"""
build_aggregate_write_states.py  --  MVP 3b: Aggregate Gate / Buffer-Commit WRITE

Gate chain: MATCH -> COMPATIBILITY -> AGGREGATE -> COMMIT

Incoming examples route to:
  - persistent states (direct update, if compatible)
  - aggregate buffers (temporary, if not compatible)
  - buffers promoted to persistent states after sufficient support

Prevents both MVP 3a failure modes:
  - over-merging  (budget_filling): incompatible examples no longer contaminate states
  - over-splitting (token_conflict): evidence must accumulate before becoming permanent

Writer variants:
  aggregate_basic            sim + count + p_state + entropy compatibility gate
  aggregate_budget_filling   adaptive promotion count to prevent budget underuse
  aggregate_conflict_aware   token conflict routing only after min_reliable_count support
  aggregate_promote_pure     promote by purity OR count+entropy jointly
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse, json, time
from collections import deque
import numpy as np
import torch

from utils import get_device, set_seed

TOP_K           = 50
EPS             = 1e-10
CHUNK           = 512
COMMIT_INTERVAL = 5000


# ── helpers ───────────────────────────────────────────────────────────────────

def _entropy_dict(counts: dict, total: float) -> float:
    if total <= 0 or not counts:
        return 0.0
    probs = np.array([c / total for c in counts.values()], np.float64)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs + EPS)))


def _purity_dict(counts: dict, total: float) -> float:
    if total <= 0 or not counts:
        return 1.0
    return float(max(counts.values()) / total)


def _ema_update(proto: torch.Tensor, h: torch.Tensor, count: float, ema_cap: int) -> torch.Tensor:
    lr  = 1.0 / max(1.0, min(count, float(ema_cap)))
    upd = (1.0 - lr) * proto + lr * h
    n   = upd.norm()
    return upd / n if n > EPS else upd


# ── compatibility predicates ──────────────────────────────────────────────────

def is_state_compatible(
    nearest_sim: float, p_state_y: float, state_count: float, state_entropy: float,
    sim_threshold: float, min_compat_prob: float, min_reliable_count: float,
    max_entropy_for_blind_update: float,
) -> bool:
    if nearest_sim < sim_threshold:
        return False
    if state_count < min_reliable_count:
        return True   # too young to judge; accept freely
    if p_state_y >= min_compat_prob:
        return True
    if state_entropy <= max_entropy_for_blind_update:
        return True
    return False


def is_state_compatible_conflict_aware(
    nearest_sim: float, p_state_y: float, state_count: float,
    sim_threshold: float, min_compat_prob: float, min_reliable_count: float,
) -> bool:
    """Strict version: no entropy fallback once a state is mature."""
    if nearest_sim < sim_threshold:
        return False
    if state_count < min_reliable_count:
        return True
    return p_state_y >= min_compat_prob


def is_buffer_compatible(
    buffer_sim: float, p_buffer_y: float, buffer_count: float,
    buffer_sim_threshold: float, buffer_min_compat_prob: float,
    min_buffer_reliable_count: float,
) -> bool:
    if buffer_sim < buffer_sim_threshold:
        return False
    if buffer_count < min_buffer_reliable_count:
        return True
    return p_buffer_y >= buffer_min_compat_prob


# ── PersistentStore ───────────────────────────────────────────────────────────

class PersistentStore:
    def __init__(self, budget: int, d: int, ema_cap: int, device, top_k: int = TOP_K):
        self.budget   = budget
        self.d        = d
        self.ema_cap  = ema_cap
        self.device   = device
        self.top_k    = top_k
        self.n        = 0
        self.proto    = torch.zeros(budget, d, device=device, dtype=torch.float32)
        self.counts   = []
        self.total    = np.zeros(budget, np.float64)
        self.created  = np.zeros(budget, np.int32)
        self.last_upd = np.zeros(budget, np.int32)
        self.sum_nll  = np.zeros(budget, np.float64)
        self.sum_ent  = np.zeros(budget, np.float64)
        self.source   = []

    def find_nearest_batch(self, h_chunk: torch.Tensor):
        if self.n == 0:
            C = len(h_chunk)
            return torch.full((C,), -1, dtype=torch.long), torch.full((C,), -1.0)
        sims = h_chunk @ self.proto[:self.n].T
        best_sims, best_ids = sims.max(dim=1)
        return best_ids.cpu(), best_sims.cpu()

    def create(self, h: torch.Tensor, y: int, nll: float, gpt_ent: float,
               idx: int, src: str = "direct_seed") -> int:
        sid = self.n
        self.proto[sid]    = h
        self.counts.append({y: 1})
        self.total[sid]    = 1.0
        self.created[sid]  = idx
        self.last_upd[sid] = idx
        self.sum_nll[sid]  = nll
        self.sum_ent[sid]  = gpt_ent
        self.source.append(src)
        self.n += 1
        return sid

    def promote_from_buffer(self, buf_proto: torch.Tensor, counts_dict: dict,
                             total: float, nll_sum: float, ent_sum: float,
                             created_at: int, last_updated: int) -> int:
        sid = self.n
        self.proto[sid]    = buf_proto.to(self.device)
        self.counts.append(dict(counts_dict))
        self.total[sid]    = total
        self.created[sid]  = created_at
        self.last_upd[sid] = last_updated
        self.sum_nll[sid]  = nll_sum
        self.sum_ent[sid]  = ent_sum
        self.source.append("promoted_buffer")
        self.n += 1
        return sid

    def merge_buffer(self, sid: int, buf_counts: dict, buf_total: float,
                     buf_proto: torch.Tensor, buf_nll: float, buf_ent: float):
        for tok, cnt in buf_counts.items():
            self.counts[sid][tok] = self.counts[sid].get(tok, 0) + cnt
        n_old = self.total[sid]
        n_new = n_old + buf_total
        if n_new > 0:
            w_old = n_old / n_new
            w_new = buf_total / n_new
            p = w_old * self.proto[sid] + w_new * buf_proto.to(self.device)
            nm = p.norm()
            self.proto[sid] = p / nm if nm > EPS else p
        self.total[sid]    = n_new
        self.sum_nll[sid] += buf_nll
        self.sum_ent[sid] += buf_ent

    def update(self, sid: int, h: torch.Tensor, y: int,
               nll: float, gpt_ent: float, idx: int):
        self.proto[sid]        = _ema_update(self.proto[sid], h, self.total[sid], self.ema_cap)
        self.counts[sid][y]    = self.counts[sid].get(y, 0) + 1
        self.total[sid]       += 1.0
        self.last_upd[sid]     = idx
        self.sum_nll[sid]     += nll
        self.sum_ent[sid]     += gpt_ent

    def p_state(self, sid: int, y: int) -> float:
        t = self.total[sid]
        return float(self.counts[sid].get(y, 0)) / t if t > 0 else 0.0

    def entropy(self, sid: int) -> float:
        return _entropy_dict(self.counts[sid], self.total[sid])

    def pack(self, assignments: np.ndarray) -> dict:
        B        = self.n
        tok_ids  = np.zeros((B, self.top_k), np.int64)
        tok_cnts = np.zeros((B, self.top_k), np.float32)
        entropy  = np.zeros(B, np.float32)
        purity   = np.zeros(B, np.float32)

        for sid in range(B):
            cd, total = self.counts[sid], self.total[sid]
            if not cd or total <= 0:
                continue
            items = sorted(cd.items(), key=lambda x: -x[1])[:self.top_k]
            for j, (tid, cnt) in enumerate(items):
                tok_ids[sid, j]  = tid
                tok_cnts[sid, j] = float(cnt)
            entropy[sid] = float(_entropy_dict(cd, total))
            purity[sid]  = float(_purity_dict(cd, total))

        proto_np  = self.proto[:B].cpu().float().numpy()
        tc        = self.total[:B]
        mean_nll  = np.where(tc > 0, self.sum_nll[:B] / (tc + EPS), 0.0)
        mean_gent = np.where(tc > 0, self.sum_ent[:B] / (tc + EPS), 0.0)

        valid = assignments >= 0
        ac2   = np.zeros(B, np.int64)
        if valid.any() and B > 0:
            ac2 = np.bincount(assignments[valid], minlength=B).astype(np.int64)

        return {
            "prototype_h":        torch.tensor(proto_np,    dtype=torch.float32),
            "top_k_token_ids":    torch.tensor(tok_ids,     dtype=torch.long),
            "top_k_token_counts": torch.tensor(tok_cnts,    dtype=torch.float32),
            "total_counts":       torch.tensor(tc.astype(np.float32)),
            "state_entropy":      torch.tensor(entropy,     dtype=torch.float32),
            "state_purity":       torch.tensor(purity,      dtype=torch.float32),
            "mean_nll":           torch.tensor(mean_nll.astype(np.float32)),
            "mean_gpt_entropy":   torch.tensor(mean_gent.astype(np.float32)),
            "assigned_count":     torch.tensor(ac2,         dtype=torch.long),
            "assigned_indices":   torch.tensor(assignments, dtype=torch.long),
            "created_at":         torch.tensor(self.created[:B],  dtype=torch.long),
            "last_updated":       torch.tensor(self.last_upd[:B], dtype=torch.long),
            "num_states":         B,
        }


# ── BufferStore ───────────────────────────────────────────────────────────────

class BufferStore:
    def __init__(self, max_buffers: int, d: int, ema_cap: int, device):
        self.max_buffers = max_buffers
        self.d           = d
        self.ema_cap     = ema_cap
        self.device      = device

        self.proto    = torch.zeros(max_buffers, d, device=device, dtype=torch.float32)
        self.active_t  = torch.zeros(max_buffers, dtype=torch.bool, device=device)
        self.active_np = np.zeros(max_buffers, bool)   # CPU mirror for Python loops
        self.counts   = [None] * max_buffers
        self.total    = np.zeros(max_buffers, np.float64)
        self.created  = np.zeros(max_buffers, np.int32)
        self.last_upd = np.zeros(max_buffers, np.int32)
        self.sum_nll  = np.zeros(max_buffers, np.float64)
        self.sum_ent  = np.zeros(max_buffers, np.float64)
        self.parent   = np.full(max_buffers, -1, np.int32)
        self.example_idxs = [[] for _ in range(max_buffers)]

        self._free      = deque(range(max_buffers))
        self._n_active  = 0
        self.n_created  = 0
        self.n_promoted = 0
        self.n_dropped  = 0

    def n_active(self) -> int:
        return self._n_active

    def find_nearest(self, h: torch.Tensor) -> tuple:
        if self._n_active == 0:
            return -1, -1.0
        sims = self.proto @ h           # [max_buffers]
        sims[~self.active_t] = -2.0
        best = int(sims.argmax())
        return best, float(sims[best])

    def _alloc(self) -> int:
        if not self._free:
            return -1
        bid = self._free.popleft()
        self._n_active += 1
        return bid

    def _release(self, bid: int):
        self.active_t[bid]     = False
        self.active_np[bid]    = False
        self._n_active        -= 1
        self._free.appendleft(bid)
        self.example_idxs[bid] = []

    def create(self, h: torch.Tensor, y: int, nll: float, gpt_ent: float,
               idx: int, parent_sid: int = -1, reason: str = "no_compat") -> int:
        bid = self._alloc()
        if bid < 0:
            return -1
        self.proto[bid]        = h
        self.active_t[bid]     = True
        self.active_np[bid]    = True
        self.counts[bid]       = {y: 1}
        self.total[bid]        = 1.0
        self.created[bid]      = idx
        self.last_upd[bid]     = idx
        self.sum_nll[bid]      = nll
        self.sum_ent[bid]      = gpt_ent
        self.parent[bid]       = parent_sid
        self.example_idxs[bid] = [idx]
        self.n_created        += 1
        return bid

    def update(self, bid: int, h: torch.Tensor, y: int,
               nll: float, gpt_ent: float, idx: int):
        self.proto[bid]        = _ema_update(self.proto[bid], h, self.total[bid], self.ema_cap)
        self.counts[bid][y]    = self.counts[bid].get(y, 0) + 1
        self.total[bid]       += 1.0
        self.last_upd[bid]     = idx
        self.sum_nll[bid]     += nll
        self.sum_ent[bid]     += gpt_ent
        self.example_idxs[bid].append(idx)

    def p_buffer(self, bid: int, y: int) -> float:
        t = self.total[bid]
        return float(self.counts[bid].get(y, 0)) / t if (t > 0 and self.counts[bid]) else 0.0

    def entropy(self, bid: int) -> float:
        return _entropy_dict(self.counts[bid] or {}, self.total[bid])

    def purity(self, bid: int) -> float:
        return _purity_dict(self.counts[bid] or {}, self.total[bid])

    def drop(self, bid: int):
        self._release(bid)
        self.n_dropped += 1

    def promote(self, bid: int):
        self._release(bid)
        self.n_promoted += 1

    def evict_weakest(self, current_idx: int):
        active = np.where(self.active_np)[0]
        if len(active) == 0:
            return
        # score: low count + old + high entropy → weak → evict
        ages    = current_idx - self.last_upd[active].astype(np.float64)
        totals  = self.total[active]
        scores  = -totals + 0.01 * ages
        worst   = active[int(scores.argmax())]
        self.drop(worst)


# ── COMMIT ────────────────────────────────────────────────────────────────────

def _should_promote(cnt: float, ent: float, pur: float,
                    eff_prom_cnt: int, max_prom_ent: float,
                    promote_mode: str, promote_purity: float) -> bool:
    if promote_mode == "purity_count":
        return ((cnt >= eff_prom_cnt and pur >= promote_purity)
                or (cnt >= 2 * eff_prom_cnt and ent <= max_prom_ent))
    return cnt >= eff_prom_cnt and ent <= max_prom_ent


def run_commit(
    persistent: PersistentStore,
    buffers: BufferStore,
    asgn: np.ndarray,
    current_idx: int,
    state_budget: int,
    effective_promote_count: int,
    max_promote_entropy: float,
    max_buffer_age: int,
    min_survive_count: int,
    hp: dict,
):
    enable_merge   = hp.get("enable_merge",    False)
    merge_sim_thr  = hp.get("merge_sim_threshold", 0.999)
    promote_mode   = hp.get("promote_mode",    "count_entropy")
    promote_purity = hp.get("promote_purity",  0.6)

    to_process = list(np.where(buffers.active_np)[0])

    for bid in to_process:
        if not buffers.active_np[bid]:
            continue

        cnt = float(buffers.total[bid])
        age = current_idx - int(buffers.created[bid])
        ent = buffers.entropy(bid)
        pur = buffers.purity(bid)

        # PROMOTE
        if (persistent.n < state_budget
                and _should_promote(cnt, ent, pur, effective_promote_count,
                                    max_promote_entropy, promote_mode, promote_purity)):
            new_sid = persistent.promote_from_buffer(
                buffers.proto[bid].clone(),
                buffers.counts[bid],
                cnt,
                float(buffers.sum_nll[bid]),
                float(buffers.sum_ent[bid]),
                int(buffers.created[bid]),
                int(buffers.last_upd[bid]),
            )
            for ex_idx in buffers.example_idxs[bid]:
                asgn[ex_idx] = new_sid
            buffers.promote(bid)
            continue

        # MERGE (disabled by default in fast mode)
        if enable_merge and persistent.n >= state_budget and persistent.n > 0:
            buf_h  = buffers.proto[bid].unsqueeze(0)
            sims_p = (buf_h @ persistent.proto[:persistent.n].T).squeeze(0)
            best_p = int(sims_p.argmax())
            if float(sims_p[best_p]) >= merge_sim_thr:
                persistent.merge_buffer(
                    best_p,
                    buffers.counts[bid], cnt,
                    buffers.proto[bid].clone(),
                    float(buffers.sum_nll[bid]),
                    float(buffers.sum_ent[bid]),
                )
                for ex_idx in buffers.example_idxs[bid]:
                    asgn[ex_idx] = best_p
                buffers.promote(bid)
                continue

        # DROP
        if max_buffer_age > 0 and age > max_buffer_age and cnt < min_survive_count:
            buffers.drop(bid)


# ── main aggregate loop ───────────────────────────────────────────────────────

def _get_effective_promote_count(
    method: str, n_states: int, state_budget: int,
    base_promote_count: int, current_idx: int, N: int,
) -> int:
    if method == "aggregate_budget_filling":
        progress = current_idx / max(N - 1, 1)
        target   = progress * state_budget
        if n_states < 0.8 * target:
            return max(2, base_promote_count // 2)
    return base_promote_count


def _aggregate_loop(
    h_gpu: torch.Tensor,
    y: np.ndarray,
    nll_gpt: np.ndarray,
    gpt_ent: np.ndarray,
    state_budget: int,
    buffer_budget: int,
    ema_cap: int,
    device,
    hp: dict,
    method: str,
) -> dict:
    N, D = h_gpu.shape
    persistent = PersistentStore(state_budget, D, ema_cap, device)
    buffers    = BufferStore(buffer_budget, D, ema_cap, device)
    asgn       = np.full(N, -1, dtype=np.int64)

    sim_thr       = hp["sim_threshold"]
    min_compat_p  = hp["min_compat_prob"]
    min_rel_cnt   = hp["min_reliable_count"]
    max_ent_blind = hp["max_entropy_for_blind_update"]
    buf_sim_thr   = hp["buffer_sim_threshold"]
    buf_min_p     = hp["buffer_min_compat_prob"]
    min_buf_rel   = hp["min_buffer_reliable_count"]
    prom_cnt      = hp["promote_count"]
    max_prom_ent  = hp["max_promote_entropy"]
    max_buf_age   = hp["max_buffer_age"]
    min_surv      = hp["min_survive_count"]
    compat_mode   = hp.get("compat_mode", "basic")

    n_direct = n_buf_upd = n_buf_cre = 0

    for start in range(0, N, CHUNK):
        end = min(start + CHUNK, N)
        blk = h_gpu[start:end]
        C   = end - start

        p_ids, p_sims = persistent.find_nearest_batch(blk)

        for i in range(C):
            idx = start + i
            yi  = int(y[idx])
            nll = float(nll_gpt[idx])
            ent = float(gpt_ent[idx])
            h_i = blk[i]

            sid = int(p_ids[i])
            sim = float(p_sims[i])

            if sid >= 0:
                p_st_y  = persistent.p_state(sid, yi)
                s_ent_v = persistent.entropy(sid)
                s_cnt   = float(persistent.total[sid])
            else:
                p_st_y = s_ent_v = s_cnt = 0.0

            if compat_mode == "conflict_aware":
                compat = (sid >= 0) and is_state_compatible_conflict_aware(
                    sim, p_st_y, s_cnt, sim_thr, min_compat_p, min_rel_cnt)
            else:
                compat = (sid >= 0) and is_state_compatible(
                    sim, p_st_y, s_cnt, s_ent_v,
                    sim_thr, min_compat_p, min_rel_cnt, max_ent_blind)

            if compat:
                persistent.update(sid, h_i, yi, nll, ent, idx)
                asgn[idx] = sid
                n_direct += 1
            else:
                bid, buf_sim = buffers.find_nearest(h_i)
                if bid >= 0:
                    p_buf_y = buffers.p_buffer(bid, yi)
                    buf_cnt = float(buffers.total[bid])
                else:
                    p_buf_y = buf_cnt = 0.0

                buf_compat = (bid >= 0) and is_buffer_compatible(
                    buf_sim, p_buf_y, buf_cnt,
                    buf_sim_thr, buf_min_p, min_buf_rel)

                if buf_compat:
                    buffers.update(bid, h_i, yi, nll, ent, idx)
                    n_buf_upd += 1
                else:
                    if buffers.n_active() >= buffer_budget:
                        buffers.evict_weakest(idx)
                    reason = "low_sim" if sim < sim_thr else "low_p_state"
                    new_bid = buffers.create(h_i, yi, nll, ent, idx,
                                             parent_sid=sid, reason=reason)
                    n_buf_cre += 1

            if (idx + 1) % COMMIT_INTERVAL == 0:
                eff_pc = _get_effective_promote_count(
                    method, persistent.n, state_budget, prom_cnt, idx, N)
                run_commit(persistent, buffers, asgn, idx, state_budget,
                           eff_pc, max_prom_ent, max_buf_age, min_surv, hp)

        if (start // CHUNK) % 20 == 0:
            pct = 100 * start / N
            print(f"    {start:,}/{N:,} ({pct:.0f}%)  "
                  f"states={persistent.n:,}  bufs={buffers.n_active():,}", end="\r")

    print(f"    {N:,}/{N:,} (100%)  "
          f"states={persistent.n:,}  bufs={buffers.n_active():,}         ")

    # Final COMMIT: promote anything that qualifies; never drop
    eff_pc = _get_effective_promote_count(method, persistent.n, state_budget, prom_cnt, N - 1, N)
    run_commit(persistent, buffers, asgn, N - 1, state_budget,
               eff_pc, max_prom_ent, max_buffer_age=0, min_survive_count=0, hp=hp)

    state_dict = persistent.pack(asgn)
    state_dict.update({
        "method":                method,
        "budget":                state_budget,
        "actual_num_states":     persistent.n,
        "num_buffers_final":     buffers.n_active(),
        "num_buffers_promoted":  buffers.n_promoted,
        "num_buffers_dropped":   buffers.n_dropped,
        "num_direct_updates":    n_direct,
        "num_buffer_updates":    n_buf_upd,
        "num_buffer_creates":    n_buf_cre,
        "hyperparams":           hp,
        "create_update_commit_trace_summary": {
            "n_direct":             n_direct,
            "n_buf_update":         n_buf_upd,
            "n_buf_create":         n_buf_cre,
            "n_promoted":           buffers.n_promoted,
            "n_dropped":            buffers.n_dropped,
            "n_remaining_buffers":  buffers.n_active(),
            "n_total":              N,
            "direct_frac":          n_direct / max(N, 1),
            "buffer_frac":          (n_buf_upd + n_buf_cre) / max(N, 1),
        },
    })
    return state_dict


# ── variant configurations ────────────────────────────────────────────────────

_BASE_HP = dict(
    buffer_sim_threshold       = 0.990,
    buffer_min_compat_prob     = 0.001,
    min_buffer_reliable_count  = 4,
    max_entropy_for_blind_update = 1.0,
    max_buffer_age             = 10000,
    min_survive_count          = 2,
    buffer_budget_frac         = 0.25,
    enable_merge               = False,
    merge_sim_threshold        = 0.999,
    overlap_threshold          = 0.25,
    promote_mode               = "count_entropy",
    promote_purity             = 0.6,
    compat_mode                = "basic",
)


def _fast_configs():
    return [
        dict(sim_threshold=0.995, min_compat_prob=0.005,
             min_reliable_count=8, promote_count=8, max_promote_entropy=2.0),
        dict(sim_threshold=0.995, min_compat_prob=0.005,
             min_reliable_count=4, promote_count=4, max_promote_entropy=2.0),
    ]


def _full_configs():
    configs = []
    for sim in [0.990, 0.995, 0.997]:
        for (rel, prom) in [(4, 4), (8, 8), (8, 16), (16, 8)]:
            for p in [0.001, 0.005, 0.01]:
                configs.append(dict(
                    sim_threshold=sim, min_compat_prob=p,
                    min_reliable_count=rel, promote_count=prom,
                    max_promote_entropy=2.0,
                ))
    return configs


def get_variants(method: str, budgets: list, fast: bool, ema_caps: list) -> list:
    """Returns list of (tag, hp, budget, ema_cap) tuples."""
    configs = _fast_configs() if fast else _full_configs()
    out = []
    for ema_cap in ema_caps:
        for cfg in configs:
            for B in budgets:
                hp = dict(_BASE_HP)
                hp.update(cfg)
                hp["ema_cap"]    = ema_cap
                buf_budget = max(int(B * hp["buffer_budget_frac"]), 64)

                # method-specific overrides
                if method == "aggregate_basic":
                    hp["compat_mode"] = "basic"
                    tag = (f"aggregate_basic_sim{cfg['sim_threshold']}"
                           f"_p{cfg['min_compat_prob']}"
                           f"_rel{cfg['min_reliable_count']}"
                           f"_prom{cfg['promote_count']}"
                           f"_ema{ema_cap}_B{B}")

                elif method == "aggregate_budget_filling":
                    hp["compat_mode"] = "basic"
                    tag = (f"aggregate_budget_filling_sim{cfg['sim_threshold']}"
                           f"_p{cfg['min_compat_prob']}"
                           f"_rel{cfg['min_reliable_count']}"
                           f"_prom{cfg['promote_count']}"
                           f"_ema{ema_cap}_B{B}")

                elif method == "aggregate_conflict_aware":
                    hp["compat_mode"] = "conflict_aware"
                    tag = (f"aggregate_conflict_aware_sim{cfg['sim_threshold']}"
                           f"_p{cfg['min_compat_prob']}"
                           f"_rel{cfg['min_reliable_count']}"
                           f"_prom{cfg['promote_count']}"
                           f"_ema{ema_cap}_B{B}")

                elif method == "aggregate_promote_pure":
                    hp["compat_mode"]   = "basic"
                    hp["promote_mode"]  = "purity_count"
                    for pp in ([0.6] if fast else [0.4, 0.6, 0.8]):
                        hp2       = dict(hp)
                        hp2["promote_purity"] = pp
                        tag2 = (f"aggregate_promote_pure_sim{cfg['sim_threshold']}"
                                f"_p{cfg['min_compat_prob']}"
                                f"_rel{cfg['min_reliable_count']}"
                                f"_prom{cfg['promote_count']}"
                                f"_pur{pp}_ema{ema_cap}_B{B}")
                        out.append((tag2, hp2, B, ema_cap, buf_budget))
                    continue
                else:
                    continue

                out.append((tag, hp, B, ema_cap, buf_budget))
    return out


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",    required=True)
    parser.add_argument("--output",    required=True)
    parser.add_argument("--methods",   nargs="+",
                        default=["aggregate_basic", "aggregate_budget_filling",
                                 "aggregate_conflict_aware"])
    parser.add_argument("--budgets",   nargs="+", type=int, default=[10000, 25000])
    parser.add_argument("--ema_caps",  nargs="+", type=int, default=[128])
    parser.add_argument("--fast",      action="store_true")
    parser.add_argument("--force",     action="store_true")
    parser.add_argument("--device",    default="cuda")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device({"device": args.device})
    src    = Path(args.source)
    out    = Path(args.output) / "states"
    out.mkdir(parents=True, exist_ok=True)

    print(f"\nbuild_aggregate_write_states")
    print(f"  Source:  {src}")
    print(f"  Output:  {out}")
    print(f"  Methods: {args.methods}")
    print(f"  Budgets: {args.budgets}")
    print(f"  Fast:    {args.fast}  Device: {device}")

    ct_path = src / "states" / "controller_train.pt"
    print(f"\nLoading CT data from {ct_path} ...")
    ct  = torch.load(ct_path, weights_only=False)
    h   = ct["h"].float()
    h   = h / (h.norm(dim=-1, keepdim=True) + EPS)
    y   = ct["y"].numpy().astype(np.int64)
    nll_gpt = ct["nll_gpt"].numpy().astype(np.float64)
    gpt_ent = ct.get("gpt_entropy", ct["nll_gpt"]).float().numpy().astype(np.float64)
    N, D_h  = h.shape
    print(f"  N={N:,}  D={D_h}")

    h_gpu = h.to(device)

    all_variants = []
    for m in args.methods:
        all_variants.extend(get_variants(m, args.budgets, args.fast, args.ema_caps))

    print(f"\nTotal variants to build: {len(all_variants)}")
    manifest = []
    t0_all   = time.time()

    for vi, (tag, hp, budget, ema_cap, buf_budget) in enumerate(all_variants):
        out_path = out / f"{tag}.pt"
        print(f"\n[{vi+1}/{len(all_variants)}] {tag}")

        if out_path.exists() and not args.force:
            print(f"  [cached] {out_path.name}")
            try:
                sd = torch.load(out_path, weights_only=False)
                B  = sd.get("actual_num_states", sd.get("num_states", "?"))
            except Exception:
                B = "?"
            manifest.append({"tag": tag, "budget": budget, "path": str(out_path),
                              "status": "cached", "actual_num_states": B})
            continue

        t0 = time.time()
        method = tag.split("_B")[0].rsplit("_ema", 1)[0]
        for m in ["aggregate_basic", "aggregate_budget_filling",
                  "aggregate_conflict_aware", "aggregate_promote_pure"]:
            if tag.startswith(m):
                method = m
                break

        print(f"  method={method}  budget={budget}  buf_budget={buf_budget}  ema={ema_cap}")
        print(f"  sim={hp['sim_threshold']}  p={hp['min_compat_prob']}"
              f"  rel={hp['min_reliable_count']}  prom={hp['promote_count']}"
              f"  max_ent={hp['max_promote_entropy']}")

        try:
            state_dict = _aggregate_loop(
                h_gpu, y, nll_gpt, gpt_ent,
                budget, buf_budget, ema_cap, device, hp, method,
            )
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback; traceback.print_exc()
            continue

        B_actual = state_dict["actual_num_states"]
        elapsed  = time.time() - t0
        trace    = state_dict.get("create_update_commit_trace_summary", {})
        print(f"  Saved {out_path.name}")
        print(f"  B={B_actual:,}  promoted={state_dict['num_buffers_promoted']:,}"
              f"  dropped={state_dict['num_buffers_dropped']:,}"
              f"  direct_frac={trace.get('direct_frac', 0):.2%}"
              f"  elapsed={elapsed:.0f}s")

        torch.save(state_dict, out_path)
        manifest.append({
            "tag": tag, "method": method, "budget": budget,
            "actual_num_states": B_actual,
            "num_buffers_promoted": state_dict["num_buffers_promoted"],
            "num_buffers_dropped":  state_dict["num_buffers_dropped"],
            "path": str(out_path), "status": "built", "elapsed_s": elapsed,
            **{k: v for k, v in hp.items() if not isinstance(v, (dict, list))},
        })

    mnf_path = Path(args.output) / "states_manifest.json"
    with open(mnf_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {mnf_path}")
    print(f"Total elapsed: {time.time()-t0_all:.0f}s")
    print(f"Built/cached: {len(manifest)} variants")


if __name__ == "__main__":
    main()

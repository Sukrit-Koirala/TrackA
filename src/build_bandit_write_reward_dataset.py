"""
build_bandit_write_reward_dataset.py  --  MVP 4b Stage 1

Replays the oracle trajectory; at sampled decision points enumerates valid
WRITE actions, evaluates each counterfactually on held-out probes, and saves
the resulting (feature, reward) dataset.

Teacher IDs drive the oracle trajectory only.  They never appear as features,
action labels, or reward targets.  Rewards come from measured probe NLL delta.

Fixed READ parameters: k=4, tau=0.05, alpha=0.75, beta=0.0
(matches empirically best fixed action from MVP 2c / Track A).

Usage:
  python src/build_bandit_write_reward_dataset.py \\
    --source              scale_200k_seed42 \\
    --oracle_dir          outputs_mvp4a0_oracle_reconstruction \\
    --output              outputs_mvp4b_bandit_write_fast \\
    --n_decision_points   10000 \\
    --k_states 8 --k_buffers 8 \\
    --n_local_probes 16 --n_history_probes 16 --n_replay_probes 16 \\
    --future_window 0 --max_exemplars 16 \\
    --promotion_support 8 \\
    --state_budget 10000 --buffer_budget 10000 \\
    --seed 42 --device cuda
"""

import sys, json, math, random, argparse
from pathlib import Path
from dataclasses import dataclass, field
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch

from utils import get_device, set_seed

VOCAB = 50257
# Fixed READ: matches best fixed action from existing eval (k2_t0.05_a0.75_b0.0)
# We use k=4 here for richer local mixing
READ_K    = 4
READ_TAU  = 0.05
READ_ALPHA = 0.75

# Action type indices
ACT_UPDATE_STATE       = 0
ACT_UPDATE_BUFFER      = 1
ACT_PROMOTE_AND_UPDATE = 2
ACT_CREATE_BUFFER      = 3
ACT_DEFER              = 4
ACT_NAMES = ["UPDATE_STATE", "UPDATE_BUFFER", "PROMOTE_AND_UPDATE",
             "CREATE_BUFFER", "DEFER"]
N_ACT_TYPES = 5

FEATURE_NAMES = [
    # global (12)
    "stream_progress", "gpt_nll", "gpt_entropy",
    "n_persistent", "n_buffers",
    "budget_frac_persistent", "budget_frac_buffers",
    "state_top1_sim", "state_top2_sim", "state_sim_gap",
    "buffer_top1_sim", "buffer_top2_sim",
    # action one-hot (5)
    "act_update_state", "act_update_buffer", "act_promote_update",
    "act_create_buffer", "act_defer",
    # candidate (8)
    "cand_sim", "cand_rank_norm", "cand_log_support",
    "cand_is_persistent", "cand_entropy", "cand_p_y_true",
    "cand_age_frac", "cand_recency_frac",
]
N_FEAT = len(FEATURE_NAMES)  # 25

REWARD_COLS = [
    "reward_local_only",    # local geometric probes
    "reward_local_history", # local + history probes
    "reward_full",          # local + history + replay
    "reward_penalized",     # full - action costs
    "nll_before",           # baseline probe NLL mean (full probe set)
    "nll_after",            # post-action probe NLL mean
    "history_damage",       # mean NLL increase on candidate history probes
]


# ── live objects ──────────────────────────────────────────────────────────────

@dataclass
class LiveObj:
    obj_id:           int
    teacher_id:       int           # oracle label; never used as feature/reward
    sum_h:            np.ndarray    # float64 [D]
    token_counts:     dict          # {tok_id: count}
    count:            int
    created_step:     int
    last_update_step: int
    is_promoted:      bool
    promoted_step:    int
    exemplars:        list          # row indices, reservoir-sampled


def _proto(obj: LiveObj) -> np.ndarray:
    n = float(np.linalg.norm(obj.sum_h))
    return (obj.sum_h / max(n, 1e-8)).astype(np.float32)


def _entropy_d(d: dict, total: int) -> float:
    if total <= 0: return 0.0
    e = 0.0
    for c in d.values():
        p = c / total
        if p > 0: e -= p * math.log(p + 1e-12)
    return e


def _copy_obj(obj: LiveObj) -> LiveObj:
    return LiveObj(
        obj_id=obj.obj_id, teacher_id=obj.teacher_id,
        sum_h=obj.sum_h.copy(), token_counts=dict(obj.token_counts),
        count=obj.count, created_step=obj.created_step,
        last_update_step=obj.last_update_step,
        is_promoted=obj.is_promoted, promoted_step=obj.promoted_step,
        exemplars=list(obj.exemplars),
    )


def _update_inplace(obj: LiveObj, h_raw: np.ndarray, y: int, step: int, max_ex: int):
    obj.sum_h += h_raw.astype(np.float64)
    obj.token_counts[y] = obj.token_counts.get(y, 0) + 1
    obj.count += 1
    obj.last_update_step = step
    if len(obj.exemplars) < max_ex:
        obj.exemplars.append(step)
    else:
        j = random.randint(0, obj.count - 1)
        if j < max_ex:
            obj.exemplars[j] = step


def _make_new_obj(obj_id: int, teacher_id: int,
                  h_raw: np.ndarray, y: int, step: int) -> LiveObj:
    return LiveObj(
        obj_id=obj_id, teacher_id=teacher_id,
        sum_h=h_raw.astype(np.float64),
        token_counts={y: 1},
        count=1, created_step=step, last_update_step=step,
        is_promoted=False, promoted_step=-1,
        exemplars=[step],
    )


# ── GPU-backed live memory ────────────────────────────────────────────────────

class LiveMemory:
    def __init__(self, state_budget: int, buffer_budget: int, D: int,
                 device: torch.device):
        self.state_budget  = state_budget
        self.buffer_budget = buffer_budget
        self.device        = device
        total = state_budget + buffer_budget
        self.protos = torch.zeros(total, D, device=device)   # prototype matrix
        self.objects:         dict[int, LiveObj] = {}         # obj_id -> obj
        self.teacher_to_obj:  dict[int, int]     = {}         # teacher_id -> obj_id
        self.obj_to_row:      dict[int, int]     = {}         # obj_id -> row
        self.row_to_obj:      dict[int, int]     = {}         # row -> obj_id
        self.persistent_rows: list               = []
        self.buffer_rows:     list               = []
        self._next_id  = 0
        self._next_row = 0

    def _alloc_row(self, obj_id: int) -> int:
        row = self._next_row
        self._next_row += 1
        self.obj_to_row[obj_id] = row
        self.row_to_obj[row] = obj_id
        return row

    def _sync_proto(self, obj: LiveObj):
        row = self.obj_to_row[obj.obj_id]
        self.protos[row] = torch.from_numpy(_proto(obj)).to(self.device)

    def has_teacher(self, tid: int) -> bool:
        return tid in self.teacher_to_obj

    def create_buffer(self, teacher_id: int, h_raw: np.ndarray,
                      y: int, step: int) -> LiveObj:
        obj = _make_new_obj(self._next_id, teacher_id, h_raw, y, step)
        self._next_id += 1
        self.objects[obj.obj_id] = obj
        self.teacher_to_obj[teacher_id] = obj.obj_id
        row = self._alloc_row(obj.obj_id)
        self.buffer_rows.append(row)
        self._sync_proto(obj)
        return obj

    def update(self, teacher_id: int, h_raw: np.ndarray, y: int,
               step: int, max_ex: int, promotion_support: int) -> LiveObj:
        obj = self.objects[self.teacher_to_obj[teacher_id]]
        _update_inplace(obj, h_raw, y, step, max_ex)
        if not obj.is_promoted and obj.count >= promotion_support:
            obj.is_promoted  = True
            obj.promoted_step = step
            row = self.obj_to_row[obj.obj_id]
            self.buffer_rows.remove(row)
            self.persistent_rows.append(row)
        self._sync_proto(obj)
        return obj

    def get_candidates(self, h_n: np.ndarray, k_states: int, k_buffers: int):
        """Returns (state_cands, buf_cands, sim_info) where cands=list[(sim, obj)]."""
        h_gpu = torch.from_numpy(h_n).to(self.device).unsqueeze(0)  # [1,D]

        state_cands, s_top1, s_top2 = [], 0.0, 0.0
        if self.persistent_rows:
            rows  = self.persistent_rows
            protos = self.protos[rows]                         # [n_p, D]
            sims  = (h_gpu @ protos.T).squeeze(0).cpu().numpy()  # [n_p]
            k_eff = min(k_states, len(rows))
            top   = np.argsort(-sims)[:k_eff]
            for ri in top:
                state_cands.append((float(sims[ri]),
                                    self.objects[self.row_to_obj[rows[ri]]]))
            sorted_all = np.sort(sims)[::-1]
            s_top1 = float(sorted_all[0]) if len(sorted_all) >= 1 else 0.0
            s_top2 = float(sorted_all[1]) if len(sorted_all) >= 2 else s_top1

        buf_cands, b_top1, b_top2 = [], 0.0, 0.0
        if self.buffer_rows:
            rows  = self.buffer_rows
            protos = self.protos[rows]
            sims  = (h_gpu @ protos.T).squeeze(0).cpu().numpy()
            k_eff = min(k_buffers, len(rows))
            top   = np.argsort(-sims)[:k_eff]
            for ri in top:
                buf_cands.append((float(sims[ri]),
                                  self.objects[self.row_to_obj[rows[ri]]]))
            sorted_all = np.sort(sims)[::-1]
            b_top1 = float(sorted_all[0]) if len(sorted_all) >= 1 else 0.0
            b_top2 = float(sorted_all[1]) if len(sorted_all) >= 2 else b_top1

        return state_cands, buf_cands, (s_top1, s_top2, b_top1, b_top2)

    @property
    def n_persistent(self): return len(self.persistent_rows)
    @property
    def n_buffers(self):    return len(self.buffer_rows)


# ── probe NLL ─────────────────────────────────────────────────────────────────

def probe_nll(
    h_probes:     np.ndarray,   # [P, D] normalized
    y_probes:     np.ndarray,   # [P] int
    p_gpt_probes: np.ndarray,   # [P] float
    pool:         list,          # list of LiveObj
    k:    int   = READ_K,
    tau:  float = READ_TAU,
    alpha: float = READ_ALPHA,
) -> np.ndarray:
    P = len(h_probes)
    if P == 0:
        return np.array([], dtype=np.float32)
    if not pool:
        return -np.log(np.maximum(p_gpt_probes, 1e-12))

    protos  = np.stack([_proto(o) for o in pool])   # [K, D]
    sims    = h_probes @ protos.T                    # [P, K]
    k_eff   = min(k, len(pool))
    top_idx = np.argsort(-sims, axis=1)[:, :k_eff]  # [P, k_eff]
    top_s   = np.take_along_axis(sims, top_idx, axis=1)

    shifted = top_s - top_s.max(axis=1, keepdims=True)
    w = np.exp(shifted / (tau + 1e-8))
    w /= w.sum(axis=1, keepdims=True) + 1e-10        # [P, k_eff]

    p_sy = np.zeros((P, k_eff), dtype=np.float32)
    for pi in range(P):
        y_p = int(y_probes[pi])
        for ki in range(k_eff):
            ci = int(top_idx[pi, ki])
            c  = pool[ci]
            p_sy[pi, ki] = c.token_counts.get(y_p, 0) / max(c.count, 1)

    p_local = (w * p_sy).sum(axis=1)
    p_final = np.maximum(alpha * p_gpt_probes + (1 - alpha) * p_local, 1e-12)
    return -np.log(p_final).astype(np.float32)


# ── feature extraction ────────────────────────────────────────────────────────

def extract_features(
    step: int, N: int,
    y_t: int, nll_gpt_t: float, gpt_entropy_t: float,
    n_persistent: int, n_buffers: int,
    state_budget: int, buffer_budget: int,
    sim_info: tuple,                  # (s_top1, s_top2, b_top1, b_top2)
    cand_obj: "LiveObj | None",
    cand_sim:  float,
    cand_rank: int,
    action_type: int,
) -> np.ndarray:
    s_top1, s_top2, b_top1, b_top2 = sim_info
    s_gap = s_top1 - s_top2

    g = [
        step / max(N - 1, 1), float(nll_gpt_t), float(gpt_entropy_t),
        float(n_persistent), float(n_buffers),
        n_persistent / max(state_budget, 1),
        n_buffers    / max(buffer_budget, 1),
        s_top1, s_top2, s_gap, b_top1, b_top2,
    ]  # 12 features

    a = [0.0] * N_ACT_TYPES
    a[action_type] = 1.0  # 5 features

    if cand_obj is not None:
        p_y = cand_obj.token_counts.get(y_t, 0) / max(cand_obj.count, 1)
        c = [
            float(cand_sim),
            cand_rank / 8.0,
            math.log1p(cand_obj.count),
            float(cand_obj.is_promoted),
            _entropy_d(cand_obj.token_counts, cand_obj.count),
            float(p_y),
            (step - cand_obj.created_step) / max(N, 1),
            (step - cand_obj.last_update_step) / max(N, 1),
        ]  # 8 features
    else:
        c = [0.0] * 8

    return np.array(g + a + c, dtype=np.float32)  # [25]


# ── stratified decision-point sampling ───────────────────────────────────────

def stratified_sample(N: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n indices across [0, N) stratified by quartile."""
    n_per = n // 4
    remainder = n - 4 * n_per
    idxs = []
    for q in range(4):
        lo = int(N * q / 4)
        hi = int(N * (q + 1) / 4)
        cnt = n_per + (1 if q < remainder else 0)
        cnt = min(cnt, hi - lo)
        idxs.append(rng.integers(lo, hi, size=cnt))
    return np.sort(np.concatenate(idxs))


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",             required=True)
    ap.add_argument("--oracle_dir",         required=True)
    ap.add_argument("--output",             required=True)
    ap.add_argument("--budget",             type=int, default=10000)
    ap.add_argument("--teacher",            default="minibatch_kmeans")
    ap.add_argument("--n_decision_points",  type=int, default=10000)
    ap.add_argument("--k_states",           type=int, default=8)
    ap.add_argument("--k_buffers",          type=int, default=8)
    ap.add_argument("--n_local_probes",     type=int, default=16)
    ap.add_argument("--n_history_probes",   type=int, default=16)
    ap.add_argument("--n_replay_probes",    type=int, default=16)
    ap.add_argument("--future_window",      type=int, default=0)
    ap.add_argument("--max_exemplars",      type=int, default=16)
    ap.add_argument("--promotion_support",  type=int, default=8)
    ap.add_argument("--state_budget",       type=int, default=10000)
    ap.add_argument("--buffer_budget",      type=int, default=10000)
    ap.add_argument("--lambda_history",     type=float, default=1.0)
    ap.add_argument("--lambda_create",      type=float, default=-1.0,
                    help="<0 = auto-calibrate from reward scale")
    ap.add_argument("--lambda_promote",     type=float, default=-1.0)
    ap.add_argument("--seed",               type=int, default=42)
    ap.add_argument("--device",             default="cuda")
    ap.add_argument("--force",              action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    rng  = np.random.default_rng(args.seed)
    device = get_device({"device": args.device})

    out_dir = Path(args.output) / "reward_dataset"
    sentinel = out_dir / "actions.parquet"
    if sentinel.exists() and not args.force:
        print(f"[cached] {sentinel}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load data ─────────────────────────────────────────────────────────────
    print("Loading datastore ...")
    ds_path = Path(args.source) / "states" / "datastore.pt"
    ds = torch.load(ds_path, weights_only=False)
    h_orig = ds["h"].float().numpy()                      # [N, D]
    y_np   = ds["y"].numpy().astype(np.int64)             # [N]
    nll_gpt     = ds.get("nll_gpt",     torch.zeros(len(y_np))).float().numpy()
    gpt_entropy = ds.get("gpt_entropy", torch.zeros(len(y_np))).float().numpy()
    N, D = h_orig.shape
    h_norm = (h_orig / (np.linalg.norm(h_orig, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
    p_gpt  = np.exp(-nll_gpt).astype(np.float32)
    print(f"  N={N:,}  D={D}")

    # ── load oracle assignments ───────────────────────────────────────────────
    tag  = f"{args.teacher}_B{args.budget}"
    npz  = np.load(Path(args.oracle_dir) / "teacher_assignments" / f"{tag}_assignments.npz")
    teacher_ids = npz["teacher_ids"].astype(np.int64)
    print(f"  Teacher IDs loaded: {tag}")

    # ── sample decision points ────────────────────────────────────────────────
    sampled = stratified_sample(N, args.n_decision_points, rng)
    sampled_set = set(sampled.tolist())
    print(f"  Sampled {len(sampled_set)} decision points")

    # ── initialize memory ─────────────────────────────────────────────────────
    mem = LiveMemory(args.state_budget, args.buffer_budget, D, device)
    recent_replay = []
    MAX_REPLAY = min(2048, N)

    records = []
    n_dp_done = 0

    print(f"\nRunning oracle replay + counterfactual evaluation ...")

    for step in range(N):
        h_raw = h_orig[step]
        h_n   = h_norm[step]
        y_t   = int(y_np[step])
        z_t   = int(teacher_ids[step])

        # ── decision point: evaluate BEFORE oracle update ─────────────────────
        if step in sampled_set and (mem.n_persistent + mem.n_buffers) > 0:
            state_cands, buf_cands, sim_info = mem.get_candidates(
                h_n, args.k_states, args.k_buffers)

            # Build probe indices
            # Local probes: random sample (approximation; no precomputed ANN)
            local_idxs = rng.integers(0, N, size=args.n_local_probes * 3)
            local_idxs = local_idxs[local_idxs != step][:args.n_local_probes]

            # History probes: from candidate exemplars
            hist_pool = []
            for _, obj in (state_cands + buf_cands):
                hist_pool.extend(obj.exemplars)
            if hist_pool:
                hist_pool = list(set(hist_pool) - {step})
                if len(hist_pool) > args.n_history_probes:
                    hist_pool = rng.choice(hist_pool, args.n_history_probes,
                                           replace=False).tolist()
            hist_idxs = np.array(hist_pool[:args.n_history_probes], dtype=np.int64)

            # Replay probes
            replay_pool = [r for r in recent_replay if r != step]
            if replay_pool:
                n_rp = min(args.n_replay_probes, len(replay_pool))
                rep_idxs = np.array(
                    rng.choice(replay_pool, n_rp, replace=False), dtype=np.int64)
            else:
                rep_idxs = np.array([], dtype=np.int64)

            # Combine and deduplicate
            all_probes = np.unique(np.concatenate([
                local_idxs.astype(np.int64),
                hist_idxs,
                rep_idxs,
            ]))
            all_probes = all_probes[all_probes != step]

            local_set   = set(local_idxs.tolist())
            hist_set    = set(hist_pool[:args.n_history_probes])
            replay_set  = set(rep_idxs.tolist()) if len(rep_idxs) > 0 else set()

            # Compute probe arrays
            h_probes     = h_norm[all_probes]
            y_probes     = y_np[all_probes].astype(np.int32)
            p_gpt_probes = p_gpt[all_probes]

            # Baseline candidate pool
            pool_objs = [obj for _, obj in (state_cands + buf_cands)]

            # Baseline NLL
            nll_base = probe_nll(h_probes, y_probes, p_gpt_probes, pool_objs)
            base_mean = float(nll_base.mean()) if len(nll_base) > 0 else 0.0

            # Masks for probe subsets
            local_mask  = np.array([i in local_set  for i in all_probes])
            hist_mask   = np.array([i in hist_set   for i in all_probes])
            replay_mask = np.array([i in replay_set for i in all_probes])

            def _reward(nll_after_arr, mask_local, mask_hist):
                if len(nll_after_arr) == 0:
                    return 0.0, 0.0, 0.0, 0.0
                after_mean = float(nll_after_arr.mean())
                local_gain = (float(nll_base[mask_local].mean())
                              - float(nll_after_arr[mask_local].mean())
                              if mask_local.any() else 0.0)
                lh_gain    = (float(nll_base.mean()) - after_mean
                              if len(nll_base) > 0 else 0.0)
                hist_dmg   = (float(np.maximum(0,
                              nll_after_arr[mask_hist] - nll_base[mask_hist]).mean())
                              if mask_hist.any() else 0.0)
                return local_gain, lh_gain, after_mean, hist_dmg

            def _make_record(act_type, cand_sim, cand_rank, cand_obj, nll_after_arr):
                local_g, full_g, nll_aft, hdmg = _reward(
                    nll_after_arr, local_mask, hist_mask)
                feat = extract_features(
                    step, N, y_t, float(nll_gpt[step]), float(gpt_entropy[step]),
                    mem.n_persistent, mem.n_buffers,
                    args.state_budget, args.buffer_budget,
                    sim_info, cand_obj, cand_sim, cand_rank, act_type,
                )
                rec = {
                    "step":            step,
                    "action_type":     act_type,
                    "action_name":     ACT_NAMES[act_type],
                    "cand_obj_id":     cand_obj.obj_id if cand_obj else -1,
                    "cand_is_promoted": bool(cand_obj.is_promoted) if cand_obj else False,
                    "n_probes":        len(all_probes),
                    "nll_before":      base_mean,
                    "nll_after":       nll_aft,
                    "history_damage":  hdmg,
                    "reward_local_only":    local_g,
                    "reward_local_history": full_g,
                    "reward_full":          full_g,
                }
                for fi, fn in enumerate(FEATURE_NAMES):
                    rec[fn] = float(feat[fi])
                return rec

            # ── DEFER (no change) ─────────────────────────────────────────────
            records.append(_make_record(
                ACT_DEFER, 0.0, 0, None, nll_base))

            # ── CREATE_BUFFER ─────────────────────────────────────────────────
            if mem.n_buffers < args.buffer_budget:
                new_obj = _make_new_obj(-1, -1, h_raw, y_t, step)
                nll_cb  = probe_nll(h_probes, y_probes, p_gpt_probes,
                                    [new_obj] + pool_objs)
                records.append(_make_record(ACT_CREATE_BUFFER, 0.0, 0, None, nll_cb))

            # ── UPDATE_STATE ──────────────────────────────────────────────────
            for rank, (sim, obj) in enumerate(state_cands):
                mod = _copy_obj(obj)
                _update_inplace(mod, h_raw, y_t, step, args.max_exemplars)
                mod_pool = [mod if o.obj_id == obj.obj_id else o for o in pool_objs]
                nll_us   = probe_nll(h_probes, y_probes, p_gpt_probes, mod_pool)
                records.append(_make_record(ACT_UPDATE_STATE, sim, rank, obj, nll_us))

            # ── UPDATE_BUFFER + PROMOTE_AND_UPDATE ────────────────────────────
            for rank, (sim, obj) in enumerate(buf_cands):
                # UPDATE_BUFFER
                mod = _copy_obj(obj)
                _update_inplace(mod, h_raw, y_t, step, args.max_exemplars)
                mod_pool = [mod if o.obj_id == obj.obj_id else o for o in pool_objs]
                nll_ub   = probe_nll(h_probes, y_probes, p_gpt_probes, mod_pool)
                records.append(_make_record(ACT_UPDATE_BUFFER, sim, rank, obj, nll_ub))

                # PROMOTE_AND_UPDATE (update first, then promote)
                mod2 = _copy_obj(obj)
                _update_inplace(mod2, h_raw, y_t, step, args.max_exemplars)
                mod2.is_promoted  = True
                mod2.promoted_step = step
                mod_pool2 = [mod2 if o.obj_id == obj.obj_id else o for o in pool_objs]
                nll_pu    = probe_nll(h_probes, y_probes, p_gpt_probes, mod_pool2)
                records.append(_make_record(ACT_PROMOTE_AND_UPDATE, sim, rank, obj, nll_pu))

            n_dp_done += 1
            if n_dp_done % 1000 == 0:
                print(f"  {n_dp_done}/{len(sampled_set)} decision points  "
                      f"(step={step:,}  n_pers={mem.n_persistent}  "
                      f"n_buf={mem.n_buffers}  n_records={len(records):,})")

        # ── oracle trajectory update ──────────────────────────────────────────
        if not mem.has_teacher(z_t):
            if mem.n_buffers < args.buffer_budget:
                mem.create_buffer(z_t, h_raw, y_t, step)
        else:
            mem.update(z_t, h_raw, y_t, step, args.max_exemplars,
                       args.promotion_support)

        recent_replay.append(step)
        if len(recent_replay) > MAX_REPLAY:
            recent_replay.pop(0)

    print(f"\n  Done. {len(records):,} action records from "
          f"{n_dp_done} decision points")

    # ── calibrate and add penalized reward ────────────────────────────────────
    df = pd.DataFrame(records)

    if len(df) > 0:
        gains = df["reward_full"].values
        gain_std = float(np.nanstd(gains))
        lam_create  = args.lambda_create  if args.lambda_create  >= 0 else 0.10 * gain_std
        lam_promote = args.lambda_promote if args.lambda_promote >= 0 else 0.05 * gain_std

        df["reward_penalized"] = (
            df["reward_full"]
            - args.lambda_history * df["history_damage"]
            - lam_create  * (df["action_type"] == ACT_CREATE_BUFFER).astype(float)
            - lam_promote * (df["action_type"] == ACT_PROMOTE_AND_UPDATE).astype(float)
        )
        print(f"  Calibrated: lambda_create={lam_create:.4f}  "
              f"lambda_promote={lam_promote:.4f}  gain_std={gain_std:.4f}")
    else:
        df["reward_penalized"] = 0.0

    # ── save ──────────────────────────────────────────────────────────────────
    df.to_parquet(out_dir / "actions.parquet", index=False)
    print(f"Saved: {out_dir / 'actions.parquet'}  ({len(df):,} rows)")

    sampled_df = pd.DataFrame({"step": sampled, "quartile": sampled * 4 // N})
    sampled_df.to_parquet(out_dir / "decision_points.parquet", index=False)

    schema = {
        "feature_names": FEATURE_NAMES,
        "reward_cols":   REWARD_COLS,
        "n_features":    N_FEAT,
        "action_types":  ACT_NAMES,
        "read_params":   {"k": READ_K, "tau": READ_TAU, "alpha": READ_ALPHA},
        "n_decision_points": n_dp_done,
        "n_records":         len(df),
    }
    with open(out_dir / "feature_schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()

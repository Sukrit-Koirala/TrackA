"""
build_adaptive_write_states.py  --  MVP 3a: Adaptive Online WRITE/UPDATE

Processes CT examples online in stream order.  For each example finds the
nearest existing state (cosine sim) and decides CREATE vs UPDATE.

Writer strategies
  adaptive_threshold    CREATE if nearest_sim < threshold
  budget_filling        CREATE while budget not full or state too large
  uncertainty_aware     CREATE if low sim AND high GPT NLL
  token_conflict        CREATE if high sim AND p_state(y) < prob_threshold

2-pass variants (controlled by --enable_split / --enable_prune):
  split_conflict        build first, then split high-entropy states
  prune_low_utility     build first, then drop tiny states

State file format (MVP 2b-compatible):
  prototype_h           [B, D]
  top_k_token_ids       [B, TOP_K]
  top_k_token_counts    [B, TOP_K]
  total_counts          [B]         true example count per state
  state_entropy         [B]
  state_purity          [B]
  mean_nll              [B]
  mean_gpt_entropy      [B]
  assigned_count        [B]         == total_counts for adaptive write
  assigned_indices      [N_ct]      state id each CT example was assigned to
  created_at            [B]
  last_updated          [B]
  num_states            int
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse, json, time
import numpy as np
import torch

from utils import get_device, set_seed

TOP_K  = 50
EPS    = 1e-10
CHUNK  = 512   # GPU matmul chunk for nearest-state search


# ── StateStore ────────────────────────────────────────────────────────────────

class StateStore:
    """Online state manager.  Prototype matrix lives on GPU; counts on CPU."""

    def __init__(self, budget: int, d: int, ema_cap: int, device, top_k: int = TOP_K):
        self.budget  = budget
        self.d       = d
        self.ema_cap = ema_cap
        self.device  = device
        self.top_k   = top_k
        self.n       = 0
        # GPU prototype matrix  [budget, D]
        self.proto   = torch.zeros(budget, d, device=device, dtype=torch.float32)
        # CPU accumulators
        self.full_counts  = []   # list[dict{tid: int}]
        self.total_cnt    = np.zeros(budget, np.float64)
        self.created_at   = np.zeros(budget, np.int32)
        self.last_upd     = np.zeros(budget, np.int32)
        self.sum_nll      = np.zeros(budget, np.float64)
        self.sum_gpt_ent  = np.zeros(budget, np.float64)

    def find_nearest_batch(self, h_chunk: torch.Tensor):
        """h_chunk [C, D] on device → (ids [C], sims [C]) CPU int/float."""
        if self.n == 0:
            C = len(h_chunk)
            return torch.full((C,), -1, dtype=torch.long), torch.full((C,), -1.0)
        sims = h_chunk @ self.proto[:self.n].T        # [C, n]
        best_sims, best_ids = sims.max(dim=1)
        return best_ids.cpu(), best_sims.cpu()

    def create(self, h_norm: torch.Tensor, y: int, nll: float, gpt_ent: float, idx: int) -> int:
        sid = self.n
        self.proto[sid]        = h_norm
        self.full_counts.append({y: 1})
        self.total_cnt[sid]    = 1.0
        self.created_at[sid]   = idx
        self.last_upd[sid]     = idx
        self.sum_nll[sid]      = nll
        self.sum_gpt_ent[sid]  = gpt_ent
        self.n += 1
        return sid

    def update(self, sid: int, h_norm: torch.Tensor, y: int,
               nll: float, gpt_ent: float, idx: int):
        cnt  = self.total_cnt[sid]
        lr   = 1.0 / max(1.0, min(cnt, float(self.ema_cap)))
        self.proto[sid] = (1.0 - lr) * self.proto[sid] + lr * h_norm
        n = self.proto[sid].norm()
        if n > EPS:
            self.proto[sid] = self.proto[sid] / n
        self.full_counts[sid][y] = self.full_counts[sid].get(y, 0) + 1
        self.total_cnt[sid]   += 1.0
        self.last_upd[sid]     = idx
        self.sum_nll[sid]     += nll
        self.sum_gpt_ent[sid] += gpt_ent

    def state_count(self, sid: int) -> float:
        return float(self.total_cnt[sid])

    def p_state(self, sid: int, y: int) -> float:
        total = self.total_cnt[sid]
        if total == 0:
            return 0.0
        return float(self.full_counts[sid].get(y, 0)) / total

    def pack(self, assignments: np.ndarray) -> dict:
        B        = self.n
        tok_ids  = np.zeros((B, self.top_k), dtype=np.int64)
        tok_cnts = np.zeros((B, self.top_k), dtype=np.float32)
        entropy  = np.zeros(B, dtype=np.float32)
        purity   = np.zeros(B, dtype=np.float32)

        for sid in range(B):
            counts = self.full_counts[sid]
            total  = self.total_cnt[sid]
            if not counts or total == 0:
                continue
            items = sorted(counts.items(), key=lambda x: -x[1])[:self.top_k]
            for j, (tid, cnt) in enumerate(items):
                tok_ids[sid, j]  = tid
                tok_cnts[sid, j] = float(cnt)
            probs = np.array([c / total for _, c in counts.items()], np.float64)
            probs /= probs.sum() + EPS
            entropy[sid] = float(-np.sum(probs * np.log(probs + EPS)))
            purity[sid]  = float(max(c / total for _, c in counts.items()))

        proto_np     = self.proto[:B].cpu().float().numpy()
        tc           = self.total_cnt[:B]
        mean_nll     = np.where(tc > 0, self.sum_nll[:B] / (tc + EPS), 0.0)
        mean_gpt_ent = np.where(tc > 0, self.sum_gpt_ent[:B] / (tc + EPS), 0.0)
        asgn_count   = np.bincount(assignments.clip(0), minlength=B).astype(np.int64)

        return {
            "prototype_h":        torch.tensor(proto_np,            dtype=torch.float32),
            "top_k_token_ids":    torch.tensor(tok_ids,             dtype=torch.long),
            "top_k_token_counts": torch.tensor(tok_cnts,            dtype=torch.float32),
            "total_counts":       torch.tensor(tc.astype(np.float32)),
            "state_entropy":      torch.tensor(entropy,             dtype=torch.float32),
            "state_purity":       torch.tensor(purity,              dtype=torch.float32),
            "mean_nll":           torch.tensor(mean_nll.astype(np.float32)),
            "mean_gpt_entropy":   torch.tensor(mean_gpt_ent.astype(np.float32)),
            "assigned_count":     torch.tensor(asgn_count,          dtype=torch.long),
            "assigned_indices":   torch.tensor(assignments,         dtype=torch.long),
            "created_at":         torch.tensor(self.created_at[:B], dtype=torch.long),
            "last_updated":       torch.tensor(self.last_upd[:B],   dtype=torch.long),
            "num_states":         B,
        }


# ── core online loop ──────────────────────────────────────────────────────────

def _online_loop(
    h_gpu: torch.Tensor,     # [N, D] on GPU (L2-normalised)
    y: np.ndarray,           # [N] int
    nll_gpt: np.ndarray,     # [N] float
    gpt_ent: np.ndarray,     # [N] float
    budget: int,
    ema_cap: int,
    device,
    should_create,           # fn(store, sid, sim, yi, nll, ent, idx) -> bool
    chunk: int = CHUNK,
) -> tuple:
    N     = len(y)
    store = StateStore(budget, h_gpu.shape[1], ema_cap, device)
    asgn  = np.full(N, -1, dtype=np.int32)

    for start in range(0, N, chunk):
        end   = min(start + chunk, N)
        blk   = h_gpu[start:end]                      # [C, D]
        C     = end - start
        b_ids, b_sims = store.find_nearest_batch(blk) # [C] CPU

        for i in range(C):
            idx  = start + i
            sid  = int(b_ids[i])
            sim  = float(b_sims[i])
            yi   = int(y[idx])
            nll  = float(nll_gpt[idx])
            ent  = float(gpt_ent[idx])
            h_i  = blk[i]

            do_create = (sid < 0) or should_create(store, sid, sim, yi, nll, ent, idx)
            if do_create and store.n < budget:
                new_sid    = store.create(h_i, yi, nll, ent, idx)
                asgn[idx]  = new_sid
            elif sid >= 0:
                store.update(sid, h_i, yi, nll, ent, idx)
                asgn[idx]  = sid
            # else: sid<0, budget full, nothing to assign (should not happen in practice)

        if (start // chunk) % 50 == 0:
            pct = 100 * start / N
            print(f"    {start:,}/{N:,}  ({pct:.0f}%)  n_states={store.n:,}", end="\r")

    print(f"    {N:,}/{N:,} (100%)  n_states={store.n:,}           ")
    return store, asgn


# ── writer variants ───────────────────────────────────────────────────────────

def build_adaptive_threshold(h_gpu, y, nll_gpt, gpt_ent, budget,
                              threshold, ema_cap, device):
    def sc(store, sid, sim, yi, nll, ent, idx):
        return sim < threshold
    store, asgn = _online_loop(h_gpu, y, nll_gpt, gpt_ent, budget, ema_cap, device, sc)
    return store.pack(asgn)


def build_budget_filling(h_gpu, y, nll_gpt, gpt_ent, budget,
                          threshold, warmup_fraction, max_state_count, ema_cap, device):
    warmup_n = int(budget * warmup_fraction)
    def sc(store, sid, sim, yi, nll, ent, idx):
        if store.n < warmup_n:
            return True   # create freely during warmup
        return sim < threshold or store.state_count(sid) > max_state_count
    store, asgn = _online_loop(h_gpu, y, nll_gpt, gpt_ent, budget, ema_cap, device, sc)
    return store.pack(asgn)


def build_uncertainty_aware(h_gpu, y, nll_gpt, gpt_ent, budget,
                             loose_threshold, nll_threshold, ema_cap, device):
    def sc(store, sid, sim, yi, nll, ent, idx):
        return sim < loose_threshold and nll > nll_threshold
    store, asgn = _online_loop(h_gpu, y, nll_gpt, gpt_ent, budget, ema_cap, device, sc)
    return store.pack(asgn)


def build_token_conflict(h_gpu, y, nll_gpt, gpt_ent, budget,
                          conflict_threshold, low_prob_threshold, ema_cap, device):
    def sc(store, sid, sim, yi, nll, ent, idx):
        # High sim but low state-probability for this token → new state
        return sim > conflict_threshold and store.p_state(sid, yi) < low_prob_threshold
    store, asgn = _online_loop(h_gpu, y, nll_gpt, gpt_ent, budget, ema_cap, device, sc)
    return store.pack(asgn)


# ── 2-pass: split high-entropy states ─────────────────────────────────────────

def split_high_entropy_states(state_dict: dict, h_gpu: torch.Tensor,
                               y: np.ndarray, nll_gpt: np.ndarray,
                               gpt_ent: np.ndarray, device,
                               entropy_threshold: float = 2.0,
                               purity_threshold: float = 0.5,
                               min_count: int = 4) -> dict:
    """Replace high-entropy states with 2 sub-states via mini-kmeans (1 iter)."""
    asgn    = state_dict["assigned_indices"].numpy().copy()
    entropy = state_dict["state_entropy"].numpy()
    purity  = state_dict["state_purity"].numpy()
    counts  = state_dict["assigned_count"].numpy()
    proto   = state_dict["prototype_h"].clone()   # [B, D]
    B       = len(proto)

    to_split = np.where(
        (entropy > entropy_threshold) & (purity < purity_threshold) & (counts >= min_count)
    )[0]
    if len(to_split) == 0:
        print(f"  [split] nothing to split (entropy>{entropy_threshold}, purity<{purity_threshold})")
        return state_dict

    print(f"  [split] splitting {len(to_split)} states (entropy>{entropy_threshold})")

    tok_ids  = state_dict["top_k_token_ids"].numpy().copy()
    tok_cnts = state_dict["top_k_token_counts"].numpy().copy()
    total_c  = state_dict["total_counts"].numpy().copy()
    cr_at    = state_dict["created_at"].numpy().copy()
    lu_at    = state_dict["last_updated"].numpy().copy()
    mnll     = state_dict["mean_nll"].numpy().copy()
    mgent    = state_dict["mean_gpt_entropy"].numpy().copy()

    new_proto  = [proto[s].clone() for s in range(B)]
    new_tokinf = [(tok_ids[s].copy(), tok_cnts[s].copy(), total_c[s],
                   cr_at[s], lu_at[s], mnll[s], mgent[s]) for s in range(B)]
    remap = np.arange(B, dtype=np.int64)   # old_sid -> new_sid

    extra_protos, extra_tokinf = [], []

    for old_sid in to_split:
        mask = asgn == old_sid
        idxs = np.where(mask)[0]
        if len(idxs) < 4:
            continue
        h_c = h_gpu[idxs]     # [M, D]

        # Init: random pair
        torch.manual_seed(int(old_sid))
        perm = torch.randperm(len(idxs))
        c0   = h_c[perm[0]].clone()
        c1   = h_c[perm[len(perm) // 2]].clone()
        c0   = c0 / (c0.norm() + EPS)
        c1   = c1 / (c1.norm() + EPS)

        # 1 iteration of k-means
        sims0 = (h_c @ c0)    # [M]
        sims1 = (h_c @ c1)    # [M]
        labels = (sims1 > sims0).cpu().numpy()   # 0 or 1 for each member

        def make_sub(label_val):
            sub_idxs = idxs[labels == label_val]
            if len(sub_idxs) == 0:
                return None
            h_sub  = h_gpu[sub_idxs]
            proto_ = h_sub.mean(0);  n_ = proto_.norm()
            if n_ > EPS:
                proto_ = proto_ / n_
            y_sub  = y[sub_idxs]
            counts_ = {}
            for t in y_sub:
                counts_[int(t)] = counts_.get(int(t), 0) + 1
            total_ = float(len(sub_idxs))
            items  = sorted(counts_.items(), key=lambda x: -x[1])[:TOP_K]
            tid_   = np.zeros(TOP_K, np.int64)
            tcnt_  = np.zeros(TOP_K, np.float32)
            for j, (t, c) in enumerate(items):
                tid_[j] = t; tcnt_[j] = float(c)
            return proto_, tid_, tcnt_, total_, sub_idxs

        sub0 = make_sub(0); sub1 = make_sub(1)
        if sub0 is None or sub1 is None:
            continue

        # Assign sub0 to old_sid (in place), add sub1 as new state
        new_sid1 = B + len(extra_protos)
        new_proto[old_sid] = sub0[0]
        new_tokinf[old_sid] = (sub0[1], sub0[2], sub0[3],
                                cr_at[old_sid], int(sub0[4].max()),
                                mnll[old_sid], mgent[old_sid])
        extra_protos.append(sub1[0])
        extra_tokinf.append((sub1[1], sub1[2], sub1[3],
                              cr_at[old_sid], int(sub1[4].max()),
                              mnll[old_sid], mgent[old_sid]))
        for xi in sub1[4]:
            asgn[xi] = new_sid1

    total_B = B + len(extra_protos)
    # Rebuild state dict
    all_proto = torch.stack(new_proto + [p for p in extra_protos])  # [total_B, D]
    all_info  = new_tokinf + extra_tokinf

    tk_ids  = np.zeros((total_B, TOP_K), np.int64)
    tk_cnts = np.zeros((total_B, TOP_K), np.float32)
    tc_arr  = np.zeros(total_B, np.float32)
    cr_arr  = np.zeros(total_B, np.int32)
    lu_arr  = np.zeros(total_B, np.int32)
    mn_arr  = np.zeros(total_B, np.float32)
    mg_arr  = np.zeros(total_B, np.float32)

    for s, (tid, tcnt, tot, cr, lu, mn, mg) in enumerate(all_info):
        tk_ids[s]  = tid;  tk_cnts[s] = tcnt
        tc_arr[s]  = float(tot)
        cr_arr[s]  = int(cr);  lu_arr[s] = int(lu)
        mn_arr[s]  = float(mn); mg_arr[s] = float(mg)

    ent_new  = np.zeros(total_B, np.float32)
    pur_new  = np.zeros(total_B, np.float32)
    for s in range(total_B):
        tot = float(tc_arr[s])
        if tot == 0:
            continue
        cnts_s = tk_cnts[s][tk_cnts[s] > 0]
        if len(cnts_s) == 0:
            continue
        p = cnts_s / (tot + EPS)
        ent_new[s] = float(-np.sum(p * np.log(p + EPS)))
        pur_new[s] = float(p.max())

    asgn_count = np.bincount(asgn.clip(0), minlength=total_B).astype(np.int64)

    return {
        "prototype_h":        all_proto.cpu().float(),
        "top_k_token_ids":    torch.tensor(tk_ids,   dtype=torch.long),
        "top_k_token_counts": torch.tensor(tk_cnts,  dtype=torch.float32),
        "total_counts":       torch.tensor(tc_arr),
        "state_entropy":      torch.tensor(ent_new,  dtype=torch.float32),
        "state_purity":       torch.tensor(pur_new,  dtype=torch.float32),
        "mean_nll":           torch.tensor(mn_arr),
        "mean_gpt_entropy":   torch.tensor(mg_arr),
        "assigned_count":     torch.tensor(asgn_count, dtype=torch.long),
        "assigned_indices":   torch.tensor(asgn,      dtype=torch.long),
        "created_at":         torch.tensor(cr_arr,    dtype=torch.long),
        "last_updated":       torch.tensor(lu_arr,    dtype=torch.long),
        "num_states":         total_B,
    }


# ── 2-pass: prune low-count states ────────────────────────────────────────────

def prune_low_count_states(state_dict: dict, min_count: int = 2) -> dict:
    """Remove states with assigned_count < min_count."""
    counts = state_dict["assigned_count"].numpy()
    keep   = np.where(counts >= min_count)[0]
    if len(keep) == len(counts):
        print(f"  [prune] nothing to prune (all >= {min_count})")
        return state_dict

    n_rm = len(counts) - len(keep)
    print(f"  [prune] removing {n_rm} states (count < {min_count}), keeping {len(keep)}")

    asgn    = state_dict["assigned_indices"].numpy().copy()
    old2new = np.full(len(counts), -1, dtype=np.int64)
    old2new[keep] = np.arange(len(keep), dtype=np.int64)

    # Reassign pruned states: find nearest kept prototype
    proto    = state_dict["prototype_h"]          # [B, D]
    keep_t   = torch.tensor(keep, dtype=torch.long)
    kept_proto = proto[keep_t]                    # [K, D]
    pruned   = np.setdiff1d(np.arange(len(counts)), keep)
    if len(pruned) > 0:
        # For examples assigned to pruned states, reassign to nearest kept
        for p_sid in pruned:
            mask = asgn == p_sid
            if not mask.any():
                continue
            p_vec = proto[p_sid].unsqueeze(0)      # [1, D]
            sims  = (p_vec @ kept_proto.T).squeeze(0)  # [K]
            nn    = int(sims.argmax())
            asgn[mask] = keep[nn]

    # Now remap assignments
    asgn_new = old2new[asgn.clip(0)]

    def sel(t):
        return t[keep_t] if isinstance(t, torch.Tensor) else t[keep]

    new_asgn_count = np.bincount(asgn_new.clip(0), minlength=len(keep)).astype(np.int64)

    return {
        "prototype_h":        sel(state_dict["prototype_h"]),
        "top_k_token_ids":    sel(state_dict["top_k_token_ids"]),
        "top_k_token_counts": sel(state_dict["top_k_token_counts"]),
        "total_counts":       sel(state_dict["total_counts"]),
        "state_entropy":      sel(state_dict["state_entropy"]),
        "state_purity":       sel(state_dict["state_purity"]),
        "mean_nll":           sel(state_dict["mean_nll"]),
        "mean_gpt_entropy":   sel(state_dict["mean_gpt_entropy"]),
        "assigned_count":     torch.tensor(new_asgn_count, dtype=torch.long),
        "assigned_indices":   torch.tensor(asgn_new, dtype=torch.long),
        "created_at":         sel(state_dict["created_at"]),
        "last_updated":       sel(state_dict["last_updated"]),
        "num_states":         len(keep),
    }


# ── variant configuration ─────────────────────────────────────────────────────

def get_variants(method: str, budgets: list, fast: bool, ema_caps: list) -> list:
    """Returns list of (tag, build_fn, budget) for a given method."""
    out = []
    for ema_cap in ema_caps:
        if method == "adaptive_threshold_write":
            thresholds = [0.990, 0.995] if fast else [0.985, 0.990, 0.995, 0.997]
            for thr in thresholds:
                for B in budgets:
                    tag = f"adaptive_threshold_thr{thr}_ema{ema_cap}_B{B}"
                    params = dict(threshold=thr, ema_cap=ema_cap)
                    out.append((tag, "adaptive_threshold", params, B))

        elif method == "budget_filling_write":
            combos = [(0.990, 0.25, 64)] if fast else [
                (0.990, 0.25, 32), (0.990, 0.25, 64),
                (0.995, 0.5, 64),  (0.995, 0.5, 128),
            ]
            for (thr, wf, ms) in combos:
                for B in budgets:
                    tag = f"budget_filling_thr{thr}_max{ms}_ema{ema_cap}_B{B}"
                    params = dict(threshold=thr, warmup_fraction=wf,
                                  max_state_count=ms, ema_cap=ema_cap)
                    out.append((tag, "budget_filling", params, B))

        elif method == "uncertainty_aware_write":
            loose_thrs = [0.995] if fast else [0.995, 0.998]
            for lt in loose_thrs:
                for B in budgets:
                    tag = f"uncertainty_aware_lt{lt}_ema{ema_cap}_B{B}"
                    params = dict(loose_threshold=lt, ema_cap=ema_cap,
                                  nll_q=75)   # 75th-pctile NLL as threshold
                    out.append((tag, "uncertainty_aware", params, B))

        elif method == "token_conflict_write":
            combos = [(0.990, 0.005)] if fast else [
                (0.990, 0.001), (0.990, 0.005), (0.995, 0.005), (0.995, 0.01),
            ]
            for (ct, lp) in combos:
                for B in budgets:
                    tag = f"token_conflict_ct{ct}_lp{lp}_ema{ema_cap}_B{B}"
                    params = dict(conflict_threshold=ct, low_prob_threshold=lp,
                                  ema_cap=ema_cap)
                    out.append((tag, "token_conflict", params, B))

        elif method == "split_conflict_states":
            base_thrs = [0.990] if fast else [0.990, 0.995]
            for thr in base_thrs:
                for B in budgets:
                    tag = f"split_conflict_thr{thr}_ema{ema_cap}_B{B}"
                    params = dict(threshold=thr, ema_cap=ema_cap,
                                  entropy_threshold=2.0, purity_threshold=0.5)
                    out.append((tag, "split_conflict", params, B))

        elif method == "prune_low_utility_states":
            base_thrs = [0.990] if fast else [0.990, 0.995]
            for thr in base_thrs:
                for B in budgets:
                    tag = f"prune_low_thr{thr}_ema{ema_cap}_B{B}"
                    params = dict(threshold=thr, ema_cap=ema_cap, min_count=2)
                    out.append((tag, "prune_low", params, B))

    return out


def run_variant(kind: str, params: dict, budget: int,
                h_gpu: torch.Tensor, y: np.ndarray,
                nll_gpt: np.ndarray, gpt_ent: np.ndarray,
                device) -> dict:
    ema = params.get("ema_cap", 128)

    if kind == "adaptive_threshold":
        return build_adaptive_threshold(
            h_gpu, y, nll_gpt, gpt_ent, budget,
            params["threshold"], ema, device)

    elif kind == "budget_filling":
        return build_budget_filling(
            h_gpu, y, nll_gpt, gpt_ent, budget,
            params["threshold"], params["warmup_fraction"],
            params["max_state_count"], ema, device)

    elif kind == "uncertainty_aware":
        nll_thr = float(np.percentile(nll_gpt, params.get("nll_q", 75)))
        print(f"    NLL threshold ({params.get('nll_q', 75)}th-pctile) = {nll_thr:.4f}")
        return build_uncertainty_aware(
            h_gpu, y, nll_gpt, gpt_ent, budget,
            params["loose_threshold"], nll_thr, ema, device)

    elif kind == "token_conflict":
        return build_token_conflict(
            h_gpu, y, nll_gpt, gpt_ent, budget,
            params["conflict_threshold"], params["low_prob_threshold"], ema, device)

    elif kind == "split_conflict":
        # Pass 1: adaptive_threshold base
        base = build_adaptive_threshold(
            h_gpu, y, nll_gpt, gpt_ent, budget,
            params["threshold"], ema, device)
        # Pass 2: split
        return split_high_entropy_states(
            base, h_gpu, y, nll_gpt, gpt_ent, device,
            params.get("entropy_threshold", 2.0),
            params.get("purity_threshold", 0.5))

    elif kind == "prune_low":
        # Pass 1: adaptive_threshold base
        base = build_adaptive_threshold(
            h_gpu, y, nll_gpt, gpt_ent, budget,
            params["threshold"], ema, device)
        # Pass 2: prune
        return prune_low_count_states(base, params.get("min_count", 2))

    else:
        raise ValueError(f"Unknown kind: {kind}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",    required=True)
    parser.add_argument("--output",    required=True)
    parser.add_argument("--methods",   nargs="+",
                        default=["budget_filling_write", "token_conflict_write"])
    parser.add_argument("--budgets",   nargs="+", type=int, default=[10000, 25000])
    parser.add_argument("--ema_caps",  nargs="+", type=int, default=[128])
    parser.add_argument("--fast",      action="store_true",
                        help="Fewer hyperparam combos (for quick first run)")
    parser.add_argument("--force",     action="store_true")
    parser.add_argument("--device",    default="cuda")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device({"device": args.device})
    src    = Path(args.source)
    out    = Path(args.output) / "states"
    out.mkdir(parents=True, exist_ok=True)

    print(f"\nbuild_adaptive_write_states")
    print(f"  Source:  {src}")
    print(f"  Output:  {out}")
    print(f"  Methods: {args.methods}")
    print(f"  Budgets: {args.budgets}")
    print(f"  Fast:    {args.fast}  Device: {device}")

    # Load CT data
    ct_path = src / "states" / "controller_train.pt"
    print(f"\nLoading CT data from {ct_path} ...")
    ct  = torch.load(ct_path, weights_only=False)
    h   = ct["h"].float()
    h   = h / (h.norm(dim=-1, keepdim=True) + EPS)   # L2-normalise
    y   = ct["y"].numpy().astype(np.int64)
    nll_gpt = ct["nll_gpt"].numpy().astype(np.float64)
    # Approximate GPT entropy using NLL (proxy)
    if "gpt_entropy" in ct:
        gpt_ent = ct["gpt_entropy"].numpy().astype(np.float64)
    else:
        gpt_ent = nll_gpt.copy()   # proxy
    N, D_h = h.shape
    print(f"  N={N:,}  D={D_h}")

    h_gpu = h.to(device)

    # Collect all variants
    all_variants = []
    for m in args.methods:
        all_variants.extend(get_variants(m, args.budgets, args.fast, args.ema_caps))

    print(f"\nTotal variants to build: {len(all_variants)}")

    manifest = []
    t0_all = time.time()

    for i, (tag, kind, params, budget) in enumerate(all_variants):
        out_path = out / f"{tag}.pt"
        print(f"\n[{i+1}/{len(all_variants)}] {tag}")

        if out_path.exists() and not args.force:
            print(f"  [cached] {out_path.name}")
            B = torch.load(out_path, weights_only=False).get("num_states", "?")
            manifest.append({"tag": tag, "kind": kind, "budget": budget,
                             "path": str(out_path), "status": "cached",
                             **{k: v for k, v in params.items() if k != "ema_cap"},
                             "ema_cap": params.get("ema_cap", 128)})
            continue

        t0 = time.time()
        print(f"  kind={kind}  budget={budget}  params={params}")
        try:
            state_dict = run_variant(kind, params, budget, h_gpu, y, nll_gpt, gpt_ent, device)
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback; traceback.print_exc()
            continue

        B_actual = state_dict["num_states"]
        elapsed  = time.time() - t0
        torch.save(state_dict, out_path)
        print(f"  Saved {out_path.name}  B={B_actual:,}  elapsed={elapsed:.0f}s")

        manifest.append({
            "tag": tag, "kind": kind, "budget": budget,
            "actual_num_states": B_actual,
            "path": str(out_path), "status": "built",
            "elapsed_s": elapsed,
            **{k: v for k, v in params.items() if k != "ema_cap"},
            "ema_cap": params.get("ema_cap", 128),
        })

    # Save manifest
    mnf_path = Path(args.output) / "states_manifest.json"
    with open(mnf_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved: {mnf_path}")
    print(f"Total elapsed: {time.time()-t0_all:.0f}s")
    print(f"Built/cached: {len(manifest)} variants")


if __name__ == "__main__":
    main()

"""
mvp4b1_common.py -- Shared definitions for MVP 4b.1: Clean On-Policy Bandit-WRITE

Key differences from MVP 4b:
  - PROMOTE_AND_UPDATE removed; promotion is automatic inside UPDATE_BUFFER
  - Single combined object_budget (n_persistent + n_buffers <= budget)
  - Four distinct reward columns: local / local_history / full / penalized
  - Atomic writes for all artifacts
  - Artifact versioning (mvp4b1_v1) with cache validation
  - 24 features (no act_promote_update)
"""

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

ARTIFACT_VERSION = "mvp4b1_v1"

# ── Action space (4 types, no PROMOTE_AND_UPDATE) ────────────────────────────
ACT_UPDATE_STATE  = 0
ACT_UPDATE_BUFFER = 1
ACT_CREATE_BUFFER = 2
ACT_DEFER         = 3
ACT_NAMES   = ["UPDATE_STATE", "UPDATE_BUFFER", "CREATE_BUFFER", "DEFER"]
N_ACT_TYPES = 4

# Fixed READ parameters (matches MVP 2c best)
READ_K     = 4
READ_TAU   = 0.05
READ_ALPHA = 0.75

# Feature schema (24 features)
FEATURE_NAMES = [
    # global (12)
    "stream_progress", "gpt_nll", "gpt_entropy",
    "n_persistent", "n_buffers",
    "budget_frac_persistent", "budget_frac_total",
    "state_top1_sim", "state_top2_sim", "state_sim_gap",
    "buffer_top1_sim", "buffer_top2_sim",
    # action one-hot (4)
    "act_update_state", "act_update_buffer",
    "act_create_buffer", "act_defer",
    # candidate (8)
    "cand_sim", "cand_rank_norm", "cand_log_support",
    "cand_is_persistent", "cand_entropy", "cand_p_y_true",
    "cand_age_frac", "cand_recency_frac",
]
N_FEAT = len(FEATURE_NAMES)  # 24

REWARD_COLS = [
    "reward_local",           # local geometric probes only
    "reward_local_history",   # local + candidate-history probes (no replay)
    "reward_full",            # local + history + replay probes
    "reward_penalized",       # full - action costs
    "nll_before",
    "nll_after",
    "history_damage",
]

TOP_K = 256   # top tokens per state in canonical schema


# ── LiveObj ───────────────────────────────────────────────────────────────────

@dataclass
class LiveObj:
    obj_id:           int
    teacher_id:       int        # oracle label; never used as feature/reward
    sum_h:            np.ndarray # [D] float64
    token_counts:     dict
    count:            int
    created_step:     int
    last_update_step: int
    is_promoted:      bool
    promoted_step:    int
    exemplars:        list       # reservoir-sampled step indices


def _proto(obj: LiveObj) -> np.ndarray:
    n = float(np.linalg.norm(obj.sum_h))
    return (obj.sum_h / max(n, 1e-8)).astype(np.float32)


def _entropy_d(d: dict, total: int) -> float:
    if total <= 0:
        return 0.0
    e = 0.0
    for c in d.values():
        p = c / total
        if p > 0:
            e -= p * math.log(p + 1e-12)
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


def _update_inplace(obj: LiveObj, h_raw: np.ndarray, y: int, step: int,
                    max_ex: int) -> None:
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
        sum_h=h_raw.astype(np.float64), token_counts={y: 1},
        count=1, created_step=step, last_update_step=step,
        is_promoted=False, promoted_step=-1, exemplars=[step],
    )


# ── LiveMemory (single combined budget) ───────────────────────────────────────

class LiveMemory:
    def __init__(self, object_budget: int, D: int, device: torch.device):
        self.object_budget = object_budget
        self.device        = device
        self.protos        = torch.zeros(object_budget, D, device=device)
        self.objects:         dict[int, LiveObj] = {}
        self.teacher_to_obj:  dict[int, int]     = {}
        self.obj_to_row:      dict[int, int]     = {}
        self.row_to_obj:      dict[int, int]     = {}
        self.persistent_rows: list               = []
        self.buffer_rows:     list               = []
        self._next_id  = 0
        self._next_row = 0

    def _alloc_row(self, obj_id: int) -> int:
        row = self._next_row
        self._next_row += 1
        self.obj_to_row[obj_id] = row
        self.row_to_obj[row]    = obj_id
        return row

    def _sync_proto(self, obj: LiveObj) -> None:
        row = self.obj_to_row[obj.obj_id]
        self.protos[row] = torch.from_numpy(_proto(obj)).to(self.device)

    def _auto_promote(self, obj: LiveObj, step: int, promotion_support: int) -> None:
        if not obj.is_promoted and obj.count >= promotion_support:
            obj.is_promoted   = True
            obj.promoted_step = step
            row = self.obj_to_row[obj.obj_id]
            if row in self.buffer_rows:
                self.buffer_rows.remove(row)
                self.persistent_rows.append(row)

    @property
    def n_persistent(self) -> int: return len(self.persistent_rows)
    @property
    def n_buffers(self)    -> int: return len(self.buffer_rows)
    @property
    def n_total(self)      -> int: return len(self.objects)
    @property
    def is_full(self)      -> bool: return self.n_total >= self.object_budget

    def has_teacher(self, tid: int) -> bool:
        return tid in self.teacher_to_obj

    def create_buffer(self, teacher_id: int, h_raw: np.ndarray,
                      y: int, step: int) -> Optional[LiveObj]:
        if self.is_full:
            return None
        obj = _make_new_obj(self._next_id, teacher_id, h_raw, y, step)
        self._next_id += 1
        self.objects[obj.obj_id]        = obj
        self.teacher_to_obj[teacher_id] = obj.obj_id
        row = self._alloc_row(obj.obj_id)
        self.buffer_rows.append(row)
        self._sync_proto(obj)
        return obj

    def update(self, obj_id: int, h_raw: np.ndarray, y: int, step: int,
               max_ex: int, promotion_support: int) -> LiveObj:
        obj = self.objects[obj_id]
        _update_inplace(obj, h_raw, y, step, max_ex)
        self._auto_promote(obj, step, promotion_support)
        self._sync_proto(obj)
        return obj

    def get_candidates(self, h_n: np.ndarray, k_states: int, k_buffers: int):
        h_gpu = torch.from_numpy(h_n).to(self.device).unsqueeze(0)

        state_cands, s_top1, s_top2 = [], 0.0, 0.0
        if self.persistent_rows:
            rows  = self.persistent_rows
            sims  = (h_gpu @ self.protos[rows].T).squeeze(0).cpu().numpy()
            k_eff = min(k_states, len(rows))
            top   = np.argsort(-sims)[:k_eff]
            for ri in top:
                state_cands.append((float(sims[ri]),
                                    self.objects[self.row_to_obj[rows[ri]]]))
            s     = np.sort(sims)[::-1]
            s_top1 = float(s[0]) if len(s) >= 1 else 0.0
            s_top2 = float(s[1]) if len(s) >= 2 else s_top1

        buf_cands, b_top1, b_top2 = [], 0.0, 0.0
        if self.buffer_rows:
            rows  = self.buffer_rows
            sims  = (h_gpu @ self.protos[rows].T).squeeze(0).cpu().numpy()
            k_eff = min(k_buffers, len(rows))
            top   = np.argsort(-sims)[:k_eff]
            for ri in top:
                buf_cands.append((float(sims[ri]),
                                  self.objects[self.row_to_obj[rows[ri]]]))
            s     = np.sort(sims)[::-1]
            b_top1 = float(s[0]) if len(s) >= 1 else 0.0
            b_top2 = float(s[1]) if len(s) >= 2 else b_top1

        return state_cands, buf_cands, (s_top1, s_top2, b_top1, b_top2)


# ── Probe NLL ─────────────────────────────────────────────────────────────────

def probe_nll(
    h_probes:     np.ndarray,
    y_probes:     np.ndarray,
    p_gpt_probes: np.ndarray,
    pool:         list,
    k:     int   = READ_K,
    tau:   float = READ_TAU,
    alpha: float = READ_ALPHA,
) -> np.ndarray:
    P = len(h_probes)
    if P == 0:
        return np.array([], dtype=np.float32)
    if not pool:
        return -np.log(np.maximum(p_gpt_probes, 1e-12)).astype(np.float32)

    protos  = np.stack([_proto(o) for o in pool])
    sims    = h_probes @ protos.T
    k_eff   = min(k, len(pool))
    top_idx = np.argsort(-sims, axis=1)[:, :k_eff]
    top_s   = np.take_along_axis(sims, top_idx, axis=1)
    shifted = top_s - top_s.max(axis=1, keepdims=True)
    w       = np.exp(shifted / (tau + 1e-8))
    w      /= w.sum(axis=1, keepdims=True) + 1e-10

    p_sy = np.zeros((P, k_eff), dtype=np.float32)
    for pi in range(P):
        y_p = int(y_probes[pi])
        for ki in range(k_eff):
            c = pool[int(top_idx[pi, ki])]
            p_sy[pi, ki] = c.token_counts.get(y_p, 0) / max(c.count, 1)

    p_local = (w * p_sy).sum(axis=1)
    p_final = np.maximum(alpha * p_gpt_probes + (1 - alpha) * p_local, 1e-12)
    return -np.log(p_final).astype(np.float32)


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(
    step: int, N: int,
    y_t: int, nll_gpt_t: float, gpt_entropy_t: float,
    n_persistent: int, n_buffers: int,
    object_budget: int,
    sim_info: tuple,
    cand_obj: Optional[LiveObj],
    cand_sim: float,
    cand_rank: int,
    action_type: int,
) -> np.ndarray:
    s_top1, s_top2, b_top1, b_top2 = sim_info
    n_total = n_persistent + n_buffers

    g = [
        step / max(N - 1, 1), float(nll_gpt_t), float(gpt_entropy_t),
        float(n_persistent), float(n_buffers),
        n_persistent / max(object_budget, 1),
        n_total      / max(object_budget, 1),
        s_top1, s_top2, s_top1 - s_top2, b_top1, b_top2,
    ]  # 12

    a = [0.0] * N_ACT_TYPES
    a[action_type] = 1.0   # 4

    if cand_obj is not None:
        p_y = cand_obj.token_counts.get(y_t, 0) / max(cand_obj.count, 1)
        c = [
            float(cand_sim),
            cand_rank / 8.0,
            math.log1p(cand_obj.count),
            float(cand_obj.is_promoted),
            _entropy_d(cand_obj.token_counts, cand_obj.count),
            float(p_y),
            (step - cand_obj.created_step)     / max(N, 1),
            (step - cand_obj.last_update_step) / max(N, 1),
        ]  # 8
    else:
        c = [0.0] * 8

    return np.array(g + a + c, dtype=np.float32)  # [24]


# ── Sampling ──────────────────────────────────────────────────────────────────

def stratified_sample(N: int, n: int, rng: np.random.Generator) -> np.ndarray:
    n_per = n // 4
    rem   = n - 4 * n_per
    idxs  = []
    for q in range(4):
        lo  = int(N * q / 4)
        hi  = int(N * (q + 1) / 4)
        cnt = min(n_per + (1 if q < rem else 0), hi - lo)
        if cnt > 0:
            idxs.append(rng.integers(lo, hi, size=cnt))
    return np.sort(np.concatenate(idxs)) if idxs else np.array([], dtype=np.int64)


# ── Atomic writes ─────────────────────────────────────────────────────────────

def atomic_write_json(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    with open(tmp) as f:
        json.load(f)            # validate; raises if corrupt
    tmp.rename(path)


def atomic_torch_save(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.pt")
    torch.save(obj, tmp)
    tmp.rename(path)


def atomic_parquet_save(df, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.rename(path)


def validate_json_cache(path: Path, required_keys: list[str],
                        expected_version: str = ARTIFACT_VERSION) -> Optional[dict]:
    """Load and validate a cached JSON. Returns dict or None if invalid/stale."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            d = json.load(f)
    except (json.JSONDecodeError, OSError):
        print(f"  [INVALID CACHE] corrupt JSON: {path}")
        path.unlink(missing_ok=True)
        return None
    if d.get("artifact_version") != expected_version:
        print(f"  [STALE CACHE] version {d.get('artifact_version')!r} != "
              f"{expected_version!r}: {path.name}")
        return None
    missing = [k for k in required_keys if k not in d]
    if missing:
        print(f"  [INVALID CACHE] missing keys {missing}: {path.name}")
        return None
    return d


# ── Canonical state schema (compatible with run_method_budget) ────────────────

def build_canonical_state(mem: LiveMemory, method: str, budget: int) -> Optional[dict]:
    """Build state dict matching build_predictive_states.compute_state_stats."""
    objs = sorted(mem.objects.values(), key=lambda o: o.obj_id)
    K = len(objs)
    if K == 0:
        return None
    D = len(objs[0].sum_h)

    proto   = np.zeros((K, D),     dtype=np.float32)
    tk_ids  = np.zeros((K, TOP_K), dtype=np.int32)
    tk_cnts = np.zeros((K, TOP_K), dtype=np.float32)
    totals  = np.zeros(K,          dtype=np.float32)
    acnt    = np.zeros(K,          dtype=np.int64)
    ent     = np.zeros(K,          dtype=np.float32)
    pur     = np.zeros(K,          dtype=np.float32)

    for i, obj in enumerate(objs):
        proto[i]   = _proto(obj)
        totals[i]  = float(obj.count)
        acnt[i]    = obj.count
        if obj.token_counts:
            toks = sorted(obj.token_counts.items(), key=lambda x: -x[1])
            for j, (tok, cnt) in enumerate(toks[:TOP_K]):
                tk_ids[i, j]  = tok
                tk_cnts[i, j] = float(cnt)
        c_arr = np.array(list(obj.token_counts.values()), dtype=np.float32)
        if len(c_arr) > 0:
            p = c_arr / (c_arr.sum() + 1e-12)
            ent[i] = float(-np.sum(p * np.log(p + 1e-12)))
            pur[i] = float(p.max())

    return {
        "artifact_version":    ARTIFACT_VERSION,
        "method":              method,
        "budget":              budget,
        "n_states":            K,
        "prototype_h":         torch.from_numpy(proto),
        "top_k_token_ids":     torch.from_numpy(tk_ids).long(),
        "top_k_token_counts":  torch.from_numpy(tk_cnts),
        "total_counts":        torch.from_numpy(totals),
        "assigned_count":      torch.from_numpy(acnt),
        "state_entropy":       torch.from_numpy(ent),
        "state_purity":        torch.from_numpy(pur),
        "mean_nll":            torch.zeros(K),
        "mean_gpt_entropy":    torch.zeros(K),
        "config":              {"method": method, "budget": budget,
                                "artifact_version": ARTIFACT_VERSION},
    }


def save_and_verify_state(mem: LiveMemory, method: str, budget: int,
                           states_dir: Path) -> Path:
    """Build canonical state, save atomically, verify round-trip load."""
    sd = build_canonical_state(mem, method, budget)
    if sd is None:
        raise RuntimeError("Cannot save state: memory is empty")

    states_dir.mkdir(parents=True, exist_ok=True)
    out_path = states_dir / f"{method}_B{budget}.pt"
    atomic_torch_save(out_path, sd)

    # Round-trip verification
    check = torch.load(out_path, map_location="cpu", weights_only=False)
    assert "prototype_h" in check, "Round-trip failed: missing prototype_h"
    assert check["prototype_h"].shape == sd["prototype_h"].shape
    K = sd["n_states"]
    print(f"  Saved state: {out_path}  ({K} objects, "
          f"{out_path.stat().st_size/1e6:.1f} MB)  [verified]")
    return out_path

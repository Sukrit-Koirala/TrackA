"""
build_write_imitation_dataset.py  --  MVP 4a Stage 2

Teacher-forced rollout over the datastore to generate supervised
WRITE action labels for imitation learning.

Gate chain: MATCH -> COMPATIBILITY -> AGGREGATE -> COMMIT
Labels:  UPDATE_STATE=0  UPDATE_BUFFER=1  CREATE_BUFFER=2  PROMOTE_BUFFER=3

Teacher ID is used ONLY for label generation, never as a policy input feature.

Saves:
  outputs_mvp4a.../datasets/<teacher_name>_write_imitation_dataset.pt
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse, json, time
import numpy as np
import torch

from utils import get_device, set_seed

# ── constants ─────────────────────────────────────────────────────────────────
TOP_K     = 50
EPS       = 1e-10
CHUNK_GPU = 512   # for nearest-neighbor GPU search batching

ACT_UPDATE_STATE   = 0
ACT_UPDATE_BUFFER  = 1
ACT_CREATE_BUFFER  = 2
ACT_PROMOTE_BUFFER = 3
ACTION_NAMES       = ["UPDATE_STATE", "UPDATE_BUFFER", "CREATE_BUFFER", "PROMOTE_BUFFER"]
N_ACTIONS          = 4

FEATURE_NAMES = [
    "nearest_state_sim",    "nearest_buffer_sim",
    "log1p_state_count",    "log1p_buffer_count",
    "state_entropy",        "buffer_entropy",
    "state_purity",         "buffer_purity",
    "p_state_y_true",       "p_buffer_y_true",
    "state_top1_prob",      "buffer_top1_prob",
    "log1p_state_age",      "log1p_buffer_age",
    "log1p_state_upd_age",  "log1p_buffer_upd_age",
    "gpt_entropy",          "gpt_nll",
    "gpt_top1_prob",        "gpt_margin",
    "state_budget_usage",   "buffer_budget_usage",
    "log1p_num_states",     "log1p_num_buffers",
    "progress",
    "has_state",            "has_buffer",
]
FEAT_DIM = len(FEATURE_NAMES)


# ── online memory management ──────────────────────────────────────────────────

def _entropy_dict(counts: dict, total: float) -> float:
    if total <= 0 or not counts:
        return 0.0
    p = np.fromiter(counts.values(), dtype=np.float64) / total
    return float(-np.sum(p * np.log(p + EPS)))


def _purity_dict(counts: dict, total: float) -> float:
    if total <= 0 or not counts:
        return 1.0
    return float(max(counts.values()) / total)


def _ema_update(proto: torch.Tensor, h: torch.Tensor, total: float, ema_cap: int) -> torch.Tensor:
    lr  = 1.0 / max(1.0, min(total, float(ema_cap)))
    upd = (1.0 - lr) * proto + lr * h
    n   = upd.norm()
    return (upd / n) if n > EPS else upd


class OnlineMemory:
    """Persistent states + aggregate buffers with teacher count tracking.

    Teacher IDs tracked only for label generation / diagnostics;
    not exposed to the learned policy at inference time.
    """

    def __init__(self, state_budget: int, buf_budget: int, d: int,
                 ema_cap: int = 128, device="cpu"):
        self.state_budget = state_budget
        self.buf_budget   = buf_budget
        self.d            = d
        self.ema_cap      = ema_cap
        self.device       = device

        # ── persistent states ─────────────────────────────────────────────────
        self.s_proto   = torch.zeros(state_budget, d, dtype=torch.float32)
        self.s_total   = np.zeros(state_budget, np.float64)
        self.s_tok     = [dict() for _ in range(state_budget)]
        self.s_teach   = [dict() for _ in range(state_budget)]
        self.s_nll_sum = np.zeros(state_budget, np.float64)
        self.s_ent_sum = np.zeros(state_budget, np.float64)
        self.s_created = np.zeros(state_budget, np.int32)
        self.s_last    = np.zeros(state_budget, np.int32)
        self.n_states  = 0
        self._s_gpu    = None    # lazy GPU copy
        self._s_stale  = True

        # ── aggregate buffers ─────────────────────────────────────────────────
        self.b_proto   = torch.zeros(buf_budget, d, dtype=torch.float32)
        self.b_total   = np.zeros(buf_budget, np.float64)
        self.b_tok     = [dict() for _ in range(buf_budget)]
        self.b_teach   = [dict() for _ in range(buf_budget)]
        self.b_nll_sum = np.zeros(buf_budget, np.float64)
        self.b_ent_sum = np.zeros(buf_budget, np.float64)
        self.b_created = np.zeros(buf_budget, np.int32)
        self.b_last    = np.zeros(buf_budget, np.int32)
        self.b_active  = np.zeros(buf_budget, bool)
        self.n_buffers = 0
        self._b_free   = list(range(buf_budget))
        self._b_gpu    = None
        self._b_ids    = None   # active buf indices (for GPU lookup)
        self._b_stale  = True

    # ── GPU nearest-neighbor ──────────────────────────────────────────────────

    def _refresh_s(self):
        if self._s_stale and self.n_states > 0:
            self._s_gpu = self.s_proto[:self.n_states].to(self.device)
            self._s_stale = False

    def _refresh_b(self):
        if self._b_stale and self.n_buffers > 0:
            ids = np.where(self.b_active)[0]
            self._b_ids = ids
            self._b_gpu = self.b_proto[ids].to(self.device)
            self._b_stale = False

    def find_nearest_state(self, h_gpu):
        """h_gpu: normalised [d] tensor on device. Returns (sid, sim)."""
        if self.n_states == 0:
            return -1, -1.0
        self._refresh_s()
        sims = self._s_gpu @ h_gpu
        best = int(sims.argmax())
        return best, float(sims[best])

    def find_nearest_buffer(self, h_gpu):
        """Returns (bid, sim) or (-1, -1.0)."""
        if self.n_buffers == 0:
            return -1, -1.0
        self._refresh_b()
        sims = self._b_gpu @ h_gpu
        best_local = int(sims.argmax())
        return int(self._b_ids[best_local]), float(sims[best_local])

    # ── state operations ──────────────────────────────────────────────────────

    def create_state(self, h_cpu, y, nll, ent, teacher_id, idx) -> int:
        if self.n_states >= self.state_budget:
            return -1
        sid = self.n_states
        self.s_proto[sid]   = h_cpu
        self.s_total[sid]   = 1.0
        self.s_tok[sid]     = {y: 1}
        self.s_teach[sid]   = {teacher_id: 1} if teacher_id >= 0 else {}
        self.s_nll_sum[sid] = nll
        self.s_ent_sum[sid] = ent
        self.s_created[sid] = idx
        self.s_last[sid]    = idx
        self.n_states      += 1
        self._s_stale       = True
        return sid

    def update_state(self, sid, h_cpu, y, nll, ent, teacher_id, idx):
        self.s_total[sid] += 1
        self.s_proto[sid]  = _ema_update(self.s_proto[sid], h_cpu, self.s_total[sid], self.ema_cap)
        self.s_tok[sid][y]  = self.s_tok[sid].get(y, 0) + 1
        if teacher_id >= 0:
            self.s_teach[sid][teacher_id] = self.s_teach[sid].get(teacher_id, 0) + 1
        self.s_nll_sum[sid] += nll
        self.s_ent_sum[sid] += ent
        self.s_last[sid]     = idx
        self._s_stale        = True

    # ── buffer operations ─────────────────────────────────────────────────────

    def create_buffer(self, h_cpu, y, nll, ent, teacher_id, idx) -> int:
        if not self._b_free:
            self._evict_weakest(idx)
        if not self._b_free:
            return -1
        bid = self._b_free.pop(0)
        self.b_proto[bid]   = h_cpu
        self.b_total[bid]   = 1.0
        self.b_tok[bid]     = {y: 1}
        self.b_teach[bid]   = {teacher_id: 1} if teacher_id >= 0 else {}
        self.b_nll_sum[bid] = nll
        self.b_ent_sum[bid] = ent
        self.b_created[bid] = idx
        self.b_last[bid]    = idx
        self.b_active[bid]  = True
        self.n_buffers     += 1
        self._b_stale       = True
        return bid

    def update_buffer(self, bid, h_cpu, y, nll, ent, teacher_id, idx):
        self.b_total[bid] += 1
        self.b_proto[bid]  = _ema_update(self.b_proto[bid], h_cpu, self.b_total[bid], self.ema_cap)
        self.b_tok[bid][y]  = self.b_tok[bid].get(y, 0) + 1
        if teacher_id >= 0:
            self.b_teach[bid][teacher_id] = self.b_teach[bid].get(teacher_id, 0) + 1
        self.b_nll_sum[bid] += nll
        self.b_ent_sum[bid] += ent
        self.b_last[bid]     = idx
        self._b_stale        = True

    def promote_buffer(self, bid, h_cpu, y, nll, ent, teacher_id, idx) -> int:
        """Promote buffer bid → persistent state; add current example to state."""
        if self.n_states >= self.state_budget:
            return -1
        sid = self.n_states
        self.s_proto[sid]   = self.b_proto[bid].clone()
        self.s_total[sid]   = self.b_total[bid]
        self.s_tok[sid]     = dict(self.b_tok[bid])
        self.s_teach[sid]   = dict(self.b_teach[bid])
        self.s_nll_sum[sid] = self.b_nll_sum[bid]
        self.s_ent_sum[sid] = self.b_ent_sum[bid]
        self.s_created[sid] = self.b_created[bid]
        self.s_last[sid]    = self.b_last[bid]
        self.n_states      += 1
        self._release_buf(bid)
        self.update_state(sid, h_cpu, y, nll, ent, teacher_id, idx)
        self._s_stale = True
        return sid

    def _release_buf(self, bid):
        self.b_active[bid]  = False
        self.b_total[bid]   = 0
        self.b_tok[bid]     = {}
        self.b_teach[bid]   = {}
        self.n_buffers     -= 1
        self._b_free.append(bid)
        self._b_stale       = True

    def _evict_weakest(self, current_idx):
        active = np.where(self.b_active)[0]
        if len(active) == 0:
            return
        ages   = current_idx - self.b_last[active].astype(np.float64)
        totals = self.b_total[active]
        scores = -totals + 0.01 * ages
        worst  = active[int(scores.argmax())]
        self._release_buf(int(worst))

    # ── state attribute helpers ───────────────────────────────────────────────

    def s_entropy(self, sid):
        return _entropy_dict(self.s_tok[sid], self.s_total[sid])

    def s_purity(self, sid):
        return _purity_dict(self.s_tok[sid], self.s_total[sid])

    def b_entropy(self, bid):
        return _entropy_dict(self.b_tok[bid], self.b_total[bid])

    def b_purity(self, bid):
        return _purity_dict(self.b_tok[bid], self.b_total[bid])

    def s_p_y(self, sid, y) -> float:
        tc = self.s_tok[sid]
        return tc.get(y, 0) / max(self.s_total[sid], 1)

    def b_p_y(self, bid, y) -> float:
        tc = self.b_tok[bid]
        return tc.get(y, 0) / max(self.b_total[bid], 1)

    def s_top1_prob(self, sid) -> float:
        tc = self.s_tok[sid]
        if not tc:
            return 0.0
        return max(tc.values()) / max(self.s_total[sid], 1)

    def b_top1_prob(self, bid) -> float:
        tc = self.b_tok[bid]
        if not tc:
            return 0.0
        return max(tc.values()) / max(self.b_total[bid], 1)

    def s_teacher_prob(self, sid, teacher_id) -> float:
        return self.s_teach[sid].get(teacher_id, 0) / max(self.s_total[sid], 1)

    def b_teacher_prob(self, bid, teacher_id) -> float:
        return self.b_teach[bid].get(teacher_id, 0) / max(self.b_total[bid], 1)

    # ── export ────────────────────────────────────────────────────────────────

    def export_state_file(self, method, teacher_name, budget, action_counts, hyperparams):
        B = self.n_states
        if B == 0:
            raise RuntimeError("No states to export.")
        top_ids  = torch.zeros(B, TOP_K, dtype=torch.int32)
        top_cnts = torch.zeros(B, TOP_K, dtype=torch.float32)
        total_c  = torch.zeros(B, dtype=torch.float32)
        asgn_c   = torch.zeros(B, dtype=torch.int64)
        s_ent    = torch.zeros(B, dtype=torch.float32)
        s_pur    = torch.zeros(B, dtype=torch.float32)
        m_nll    = torch.zeros(B, dtype=torch.float32)
        m_gent   = torch.zeros(B, dtype=torch.float32)
        t_purity = []

        for i in range(B):
            tc   = self.s_tok[i]
            tot  = float(self.s_total[i])
            total_c[i] = tot
            asgn_c[i]  = int(tot)
            if tc:
                items = sorted(tc.items(), key=lambda x: -x[1])[:TOP_K]
                for j, (tok, cnt) in enumerate(items):
                    top_ids[i, j]  = tok
                    top_cnts[i, j] = cnt
            s_ent[i] = self.s_entropy(i)
            s_pur[i] = self.s_purity(i)
            m_nll[i] = float(self.s_nll_sum[i] / max(tot, 1))
            m_gent[i] = float(self.s_ent_sum[i] / max(tot, 1))
            teach = self.s_teach[i]
            if teach and tot > 0:
                t_purity.append(max(teach.values()) / tot)
            else:
                t_purity.append(1.0)

        t_stats = {
            "mean_teacher_purity":   float(np.mean(t_purity)),
            "median_teacher_purity": float(np.median(t_purity)),
        }
        ac = {ACTION_NAMES[k]: int(v) for k, v in action_counts.items()}

        return {
            "prototype_h":        self.s_proto[:B].clone(),
            "top_k_token_ids":    top_ids,
            "top_k_token_counts": top_cnts,
            "total_counts":       total_c,
            "assigned_count":     asgn_c,
            "state_entropy":      s_ent,
            "state_purity":       s_pur,
            "mean_nll":           m_nll,
            "mean_gpt_entropy":   m_gent,
            "method":             method,
            "teacher_name":       teacher_name,
            "budget":             budget,
            "actual_num_states":  B,
            "num_buffers_final":  int(self.n_buffers),
            "action_counts":      ac,
            "teacher_alignment_stats": t_stats,
            "hyperparams":        hyperparams,
        }


# ── feature building ──────────────────────────────────────────────────────────

def build_features(
    mem: OnlineMemory,
    h_gpu,       # normalised [d] tensor on device
    y: int,
    gpt_nll: float,
    gpt_entropy: float,
    gpt_top1_prob: float,
    gpt_top2_prob: float,
    idx: int,
    N: int,
    s_id: int,
    s_sim: float,
    b_id: int,
    b_sim: float,
) -> np.ndarray:
    feat = np.zeros(FEAT_DIM, dtype=np.float32)

    has_s = s_id >= 0
    has_b = b_id >= 0

    # State features
    if has_s:
        s_cnt = float(mem.s_total[s_id])
        s_ent = mem.s_entropy(s_id)
        s_pur = mem.s_purity(s_id)
        p_s_y = mem.s_p_y(s_id, y)
        s_t1  = mem.s_top1_prob(s_id)
        s_age = idx - int(mem.s_created[s_id])
        s_upd = idx - int(mem.s_last[s_id])
    else:
        s_cnt = s_ent = s_pur = p_s_y = s_t1 = s_age = s_upd = 0.0

    # Buffer features
    if has_b:
        b_cnt = float(mem.b_total[b_id])
        b_ent = mem.b_entropy(b_id)
        b_pur = mem.b_purity(b_id)
        p_b_y = mem.b_p_y(b_id, y)
        b_t1  = mem.b_top1_prob(b_id)
        b_age = idx - int(mem.b_created[b_id])
        b_upd = idx - int(mem.b_last[b_id])
    else:
        b_cnt = b_ent = b_pur = p_b_y = b_t1 = b_age = b_upd = 0.0

    feat[0]  = float(s_sim) if has_s else 0.0
    feat[1]  = float(b_sim) if has_b else 0.0
    feat[2]  = float(np.log1p(s_cnt))
    feat[3]  = float(np.log1p(b_cnt))
    feat[4]  = s_ent
    feat[5]  = b_ent
    feat[6]  = s_pur
    feat[7]  = b_pur
    feat[8]  = p_s_y
    feat[9]  = p_b_y
    feat[10] = s_t1
    feat[11] = b_t1
    feat[12] = float(np.log1p(s_age))
    feat[13] = float(np.log1p(b_age))
    feat[14] = float(np.log1p(s_upd))
    feat[15] = float(np.log1p(b_upd))
    feat[16] = gpt_entropy
    feat[17] = gpt_nll
    feat[18] = gpt_top1_prob
    feat[19] = gpt_top1_prob - gpt_top2_prob   # margin
    feat[20] = mem.n_states  / max(mem.state_budget, 1)
    feat[21] = mem.n_buffers / max(mem.buf_budget, 1)
    feat[22] = float(np.log1p(mem.n_states))
    feat[23] = float(np.log1p(mem.n_buffers))
    feat[24] = idx / max(N - 1, 1)
    feat[25] = 1.0 if has_s else 0.0
    feat[26] = 1.0 if has_b else 0.0
    return feat


# ── label generation ──────────────────────────────────────────────────────────

def get_action_label(
    mem: OnlineMemory,
    s_id: int, b_id: int,
    s_sim: float, b_sim: float,
    teacher_id: int,
    y_i: int, nll_i: float,
    hp: dict,
) -> int:
    """Priority: PROMOTE > UPDATE_STATE > UPDATE_BUFFER > CREATE_BUFFER.

    UPDATE_STATE uses a compatibility criterion: state predicts the true token
    better than GPT-2 does. This signal (p_state_y_true vs exp(-gpt_nll)) is
    already in the feature vector, so the MLP can learn it directly. Teacher ID
    is retained only for diagnostics; it does not drive any label decision.

    PROMOTE is count-based with a minimum similarity check. UPDATE_BUFFER is
    geometry-based (buffer exists and is close enough, or too young to judge).
    """
    buf_sim_thr = hp.get("buf_sim_threshold", 0.3)
    gpt_p_y     = float(np.exp(-nll_i))   # GPT-2's prob for true token y_i

    # 1. PROMOTE_BUFFER? — buffer mature, token-pure, AND still attracting examples
    if b_id >= 0 and mem.n_states < mem.state_budget:
        b_cnt = float(mem.b_total[b_id])
        b_pur = mem.b_purity(b_id)          # max token fraction (self-purity, not teacher)
        promo_purity = hp.get("promote_purity", 0.0)
        if b_cnt >= hp["promote_count"] and b_pur >= promo_purity and b_sim >= buf_sim_thr:
            return ACT_PROMOTE_BUFFER

    # 2. UPDATE_STATE? — state is a better predictor of y_i than GPT-2
    if s_id >= 0:
        s_cnt = float(mem.s_total[s_id])
        s_p_y = mem.s_p_y(s_id, y_i)
        if s_p_y >= gpt_p_y and s_cnt >= hp["min_state_count"]:
            return ACT_UPDATE_STATE

    # 3. UPDATE_BUFFER? — buffer close enough (no age exemption; sim-only criterion)
    if b_id >= 0 and b_sim >= buf_sim_thr:
        return ACT_UPDATE_BUFFER

    # 4. CREATE_BUFFER (default)
    return ACT_CREATE_BUFFER


# ── rollout ───────────────────────────────────────────────────────────────────

def run_rollout(ct_data, teacher_ids, state_budget, buf_budget, device, hp):
    N  = len(teacher_ids)
    d  = ct_data["h"].shape[1]
    mem = OnlineMemory(state_budget, buf_budget, d, hp.get("ema_cap", 128), device)

    h_all   = ct_data["h"].float()
    y_all   = ct_data["y"].numpy()
    nll_all = ct_data["nll_gpt"].numpy()
    ent_all = ct_data["gpt_entropy"].numpy()
    t1_all  = ct_data["gpt_top1_prob"].numpy()
    t2_all  = ct_data["gpt_top2_prob"].numpy()

    X       = np.zeros((N, FEAT_DIM), dtype=np.float32)
    y_act   = np.zeros(N, dtype=np.int64)
    act_cnt = {k: 0 for k in range(N_ACTIONS)}

    _norm_fn = lambda h: h / (h.norm() + EPS)

    print(f"    Rollout N={N:,} state_budget={state_budget} buf_budget={buf_budget}")

    for i in range(N):
        h_cpu = h_all[i]                              # [d] float32 cpu
        h_gpu = _norm_fn(h_cpu).to(device)
        y_i   = int(y_all[i])
        nll_i = float(nll_all[i])
        ent_i = float(ent_all[i])
        t1_i  = float(t1_all[i])
        t2_i  = float(t2_all[i])
        tid   = int(teacher_ids[i])

        s_id, s_sim = mem.find_nearest_state(h_gpu)
        b_id, b_sim = mem.find_nearest_buffer(h_gpu)

        feat = build_features(
            mem, h_gpu, y_i, nll_i, ent_i, t1_i, t2_i,
            i, N, s_id, s_sim, b_id, b_sim,
        )

        label = get_action_label(mem, s_id, b_id, s_sim, b_sim, tid, y_i, nll_i, hp)

        # Execute teacher action on memory
        if label == ACT_PROMOTE_BUFFER:
            new_sid = mem.promote_buffer(b_id, h_cpu, y_i, nll_i, ent_i, tid, i)
            if new_sid < 0:   # fallback: budget full
                if b_id >= 0:
                    mem.update_buffer(b_id, h_cpu, y_i, nll_i, ent_i, tid, i)
                    label = ACT_UPDATE_BUFFER
                else:
                    mem.create_buffer(h_cpu, y_i, nll_i, ent_i, tid, i)
                    label = ACT_CREATE_BUFFER

        elif label == ACT_UPDATE_STATE:
            if s_id >= 0:
                mem.update_state(s_id, h_cpu, y_i, nll_i, ent_i, tid, i)
            else:
                mem.create_buffer(h_cpu, y_i, nll_i, ent_i, tid, i)
                label = ACT_CREATE_BUFFER

        elif label == ACT_UPDATE_BUFFER:
            if b_id >= 0:
                mem.update_buffer(b_id, h_cpu, y_i, nll_i, ent_i, tid, i)
            else:
                mem.create_buffer(h_cpu, y_i, nll_i, ent_i, tid, i)
                label = ACT_CREATE_BUFFER

        else:   # CREATE_BUFFER
            mem.create_buffer(h_cpu, y_i, nll_i, ent_i, tid, i)

        X[i]      = feat
        y_act[i]  = label
        act_cnt[label] = act_cnt.get(label, 0) + 1

        if (i + 1) % 10000 == 0:
            pct = (i + 1) / N * 100
            dist = " ".join(f"{ACTION_NAMES[k]}={act_cnt[k]}" for k in range(N_ACTIONS))
            print(f"      {i+1}/{N} ({pct:.0f}%)  states={mem.n_states}  bufs={mem.n_buffers}  {dist}")

    return X, y_act, act_cnt, mem


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",        required=True)
    parser.add_argument("--teachers_dir",  required=True,
                        help="Directory with *_assignments.pt files")
    parser.add_argument("--teachers",      nargs="+",
                        default=["utility_weighted_B10000", "minibatch_kmeans_B10000"])
    parser.add_argument("--output",        required=True)
    parser.add_argument("--state_budget",  type=int, default=10000)
    parser.add_argument("--buf_frac",      type=float, default=0.05,
                        help="Buffer budget as fraction of state_budget")
    parser.add_argument("--ema_cap",       type=int, default=128)
    # Label hyperparameters
    parser.add_argument("--teacher_match",      type=float, default=0.3)
    parser.add_argument("--promote_count",      type=int,   default=4)
    parser.add_argument("--promote_purity",     type=float, default=0.5)
    parser.add_argument("--state_sim_threshold",type=float, default=0.5,
                        help="Cosine sim threshold for UPDATE_STATE label")
    parser.add_argument("--buf_sim_threshold",  type=float, default=0.3,
                        help="Cosine sim threshold for UPDATE_BUFFER/PROMOTE label")
    parser.add_argument("--min_state_cnt", type=int,   default=2)
    parser.add_argument("--min_buf_cnt",   type=int,   default=4)
    parser.add_argument("--device",        default="cuda")
    parser.add_argument("--force",         action="store_true")
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device  = get_device({"device": args.device})
    src     = Path(args.source)
    tdir    = Path(args.teachers_dir)
    out     = Path(args.output) / "datasets"
    out.mkdir(parents=True, exist_ok=True)

    buf_budget = max(200, int(args.state_budget * args.buf_frac))

    hp = {
        "teacher_match_threshold":  args.teacher_match,
        "promote_count":            args.promote_count,
        "promote_purity":           args.promote_purity,   # token self-purity for promotion
        "state_sim_threshold":      args.state_sim_threshold,
        "buf_sim_threshold":        args.buf_sim_threshold,
        "min_state_count":          args.min_state_cnt,
        "min_buf_count_for_judgment": args.min_buf_cnt,
        "ema_cap":                  args.ema_cap,
        "state_budget":             args.state_budget,
        "buf_budget":               buf_budget,
    }

    print(f"\nbuild_write_imitation_dataset")
    print(f"  Source:  {src}")
    print(f"  HP:      {hp}")

    ct_data = torch.load(src / "states" / "controller_train.pt", weights_only=False)

    for teacher_name in args.teachers:
        out_path = out / f"{teacher_name}_write_imitation_dataset.pt"
        if out_path.exists() and not args.force:
            print(f"\n  [cached] {teacher_name}")
            continue

        asgn_path = tdir / f"{teacher_name}_assignments.pt"
        if not asgn_path.exists():
            print(f"\n  [SKIP] {teacher_name} — assignments not found: {asgn_path}")
            continue

        print(f"\n  [{teacher_name}]")
        t0 = time.time()

        asgn = torch.load(asgn_path, weights_only=False)
        teacher_ids = asgn["teacher_ids"].numpy()   # [N]

        X, y_act, act_cnt, mem = run_rollout(
            ct_data, teacher_ids, args.state_budget, buf_budget, device, hp
        )

        act_dist = {ACTION_NAMES[k]: int(act_cnt[k]) for k in range(N_ACTIONS)}
        print(f"    Action distribution: {act_dist}")
        print(f"    Final states={mem.n_states}  buffers={mem.n_buffers}  elapsed={time.time()-t0:.0f}s")

        save = {
            "X":             torch.tensor(X),
            "y_action":      torch.tensor(y_act),
            "feature_names": FEATURE_NAMES,
            "action_names":  ACTION_NAMES,
            "teacher_name":  teacher_name,
            "hyperparams":   hp,
            "action_distribution": act_dist,
        }
        torch.save(save, out_path)
        print(f"    Saved {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

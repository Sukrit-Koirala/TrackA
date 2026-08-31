"""
train_q_state_read.py  —  MVP 2c: Q-State-Read Controller

Core module.  Provides:
  build_action_grid        build full or fast 217/80-action grid
  build_action_features    [A, 6] feature matrix
  build_obs_features       [N, 22] observation features per query
  compute_reward_matrix    [N, A]  reward = NLL_GPT - NLL_final
  find_best_fixed          (idx, action) with highest mean CT reward
  QStateReadMLP            28 -> 128 -> 128 -> 1
  train_q_mlp              offline fitted-Q with early stopping
  apply_q_controller       [N] argmax-Q chosen action indices
  eval_on_val              returns dict of NLL stats
  compute_oracle_nll       per-example best action NLL on val
  run_method_budget        end-to-end for one (method, budget)

Observation features (22):
  0  gpt_entropy (nll_gpt proxy)
  1  log_gpt_top1_prob
  2  gpt_margin                  (0 -- full dist unavailable)
  3  nearest_state_sim
  4  mean_top2_state_sim
  5  mean_top4_state_sim
  6  mean_top8_state_sim
  7  std_top8_state_sim
  8  top1_state_entropy
  9  mean_top4_state_entropy
 10  mean_top8_state_entropy
 11  top1_state_purity
 12  mean_top4_state_purity
 13  top1_state_count_log
 14  mean_top4_state_count_log
 15  mean_top8_state_count_log
 16  agreement_gpt_top1_vs_state_top1  (0 -- GPT top-1 token unavailable)
 17  agreement_gpt_top1_vs_top4_state  (0 -- GPT top-1 token unavailable)
 18  top1_state_top_token_prob
 19  mean_top4_state_top_token_prob
 20  sim_gap_top1_top2
 21  sim_gap_top1_top4

Action features (6):
  0  k_states / 32
  1  tau
  2  alpha
  3  beta / 10
  4  is_gpt_only
  5  uses_state_read
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

from utils import get_device, set_seed, ppl
from build_neighbors import cosine_top_k
from evaluate_predictive_states import (
    compute_top_k_state_sims,
    precompute_lookup,
    build_global_freq,
)

MAX_K  = 32
OBS_DIM = 22
ACT_DIM = 6
Q_DIM   = OBS_DIM + ACT_DIM  # 28
EPS     = 1e-10
VOCAB   = 50257
Q_CHUNK = 256
DS_CHUNK = 8192


# ── action grid ───────────────────────────────────────────────────────────────

def build_action_grid(fast: bool = False) -> list[dict]:
    acts = [{"name": "gpt_only", "k_states": 0, "tau": 0.1, "alpha": 1.0, "beta": 0.0}]
    k_vals   = [1, 2, 4, 8, 16]         if fast else [1, 2, 4, 8, 16, 32]
    tau_vals = [0.05, 0.1]              if fast else [0.05, 0.1, 0.2]
    a_vals   = [0.5, 0.75]              if fast else [0.25, 0.5, 0.75]
    b_vals   = [0.0, 5.0]              if fast else [0.0, 1.0, 5.0, 10.0]
    for k in k_vals:
        for tau in tau_vals:
            for alpha in a_vals:
                for beta in b_vals:
                    acts.append({"name": f"k{k}_t{tau}_a{alpha}_b{beta}",
                                  "k_states": k, "tau": tau,
                                  "alpha": alpha, "beta": beta})
    return acts


def build_action_features(actions: list[dict]) -> torch.Tensor:
    """[A, 6]"""
    rows = []
    for a in actions:
        k = a["k_states"]
        rows.append([k / MAX_K, a["tau"], a["alpha"], a["beta"] / 10.0,
                     float(k == 0), float(k > 0)])
    return torch.tensor(rows, dtype=torch.float32)


# ── observation features ──────────────────────────────────────────────────────

def build_obs_features(q_data: dict,
                        top_ids: torch.Tensor,   # [N, 32]
                        top_sims: torch.Tensor,  # [N, 32]
                        states: dict) -> torch.Tensor:
    """[N, 22]"""
    N  = len(q_data["y"])
    K  = top_ids.shape[1]
    k2 = min(2, K); k4 = min(4, K); k8 = min(8, K)

    nll_gpt = q_data["nll_gpt"].float()
    p_gpt   = q_data["p_gpt_true"].float()

    st_ent = states["state_entropy"].float()         # [B]
    st_pur = states["state_purity"].float()          # [B]
    st_cnt = states["total_counts"].float()          # [B]
    tk_cnt = states["top_k_token_counts"].float()    # [B, TOP_K]

    # Similarity stats
    s1   = top_sims[:, 0]
    s2   = top_sims[:, :k2].mean(-1)
    s4   = top_sims[:, :k4].mean(-1)
    s8   = top_sims[:, :k8].mean(-1)
    std8 = top_sims[:, :k8].std(-1).clamp(0)
    g12  = top_sims[:, 0] - top_sims[:, min(1, K-1)]
    g14  = top_sims[:, 0] - s4

    # State metadata
    ids1 = top_ids[:, 0]
    ids4 = top_ids[:, :k4]
    ids8 = top_ids[:, :k8]

    ent1 = st_ent[ids1]
    ent4 = st_ent[ids4].mean(-1)
    ent8 = st_ent[ids8].mean(-1)
    pur1 = st_pur[ids1]
    pur4 = st_pur[ids4].mean(-1)
    c1   = torch.log1p(st_cnt[ids1])
    c4   = torch.log1p(st_cnt[ids4]).mean(-1)
    c8   = torch.log1p(st_cnt[ids8]).mean(-1)

    # Top-token probability of nearest states
    tt1 = tk_cnt[ids1, 0] / (st_cnt[ids1] + EPS)
    tt4 = torch.stack([
        tk_cnt[top_ids[:, i], 0] / (st_cnt[top_ids[:, i]] + EPS)
        for i in range(k4)
    ], dim=-1).mean(-1)

    obs = torch.stack([
        nll_gpt.clamp(0, 15),           # 0
        p_gpt.clamp(EPS, 1).log(),      # 1
        torch.zeros(N),                  # 2  gpt_margin (unavail)
        s1, s2, s4, s8, std8,           # 3-7
        ent1, ent4, ent8,               # 8-10
        pur1, pur4,                     # 11-12
        c1, c4, c8,                     # 13-15
        torch.zeros(N),                  # 16 agreement (unavail)
        torch.zeros(N),                  # 17 agreement (unavail)
        tt1, tt4,                       # 18-19
        g12, g14,                       # 20-21
    ], dim=-1)

    return obs  # [N, 22]


# ── reward matrix ─────────────────────────────────────────────────────────────

def compute_reward_matrix(precomp: dict, actions: list[dict]) -> torch.Tensor:
    """[N, A]  reward = NLL_GPT - NLL_final  (positive = retrieval helped)"""
    N  = len(precomp["p_gpt"])
    A  = len(actions)
    rew = torch.zeros(N, A)

    rc  = precomp["raw_count"]    # [N, 32]
    st  = precomp["sel_total"]    # [N, 32]
    sim = precomp["top_sims"]     # [N, 32]
    p_g = precomp["p_gpt"]        # [N]
    pgy = precomp["p_global_y"]   # [N]
    nlg = precomp["nll_gpt"]      # [N]

    for ai, act in enumerate(actions):
        k = act["k_states"]
        if k == 0:
            continue  # reward = 0
        tau = act["tau"]; alpha = act["alpha"]; beta = act["beta"]
        w   = F.softmax(sim[:, :k] / (tau + EPS), dim=-1)
        ps  = (rc[:, :k] + beta * pgy.unsqueeze(1)) / (st[:, :k] + beta + EPS)
        pl  = (w * ps).sum(-1).clamp(EPS)
        pf  = (alpha * p_g + (1 - alpha) * pl).clamp(EPS)
        rew[:, ai] = nlg - (-pf.log())

    return rew


# ── best fixed ────────────────────────────────────────────────────────────────

def find_best_fixed(ct_rewards: torch.Tensor,
                     actions: list[dict]) -> tuple[int, dict]:
    best_ai = int(ct_rewards.mean(0).argmax())
    return best_ai, actions[best_ai]


# ── oracle ────────────────────────────────────────────────────────────────────

def compute_oracle(nll_gpt: torch.Tensor,
                    val_rewards: torch.Tensor) -> tuple[float, torch.Tensor]:
    """Per-example best action. Returns (mean_nll, per_example_nll)."""
    best_rew, best_ai = val_rewards.max(dim=-1)         # [N]
    nlls = (nll_gpt - best_rew).clamp(0)               # [N]
    return float(nlls.mean()), nlls


# ── Q-MLP ─────────────────────────────────────────────────────────────────────

class QStateReadMLP(nn.Module):
    def __init__(self, in_dim: int = Q_DIM, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── training ──────────────────────────────────────────────────────────────────

def train_q_mlp(obs: torch.Tensor,         # [N_ct, OBS_DIM]
                 act_feats: torch.Tensor,   # [A, ACT_DIM]
                 rewards: torch.Tensor,     # [N_ct, A]
                 device: torch.device,
                 max_q_samples: int = 800_000,
                 n_epochs: int = 30,
                 batch_size: int = 1024,
                 lr: float = 1e-3,
                 dev_frac: float = 0.2,
                 seed: int = 42) -> tuple["QStateReadMLP", dict]:
    set_seed(seed)
    N_ct, A = rewards.shape

    # Subsample (q, a) pairs
    total = N_ct * A
    if total > max_q_samples:
        perm  = torch.randperm(total)[:max_q_samples]
        q_idx = perm // A
        a_idx = perm % A
    else:
        q_idx = torch.arange(N_ct).unsqueeze(1).expand(N_ct, A).reshape(-1)
        a_idx = torch.arange(A).unsqueeze(0).expand(N_ct, A).reshape(-1)

    obs_s = obs[q_idx]              # [S, OBS_DIM]
    act_s = act_feats[a_idx]        # [S, ACT_DIM]
    rew_s = rewards[q_idx, a_idx]   # [S]
    X     = torch.cat([obs_s, act_s], dim=-1)  # [S, Q_DIM]

    # Normalise
    x_mu  = X.mean(0, keepdim=True)
    x_std = X.std(0, keepdim=True).clamp(1e-6)
    y_mu  = rew_s.mean();  y_std = rew_s.std().clamp(1e-6)
    Xn    = (X - x_mu) / x_std
    yn    = (rew_s - y_mu) / y_std

    S     = len(Xn)
    n_dev = max(1, int(S * dev_frac))
    perm2 = torch.randperm(S)
    dev_i, tr_i = perm2[:n_dev], perm2[n_dev:]

    dl = DataLoader(TensorDataset(Xn[tr_i], yn[tr_i]),
                    batch_size=batch_size, shuffle=True, num_workers=0)

    model = QStateReadMLP(Q_DIM).to(device)
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit  = nn.MSELoss()

    best_dev, best_sd, patience, max_pat = float("inf"), None, 0, 10
    history = []

    for ep in range(n_epochs):
        model.train()
        tr_loss = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tr_loss += float(loss) * len(xb)
        tr_loss /= max(len(tr_i), 1)

        model.eval()
        with torch.no_grad():
            dp = model(Xn[dev_i].to(device)).cpu()
        dev_loss = float(crit(dp, yn[dev_i]))

        history.append({"ep": ep+1, "tr": tr_loss, "dev": dev_loss})
        if (ep + 1) % 5 == 0:
            print(f"      ep{ep+1:3d}  tr={tr_loss:.5f}  dev={dev_loss:.5f}")

        if dev_loss < best_dev:
            best_dev = dev_loss
            best_sd  = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= max_pat:
                print(f"      early stop ep{ep+1}  best_dev={best_dev:.5f}")
                break

    if best_sd:
        model.load_state_dict(best_sd)
    model.cpu()
    model.x_mu  = x_mu    # [1, Q_DIM]
    model.x_std = x_std   # [1, Q_DIM]
    model.y_mu  = y_mu
    model.y_std = y_std

    return model, {"best_dev_mse": best_dev, "n_samples": S,
                   "n_epochs_run": len(history), "history": history[-10:]}


# ── inference ─────────────────────────────────────────────────────────────────

def apply_q_controller(model: "QStateReadMLP",
                        obs: torch.Tensor,        # [N, OBS_DIM]
                        act_feats: torch.Tensor,  # [A, ACT_DIM]
                        device: torch.device,
                        batch_q: int = 2048) -> torch.Tensor:
    """[N] chosen action indices."""
    model.eval()
    N, A = len(obs), len(act_feats)
    chosen = torch.zeros(N, dtype=torch.long)

    obs_n = (obs - model.x_mu[:, :OBS_DIM]) / model.x_std[:, :OBS_DIM]
    act_n = (act_feats - model.x_mu[:, OBS_DIM:]) / model.x_std[:, OBS_DIM:]

    for s in range(0, N, batch_q):
        e   = min(s + batch_q, N)
        C   = e - s
        ob  = obs_n[s:e].unsqueeze(1).expand(C, A, OBS_DIM).reshape(C * A, OBS_DIM)
        ac  = act_n.unsqueeze(0).expand(C, A, ACT_DIM).reshape(C * A, ACT_DIM)
        inp = torch.cat([ob, ac], dim=-1)
        with torch.no_grad():
            q = model.to(device)(inp.to(device)).cpu().view(C, A)
        chosen[s:e] = q.argmax(-1)

    return chosen


# ── val evaluation ────────────────────────────────────────────────────────────

def eval_fixed_on_val(best_ai: int, val_rewards: torch.Tensor,
                       val_precomp: dict) -> dict:
    nll_gpt = val_precomp["nll_gpt"]
    nlls    = (nll_gpt - val_rewards[:, best_ai]).clamp(0)
    return {"mean_nll": float(nlls.mean()), "per_example_nll": nlls}


def eval_q_on_val(model: "QStateReadMLP",
                   obs_val: torch.Tensor,
                   act_feats: torch.Tensor,
                   val_rewards: torch.Tensor,
                   val_precomp: dict,
                   device: torch.device,
                   actions: list[dict]) -> dict:
    chosen   = apply_q_controller(model, obs_val, act_feats, device)   # [N]
    nll_gpt  = val_precomp["nll_gpt"]
    best_rew = val_rewards.gather(1, chosen.unsqueeze(1)).squeeze(1)   # [N]
    nlls     = (nll_gpt - best_rew).clamp(0)

    # Action distribution
    counts   = torch.bincount(chosen, minlength=len(actions)).float()
    frac     = counts / counts.sum()

    # Stats
    k_chosen = torch.tensor([actions[i]["k_states"] for i in chosen.tolist()], dtype=torch.float)
    ret_mask = k_chosen > 0

    return {
        "mean_nll":         float(nlls.mean()),
        "per_example_nll":  nlls,
        "chosen":           chosen,
        "action_frac":      frac,
        "avg_k_states":     float(k_chosen.mean()),
        "retrieval_usage":  float(ret_mask.float().mean()),
    }


# ── action distribution table ─────────────────────────────────────────────────

def action_distribution_df(chosen: torch.Tensor, actions: list[dict]) -> pd.DataFrame:
    A      = len(actions)
    counts = torch.bincount(chosen, minlength=A).numpy()
    N      = counts.sum()
    rows   = []
    for ai, (act, cnt) in enumerate(zip(actions, counts)):
        if cnt > 0:
            rows.append({**act, "count": int(cnt), "frac": cnt / N})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("count", ascending=False)
    return df


# ── inspection files ───────────────────────────────────────────────────────────

def write_inspection_files(
    insp_dir: Path,
    val_data: dict,
    val_precomp: dict,
    top_ids_val: torch.Tensor,   # [N, 32]
    top_sims_val: torch.Tensor,  # [N, 32]
    states: dict,
    q_nlls: torch.Tensor,        # [N]
    fixed_nlls: torch.Tensor,    # [N]
    chosen: torch.Tensor,        # [N]
    best_ai: int,
    actions: list[dict],
    n_examples: int = 50,
):
    insp_dir.mkdir(parents=True, exist_ok=True)
    nll_gpt = val_precomp["nll_gpt"]
    y       = val_data["y"]
    tk_ids  = states["top_k_token_ids"]     # [B, TOP_K]
    st_ent  = states["state_entropy"]
    st_pur  = states["state_purity"]
    st_cnt  = states["total_counts"]

    delta_q_fixed = fixed_nlls - q_nlls     # positive = Q wins
    delta_q_gpt   = nll_gpt - q_nlls        # positive = Q beats GPT

    def format_example(i: int) -> str:
        qi = int(i)
        s1 = float(top_sims_val[qi, 0])
        st_i = int(top_ids_val[qi, 0])
        top_toks = tk_ids[st_i, :5].tolist()
        lines = [
            f"idx={qi}",
            f"y_true_token_id={int(y[qi])}",
            f"gpt_nll={float(nll_gpt[qi]):.4f}",
            f"q_nll={float(q_nlls[qi]):.4f}",
            f"fixed_nll={float(fixed_nlls[qi]):.4f}",
            f"delta_q_vs_fixed={float(delta_q_fixed[qi]):+.4f}",
            f"nearest_state_sim={s1:.4f}",
            f"nearest_state_entropy={float(st_ent[st_i]):.3f}",
            f"nearest_state_purity={float(st_pur[st_i]):.3f}",
            f"nearest_state_count={int(st_cnt[st_i])}",
            f"nearest_state_top5_tokens={top_toks}",
            f"chosen_action={actions[int(chosen[qi])]['name']}",
            f"best_fixed_action={actions[best_ai]['name']}",
            "---",
        ]
        return "\n".join(lines)

    files = {
        "q_wins_vs_fixed.txt":   (delta_q_fixed > 0.01).nonzero(as_tuple=True)[0],
        "q_fails_vs_fixed.txt":  (delta_q_fixed < -0.01).nonzero(as_tuple=True)[0],
        "biggest_help_vs_gpt.txt": delta_q_gpt.topk(n_examples).indices,
        "biggest_hurt_vs_gpt.txt": (-delta_q_gpt).topk(n_examples).indices,
        "gpt_only_chosen.txt":   (chosen == 0).nonzero(as_tuple=True)[0],
        "state_read_chosen.txt": (chosen != 0).nonzero(as_tuple=True)[0],
    }

    for fname, idxs in files.items():
        idxs = idxs[:n_examples]
        with open(insp_dir / fname, "w", encoding="utf-8") as f:
            f.write(f"# {fname}  n={len(idxs)}\n\n")
            for i in idxs:
                f.write(format_example(int(i)) + "\n")


# ── load raw random baseline ──────────────────────────────────────────────────

def load_random_baseline(csv_path: str | None, budget: int) -> float:
    if csv_path is None or not Path(csv_path).exists():
        return float("nan")
    try:
        df = pd.read_csv(csv_path)
        r  = df[df["method"].str.startswith("random") &
                (df["budget"] == budget) &
                (df["read_policy"] == "best_fixed")]
        if not r.empty:
            return float(r["val_nll"].mean())
    except Exception:
        pass
    return float("nan")


def load_full_ds_baselines(src: Path) -> dict:
    out = {"gpt": 2.5533, "fixed": 2.3873, "qmlp": 2.2977}
    bl  = src / "reports" / "fixed_baselines.json"
    if bl.exists():
        with open(bl) as f:
            d = json.load(f)
        out["gpt"]   = d.get("gpt_only_val_nll",           out["gpt"])
        out["fixed"] = d.get("best_train_selected_val_nll", out["fixed"])
    qm = src / "reports" / "q_controller_metrics.json"
    if qm.exists():
        with open(qm) as f:
            d = json.load(f)
        if "Q-MLP-full" in d:
            out["qmlp"] = d["Q-MLP-full"].get("mean_nll", out["qmlp"])
    return out


# ── per-method/budget runner ──────────────────────────────────────────────────

def run_method_budget(
    method: str,
    budget: int,
    src: Path,
    states_dir: Path,
    out: Path,
    device: torch.device,
    raw_memory_csv: str | None = None,
    max_q_samples: int = 800_000,
    n_epochs: int = 30,
    fast_grid: bool = False,
    force_neighbors: bool = False,
    force_rewards: bool = False,
    force_train: bool = False,
    force_eval: bool = False,
) -> dict | None:
    tag = f"{method}_B{budget}"
    state_path = states_dir / f"{tag}.pt"
    if not state_path.exists():
        print(f"  [SKIP] state file not found: {state_path}")
        return None

    mb_out   = out / tag
    mb_out.mkdir(parents=True, exist_ok=True)
    nbr_dir  = out / "state_neighbors"
    nbr_dir.mkdir(parents=True, exist_ok=True)
    insp_dir = out / "inspection" / tag

    # ── load data ────────────────────────────────────────────────────────────
    print(f"\n  Loading data ...")
    ct_data  = torch.load(src / "states" / "controller_train.pt", weights_only=False)
    val_data = torch.load(src / "states" / "val.pt",              weights_only=False)
    ds_data  = torch.load(src / "states" / "datastore.pt",        weights_only=False)
    states   = torch.load(state_path, weights_only=False)

    full_bl  = load_full_ds_baselines(src)
    rand_nll = load_random_baseline(raw_memory_csv, budget)

    P_global  = build_global_freq(ds_data["y"], VOCAB)

    def norm_h(data):
        h = data["h"].float()
        return h / (h.norm(dim=-1, keepdim=True) + 1e-8)

    ct_h  = norm_h(ct_data)
    val_h = norm_h(val_data)
    proto = states["prototype_h"]   # [B, D]  already normalised

    # ── top-k state neighbors (cached) ───────────────────────────────────────
    ct_nbr_path  = nbr_dir / f"{tag}_ct_top{MAX_K}.pt"
    val_nbr_path = nbr_dir / f"{tag}_val_top{MAX_K}.pt"

    if ct_nbr_path.exists() and not force_neighbors:
        print(f"  Loading CT neighbors ...")
        ct_nbrs = torch.load(ct_nbr_path, weights_only=False)
        ct_ids, ct_sims = ct_nbrs["ids"], ct_nbrs["sims"]
    else:
        print(f"  Computing CT top-{MAX_K} state sims ...")
        ct_ids, ct_sims = compute_top_k_state_sims(ct_h, proto, MAX_K, device)
        torch.save({"ids": ct_ids, "sims": ct_sims}, ct_nbr_path)

    if val_nbr_path.exists() and not force_neighbors:
        print(f"  Loading val neighbors ...")
        val_nbrs = torch.load(val_nbr_path, weights_only=False)
        val_ids, val_sims = val_nbrs["ids"], val_nbrs["sims"]
    else:
        print(f"  Computing val top-{MAX_K} state sims ...")
        val_ids, val_sims = compute_top_k_state_sims(val_h, proto, MAX_K, device)
        torch.save({"ids": val_ids, "sims": val_sims}, val_nbr_path)

    # ── precompute token lookups ──────────────────────────────────────────────
    ct_precomp  = precompute_lookup(ct_data,  ct_ids,  ct_sims,  states, P_global)
    val_precomp = precompute_lookup(val_data, val_ids, val_sims, states, P_global)

    # ── observation features ──────────────────────────────────────────────────
    obs_ct  = build_obs_features(ct_data,  ct_ids,  ct_sims,  states)  # [N_ct,  22]
    obs_val = build_obs_features(val_data, val_ids, val_sims, states)  # [N_val, 22]

    actions   = build_action_grid(fast=fast_grid)
    act_feats = build_action_features(actions)                          # [A, 6]
    A         = len(actions)
    print(f"  Action grid: {A} actions  (fast={fast_grid})")

    # ── reward matrices (cached) ──────────────────────────────────────────────
    ct_rew_path  = mb_out / "ct_rewards.pt"
    val_rew_path = mb_out / "val_rewards.pt"

    if ct_rew_path.exists() and not force_rewards:
        print(f"  Loading CT rewards ...")
        ct_rewards  = torch.load(ct_rew_path,  weights_only=False)
        val_rewards = torch.load(val_rew_path, weights_only=False)
    else:
        print(f"  Computing CT rewards  [N_ct={len(ct_data['y']):,}  A={A}] ...")
        ct_rewards  = compute_reward_matrix(ct_precomp,  actions)
        print(f"  Computing val rewards [N_val={len(val_data['y']):,} A={A}] ...")
        val_rewards = compute_reward_matrix(val_precomp, actions)
        torch.save(ct_rewards,  ct_rew_path)
        torch.save(val_rewards, val_rew_path)

    # ── best fixed action ─────────────────────────────────────────────────────
    best_ai, best_act = find_best_fixed(ct_rewards, actions)
    fixed_m = eval_fixed_on_val(best_ai, val_rewards, val_precomp)
    fixed_nll = fixed_m["mean_nll"]
    print(f"  Best-fixed action: {best_act['name']}  val_NLL={fixed_nll:.4f}")

    # ── oracle ────────────────────────────────────────────────────────────────
    oracle_nll, oracle_nlls = compute_oracle(val_precomp["nll_gpt"], val_rewards)
    print(f"  Oracle NLL: {oracle_nll:.4f}")

    # ── train Q-MLP ──────────────────────────────────────────────────────────
    q_path = mb_out / "q_model.pt"
    if q_path.exists() and not force_train:
        print(f"  Loading Q model ...")
        ckpt  = torch.load(q_path, weights_only=False)
        model = QStateReadMLP(Q_DIM)
        model.load_state_dict(ckpt["state_dict"])
        model.x_mu  = ckpt["x_mu"]
        model.x_std = ckpt["x_std"]
        model.y_mu  = ckpt["y_mu"]
        model.y_std = ckpt["y_std"]
        q_train_metrics = ckpt.get("train_metrics", {})
    else:
        print(f"  Training Q-MLP  max_q_samples={max_q_samples:,} ...")
        model, q_train_metrics = train_q_mlp(
            obs_ct, act_feats, ct_rewards, device,
            max_q_samples=max_q_samples, n_epochs=n_epochs,
        )
        torch.save({
            "state_dict":    model.state_dict(),
            "x_mu":          model.x_mu,
            "x_std":         model.x_std,
            "y_mu":          model.y_mu,
            "y_std":         model.y_std,
            "train_metrics": q_train_metrics,
            "actions":       actions,
            "method":        method,
            "budget":        budget,
        }, q_path)

    # ── Q eval on val ─────────────────────────────────────────────────────────
    print(f"  Evaluating Q-controller on val ...")
    q_m = eval_q_on_val(model, obs_val, act_feats, val_rewards,
                         val_precomp, device, actions)
    q_nll = q_m["mean_nll"]
    print(f"  Q-state-read NLL={q_nll:.4f}  "
          f"d_vs_fixed={q_nll - fixed_nll:+.4f}  "
          f"d_vs_random={q_nll - rand_nll:+.4f}")

    # ── action distribution ───────────────────────────────────────────────────
    act_df = action_distribution_df(q_m["chosen"], actions)
    act_df.to_csv(mb_out / "action_distribution.csv", index=False)

    top_actions = act_df.head(2)[["name", "frac"]].to_dict("records") if not act_df.empty else []

    # ── save metrics ──────────────────────────────────────────────────────────
    metrics = {
        "method":             method,
        "budget":             budget,
        "n_states":           int(len(states["prototype_h"])),
        "action_grid_size":   A,
        "best_fixed_action":  best_act["name"],
        "best_fixed_nll":     fixed_nll,
        "q_state_nll":        q_nll,
        "oracle_nll":         oracle_nll,
        "gpt_nll":            float(val_precomp["nll_gpt"].mean()),
        "full_ds_fixed_nll":  full_bl["fixed"],
        "full_ds_qmlp_nll":   full_bl["qmlp"],
        "raw_random_nll":     rand_nll,
        "delta_q_vs_fixed":   q_nll - fixed_nll,
        "delta_q_vs_random":  q_nll - rand_nll,
        "delta_q_vs_ds_fixed":q_nll - full_bl["fixed"],
        "delta_q_vs_ds_qmlp": q_nll - full_bl["qmlp"],
        "avg_k_states":       q_m["avg_k_states"],
        "retrieval_usage":    q_m["retrieval_usage"],
        "oracle_gap_vs_fixed": oracle_nll - fixed_nll,
        "oracle_gap_vs_q":    oracle_nll - q_nll,
        "q_train_metrics":    q_train_metrics,
        "top_actions":        top_actions,
    }
    with open(mb_out / "q_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    fixed_m_out = {
        "action": best_act["name"], "val_nll": fixed_nll,
        "val_ppl": ppl(fixed_nll),
    }
    oracle_m_out = {
        "val_nll": oracle_nll, "val_ppl": ppl(oracle_nll),
        "oracle_gap_vs_fixed": oracle_nll - fixed_nll,
        "oracle_gap_vs_q": oracle_nll - q_nll,
    }
    with open(mb_out / "best_fixed_metrics.json", "w") as f:
        json.dump(fixed_m_out, f, indent=2)
    with open(mb_out / "oracle_metrics.json", "w") as f:
        json.dump(oracle_m_out, f, indent=2)

    torch.save({"q_nlls": q_m["per_example_nll"],
                "fixed_nlls": fixed_m["per_example_nll"],
                "oracle_nlls": oracle_nlls,
                "chosen": q_m["chosen"]}, mb_out / "val_predictions.pt")

    # ── inspection ────────────────────────────────────────────────────────────
    write_inspection_files(
        insp_dir, val_data, val_precomp,
        val_ids, val_sims, states,
        q_m["per_example_nll"], fixed_m["per_example_nll"],
        q_m["chosen"], best_ai, actions,
    )

    return metrics


# ── report ────────────────────────────────────────────────────────────────────

def generate_report(all_metrics: list[dict]) -> str:
    lines = []
    p = lambda t="": lines.append(t)
    h = lambda n, t: (lines.append("#"*n + " " + t), lines.append(""))

    if not all_metrics:
        return "No results.\n"

    m0  = all_metrics[0]
    gpt = m0.get("gpt_nll", 2.5533)
    dsf = m0.get("full_ds_fixed_nll", 2.3873)
    dsq = m0.get("full_ds_qmlp_nll",  2.2977)

    h(1, "MVP 2c: Q-State-Read Controller — Report")
    p(f"Full-datastore: GPT={gpt:.4f}  Fixed={dsf:.4f}  Q-MLP={dsq:.4f}")
    p(f"MVP 2b best:   minibatch_kmeans B=50k = 2.4205  (best-fixed state-read)")
    p()

    h(2, "1. Goal")
    p("MVP 2b showed persistent predictive states beat raw memory under best-fixed read.")
    p("MVP 2c tests whether dynamic Q-read control improves state usage.")
    p("Question: does Q-MLP select better (k_states, tau, alpha, beta) per query?")
    p()

    h(2, "2. Setup")
    n_act = m0.get("action_grid_size", "?")
    p(f"- Action grid: {n_act} actions (k_states x tau x alpha x beta + gpt_only)")
    p("- Obs features: 22-dim (GPT uncertainty, state sims, state quality, count info)")
    p("- Q-MLP: 28-dim input -> 128 -> 128 -> 1  |  MSE loss  |  Adam")
    p("- Training: max 800k (q,a) pairs subsampled from CT × actions")
    p("- Oracle: per-example best action on val (diagnostic only)")
    p()

    h(2, "3. Main Results")
    p("| method | budget | best-fixed | Q-state | delta | oracle | random_B | ds_fixed | ds_qmlp |")
    p("|--------|--------|-----------|---------|-------|--------|----------|----------|---------|")
    for m in sorted(all_metrics, key=lambda x: (x["method"], x["budget"])):
        d  = m["delta_q_vs_fixed"]
        sg = "+" if d >= 0 else ""
        p(f"| {m['method']:<28} | {m['budget']:>6} | "
          f"{m['best_fixed_nll']:.4f} | {m['q_state_nll']:.4f} | {sg}{d:.4f} | "
          f"{m['oracle_nll']:.4f} | {m.get('raw_random_nll', float('nan')):.4f} | "
          f"{dsf:.4f} | {dsq:.4f} |")
    p()

    h(2, "4. Best Result")
    best = min(all_metrics, key=lambda x: x["q_state_nll"])
    p(f"Best Q-state-read: {best['method']}  B={best['budget']}  "
      f"NLL={best['q_state_nll']:.4f}")
    p(f"  vs MVP 2b best-fixed (minibatch_kmeans B=50k=2.4205): "
      f"{best['q_state_nll'] - 2.4205:+.4f}")
    p(f"  vs full-ds fixed ({dsf:.4f}): {best['q_state_nll'] - dsf:+.4f}")
    p(f"  vs full-ds Q-MLP ({dsq:.4f}): {best['q_state_nll'] - dsq:+.4f}")
    p()

    h(2, "5. Does Q-Read Help?")
    wins = sum(1 for m in all_metrics if m["delta_q_vs_fixed"] < 0)
    tot  = len(all_metrics)
    frac = wins / tot if tot else 0
    p(f"Q-state-read beats best-fixed on {wins}/{tot} ({frac:.0%}) method/budget combos.")
    p()

    if frac >= 0.7:
        verdict = "SUCCESS — Q-state-read consistently improves over fixed read policy"
    elif frac >= 0.5:
        verdict = "PARTIAL SUCCESS — Q-state-read improves in most cases"
    elif frac >= 0.3:
        verdict = "WEAK — Q-state-read only occasionally helps"
    else:
        verdict = "FAILURE — Q-state-read does not beat best-fixed state-read"

    p(f"**[{verdict}]**")
    p()

    h(2, "6. Action Behavior")
    for m in sorted(all_metrics, key=lambda x: (x["method"], x["budget"])):
        tops = m.get("top_actions", [])
        top1 = tops[0] if tops else {}
        top2 = tops[1] if len(tops) > 1 else {}
        p(f"  {m['method']} B={m['budget']}: "
          f"ret_usage={m['retrieval_usage']:.2f}  "
          f"avg_k={m['avg_k_states']:.1f}  "
          f"top1={top1.get('name','?')} ({top1.get('frac',0):.2f})  "
          f"top2={top2.get('name','?')} ({top2.get('frac',0):.2f})")
    p()

    h(2, "7. Oracle Gap")
    p("How much headroom exists in the action grid?")
    p()
    p("| method | budget | best-fixed | Q-state | oracle | gap_fixed | gap_Q |")
    p("|--------|--------|-----------|---------|--------|-----------|-------|")
    for m in sorted(all_metrics, key=lambda x: (x["method"], x["budget"])):
        gf = m["oracle_nll"] - m["best_fixed_nll"]
        gq = m["oracle_nll"] - m["q_state_nll"]
        p(f"| {m['method']:<28} | {m['budget']:>6} | "
          f"{m['best_fixed_nll']:.4f} | {m['q_state_nll']:.4f} | "
          f"{m['oracle_nll']:.4f} | {gf:+.4f} | {gq:+.4f} |")
    p()
    p("oracle gap vs Q: how much Q leaves on the table vs per-example best action.")
    p()

    h(2, "8. What This Does Not Test")
    for item in ["PRUNE", "New WRITE methods", "Semantic labels / manual regions",
                 "Hierarchy", "PPO", "Transformer fine-tuning", "Branch B"]:
        p(f"- {item}")
    p()
    p("MVP 2c only tests: **can Q-MLP improve state-reading over a fixed policy?**")
    p()

    return "\n".join(lines)


# ── plots ─────────────────────────────────────────────────────────────────────

def try_plots(all_metrics: list[dict], out: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_dir = out / "plots"
        plot_dir.mkdir(exist_ok=True)
        methods  = sorted(set(m["method"] for m in all_metrics))
        budgets  = sorted(set(m["budget"] for m in all_metrics))
        colors   = plt.cm.tab10(np.linspace(0, 1, max(len(methods), 1)))

        def get(m, method, budget, key):
            r = [x for x in m if x["method"] == method and x["budget"] == budget]
            return r[0].get(key, float("nan")) if r else float("nan")

        dsf = all_metrics[0]["full_ds_fixed_nll"] if all_metrics else 2.3873
        dsq = all_metrics[0]["full_ds_qmlp_nll"]  if all_metrics else 2.2977

        for metric_key, ylabel, fname in [
            ("q_state_nll",    "Q-state-read NLL",                  "q_vs_fixed_by_budget.png"),
            ("delta_q_vs_fixed","Delta NLL (Q vs fixed, neg=better)", "delta_q_vs_fixed.png"),
            ("retrieval_usage", "Retrieval usage",                   "retrieval_usage_by_budget.png"),
            ("avg_k_states",    "Avg k_states chosen",               "avg_k_states_by_budget.png"),
            ("oracle_gap_vs_q", "Oracle gap vs Q (neg=Q better)",    "oracle_gap_by_budget.png"),
        ]:
            fig, ax = plt.subplots(figsize=(9, 5))
            for i, meth in enumerate(methods):
                xs = [b for b in budgets]
                ys = [get(all_metrics, meth, b, metric_key) for b in budgets]
                ax.plot(xs, ys, "-o", label=meth, color=colors[i], linewidth=1.5, markersize=5)
            if metric_key == "q_state_nll":
                ax.axhline(dsf, color="black",  linestyle=":", alpha=0.6, label="full_ds_fixed")
                ax.axhline(dsq, color="purple", linestyle=":", alpha=0.6, label="full_ds_Q-MLP")
            ax.set_xlabel("Budget"); ax.set_ylabel(ylabel)
            ax.set_title(ylabel); ax.set_xscale("log")
            ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
            plt.tight_layout()
            plt.savefig(plot_dir / fname, dpi=150)
            plt.close()

        # Q vs fixed vs full-ds
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, meth in enumerate(methods):
            xs = budgets
            q  = [get(all_metrics, meth, b, "q_state_nll")  for b in budgets]
            bf = [get(all_metrics, meth, b, "best_fixed_nll") for b in budgets]
            ax.plot(xs, q,  "-o", color=colors[i], linewidth=1.5, markersize=5, label=f"{meth} Q")
            ax.plot(xs, bf, "--", color=colors[i], linewidth=1.0, markersize=3, alpha=0.6, label=f"{meth} fixed")
        ax.axhline(dsf, color="black",  linestyle=":", label="full_ds_fixed")
        ax.axhline(dsq, color="purple", linestyle=":", label="full_ds_Q-MLP")
        ax.set_xlabel("Budget"); ax.set_ylabel("Val NLL")
        ax.set_title("Q-state vs best-fixed vs full-DS baselines")
        ax.set_xscale("log")
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=6)
        plt.tight_layout()
        plt.savefig(plot_dir / "best_q_state_vs_full_baselines.png", dpi=150)
        plt.close()

        print(f"  Plots -> {plot_dir}")
    except Exception as e:
        print(f"  Plots skipped: {e}")

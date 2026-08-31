"""
train_state_read_controller.py

Q-MLP controller that selects state-read actions per query.

Observation (12 features per query):
  gpt_entropy, gpt_top1_prob, gpt_margin,
  nearest_state_sim, mean_top2_state_sim, mean_top4_state_sim, mean_top8_state_sim,
  top1_state_entropy, mean_top4_state_entropy,
  log1p(top1_state_count), mean_log_top4_state_count,
  gpt_top1_in_nearest_state   (binary)

Action features (5):
  k_states_norm, tau, alpha, beta_norm, is_gpt_only

Q input: obs [12] || action_feats [5] = 17-dim
Q output: scalar (predicted reward)

Training: offline fitted-Q on CT data with reward = -NLL
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils import get_device, set_seed

MAX_K_STATES = 32
OBS_DIM      = 12
ACT_DIM      = 5
Q_DIM        = OBS_DIM + ACT_DIM   # 17


# ── action grid ───────────────────────────────────────────────────────────────

K_VALS     = [1, 2, 4, 8, 16, 32]
TAU_VALS   = [0.05, 0.1, 0.2]
ALPHA_VALS = [0.25, 0.5, 0.75]
BETA_VALS  = [0.0, 1.0, 5.0, 10.0]

def build_state_action_grid(max_k: int = MAX_K_STATES) -> list[dict]:
    acts = [{"name": "gpt_only", "k_states": 0, "tau": 0.1,
              "alpha": 1.0, "beta": 0.0}]
    for k in K_VALS:
        if k > max_k: continue
        for tau in TAU_VALS:
            for alpha in ALPHA_VALS:
                for beta in BETA_VALS:
                    name = f"k{k}_t{tau}_a{alpha}_b{beta}"
                    acts.append({"name": name, "k_states": k, "tau": tau,
                                  "alpha": alpha, "beta": beta})
    return acts


# ── action feature vectors ─────────────────────────────────────────────────────

def build_state_action_features_all(actions: list[dict],
                                     max_k: int = MAX_K_STATES) -> torch.Tensor:
    """Returns [A, 5] action feature matrix."""
    rows = []
    for a in actions:
        k    = a["k_states"]
        tau  = a["tau"]
        alpha= a["alpha"]
        beta = a["beta"]
        rows.append([
            k / max_k,                   # k_states_norm
            tau,                         # tau (already in [0,1] range)
            alpha,                       # alpha
            beta / 10.0,                 # beta_norm
            float(k == 0),               # is_gpt_only
        ])
    return torch.tensor(rows, dtype=torch.float32)  # [A, 5]


# ── observation features ───────────────────────────────────────────────────────

def build_state_obs_features(q_data: dict, precomp: dict, states: dict) -> torch.Tensor:
    """
    Builds [N, 12] observation feature tensor for state-read controller.

    precomp keys needed: top_sims [N, MAX_K], raw_count [N, MAX_K],
                         sel_total [N, MAX_K], p_gpt [N], nll_gpt [N]
    states keys needed: state_entropy [B]
    """
    N = len(q_data["y"])
    top_sims  = precomp["top_sims"].float()   # [N, MAX_K]
    raw_count = precomp["raw_count"].float()
    sel_total = precomp["sel_total"].float()
    p_gpt     = precomp["p_gpt"].float()      # [N]  p(y_true) from GPT
    nll_gpt   = precomp["nll_gpt"].float()    # [N]

    B_states = len(states["state_entropy"])
    state_entropy_all = states["state_entropy"].float()  # [B]

    # GPT-based features
    gpt_entropy = nll_gpt.clamp(0, 12)          # proxy: higher loss → higher entropy
    gpt_top1    = p_gpt.clamp(1e-10, 1.0)
    gpt_margin  = torch.zeros(N)                 # placeholder (we don't have sorted probs)

    # State similarity features
    sim_top1 = top_sims[:, 0]
    sim_top2 = top_sims[:, :2].mean(-1)
    sim_top4 = top_sims[:, :4].mean(-1)
    sim_top8 = top_sims[:, :8].mean(-1)

    # State entropy features  (needs index lookup)
    # We stored top-k state ids in precomp via top_ids — but precomp only has raw_count/sel_total
    # Approx state entropy from sel_total as proxy for state size
    top1_entropy = (raw_count[:, 0] / (sel_total[:, 0] + 1e-8)).clamp(0, 1)  # purity proxy
    top4_entropy = (raw_count[:, :4] / (sel_total[:, :4] + 1e-8)).mean(-1)

    # Count features
    log_top1_count = torch.log1p(sel_total[:, 0])
    log_top4_count = torch.log1p(sel_total[:, :4]).mean(-1)

    # True-token in nearest state
    true_in_nearest = (raw_count[:, 0] > 0).float()

    obs = torch.stack([
        gpt_entropy,       # 0
        gpt_top1.log(),    # 1  log prob (negative)
        gpt_margin,        # 2
        sim_top1,          # 3
        sim_top2,          # 4
        sim_top4,          # 5
        sim_top8,          # 6
        top1_entropy,      # 7  (purity proxy)
        top4_entropy,      # 8
        log_top1_count,    # 9
        log_top4_count,    # 10
        true_in_nearest,   # 11
    ], dim=-1)             # [N, 12]

    return obs


# ── Q-MLP model ───────────────────────────────────────────────────────────────

class QStateMLP(nn.Module):
    def __init__(self, in_dim: int = Q_DIM, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── training ──────────────────────────────────────────────────────────────────

def train_q_state_mlp(obs: torch.Tensor,       # [N, OBS_DIM]
                       act_feats: torch.Tensor, # [A, ACT_DIM]
                       rewards: torch.Tensor,   # [N, A] (reward per action per query)
                       device: torch.device,
                       n_epochs: int = 30,
                       batch_size: int = 4096,
                       lr: float = 1e-3,
                       seed: int = 42) -> QStateMLP:
    set_seed(seed)
    N, A = rewards.shape

    # Normalise observations
    obs_mu  = obs.mean(0, keepdim=True)
    obs_std = obs.std(0, keepdim=True).clamp(min=1e-6)
    obs_n   = (obs - obs_mu) / obs_std

    # Expand: each (query, action) is a training sample
    obs_exp = obs_n.unsqueeze(1).expand(N, A, OBS_DIM).reshape(N * A, OBS_DIM)
    act_exp = act_feats.unsqueeze(0).expand(N, A, ACT_DIM).reshape(N * A, ACT_DIM)
    rew_exp = rewards.reshape(N * A)

    inp = torch.cat([obs_exp, act_exp], dim=-1)  # [N*A, Q_DIM]

    ds      = TensorDataset(inp, rew_exp)
    loader  = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model   = QStateMLP(Q_DIM).to(device)
    opt     = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit    = nn.MSELoss()

    for ep in range(n_epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += float(loss) * len(xb)
        if (ep + 1) % 10 == 0:
            print(f"    epoch {ep+1}/{n_epochs}  loss={total_loss / (N*A):.5f}")

    model.cpu()
    # Store normalisation stats
    model.obs_mu  = obs_mu
    model.obs_std = obs_std
    return model


# ── inference ─────────────────────────────────────────────────────────────────

def apply_q_state_controller(model: QStateMLP,
                              obs: torch.Tensor,     # [N, OBS_DIM]
                              act_feats: torch.Tensor,  # [A, ACT_DIM]
                              device: torch.device) -> torch.Tensor:
    """Returns [N] indices into action list (argmax Q)."""
    model.eval()
    N = len(obs)
    A = len(act_feats)

    obs_n = (obs - model.obs_mu) / model.obs_std

    obs_exp = obs_n.unsqueeze(1).expand(N, A, OBS_DIM).reshape(N * A, OBS_DIM)
    act_exp = act_feats.unsqueeze(0).expand(N, A, ACT_DIM).reshape(N * A, ACT_DIM)
    inp     = torch.cat([obs_exp, act_exp], dim=-1)

    with torch.no_grad():
        q = model.to(device)(inp.to(device)).cpu()  # [N*A]

    q = q.view(N, A)
    return q.argmax(dim=-1)   # [N]


# ── offline reward computation ────────────────────────────────────────────────

def compute_per_action_rewards(precomp: dict,
                                act_feats: torch.Tensor,
                                actions: list[dict],
                                chunk: int = 2048) -> torch.Tensor:
    """
    Returns [N, A] reward tensor.  reward = -NLL (higher is better).
    Imports eval_action from evaluate_predictive_states inline to avoid circular dep.
    """
    import torch.nn.functional as F

    N = len(precomp["p_gpt"])
    A = len(actions)
    rewards = torch.zeros(N, A)

    for ai, act in enumerate(actions):
        k    = act["k_states"]
        tau  = act["tau"]
        alpha= act["alpha"]
        beta = act["beta"]
        p_gpt = precomp["p_gpt"]
        p_gy  = precomp["p_global_y"]

        if k == 0:
            nll = -p_gpt.clamp(1e-10).log()
        else:
            sims_k  = precomp["top_sims"][:, :k]
            raw_k   = precomp["raw_count"][:, :k]
            total_k = precomp["sel_total"][:, :k]
            weights = F.softmax(sims_k / (tau + 1e-8), dim=-1)
            p_state = (raw_k + beta * p_gy.unsqueeze(1)) / (total_k + beta + 1e-10)
            p_local = (weights * p_state).sum(-1).clamp(1e-10)
            p_final = (alpha * p_gpt + (1 - alpha) * p_local).clamp(1e-10)
            nll     = -p_final.log()

        rewards[:, ai] = -nll  # higher reward = lower NLL

    return rewards


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",    required=True)
    parser.add_argument("--output",    required=True)
    parser.add_argument("--method",    required=True)
    parser.add_argument("--budget",    type=int, required=True)
    parser.add_argument("--n_epochs",  type=int, default=30)
    parser.add_argument("--batch",     type=int, default=4096)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--force",     action="store_true")
    args = parser.parse_args()

    src = Path(args.source)
    out = Path(args.output)
    (out / "q_state_models").mkdir(parents=True, exist_ok=True)

    save_path = out / "q_state_models" / f"q_state_{args.method}_B{args.budget}.pt"
    if save_path.exists() and not args.force:
        print(f"Model already exists: {save_path}  (--force to retrain)")
        return

    device = get_device({"device": "cuda"})
    set_seed(42)

    print(f"\ntrain_state_read_controller: {args.method}  B={args.budget}")

    # Load data
    ct_data  = torch.load(src / "states" / "controller_train.pt", weights_only=False)
    val_data = torch.load(src / "states" / "val.pt",              weights_only=False)
    ds_data  = torch.load(src / "states" / "datastore.pt",        weights_only=False)

    vocab_size = 50257
    counts = torch.bincount(ds_data["y"].long(), minlength=vocab_size).float()
    P_global = (counts / counts.sum()).clamp(1e-10)

    # Load states
    state_path = out / "states" / f"{args.method}_B{args.budget}.pt"
    if not state_path.exists():
        print(f"State file not found: {state_path}")
        return
    states = torch.load(state_path, weights_only=False)
    proto  = states["prototype_h"]

    def norm_h(data):
        h = data["h"].float()
        return h / (h.norm(dim=-1, keepdim=True) + 1e-8)

    from build_neighbors import cosine_top_k
    from evaluate_predictive_states import (
        compute_top_k_state_sims, precompute_lookup
    )

    ct_h_norm = norm_h(ct_data)
    print(f"  Computing CT top-{MAX_K_STATES} state sims ...")
    ct_idx, ct_sims = compute_top_k_state_sims(ct_h_norm, proto, MAX_K_STATES, device)
    ct_precomp = precompute_lookup(ct_data, ct_idx, ct_sims, states, P_global)

    actions   = build_state_action_grid(MAX_K_STATES)
    act_feats = build_state_action_features_all(actions, MAX_K_STATES)

    print(f"  Computing CT per-action rewards ({len(actions)} actions) ...")
    ct_precomp["p_global_y"] = P_global[ct_data["y"].long()]
    rewards = compute_per_action_rewards(ct_precomp, act_feats, actions)

    obs = build_state_obs_features(ct_data, ct_precomp, states)

    print(f"  Training Q-state-MLP  N={len(obs):,}  A={len(actions)}  "
          f"epochs={args.n_epochs} ...")
    model = train_q_state_mlp(obs, act_feats, rewards, device,
                               n_epochs=args.n_epochs,
                               batch_size=args.batch,
                               lr=args.lr)

    torch.save({
        "model":   model.state_dict(),
        "obs_mu":  model.obs_mu,
        "obs_std": model.obs_std,
        "actions": actions,
        "method":  args.method,
        "budget":  args.budget,
        "q_dim":   Q_DIM,
    }, save_path)
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    main()

"""
train_q_controller.py

Q-style contextual bandit controller for the MATCH→SELECT→PREDICT→MIX gate chain.

Problem with the old classifier controller:
  - GPT-only is the best per-example action for many queries.
  - Classifier collapses: learns to predict GPT-only almost always.

Q-controller instead learns: (observation, action) → expected reward.
At inference, all candidate actions are scored and the argmax is chosen.
This avoids collapse because the model learns the *value* of each action,
not just noisy argmax labels.

Also trains on reduced action sets A, B, C to test whether action-space
noise is the bottleneck for the full 46-action set.

No regions, modules, hierarchy, WRITE/PRUNE gates.
Controller only chooses gate parameters (k, tau, alpha).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from utils import load_config, ensure_dirs, get_device, set_seed, build_action_grid, ppl
from gate_chain import evaluate_action_for_queries
from train_controller import build_features


OBS_DIM    = 13
ACTION_DIM = 5
INPUT_DIM  = OBS_DIM + ACTION_DIM   # 18


# ── action features ────────────────────────────────────────────────────────────

def build_action_features(action: dict, max_k: int) -> torch.Tensor:
    """5-dim vector describing one candidate action."""
    k = action["k"]
    return torch.tensor([
        k / max_k,
        float(np.log1p(k) / np.log1p(max_k)),
        float(action["tau"]),
        float(action["alpha"]),
        float(k > 0),                           # is_retrieval flag
    ], dtype=torch.float32)


def build_all_action_features(actions: list[dict], max_k: int) -> torch.Tensor:
    """[A, 5] action feature matrix for all candidate actions."""
    return torch.stack([build_action_features(a, max_k) for a in actions])


# ── base Q data (cacheable) ────────────────────────────────────────────────────

def compute_q_base_data(
    query_data: dict,
    neighbor_data: dict,
    actions: list[dict],
    cfg: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute observation features and per-example per-action rewards on a split.
    Returns:
        obs               [N, 13]
        per_action_rewards [N, A]   reward = NLL_GPT - NLL_action - lambda*k/max_k

    Expensive to recompute (one gate-chain pass per action); cache to disk.
    """
    obs = build_features(query_data, neighbor_data)   # [N, 13]
    N, A = len(query_data["y"]), len(actions)
    per_action_rewards = torch.zeros(N, A)

    for ai, action in enumerate(tqdm(actions, desc="  rewards")):
        m = evaluate_action_for_queries(
            query_data, neighbor_data, action,
            max_k=cfg["max_k"],
            lambda_cost=cfg.get("lambda_cost", 0.0),
            eps=cfg.get("eps", 1e-12),
        )
        per_action_rewards[:, ai] = m["per_example_reward"]

    return obs, per_action_rewards


def build_q_xy(
    obs: torch.Tensor,                 # [N, 13]
    per_action_rewards: torch.Tensor,  # [N, A_full]
    action_feats: torch.Tensor,        # [A_full, 5]
    subset_indices: list[int] | None = None,
    max_samples: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Flatten into (X [N*A_sub, 18], y [N*A_sub]) for Q-controller training.
    subset_indices selects a subset of actions; None = use all.
    """
    N = len(obs)
    if subset_indices is None:
        subset_indices = list(range(len(action_feats)))
    A_sub = len(subset_indices)

    act_sub = action_feats[subset_indices]             # [A_sub, 5]
    rew_sub = per_action_rewards[:, subset_indices]    # [N, A_sub]

    obs_rep = obs.unsqueeze(1).expand(N, A_sub, -1).reshape(N * A_sub, -1)
    act_rep = act_sub.unsqueeze(0).expand(N, A_sub, -1).reshape(N * A_sub, -1)
    X = torch.cat([obs_rep, act_rep], dim=-1)          # [N*A_sub, 18]
    y = rew_sub.reshape(N * A_sub)                     # [N*A_sub]

    if max_samples > 0 and len(y) > max_samples:
        idx = torch.randperm(len(y))[:max_samples]
        X, y = X[idx], y[idx]

    return X, y


# ── QMLP ───────────────────────────────────────────────────────────────────────

class QMLP(nn.Module):
    """Reward predictor: (obs_features || action_features) → scalar reward."""
    def __init__(self, input_dim: int = INPUT_DIM, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)   # [N]


def train_q_mlp(
    X: torch.Tensor,   # [N, feat_dim]
    y: torch.Tensor,   # [N] rewards
    cfg: dict,
    device: torch.device,
) -> QMLP:
    """Train QMLP with MSE loss; early-stop on q_dev split."""
    mu  = X.mean(0, keepdim=True)
    std = X.std(0, keepdim=True).clamp(min=1e-6)
    X_n = (X - mu) / std

    dev_frac = cfg.get("q_dev_fraction", 0.2)
    n_train  = int(len(X) * (1.0 - dev_frac))
    perm     = torch.randperm(len(X))
    tr_idx, dev_idx = perm[:n_train], perm[n_train:]
    X_tr, y_tr   = X_n[tr_idx], y[tr_idx]
    X_dev, y_dev = X_n[dev_idx].to(device), y[dev_idx].to(device)

    model   = QMLP(X.shape[1], hidden_dim=cfg.get("q_controller_hidden", 128)).to(device)
    opt     = torch.optim.Adam(
        model.parameters(),
        lr=cfg.get("q_controller_lr", 1e-3),
        weight_decay=cfg.get("q_controller_weight_decay", 1e-4),
    )
    loss_fn = nn.MSELoss()
    loader  = DataLoader(TensorDataset(X_tr, y_tr),
                         batch_size=cfg.get("q_controller_batch_size", 2048),
                         shuffle=True)

    epochs       = cfg.get("q_controller_epochs", 30)
    best_dev_mse = float("inf")
    best_state: dict | None = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * len(xb)

        model.eval()
        with torch.no_grad():
            dev_mse = loss_fn(model(X_dev), y_dev).item()
        if dev_mse < best_dev_mse:
            best_dev_mse = dev_mse
            best_state   = {k2: v.clone() for k2, v in model.state_dict().items()}
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs}  "
                  f"train_mse={total_loss/len(X_tr):.5f}  dev_mse={dev_mse:.5f}")

    if best_state:
        model.load_state_dict(best_state)

    model.register_buffer("feat_mu",  mu)
    model.register_buffer("feat_std", std)
    return model


# ── inference ──────────────────────────────────────────────────────────────────

def apply_q_controller(
    model: QMLP,
    obs_features: torch.Tensor,    # [N, 13]
    action_feats: torch.Tensor,    # [A_sub, 5]
    device: torch.device,
) -> torch.Tensor:
    """Score all actions per query; return chosen action indices [N] into action_feats."""
    model.eval()
    mu  = model.get_buffer("feat_mu").to(device)
    std = model.get_buffer("feat_std").to(device)

    N, A = len(obs_features), len(action_feats)
    obs_rep = obs_features.unsqueeze(1).expand(N, A, -1)
    act_rep = action_feats.unsqueeze(0).expand(N, A, -1)
    X   = torch.cat([obs_rep, act_rep], dim=-1).reshape(N * A, -1).to(device)
    X_n = (X - mu) / std

    with torch.no_grad():
        scores = model(X_n).reshape(N, A)

    return scores.argmax(-1).cpu()


# ── val evaluation ─────────────────────────────────────────────────────────────

def eval_chosen_on_val(
    chosen: torch.Tensor,          # [N] indices into actions_subset
    actions_subset: list[dict],
    val_data: dict,
    val_nbrs: dict,
    cfg: dict,
) -> dict:
    """Evaluate chosen actions on val. Reports pure NLL (lambda_cost=0)."""
    nll_list, k_list, ret_list = [], [], []
    action_hist: dict[str, int] = {}

    for ai, action in enumerate(actions_subset):
        mask = (chosen == ai)
        if mask.sum() == 0:
            continue
        action_hist[action["name"]] = int(mask.sum())
        m = evaluate_action_for_queries(
            {k2: v[mask] if k2 != "metadata" else v for k2, v in val_data.items()},
            {k2: v[mask] for k2, v in val_nbrs.items()},
            action,
            max_k=cfg["max_k"], lambda_cost=0.0, eps=cfg.get("eps", 1e-12),
        )
        nll_list.extend(m["per_example_nll"].tolist())
        k_list.extend([action["k"]] * int(mask.sum()))
        ret_list.extend([float(action["k"] > 0)] * int(mask.sum()))

    mean_nll = float(np.mean(nll_list))
    return {
        "mean_nll":        mean_nll,
        "mean_ppl":        ppl(mean_nll),
        "mean_k":          float(np.mean(k_list)),
        "retrieval_usage": float(np.mean(ret_list)),
        "action_hist":     action_hist,
    }


# ── save / load ────────────────────────────────────────────────────────────────

def save_q_model(model: QMLP, action_subset_names: list[str], path: Path):
    net_state = {k2: v for k2, v in model.state_dict().items()
                 if k2 not in ("feat_mu", "feat_std")}
    torch.save({
        "state_dict":          net_state,
        "feat_mu":             model.get_buffer("feat_mu"),
        "feat_std":            model.get_buffer("feat_std"),
        "action_subset_names": action_subset_names,
        "input_dim":           model.net[0].in_features,
        "hidden_dim":          model.net[0].out_features,
    }, path)


def load_q_model(path: Path, device: torch.device = torch.device("cpu")) -> tuple["QMLP", list[str]]:
    ckpt  = torch.load(path, map_location="cpu", weights_only=False)
    model = QMLP(ckpt["input_dim"], ckpt["hidden_dim"]).to(device)
    model.register_buffer("feat_mu",  ckpt["feat_mu"])
    model.register_buffer("feat_std", ckpt["feat_std"])
    model.load_state_dict(ckpt["state_dict"], strict=False)
    return model, ckpt["action_subset_names"]


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train Q-style contextual bandit controller(s)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Recompute Q base data even if cached")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    ensure_dirs(cfg)
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg)
    print(f"Device: {device}")

    states_dir  = Path(cfg["states_dir"])
    nbrs_dir    = Path(cfg["neighbors_dir"])
    reports_dir = Path(cfg["reports_dir"])
    models_dir  = Path(cfg["models_dir"])
    k_max       = cfg["max_k"]

    print("\nLoading data …")
    ct_data  = torch.load(states_dir / "controller_train.pt", weights_only=False)
    val_data = torch.load(states_dir / "val.pt",             weights_only=False)
    ct_nbrs  = torch.load(nbrs_dir   / f"controller_train_top{k_max}.pt", weights_only=False)
    val_nbrs = torch.load(nbrs_dir   / f"val_top{k_max}.pt",              weights_only=False)

    actions       = build_action_grid(cfg)
    all_act_feats = build_all_action_features(actions, k_max)   # [A, 5]
    print(f"Action grid: {len(actions)} actions")

    # ── Q base data (compute once, cache for sklearn script too) ──────────────
    cache_path = models_dir / "q_base_data.pt"
    if cache_path.exists() and not args.force_recompute:
        print("\nLoading cached Q base data …")
        cache              = torch.load(cache_path, weights_only=False)
        X_obs              = cache["X_obs"]
        per_action_rewards = cache["per_action_rewards"]
    else:
        print("\n[1] Computing Q base data …")
        X_obs, per_action_rewards = compute_q_base_data(ct_data, ct_nbrs, actions, cfg)
        torch.save({"X_obs": X_obs, "per_action_rewards": per_action_rewards}, cache_path)
        print(f"    Cached → {cache_path}")
    print(f"    obs={X_obs.shape}  rewards={per_action_rewards.shape}")

    print("\nBuilding val observation features …")
    X_obs_val = build_features(val_data, val_nbrs)   # [N_val, 13]

    # ── best fixed action for set C ───────────────────────────────────────────
    with open(reports_dir / "fixed_baselines.json") as f:
        baselines = json.load(f)
    best_fixed_name = baselines["best_train_selected_name"]
    print(f"Best fixed action (CT-selected): {best_fixed_name}")

    # ── action subsets ────────────────────────────────────────────────────────
    name_to_idx = {a["name"]: i for i, a in enumerate(actions)}
    action_subsets: dict[str, list[int]] = {
        "full": list(range(len(actions))),
        "A": [name_to_idx[n] for n in
              ["gpt_only", "k16_t0.05_a0.75", "k32_t0.05_a0.75"]],
        "B": [name_to_idx[n] for n in
              ["gpt_only", "k8_t0.05_a0.75", "k16_t0.05_a0.75",
               "k32_t0.05_a0.75", "k64_t0.05_a0.75"]],
        "C": [name_to_idx["gpt_only"], name_to_idx[best_fixed_name]],
    }

    max_q_samples = cfg.get("max_q_samples", 300000)
    all_results: dict[str, dict] = {}

    for set_name, subset_indices in action_subsets.items():
        actions_sub   = [actions[i] for i in subset_indices]
        act_feats_sub = all_act_feats[subset_indices]

        print(f"\n[Q-MLP set={set_name}]  {len(subset_indices)} actions …")
        X_q, y_q = build_q_xy(
            X_obs, per_action_rewards, all_act_feats,
            subset_indices=subset_indices, max_samples=max_q_samples,
        )
        print(f"  Q samples={len(X_q)}  "
              f"reward mean={y_q.mean():.4f}  std={y_q.std():.4f}")

        model   = train_q_mlp(X_q, y_q, cfg, device)
        chosen  = apply_q_controller(model, X_obs_val, act_feats_sub, device)
        metrics = eval_chosen_on_val(chosen, actions_sub, val_data, val_nbrs, cfg)
        all_results[f"Q-MLP-{set_name}"] = metrics

        print(f"  val NLL={metrics['mean_nll']:.4f}  PPL={metrics['mean_ppl']:.2f}  "
              f"avg_k={metrics['mean_k']:.1f}  ret={metrics['retrieval_usage']*100:.0f}%")
        top5 = sorted(metrics["action_hist"].items(), key=lambda x: -x[1])[:5]
        print(f"  Top actions: {top5}")

        save_q_model(model, [a["name"] for a in actions_sub],
                     models_dir / f"q_mlp_{set_name}.pt")
        print(f"  Saved → models/q_mlp_{set_name}.pt")

    # ── save ──────────────────────────────────────────────────────────────────
    out = {k2: {k3: v for k3, v in m.items()} for k2, m in all_results.items()}
    with open(reports_dir / "q_controller_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {reports_dir / 'q_controller_metrics.json'}")

    import pandas as pd
    for method, m in all_results.items():
        tag  = method.replace("/", "_").replace(" ", "_")
        hist = m.get("action_hist", {})
        total = sum(hist.values()) or 1
        rows  = [{"action": n, "count": c, "fraction": c / total}
                 for n, c in sorted(hist.items(), key=lambda x: -x[1])]
        pd.DataFrame(rows).to_csv(reports_dir / f"q_dist_{tag}.csv", index=False)

    print("\nDone.")


if __name__ == "__main__":
    main()

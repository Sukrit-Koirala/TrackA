"""
train_controller.py

Steps:
  1. Build observation features for each controller_train and val example.
     Features are inference-time-only (no true labels).
  2. For each controller_train example, evaluate all candidate actions and
     label it with the best action (min NLL).  Supervised best-action.
  3. Train a small MLP: features → best_action_index.
  4. Search an entropy-threshold heuristic on controller_train.
  5. Evaluate controller, heuristic, and oracle on val.
  6. Save model weights and metrics.

Observation features (13-dim, no true label used):
  gpt_entropy
  gpt_top1_prob
  gpt_margin                     = top1_prob - top2_prob
  nearest_neighbor_similarity    = sims[:, 0]
  mean_top4_similarity
  mean_top8_similarity
  mean_top16_similarity
  mean_top32_similarity
  std_top32_similarity
  neighbor_label_entropy_top8
  neighbor_label_entropy_top32
  gpt_top1_matches_neighbor_mode_top8
  gpt_top1_matches_neighbor_mode_top32
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


# ── observation features ───────────────────────────────────────────────────────

def _label_entropy(neighbor_y: torch.Tensor, k: int) -> torch.Tensor:
    """
    Compute entropy of the next-token label distribution in top-k neighbors.
    [N, max_k] → [N]  (only labels[:k] used, no true y_true involved)
    """
    y_np = neighbor_y[:, :k].numpy()
    N    = y_np.shape[0]
    ents = np.zeros(N, dtype=np.float32)
    for i in range(N):
        labels = y_np[i]
        _, counts = np.unique(labels, return_counts=True)
        probs  = counts / counts.sum()
        ents[i] = -float(np.sum(probs * np.log(probs + 1e-10)))
    return torch.from_numpy(ents)


def _mode_match(gpt_top1_id: torch.Tensor, neighbor_y: torch.Tensor, k: int) -> torch.Tensor:
    """
    1 if GPT's top-1 prediction matches the mode (most frequent label)
    among the top-k neighbor labels, else 0.
    """
    y_np = neighbor_y[:, :k].numpy()
    g_np = gpt_top1_id.numpy()
    N    = y_np.shape[0]
    out  = np.zeros(N, dtype=np.float32)
    for i in range(N):
        labels = y_np[i]
        mode   = np.bincount(labels).argmax()
        out[i] = float(g_np[i] == mode)
    return torch.from_numpy(out)


def build_features(query_data: dict, neighbor_data: dict) -> torch.Tensor:
    """Return [N, 13] float feature matrix. No true labels used."""
    sims     = neighbor_data["neighbor_sims"]          # [N, 64]
    nbr_y    = neighbor_data["neighbor_y"]             # [N, 64]
    top1_id  = query_data["gpt_top1_id"]               # [N]

    feats = torch.stack([
        query_data["gpt_entropy"],                     # f0
        query_data["gpt_top1_prob"],                   # f1
        query_data["gpt_top1_prob"] - query_data["gpt_top2_prob"],   # f2 margin
        sims[:, 0],                                    # f3 nearest sim
        sims[:, :4].mean(-1),                          # f4
        sims[:, :8].mean(-1),                          # f5
        sims[:, :16].mean(-1),                         # f6
        sims[:, :32].mean(-1),                         # f7
        sims[:, :32].std(-1),                          # f8
        _label_entropy(nbr_y, 8),                      # f9
        _label_entropy(nbr_y, 32),                     # f10
        _mode_match(top1_id, nbr_y, 8),                # f11
        _mode_match(top1_id, nbr_y, 32),               # f12
    ], dim=-1)                                         # [N, 13]

    return feats.float()


# ── best-action labelling ──────────────────────────────────────────────────────

def label_best_actions(
    query_data: dict,
    neighbor_data: dict,
    actions: list[dict],
    cfg: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each example, evaluate all actions and label with the one that
    minimises NLL on that example.  Returns (best_indices [N], per_action_nlls [N, A]).
    """
    N = len(query_data["y"])
    A = len(actions)
    per_action_nlls = torch.zeros(N, A)

    for ai, action in enumerate(tqdm(actions, desc="  Labelling actions")):
        m = evaluate_action_for_queries(
            query_data, neighbor_data, action,
            max_k=cfg["max_k"], lambda_cost=cfg.get("lambda_cost", 0.0),
            eps=cfg.get("eps", 1e-12),
        )
        per_action_nlls[:, ai] = m["per_example_nll"]

    best_indices = per_action_nlls.argmin(dim=-1)   # [N]
    return best_indices, per_action_nlls


# ── MLP controller ─────────────────────────────────────────────────────────────

class ControllerMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp(
    X: torch.Tensor,          # [N, feat_dim]
    y: torch.Tensor,          # [N] class indices
    cfg: dict,
    device: torch.device,
    num_actions: int = 0,     # must be passed explicitly; y.max()+1 may miss unseen actions
) -> ControllerMLP:
    # Standardise features
    mu  = X.mean(0, keepdim=True)
    std = X.std(0, keepdim=True).clamp(min=1e-6)
    X_n = (X - mu) / std

    if num_actions <= 0:
        raise ValueError("pass num_actions=len(actions) explicitly to avoid missing-class bugs")
    model = ControllerMLP(X.shape[1], cfg["controller_hidden"], num_actions).to(device)
    opt   = torch.optim.Adam(model.parameters(),
                              lr=cfg["controller_lr"],
                              weight_decay=cfg.get("controller_weight_decay", 1e-4))
    loss_fn = nn.CrossEntropyLoss()

    ds     = TensorDataset(X_n, y)
    loader = DataLoader(ds, batch_size=cfg["controller_batch_size"], shuffle=True)

    for epoch in range(cfg["controller_epochs"]):
        model.train()
        total_loss = 0.0
        correct    = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss   = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
            correct    += (logits.argmax(-1) == yb).sum().item()
        acc = correct / len(X)
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1:3d}/{cfg['controller_epochs']}  "
                  f"loss={total_loss/len(X):.4f}  acc={acc:.3f}")

    # Store normalisation stats on the model for later use
    model.register_buffer("feat_mu",  mu)
    model.register_buffer("feat_std", std)
    return model


def apply_controller(
    model: ControllerMLP,
    X: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Return action indices chosen by the controller. [N] LongTensor."""
    model.eval()
    X_d  = X.to(device)
    mu   = model.get_buffer("feat_mu").to(device)
    std  = model.get_buffer("feat_std").to(device)
    X_n  = (X_d - mu) / std
    with torch.no_grad():
        logits = model(X_n)
    return logits.argmax(-1).cpu()


# ── entropy heuristic ──────────────────────────────────────────────────────────

def entropy_heuristic_search(
    ct_data: dict,
    ct_nbrs: dict,
    ct_features: torch.Tensor,
    actions: list[dict],
    per_action_nlls_ct: torch.Tensor,   # [N_ct, A]
    cfg: dict,
) -> dict:
    """
    Search (threshold, retrieval_action) on controller_train.
    If gpt_entropy > threshold → use retrieval_action; else GPT-only.
    Returns the best config found.
    """
    retrieval_actions = [a for a in actions if a["k"] > 0]
    ret_action_names  = [a["name"] for a in actions]

    nll_gpt = ct_data["nll_gpt"].numpy()
    entropy = ct_features[:, 0].numpy()   # gpt_entropy is feature 0

    thresholds = np.linspace(
        cfg.get("entropy_threshold_min", 0.3),
        cfg.get("entropy_threshold_max", 5.0),
        cfg.get("entropy_threshold_steps", 30),
    )

    best_nll   = float("inf")
    best_cfg   = None

    for ai, ret_action in enumerate(retrieval_actions):
        action_idx_in_grid = ret_action_names.index(ret_action["name"])
        nll_ret = per_action_nlls_ct[:, action_idx_in_grid].numpy()

        for threshold in thresholds:
            mask       = entropy > threshold
            nll_hybrid = np.where(mask, nll_ret, nll_gpt)
            mean_nll   = float(nll_hybrid.mean())
            if mean_nll < best_nll:
                best_nll = mean_nll
                best_cfg = {
                    "threshold":      float(threshold),
                    "retrieval_action": ret_action,
                    "ct_nll":         mean_nll,
                    "ct_ppl":         float(np.exp(mean_nll)),
                    "retrieval_frac": float(mask.mean()),
                }

    return best_cfg


def apply_entropy_heuristic(
    data: dict,
    neighbors: dict,
    features: torch.Tensor,
    heuristic: dict,
    actions: list[dict],
    cfg: dict,
) -> dict:
    """Apply the found heuristic to a dataset and return metrics."""
    threshold  = heuristic["threshold"]
    ret_action = heuristic["retrieval_action"]

    entropy = features[:, 0].numpy()
    mask    = entropy > threshold

    nll_gpt = data["nll_gpt"].numpy()
    m_ret   = evaluate_action_for_queries(
        data, neighbors, ret_action,
        max_k=cfg["max_k"], lambda_cost=cfg.get("lambda_cost", 0.0),
        eps=cfg.get("eps", 1e-12),
    )
    nll_ret  = m_ret["per_example_nll"].numpy()
    nll_hyb  = np.where(mask, nll_ret, nll_gpt)

    k_used   = np.where(mask, float(ret_action["k"]), 0.0)

    return {
        "mean_nll":        float(nll_hyb.mean()),
        "mean_ppl":        float(np.exp(nll_hyb.mean())),
        "mean_k":          float(k_used.mean()),
        "retrieval_usage": float(mask.mean()),
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg)
    print(f"Device: {device}")

    states_dir  = Path(cfg["states_dir"])
    nbrs_dir    = Path(cfg["neighbors_dir"])
    reports_dir = Path(cfg["reports_dir"])
    models_dir  = Path(cfg["models_dir"])
    k           = cfg["max_k"]

    print("\nLoading data …")
    ct_data  = torch.load(states_dir / "controller_train.pt", weights_only=False)
    val_data = torch.load(states_dir / "val.pt",             weights_only=False)
    ct_nbrs  = torch.load(nbrs_dir   / f"controller_train_top{k}.pt", weights_only=False)
    val_nbrs = torch.load(nbrs_dir   / f"val_top{k}.pt",              weights_only=False)

    actions = build_action_grid(cfg)
    print(f"Action grid: {len(actions)} actions")

    # ── build features ────────────────────────────────────────────────────────
    print("\n[1/5] Building observation features …")
    print("  controller_train …")
    X_ct  = build_features(ct_data,  ct_nbrs)
    print("  val …")
    X_val = build_features(val_data, val_nbrs)
    print(f"  Feature dim: {X_ct.shape[1]}")

    # ── label best actions on controller_train ────────────────────────────────
    print("\n[2/5] Labelling best actions (controller_train) …")
    best_ct_idx, per_action_nlls_ct = label_best_actions(ct_data, ct_nbrs, actions, cfg)

    # Action distribution on train
    action_counts = torch.bincount(best_ct_idx, minlength=len(actions))
    print("  Top-5 best actions by frequency:")
    for ai in action_counts.argsort(descending=True)[:5]:
        print(f"    [{ai:2d}] {actions[ai]['name']:30s}  count={action_counts[ai].item()}")

    # ── train MLP ─────────────────────────────────────────────────────────────
    print("\n[3/5] Training MLP controller …")
    controller = train_mlp(X_ct, best_ct_idx, cfg, device, num_actions=len(actions))
    # Save network params only — buffers are stored separately so evaluate.py
    # can construct a model of the right size before calling load_state_dict.
    net_state = {k2: v for k2, v in controller.state_dict().items()
                 if k2 not in ("feat_mu", "feat_std")}
    torch.save(net_state, models_dir / "controller.pt")
    torch.save({"feat_mu": controller.feat_mu, "feat_std": controller.feat_std,
                "num_actions": len(actions)},
               models_dir / "controller_norm.pt")
    print(f"  Saved model -> {models_dir / 'controller.pt'}")

    # ── entropy heuristic search ──────────────────────────────────────────────
    print("\n[4/5] Entropy heuristic search (controller_train) …")
    heuristic = entropy_heuristic_search(
        ct_data, ct_nbrs, X_ct, actions, per_action_nlls_ct, cfg
    )
    print(f"  Best threshold: {heuristic['threshold']:.2f}")
    print(f"  Retrieval action: {heuristic['retrieval_action']['name']}")
    print(f"  CT NLL: {heuristic['ct_nll']:.4f}  PPL: {heuristic['ct_ppl']:.2f}")
    print(f"  Retrieval fraction: {heuristic['retrieval_frac']:.3f}")

    # ── evaluate on val ───────────────────────────────────────────────────────
    print("\n[5/5] Evaluating on val …")

    # GPT-only
    nll_gpt_val = float(val_data["nll_gpt"].mean())
    ppl_gpt_val = ppl(nll_gpt_val)

    # Oracle per-example best (val-selected, diagnostic)
    print("  Oracle (val best-per-example) …")
    _, per_action_nlls_val = label_best_actions(val_data, val_nbrs, actions, cfg)
    oracle_nll_val = float(per_action_nlls_val.min(dim=-1).values.mean())
    oracle_ppl_val = ppl(oracle_nll_val)

    # Entropy heuristic on val
    heur_val = apply_entropy_heuristic(val_data, val_nbrs, X_val, heuristic, actions, cfg)

    # Learned controller on val
    chosen_val_idx = apply_controller(controller, X_val, device)
    val_nlls_ctrl  = []
    val_k_ctrl     = []
    val_ret_ctrl   = []
    action_hist    = {}
    for ai, action in enumerate(actions):
        mask = (chosen_val_idx == ai)
        if mask.sum() == 0:
            continue
        action_hist[action["name"]] = int(mask.sum())
        # Evaluate this action for examples where controller chose it
        m_ai = evaluate_action_for_queries(
            {k2: v[mask] if k2 != "metadata" else v for k2, v in val_data.items()},
            {k2: v[mask] for k2, v in val_nbrs.items()},
            action,
            max_k=cfg["max_k"], lambda_cost=cfg.get("lambda_cost", 0.0),
            eps=cfg.get("eps", 1e-12),
        )
        val_nlls_ctrl.extend(m_ai["per_example_nll"].tolist())
        val_k_ctrl.extend([action["k"]] * int(mask.sum()))
        val_ret_ctrl.extend([1.0 if action["k"] > 0 else 0.0] * int(mask.sum()))

    ctrl_nll = float(np.mean(val_nlls_ctrl))
    ctrl_ppl = ppl(ctrl_nll)
    ctrl_k   = float(np.mean(val_k_ctrl))
    ctrl_ret = float(np.mean(val_ret_ctrl))

    # ── print val results ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CONTROLLER RESULTS (val)")
    print("=" * 70)
    print(f"{'Method':<35} {'NLL':>7}  {'PPL':>8}  {'avg_k':>6}  {'ret%':>6}")
    print("-" * 70)
    print(f"  {'GPT-only':<33} {nll_gpt_val:>7.4f}  {ppl_gpt_val:>8.2f}  {'0.0':>6}  {'0%':>6}")
    print(f"  {'Entropy heuristic':<33} {heur_val['mean_nll']:>7.4f}  "
          f"{heur_val['mean_ppl']:>8.2f}  {heur_val['mean_k']:>6.1f}  "
          f"{heur_val['retrieval_usage']*100:>5.0f}%")
    print(f"  {'Learned controller':<33} {ctrl_nll:>7.4f}  {ctrl_ppl:>8.2f}  "
          f"{ctrl_k:>6.1f}  {ctrl_ret*100:>5.0f}%")
    print(f"  {'[DIAG] Oracle per-example (val)':<33} {oracle_nll_val:>7.4f}  "
          f"{oracle_ppl_val:>8.2f}")
    print("=" * 70)

    # ── save results ──────────────────────────────────────────────────────────
    results = {
        "gpt_only_val": {"nll": nll_gpt_val, "ppl": ppl_gpt_val, "mean_k": 0.0, "retrieval_usage": 0.0},
        "entropy_heuristic_val": heur_val,
        "entropy_heuristic_config": {
            k2: (v if k2 != "retrieval_action" else v["name"])
            for k2, v in heuristic.items()
        },
        "learned_controller_val": {
            "nll": ctrl_nll, "ppl": ctrl_ppl,
            "mean_k": ctrl_k, "retrieval_usage": ctrl_ret,
        },
        "oracle_val": {"nll": oracle_nll_val, "ppl": oracle_ppl_val},
        "action_distribution_val": action_hist,
    }
    with open(reports_dir / "controller_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {reports_dir / 'controller_results.json'}")

    # Action distribution CSV
    import pandas as pd
    dist_rows = [{"action": k2, "count": v, "action_idx": actions.index(
        next(a for a in actions if a["name"] == k2))}
        for k2, v in sorted(action_hist.items(), key=lambda x: -x[1])]
    pd.DataFrame(dist_rows).to_csv(reports_dir / "action_distribution.csv", index=False)
    print(f"Saved: {reports_dir / 'action_distribution.csv'}")

    print("\nDone.")


if __name__ == "__main__":
    main()

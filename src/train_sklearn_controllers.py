"""
train_sklearn_controllers.py

Scikit-learn Q-style contextual bandit controllers.

Trains DecisionTree, RandomForest, and HistGradientBoosting regressors on
the same Q-style (obs || action_features) → reward data as the Q-MLP.
Tree-based methods are scale-invariant and serve as strong non-neural baselines
that can reveal whether the features are useful regardless of neural capacity.

At inference, all 46 candidate actions are scored per query and the argmax taken.

Reuses the Q base data cache from train_q_controller.py (outputs/models/q_base_data.pt).

No regions, modules, hierarchy, WRITE/PRUNE gates.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json
import numpy as np
import torch
import joblib

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

from utils import load_config, ensure_dirs, set_seed, build_action_grid, ppl
from train_controller import build_features
from train_q_controller import (
    build_all_action_features, compute_q_base_data, build_q_xy, eval_chosen_on_val,
)


def apply_sklearn_q(
    model,
    obs_np: np.ndarray,       # [N, 13]  unnormalized
    act_np: np.ndarray,       # [A, 5]   unnormalized
) -> np.ndarray:
    """Score all actions for each query; return chosen action indices [N]."""
    N, A = len(obs_np), len(act_np)
    obs_rep = np.repeat(obs_np[:, np.newaxis, :], A, axis=1).reshape(N * A, -1)
    act_rep = np.tile(act_np[np.newaxis, :, :], (N, 1, 1)).reshape(N * A, -1)
    X       = np.concatenate([obs_rep, act_rep], axis=-1)        # [N*A, 18]
    scores  = model.predict(X).reshape(N, A)                     # [N, A]
    return scores.argmax(axis=1)                                  # [N]


def main():
    parser = argparse.ArgumentParser(
        description="Train sklearn Q-controllers for the gate chain")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Recompute Q base data even if cached")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    set_seed(cfg.get("seed", 42))

    states_dir  = Path(cfg["states_dir"])
    nbrs_dir    = Path(cfg["neighbors_dir"])
    reports_dir = Path(cfg["reports_dir"])
    models_dir  = Path(cfg["models_dir"])
    k_max       = cfg["max_k"]

    print("Loading data …")
    ct_data  = torch.load(states_dir / "controller_train.pt", weights_only=False)
    val_data = torch.load(states_dir / "val.pt",             weights_only=False)
    ct_nbrs  = torch.load(nbrs_dir   / f"controller_train_top{k_max}.pt", weights_only=False)
    val_nbrs = torch.load(nbrs_dir   / f"val_top{k_max}.pt",              weights_only=False)

    actions       = build_action_grid(cfg)
    all_act_feats = build_all_action_features(actions, k_max)   # [A, 5]
    print(f"Action grid: {len(actions)} actions")

    # ── Q base data ───────────────────────────────────────────────────────────
    cache_path = models_dir / "q_base_data.pt"
    if cache_path.exists() and not args.force_recompute:
        print("Loading cached Q base data …")
        cache              = torch.load(cache_path, weights_only=False)
        X_obs              = cache["X_obs"]
        per_action_rewards = cache["per_action_rewards"]
    else:
        print("Computing Q base data (run train_q_controller.py first to cache this) …")
        X_obs, per_action_rewards = compute_q_base_data(ct_data, ct_nbrs, actions, cfg)
        torch.save({"X_obs": X_obs, "per_action_rewards": per_action_rewards}, cache_path)

    max_q = cfg.get("max_q_samples", 300000)
    X_q, y_q = build_q_xy(
        X_obs, per_action_rewards, all_act_feats,
        subset_indices=None, max_samples=max_q,
    )
    X_np = X_q.numpy()
    y_np = y_q.numpy()
    print(f"Q dataset: {len(X_np)} samples  input_dim={X_np.shape[1]}")

    # Val obs and action features (numpy, unnormalized — trees are scale-invariant)
    X_obs_val_np  = build_features(val_data, val_nbrs).numpy()   # [N_val, 13]
    act_feats_np  = all_act_feats.numpy()                        # [A, 5]

    sklearn_models = {
        "Q-DecisionTree": DecisionTreeRegressor(
            max_depth=cfg.get("dt_max_depth", 8),
            random_state=cfg.get("seed", 42),
        ),
        "Q-RandomForest": RandomForestRegressor(
            n_estimators=cfg.get("rf_n_estimators", 100),
            max_depth=cfg.get("rf_max_depth", 8),
            n_jobs=-1,
            random_state=cfg.get("seed", 42),
        ),
        "Q-GradientBoosting": HistGradientBoostingRegressor(
            max_iter=cfg.get("gb_max_iter", 100),
            max_depth=cfg.get("gb_max_depth", 4),
            random_state=cfg.get("seed", 42),
        ),
    }

    all_results: dict[str, dict] = {}

    for name, model in sklearn_models.items():
        print(f"\n[{name}] training on {len(X_np)} samples …")
        model.fit(X_np, y_np)
        print(f"  Training done.")

        print(f"  Scoring {len(X_obs_val_np)} val examples × {len(actions)} actions …")
        chosen_np = apply_sklearn_q(model, X_obs_val_np, act_feats_np)
        chosen    = torch.from_numpy(chosen_np.astype(np.int64))

        metrics = eval_chosen_on_val(chosen, actions, val_data, val_nbrs, cfg)
        all_results[name] = metrics

        print(f"  val NLL={metrics['mean_nll']:.4f}  PPL={metrics['mean_ppl']:.2f}  "
              f"avg_k={metrics['mean_k']:.1f}  ret={metrics['retrieval_usage']*100:.0f}%")
        top5 = sorted(metrics["action_hist"].items(), key=lambda x: -x[1])[:5]
        print(f"  Top actions: {top5}")

        tag  = name.lower().replace("-", "_").replace(" ", "_")
        path = models_dir / f"{tag}.joblib"
        joblib.dump(model, path)
        print(f"  Saved → {path}")

    # ── save ──────────────────────────────────────────────────────────────────
    out = {}
    for method, m in all_results.items():
        out[method] = {
            "mean_nll":        m["mean_nll"],
            "mean_ppl":        m["mean_ppl"],
            "mean_k":          m["mean_k"],
            "retrieval_usage": m["retrieval_usage"],
            "action_hist":     m["action_hist"],
        }
    with open(reports_dir / "sklearn_controller_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {reports_dir / 'sklearn_controller_metrics.json'}")
    print("\nDone.")


if __name__ == "__main__":
    main()

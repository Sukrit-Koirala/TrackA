"""
train_write_scorer.py

Trains learned WRITE scorers that predict per-datastore-entry utility.

Models:
  1. Ridge regression      (sklearn)
  2. GradientBoostingRegressor (sklearn)
  3. MLP                   (PyTorch, 2 × 128 hidden)

Training target: utility_net from Mode A (default) or Mode B.
80/20 train/dev split of datastore entries.

Evaluation: dev MSE, Spearman rank correlation, top-B overlap with oracle.

Saves:
  models/write_scorer_linear.joblib
  models/write_scorer_gbr.joblib
  models/write_scorer_mlp.pt
  write_scores/linear_scores.pt
  write_scores/gbr_scores.pt
  write_scores/mlp_scores.pt

Usage:
  python src/train_write_scorer.py \\
    --source outputs_scale_sweep/scale_200k_seed42 \\
    --output outputs_mvp2_write_memory
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

from utils import get_device, set_seed


def spearman_r(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr
        r, _ = spearmanr(a, b)
        return float(r) if not np.isnan(r) else 0.0
    except ImportError:
        # Fallback: rank correlation via argsort
        rank_a = np.argsort(np.argsort(a)).astype(float)
        rank_b = np.argsort(np.argsort(b)).astype(float)
        n = len(a)
        cov = np.mean((rank_a - rank_a.mean()) * (rank_b - rank_b.mean()))
        return float(cov / (rank_a.std() * rank_b.std() + 1e-10))


def top_b_overlap(pred: np.ndarray, truth: np.ndarray, b: int) -> float:
    """Fraction of oracle top-B that appear in predicted top-B."""
    top_pred  = set(np.argsort(pred)[-b:])
    top_truth = set(np.argsort(truth)[-b:])
    return len(top_pred & top_truth) / b


class WriteScorerMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",       required=True)
    parser.add_argument("--output",       required=True)
    parser.add_argument("--utility_mode", default="A", choices=["A", "B", "C"])
    parser.add_argument("--target",       default="utility_net",
                        choices=["utility_net", "utility_positive"])
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--eval_budgets", nargs="+", type=int,
                        default=[1000, 5000, 10000, 25000])
    args = parser.parse_args()

    src = Path(args.source)
    out = Path(args.output)
    models_dir = out / "models"
    scores_dir = out / "write_scores"
    models_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    device = get_device({"device": "cuda"})
    set_seed(args.seed)

    print(f"\ntrain_write_scorer")
    print(f"Source: {src}  |  Output: {out}")
    print(f"Utility mode: {args.utility_mode}  |  Target: {args.target}")

    # ── load features ─────────────────────────────────────────────────────────
    feat_path = out / "features" / "write_features.pt"
    if not feat_path.exists():
        print(f"ERROR: {feat_path} not found. Run build_write_features.py first.")
        sys.exit(1)
    feat_data = torch.load(feat_path, weights_only=False)
    X      = feat_data["X"].numpy().astype(np.float32)   # [N_ds, D_feat]
    N, D_feat = X.shape
    print(f"\nFeatures: {X.shape}")

    # ── load utility labels ───────────────────────────────────────────────────
    util_path = out / "utility" / f"write_utilities_mode{args.utility_mode}.pt"
    if not util_path.exists():
        print(f"ERROR: {util_path} not found. Run compute_write_utilities.py first.")
        sys.exit(1)
    util_data = torch.load(util_path, weights_only=False)
    y_all    = util_data[args.target].numpy().astype(np.float32)   # [N_ds]
    y_oracle = util_data["utility_positive"].numpy().astype(np.float32)  # for overlap eval
    print(f"Target: {args.target}  mean={y_all.mean():.6f}  std={y_all.std():.6f}")
    print(f"Non-zero targets: {(y_all > 0).sum():,}/{N:,}")

    # ── 80/20 split ───────────────────────────────────────────────────────────
    rng     = np.random.default_rng(args.seed)
    perm    = rng.permutation(N)
    n_train = int(N * 0.8)
    tr_idx  = perm[:n_train]
    dev_idx = perm[n_train:]
    X_tr, y_tr   = X[tr_idx], y_all[tr_idx]
    X_dev, y_dev = X[dev_idx], y_all[dev_idx]
    print(f"\nTrain: {len(X_tr):,}  Dev: {len(X_dev):,}")

    # Standardise features (fit on train only)
    mu    = X_tr.mean(axis=0, keepdims=True)
    sigma = X_tr.std(axis=0, keepdims=True) + 1e-6
    X_tr_n  = (X_tr  - mu) / sigma
    X_dev_n = (X_dev - mu) / sigma
    X_all_n = (X     - mu) / sigma

    summary: dict[str, dict] = {}

    # ── 1. Ridge regression ───────────────────────────────────────────────────
    print("\n[1] Ridge regression ...")
    from sklearn.linear_model import Ridge
    import joblib
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr_n, y_tr)
    pred_dev  = ridge.predict(X_dev_n)
    dev_mse   = float(np.mean((pred_dev - y_dev) ** 2))
    dev_r     = spearman_r(pred_dev, y_dev)
    all_preds = ridge.predict(X_all_n)

    overlaps = {b: top_b_overlap(all_preds, y_oracle, b) for b in args.eval_budgets}
    print(f"  dev MSE={dev_mse:.6f}  Spearman={dev_r:.4f}")
    for b, ov in overlaps.items():
        print(f"  top-{b} overlap with oracle: {ov:.3f}")

    joblib.dump({"model": ridge, "mu": mu, "sigma": sigma},
                models_dir / "write_scorer_linear.joblib")
    torch.save({"scores": torch.from_numpy(all_preds), "method": "linear"},
               scores_dir / "linear_scores.pt")
    summary["linear"] = {"dev_mse": dev_mse, "spearman": dev_r, "overlaps": overlaps}

    # ── 2. GradientBoostingRegressor ──────────────────────────────────────────
    print("\n[2] GradientBoostingRegressor ...")
    from sklearn.ensemble import GradientBoostingRegressor
    gbr = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=args.seed,
    )
    gbr.fit(X_tr_n, y_tr)
    pred_dev  = gbr.predict(X_dev_n)
    dev_mse   = float(np.mean((pred_dev - y_dev) ** 2))
    dev_r     = spearman_r(pred_dev, y_dev)
    all_preds = gbr.predict(X_all_n)

    overlaps = {b: top_b_overlap(all_preds, y_oracle, b) for b in args.eval_budgets}
    print(f"  dev MSE={dev_mse:.6f}  Spearman={dev_r:.4f}")
    for b, ov in overlaps.items():
        print(f"  top-{b} overlap with oracle: {ov:.3f}")

    joblib.dump({"model": gbr, "mu": mu, "sigma": sigma},
                models_dir / "write_scorer_gbr.joblib")
    torch.save({"scores": torch.from_numpy(all_preds).float(), "method": "gbr"},
               scores_dir / "gbr_scores.pt")
    summary["gbr"] = {"dev_mse": dev_mse, "spearman": dev_r, "overlaps": overlaps}

    # ── 3. MLP ────────────────────────────────────────────────────────────────
    print("\n[3] MLP write scorer ...")
    X_tr_t  = torch.from_numpy(X_tr_n).float()
    y_tr_t  = torch.from_numpy(y_tr).float()
    X_dev_t = torch.from_numpy(X_dev_n).float().to(device)
    y_dev_t = torch.from_numpy(y_dev).float().to(device)

    model   = WriteScorerMLP(D_feat, hidden=128).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    loader  = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=2048, shuffle=True)

    best_dev_mse = float("inf")
    best_state   = None
    for epoch in range(40):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            dev_mse_ep = loss_fn(model(X_dev_t), y_dev_t).item()
        if dev_mse_ep < best_dev_mse:
            best_dev_mse = dev_mse_ep
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}  dev_mse={dev_mse_ep:.6f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        X_all_t   = torch.from_numpy(X_all_n).float().to(device)
        all_preds = model(X_all_t).cpu().numpy()
        pred_dev_n = model(X_dev_t).cpu().numpy()

    dev_mse = float(np.mean((pred_dev_n - y_dev) ** 2))
    dev_r   = spearman_r(pred_dev_n, y_dev)
    overlaps = {b: top_b_overlap(all_preds, y_oracle, b) for b in args.eval_budgets}
    print(f"  Best dev MSE={best_dev_mse:.6f}  Final dev MSE={dev_mse:.6f}  Spearman={dev_r:.4f}")
    for b, ov in overlaps.items():
        print(f"  top-{b} overlap with oracle: {ov:.3f}")

    torch.save({
        "model_state_dict": best_state,
        "input_dim":        D_feat,
        "hidden_dim":       128,
        "mu":               torch.from_numpy(mu).float(),
        "sigma":            torch.from_numpy(sigma).float(),
        "feature_names":    feat_data["feature_names"],
        "utility_mode":     args.utility_mode,
        "target":           args.target,
    }, models_dir / "write_scorer_mlp.pt")
    torch.save({"scores": torch.from_numpy(all_preds).float(), "method": "mlp"},
               scores_dir / "mlp_scores.pt")
    summary["mlp"] = {"dev_mse": dev_mse, "spearman": dev_r, "overlaps": overlaps}

    # ── save summary ──────────────────────────────────────────────────────────
    with open(out / "reports" / "write_scorer_summary.json"
              if (out / "reports").exists()
              else models_dir / "write_scorer_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("WRITE SCORER SUMMARY")
    print("=" * 60)
    for name, s in summary.items():
        print(f"  {name:<10}  MSE={s['dev_mse']:.6f}  Spearman={s['spearman']:.4f}")
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()

"""
train_bandit_write_scorer.py  --  MVP 4b Stage 2a

Trains action-value scorer on the counterfactual reward dataset.

Models:
  MLP  -- PyTorch, Huber loss + pairwise ranking loss (within decision points)
  GBM  -- sklearn GradientBoostingRegressor

Train/val/test split by stream quartile (temporal order preserved):
  Q1+Q2+Q3 (steps 0-75%) --> train
  Q3 last 10%             --> val   (held out of GBM early stopping)
  Q4 (steps 75-100%)      --> test  (never touched during training)

Usage:
  python src/train_bandit_write_scorer.py \\
    --output outputs_mvp4b_bandit_write_fast \\
    --reward_col reward_penalized \\
    --model mlp gbm \\
    --mlp_hidden 128 64 \\
    --mlp_epochs 50 \\
    --gbm_n_estimators 400 \\
    --seed 42 --device cuda
"""

import json
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error


FEATURE_NAMES = [
    "stream_progress", "gpt_nll", "gpt_entropy",
    "n_persistent", "n_buffers",
    "budget_frac_persistent", "budget_frac_buffers",
    "state_top1_sim", "state_top2_sim", "state_sim_gap",
    "buffer_top1_sim", "buffer_top2_sim",
    "act_update_state", "act_update_buffer", "act_promote_update",
    "act_create_buffer", "act_defer",
    "cand_sim", "cand_rank_norm", "cand_log_support",
    "cand_is_promoted", "cand_entropy", "cand_p_y_true",
    "cand_age_frac", "cand_recency_frac",
]


# ── MLP ───────────────────────────────────────────────────────────────────────

class ActionValueMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(),
                       nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _ranking_loss(pred: torch.Tensor, target: torch.Tensor,
                  groups: torch.Tensor, margin: float = 0.001) -> torch.Tensor:
    """Pairwise ranking loss within each decision-point group."""
    loss = torch.tensor(0.0, device=pred.device)
    n_pairs = 0
    for gid in groups.unique():
        mask = groups == gid
        if mask.sum() < 2:
            continue
        p = pred[mask]
        t = target[mask]
        # all pairs where target differs by >= margin
        diff_t = t.unsqueeze(0) - t.unsqueeze(1)   # [n, n]
        diff_p = p.unsqueeze(0) - p.unsqueeze(1)
        pair_mask = diff_t > margin                  # i better than j
        if not pair_mask.any():
            continue
        # ReLU hinge: want diff_p > 0 when diff_t > 0
        loss += torch.clamp(-diff_p[pair_mask], min=0.0).mean()
        n_pairs += pair_mask.sum().item()
    return loss / max(n_pairs, 1)


def train_mlp(
    X_tr: np.ndarray, y_tr: np.ndarray, grp_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    hidden: list[int], epochs: int, lr: float,
    rank_weight: float, device: torch.device,
    out_dir: Path,
) -> ActionValueMLP:
    in_dim = X_tr.shape[1]
    model  = ActionValueMLP(in_dim, hidden).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    huber  = nn.HuberLoss(delta=0.1)

    X_t  = torch.from_numpy(X_tr).float().to(device)
    y_t  = torch.from_numpy(y_tr).float().to(device)
    g_t  = torch.from_numpy(grp_tr).long().to(device)
    X_v  = torch.from_numpy(X_val).float().to(device)
    y_v  = torch.from_numpy(y_val).float().to(device)

    # Mini-batch loader
    ds      = TensorDataset(X_t, y_t, g_t)
    loader  = DataLoader(ds, batch_size=2048, shuffle=True)

    best_val = np.inf
    best_state = None
    history = []

    print(f"  MLP training: {in_dim} -> {hidden} -> 1  "
          f"({sum(p.numel() for p in model.parameters()):,} params)")

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for xb, yb, gb in loader:
            pred  = model(xb)
            hl    = huber(pred, yb)
            rl    = _ranking_loss(pred, yb, gb) if rank_weight > 0 else torch.tensor(0.0)
            loss  = hl + rank_weight * rl
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_v).cpu().numpy()
        val_mae  = float(mean_absolute_error(y_val, val_pred))
        val_loss = float(huber(torch.from_numpy(val_pred).to(device), y_v).item())

        history.append({"epoch": ep, "train_loss": ep_loss / len(loader),
                        "val_mae": val_mae, "val_loss": val_loss})
        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_ep    = ep

        if ep % 10 == 0 or ep == epochs:
            print(f"  epoch {ep:3d}/{epochs}  train_loss={ep_loss/len(loader):.5f}  "
                  f"val_mae={val_mae:.5f}  val_loss={val_loss:.5f}"
                  + ("  <-- best" if ep == best_ep else ""))

    model.load_state_dict(best_state)
    print(f"  Best epoch: {best_ep}  val_loss={best_val:.5f}")
    with open(out_dir / "mlp_train_history.json", "w") as f:
        json.dump(history, f, indent=2)
    return model


# ── GBM ───────────────────────────────────────────────────────────────────────

def train_gbm(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    n_estimators: int, max_depth: int, lr: float,
    out_dir: Path,
) -> GradientBoostingRegressor:
    print(f"  GBM training: n_estimators={n_estimators}  "
          f"depth={max_depth}  lr={lr}")
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=lr,
        subsample=0.8,
        min_samples_leaf=20,
        validation_fraction=None,   # we handle val ourselves
        random_state=42,
        verbose=0,
    )
    model.fit(X_tr, y_tr)
    val_pred = model.predict(X_val)
    val_mae  = float(mean_absolute_error(y_val, val_pred))
    print(f"  GBM val MAE: {val_mae:.5f}")

    # Feature importances
    fi = pd.DataFrame({
        "feature":    FEATURE_NAMES[:X_tr.shape[1]],
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    fi.to_csv(out_dir / "gbm_feature_importance.csv", index=False)
    print(f"  Top-5 features:\n{fi.head(5).to_string(index=False)}")
    return model


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output",           required=True)
    ap.add_argument("--reward_col",       default="reward_penalized")
    ap.add_argument("--model",            nargs="+", default=["mlp", "gbm"],
                    choices=["mlp", "gbm"])
    ap.add_argument("--mlp_hidden",       nargs="+", type=int, default=[128, 64])
    ap.add_argument("--mlp_epochs",       type=int, default=50)
    ap.add_argument("--mlp_lr",           type=float, default=3e-4)
    ap.add_argument("--mlp_rank_weight",  type=float, default=0.5)
    ap.add_argument("--gbm_n_estimators", type=int, default=400)
    ap.add_argument("--gbm_max_depth",    type=int, default=5)
    ap.add_argument("--gbm_lr",           type=float, default=0.05)
    ap.add_argument("--val_frac",         type=float, default=0.10,
                    help="fraction of train set used for validation (tail of Q3)")
    ap.add_argument("--seed",    type=int,   default=42)
    ap.add_argument("--device",  default="cuda")
    ap.add_argument("--force",   action="store_true")
    args = ap.parse_args()

    import random, sys
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir   = Path(args.output)
    data_dir  = out_dir / "reward_dataset"
    model_dir = out_dir / "scorer_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    sentinel = model_dir / "training_complete.json"
    if sentinel.exists() and not args.force:
        print(f"[cached] {sentinel}")
        return

    # ── load data ─────────────────────────────────────────────────────────────
    print("Loading reward dataset ...")
    df = pd.read_parquet(data_dir / "actions.parquet")
    print(f"  {len(df):,} records")

    rcol = args.reward_col
    if rcol not in df.columns:
        print(f"WARNING: {rcol} not found, falling back to reward_full")
        rcol = "reward_full"

    feat_cols = [f for f in FEATURE_NAMES if f in df.columns]
    X_all  = df[feat_cols].values.astype(np.float32)
    y_all  = df[rcol].values.astype(np.float32)
    steps  = df["step"].values.astype(np.int64)
    n_all  = len(df)

    # ── temporal split (by step quartile) ─────────────────────────────────────
    step_max   = int(steps.max()) + 1
    test_lo    = int(step_max * 0.75)
    train_hi   = int(step_max * 0.75)
    val_lo     = int(train_hi * (1 - args.val_frac))

    test_mask  = steps >= test_lo
    val_mask   = (steps >= val_lo) & (steps < train_hi)
    train_mask = steps < val_lo

    X_tr, y_tr = X_all[train_mask], y_all[train_mask]
    X_v,  y_v  = X_all[val_mask],   y_all[val_mask]
    X_te, y_te = X_all[test_mask],  y_all[test_mask]
    grp_tr     = steps[train_mask].astype(np.int64)

    print(f"  Train: {len(X_tr):,}  Val: {len(X_v):,}  Test: {len(X_te):,}")
    print(f"  Features: {len(feat_cols)}")

    # ── normalize features ─────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_tr_n = scaler.fit_transform(X_tr)
    X_v_n  = scaler.transform(X_v)
    X_te_n = scaler.transform(X_te)

    scaler_path = model_dir / "feature_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Saved scaler: {scaler_path}")

    # Save feature schema
    schema = {
        "feature_names": feat_cols,
        "reward_col":    rcol,
        "n_features":    len(feat_cols),
        "n_train":       int(len(X_tr)),
        "n_val":         int(len(X_v)),
        "n_test":        int(len(X_te)),
        "step_split": {
            "train_max": int(val_lo),
            "val_lo":    int(val_lo),
            "test_lo":   int(test_lo),
        },
    }

    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda"
                          else "cpu")
    print(f"  Device: {device}")

    # ── MLP ───────────────────────────────────────────────────────────────────
    if "mlp" in args.model:
        print("\n--- MLP ---")
        mlp = train_mlp(
            X_tr_n.astype(np.float32), y_tr, grp_tr,
            X_v_n.astype(np.float32),  y_v,
            hidden=args.mlp_hidden, epochs=args.mlp_epochs,
            lr=args.mlp_lr, rank_weight=args.mlp_rank_weight,
            device=device, out_dir=model_dir,
        )
        mlp_path = model_dir / "mlp.pt"
        torch.save({
            "state_dict": mlp.state_dict(),
            "hidden":     args.mlp_hidden,
            "in_dim":     X_tr_n.shape[1],
        }, mlp_path)
        print(f"  Saved: {mlp_path}")

        mlp.eval()
        with torch.no_grad():
            te_pred = mlp(torch.from_numpy(X_te_n.astype(np.float32)).to(device)
                          ).cpu().numpy()
        te_mae = float(mean_absolute_error(y_te, te_pred))
        te_rho, _ = __import__("scipy.stats", fromlist=["spearmanr"]).spearmanr(y_te, te_pred)
        results["mlp"] = {"test_mae": te_mae, "test_spearman": float(te_rho)}
        print(f"  MLP test MAE={te_mae:.5f}  Spearman={te_rho:.4f}")

    # ── GBM ───────────────────────────────────────────────────────────────────
    if "gbm" in args.model:
        print("\n--- GBM ---")
        gbm = train_gbm(
            X_tr_n, y_tr, X_v_n, y_v,
            n_estimators=args.gbm_n_estimators,
            max_depth=args.gbm_max_depth,
            lr=args.gbm_lr,
            out_dir=model_dir,
        )
        gbm_path = model_dir / "gbm.pkl"
        with open(gbm_path, "wb") as f:
            pickle.dump(gbm, f)
        print(f"  Saved: {gbm_path}")

        te_pred  = gbm.predict(X_te_n)
        te_mae   = float(mean_absolute_error(y_te, te_pred))
        te_rho, _ = __import__("scipy.stats", fromlist=["spearmanr"]).spearmanr(y_te, te_pred)
        results["gbm"] = {"test_mae": te_mae, "test_spearman": float(te_rho)}
        print(f"  GBM test MAE={te_mae:.5f}  Spearman={te_rho:.4f}")

    # ── save summary ──────────────────────────────────────────────────────────
    schema["model_results"] = results
    with open(model_dir / "training_schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    summary = {
        "models_trained": list(results.keys()),
        "feature_names":  feat_cols,
        "reward_col":     rcol,
        "results":        results,
        "scaler":         str(scaler_path),
    }
    with open(sentinel, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTraining complete. Results:")
    for m, r in results.items():
        print(f"  {m.upper()}:  MAE={r['test_mae']:.5f}  "
              f"Spearman={r['test_spearman']:.4f}")
    print(f"\nSaved: {model_dir}/")


if __name__ == "__main__":
    main()

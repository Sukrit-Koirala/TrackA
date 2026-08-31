"""
train_mvp4b1_scorer.py  --  MVP 4b.1 Stage 3

Trains MLP + GBM scorer on mixed-source reward dataset.
Source weights: oracle=0.35, low_memory=0.25, onpolicy=0.40

Usage:
  python src/train_mvp4b1_scorer.py \\
    --output outputs_mvp4b1_onpolicy_bandit_write_clean_fast \\
    --reward_col reward_penalized \\
    --trajectory_types oracle low_memory \\
    --model mlp gbm \\
    --seed 42 --device cuda
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from mvp4b1_common import (
    ARTIFACT_VERSION, FEATURE_NAMES, N_FEAT, REWARD_COLS,
    atomic_write_json, atomic_parquet_save, validate_json_cache,
)

SOURCE_WEIGHTS = {"oracle": 0.35, "low_memory": 0.25, "onpolicy": 0.40}


# ── MLP ───────────────────────────────────────────────────────────────────────

class BanditMLP(nn.Module):
    def __init__(self, n_feat: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _temporal_split(df: pd.DataFrame):
    max_step  = df["step"].max()
    q3_cutoff = int(max_step * 0.75)
    train = df[df["step"] <  q3_cutoff]
    val   = df[(df["step"] >= q3_cutoff) & (df["step"] < int(max_step * 0.875))]
    test  = df[df["step"] >= int(max_step * 0.875)]
    return train, val, test


def train_mlp(X_tr, y_tr, X_val, y_val,
              sample_w_tr, n_epochs: int, device: torch.device, seed: int):
    torch.manual_seed(seed)
    model = BanditMLP(N_FEAT).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    huber = nn.HuberLoss(delta=1.0, reduction="none")

    X_t  = torch.from_numpy(X_tr).float().to(device)
    y_t  = torch.from_numpy(y_tr).float().to(device)
    w_t  = torch.from_numpy(sample_w_tr).float().to(device)
    X_v  = torch.from_numpy(X_val).float().to(device)
    y_v  = torch.from_numpy(y_val).float().to(device)

    best_val = float("inf")
    best_sd  = None
    n_tr     = len(X_t)
    bs       = min(1024, n_tr)

    for ep in range(n_epochs):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        ep_loss = 0.0; n_batches = 0
        for i in range(0, n_tr, bs):
            idx   = perm[i: i + bs]
            xb, yb, wb = X_t[idx], y_t[idx], w_t[idx]
            opt.zero_grad()
            pred  = model(xb)
            loss  = (huber(pred, yb) * wb).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item(); n_batches += 1
        sched.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(X_v).cpu().numpy()
            val_loss = float(huber(
                torch.from_numpy(val_pred).to(device), y_v
            ).mean().item())
        if val_loss < best_val:
            best_val = val_loss
            best_sd  = {k: v.cpu() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == n_epochs - 1:
            print(f"    ep {ep:3d}/{n_epochs}  "
                  f"train={ep_loss/max(n_batches,1):.4f}  val={val_loss:.4f}")

    model.load_state_dict(best_sd)
    return model


def train_gbm(X_tr, y_tr, X_val, y_val,
              sample_w_tr, n_estimators: int, seed: int):
    try:
        from lightgbm import LGBMRegressor
        gbm = LGBMRegressor(
            n_estimators=n_estimators, learning_rate=0.05,
            num_leaves=63, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1,
            n_jobs=4, random_state=seed, verbose=-1,
        )
        gbm.fit(X_tr, y_tr, sample_weight=sample_w_tr,
                eval_set=[(X_val, y_val)], callbacks=[])
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        gbm = GradientBoostingRegressor(
            n_estimators=min(n_estimators, 200), learning_rate=0.05,
            max_depth=5, subsample=0.8, random_state=seed, verbose=0,
        )
        gbm.fit(X_tr, y_tr, sample_weight=sample_w_tr)
    return gbm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output",            required=True)
    ap.add_argument("--reward_col",        default="reward_penalized")
    ap.add_argument("--trajectory_types",  nargs="+",
                    default=["oracle", "low_memory"])
    ap.add_argument("--model",             nargs="+",
                    default=["mlp", "gbm"],
                    choices=["mlp", "gbm"])
    ap.add_argument("--mlp_epochs",        type=int, default=60)
    ap.add_argument("--gbm_n_estimators",  type=int, default=400)
    ap.add_argument("--seed",              type=int, default=42)
    ap.add_argument("--device",            default="cuda")
    ap.add_argument("--force",             action="store_true")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    out_dir    = Path(args.output)
    model_dir  = out_dir / "scorer_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    sentinel   = model_dir / "training_schema.json"

    cached = validate_json_cache(sentinel, ["artifact_version", "models_trained"])
    if cached and not args.force:
        print(f"[cached] {sentinel}")
        return

    rc = args.reward_col
    frames = []
    for ttype in args.trajectory_types:
        rd = out_dir / "reward_datasets" / ttype / "actions.parquet"
        if not rd.exists():
            print(f"  WARNING: missing {rd}, skipping")
            continue
        df = pd.read_parquet(rd)
        df["_source"] = ttype
        frames.append(df)
        print(f"  Loaded {ttype}: {len(df):,} records")

    if not frames:
        raise RuntimeError("No reward datasets found!")

    all_df = pd.concat(frames, ignore_index=True)

    # Per-source temporal splits + sample weights
    train_parts, val_parts, test_parts = [], [], []
    for ttype, grp in all_df.groupby("_source"):
        tr, va, te = _temporal_split(grp.reset_index(drop=True))
        w_raw = SOURCE_WEIGHTS.get(ttype, 0.25)
        for part, store in [(tr, train_parts), (va, val_parts), (te, test_parts)]:
            part = part.copy()
            part["_sample_w"] = w_raw / max(len(part), 1)
            store.append(part)

    tr_df  = pd.concat(train_parts,  ignore_index=True)
    val_df = pd.concat(val_parts,    ignore_index=True)
    test_df = pd.concat(test_parts,  ignore_index=True)

    # Normalise sample weights to [0,1] across training set
    tr_df["_sample_w"] = (tr_df["_sample_w"] /
                          (tr_df["_sample_w"].sum() / len(tr_df)))

    feat_cols = [f for f in FEATURE_NAMES if f in all_df.columns]
    missing   = [f for f in FEATURE_NAMES if f not in all_df.columns]
    if missing:
        print(f"  WARNING: missing feature cols: {missing}")

    def _arrays(df):
        X = df[feat_cols].fillna(0).values.astype(np.float32)
        y = df[rc].fillna(0).values.astype(np.float32)
        w = df["_sample_w"].values.astype(np.float32)
        return X, y, w

    X_tr,  y_tr,  w_tr  = _arrays(tr_df)
    X_val, y_val, _     = _arrays(val_df)
    X_te,  y_te,  _     = _arrays(test_df)

    # Fit StandardScaler on training features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_tr_n  = scaler.fit_transform(X_tr).astype(np.float32)
    X_val_n = scaler.transform(X_val).astype(np.float32)
    X_te_n  = scaler.transform(X_te).astype(np.float32)

    with open(model_dir / "feature_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    models_trained = []

    if "mlp" in args.model:
        print(f"\nTraining MLP ({args.mlp_epochs} epochs) ...")
        mlp = train_mlp(X_tr_n, y_tr, X_val_n, y_val,
                        w_tr, args.mlp_epochs, device, args.seed)
        torch.save(mlp.state_dict(), model_dir / "mlp_weights.pt")
        mlp_preds_te = mlp(torch.from_numpy(X_te_n).float().to(device)
                           ).detach().cpu().numpy()
        mlp_mae = float(np.abs(mlp_preds_te - y_te).mean())
        print(f"  MLP test MAE = {mlp_mae:.4f}")
        models_trained.append("mlp")

    if "gbm" in args.model:
        print(f"\nTraining GBM ({args.gbm_n_estimators} estimators) ...")
        gbm = train_gbm(X_tr_n, y_tr, X_val_n, y_val,
                        w_tr, args.gbm_n_estimators, args.seed)
        with open(model_dir / "gbm_model.pkl", "wb") as f:
            pickle.dump(gbm, f)
        gbm_preds_te = gbm.predict(X_te_n).astype(np.float32)
        gbm_mae = float(np.abs(gbm_preds_te - y_te).mean())
        print(f"  GBM test MAE = {gbm_mae:.4f}")
        models_trained.append("gbm")

    schema = {
        "artifact_version": ARTIFACT_VERSION,
        "reward_col":       rc,
        "trajectory_types": args.trajectory_types,
        "source_weights":   SOURCE_WEIGHTS,
        "feature_names":    feat_cols,
        "n_features":       len(feat_cols),
        "n_train":          len(X_tr),
        "n_val":            len(X_val),
        "n_test":           len(X_te),
        "models_trained":   models_trained,
        "mlp_epochs":       args.mlp_epochs,
        "gbm_n_estimators": args.gbm_n_estimators,
    }
    atomic_write_json(sentinel, schema)
    print(f"\nSaved: {sentinel}")
    print(f"Models trained: {models_trained}")


if __name__ == "__main__":
    main()

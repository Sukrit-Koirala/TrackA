"""
predict_future_object_utility.py  --  MVP 4c-0 Stage 7

Trains linear / GBM / MLP models to predict future object utility
from features available at write time.

Targets (each trained separately, saved in utility_prediction/{target}/):
  G_raw                          -- raw total gain (default from compute_object_utility)
  shrunk_mean_lambda10           -- shrinkage-regularized per-retrieval utility
  shrunk_opportunity_lambda10    -- opportunity-normalized shrunk utility
  unique_gain_after_replacement  -- LOO counterfactual gain

Write-time features:
  support_after, entropy_before/after/delta, purity_before/after/delta,
  p_y_true_before/after, proto_cosine_sim, n_update_buf, n_update_sta,
  is_promoted, creation_step_norm

Outputs (per target in {output}/utility_prediction/{target}/):
  predictor_linear.pkl
  predictor_gbm.pkl
  predictor_mlp.pt
  predictor_results.json

Summary: {output}/utility_prediction/multi_target_summary.json

Answers:
  Q7:  Which utility metric is most predictable from WRITE-time features?

Usage:
  python src/predict_future_object_utility.py \\
    --output outputs_mvp4c0_delayed_credit_audit_fast \\
    --trajectories oracle bandit \\
    --targets G_raw shrunk_mean_lambda10 shrunk_opportunity_lambda10 \\
    --test_frac 0.2 \\
    --seed 42 \\
    --force
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

import torch
import torch.nn as nn


WRITE_TIME_FEATURES = [
    "support_after",
    "entropy_before", "entropy_after", "entropy_delta",
    "purity_before",  "purity_after",  "purity_delta",
    "p_y_true_before", "p_y_true_after",
    "proto_cosine_sim",
    "n_update_buf", "n_update_sta",
    "is_promoted",
    "creation_step_norm",
]

# Normalized targets live in normalized_object_utility.parquet
NORM_TARGETS = {
    "shrunk_mean_lambda5",  "shrunk_mean_lambda10",  "shrunk_mean_lambda20",
    "shrunk_opportunity_lambda5", "shrunk_opportunity_lambda10", "shrunk_opportunity_lambda20",
    "unique_gain_after_replacement", "mean_gain_per_retrieval",
    "utility_per_support", "utility_per_write_event", "mean_gain_per_opportunity",
}


class MLP(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64),  nn.LayerNorm(64),  nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_features(objects_df: pd.DataFrame, events_df: pd.DataFrame,
                   max_step: float) -> pd.DataFrame:
    create_evs = events_df[events_df["action_type"].str.upper().isin(
        ["CREATE", "CREATE_BUFFER"])].copy()
    update_buf  = events_df[events_df["action_type"].str.upper() == "UPDATE_BUFFER"]
    update_sta  = events_df[events_df["action_type"].str.upper() == "UPDATE_STATE"]
    n_buf = update_buf.groupby("object_id").size().rename("n_update_buf")
    n_sta = update_sta.groupby("object_id").size().rename("n_update_sta")

    feat_rows = []
    for obj_id, grp in create_evs.groupby("object_id"):
        ev      = grp.sort_values("stream_step").iloc[0]
        obj_row = objects_df[objects_df["object_id"] == obj_id]
        if obj_row.empty:
            continue
        obj_row = obj_row.iloc[0]
        is_promoted = (bool(obj_row.get("is_persistent", False)) or
                       float(obj_row.get("promotion_step", -1)) >= 0)
        feat_rows.append({
            "object_id":          int(obj_id),
            "support_after":      float(ev.get("support_after", 0)),
            "entropy_before":     float(ev.get("entropy_before", 0)),
            "entropy_after":      float(ev.get("entropy_after", 0)),
            "entropy_delta":      float(ev.get("entropy_before", 0)) - float(ev.get("entropy_after", 0)),
            "purity_before":      float(ev.get("purity_before", 0)),
            "purity_after":       float(ev.get("purity_after", 0)),
            "purity_delta":       float(ev.get("purity_after", 0)) - float(ev.get("purity_before", 0)),
            "p_y_true_before":    float(ev.get("p_y_true_before", 0)),
            "p_y_true_after":     float(ev.get("p_y_true_after", 0)),
            "proto_cosine_sim":   float(ev.get("proto_cosine_sim", 0)),
            "n_update_buf":       float(n_buf.get(obj_id, 0)),
            "n_update_sta":       float(n_sta.get(obj_id, 0)),
            "is_promoted":        float(is_promoted),
            "creation_step_norm": float(ev.get("stream_step", 0)) / max(max_step, 1),
        })
    return pd.DataFrame(feat_rows)


def _spearman(y_true, y_pred):
    try:
        return float(spearmanr(y_true, y_pred).statistic)
    except Exception:
        return float("nan")


def _top_quartile_auroc(y_true, y_pred, pct=75.0):
    """AUROC for predicting membership in the top-pct percentile."""
    thr   = np.percentile(y_true, pct)
    label = (y_true >= thr).astype(int)
    if label.sum() == 0 or label.sum() == len(label):
        return float("nan")
    try:
        return float(roc_auc_score(label, y_pred))
    except Exception:
        return float("nan")


def evaluate(y_true, y_pred, label):
    rho  = _spearman(y_true, y_pred)
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    auroc_q75 = _top_quartile_auroc(y_true, y_pred, 75.0)
    auroc_q90 = _top_quartile_auroc(y_true, y_pred, 90.0)
    print(f"  [{label}]  spearman={rho:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}  "
          f"AUROC@Q75={auroc_q75:.3f}  AUROC@Q90={auroc_q90:.3f}  n={len(y_true)}")
    return {
        "spearman": rho, "mae": mae, "rmse": rmse,
        "auroc_q75": auroc_q75, "auroc_q90": auroc_q90,
        "n": len(y_true),
    }


def train_for_target(
    df: pd.DataFrame,
    avail_feats: list,
    target: str,
    target_dir: Path,
    seed: int,
    test_frac: float,
    mlp_epochs: int,
    force: bool,
) -> dict:
    """Train linear/GBM/MLP for one target. Returns results dict."""
    sentinel = target_dir / "predictor_results.json"
    if sentinel.exists() and not force:
        print(f"  [cached] {target}")
        with open(sentinel) as f:
            return json.load(f)

    if target not in df.columns:
        print(f"  WARNING: target '{target}' not in dataframe, skipping")
        return {}

    X = df[avail_feats].fillna(0.0).astype(float).values
    y = df[target].fillna(0.0).astype(float).values

    rng   = np.random.default_rng(seed)
    order = (np.argsort(df["creation_step_norm"].values)
             if "creation_step_norm" in df.columns
             else rng.permutation(len(df)))

    n_test    = max(1, int(len(df) * test_frac))
    n_train   = len(df) - n_test
    train_idx = order[:n_train]
    test_idx  = order[n_train:]
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_te, y_te = X[test_idx],  y[test_idx]

    results = {}

    # Linear
    lin = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    lin.fit(X_tr, y_tr)
    results["linear"] = evaluate(y_te, lin.predict(X_te), f"{target}/linear")
    with open(target_dir / "predictor_linear.pkl", "wb") as f:
        pickle.dump(lin, f)

    # Feature importance (Ridge)
    scaler = lin.named_steps["scaler"]
    coef   = lin.named_steps["ridge"].coef_ * scaler.scale_
    fimp   = pd.DataFrame({
        "feature": avail_feats,
        "abs_coef": np.abs(coef),
        "coef": coef,
    }).sort_values("abs_coef", ascending=False)
    fimp.to_csv(target_dir / "feature_importance.csv", index=False)

    # GBM
    if HAS_LGB:
        gbm = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            random_state=seed, n_jobs=4, verbosity=-1)
        gbm.fit(X_tr, y_tr,
                eval_set=[(X_te, y_te)],
                callbacks=[lgb.early_stopping(30, verbose=False),
                           lgb.log_evaluation(-1)])
        results["gbm"] = evaluate(y_te, gbm.predict(X_te), f"{target}/gbm")
        with open(target_dir / "predictor_gbm.pkl", "wb") as f:
            pickle.dump(gbm, f)
    else:
        print(f"  [GBM] lightgbm not installed, skipping")

    # MLP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr_t  = torch.tensor(X_tr, dtype=torch.float32).to(device)
    ytr_t  = torch.tensor(y_tr, dtype=torch.float32).to(device)
    Xte_t  = torch.tensor(X_te, dtype=torch.float32).to(device)

    y_mu, y_std = float(ytr_t.mean()), float(ytr_t.std()) + 1e-8
    x_mu, x_std = Xtr_t.mean(0), Xtr_t.std(0) + 1e-8
    Xtr_n = (Xtr_t - x_mu) / x_std
    Xte_n = (Xte_t - x_mu) / x_std
    ytr_n = (ytr_t - y_mu) / y_std

    mlp   = MLP(len(avail_feats)).to(device)
    opt   = torch.optim.Adam(mlp.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=mlp_epochs)
    mlp.train()
    for _ in range(mlp_epochs):
        opt.zero_grad()
        nn.functional.mse_loss(mlp(Xtr_n), ytr_n).backward()
        opt.step()
        sched.step()

    mlp.eval()
    with torch.no_grad():
        y_pred_mlp = mlp(Xte_n).cpu().numpy() * y_std + y_mu
    results["mlp"] = evaluate(y_te, y_pred_mlp, f"{target}/mlp")
    torch.save({
        "state_dict": mlp.state_dict(),
        "x_mu": x_mu.cpu().numpy(), "x_std": x_std.cpu().numpy(),
        "y_mu": y_mu, "y_std": y_std, "features": avail_feats,
    }, target_dir / "predictor_mlp.pt")

    best_rho = max(v.get("spearman", -99) for v in results.values() if isinstance(v, dict))
    results["best_spearman"] = float(best_rho)
    results["target"] = target
    results["n_train"] = int(n_train)
    results["n_test"]  = int(n_test)

    with open(sentinel, "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output",       required=True)
    ap.add_argument("--trajectories", nargs="+", default=["oracle", "bandit"])
    ap.add_argument("--targets",      nargs="+",
                    default=["G_raw", "shrunk_mean_lambda10",
                             "shrunk_opportunity_lambda10", "unique_gain_after_replacement"],
                    help="Utility metrics to train predictors for (each gets its own subdir)")
    # legacy single-target support
    ap.add_argument("--target",       default=None,
                    help="If set, overrides --targets to single target (backwards-compat)")
    ap.add_argument("--test_frac",    type=float, default=0.2)
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--mlp_epochs",   type=int,   default=80)
    ap.add_argument("--force",        action="store_true")
    args = ap.parse_args()

    if args.target is not None:
        targets = [args.target]
    else:
        targets = args.targets

    out_dir  = Path(args.output)
    pred_dir = out_dir / "predictors"         # legacy single-target location
    upred_dir = out_dir / "utility_prediction" # multi-target location
    pred_dir.mkdir(parents=True, exist_ok=True)
    upred_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # ── Load features ──────────────────────────────────────────────────────────
    all_feat = []
    norm_dfs = []
    for traj in args.trajectories:
        ev_path   = out_dir / "provenance"     / f"{traj}_events.parquet"
        obj_path  = out_dir / "provenance"     / f"{traj}_objects.parquet"
        ret_path  = out_dir / "object_utility" / f"{traj}_object_returns.parquet"
        norm_path = out_dir / "object_utility" / "normalized_object_utility.parquet"

        if not all(p.exists() for p in [ev_path, obj_path, ret_path]):
            print(f"  WARNING: skipping {traj} (missing provenance/returns files)")
            continue

        events_df  = pd.read_parquet(ev_path)
        objects_df = pd.read_parquet(obj_path)
        returns_df = pd.read_parquet(ret_path)
        max_step   = float(events_df["stream_step"].max()) if len(events_df) > 0 else 1.0
        feat_df    = build_features(objects_df, events_df, max_step)

        # Merge raw G_raw
        merged = feat_df.merge(
            returns_df[["object_id", "G_raw"]].rename(columns={"G_raw": "G_raw"}),
            on="object_id", how="inner")

        # Merge normalized targets if available
        if norm_path.exists():
            norm_df = pd.read_parquet(norm_path)
            norm_traj = norm_df[norm_df["trajectory"] == traj] if "trajectory" in norm_df.columns else norm_df
            norm_cols = [c for c in norm_traj.columns
                         if c in NORM_TARGETS and c not in merged.columns]
            if norm_cols:
                merged = merged.merge(
                    norm_traj[["object_id"] + norm_cols], on="object_id", how="left")

        merged["trajectory"] = traj
        all_feat.append(merged)
        print(f"[{traj}] {len(merged):,} objects with features")

    if not all_feat:
        print("ERROR: no data available")
        return

    df          = pd.concat(all_feat, ignore_index=True)
    avail_feats = [c for c in WRITE_TIME_FEATURES if c in df.columns]
    print(f"\nFeatures ({len(avail_feats)}): {avail_feats}")
    print(f"Targets:  {targets}")

    # ── Train per target ───────────────────────────────────────────────────────
    all_results = {}
    for tgt in targets:
        safe_name = tgt.replace("/", "_").replace(" ", "_")
        tgt_dir   = upred_dir / safe_name
        tgt_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- Target: {tgt} ---")
        res = train_for_target(df, avail_feats, tgt, tgt_dir,
                               args.seed, args.test_frac, args.mlp_epochs, args.force)
        if res:
            all_results[tgt] = res

    # ── Multi-target summary ───────────────────────────────────────────────────
    if all_results:
        print("\n--- Multi-target summary ---")
        rows = []
        best_tgt, best_rho = None, -99.0
        for tgt, res in all_results.items():
            rho = res.get("best_spearman", -99)
            if isinstance(rho, float) and rho > best_rho:
                best_rho = rho
                best_tgt = tgt
            gbm_auroc = res.get("gbm", {}).get("auroc_q75", float("nan"))
            lin_rho   = res.get("linear", {}).get("spearman", float("nan"))
            gbm_rho   = res.get("gbm", {}).get("spearman", float("nan"))
            mlp_rho   = res.get("mlp", {}).get("spearman", float("nan"))
            rows.append({
                "target":         tgt,
                "best_spearman":  rho,
                "linear_rho":     lin_rho,
                "gbm_rho":        gbm_rho,
                "mlp_rho":        mlp_rho,
                "gbm_auroc_q75":  gbm_auroc,
            })
            print(f"  {tgt:<45}  best_rho={rho:.4f}  AUROC@Q75={gbm_auroc:.3f}")

        summary_df = pd.DataFrame(rows).sort_values("best_spearman", ascending=False)
        summary_df.to_csv(upred_dir / "target_comparison.csv", index=False)

        q7_answer = ("STRONG"   if best_rho > 0.50 else
                     "MODERATE" if best_rho > 0.25 else
                     "WEAK"     if best_rho > 0.10 else "NONE")
        print(f"\nQ7: Best target for prediction = '{best_tgt}'  "
              f"(rho={best_rho:.4f}) → {q7_answer}")

        multi_summary = {
            "targets":          targets,
            "best_target":      best_tgt,
            "best_spearman":    float(best_rho),
            "q7_predictability": q7_answer,
            "per_target":       all_results,
        }
        with open(upred_dir / "multi_target_summary.json", "w") as f:
            json.dump(multi_summary, f, indent=2, default=str)

        # ── Backwards-compat: write legacy predictor_results.json ─────────────
        if "G_raw" in all_results:
            compat = all_results["G_raw"].copy()
            compat["q7_predictability"] = q7_answer
            compat["best_spearman"]     = float(best_rho)
        else:
            compat = {
                "q7_predictability": q7_answer,
                "best_spearman":     float(best_rho),
                "target":            best_tgt,
            }
        with open(pred_dir / "predictor_results.json", "w") as f:
            json.dump(compat, f, indent=2, default=str)
        print(f"\nSaved: {upred_dir / 'multi_target_summary.json'}")


if __name__ == "__main__":
    main()

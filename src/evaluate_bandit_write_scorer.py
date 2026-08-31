"""
evaluate_bandit_write_scorer.py  --  MVP 4b Stage 2b

Evaluates trained action-value scorers on the held-out test set.

Metrics:
  - Regression: MAE, RMSE, Spearman rho
  - Regret: fraction of decision points where predicted best != true best
  - Regret gap: mean reward lost by following predicted best vs oracle best
  - Baseline comparison: random, always-defer, always-update-state

Usage:
  python src/evaluate_bandit_write_scorer.py \\
    --output outputs_mvp4b_bandit_write_fast \\
    --reward_col reward_penalized \\
    --model mlp gbm
"""

import json
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

from train_bandit_write_scorer import ActionValueMLP, FEATURE_NAMES

ACT_NAMES = ["UPDATE_STATE", "UPDATE_BUFFER", "PROMOTE_AND_UPDATE",
             "CREATE_BUFFER", "DEFER"]


def _load_mlp(model_dir: Path, device: torch.device) -> ActionValueMLP | None:
    p = model_dir / "mlp.pt"
    if not p.exists():
        return None
    ckpt   = torch.load(p, map_location=device, weights_only=False)
    model  = ActionValueMLP(ckpt["in_dim"], ckpt["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model


def _load_gbm(model_dir: Path):
    p = model_dir / "gbm.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def _load_scaler(model_dir: Path):
    p = model_dir / "feature_scaler.pkl"
    with open(p, "rb") as f:
        return pickle.load(f)


def _regret_analysis(
    df_te: pd.DataFrame,
    y_pred: np.ndarray,
    rcol: str,
    model_name: str,
) -> dict:
    """Per-decision-point regret: how much reward is lost vs oracle best."""
    df_te = df_te.copy()
    df_te["_pred"] = y_pred

    dp_stats = []
    for step_val, grp in df_te.groupby("step"):
        oracle_best = grp[rcol].max()
        pred_best_idx = grp["_pred"].idxmax()
        pred_best_reward = grp.loc[pred_best_idx, rcol]
        regret = oracle_best - pred_best_reward
        pred_act = grp.loc[pred_best_idx, "action_type"]
        oracle_act_idx = grp[rcol].idxmax()
        oracle_act = grp.loc[oracle_act_idx, "action_type"]
        dp_stats.append({
            "step":            step_val,
            "oracle_reward":   oracle_best,
            "pred_reward":     pred_best_reward,
            "regret":          regret,
            "match":           int(pred_act == oracle_act),
            "pred_action":     int(pred_act),
            "oracle_action":   int(oracle_act),
        })

    dp_df = pd.DataFrame(dp_stats)
    frac_match   = float(dp_df["match"].mean())
    mean_regret  = float(dp_df["regret"].mean())
    frac_zero_regret = float((dp_df["regret"] <= 1e-6).mean())

    print(f"  {model_name}:")
    print(f"    Action match:    {frac_match*100:.1f}%")
    print(f"    Mean regret:     {mean_regret:.5f}")
    print(f"    Zero-regret DPs: {frac_zero_regret*100:.1f}%")

    return {
        "action_match_frac":    frac_match,
        "mean_regret":          mean_regret,
        "frac_zero_regret":     frac_zero_regret,
        "p50_regret":           float(dp_df["regret"].quantile(0.50)),
        "p90_regret":           float(dp_df["regret"].quantile(0.90)),
        "action_confusion":     {f"{k[0]},{k[1]}": int(v)
                                 for k, v in dp_df.groupby(
                                     ["oracle_action", "pred_action"])
                                     .size().to_dict().items()},
    }


def _baseline_regret(df_te: pd.DataFrame, rcol: str, strategy: str) -> dict:
    """Compute regret for a fixed policy baseline."""
    ACT_DEFER = 4
    ACT_UPDATE_STATE = 0
    dp_stats = []
    for step_val, grp in df_te.groupby("step"):
        oracle_best = grp[rcol].max()
        if strategy == "random":
            chosen_idx = grp.sample(1).index[0]
        elif strategy == "always_defer":
            candidates = grp[grp["action_type"] == ACT_DEFER]
            chosen_idx = candidates.index[0] if len(candidates) > 0 else grp.index[0]
        elif strategy == "always_update_state":
            candidates = grp[grp["action_type"] == ACT_UPDATE_STATE]
            chosen_idx = candidates.index[0] if len(candidates) > 0 else grp.index[0]
        else:
            chosen_idx = grp.index[0]
        chosen_reward = grp.loc[chosen_idx, rcol]
        dp_stats.append(oracle_best - chosen_reward)
    arr = np.array(dp_stats)
    return {
        "mean_regret": float(arr.mean()),
        "p50_regret":  float(np.percentile(arr, 50)),
        "p90_regret":  float(np.percentile(arr, 90)),
    }


def evaluate_model(
    name: str, pred: np.ndarray, y: np.ndarray,
    df_te: pd.DataFrame, rcol: str,
) -> dict:
    mae  = float(mean_absolute_error(y, pred))
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    rho, p_rho = stats.spearmanr(y, pred)
    r_rho = stats.pearsonr(y, pred)

    print(f"\n  {name.upper()}:")
    print(f"    MAE={mae:.5f}  RMSE={rmse:.5f}  "
          f"Spearman={float(rho):.4f}  Pearson={float(r_rho[0]):.4f}")

    reg = _regret_analysis(df_te, pred, rcol, name)
    return {
        "mae": mae, "rmse": rmse,
        "spearman_rho": float(rho), "spearman_p": float(p_rho),
        "pearson_r": float(r_rho[0]),
        "regret": reg,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output",     required=True)
    ap.add_argument("--reward_col", default="reward_penalized")
    ap.add_argument("--model",      nargs="+", default=["mlp", "gbm"],
                    choices=["mlp", "gbm"])
    ap.add_argument("--device",     default="cuda")
    ap.add_argument("--force",      action="store_true")
    args = ap.parse_args()

    out_dir   = Path(args.output)
    data_dir  = out_dir / "reward_dataset"
    model_dir = out_dir / "scorer_models"
    eval_dir  = out_dir / "scorer_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    sentinel = eval_dir / "scorer_eval.json"
    if sentinel.exists() and not args.force:
        print(f"[cached] {sentinel}")
        return

    device = torch.device("cuda" if torch.cuda.is_available()
                          and args.device == "cuda" else "cpu")

    # ── load schema for test split ────────────────────────────────────────────
    schema_path = model_dir / "training_schema.json"
    schema = json.load(open(schema_path))
    test_lo     = schema["step_split"]["test_lo"]
    feat_cols   = schema["feature_names"]
    rcol        = args.reward_col

    # ── load data ─────────────────────────────────────────────────────────────
    print("Loading reward dataset (test split) ...")
    df = pd.read_parquet(data_dir / "actions.parquet")
    df_te = df[df["step"] >= test_lo].copy().reset_index(drop=True)
    print(f"  Test records: {len(df_te):,}  "
          f"Decision points: {df_te['step'].nunique():,}")

    if rcol not in df_te.columns:
        rcol = "reward_full"
    y_te = df_te[rcol].values.astype(np.float32)

    feat_cols_avail = [f for f in feat_cols if f in df_te.columns]
    X_te = df_te[feat_cols_avail].values.astype(np.float32)

    # ── normalize ─────────────────────────────────────────────────────────────
    scaler  = _load_scaler(model_dir)
    X_te_n  = scaler.transform(X_te).astype(np.float32)

    # ── evaluate models ───────────────────────────────────────────────────────
    results = {}
    print(section := f"\n{'='*60}\nModel evaluation on test set\n{'='*60}")

    if "mlp" in args.model:
        mlp = _load_mlp(model_dir, device)
        if mlp is not None:
            with torch.no_grad():
                pred = mlp(torch.from_numpy(X_te_n).to(device)).cpu().numpy()
            results["mlp"] = evaluate_model("MLP", pred, y_te, df_te, rcol)
        else:
            print("  MLP model not found, skipping.")

    if "gbm" in args.model:
        gbm = _load_gbm(model_dir)
        if gbm is not None:
            pred = gbm.predict(X_te_n).astype(np.float32)
            results["gbm"] = evaluate_model("GBM", pred, y_te, df_te, rcol)
        else:
            print("  GBM model not found, skipping.")

    # ── baselines ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}\nBaseline regret\n{'='*60}")
    baselines = {}
    for strat in ["random", "always_defer", "always_update_state"]:
        b = _baseline_regret(df_te, rcol, strat)
        baselines[strat] = b
        print(f"  {strat:<24}  mean_regret={b['mean_regret']:.5f}  "
              f"p90_regret={b['p90_regret']:.5f}")

    # ── go/no-go for Stage 2 ──────────────────────────────────────────────────
    print(f"\n{'='*60}\nStage 2 go/no-go\n{'='*60}")
    best_model = max(results.items(),
                     key=lambda x: x[1]["spearman_rho"]) if results else None
    gates = {}
    if best_model:
        name, m = best_model
        rng = baselines.get("random", {}).get("mean_regret", np.inf)
        gates = {
            "spearman_gt_0.1":      m["spearman_rho"] > 0.10,
            "beats_random_regret":  m["regret"]["mean_regret"] < rng * 0.9,
            "action_match_gt_30pct": m["regret"]["action_match_frac"] > 0.30,
        }
        for g, v in gates.items():
            print(f"  {'PASS' if v else 'FAIL'}  {g}")
        verdict = "GO" if all(gates.values()) else "WARN"
        print(f"\n  ==> Stage 2 verdict: {verdict} (best model: {name})")
    else:
        verdict = "NO-GO"
        print("  No models evaluated.")

    # ── save ──────────────────────────────────────────────────────────────────
    report = {
        "reward_col":   rcol,
        "n_test":       len(df_te),
        "n_dp_test":    int(df_te["step"].nunique()),
        "models":       results,
        "baselines":    baselines,
        "gates":        gates,
        "verdict":      verdict,
    }
    with open(sentinel, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved: {sentinel}")

    # Per-decision-point regret CSV for the best model
    if best_model:
        name, _ = best_model
        if name == "mlp":
            mlp = _load_mlp(model_dir, device)
            if mlp is not None:
                with torch.no_grad():
                    pred = mlp(torch.from_numpy(X_te_n).to(device)).cpu().numpy()
        else:
            pred = _load_gbm(model_dir).predict(X_te_n)

        dp_rows = []
        df_te["_pred"] = pred
        for step_val, grp in df_te.groupby("step"):
            oracle_best   = grp[rcol].max()
            pred_best_idx = grp["_pred"].idxmax()
            pred_reward   = grp.loc[pred_best_idx, rcol]
            dp_rows.append({
                "step":          step_val,
                "oracle_reward": oracle_best,
                "pred_reward":   pred_reward,
                "regret":        oracle_best - pred_reward,
                "pred_action":   int(grp.loc[pred_best_idx, "action_type"]),
            })
        dp_df = pd.DataFrame(dp_rows)
        dp_df.to_parquet(eval_dir / "per_dp_regret.parquet", index=False)
        print(f"Saved: {eval_dir}/per_dp_regret.parquet")


if __name__ == "__main__":
    main()

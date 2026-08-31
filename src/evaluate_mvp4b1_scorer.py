"""
evaluate_mvp4b1_scorer.py  --  MVP 4b.1 Stage 4

Evaluates scorer by trajectory source (oracle / low_memory / onpolicy)
and reports regret metrics.

Usage:
  python src/evaluate_mvp4b1_scorer.py \\
    --output outputs_mvp4b1_onpolicy_bandit_write_clean_fast \\
    --reward_col reward_penalized \\
    --trajectory_types oracle low_memory \\
    --model mlp gbm \\
    --device cuda --force
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from mvp4b1_common import (
    ARTIFACT_VERSION, FEATURE_NAMES, N_FEAT, ACT_NAMES, N_ACT_TYPES,
    atomic_write_json, validate_json_cache,
)
from train_mvp4b1_scorer import BanditMLP


def _load_model_mlp(model_dir: Path, device: torch.device):
    weights = torch.load(model_dir / "mlp_weights.pt",
                         map_location=device, weights_only=True)
    model = BanditMLP(N_FEAT).to(device)
    model.load_state_dict(weights)
    model.eval()
    return model


def _load_scaler(model_dir: Path):
    with open(model_dir / "feature_scaler.pkl", "rb") as f:
        return pickle.load(f)


def _load_gbm(model_dir: Path):
    with open(model_dir / "gbm_model.pkl", "rb") as f:
        return pickle.load(f)


def _predict(model, model_name: str, X_n: np.ndarray, device: torch.device) -> np.ndarray:
    if model_name == "mlp":
        with torch.no_grad():
            return model(torch.from_numpy(X_n).float().to(device)
                         ).cpu().numpy().astype(np.float32)
    else:
        return model.predict(X_n).astype(np.float32)


def _regret_metrics(df: pd.DataFrame, pred_col: str, rc: str) -> dict:
    """Per-decision-point regret: oracle_best - predicted_best."""
    regrets, matches = [], []
    for step, grp in df.groupby("step"):
        g   = grp[rc].values
        p   = grp[pred_col].values
        if len(g) == 0:
            continue
        oracle_best  = float(g.max())
        oracle_act   = int(grp["action_type"].values[np.argmax(g)])
        pred_act     = int(grp["action_type"].values[np.argmax(p)])
        pred_val     = float(g[np.argmax(p)])   # true reward at predicted action
        regrets.append(oracle_best - pred_val)
        matches.append(int(oracle_act == pred_act))
    if not regrets:
        return {}
    return {
        "mean_regret":       float(np.mean(regrets)),
        "median_regret":     float(np.median(regrets)),
        "p90_regret":        float(np.percentile(regrets, 90)),
        "action_match_frac": float(np.mean(matches)),
        "n_dp":              len(regrets),
    }


def _eval_one(df: pd.DataFrame, model, model_name: str, scaler,
              rc: str, device: torch.device, source_tag: str) -> dict:
    feat_cols = [f for f in FEATURE_NAMES if f in df.columns]
    X    = df[feat_cols].fillna(0).values.astype(np.float32)
    y    = df[rc].fillna(0).values.astype(np.float32)
    X_n  = scaler.transform(X).astype(np.float32)
    pred = _predict(model, model_name, X_n, device)

    df   = df.copy()
    df["_pred"] = pred

    # Correlation metrics
    from scipy.stats import spearmanr
    rho = float(spearmanr(y, pred).correlation)
    mae = float(np.abs(y - pred).mean())

    # Regret
    regret = _regret_metrics(df, "_pred", rc)

    # Per-action MAE
    per_action_mae = {}
    for act, name in enumerate(ACT_NAMES):
        mask = df["action_type"].values == act
        if mask.sum() == 0:
            continue
        per_action_mae[name] = float(np.abs(y[mask] - pred[mask]).mean())

    return {
        "source_tag":      source_tag,
        "n_records":       len(df),
        "spearman_rho":    rho,
        "mae":             mae,
        "regret":          regret,
        "per_action_mae":  per_action_mae,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output",            required=True)
    ap.add_argument("--reward_col",        default="reward_penalized")
    ap.add_argument("--trajectory_types",  nargs="+",
                    default=["oracle", "low_memory"])
    ap.add_argument("--model",             nargs="+",
                    default=["mlp", "gbm"],
                    choices=["mlp", "gbm"])
    ap.add_argument("--device",            default="cuda")
    ap.add_argument("--force",             action="store_true")
    args = ap.parse_args()

    device    = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir   = Path(args.output)
    model_dir = out_dir / "scorer_models"
    eval_dir  = out_dir / "scorer_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    sentinel  = eval_dir / "scorer_eval.json"

    cached = validate_json_cache(sentinel, ["artifact_version", "models"])
    if cached and not args.force:
        print(f"[cached] {sentinel}")
        return

    rc     = args.reward_col
    scaler = _load_scaler(model_dir)

    # Load datasets
    frames_by_source = {}
    for ttype in args.trajectory_types:
        rd = out_dir / "reward_datasets" / ttype / "actions.parquet"
        if rd.exists():
            df = pd.read_parquet(rd)
            df["_source"] = ttype
            frames_by_source[ttype] = df
            print(f"  Loaded {ttype}: {len(df):,} records")

    # Also load onpolicy if present
    op_rd = out_dir / "reward_datasets" / "onpolicy" / "actions.parquet"
    if op_rd.exists():
        df = pd.read_parquet(op_rd)
        df["_source"] = "onpolicy"
        frames_by_source["onpolicy"] = df
        print(f"  Loaded onpolicy: {len(df):,} records")

    if not frames_by_source:
        raise RuntimeError("No reward datasets found!")

    all_df = pd.concat(list(frames_by_source.values()), ignore_index=True)
    sources = list(frames_by_source.keys())

    models_results = {}
    for mname in args.model:
        print(f"\nEvaluating {mname.upper()} ...")
        if mname == "mlp":
            model = _load_model_mlp(model_dir, device)
        else:
            model = _load_gbm(model_dir)

        # Combined eval
        combined = _eval_one(all_df, model, mname, scaler, rc, device, "combined")
        per_source_res = {}
        for src, df in frames_by_source.items():
            per_source_res[src] = _eval_one(
                df, model, mname, scaler, rc, device, src)
            print(f"  {src}: rho={per_source_res[src]['spearman_rho']:.3f}  "
                  f"MAE={per_source_res[src]['mae']:.4f}  "
                  f"regret={per_source_res[src].get('regret',{}).get('mean_regret','?'):.4f}")

        print(f"  combined: rho={combined['spearman_rho']:.3f}  "
              f"MAE={combined['mae']:.4f}  "
              f"regret={combined.get('regret',{}).get('mean_regret','?'):.4f}")

        models_results[mname] = {
            "combined":  combined,
            "per_source": per_source_res,
            # top-level shortcuts for compatibility with run_mvp4b_bandit_write analyze
            "spearman_rho": combined["spearman_rho"],
            "mae":          combined["mae"],
            "regret":       combined.get("regret", {}),
        }

    report = {
        "artifact_version": ARTIFACT_VERSION,
        "reward_col":       rc,
        "trajectory_types": sources,
        "models":           models_results,
    }
    atomic_write_json(sentinel, report)
    print(f"\nSaved: {sentinel}")


if __name__ == "__main__":
    main()

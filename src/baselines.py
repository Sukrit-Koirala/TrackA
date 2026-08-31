"""
baselines.py

Evaluate every fixed (k, tau, alpha) action on controller_train and val.

Selection rule (no leakage):
  - Choose best fixed action by controller_train NLL.
  - Report its val NLL as the "best fixed kNN" result.
  - Also report val-oracle best-fixed-action as a diagnostic (labelled).

Outputs:
  outputs/reports/fixed_baselines.csv
  outputs/reports/fixed_baselines.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json
import math
import pandas as pd
import torch

from utils import load_config, ensure_dirs, get_device, set_seed, build_action_grid, ppl
from gate_chain import evaluate_action_for_queries


def evaluate_all_actions(
    actions: list[dict],
    query_data: dict,
    neighbor_data: dict,
    cfg: dict,
) -> list[dict]:
    results = []
    for action in actions:
        m = evaluate_action_for_queries(
            query_data, neighbor_data, action,
            max_k=cfg["max_k"],
            lambda_cost=cfg.get("lambda_cost", 0.0),
            eps=cfg.get("eps", 1e-12),
        )
        results.append({
            "name":            action["name"],
            "k":               action["k"],
            "tau":             action["tau"],
            "alpha":           action["alpha"],
            "mean_nll":        m["mean_nll"],
            "mean_ppl":        ppl(m["mean_nll"]),
            "mean_reward":     m["mean_reward"],
            "mean_k":          m["mean_k"],
            "retrieval_usage": m["retrieval_usage"],
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    set_seed(cfg.get("seed", 42))

    states_dir = Path(cfg["states_dir"])
    nbrs_dir   = Path(cfg["neighbors_dir"])
    reports_dir = Path(cfg["reports_dir"])

    k = cfg["max_k"]

    print("Loading data …")
    ct_data    = torch.load(states_dir / "controller_train.pt", weights_only=False)
    val_data   = torch.load(states_dir / "val.pt",             weights_only=False)
    ct_nbrs    = torch.load(nbrs_dir   / f"controller_train_top{k}.pt", weights_only=False)
    val_nbrs   = torch.load(nbrs_dir   / f"val_top{k}.pt",              weights_only=False)

    actions = build_action_grid(cfg)
    print(f"Action grid: {len(actions)} actions")

    # ── evaluate on controller_train ──────────────────────────────────────────
    print("\nEvaluating on controller_train …")
    ct_results  = evaluate_all_actions(actions, ct_data, ct_nbrs, cfg)
    ct_df       = pd.DataFrame(ct_results).sort_values("mean_nll")

    # Best action selected on train (no validation leakage)
    best_ct     = ct_df.iloc[0]
    best_name   = best_ct["name"]
    print(f"  Best train-selected action: {best_name}  NLL={best_ct['mean_nll']:.4f}")

    # GPT-only reference
    gpt_ct = next(r for r in ct_results if r["name"] == "gpt_only")
    print(f"  GPT-only train NLL:         {gpt_ct['mean_nll']:.4f}")

    # ── evaluate on val ───────────────────────────────────────────────────────
    print("\nEvaluating on val …")
    val_results = evaluate_all_actions(actions, val_data, val_nbrs, cfg)
    val_df      = pd.DataFrame(val_results).sort_values("mean_nll")

    gpt_val     = next(r for r in val_results if r["name"] == "gpt_only")
    best_val_row = next(r for r in val_results if r["name"] == best_name)

    # Val-oracle: best action as chosen on val (diagnostic only — uses val implicitly)
    oracle_val  = val_df.iloc[0]

    # ── print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FIXED BASELINES SUMMARY")
    print("=" * 70)
    print(f"{'Method':<40} {'NLL':>7}  {'PPL':>8}  {'avg_k':>6}  {'ret%':>6}")
    print("-" * 70)

    def row(label, r):
        print(f"  {label:<38} {r['mean_nll']:>7.4f}  {r['mean_ppl']:>8.2f}"
              f"  {r['mean_k']:>6.1f}  {r['retrieval_usage']*100:>5.0f}%")

    row("GPT-only (val)",          gpt_val)
    row("Best train-selected (val)", best_val_row)
    row("[DIAG] Val-oracle best (val)", dict(oracle_val))
    print("=" * 70)

    # Top-10 actions by val NLL
    print("\nTop-10 actions (val NLL):")
    print(val_df[["name", "k", "tau", "alpha", "mean_nll", "mean_ppl"]].head(10).to_string(index=False))

    # ── save ──────────────────────────────────────────────────────────────────
    # Merge train + val results
    ct_df  = ct_df.rename(columns={"mean_nll": "ct_nll", "mean_ppl": "ct_ppl",
                                    "mean_reward": "ct_reward"})
    val_df2 = pd.DataFrame(val_results).rename(
        columns={"mean_nll": "val_nll", "mean_ppl": "val_ppl",
                 "mean_reward": "val_reward", "retrieval_usage": "val_ret_usage"})
    merged = ct_df.merge(val_df2[["name", "val_nll", "val_ppl", "val_reward", "val_ret_usage"]],
                         on="name")
    merged = merged.sort_values("ct_nll")
    merged.to_csv(reports_dir / "fixed_baselines.csv", index=False)

    summary = {
        "gpt_only_val_nll":             gpt_val["mean_nll"],
        "gpt_only_val_ppl":             gpt_val["mean_ppl"],
        "best_train_selected_name":     best_name,
        "best_train_selected_val_nll":  best_val_row["mean_nll"],
        "best_train_selected_val_ppl":  best_val_row["mean_ppl"],
        "best_train_selected_val_k":    best_val_row["mean_k"],
        "best_train_selected_val_ret":  best_val_row["retrieval_usage"],
        "oracle_val_name":              oracle_val["name"],
        "oracle_val_nll":               oracle_val["mean_nll"],
        "oracle_val_ppl":               oracle_val["mean_ppl"],
        "all_val_results":              val_results,
        "all_ct_results":               ct_results,
    }
    with open(reports_dir / "fixed_baselines.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {reports_dir / 'fixed_baselines.csv'}")
    print(f"Saved: {reports_dir / 'fixed_baselines.json'}")
    print("\nDone.")


if __name__ == "__main__":
    main()

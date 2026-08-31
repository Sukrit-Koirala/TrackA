"""
evaluate.py

Assembles the final comparison table from saved results and re-evaluates the
learned controller on val for a fresh measurement.

Outputs:
  outputs/reports/final_metrics.json
  outputs/reports/final_metrics.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json
import math
import numpy as np
import pandas as pd
import torch

from utils import load_config, ensure_dirs, get_device, set_seed, build_action_grid, ppl
from gate_chain import evaluate_action_for_queries
from train_controller import ControllerMLP, build_features, apply_controller


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    device = get_device(cfg)

    states_dir  = Path(cfg["states_dir"])
    nbrs_dir    = Path(cfg["neighbors_dir"])
    reports_dir = Path(cfg["reports_dir"])
    models_dir  = Path(cfg["models_dir"])
    k           = cfg["max_k"]

    # ── load saved summaries ──────────────────────────────────────────────────
    with open(reports_dir / "fixed_baselines.json") as f:
        baselines = json.load(f)
    with open(reports_dir / "controller_results.json") as f:
        ctrl_res = json.load(f)

    # ── load val data for fresh controller eval ───────────────────────────────
    print("Loading val data …")
    val_data = torch.load(states_dir / "val.pt", weights_only=False)
    val_nbrs = torch.load(nbrs_dir   / f"val_top{k}.pt", weights_only=False)
    actions  = build_action_grid(cfg)

    # ── re-run learned controller on val ─────────────────────────────────────
    print("Re-evaluating learned controller on val …")
    feat_dim = 13
    norm = torch.load(models_dir / "controller_norm.pt", map_location="cpu", weights_only=True)
    num_act = int(norm["num_actions"])
    ctrl_model = ControllerMLP(feat_dim, cfg["controller_hidden"], num_act)
    ctrl_model.register_buffer("feat_mu",  norm["feat_mu"])
    ctrl_model.register_buffer("feat_std", norm["feat_std"])
    ctrl_model.load_state_dict(
        torch.load(models_dir / "controller.pt", map_location="cpu", weights_only=True),
        strict=False,
    )
    ctrl_model = ctrl_model.to(device)

    X_val = build_features(val_data, val_nbrs)
    chosen = apply_controller(ctrl_model, X_val, device)   # [N_val]

    val_nll_ctrl_list = []
    val_k_ctrl_list   = []
    val_ret_list      = []

    for ai, action in enumerate(actions):
        mask = (chosen == ai)
        if mask.sum() == 0:
            continue
        m = evaluate_action_for_queries(
            {k2: v[mask] if k2 != "metadata" else v for k2, v in val_data.items()},
            {k2: v[mask] for k2, v in val_nbrs.items()},
            action,
            max_k=cfg["max_k"], lambda_cost=cfg.get("lambda_cost", 0.0),
            eps=cfg.get("eps", 1e-12),
        )
        val_nll_ctrl_list.extend(m["per_example_nll"].tolist())
        val_k_ctrl_list.extend([action["k"]] * int(mask.sum()))
        val_ret_list.extend([float(action["k"] > 0)] * int(mask.sum()))

    ctrl_nll = float(np.mean(val_nll_ctrl_list))
    ctrl_k   = float(np.mean(val_k_ctrl_list))
    ctrl_ret = float(np.mean(val_ret_list))

    # ── assemble table rows ───────────────────────────────────────────────────
    rows = [
        {
            "method":            "GPT only",
            "val_nll":           baselines["gpt_only_val_nll"],
            "val_ppl":           baselines["gpt_only_val_ppl"],
            "avg_k":             0.0,
            "retrieval_usage":   0.0,
            "note": "",
        },
        {
            "method":            "Best fixed kNN (train-selected)",
            "val_nll":           baselines["best_train_selected_val_nll"],
            "val_ppl":           baselines["best_train_selected_val_ppl"],
            "avg_k":             baselines["best_train_selected_val_k"],
            "retrieval_usage":   baselines["best_train_selected_val_ret"],
            "note":              baselines["best_train_selected_name"],
        },
        {
            "method":            "Entropy heuristic",
            "val_nll":           ctrl_res["entropy_heuristic_val"]["mean_nll"],
            "val_ppl":           ctrl_res["entropy_heuristic_val"]["mean_ppl"],
            "avg_k":             ctrl_res["entropy_heuristic_val"]["mean_k"],
            "retrieval_usage":   ctrl_res["entropy_heuristic_val"]["retrieval_usage"],
            "note":              (f"thresh={ctrl_res['entropy_heuristic_config']['threshold']:.2f} "
                                  f"action={ctrl_res['entropy_heuristic_config']['retrieval_action']}"),
        },
        {
            "method":            "Learned controller",
            "val_nll":           ctrl_nll,
            "val_ppl":           ppl(ctrl_nll),
            "avg_k":             ctrl_k,
            "retrieval_usage":   ctrl_ret,
            "note": "",
        },
        {
            "method":            "[DIAG] Oracle per-example (val)",
            "val_nll":           ctrl_res["oracle_val"]["nll"],
            "val_ppl":           ctrl_res["oracle_val"]["ppl"],
            "avg_k":             float("nan"),
            "retrieval_usage":   float("nan"),
            "note":              "val-oracle — not a fair comparison",
        },
        {
            "method":            "[DIAG] Best fixed kNN (val-oracle)",
            "val_nll":           baselines["oracle_val_nll"],
            "val_ppl":           baselines["oracle_val_ppl"],
            "avg_k":             float("nan"),
            "retrieval_usage":   float("nan"),
            "note":              f"action={baselines['oracle_val_name']}",
        },
    ]

    # ── print table ───────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("FINAL EVALUATION TABLE")
    print("=" * 80)
    hdr = f"{'Method':<40} {'val_nll':>8}  {'val_ppl':>9}  {'avg_k':>6}  {'ret%':>6}"
    print(hdr)
    print("-" * 80)
    for r in rows:
        avg_k = f"{r['avg_k']:>6.1f}" if not math.isnan(r["avg_k"]) else "  —   "
        ret   = f"{r['retrieval_usage']*100:>5.0f}%" if not math.isnan(r["retrieval_usage"]) else "  —   "
        print(f"  {r['method']:<38} {r['val_nll']:>8.4f}  {r['val_ppl']:>9.2f}  {avg_k}  {ret}")
    print("=" * 80)

    # ── success condition check ───────────────────────────────────────────────
    best_fixed_nll = baselines["best_train_selected_val_nll"]
    if ctrl_nll < best_fixed_nll:
        print(f"\n✓ SUCCESS: controller NLL {ctrl_nll:.4f} < best fixed NLL {best_fixed_nll:.4f}")
    elif ctrl_k < baselines["best_train_selected_val_k"] and ctrl_nll <= best_fixed_nll * 1.01:
        print(f"\n✓ SUCCESS (efficiency): controller matches best fixed NLL at lower avg k "
              f"({ctrl_k:.1f} vs {baselines['best_train_selected_val_k']:.1f})")
    else:
        delta = ctrl_nll - best_fixed_nll
        print(f"\n✗ Controller did not beat best fixed (delta NLL = {delta:+.4f})")

    # ── save ──────────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(reports_dir / "final_metrics.csv", index=False)
    with open(reports_dir / "final_metrics.json", "w") as f:
        json.dump(rows, f, indent=2)

    print(f"\nSaved: {reports_dir / 'final_metrics.csv'}")
    print(f"Saved: {reports_dir / 'final_metrics.json'}")
    print("\nDone.")


if __name__ == "__main__":
    main()

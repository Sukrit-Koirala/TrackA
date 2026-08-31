"""
run_mvp2c_q_state_read.py  —  MVP 2c pipeline runner

Stages (all sentinel-cached):
  1. Run each (method, budget) via train_q_state_read.run_method_budget
  2. Aggregate results -> CSV + JSON
  3. Generate report
  4. Generate plots

Usage:
  python branch_a_gate_mvp/src/run_mvp2c_q_state_read.py \\
    --source branch_a_gate_mvp/outputs_scale_sweep/scale_200k_seed42 \\
    --states_dir branch_a_gate_mvp/outputs_mvp2b_state_write/states \\
    --raw_memory_csv branch_a_gate_mvp/outputs_mvp2_write_memory/reports/write_memory_results.csv \\
    --output branch_a_gate_mvp/outputs_mvp2c_q_state_read \\
    --methods minibatch_kmeans query_kmeans streaming_write_thr0.995 utility_weighted \\
    --budgets 5000 10000 25000 50000 \\
    --max_q_samples 800000

Fast mode:
  python branch_a_gate_mvp/src/run_mvp2c_q_state_read.py \\
    --source branch_a_gate_mvp/outputs_scale_sweep/scale_200k_seed42 \\
    --states_dir branch_a_gate_mvp/outputs_mvp2b_state_write/states \\
    --output branch_a_gate_mvp/outputs_mvp2c_q_state_read_fast \\
    --methods minibatch_kmeans query_kmeans \\
    --budgets 10000 25000 \\
    --fast_action_grid \\
    --max_q_samples 300000
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json
import time
from datetime import datetime
import pandas as pd
import torch

from utils import get_device, set_seed
from train_q_state_read import (
    run_method_budget,
    generate_report,
    try_plots,
)

DEFAULT_METHODS = [
    "minibatch_kmeans",
    "query_kmeans",
    "streaming_write_thr0.995",
    "utility_weighted",
]
DEFAULT_BUDGETS = [5000, 10000, 25000, 50000]


# ── sentinel caching ───────────────────────────────────────────────────────────

def sentinel(out: Path, name: str) -> Path:
    p = out / "sentinels"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{name}.done"


def is_done(out: Path, name: str) -> bool:
    return sentinel(out, name).exists()


def mark_done(out: Path, name: str):
    sentinel(out, name).write_text(datetime.now().isoformat())


def clear(out: Path, name: str):
    s = sentinel(out, name)
    if s.exists():
        s.unlink()


# ── aggregate ─────────────────────────────────────────────────────────────────

def aggregate(all_metrics: list[dict], out: Path):
    if not all_metrics:
        print("  No results to aggregate.")
        return

    (out / "reports").mkdir(parents=True, exist_ok=True)

    rows = []
    for m in all_metrics:
        tops = m.get("top_actions", [])
        rows.append({
            "state_method":               m["method"],
            "budget":                     m["budget"],
            "read_policy":                "q_state_read",
            "val_nll":                    m["q_state_nll"],
            "val_ppl":                    round(2.71828 ** m["q_state_nll"], 4),
            "avg_k_states":               m["avg_k_states"],
            "retrieval_usage":            m["retrieval_usage"],
            "best_fixed_state_nll":       m["best_fixed_nll"],
            "q_state_nll":                m["q_state_nll"],
            "oracle_state_nll":           m["oracle_nll"],
            "delta_q_vs_best_fixed_state":m["delta_q_vs_fixed"],
            "delta_q_vs_raw_random_same_budget": m["delta_q_vs_random"],
            "delta_q_vs_full_ds_fixed":   m["delta_q_vs_ds_fixed"],
            "delta_q_vs_full_ds_qmlp":    m["delta_q_vs_ds_qmlp"],
            "top_action_1":               tops[0]["name"] if tops else "",
            "top_action_1_frac":          tops[0]["frac"] if tops else 0.0,
            "top_action_2":               tops[1]["name"] if len(tops) > 1 else "",
            "top_action_2_frac":          tops[1]["frac"] if len(tops) > 1 else 0.0,
            "oracle_gap_vs_fixed":        m["oracle_gap_vs_fixed"],
            "oracle_gap_vs_q":            m["oracle_gap_vs_q"],
            "raw_random_nll":             m["raw_random_nll"],
            "full_ds_fixed_nll":          m["full_ds_fixed_nll"],
            "full_ds_qmlp_nll":           m["full_ds_qmlp_nll"],
        })
        # Also add best_fixed_state_read row
        rows.append({
            "state_method":               m["method"],
            "budget":                     m["budget"],
            "read_policy":                "best_fixed_state_read",
            "val_nll":                    m["best_fixed_nll"],
            "val_ppl":                    round(2.71828 ** m["best_fixed_nll"], 4),
            "avg_k_states":               float("nan"),
            "retrieval_usage":            float("nan"),
            "best_fixed_state_nll":       m["best_fixed_nll"],
            "q_state_nll":                m["q_state_nll"],
            "oracle_state_nll":           m["oracle_nll"],
            "delta_q_vs_best_fixed_state":float("nan"),
            "delta_q_vs_raw_random_same_budget": m["best_fixed_nll"] - m["raw_random_nll"],
            "delta_q_vs_full_ds_fixed":   m["best_fixed_nll"] - m["full_ds_fixed_nll"],
            "delta_q_vs_full_ds_qmlp":    m["best_fixed_nll"] - m["full_ds_qmlp_nll"],
            "top_action_1":               m["best_fixed_action"],
            "top_action_1_frac":          1.0,
            "top_action_2":               "",
            "top_action_2_frac":          0.0,
            "oracle_gap_vs_fixed":        m["oracle_gap_vs_fixed"],
            "oracle_gap_vs_q":            m["oracle_gap_vs_q"],
            "raw_random_nll":             m["raw_random_nll"],
            "full_ds_fixed_nll":          m["full_ds_fixed_nll"],
            "full_ds_qmlp_nll":           m["full_ds_qmlp_nll"],
        })
        # Oracle row
        rows.append({
            "state_method":               m["method"],
            "budget":                     m["budget"],
            "read_policy":                "oracle_state_action_diag",
            "val_nll":                    m["oracle_nll"],
            "val_ppl":                    round(2.71828 ** m["oracle_nll"], 4),
            "avg_k_states":               float("nan"),
            "retrieval_usage":            float("nan"),
            "best_fixed_state_nll":       m["best_fixed_nll"],
            "q_state_nll":                m["q_state_nll"],
            "oracle_state_nll":           m["oracle_nll"],
            "delta_q_vs_best_fixed_state":float("nan"),
            "delta_q_vs_raw_random_same_budget": m["oracle_nll"] - m["raw_random_nll"],
            "delta_q_vs_full_ds_fixed":   m["oracle_nll"] - m["full_ds_fixed_nll"],
            "delta_q_vs_full_ds_qmlp":    m["oracle_nll"] - m["full_ds_qmlp_nll"],
            "top_action_1":               "",
            "top_action_1_frac":          float("nan"),
            "top_action_2":               "",
            "top_action_2_frac":          float("nan"),
            "oracle_gap_vs_fixed":        m["oracle_gap_vs_fixed"],
            "oracle_gap_vs_q":            m["oracle_gap_vs_q"],
            "raw_random_nll":             m["raw_random_nll"],
            "full_ds_fixed_nll":          m["full_ds_fixed_nll"],
            "full_ds_qmlp_nll":           m["full_ds_qmlp_nll"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(out / "reports" / "q_state_read_results.csv", index=False)
    with open(out / "reports" / "q_state_read_results.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  Saved reports/q_state_read_results.csv ({len(df)} rows)")

    # Terminal summary
    q_only = df[df["read_policy"] == "q_state_read"]
    print(f"\n{'='*75}")
    print(f"  {'Method':<30} {'Budget':>6}  {'Fixed':>7}  {'Q':>7}  {'Oracle':>7}  {'dQ_fix':>7}")
    print(f"  {'-'*75}")
    for _, r in q_only.sort_values(["state_method", "budget"]).iterrows():
        print(f"  {r['state_method']:<30} {int(r['budget']):>6}  "
              f"{r['best_fixed_state_nll']:>7.4f}  {r['q_state_nll']:>7.4f}  "
              f"{r['oracle_state_nll']:>7.4f}  {r['delta_q_vs_best_fixed_state']:>+7.4f}")
    print(f"\n  full_ds_fixed={q_only['full_ds_fixed_nll'].iloc[0]:.4f}  "
          f"full_ds_qmlp={q_only['full_ds_qmlp_nll'].iloc[0]:.4f}")
    print(f"{'='*75}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",          required=True)
    parser.add_argument("--states_dir",      required=True)
    parser.add_argument("--output",          required=True)
    parser.add_argument("--raw_memory_csv",  default=None)

    parser.add_argument("--methods",  nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--budgets",  nargs="+", type=int, default=DEFAULT_BUDGETS)

    parser.add_argument("--max_q_samples",   type=int, default=800_000)
    parser.add_argument("--n_epochs",        type=int, default=30)
    parser.add_argument("--fast_action_grid",action="store_true",
                        help="Use 80-action fast grid instead of 217")

    parser.add_argument("--force_neighbors", action="store_true")
    parser.add_argument("--force_rewards",   action="store_true")
    parser.add_argument("--force_train",     action="store_true")
    parser.add_argument("--force_eval",      action="store_true")
    parser.add_argument("--force_all",       action="store_true")

    args = parser.parse_args()

    if args.force_all:
        args.force_neighbors = args.force_rewards = args.force_train = args.force_eval = True

    src        = Path(args.source)
    states_dir = Path(args.states_dir)
    out        = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    device = get_device({"device": "cuda"})
    set_seed(42)

    print(f"\nMVP 2c: Q-State-Read Controller")
    print(f"  Source:     {src}")
    print(f"  States:     {states_dir}")
    print(f"  Output:     {out}")
    print(f"  Methods:    {args.methods}")
    print(f"  Budgets:    {args.budgets}")
    print(f"  Device:     {device}")
    print(f"  max_q_samples: {args.max_q_samples:,}")
    print(f"  fast_grid:  {args.fast_action_grid}")

    t0          = time.time()
    all_metrics = []
    failed      = []

    for method in args.methods:
        for budget in args.budgets:
            tag = f"{method}_B{budget}"
            key = f"qread_{tag}"

            cached_path = out / tag / "q_metrics.json"
            if cached_path.exists() and not args.force_eval and not args.force_train:
                print(f"\n  [cached] {tag}")
                with open(cached_path) as f:
                    all_metrics.append(json.load(f))
                continue

            print(f"\n{'='*60}")
            print(f"  {tag}")
            print(f"{'='*60}")

            try:
                m = run_method_budget(
                    method=method,
                    budget=budget,
                    src=src,
                    states_dir=states_dir,
                    out=out,
                    device=device,
                    raw_memory_csv=args.raw_memory_csv,
                    max_q_samples=args.max_q_samples,
                    n_epochs=args.n_epochs,
                    fast_grid=args.fast_action_grid,
                    force_neighbors=args.force_neighbors,
                    force_rewards=args.force_rewards,
                    force_train=args.force_train,
                    force_eval=args.force_eval,
                )
                if m is not None:
                    all_metrics.append(m)
                    mark_done(out, key)
            except Exception as e:
                import traceback
                print(f"  [FAILED] {tag}: {e}")
                traceback.print_exc()
                failed.append(tag)

    # Aggregate
    print(f"\n{'='*60}\nAggregating results ...\n{'='*60}")
    aggregate(all_metrics, out)

    # Report
    report = generate_report(all_metrics)
    rpath  = out / "Q_STATE_READ_REPORT.md"
    with open(rpath, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport: {rpath}")

    # Plots
    try_plots(all_metrics, out)

    elapsed = time.time() - t0
    status  = "PARTIAL" if failed else "OK"
    print(f"\nPipeline {status}  elapsed={elapsed:.0f}s ({elapsed/60:.1f}m)")
    if failed:
        print(f"  Failed: {failed}")


if __name__ == "__main__":
    main()

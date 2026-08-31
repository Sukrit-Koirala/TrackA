"""
run_state_analysis.py  —  MVP 2c state analysis pipeline

Stages:
  1. analyze_state_behavior    per-state Q-usage stats
  2. inspect_top_states        human-readable inspection files
  3. analyze_q_state_actions   action distribution + behavior by bins
  4. compare_state_methods     cross-method comparison + final report

Usage:
  python branch_a_gate_mvp/src/run_state_analysis.py \\
    --source branch_a_gate_mvp/outputs_scale_sweep/scale_200k_seed42 \\
    --states_dir branch_a_gate_mvp/outputs_mvp2b_state_write/states \\
    --q_read_dir branch_a_gate_mvp/outputs_mvp2c_q_state_read \\
    --output branch_a_gate_mvp/outputs_mvp2c_state_analysis \\
    --methods minibatch_kmeans utility_weighted query_kmeans \\
    --budgets 10000 25000 50000
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import subprocess
import time
from datetime import datetime

DEFAULT_METHODS = ["minibatch_kmeans", "utility_weighted", "query_kmeans"]
DEFAULT_BUDGETS = [10000, 25000, 50000]


def sentinel(out: Path, name: str) -> Path:
    p = out / "sentinels"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{name}.done"


def is_done(out: Path, name: str) -> bool:
    return sentinel(out, name).exists()


def mark_done(out: Path, name: str):
    sentinel(out, name).write_text(datetime.now().isoformat())


def run(cmd: list, dry_run: bool = False) -> int:
    print(f"\n  $ {' '.join(str(c) for c in cmd)}")
    if dry_run:
        print("  [dry_run]"); return 0
    t0  = time.time()
    ret = subprocess.call([sys.executable] + [str(c) for c in cmd])
    print(f"  -> exit={ret}  {time.time()-t0:.0f}s")
    return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",          required=True)
    parser.add_argument("--states_dir",      required=True)
    parser.add_argument("--q_read_dir",      required=True)
    parser.add_argument("--output",          required=True)
    parser.add_argument("--methods",   nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--budgets",   nargs="+", type=int, default=DEFAULT_BUDGETS)
    parser.add_argument("--max_examples_per_state", type=int, default=8)
    parser.add_argument("--force",     action="store_true")
    parser.add_argument("--dry_run",   action="store_true")
    args = parser.parse_args()

    src        = Path(args.source)
    states_dir = Path(args.states_dir)
    q_read_dir = Path(args.q_read_dir)
    out        = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    scripts = Path(__file__).parent

    print(f"\nMVP 2c State Analysis Pipeline")
    print(f"  Source:    {src}")
    print(f"  States:    {states_dir}")
    print(f"  Q-read:    {q_read_dir}")
    print(f"  Output:    {out}")
    print(f"  Methods:   {args.methods}")
    print(f"  Budgets:   {args.budgets}")
    t0 = time.time()

    # Stage 1: per-state behavior stats
    key1 = "analyze_state_behavior"
    if args.force or not is_done(out, key1):
        print(f"\n{'='*55}\nStage 1: analyze_state_behavior\n{'='*55}")
        ret = run([
            scripts / "analyze_state_behavior.py",
            "--source",     src,
            "--states_dir", states_dir,
            "--q_read_dir", q_read_dir,
            "--output",     out,
            "--methods",    *args.methods,
            "--budgets",    *map(str, args.budgets),
        ], args.dry_run)
        if ret == 0: mark_done(out, key1)
    else:
        print(f"[skip] {key1}")

    # Stage 2: inspection files
    key2 = "inspect_top_states"
    if args.force or not is_done(out, key2):
        print(f"\n{'='*55}\nStage 2: inspect_top_states\n{'='*55}")
        ret = run([
            scripts / "inspect_top_states.py",
            "--source",     src,
            "--states_dir", states_dir,
            "--q_read_dir", q_read_dir,
            "--stats_dir",  out / "state_stats",
            "--output",     out,
            "--methods",    *args.methods,
            "--budgets",    *map(str, args.budgets),
            "--n_examples", str(args.max_examples_per_state),
        ], args.dry_run)
        if ret == 0: mark_done(out, key2)
    else:
        print(f"[skip] {key2}")

    # Stage 3: action analysis
    key3 = "analyze_q_state_actions"
    if args.force or not is_done(out, key3):
        print(f"\n{'='*55}\nStage 3: analyze_q_state_actions\n{'='*55}")
        ret = run([
            scripts / "analyze_q_state_actions.py",
            "--source",     src,
            "--states_dir", states_dir,
            "--q_read_dir", q_read_dir,
            "--output",     out,
            "--methods",    *args.methods,
            "--budgets",    *map(str, args.budgets),
        ], args.dry_run)
        if ret == 0: mark_done(out, key3)
    else:
        print(f"[skip] {key3}")

    # Stage 4: cross-method comparison + report
    key4 = "compare_state_methods"
    if args.force or not is_done(out, key4):
        print(f"\n{'='*55}\nStage 4: compare_state_methods + report\n{'='*55}")
        ret = run([
            scripts / "compare_state_methods.py",
            "--q_read_dir", q_read_dir,
            "--output",     out,
            "--methods",    *args.methods,
            "--budgets",    *map(str, args.budgets),
        ], args.dry_run)
        if ret == 0: mark_done(out, key4)
    else:
        print(f"[skip] {key4}")

    elapsed = time.time() - t0
    print(f"\n{'='*55}")
    print(f"Analysis complete  elapsed={elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"Output: {out}")
    report = out / "STATE_ANALYSIS_REPORT.md"
    if report.exists():
        print(f"Report: {report}")

    # Quick summary from state_stats
    try:
        import pandas as pd
        stats_dir = out / "state_stats"
        dfs = list(stats_dir.glob("*_state_stats.csv"))
        if dfs:
            all_df = pd.concat([pd.read_csv(f) for f in dfs], ignore_index=True)
            sel    = all_df[all_df["times_selected"] > 0]
            print(f"\n  Total states:   {len(all_df):,}")
            print(f"  Active states:  {len(sel):,}  ({len(sel)/len(all_df):.1%})")
            if not sel.empty:
                best  = sel.loc[sel["net_util_vs_gpt"].idxmax()]
                print(f"  Best state:     {best['method']} B={int(best['budget'])}  "
                      f"state={int(best['state_id'])}  "
                      f"util={best['net_util_vs_gpt']:.4f}  "
                      f"top1='{best.get('top1_token_str','?')}'")
    except Exception as e:
        pass


if __name__ == "__main__":
    main()

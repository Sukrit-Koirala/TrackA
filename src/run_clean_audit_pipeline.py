"""
run_clean_audit_pipeline.py

Convenience script: runs the full clean story-split audit pipeline in order.

Steps:
  1. extract_states.py       — story-level split extraction
  2. build_neighbors.py      — cosine neighbors + story_id metadata
  3. audit_splits.py         — split overlap + neighbor source audit
  4. baselines.py            — fixed kNN baselines
  5. train_controller.py     — old MLP controller (for comparison)
  6. train_q_controller.py   — Q-MLP full + small action sets (also caches Q base data)
  7. train_sklearn_controllers.py — sklearn Q-controllers
  8. run_heuristics.py       — extended heuristics + cost sweep (NaN-fixed)
  9. similarity_diagnostics.py   — similarity distribution diagnostics
  10. evaluate_v2.py          — unified comparison table
  11. inspect_results_v2.py   — per-method inspection files
  12. compare_split_results.py — old vs clean comparison + AUDIT_REPORT.md

Usage:
  python src/run_clean_audit_pipeline.py --config configs/clean_storysplit.yaml

Flags:
  --start-step N    Skip to step N (useful for resuming after a failure)
  --dry-run         Print commands without executing
"""

import argparse
import subprocess
import sys
from pathlib import Path

STEPS = [
    (1,  "extract_states.py",            "Story-level state extraction"),
    (2,  "build_neighbors.py",           "Cosine neighbor search + story metadata"),
    (3,  "audit_splits.py",              "Split overlap audit"),
    (4,  "baselines.py",                 "Fixed kNN baselines"),
    (5,  "train_controller.py",          "Old MLP classifier controller"),
    (6,  "train_q_controller.py",        "Q-MLP controllers (full + A/B/C)"),
    (7,  "train_sklearn_controllers.py", "Sklearn Q-controllers"),
    (8,  "run_heuristics.py",            "Extended heuristics + cost sweep"),
    (9,  "similarity_diagnostics.py",    "Similarity distribution diagnostics"),
    (10, "evaluate_v2.py",               "Unified evaluation table v2"),
    (11, "inspect_results_v2.py",        "Per-method inspection files"),
    (12, "compare_split_results.py",     "Old vs clean comparison + AUDIT_REPORT.md"),
]


def run_step(script: str, config: str, dry_run: bool) -> bool:
    src_dir = Path(__file__).parent
    cmd = [sys.executable, str(src_dir / script), "--config", config]
    print(f"\n{'='*70}")
    print(f"  Running: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    if dry_run:
        print("  [dry-run: skipping execution]")
        return True

    import os
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"\n*** Step FAILED (exit code {result.returncode}): {script} ***")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Full clean audit pipeline for Branch A Gate-Chain MVP")
    parser.add_argument("--config",     default="configs/clean_storysplit.yaml")
    parser.add_argument("--start-step", type=int, default=1,
                        help="Start from this step number (for resuming)")
    parser.add_argument("--only-step",  type=int, default=None,
                        help="Run only this step number")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Print commands without running them")
    args = parser.parse_args()

    print(f"\nBranch A Gate-Chain MVP — Clean Audit Pipeline")
    print(f"Config:     {args.config}")
    print(f"Start step: {args.start_step}")
    print(f"Dry run:    {args.dry_run}")

    for step_n, script, desc in STEPS:
        if args.only_step is not None and step_n != args.only_step:
            continue
        if step_n < args.start_step:
            print(f"  [skipping step {step_n}: {script}]")
            continue

        print(f"\n[Step {step_n}/12] {desc}")
        ok = run_step(script, args.config, args.dry_run)
        if not ok:
            print(f"\nPipeline aborted at step {step_n}.  "
                  f"Re-run with --start-step {step_n} to resume.")
            sys.exit(1)

    print("\n" + "="*70)
    print("  ALL STEPS COMPLETE")
    print(f"  See: {Path(args.config).stem.replace('clean_storysplit', 'outputs_clean_storysplit')}")
    print("  AUDIT_REPORT.md is in outputs_clean_storysplit/audit/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

"""
run_mvp2_write_memory.py

Convenience runner for the MVP 2 WRITE memory gate pipeline.

Pipeline stages (in order):
  1. compute_write_utilities   -- per-datastore-entry utility from CT retrieval
  2. build_write_features      -- 11 scalar + density features per entry
  3. train_write_scorer        -- Ridge / GBR / MLP scorers on utility targets
  4. select_write_memories     -- B entries per method, per budget
  5. build_memory_neighbors    -- CT + val neighbors into each written memory
  6. evaluate_write_memory     -- best-fixed + Q-MLP transfer eval for all memories
  7. analyze_written_memory    -- content statistics per memory subset

Sentinel-based caching: each stage leaves a sentinel file in
  <output>/sentinels/<stage_name>.done
If the sentinel exists, the stage is skipped unless the corresponding
--force_<stage> flag is set.

Flags:
  --source             path to scale_200k_seed42 output (required)
  --output             output root dir (default: outputs_mvp2_write_memory)
  --budgets            memory budgets (default: 1000 5000 10000 25000 50000)
  --force_utilities    re-run compute_write_utilities
  --force_features     re-run build_write_features
  --force_train        re-run train_write_scorer
  --force_memories     re-run select_write_memories
  --force_neighbors    re-run build_memory_neighbors
  --force_eval         re-run evaluate_write_memory
  --force_analysis     re-run analyze_written_memory
  --force_all          re-run every stage
  --dry_run            print commands without running them
  --no_plots           skip matplotlib plots in evaluate_write_memory
  --methods            subset of selection methods (default: all)

Usage:
  python src/run_mvp2_write_memory.py \\
    --source outputs_scale_sweep/scale_200k_seed42 \\
    --output outputs_mvp2_write_memory
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import subprocess
import time
from datetime import datetime


STAGES = [
    "utilities",
    "features",
    "train",
    "memories",
    "neighbors",
    "eval",
    "analysis",
]

STAGE_TO_SCRIPT = {
    "utilities": "compute_write_utilities.py",
    "features":  "build_write_features.py",
    "train":     "train_write_scorer.py",
    "memories":  "select_write_memories.py",
    "neighbors": "build_memory_neighbors.py",
    "eval":      "evaluate_write_memory.py",
    "analysis":  "analyze_written_memory.py",
}


def sentinel_path(out: Path, stage: str) -> Path:
    return out / "sentinels" / f"{stage}.done"


def is_done(out: Path, stage: str) -> bool:
    return sentinel_path(out, stage).exists()


def mark_done(out: Path, stage: str):
    p = sentinel_path(out, stage)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(datetime.now().isoformat())


def run_stage(
    stage: str,
    cmd: list[str],
    out: Path,
    dry_run: bool,
    force: bool,
) -> bool:
    """Returns True if stage ran (or was skipped as already done)."""
    if not force and is_done(out, stage):
        print(f"\n[{stage}] Already done (sentinel exists). Skipping.")
        return True

    label = " ".join(str(c) for c in cmd)
    print(f"\n{'='*70}")
    print(f"[{stage}] Running: {label}")
    print(f"{'='*70}")

    if dry_run:
        print("  (dry_run: not executing)")
        return True

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n[{stage}] FAILED with exit code {result.returncode}.")
        return False

    print(f"\n[{stage}] Done in {elapsed:.1f}s.")
    mark_done(out, stage)
    return True


def build_cmd(script: str, src: str, out: str, extra: list[str]) -> list[str]:
    return [sys.executable, str(Path(__file__).parent / script),
            "--source", src, "--output", out] + extra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",           required=True)
    parser.add_argument("--output",           default="outputs_mvp2_write_memory")
    parser.add_argument("--budgets",          nargs="+", type=int,
                        default=[1000, 5000, 10000, 25000, 50000])
    parser.add_argument("--methods",          nargs="+", default=None)
    parser.add_argument("--force_utilities",  action="store_true")
    parser.add_argument("--force_features",   action="store_true")
    parser.add_argument("--force_train",      action="store_true")
    parser.add_argument("--force_memories",   action="store_true")
    parser.add_argument("--force_neighbors",  action="store_true")
    parser.add_argument("--force_eval",       action="store_true")
    parser.add_argument("--force_analysis",   action="store_true")
    parser.add_argument("--force_all",        action="store_true")
    parser.add_argument("--dry_run",          action="store_true")
    parser.add_argument("--no_plots",         action="store_true")
    args = parser.parse_args()

    src = args.source
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    budgets_str = [str(b) for b in args.budgets]

    force: dict[str, bool] = {
        "utilities": args.force_utilities or args.force_all,
        "features":  args.force_features  or args.force_all,
        "train":     args.force_train      or args.force_all,
        "memories":  args.force_memories   or args.force_all,
        "neighbors": args.force_neighbors  or args.force_all,
        "eval":      args.force_eval       or args.force_all,
        "analysis":  args.force_analysis   or args.force_all,
    }

    print(f"\nMVP 2 Write Memory Pipeline")
    print(f"Source : {src}")
    print(f"Output : {out}")
    print(f"Budgets: {args.budgets}")
    if args.dry_run:
        print("DRY RUN -- no commands will actually execute")

    t_total = time.time()

    # ── 1. compute utilities ──────────────────────────────────────────────────
    cmd = build_cmd("compute_write_utilities.py", src, str(out), [])
    ok  = run_stage("utilities", cmd, out, args.dry_run, force["utilities"])
    if not ok:
        print("Pipeline aborted at: utilities"); sys.exit(1)

    # ── 2. build write features ───────────────────────────────────────────────
    cmd = build_cmd("build_write_features.py", src, str(out), [])
    ok  = run_stage("features", cmd, out, args.dry_run, force["features"])
    if not ok:
        print("Pipeline aborted at: features"); sys.exit(1)

    # ── 3. train write scorer ─────────────────────────────────────────────────
    cmd = build_cmd("train_write_scorer.py", src, str(out), [])
    ok  = run_stage("train", cmd, out, args.dry_run, force["train"])
    if not ok:
        print("Pipeline aborted at: train"); sys.exit(1)

    # ── 4. select write memories ──────────────────────────────────────────────
    extra = ["--budgets"] + budgets_str
    if args.methods:
        extra += ["--methods"] + args.methods
    cmd = build_cmd("select_write_memories.py", src, str(out), extra)
    ok  = run_stage("memories", cmd, out, args.dry_run, force["memories"])
    if not ok:
        print("Pipeline aborted at: memories"); sys.exit(1)

    # ── 5. build memory neighbors ─────────────────────────────────────────────
    extra = []
    if args.methods:
        extra += ["--methods"] + args.methods
    if force["neighbors"]:
        extra += ["--force"]
    cmd = build_cmd("build_memory_neighbors.py", src, str(out), extra)
    ok  = run_stage("neighbors", cmd, out, args.dry_run, force["neighbors"])
    if not ok:
        print("Pipeline aborted at: neighbors"); sys.exit(1)

    # ── 6. evaluate write memory ──────────────────────────────────────────────
    extra = ["--no_plots"] if args.no_plots else []
    cmd = build_cmd("evaluate_write_memory.py", src, str(out), extra)
    ok  = run_stage("eval", cmd, out, args.dry_run, force["eval"])
    if not ok:
        print("Pipeline aborted at: eval"); sys.exit(1)

    # ── 7. analyze written memory ─────────────────────────────────────────────
    extra = []
    if args.methods:
        extra += ["--methods"] + args.methods
    cmd = build_cmd("analyze_written_memory.py", src, str(out), extra)
    ok  = run_stage("analysis", cmd, out, args.dry_run, force["analysis"])
    if not ok:
        print("Pipeline aborted at: analysis"); sys.exit(1)

    total = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"MVP 2 pipeline complete in {total:.1f}s")
    print(f"Output: {out.resolve()}")
    print(f"  reports/write_memory_results.csv")
    print(f"  reports/memory_content_analysis.csv")
    print(f"  MVP2_WRITE_MEMORY_REPORT.md")
    if not args.no_plots:
        print(f"  plots/nll_vs_budget.png")
        print(f"  plots/delta_vs_random_by_budget.png")
        print(f"  plots/memory_fraction_vs_nll.png")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

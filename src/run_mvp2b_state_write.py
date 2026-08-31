"""
run_mvp2b_state_write.py

Pipeline runner for MVP 2b: State-Forming WRITE.

Stages:
  1. states  — build_predictive_states.py     (per method × budget)
  2. eval    — evaluate_predictive_states.py
  3. analysis— analyze_predictive_states.py
  4. report  — final comparison vs MVP2a raw memory

Caching: sentinels in <output>/sentinels/<stage>.done
Each stage sentinel is method-budget-specific where appropriate.

Usage:
  python branch_a_gate_mvp/src/run_mvp2b_state_write.py \\
    --source outputs_scale_sweep/scale_200k_seed42 \\
    --output outputs_mvp2b_state_write

  python branch_a_gate_mvp/src/run_mvp2b_state_write.py \\
    --source outputs_scale_sweep/scale_200k_seed42 \\
    --output outputs_mvp2b_state_write \\
    --raw_memory_csv outputs_mvp2_write_memory/reports/write_memory_results.csv \\
    --methods minibatch_kmeans random_partition \\
    --budgets 5000 25000

Flags:
  --force_states --force_eval --force_analysis --force_all
  --dry_run  --no_plots  --methods M1 M2 ...  --budgets B1 B2 ...
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import subprocess
import time
from datetime import datetime

METHODS  = [
    "minibatch_kmeans",
    "query_kmeans",
    "balanced_kmeans",
    "streaming_write",
    "utility_weighted",
    "random_partition",
]
BUDGETS  = [1000, 5000, 10000, 25000, 50000]


# ── sentinel helpers ───────────────────────────────────────────────────────────

def sentinel_path(out: Path, name: str) -> Path:
    p = out / "sentinels"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{name}.done"


def is_done(out: Path, name: str) -> bool:
    return sentinel_path(out, name).exists()


def mark_done(out: Path, name: str):
    sentinel_path(out, name).write_text(datetime.now().isoformat())


def clear_sentinel(out: Path, name: str):
    s = sentinel_path(out, name)
    if s.exists():
        s.unlink()


# ── runner ─────────────────────────────────────────────────────────────────────

def run(cmd: list[str], dry_run: bool = False) -> int:
    print(f"\n  $ {' '.join(str(c) for c in cmd)}")
    if dry_run:
        print("  [dry_run] skipped")
        return 0
    t0  = time.time()
    ret = subprocess.call([sys.executable] + [str(c) for c in cmd])
    dt  = time.time() - t0
    print(f"  -> exit={ret}  elapsed={dt:.1f}s")
    return ret


# ── stage 1: build states ──────────────────────────────────────────────────────

def stage_states(args, out: Path, src: Path):
    print(f"\n{'='*60}\nSTAGE 1: Build persistent predictive states\n{'='*60}")

    methods = args.methods
    budgets = args.budgets
    force   = args.force_states or args.force_all

    script = Path(__file__).parent / "build_predictive_states.py"

    failures = []
    for method in methods:
        for budget in budgets:
            key = f"states_{method}_B{budget}"
            if not force and is_done(out, key):
                print(f"  [skip] {key}")
                continue

            extra = []
            if method in ("utility_weighted",):
                util_path = src / "utilities" / "utilities.pt"
                if util_path.exists():
                    extra = ["--util_path", str(util_path)]

            cmd = [
                script,
                "--source",  src,
                "--output",  out,
                "--method",  method,
                "--budget",  budget,
            ] + extra

            ret = run(cmd, args.dry_run)
            if ret != 0:
                print(f"  [WARN] {key} exited {ret}")
                failures.append(key)
            else:
                mark_done(out, key)

    if failures:
        print(f"\n  [WARN] {len(failures)} state builds failed: {failures}")
    return len(failures) == 0


# ── stage 2: evaluate states ───────────────────────────────────────────────────

def stage_eval(args, out: Path, src: Path):
    print(f"\n{'='*60}\nSTAGE 2: Evaluate persistent predictive states\n{'='*60}")

    key    = "eval_states"
    force  = args.force_eval or args.force_all
    if not force and is_done(out, key):
        print(f"  [skip] {key}")
        return True

    script = Path(__file__).parent / "evaluate_predictive_states.py"
    cmd    = [
        script,
        "--source", src,
        "--output", out,
    ]
    if args.raw_memory_csv:
        cmd += ["--raw_memory_csv", args.raw_memory_csv]

    ret = run(cmd, args.dry_run)
    if ret == 0:
        mark_done(out, key)
    return ret == 0


# ── stage 3: analyze states ────────────────────────────────────────────────────

def stage_analysis(args, out: Path, src: Path):
    print(f"\n{'='*60}\nSTAGE 3: Analyze state quality\n{'='*60}")

    key   = "analysis_states"
    force = args.force_analysis or args.force_all
    if not force and is_done(out, key):
        print(f"  [skip] {key}")
        return True

    script = Path(__file__).parent / "analyze_predictive_states.py"
    cmd    = [
        script,
        "--source", src,
        "--output", out,
    ]

    ret = run(cmd, args.dry_run)
    if ret == 0:
        mark_done(out, key)
    return ret == 0


# ── report comparison ──────────────────────────────────────────────────────────

def print_comparison(out: Path, raw_csv: str | None):
    try:
        import pandas as pd

        eval_csv = out / "reports" / "state_write_results.csv"
        if not eval_csv.exists():
            return

        df = pd.read_csv(eval_csv)
        sub = df[df["read_policy"] == "best_fixed"]
        if sub.empty:
            return

        print(f"\n{'='*70}")
        print("FINAL COMPARISON: State-forming WRITE vs raw memory")
        print(f"{'='*70}")

        if raw_csv and Path(raw_csv).exists():
            raw_df = pd.read_csv(raw_csv)
            rand = raw_df[(raw_df["method"].str.startswith("random")) &
                          (raw_df["read_policy"] == "best_fixed")]
            if not rand.empty:
                for b in sorted(sub["budget"].unique()):
                    r = rand[rand["budget"] == b]["val_nll"]
                    rand_nll = float(r.mean()) if not r.empty else float("nan")
                    b_sub = sub[sub["budget"] == b]
                    if b_sub.empty:
                        continue
                    best = b_sub.loc[b_sub["val_nll"].idxmin()]
                    delta = best["val_nll"] - rand_nll
                    sign  = "BEATS" if delta < 0 else "LOSES"
                    print(f"  B={b:>6}  best_state={best['method']:<25}  "
                          f"NLL={best['val_nll']:.4f}  random={rand_nll:.4f}  "
                          f"d={delta:+.4f}  [{sign}]")

        print()
        gpt_nll   = float(sub["full_gpt_nll"].iloc[0])  if "full_gpt_nll" in sub else float("nan")
        fixed_nll = float(sub["full_fixed_nll"].iloc[0]) if "full_fixed_nll" in sub else float("nan")
        qmlp_nll  = float(sub["full_qmlp_nll"].iloc[0])  if "full_qmlp_nll" in sub else float("nan")
        print(f"  full_ds baselines: GPT={gpt_nll:.4f}  Fixed={fixed_nll:.4f}  Q-MLP={qmlp_nll:.4f}")
        best_state_nll = float(sub["val_nll"].min())
        print(f"  best state NLL:    {best_state_nll:.4f}  "
              f"(d vs fixed={best_state_nll - fixed_nll:+.4f})")
        print(f"{'='*70}")
    except Exception as e:
        print(f"  [comparison skipped: {e}]")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",         required=True)
    parser.add_argument("--output",         required=True)
    parser.add_argument("--raw_memory_csv", default=None,
                        help="Path to write_memory_results.csv from MVP2a")

    parser.add_argument("--methods",    nargs="+", default=METHODS)
    parser.add_argument("--budgets",    nargs="+", type=int, default=BUDGETS)

    parser.add_argument("--force_states",   action="store_true")
    parser.add_argument("--force_eval",     action="store_true")
    parser.add_argument("--force_analysis", action="store_true")
    parser.add_argument("--force_all",      action="store_true")

    parser.add_argument("--dry_run",    action="store_true")
    parser.add_argument("--no_plots",   action="store_true")

    # Only states, skip eval/analysis
    parser.add_argument("--states_only",  action="store_true")
    # Only eval (states already built)
    parser.add_argument("--eval_only",    action="store_true")
    # Skip build, run eval+analysis only (alias for --eval_only)
    parser.add_argument("--skip_build",   action="store_true")

    args = parser.parse_args()

    src = Path(args.source)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\nMVP 2b State-Forming WRITE Pipeline")
    print(f"  Source:  {src}")
    print(f"  Output:  {out}")
    print(f"  Methods: {args.methods}")
    print(f"  Budgets: {args.budgets}")
    if args.raw_memory_csv:
        print(f"  Raw memory CSV: {args.raw_memory_csv}")

    t0 = time.time()

    if args.eval_only or args.skip_build:
        stage_eval(args, out, src)
        stage_analysis(args, out, src)
        print_comparison(out, args.raw_memory_csv)
        return

    ok1 = stage_states(args, out, src)

    if args.states_only:
        print(f"\nDone (states only)  elapsed={time.time()-t0:.0f}s")
        return

    ok2 = stage_eval(args, out, src)
    ok3 = stage_analysis(args, out, src)

    print_comparison(out, args.raw_memory_csv)

    elapsed = time.time() - t0
    status  = "OK" if (ok1 and ok2 and ok3) else "PARTIAL"
    print(f"\nPipeline {status}  total elapsed={elapsed:.0f}s ({elapsed/60:.1f}m)")

    report = out / "MVP2B_STATE_WRITE_REPORT.md"
    if report.exists():
        print(f"Report: {report}")


if __name__ == "__main__":
    main()

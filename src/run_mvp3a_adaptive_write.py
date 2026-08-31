"""
run_mvp3a_adaptive_write.py  --  MVP 3a pipeline runner

Stages:
  1. build    build_adaptive_write_states.py
  2. eval     evaluate_adaptive_write_states.py
  3. analyze  analyze_adaptive_write_states.py

Fast first run:
  python branch_a_gate_mvp/src/run_mvp3a_adaptive_write.py \\
    --source branch_a_gate_mvp/outputs_scale_sweep/scale_200k_seed42 \\
    --output branch_a_gate_mvp/outputs_mvp3a_adaptive_write_fast \\
    --methods budget_filling_write token_conflict_write \\
    --budgets 10000 25000 \\
    --fast_action_grid \\
    --max_q_samples 300000

Full run:
  python branch_a_gate_mvp/src/run_mvp3a_adaptive_write.py \\
    --source branch_a_gate_mvp/outputs_scale_sweep/scale_200k_seed42 \\
    --output branch_a_gate_mvp/outputs_mvp3a_adaptive_write \\
    --methods budget_filling_write uncertainty_aware_write token_conflict_write \\
    --budgets 10000 25000 50000 \\
    --max_q_samples 800000

With 2-pass variants (slow, run only if fast results are promising):
  Add: --methods ... split_conflict_states prune_low_utility_states
       --enable_prune_refill
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse, subprocess, time
from datetime import datetime

DEFAULT_METHODS = ["budget_filling_write", "token_conflict_write"]
DEFAULT_BUDGETS = [10000, 25000]


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
    parser.add_argument("--source",        required=True)
    parser.add_argument("--output",        required=True)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--budgets", nargs="+", type=int, default=DEFAULT_BUDGETS)
    parser.add_argument("--ema_caps", nargs="+", type=int, default=[128])
    parser.add_argument("--max_q_samples",  type=int, default=800_000)
    parser.add_argument("--n_epochs",       type=int, default=30)
    parser.add_argument("--fast",           action="store_true",
                        help="Fewer hyperparam combos in build stage")
    parser.add_argument("--fast_action_grid", action="store_true",
                        help="80-action grid instead of 217 in eval stage")
    parser.add_argument("--enable_prune_refill", action="store_true",
                        help="Also run prune_low_utility_states")
    parser.add_argument("--device",         default="cuda")
    parser.add_argument("--seed",           type=int, default=42)

    # Skip flags
    parser.add_argument("--skip_build",    action="store_true")
    parser.add_argument("--skip_eval",     action="store_true")
    parser.add_argument("--skip_analysis", action="store_true")

    # Force flags
    parser.add_argument("--force_build",     action="store_true")
    parser.add_argument("--force_neighbors", action="store_true")
    parser.add_argument("--force_rewards",   action="store_true")
    parser.add_argument("--force_train",     action="store_true")
    parser.add_argument("--force_eval",      action="store_true")
    parser.add_argument("--force_analysis",  action="store_true")
    parser.add_argument("--force_all",       action="store_true")
    parser.add_argument("--dry_run",         action="store_true")

    args = parser.parse_args()

    if args.force_all:
        args.force_build = args.force_neighbors = args.force_rewards = True
        args.force_train = args.force_eval = args.force_analysis = True

    src    = Path(args.source)
    out    = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).parent

    methods = args.methods[:]
    if args.enable_prune_refill and "prune_low_utility_states" not in methods:
        methods.append("prune_low_utility_states")

    states_dir = out / "states"

    print(f"\nMVP 3a: Adaptive Write States Pipeline")
    print(f"  Source:  {src}")
    print(f"  Output:  {out}")
    print(f"  Methods: {methods}")
    print(f"  Budgets: {args.budgets}")
    print(f"  Fast:    {args.fast}  FastGrid: {args.fast_action_grid}")
    t0 = time.time()

    # ── Stage 1: Build States ─────────────────────────────────────────────────
    key1 = "build_states"
    if args.skip_build:
        print(f"\n[skip] Stage 1 (--skip_build)")
    elif not args.force_build and is_done(out, key1):
        print(f"\n[cached] Stage 1: build_states")
    else:
        print(f"\n{'='*55}\nStage 1: build_adaptive_write_states\n{'='*55}")
        cmd = [
            scripts / "build_adaptive_write_states.py",
            "--source",  src,
            "--output",  out,
            "--methods", *methods,
            "--budgets", *map(str, args.budgets),
            "--ema_caps", *map(str, args.ema_caps),
            "--device",  args.device,
            "--seed",    str(args.seed),
        ]
        if args.fast:
            cmd.append("--fast")
        if args.force_build:
            cmd.append("--force")
        ret = run(cmd, args.dry_run)
        if ret == 0:
            mark_done(out, key1)
        else:
            print(f"Stage 1 failed (exit={ret}).  Aborting.")
            return

    # ── Stage 2: Evaluate States ──────────────────────────────────────────────
    key2 = "eval_states"
    if args.skip_eval:
        print(f"\n[skip] Stage 2 (--skip_eval)")
    elif not args.force_eval and not args.force_neighbors and \
            not args.force_rewards and not args.force_train and is_done(out, key2):
        print(f"\n[cached] Stage 2: eval_states")
    else:
        print(f"\n{'='*55}\nStage 2: evaluate_adaptive_write_states\n{'='*55}")
        cmd = [
            scripts / "evaluate_adaptive_write_states.py",
            "--source",         src,
            "--states_dir",     states_dir,
            "--output",         out,
            "--max_q_samples",  str(args.max_q_samples),
            "--n_epochs",       str(args.n_epochs),
            "--device",         args.device,
            "--seed",           str(args.seed),
        ]
        if args.fast_action_grid:
            cmd.append("--fast_action_grid")
        if args.force_neighbors:
            cmd.append("--force_neighbors")
        if args.force_rewards:
            cmd.append("--force_rewards")
        if args.force_train:
            cmd.append("--force_train")
        if args.force_eval:
            cmd.append("--force_eval")
        ret = run(cmd, args.dry_run)
        if ret == 0:
            mark_done(out, key2)
        else:
            print(f"Stage 2 failed (exit={ret}).  Analysis may be incomplete.")

    # ── Stage 3: Analyze ─────────────────────────────────────────────────────
    key3 = "analyze"
    if args.skip_analysis:
        print(f"\n[skip] Stage 3 (--skip_analysis)")
    elif not args.force_analysis and is_done(out, key3):
        print(f"\n[cached] Stage 3: analyze")
    else:
        print(f"\n{'='*55}\nStage 3: analyze_adaptive_write_states\n{'='*55}")
        cmd = [
            scripts / "analyze_adaptive_write_states.py",
            "--output",     out,
            "--states_dir", states_dir,
        ]
        if args.force_analysis:
            cmd.append("--force")
        ret = run(cmd, args.dry_run)
        if ret == 0:
            mark_done(out, key3)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*55}")
    print(f"MVP 3a complete  elapsed={elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"Output: {out}")

    report = out / "ADAPTIVE_WRITE_REPORT.md"
    csv    = out / "adaptive_write_results.csv"
    if report.exists():
        print(f"Report: {report}")
    if csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(csv)
            if not df.empty and "q_state_nll" in df.columns:
                best = df.loc[df["q_state_nll"].idxmin()]
                print(f"\nBest result:")
                print(f"  tag:       {best['tag']}")
                print(f"  Q NLL:     {best['q_state_nll']:.4f}")
                print(f"  d_qmlp:    {best['delta_q_vs_full_ds_qmlp']:+.4f}")
                print(f"  d_mvp2c:   {best['delta_q_vs_best_mvp2c_overall']:+.4f}")
                n_beat = int((df["q_state_nll"] < 2.2977).sum())
                print(f"  Variants beating full_ds_qmlp: {n_beat}/{len(df)}")
        except Exception:
            pass


if __name__ == "__main__":
    main()

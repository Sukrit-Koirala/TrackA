"""
run_mvp3b_aggregate_write.py  --  MVP 3b pipeline runner

Stages:
  1. build    build_aggregate_write_states.py
  2. health   evaluate_aggregate_write_states.py  (health check only)
  3. eval     evaluate_aggregate_write_states.py  (Q-state-read, health-passing only)
  4. analyze  analyze_aggregate_write_states.py

Fast first run:
  python branch_a_gate_mvp/src/run_mvp3b_aggregate_write.py \\
    --source branch_a_gate_mvp/outputs_scale_sweep/scale_200k_seed42 \\
    --output branch_a_gate_mvp/outputs_mvp3b_aggregate_write_fast \\
    --methods aggregate_basic aggregate_budget_filling aggregate_conflict_aware \\
    --budgets 10000 25000 \\
    --fast \\
    --fast_action_grid \\
    --max_q_samples 300000

Full selected run:
  python branch_a_gate_mvp/src/run_mvp3b_aggregate_write.py \\
    --source branch_a_gate_mvp/outputs_scale_sweep/scale_200k_seed42 \\
    --output branch_a_gate_mvp/outputs_mvp3b_aggregate_write \\
    --methods aggregate_basic aggregate_budget_filling aggregate_conflict_aware aggregate_promote_pure \\
    --budgets 10000 25000 50000 \\
    --max_q_samples 800000
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse, subprocess, time
from datetime import datetime

DEFAULT_METHODS = ["aggregate_basic", "aggregate_budget_filling", "aggregate_conflict_aware"]
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
    print(f"  -> exit={ret}  elapsed={time.time()-t0:.0f}s")
    return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",        required=True)
    parser.add_argument("--output",        required=True)
    parser.add_argument("--methods",       nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--budgets",       nargs="+", type=int, default=DEFAULT_BUDGETS)
    parser.add_argument("--ema_caps",      nargs="+", type=int, default=[128])
    parser.add_argument("--max_q_samples", type=int, default=800_000)
    parser.add_argument("--n_epochs",      type=int, default=30)
    parser.add_argument("--fast",          action="store_true",
                        help="Fewer hyperparam combos in build stage")
    parser.add_argument("--fast_action_grid", action="store_true",
                        help="80-action grid instead of 217 in eval stage")
    parser.add_argument("--eval_all",      action="store_true",
                        help="Q-eval even health-failing configs")
    parser.add_argument("--device",        default="cuda")
    parser.add_argument("--seed",          type=int, default=42)

    # Skip flags
    parser.add_argument("--skip_build",    action="store_true")
    parser.add_argument("--skip_health",   action="store_true")
    parser.add_argument("--skip_eval",     action="store_true")
    parser.add_argument("--skip_analysis", action="store_true")

    # Force flags
    parser.add_argument("--force_build",     action="store_true")
    parser.add_argument("--force_health",    action="store_true")
    parser.add_argument("--force_neighbors", action="store_true")
    parser.add_argument("--force_rewards",   action="store_true")
    parser.add_argument("--force_train",     action="store_true")
    parser.add_argument("--force_eval",      action="store_true")
    parser.add_argument("--force_analysis",  action="store_true")
    parser.add_argument("--force_all",       action="store_true")
    parser.add_argument("--dry_run",         action="store_true")
    args = parser.parse_args()

    if args.force_all:
        args.force_build = args.force_health = args.force_neighbors = \
            args.force_rewards = args.force_train = args.force_eval = \
            args.force_analysis = True

    src     = Path(args.source)
    out     = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).parent

    states_dir = out / "states"

    print(f"\nMVP 3b: Aggregate Gate / Buffer-Commit WRITE Pipeline")
    print(f"  Source:  {src}")
    print(f"  Output:  {out}")
    print(f"  Methods: {args.methods}")
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
        print(f"\n{'='*55}\nStage 1: build_aggregate_write_states\n{'='*55}")
        cmd = [
            scripts / "build_aggregate_write_states.py",
            "--source",  src,
            "--output",  out,
            "--methods", *args.methods,
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

    # ── Stage 2: Health Check + Q-eval ────────────────────────────────────────
    key2 = "eval_states"
    any_eval_force = (args.force_health or args.force_neighbors or args.force_rewards
                      or args.force_train or args.force_eval)

    if args.skip_eval:
        print(f"\n[skip] Stage 2 (--skip_eval)")
    elif not any_eval_force and is_done(out, key2):
        print(f"\n[cached] Stage 2: eval_states")
    else:
        print(f"\n{'='*55}\nStage 2: evaluate_aggregate_write_states\n{'='*55}")
        cmd = [
            scripts / "evaluate_aggregate_write_states.py",
            "--source",          src,
            "--states_dir",      states_dir,
            "--output",          out,
            "--max_q_samples",   str(args.max_q_samples),
            "--n_epochs",        str(args.n_epochs),
            "--device",          args.device,
            "--seed",            str(args.seed),
        ]
        if args.fast_action_grid:
            cmd.append("--fast_action_grid")
        if args.skip_health:
            cmd.append("--skip_health")
        if args.eval_all:
            cmd.append("--eval_all")
        if args.force_health:
            cmd.append("--force_health")
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
        print(f"\n{'='*55}\nStage 3: analyze_aggregate_write_states\n{'='*55}")
        cmd = [
            scripts / "analyze_aggregate_write_states.py",
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
    print(f"MVP 3b complete  elapsed={elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"Output: {out}")

    report = out / "AGGREGATE_WRITE_REPORT.md"
    if report.exists():
        print(f"Report: {report}")

    csv = out / "reports" / "aggregate_write_results.csv"
    if csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(csv)
            if not df.empty:
                print(f"\nResults ({len(df)} variants evaluated):")

                # Health summary
                if "health_pass" in df.columns:
                    n_pass = int(df["health_pass"].sum())
                    print(f"  Health: {n_pass}/{len(df)} passed")
                    if "hit_at_4" in df.columns:
                        pass_df = df[df["health_pass"] == True]
                        if not pass_df.empty:
                            print(f"  Best hit@4 (passing): {pass_df['hit_at_4'].max():.2%}")
                            print(f"  Best median count (passing): {pass_df['median_state_count'].max():.1f}")

                # Q-NLL summary
                df_q = df.dropna(subset=["q_state_nll"])
                if not df_q.empty:
                    best = df_q.loc[df_q["q_state_nll"].idxmin()]
                    print(f"\n  Best Q NLL: {best['q_state_nll']:.4f}  ({best['tag']})")
                    print(f"  delta_vs_mvp3a_best = {best['delta_vs_mvp3a_best']:+.4f}")
                    print(f"  delta_vs_full_ds_qmlp = {best['delta_vs_full_ds_qmlp']:+.4f}")
                    print(f"  delta_vs_mvp2c_best = {best['delta_vs_mvp2c_best']:+.4f}")

                    n_beat_mvp3a = int((df_q["q_state_nll"] < 2.4391).sum())
                    n_beat_qmlp  = int((df_q["q_state_nll"] < 2.2977).sum())
                    print(f"\n  Beating MVP 3a best (2.4391): {n_beat_mvp3a}/{len(df_q)}")
                    print(f"  Beating full_ds_qmlp (2.2977): {n_beat_qmlp}/{len(df_q)}")
        except Exception as e:
            print(f"  [summary failed: {e}]")


if __name__ == "__main__":
    main()

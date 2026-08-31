"""
run_mvp4a_write_imitation.py  --  MVP 4a pipeline runner

Stages:
  1. teacher_assignments  build_offline_teacher_assignments.py
  2. dataset             build_write_imitation_dataset.py
  3. train               train_write_imitation.py
  4. build_states        build_imitation_write_states.py
  5. eval                evaluate_imitation_write_states.py  (health + Q-read)
  6. analyze             analyze_write_imitation.py

Fast command:
  python src/run_mvp4a_write_imitation.py \\
    --source branch_a_gate_mvp/outputs_scale_sweep/scale_200k_seed42 \\
    --offline_states_dir branch_a_gate_mvp/outputs_mvp2b_state_write/states \\
    --output branch_a_gate_mvp/outputs_mvp4a_write_imitation_fast \\
    --teachers utility_weighted_B10000 minibatch_kmeans_B10000 \\
    --budgets 10000 \\
    --fast_action_grid \\
    --max_q_samples 300000

Full command:
  python src/run_mvp4a_write_imitation.py \\
    --source branch_a_gate_mvp/outputs_scale_sweep/scale_200k_seed42 \\
    --offline_states_dir branch_a_gate_mvp/outputs_mvp2b_state_write/states \\
    --output branch_a_gate_mvp/outputs_mvp4a_write_imitation \\
    --teachers utility_weighted_B10000 minibatch_kmeans_B10000 \\
              utility_weighted_B25000 minibatch_kmeans_B25000 \\
    --budgets 10000 25000 \\
    --max_q_samples 800000
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse, subprocess, time
from datetime import datetime

DEFAULT_TEACHERS = ["utility_weighted_B10000", "minibatch_kmeans_B10000"]
DEFAULT_BUDGETS  = [10000]


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
    parser.add_argument("--source",            required=True)
    parser.add_argument("--offline_states_dir",required=True)
    parser.add_argument("--output",            required=True)
    parser.add_argument("--teachers",    nargs="+", default=DEFAULT_TEACHERS)
    parser.add_argument("--budgets",     nargs="+", type=int, default=DEFAULT_BUDGETS)
    parser.add_argument("--max_q_samples", type=int, default=800_000)
    parser.add_argument("--n_epochs",    type=int, default=30)
    parser.add_argument("--buf_frac",          type=float, default=0.05)
    parser.add_argument("--ema_cap",           type=int,   default=128)
    parser.add_argument("--state_sim_threshold",type=float, default=0.5)
    parser.add_argument("--buf_sim_threshold",  type=float, default=0.3)
    parser.add_argument("--promote_count",      type=int,   default=4)
    parser.add_argument("--promote_purity",     type=float, default=0.5,
                        help="Min token self-purity for buffer promotion")
    parser.add_argument("--fast_action_grid", action="store_true")
    parser.add_argument("--eval_all",    action="store_true", default=True,
                        help="Q-eval even health-failing configs (default True for diagnostics)")
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--seed",        type=int, default=42)

    # Skip flags
    parser.add_argument("--skip_teacher_assignments", action="store_true")
    parser.add_argument("--skip_dataset",             action="store_true")
    parser.add_argument("--skip_train",               action="store_true")
    parser.add_argument("--skip_build_states",        action="store_true")
    parser.add_argument("--skip_health",              action="store_true")
    parser.add_argument("--skip_eval",                action="store_true")
    parser.add_argument("--skip_analysis",            action="store_true")

    # Force flags
    parser.add_argument("--force_teacher_assignments", action="store_true")
    parser.add_argument("--force_dataset",             action="store_true")
    parser.add_argument("--force_train",               action="store_true")
    parser.add_argument("--force_build_states",        action="store_true")
    parser.add_argument("--force_health",              action="store_true")
    parser.add_argument("--force_neighbors",           action="store_true")
    parser.add_argument("--force_rewards",             action="store_true")
    parser.add_argument("--force_eval",                action="store_true")
    parser.add_argument("--force_analysis",            action="store_true")
    parser.add_argument("--force_all",                 action="store_true")
    parser.add_argument("--dry_run",                   action="store_true")
    args = parser.parse_args()

    if args.force_all:
        for attr in ["force_teacher_assignments", "force_dataset", "force_train",
                     "force_build_states", "force_health", "force_neighbors",
                     "force_rewards", "force_eval", "force_analysis"]:
            setattr(args, attr, True)

    src     = Path(args.source)
    sdir    = Path(args.offline_states_dir)
    out     = Path(args.output)
    scripts = Path(__file__).parent
    out.mkdir(parents=True, exist_ok=True)

    teachers_dir = out / "teachers"
    datasets_dir = out / "datasets"
    models_dir   = out / "models"
    states_dir   = out / "states"

    print(f"\nMVP 4a: Offline-Teacher WRITE Imitation Pipeline")
    print(f"  Source:   {src}")
    print(f"  Teachers: {args.teachers}")
    print(f"  Budgets:  {args.budgets}")
    print(f"  Output:   {out}")
    t0 = time.time()

    # ── Stage 1: teacher assignments ──────────────────────────────────────────
    key1 = "teacher_assignments"
    if args.skip_teacher_assignments:
        print(f"\n[skip] Stage 1 (--skip_teacher_assignments)")
    elif not args.force_teacher_assignments and is_done(out, key1):
        print(f"\n[cached] Stage 1: teacher_assignments")
    else:
        print(f"\n{'='*55}\nStage 1: build_offline_teacher_assignments\n{'='*55}")
        cmd = [
            scripts / "build_offline_teacher_assignments.py",
            "--source",     src,
            "--states_dir", sdir,
            "--teachers",   *args.teachers,
            "--output",     out,
            "--device",     args.device,
            "--seed",       str(args.seed),
        ]
        if args.force_teacher_assignments:
            cmd.append("--force")
        ret = run(cmd, args.dry_run)
        if ret == 0:
            mark_done(out, key1)
        else:
            print(f"Stage 1 failed.  Aborting."); return

    # ── Stage 2: dataset ──────────────────────────────────────────────────────
    key2 = "dataset"
    if args.skip_dataset:
        print(f"\n[skip] Stage 2 (--skip_dataset)")
    elif not args.force_dataset and is_done(out, key2):
        print(f"\n[cached] Stage 2: dataset")
    else:
        print(f"\n{'='*55}\nStage 2: build_write_imitation_dataset\n{'='*55}")
        cmd = [
            scripts / "build_write_imitation_dataset.py",
            "--source",       src,
            "--teachers_dir", teachers_dir,
            "--teachers",     *args.teachers,
            "--output",       out,
            "--state_budget",        str(args.budgets[0]),
            "--buf_frac",            str(args.buf_frac),
            "--ema_cap",             str(args.ema_cap),
            "--state_sim_threshold", str(args.state_sim_threshold),
            "--buf_sim_threshold",   str(args.buf_sim_threshold),
            "--promote_count",       str(args.promote_count),
            "--promote_purity",      str(args.promote_purity),
            "--device",              args.device,
            "--seed",                str(args.seed),
        ]
        if args.force_dataset:
            cmd.append("--force")
        ret = run(cmd, args.dry_run)
        if ret == 0:
            mark_done(out, key2)
        else:
            print(f"Stage 2 failed.  Aborting."); return

    # ── Stage 3: train ────────────────────────────────────────────────────────
    key3 = "train"
    if args.skip_train:
        print(f"\n[skip] Stage 3 (--skip_train)")
    elif not args.force_train and is_done(out, key3):
        print(f"\n[cached] Stage 3: train")
    else:
        print(f"\n{'='*55}\nStage 3: train_write_imitation\n{'='*55}")
        cmd = [
            scripts / "train_write_imitation.py",
            "--datasets_dir", datasets_dir,
            "--teachers",     *args.teachers,
            "--output",       out,
            "--n_epochs",     str(args.n_epochs),
            "--device",       args.device,
            "--seed",         str(args.seed),
        ]
        if args.force_train:
            cmd.append("--force")
        ret = run(cmd, args.dry_run)
        if ret == 0:
            mark_done(out, key3)
        else:
            print(f"Stage 3 failed.  Aborting."); return

    # ── Stage 4: build states ─────────────────────────────────────────────────
    key4 = "build_states"
    if args.skip_build_states:
        print(f"\n[skip] Stage 4 (--skip_build_states)")
    elif not args.force_build_states and is_done(out, key4):
        print(f"\n[cached] Stage 4: build_states")
    else:
        print(f"\n{'='*55}\nStage 4: build_imitation_write_states\n{'='*55}")
        cmd = [
            scripts / "build_imitation_write_states.py",
            "--source",     src,
            "--models_dir", models_dir,
            "--teachers",   *args.teachers,
            "--budgets",    *map(str, args.budgets),
            "--output",     out,
            "--buf_frac",   str(args.buf_frac),
            "--ema_cap",    str(args.ema_cap),
            "--device",     args.device,
            "--seed",       str(args.seed),
        ]
        if args.force_build_states:
            cmd.append("--force")
        ret = run(cmd, args.dry_run)
        if ret == 0:
            mark_done(out, key4)
        else:
            print(f"Stage 4 failed.  Aborting."); return

    # ── Stage 5: eval (health + Q-read) ──────────────────────────────────────
    key5 = "eval"
    any_eval_force = (args.force_health or args.force_neighbors or args.force_rewards
                      or args.force_eval)
    if args.skip_eval:
        print(f"\n[skip] Stage 5 (--skip_eval)")
    elif not any_eval_force and is_done(out, key5):
        print(f"\n[cached] Stage 5: eval")
    else:
        print(f"\n{'='*55}\nStage 5: evaluate_imitation_write_states\n{'='*55}")
        cmd = [
            scripts / "evaluate_imitation_write_states.py",
            "--source",        src,
            "--states_dir",    states_dir,
            "--output",        out,
            "--max_q_samples", str(args.max_q_samples),
            "--n_epochs",      str(args.n_epochs),
            "--device",        args.device,
            "--seed",          str(args.seed),
        ]
        if args.fast_action_grid:
            cmd.append("--fast_action_grid")
        if args.eval_all:
            cmd.append("--eval_all")
        if args.skip_health:
            cmd.append("--skip_health")
        if args.force_health:
            cmd.append("--force_health")
        if args.force_neighbors:
            cmd.append("--force_neighbors")
        if args.force_rewards:
            cmd.append("--force_rewards")
        if args.force_eval:
            cmd.append("--force_eval")
        ret = run(cmd, args.dry_run)
        if ret == 0:
            mark_done(out, key5)
        else:
            print(f"Stage 5 failed (exit={ret}).  Analysis may be incomplete.")

    # ── Stage 6: analyze ──────────────────────────────────────────────────────
    key6 = "analyze"
    if args.skip_analysis:
        print(f"\n[skip] Stage 6 (--skip_analysis)")
    elif not args.force_analysis and is_done(out, key6):
        print(f"\n[cached] Stage 6: analyze")
    else:
        print(f"\n{'='*55}\nStage 6: analyze_write_imitation\n{'='*55}")
        cmd = [
            scripts / "analyze_write_imitation.py",
            "--output",     out,
            "--states_dir", str(states_dir),
            "--models_dir", str(models_dir),
        ]
        if args.force_analysis:
            cmd.append("--force")
        ret = run(cmd, args.dry_run)
        if ret == 0:
            mark_done(out, key6)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*55}")
    print(f"MVP 4a complete  elapsed={elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"Output: {out}")

    report = out / "WRITE_IMITATION_REPORT.md"
    if report.exists():
        print(f"Report: {report}")

    csv = out / "reports" / "write_imitation_results.csv"
    if csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(csv)
            if not df.empty:
                print(f"\nResults ({len(df)} variants evaluated):")
                df_q = df.dropna(subset=["q_state_nll"])
                if not df_q.empty:
                    best = df_q.loc[df_q["q_state_nll"].idxmin()]
                    print(f"  Best Q NLL: {best['q_state_nll']:.4f}  ({best['tag']})")
                    print(f"  delta_vs_mvp3b_best    = {best.get('delta_vs_mvp3b_best', float('nan')):+.4f}")
                    print(f"  delta_vs_full_ds_qmlp  = {best.get('delta_vs_full_ds_qmlp', float('nan')):+.4f}")
                    print(f"  delta_vs_mvp2c_same    = {best.get('delta_vs_mvp2c_same_teacher', float('nan')):+.4f}")
                    n_beat = int((df_q["q_state_nll"] < 2.4204).sum())
                    print(f"\n  Beating MVP 3b best (2.4204): {n_beat}/{len(df_q)}")
        except Exception as e:
            print(f"  [summary failed: {e}]")


if __name__ == "__main__":
    main()

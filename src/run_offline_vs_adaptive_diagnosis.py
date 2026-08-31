"""
run_offline_vs_adaptive_diagnosis.py  --  MVP 3a diagnosis pipeline runner

Usage:
  python branch_a_gate_mvp/src/run_offline_vs_adaptive_diagnosis.py \\
    --source branch_a_gate_mvp/outputs_scale_sweep/scale_200k_seed42 \\
    --offline_states_dir branch_a_gate_mvp/outputs_mvp2b_state_write/states \\
    --offline_q_dir branch_a_gate_mvp/outputs_mvp2c_q_state_read \\
    --adaptive_root branch_a_gate_mvp/outputs_mvp3a_adaptive_write_fast \\
    --output branch_a_gate_mvp/outputs_mvp3a_offline_vs_adaptive_diagnosis \\
    --budgets 10000 25000
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse, subprocess, time
from datetime import datetime


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


def resolve_adaptive_dirs(adaptive_root: Path):
    """Find adaptive states and q_read dirs: prefer fast run, fall back to full."""
    for candidate in [adaptive_root,
                      adaptive_root.parent / (adaptive_root.name + "_fast"),
                      adaptive_root.parent / (adaptive_root.name.replace("_fast", ""))]:
        sdir = candidate / "states"
        qdir = candidate / "q_read"
        if sdir.exists() and qdir.exists() and any(sdir.glob("*.pt")):
            return sdir, qdir, candidate
    # Try parent for fast/full search
    parent = adaptive_root.parent
    for d in [parent / "outputs_mvp3a_adaptive_write_fast",
              parent / "outputs_mvp3a_adaptive_write"]:
        sdir = d / "states"
        qdir = d / "q_read"
        if sdir.exists() and qdir.exists() and any(sdir.glob("*.pt")):
            return sdir, qdir, d
    return None, None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",             required=True)
    parser.add_argument("--offline_states_dir", required=True)
    parser.add_argument("--offline_q_dir",      required=True)
    parser.add_argument("--adaptive_root",       required=True,
                        help="Root of adaptive output dir (e.g. outputs_mvp3a_adaptive_write_fast)")
    parser.add_argument("--adaptive_states_dir", default=None,
                        help="Override: direct path to adaptive states dir")
    parser.add_argument("--adaptive_q_dir",      default=None,
                        help="Override: direct path to adaptive q_read dir")
    parser.add_argument("--output",             required=True)
    parser.add_argument("--budgets",  nargs="+", type=int, default=[10000, 25000])
    parser.add_argument("--max_examples", type=int, default=50)
    parser.add_argument("--skip_inspection", action="store_true")
    parser.add_argument("--force",   action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    src      = Path(args.source)
    out      = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    scripts  = Path(__file__).parent

    # Resolve adaptive dirs
    if args.adaptive_states_dir and args.adaptive_q_dir:
        ada_sdir = Path(args.adaptive_states_dir)
        ada_qdir = Path(args.adaptive_q_dir)
        ada_root = Path(args.adaptive_root)
    else:
        ada_sdir, ada_qdir, ada_root = resolve_adaptive_dirs(Path(args.adaptive_root))
        if ada_sdir is None:
            # Try treating adaptive_root directly
            ada_root = Path(args.adaptive_root)
            ada_sdir = ada_root / "states"
            ada_qdir = ada_root / "q_read"

    print(f"\nMVP 3a Offline vs Adaptive Diagnosis")
    print(f"  Source:          {src}")
    print(f"  Offline states:  {args.offline_states_dir}")
    print(f"  Offline Q-read:  {args.offline_q_dir}")
    print(f"  Adaptive states: {ada_sdir}")
    print(f"  Adaptive Q-read: {ada_qdir}")
    print(f"  Output:          {out}")
    print(f"  Budgets:         {args.budgets}")

    if not ada_sdir or not ada_sdir.exists():
        print(f"\nERROR: adaptive states dir not found: {ada_sdir}")
        print("Pass --adaptive_root pointing to an mvp3a output directory "
              "containing states/ and q_read/ subdirs.")
        return

    t0 = time.time()

    # ── Stage 1: quantitative diagnostics ────────────────────────────────────
    key1 = "diagnose"
    if not args.force and is_done(out, key1):
        print(f"\n[cached] Stage 1: quantitative diagnosis")
    else:
        print(f"\n{'='*55}\nStage 1: diagnose_offline_vs_adaptive_states\n{'='*55}")
        cmd = [
            scripts / "diagnose_offline_vs_adaptive_states.py",
            "--source",              src,
            "--offline_states_dir",  args.offline_states_dir,
            "--offline_q_dir",       args.offline_q_dir,
            "--adaptive_states_dir", ada_sdir,
            "--adaptive_q_dir",      ada_qdir,
            "--output",              out,
            "--budgets",             *map(str, args.budgets),
        ]
        if args.force:
            cmd.append("--force")
        ret = run(cmd, args.dry_run)
        if ret == 0:
            mark_done(out, key1)
        else:
            print(f"Stage 1 failed (exit={ret})")

    # ── Stage 2: qualitative inspection ───────────────────────────────────────
    if args.skip_inspection:
        print(f"\n[skip] Stage 2 (--skip_inspection)")
    else:
        key2 = "inspect"
        if not args.force and is_done(out, key2):
            print(f"\n[cached] Stage 2: inspection")
        else:
            print(f"\n{'='*55}\nStage 2: inspect_offline_adaptive_failures\n{'='*55}")
            cmd = [
                scripts / "inspect_offline_adaptive_failures.py",
                "--source",               src,
                "--offline_states_dir",   args.offline_states_dir,
                "--offline_q_dir",        args.offline_q_dir,
                "--adaptive_states_dir",  ada_sdir,
                "--adaptive_q_dir",       ada_qdir,
                "--output",               out,
                "--budgets",              *map(str, args.budgets),
                "--max_examples",         str(args.max_examples),
            ]
            ret = run(cmd, args.dry_run)
            if ret == 0:
                mark_done(out, key2)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*55}")
    print(f"Diagnosis complete  elapsed={elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"Output: {out}")

    report = out / "OFFLINE_VS_ADAPTIVE_DIAGNOSIS_REPORT.md"
    if report.exists():
        print(f"Report: {report}")

    csv = out / "reports" / "offline_vs_adaptive_summary.csv"
    if csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(csv)
            print(f"\nSummary ({len(df)} groups):")
            for _, r in df.iterrows():
                print(f"  {r['method_type']:<10} {r['tag']:<55} "
                      f"actual_B={int(r['actual_num_states']):<6} "
                      f"usage={r['budget_usage_fraction']:.1%}")
        except Exception:
            pass

    qbeh = out / "reports" / "q_behavior_comparison.csv"
    if qbeh.exists():
        try:
            import pandas as pd
            df = pd.read_csv(qbeh)
            print(f"\nQ NLL comparison:")
            for _, r in df.iterrows():
                print(f"  {r['method_type']:<10} {r['tag']:<55} "
                      f"fixed={r['fixed_state_nll']:.4f}  Q={r['q_state_nll']:.4f}")
        except Exception:
            pass


if __name__ == "__main__":
    main()

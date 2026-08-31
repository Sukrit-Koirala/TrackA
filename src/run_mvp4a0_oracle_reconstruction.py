"""
run_mvp4a0_oracle_reconstruction.py  --  MVP 4a-0: Oracle Sequential Reconstruction

Orchestrates all steps:
  1. build_oracle_teacher_assignments.py   (re-derive datastore labels)
  2. run_oracle_write_reconstruction.py    (sequential stream, build artifacts)
  3. evaluate_oracle_write_reconstruction.py (Q-read evaluation per mode)
  4. analyze_oracle_write_reconstruction.py  (comparison, report)

Usage:
  python src/run_mvp4a0_oracle_reconstruction.py \\
    --source            outputs_scale_sweep/scale_200k_seed42 \\
    --offline_states_dir outputs_mvp2b_state_write/states \\
    --offline_qread_dir outputs_mvp2c_q_state_read \\
    --output            outputs_mvp4a0_oracle_reconstruction \\
    --teacher           minibatch_kmeans \\
    --budget            10000 \\
    --promotion_support 8 \\
    --seed              42 \\
    --device            cuda

Skip flags (if intermediate outputs already exist):
  --skip_assignment_build
  --skip_reconstruction
  --skip_evaluation
  --skip_analysis

Fast mode:
  --fast               uses fast_grid + 300k Q samples
  --max_q_samples N    override Q samples (default 800000)
  --n_epochs N         override Q training epochs (default 30)
"""

import sys
import json
import subprocess
import time
from pathlib import Path

import argparse


def run_step(name: str, cmd: list[str], check: bool = True) -> int:
    print(f"\n{'='*70}")
    print(f"STEP: {name}")
    print(f"CMD:  {' '.join(str(c) for c in cmd)}")
    print(f"{'='*70}")
    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
    print(f"\n[{name}] {status}  ({elapsed:.0f}s)")
    if check and result.returncode != 0:
        print(f"ERROR: step '{name}' failed — aborting.")
        sys.exit(result.returncode)
    return result.returncode


def _load_json(p: Path) -> dict:
    return json.load(open(p)) if p.exists() else {}


def print_summary(out: Path, teacher: str, budget: int) -> None:
    tag   = f"{teacher}_B{budget}"
    ed    = out / "evaluation"
    ad    = out / "analysis"
    rd    = out / "match_diagnostics"

    teacher_m = _load_json(ed / "teacher_reference_metrics.json")
    exact_m   = _load_json(ed / "exact_teacher_prototype_metrics.json")
    seq_m     = _load_json(ed / "sequential_running_mean_metrics.json")
    agg       = _load_json(ad / "aggregate_state_metrics.json")
    recall    = _load_json(rd / "target_recall.json")
    action    = _load_json(out / "oracle_action_stats.json")
    val       = _load_json(out / "teacher_assignments" / "assignment_validation.json")

    def _q(m): return m.get("q_state_nll", "—") if m else "—"
    def _fmt(v): return f"{v:.4f}" if isinstance(v, float) else str(v)

    print("\n" + "="*70)
    print("MVP 4a-0 ORACLE RECONSTRUCTION SUMMARY")
    print("="*70)
    print(f"Teacher:     {tag}")
    print(f"Assignment validation: {'PASS' if val.get('PASS') else 'FAIL/missing'}")
    if action:
        print(f"Stream actions: create={action.get('n_create','?'):,}  "
              f"promote={action.get('n_promote','?'):,}  "
              f"objects={action.get('final_n_objects','?'):,}")

    print("\n--- Prototype similarity ---")
    for mode in ["exact_teacher_prototype", "sequential_running_mean"]:
        if mode in agg:
            cs  = agg[mode].get("proto_cos_sim",    {}).get("mean")
            l2  = agg[mode].get("proto_l2_dist",    {}).get("mean")
            kl  = agg[mode].get("kl_teacher_recon", {}).get("mean")
            cf  = agg[mode].get("count_match_fraction")
            print(f"  {mode[:24]:<24}  cos={_fmt(cs)}  l2={_fmt(l2)}"
                  f"  kl={_fmt(kl)}  cnt_match={_fmt(cf)}")

    print("\n--- MATCH recall (combined) ---")
    comb = recall.get("combined", {})
    for k in [1, 4, 8, 16]:
        v = comb.get(f"recall@{k}")
        if v is not None:
            print(f"  recall@{k}:  {v:.4f}")

    print("\n--- Q-read NLL ---")
    rows = [
        ("teacher_reference",       _q(teacher_m)),
        ("exact_teacher_prototype", _q(exact_m)),
        ("sequential_running_mean", _q(seq_m)),
        ("MVP3b_best (reference)",  2.4204),
        ("full_raw  (reference)",   2.2977),
    ]
    for name, q in rows:
        print(f"  {name:<32}  Q-NLL={_fmt(q) if isinstance(q, float) else q}")

    ref_q = (teacher_m or {}).get("q_state_nll") or 2.3015
    for mode_label, q in [("exact", exact_m), ("sequential", seq_m)]:
        qv = (q or {}).get("q_state_nll")
        if qv is not None:
            gap = qv - ref_q
            verdict = "PASS" if abs(gap) <= 0.02 else ("WARNING" if abs(gap) <= 0.05 else "FAIL")
            print(f"\n  {mode_label} vs teacher: gap={gap:+.4f}  --> {verdict}")

    rp = out / "report.md"
    if rp.exists():
        print(f"\nFull report: {rp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",              required=True)
    ap.add_argument("--offline_states_dir",  required=True)
    ap.add_argument("--offline_qread_dir",   required=True)
    ap.add_argument("--output",              required=True)
    ap.add_argument("--teacher",             default="minibatch_kmeans")
    ap.add_argument("--budget",              type=int, default=10000)
    ap.add_argument("--promotion_support",   type=int, default=8)
    ap.add_argument("--seed",                type=int, default=42)
    ap.add_argument("--device",              default="cuda")
    ap.add_argument("--max_q_samples",       type=int, default=800_000)
    ap.add_argument("--n_epochs",            type=int, default=30)
    ap.add_argument("--fast",                action="store_true")
    ap.add_argument("--force",               action="store_true")
    ap.add_argument("--skip_assignment_build", action="store_true")
    ap.add_argument("--skip_reconstruction",   action="store_true")
    ap.add_argument("--skip_evaluation",       action="store_true")
    ap.add_argument("--skip_analysis",         action="store_true")
    args = ap.parse_args()

    src_dir  = Path(args.source)
    out      = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    src    = Path(__file__).parent

    cfg = {
        "source":              str(src_dir),
        "offline_states_dir":  args.offline_states_dir,
        "offline_qread_dir":   args.offline_qread_dir,
        "output":              str(out),
        "teacher":             args.teacher,
        "budget":              args.budget,
        "promotion_support":   args.promotion_support,
        "seed":                args.seed,
        "device":              args.device,
        "max_q_samples":       args.max_q_samples,
        "n_epochs":            args.n_epochs,
        "fast":                args.fast,
    }
    with open(out / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    force_flag = ["--force"] if args.force else []

    base_args = [
        "--source",   str(src_dir),
        "--output",   str(out),
        "--teacher",  args.teacher,
        "--budget",   str(args.budget),
        "--seed",     str(args.seed),
    ]

    # ── step 1: assignments ───────────────────────────────────────────────────
    if not args.skip_assignment_build:
        run_step("1. build_oracle_teacher_assignments",
                 [python, str(src / "build_oracle_teacher_assignments.py"),
                  *base_args,
                  "--offline_states_dir", args.offline_states_dir,
                  "--device", args.device,
                  *force_flag])
    else:
        print("\n[SKIP] build_oracle_teacher_assignments")

    # ── step 2: reconstruction ────────────────────────────────────────────────
    if not args.skip_reconstruction:
        run_step("2. run_oracle_write_reconstruction",
                 [python, str(src / "run_oracle_write_reconstruction.py"),
                  *base_args,
                  "--offline_states_dir", args.offline_states_dir,
                  "--device",             args.device,
                  "--promotion_support",  str(args.promotion_support),
                  *force_flag])
    else:
        print("\n[SKIP] run_oracle_write_reconstruction")

    # ── step 3: evaluation ────────────────────────────────────────────────────
    if not args.skip_evaluation:
        q_flags = [
            "--offline_qread_dir", args.offline_qread_dir,
            "--device",            args.device,
            "--max_q_samples",     str(args.max_q_samples),
            "--n_epochs",          str(args.n_epochs),
        ]
        if args.fast:
            q_flags.append("--fast")
        run_step("3. evaluate_oracle_write_reconstruction",
                 [python, str(src / "evaluate_oracle_write_reconstruction.py"),
                  *base_args, *q_flags, *force_flag])
    else:
        print("\n[SKIP] evaluate_oracle_write_reconstruction")

    # ── step 4: analysis ──────────────────────────────────────────────────────
    if not args.skip_analysis:
        run_step("4. analyze_oracle_write_reconstruction",
                 [python, str(src / "analyze_oracle_write_reconstruction.py"),
                  "--offline_states_dir", args.offline_states_dir,
                  "--output",             str(out),
                  "--teacher",            args.teacher,
                  "--budget",             str(args.budget),
                  "--seed",               str(args.seed),
                  "--promotion_support",  str(args.promotion_support),
                  *force_flag])
    else:
        print("\n[SKIP] analyze_oracle_write_reconstruction")

    # ── summary ───────────────────────────────────────────────────────────────
    print_summary(out, args.teacher, args.budget)


if __name__ == "__main__":
    main()

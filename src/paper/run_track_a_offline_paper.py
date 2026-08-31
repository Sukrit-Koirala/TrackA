"""
run_track_a_offline_paper.py  --  Track A: Offline Predictive State Codebook Paper

Main pipeline runner.  All stages are sentinel-cached and can be
selectively forced or skipped.

Smoke test:
  python src/paper/run_track_a_offline_paper.py \\
    --source branch_a_gate_mvp/outputs_scale_sweep/scale_200k_seed42 \\
    --output branch_a_gate_mvp/outputs_track_a_offline_paper_smoke \\
    --seeds 42 --budgets 5000 10000 \\
    --methods minibatch_kmeans utility_weighted \\
    --raw_baselines raw_random raw_high_gpt_loss raw_high_gpt_entropy cluster_medoids \\
    --run_equal_budget --run_q_ablation --run_efficiency --run_diagnostics \\
    --fast --fast_action_grid --max_q_samples 200000 --device cuda

Full TinyStories run:
  python src/paper/run_track_a_offline_paper.py \\
    --source branch_a_gate_mvp/outputs_scale_sweep/scale_200k_seed42 \\
    --output branch_a_gate_mvp/outputs_track_a_offline_paper \\
    --seeds 42 123 999 \\
    --budgets 1000 5000 10000 25000 50000 \\
    --methods minibatch_kmeans utility_weighted query_kmeans \\
    --raw_baselines raw_random raw_high_gpt_loss raw_high_gpt_entropy \\
                    raw_token_rarity raw_coverage cluster_medoids \\
    --run_seed_sweep --run_equal_budget --run_efficiency \\
    --run_state_ablations --run_q_ablation --run_diagnostics \\
    --max_q_samples 800000 --device cuda
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse, json, subprocess, time
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent
SRC_DIR    = SCRIPT_DIR.parent

GPT_NLL       = 2.5533
FULL_DS_FIXED = 2.3873
FULL_DS_QMLP  = 2.2977


# ── sentinel helpers ──────────────────────────────────────────────────────────

def sent_path(out: Path, name: str) -> Path:
    p = out / "sentinels"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{name}.done"


def is_done(out: Path, name: str) -> bool:
    return sent_path(out, name).exists()


def mark_done(out: Path, name: str):
    sent_path(out, name).write_text(datetime.now().isoformat())


def run_stage(
    name: str,
    cmd: list,
    out: Path,
    sentinel_name: str,
    force: bool = False,
    skip: bool = False,
) -> bool:
    if skip:
        print(f"  [skip] {name}")
        return True
    if is_done(out, sentinel_name) and not force:
        print(f"  [cached] {name}")
        return True

    print(f"\n{'='*60}")
    print(f"  Stage: {name}")
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    print(f"{'='*60}")
    t0     = time.time()
    result = subprocess.run([str(c) for c in cmd], check=False)
    elapsed = time.time() - t0
    ok = result.returncode == 0
    print(f"  -> exit={result.returncode}  elapsed={elapsed:.0f}s")
    if ok:
        mark_done(out, sentinel_name)
    return ok


def py(script: Path, *args) -> list:
    return [sys.executable, str(script), *[str(a) for a in args]]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Track A Offline Paper — full experiment pipeline")

    # Core paths
    parser.add_argument("--source",         default=None,
                        help="Existing scale_200k datastore dir. "
                             "Required unless --run_extra_dataset or --run_second_model only.")
    parser.add_argument("--output",         required=True)
    parser.add_argument("--offline_states_dir", default=None,
                        help="Existing MVP 2b state files to reuse "
                             "(skips build_state_codebooks for those files)")

    # Experiment scope
    parser.add_argument("--dataset",        default="TinyStories")
    parser.add_argument("--model_name",     default="gpt2")
    parser.add_argument("--datastore_size", type=int, default=200_000)

    parser.add_argument("--seeds",    nargs="+", type=int, default=[42])
    parser.add_argument("--budgets",  nargs="+", type=int,
                        default=[5000, 10000, 25000, 50000])
    parser.add_argument("--methods",  nargs="+",
                        default=["minibatch_kmeans", "utility_weighted"])
    parser.add_argument("--raw_baselines", nargs="+",
                        default=["raw_random", "raw_high_gpt_loss",
                                 "raw_high_gpt_entropy", "cluster_medoids"])

    # Which experiments to run
    parser.add_argument("--run_seed_sweep",    action="store_true")
    parser.add_argument("--run_equal_budget",  action="store_true")
    parser.add_argument("--run_efficiency",    action="store_true")
    parser.add_argument("--run_state_ablations", action="store_true")
    parser.add_argument("--run_q_ablation",    action="store_true")
    parser.add_argument("--run_diagnostics",   action="store_true")
    parser.add_argument("--run_extra_dataset", action="store_true")
    parser.add_argument("--run_second_model",  action="store_true")

    # Skip / force flags
    parser.add_argument("--skip_build_states",  action="store_true")
    parser.add_argument("--skip_raw_baselines", action="store_true")
    parser.add_argument("--skip_q_read",        action="store_true")
    parser.add_argument("--skip_efficiency",     action="store_true")
    parser.add_argument("--skip_ablations",      action="store_true")
    parser.add_argument("--skip_diagnostics",    action="store_true")
    parser.add_argument("--skip_reports",        action="store_true")

    parser.add_argument("--force_build_states",  action="store_true")
    parser.add_argument("--force_raw_baselines", action="store_true")
    parser.add_argument("--force_q_read",        action="store_true")
    parser.add_argument("--force_efficiency",     action="store_true")
    parser.add_argument("--force_ablations",      action="store_true")
    parser.add_argument("--force_diagnostics",    action="store_true")
    parser.add_argument("--force_reports",        action="store_true")

    # Q-read config
    parser.add_argument("--max_q_samples",  type=int, default=800_000)
    parser.add_argument("--n_epochs",       type=int, default=30)
    parser.add_argument("--fast_action_grid", action="store_true")
    parser.add_argument("--fast", action="store_true",
                        help="Shortcut: enable fast_action_grid and cap max_q_samples=200k")

    # Extra dataset / model
    parser.add_argument("--extra_dataset",      default="wikitext2_raw")
    parser.add_argument("--extra_dataset_size", type=int, default=100_000)
    parser.add_argument("--second_model",       default="gpt2-medium")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed",   type=int, default=42)

    args = parser.parse_args()

    # Fast mode shortcut
    if args.fast:
        args.fast_action_grid = True
        if args.max_q_samples > 200_000:
            args.max_q_samples = 200_000

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    (out / "configs").mkdir(exist_ok=True)

    # Save run config
    with open(out / "configs" / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"  Track A: Offline Predictive State Codebook Paper")
    print(f"  Output: {out}")
    print(f"  Seeds:  {args.seeds}")
    print(f"  Budgets:{args.budgets}")
    print(f"{'='*70}")

    t0_all = time.time()

    # ── Determine source directory ────────────────────────────────────────────
    src = Path(args.source) if args.source else None

    # ── For each seed, run core TinyStories pipeline ──────────────────────────
    seed_results = []

    for seed in args.seeds:
        seed_tag = f"seed{seed}" if len(args.seeds) > 1 else "main"
        seed_out  = out / seed_tag if len(args.seeds) > 1 else out
        seed_out.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"  Seed {seed}  ({seed_tag})")
        print(f"{'='*70}")

        # ── Stage 1: Build state codebooks ────────────────────────────────────
        states_dir = seed_out / "states"
        if args.offline_states_dir and seed == args.seeds[0]:
            # Reuse existing MVP 2b states for primary seed
            import shutil
            offline_dir = Path(args.offline_states_dir)
            if offline_dir.exists():
                states_dir.mkdir(parents=True, exist_ok=True)
                for sfp in offline_dir.glob("*.pt"):
                    dest = states_dir / sfp.name
                    if not dest.exists() or args.force_build_states:
                        shutil.copy2(sfp, dest)
                print(f"  Copied existing states from {offline_dir} → {states_dir}")
                mark_done(seed_out, "build_states")

        if src:
            run_stage(
                "build_state_codebooks",
                py(SCRIPT_DIR / "build_state_codebooks.py",
                   "--source", src, "--output", seed_out,
                   "--methods", *args.methods,
                   "--budgets", *map(str, args.budgets),
                   "--device", args.device, "--seed", str(seed),
                   *(["--force"] if args.force_build_states else [])),
                seed_out, "build_states",
                force=args.force_build_states,
                skip=args.skip_build_states,
            )

        # ── Stage 2: Build raw memory baselines ───────────────────────────────
        needs_raw = (args.run_equal_budget or args.run_q_ablation
                     or args.run_efficiency)
        if src and needs_raw:
            run_stage(
                "build_raw_memory_baselines",
                py(SCRIPT_DIR / "build_raw_memory_baselines.py",
                   "--source", src, "--output", seed_out,
                   "--methods", *args.raw_baselines,
                   "--budgets", *map(str, args.budgets),
                   "--device", args.device, "--seed", str(seed),
                   *(["--force"] if args.force_raw_baselines else [])),
                seed_out, "raw_baselines",
                force=args.force_raw_baselines,
                skip=args.skip_raw_baselines,
            )

        if src is None:
            continue   # no source → can't run Q-read stages

        # ── Stage 3: Q-read for state codebooks ───────────────────────────────
        run_stage(
            "evaluate_q_read_states",
            py(SCRIPT_DIR / "evaluate_q_read.py",
               "--source", src,
               "--states_dir", states_dir,
               "--output", seed_out / "q_read" / "states",
               "--methods", *args.methods,
               "--budgets", *map(str, args.budgets),
               "--max_q_samples", str(args.max_q_samples),
               "--n_epochs", str(args.n_epochs),
               "--device", args.device,
               *(["--fast_grid"] if args.fast_action_grid else []),
               *(["--force"] if args.force_q_read else [])),
            seed_out, "q_read_states",
            force=args.force_q_read,
            skip=args.skip_q_read,
        )

        # ── Stage 4: Q-read for raw baselines ────────────────────────────────
        if needs_raw and not args.skip_raw_baselines:
            run_stage(
                "evaluate_q_read_raw",
                py(SCRIPT_DIR / "evaluate_q_read.py",
                   "--source", src,
                   "--states_dir", seed_out / "raw_baselines",
                   "--output", seed_out / "q_read" / "raw_baselines",
                   "--methods", *args.raw_baselines,
                   "--budgets", *map(str, args.budgets),
                   "--max_q_samples", str(args.max_q_samples),
                   "--n_epochs", str(args.n_epochs),
                   "--device", args.device,
                   *(["--fast_grid"] if args.fast_action_grid else []),
                   *(["--force"] if args.force_q_read else [])),
                seed_out, "q_read_raw",
                force=args.force_q_read,
                skip=args.skip_raw_baselines or args.skip_q_read,
            )

        # ── Stage 5: Efficiency ───────────────────────────────────────────────
        if args.run_efficiency:
            run_stage(
                "measure_memory_speed",
                py(SCRIPT_DIR / "measure_memory_speed.py",
                   "--source", src, "--output", seed_out,
                   "--device", args.device,
                   *(["--force"] if args.force_efficiency else [])),
                seed_out, "efficiency",
                force=args.force_efficiency,
                skip=args.skip_efficiency,
            )

        # ── Stage 6: State-object ablations ──────────────────────────────────
        if args.run_state_ablations:
            run_stage(
                "run_state_object_ablations",
                py(SCRIPT_DIR / "run_state_object_ablations.py",
                   "--source", src, "--output", seed_out,
                   "--methods", *args.methods,
                   "--budgets", *map(str, args.budgets[:2]),  # top-2 budgets only
                   "--max_q_samples", str(min(args.max_q_samples, 200_000)),
                   "--device", args.device, "--seed", str(seed),
                   *(["--force"] if args.force_ablations else [])),
                seed_out, "state_ablations",
                force=args.force_ablations,
                skip=args.skip_ablations,
            )

        # ── Stage 7: Q-read ablation (2×2) ───────────────────────────────────
        if args.run_q_ablation:
            run_stage(
                "run_q_read_ablation",
                py(SCRIPT_DIR / "run_q_read_ablation.py",
                   "--output", seed_out,
                   "--budgets", *map(str, args.budgets),
                   *(["--force"] if args.force_reports else [])),
                seed_out, "q_ablation",
                force=args.force_reports,
                skip=args.skip_reports,
            )

        # ── Stage 8: Mechanism diagnostics ───────────────────────────────────
        if args.run_diagnostics:
            run_stage(
                "analyze_mechanisms",
                py(SCRIPT_DIR / "analyze_mechanisms.py",
                   "--source", src, "--output", seed_out,
                   "--device", args.device,
                   *(["--force"] if args.force_diagnostics else [])),
                seed_out, "diagnostics",
                force=args.force_diagnostics,
                skip=args.skip_diagnostics,
            )

        # ── Stage 9: Aggregate results ────────────────────────────────────────
        if not args.skip_reports:
            run_stage(
                "aggregate_paper_results",
                py(SCRIPT_DIR / "aggregate_paper_results.py",
                   "--output", seed_out,
                   "--dataset", args.dataset,
                   "--model", args.model_name,
                   *(["--force"] if args.force_reports else [])),
                seed_out, "aggregate",
                force=args.force_reports,
                skip=args.skip_reports,
            )

        # Collect verdict for this seed
        vp = seed_out / "reports" / "verdicts.json"
        if vp.exists():
            with open(vp) as f:
                v = json.load(f)
            seed_results.append({"seed": seed, **v})

    # ── Seed stability summary (Exp 1) ────────────────────────────────────────
    if args.run_seed_sweep and len(args.seeds) > 1 and seed_results:
        print(f"\n{'='*70}")
        print(f"  Seed Stability Summary (Exp 1)")
        print(f"{'='*70}")

        q_nlls = [r.get("best_q_nll", float("nan")) for r in seed_results]
        valid  = [v for v in q_nlls if v == v]  # filter nan
        if valid:
            import statistics
            mean_q = statistics.mean(valid)
            std_q  = statistics.stdev(valid) if len(valid) > 1 else 0.0
            print(f"  Q NLL:  mean={mean_q:.4f}  std={std_q:.4f}  "
                  f"n={len(valid)}")
            print(f"  vs GPT:      {mean_q - GPT_NLL:+.4f}")
            print(f"  vs full_ds_Q:{mean_q - FULL_DS_QMLP:+.4f}")

        (out / "reports").mkdir(parents=True, exist_ok=True)
        with open(out / "reports" / "seed_stability_results.csv", "w") as f:
            f.write("seed,overall,best_q_nll,delta_vs_full_ds_qmlp,reason\n")
            for r in seed_results:
                qn  = r.get("best_q_nll", float("nan"))
                dq  = (qn - FULL_DS_QMLP) if qn == qn else float("nan")
                f.write(f"{r['seed']},{r.get('overall','?')},{qn:.4f},"
                        f"{dq:.4f},{r.get('reason','')}\n")

    # ── Extra dataset (Exp 7) ─────────────────────────────────────────────────
    if args.run_extra_dataset:
        print(f"\n{'='*70}")
        print(f"  Experiment 7: Extra Dataset ({args.extra_dataset})")
        print(f"{'='*70}")
        extra_out = out / f"dataset_{args.extra_dataset.replace('/', '_')}"
        run_stage(
            "run_dataset_replication",
            py(SCRIPT_DIR / "run_dataset_replication.py",
               "--dataset", args.extra_dataset,
               "--model_name", args.model_name,
               "--output", extra_out,
               "--datastore_size", str(args.extra_dataset_size),
               "--seeds", *map(str, args.seeds[:1]),  # first seed only
               "--budgets", *map(str, args.budgets),
               "--methods", *args.methods,
               "--raw_baselines", *args.raw_baselines[:2],
               "--max_q_samples", str(min(args.max_q_samples, 500_000)),
               "--device", args.device,
               *(["--fast_grid"] if args.fast_action_grid else []),
               *(["--force"] if args.force_reports else [])),
            out, "extra_dataset",
            force=args.force_reports,
        )

    # ── Second model (Exp 8) ─────────────────────────────────────────────────
    if args.run_second_model:
        print(f"\n{'='*70}")
        print(f"  Experiment 8: Second Model ({args.second_model})")
        print(f"{'='*70}")
        model_out = out / f"model_{args.second_model.replace('/', '_')}"
        run_stage(
            "run_model_replication",
            py(SCRIPT_DIR / "run_model_replication.py",
               "--model_name", args.second_model,
               "--dataset", args.dataset,
               "--output", model_out,
               "--datastore_size", str(min(args.datastore_size, 100_000)),
               "--seeds", *map(str, args.seeds[:1]),
               "--budgets", *map(str, args.budgets[-2:]),
               "--methods", *args.methods[:2],
               "--raw_baselines", *args.raw_baselines[:2],
               "--max_q_samples", str(min(args.max_q_samples, 500_000)),
               "--device", args.device,
               *(["--fast_grid"] if args.fast_action_grid else []),
               *(["--force"] if args.force_reports else [])),
            out, "second_model",
            force=args.force_reports,
        )

    # ── Final report ─────────────────────────────────────────────────────────
    if len(args.seeds) == 1 and not args.skip_reports:
        primary_out = out / "main" if len(args.seeds) > 1 else out
        report_path = primary_out / "reports" / "TRACK_A_OFFLINE_PAPER_REPORT.md"
        if report_path.exists():
            print(f"\nReport: {report_path}")
            with open(report_path) as f:
                header = [l.strip() for l in f.readlines()[:20] if l.strip()]
            for line in header:
                print(f"  {line}")

    elapsed = time.time() - t0_all
    print(f"\n{'='*70}")
    print(f"  Track A pipeline complete  elapsed={elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"  Output: {out}")
    print(f"{'='*70}")

    # Final verdict summary
    if seed_results:
        overall = [r.get("overall", "?") for r in seed_results]
        print(f"  Verdicts: {overall}")
        best_q = min((r.get("best_q_nll", float("inf")) for r in seed_results),
                     default=float("nan"))
        if best_q < float("inf"):
            print(f"  Best Q NLL: {best_q:.4f}  "
                  f"(delta vs full_ds_Q: {best_q - FULL_DS_QMLP:+.4f})")


if __name__ == "__main__":
    main()

"""
run_seed_sweep.py

Seed stability sweep for Branch A Gate-Chain MVP.

For each seed:
  1. Generates a config YAML under configs/seed_sweep/seed_<N>.yaml
  2. Runs the clean story-level split pipeline
  3. Saves stdout+stderr to outputs_seed_sweep/logs/seed_<N>.log

After all seeds finish, calls aggregate_seed_results.py.

Usage:
  python src/run_seed_sweep.py --seeds 42 123 999
  python src/run_seed_sweep.py --seeds 42 123 999 2024 7777 --start-step 5
  python src/run_seed_sweep.py --seeds 42 --only-step 9 --dry-run
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

BASE_CONFIG = "configs/clean_storysplit.yaml"

STEPS = [
    (1, "extract_states.py",            "Story-level state extraction"),
    (2, "build_neighbors.py",           "Cosine neighbor search + story metadata"),
    (3, "audit_splits.py",              "Split integrity audit"),
    (4, "baselines.py",                 "Fixed kNN baselines"),
    (5, "train_q_controller.py",        "Q-MLP controllers (full + A/B/C) + Q base data cache"),
    (6, "train_sklearn_controllers.py", "Sklearn Q-controllers"),
    (7, "run_heuristics.py",            "Extended heuristics + cost sweep"),
    (8, "similarity_diagnostics.py",    "Similarity distribution diagnostics"),
    (9, "evaluate_v2.py",               "Unified evaluation table v2"),
]


# ── config generation ──────────────────────────────────────────────────────────

def load_base_config(base_path: str) -> dict:
    with open(base_path) as f:
        return yaml.safe_load(f)


def make_seed_config(base_cfg: dict, seed: int) -> dict:
    cfg = dict(base_cfg)
    out = f"outputs_seed_sweep/seed_{seed}"
    cfg["seed"]              = seed
    cfg["split_seed"]        = seed
    cfg["output_dir"]        = out
    cfg["states_dir"]        = f"{out}/states"
    cfg["neighbors_dir"]     = f"{out}/neighbors"
    cfg["reports_dir"]       = f"{out}/reports"
    cfg["inspection_dir"]    = f"{out}/inspection"
    cfg["inspection_v2_dir"] = f"{out}/inspection_v2"
    cfg["models_dir"]        = f"{out}/models"
    cfg["audit_dir"]         = f"{out}/audit"
    return cfg


def generate_config(seed: int, base_path: str) -> Path:
    cfg_dir = Path("configs/seed_sweep")
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f"seed_{seed}.yaml"

    base_cfg = load_base_config(base_path)
    seed_cfg = make_seed_config(base_cfg, seed)

    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(f"# Auto-generated seed config -- seed {seed}\n")
        yaml.dump(seed_cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    return cfg_path


# ── step runner ────────────────────────────────────────────────────────────────

def _print_condensed(lines: list[str], prefix: str = "    ") -> None:
    """Print first/last 6 lines of output to keep terminal readable."""
    if len(lines) <= 20:
        for ln in lines:
            print(f"{prefix}{ln}")
    else:
        for ln in lines[:6]:
            print(f"{prefix}{ln}")
        print(f"{prefix}... ({len(lines) - 12} lines omitted) ...")
        for ln in lines[-6:]:
            print(f"{prefix}{ln}")


def run_step(
    seed: int,
    step_n: int,
    n_steps: int,
    script: str,
    desc: str,
    config_path: str,
    log_fh,
    dry_run: bool,
) -> bool:
    src_dir = Path(__file__).parent
    cmd = [sys.executable, str(src_dir / script), "--config", config_path]

    header = f"\n[Seed {seed} | Step {step_n}/{n_steps}] {desc}"
    print(header)
    log_fh.write(header + "\n")
    log_fh.write(f"  cmd: {' '.join(cmd)}\n")
    log_fh.flush()

    if dry_run:
        print("    [dry-run: skipping]")
        log_fh.write("  [dry-run: skipping]\n")
        return True

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )

    log_fh.write(result.stdout)
    log_fh.flush()
    _print_condensed(result.stdout.strip().splitlines())

    if result.returncode != 0:
        print(f"    *** STEP FAILED (exit code {result.returncode}) ***")
        log_fh.write(f"\n*** STEP FAILED (exit code {result.returncode}) ***\n")
        return False
    return True


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Seed sweep for Branch A Gate-Chain MVP")
    parser.add_argument("--seeds",       nargs="+", type=int, default=[42, 123, 999],
                        help="Seeds to sweep (default: 42 123 999)")
    parser.add_argument("--base-config", default=BASE_CONFIG,
                        help="Base YAML config to clone per seed")
    parser.add_argument("--start-step",  type=int, default=1,
                        help="Start from this step number (for resuming)")
    parser.add_argument("--only-step",   type=int, default=None,
                        help="Run only this step for all seeds")
    parser.add_argument("--only-seed",   type=int, default=None,
                        help="Run only this seed")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--skip-aggregate", action="store_true",
                        help="Skip the aggregate step at the end")
    args = parser.parse_args()

    seeds = args.seeds
    if args.only_seed is not None:
        seeds = [args.only_seed]

    print(f"\nBranch A Gate-Chain MVP -- Seed Sweep")
    print(f"Seeds:      {seeds}")
    print(f"Base cfg:   {args.base_config}")
    print(f"Start step: {args.start_step}")
    print(f"Dry run:    {args.dry_run}")

    log_dir = Path("outputs_seed_sweep/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    n_steps = len(STEPS)
    failed_seeds: list[int] = []
    completed_seeds: list[int] = []

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"  SEED {seed}")
        print(f"{'='*70}")

        cfg_path = generate_config(seed, args.base_config)
        print(f"  Config: {cfg_path}")

        log_path = log_dir / f"seed_{seed}.log"
        with open(log_path, "w", encoding="utf-8") as log_fh:
            log_fh.write(f"Seed sweep -- seed {seed}\n")
            log_fh.write(f"Config: {cfg_path}\n")
            log_fh.write(f"{'='*70}\n\n")

            seed_ok = True
            for step_n, script, desc in STEPS:
                if args.only_step is not None and step_n != args.only_step:
                    continue
                if step_n < args.start_step:
                    skip_msg = f"  [skipping step {step_n}: {script}]"
                    print(skip_msg)
                    log_fh.write(skip_msg + "\n")
                    continue

                ok = run_step(
                    seed, step_n, n_steps, script, desc,
                    str(cfg_path), log_fh, args.dry_run,
                )
                if not ok:
                    log_fh.write(f"\n*** Seed {seed}: aborted at step {step_n} ***\n")
                    failed_seeds.append(seed)
                    seed_ok = False
                    break

            if seed_ok:
                completed_seeds.append(seed)

        status = "OK" if seed_ok else "FAILED"
        print(f"\n  Seed {seed}: {status}  (log: {log_path})")

    # ── summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Completed seeds: {completed_seeds}")
    if failed_seeds:
        print(f"  Failed seeds:    {failed_seeds}")
    print(f"{'='*70}")

    if args.skip_aggregate or args.dry_run:
        return

    # Run aggregator for all requested seeds (it gracefully skips missing ones)
    src_dir = Path(__file__).parent
    agg_cmd = [
        sys.executable,
        str(src_dir / "aggregate_seed_results.py"),
        "--seeds", *[str(s) for s in args.seeds],
    ]
    print(f"\nRunning aggregator: {' '.join(agg_cmd)}")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    subprocess.run(agg_cmd, env=env)


if __name__ == "__main__":
    main()

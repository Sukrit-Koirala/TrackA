"""
run_scale_sweep.py

Scale robustness sweep for Branch A Gate-Chain MVP.

Tests whether learned gate configuration continues to beat best fixed
(k, tau, alpha) as datastore and query sizes increase.

Steps per scale:
  1. extract_states.py       -- story-level extraction
  2. build_neighbors.py      -- dual-chunked cosine search
  3. audit_splits.py         -- story overlap / same-story neighbor check
  4. baselines.py            -- fixed kNN baselines
  5. train_q_controller.py   -- Q-MLP (full + A/B/C) + Q base data
  6. train_sklearn_controllers.py -- Q-GradientBoosting (+ DT/RF)
  7. run_heuristics.py       -- extended heuristics + cost sweep
  8. evaluate_v2.py          -- unified comparison table

Usage:
  python src/run_scale_sweep.py --seed 42 --scales 50000 100000 200000
  python src/run_scale_sweep.py --seed 42 --scales 50000 100000 200000 500000
  python src/run_scale_sweep.py --seed 42 --scales 200000 --start-step 5
  python src/run_scale_sweep.py --seed 42 --scales 100000 --force

Flags:
  --seed          Random seed (default: 42)
  --scales        Datastore sizes to test
  --start-step    Resume from this step number (default: 1)
  --force         Re-run all steps even if output already exists
  --dry-run       Print commands without executing
  --skip-aggregate  Skip aggregate_scale_results.py at the end
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

BASE_CONFIG = "configs/clean_storysplit.yaml"

# Datastore size -> {controller_train, val, max_q_samples}
SCALE_PARAMS: dict[int, dict] = {
    50_000:   {"ct": 10_000, "val": 10_000, "max_q":   300_000},
    100_000:  {"ct": 20_000, "val": 20_000, "max_q":   500_000},
    200_000:  {"ct": 40_000, "val": 40_000, "max_q":   800_000},
    500_000:  {"ct": 50_000, "val": 50_000, "max_q": 1_000_000},
}

# (step_num, script, description, sentinel_file, sentinel_dir_cfg_key)
# sentinel_file = None means always run; may include {max_k}
STEPS = [
    (1, "extract_states.py",            "Story-level state extraction",
     "val.pt",                       "states_dir"),
    (2, "build_neighbors.py",           "Dual-chunked cosine neighbor search",
     "val_top{max_k}.pt",            "neighbors_dir"),
    (3, "audit_splits.py",              "Split integrity audit",
     "split_overlap_report.json",    "audit_dir"),
    (4, "baselines.py",                 "Fixed kNN baselines",
     "fixed_baselines.json",         "reports_dir"),
    (5, "train_q_controller.py",        "Q-MLP controllers (full + A/B/C)",
     "q_mlp_full.pt",                "models_dir"),
    (6, "train_sklearn_controllers.py", "Sklearn Q-controllers (DT / RF / GB)",
     "q_gradientboosting.joblib",    "models_dir"),
    (7, "run_heuristics.py",            "Extended heuristics + cost sweep",
     "best_heuristics.json",         "reports_dir"),
    (8, "evaluate_v2.py",               "Unified evaluation table v2",
     None,                           None),
]


# ── helpers ────────────────────────────────────────────────────────────────────

def scale_label(n: int) -> str:
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1_000:
        return f"{n // 1_000}k"
    return str(n)


def load_base_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_scale_config(base_cfg: dict, ds_size: int, seed: int) -> dict:
    params = SCALE_PARAMS.get(ds_size)
    if params is None:
        ct  = max(10_000, ds_size // 5)
        val = max(10_000, ds_size // 5)
        max_q = min(1_000_000, ct * 46)
        params = {"ct": ct, "val": val, "max_q": max_q}

    cfg   = dict(base_cfg)
    label = scale_label(ds_size)
    out   = f"outputs_scale_sweep/scale_{label}_seed{seed}"

    cfg["seed"]                       = seed
    cfg["split_seed"]                 = seed
    cfg["datastore_positions"]        = ds_size
    cfg["controller_train_positions"] = params["ct"]
    cfg["val_positions"]              = params["val"]
    cfg["max_q_samples"]              = params["max_q"]

    # Chunked neighbor search -- safe for any datastore size
    cfg["neighbor_query_chunk_size"]  = 256
    cfg["neighbor_ds_chunk_size"]     = 50_000

    cfg["output_dir"]                 = out
    cfg["states_dir"]                 = f"{out}/states"
    cfg["neighbors_dir"]              = f"{out}/neighbors"
    cfg["reports_dir"]                = f"{out}/reports"
    cfg["inspection_dir"]             = f"{out}/inspection"
    cfg["inspection_v2_dir"]          = f"{out}/inspection_v2"
    cfg["models_dir"]                 = f"{out}/models"
    cfg["audit_dir"]                  = f"{out}/audit"
    return cfg


def generate_config(ds_size: int, seed: int, base_path: str) -> Path:
    label    = scale_label(ds_size)
    cfg_dir  = Path("configs/scale_sweep")
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f"scale_{label}_seed{seed}.yaml"

    base_cfg  = load_base_config(base_path)
    scale_cfg = make_scale_config(base_cfg, ds_size, seed)

    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(f"# Auto-generated scale config -- datastore={ds_size:,}, seed={seed}\n")
        yaml.dump(scale_cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return cfg_path


def step_done(sentinel: str | None, dir_key: str | None, cfg: dict) -> bool:
    """Return True if this step's sentinel output file already exists."""
    if sentinel is None:
        return False
    max_k    = cfg.get("max_k", 64)
    filename = sentinel.format(max_k=max_k)
    dir_path = cfg.get(dir_key, "")
    if not dir_path:
        return False
    return (Path(dir_path) / filename).exists()


def _print_condensed(lines: list[str]) -> None:
    if len(lines) <= 20:
        for ln in lines:
            print(f"    {ln}")
    else:
        for ln in lines[:6]:
            print(f"    {ln}")
        print(f"    ... ({len(lines) - 12} lines omitted) ...")
        for ln in lines[-6:]:
            print(f"    {ln}")


def run_step(
    ds_size: int, seed: int,
    step_n: int, n_steps: int,
    script: str, desc: str,
    sentinel: str | None, dir_key: str | None,
    cfg: dict, config_path: str,
    log_fh, dry_run: bool, force: bool,
) -> bool:
    label = scale_label(ds_size)

    if not force and step_done(sentinel, dir_key, cfg):
        msg = f"  [Step {step_n}/{n_steps}] {desc} -- output exists, skipping (--force to rerun)"
        print(msg)
        log_fh.write(msg + "\n")
        return True

    src_dir = Path(__file__).parent
    cmd     = [sys.executable, str(src_dir / script), "--config", config_path]
    header  = f"\n[{label}_seed{seed} | Step {step_n}/{n_steps}] {desc}"
    print(header)
    log_fh.write(header + "\n")
    log_fh.write(f"  cmd: {' '.join(cmd)}\n")
    log_fh.flush()

    if dry_run:
        print("    [dry-run: skipping]")
        log_fh.write("  [dry-run]\n")
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
        log_fh.write(f"\n*** FAILED (exit code {result.returncode}) ***\n")
        return False
    return True


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scale robustness sweep for Branch A Gate-Chain MVP")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--scales",  nargs="+", type=int,
                        default=[50_000, 100_000, 200_000])
    parser.add_argument("--base-config", default=BASE_CONFIG)
    parser.add_argument("--start-step", type=int, default=1)
    parser.add_argument("--force",       action="store_true",
                        help="Rerun all steps even if outputs exist")
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    args = parser.parse_args()

    print(f"\nBranch A Gate-Chain MVP -- Scale Sweep")
    print(f"Seed:       {args.seed}")
    print(f"Scales:     {[scale_label(s) for s in args.scales]}")
    print(f"Start step: {args.start_step}")
    print(f"Force:      {args.force}  Dry-run: {args.dry_run}")

    log_dir = Path("outputs_scale_sweep/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    n_steps   = len(STEPS)
    failed    = []
    succeeded = []

    for ds_size in args.scales:
        label = scale_label(ds_size)
        print(f"\n{'='*70}")
        print(f"  SCALE {label}  (ds={ds_size:,}  seed={args.seed})")
        print(f"{'='*70}")

        cfg_path = generate_config(ds_size, args.seed, args.base_config)
        print(f"  Config: {cfg_path}")

        with open(cfg_path) as f:
            cfg_loaded = yaml.safe_load(f)

        log_path = log_dir / f"scale_{label}_seed{args.seed}.log"
        with open(log_path, "w", encoding="utf-8") as log_fh:
            log_fh.write(f"Scale sweep -- scale={label} ds={ds_size:,} seed={args.seed}\n")
            log_fh.write(f"Config: {cfg_path}\n")
            log_fh.write(f"CT={cfg_loaded.get('controller_train_positions')}  "
                         f"Val={cfg_loaded.get('val_positions')}  "
                         f"max_q={cfg_loaded.get('max_q_samples')}\n")
            log_fh.write(f"{'='*70}\n\n")

            scale_ok = True
            for step_n, script, desc, sentinel, dir_key in STEPS:
                if step_n < args.start_step:
                    skip = f"  [skipping step {step_n}: {script}]"
                    print(skip)
                    log_fh.write(skip + "\n")
                    continue

                ok = run_step(
                    ds_size, args.seed,
                    step_n, n_steps,
                    script, desc,
                    sentinel, dir_key,
                    cfg_loaded, str(cfg_path),
                    log_fh, args.dry_run, args.force,
                )
                if not ok:
                    log_fh.write(f"\n*** Aborted at step {step_n} ***\n")
                    failed.append(ds_size)
                    scale_ok = False
                    break

            if scale_ok:
                succeeded.append(ds_size)

        status = "OK" if scale_ok else "FAILED"
        print(f"\n  Scale {label}_seed{args.seed}: {status}  (log: {log_path})")

    # Summary
    print(f"\n{'='*70}")
    print(f"  Completed: {[scale_label(s) for s in succeeded]}")
    if failed:
        print(f"  Failed:    {[scale_label(s) for s in failed]}")
    print(f"{'='*70}")

    if args.skip_aggregate or args.dry_run:
        return

    src_dir = Path(__file__).parent
    agg_cmd = [
        sys.executable,
        str(src_dir / "aggregate_scale_results.py"),
        "--seed",   str(args.seed),
        "--scales", *[str(s) for s in args.scales],
    ]
    print(f"\nAggregating: {' '.join(agg_cmd)}")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    subprocess.run(agg_cmd, env=env)


if __name__ == "__main__":
    main()

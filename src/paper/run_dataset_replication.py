"""
run_dataset_replication.py  --  Track A Paper  (Experiment 7)

Tests whether the predictive state codebook result generalises beyond
TinyStories.  Runs the full extract → build → Q-read pipeline on
WikiText-2 raw (or other HuggingFace datasets).

Outputs to: <output>/

Pipeline:
  Stage 1: extract hidden states from the target dataset using GPT-2
  Stage 2: build state codebooks (minibatch_kmeans, utility_weighted)
  Stage 3: build raw baselines (raw_random, cluster_medoids)
  Stage 4: Q-read evaluation for states + raw baselines
  Stage 5: efficiency measurement
  Stage 6: mechanism diagnostics
  Stage 7: aggregation

Usage (WikiText-2):
  python run_dataset_replication.py \\
    --dataset wikitext2_raw \\
    --model_name gpt2 \\
    --output branch_a_gate_mvp/outputs_track_a_offline_paper_wikitext2 \\
    --datastore_size 100000 \\
    --seeds 42 123 999 \\
    --budgets 5000 10000 25000 \\
    --device cuda
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse, json, subprocess, time, yaml, math

SCRIPT_DIR = Path(__file__).parent
SRC_DIR    = SCRIPT_DIR.parent


# ── dataset configs ───────────────────────────────────────────────────────────

DATASET_CONFIGS = {
    "wikitext2_raw": {
        "hf_dataset":  "wikitext",
        "hf_config":   "wikitext-2-raw-v1",
        "text_field":  "text",
        "split_train": "train",
        "split_valid": "validation",
        "use_story_split": False,
    },
    "wikitext103_raw": {
        "hf_dataset":  "wikitext",
        "hf_config":   "wikitext-103-raw-v1",
        "text_field":  "text",
        "split_train": "train",
        "split_valid": "validation",
        "use_story_split": False,
    },
    "TinyStories": {
        "hf_dataset":  "roneneldan/TinyStories",
        "hf_config":   None,
        "text_field":  "text",
        "split_train": "train",
        "split_valid": "validation",
        "use_story_split": True,
    },
}


def extract_states_for_dataset(
    dataset: str,
    model_name: str,
    output_dir: Path,
    datastore_size: int,
    seed: int,
    device: str,
    force: bool = False,
) -> Path:
    """
    Run extract_states.py via a temporary YAML config.
    Returns the source directory containing datastore.pt, controller_train.pt, val.pt.
    """
    states_dir = output_dir / "states"
    sentinel   = output_dir / "sentinels" / "extract_done.txt"

    if sentinel.exists() and not force:
        print(f"  [cached] extract_states  ({states_dir})")
        return output_dir

    cfg = DATASET_CONFIGS.get(dataset, DATASET_CONFIGS["wikitext2_raw"])
    states_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sentinels").mkdir(parents=True, exist_ok=True)

    # Build YAML config for extract_states.py
    # Partition: 70% datastore, 20% controller_train, 10% val
    total = datastore_size + int(datastore_size * 0.2) + int(datastore_size * 0.05)
    ds_n  = datastore_size
    ct_n  = max(1000, int(datastore_size * 0.2))
    val_n = max(500,  int(datastore_size * 0.05))

    yaml_cfg = {
        "model_name":     model_name,
        "seq_len":        128,
        "dataset":        cfg["hf_dataset"],
        "dataset_config": cfg.get("hf_config"),
        "text_field":     cfg["text_field"],
        "split_train":    cfg["split_train"],
        "split_valid":    cfg["split_valid"],
        "use_story_split": cfg.get("use_story_split", False),
        "datastore_positions":        ds_n,
        "controller_train_positions": ct_n,
        "val_positions":              val_n,
        "output_dir":     str(output_dir),
        "states_dir":     str(output_dir / "states"),
        "device":         device,
        "seed":           seed,
        "dtype":          "fp16",
    }

    cfg_path = output_dir / "extract_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(yaml_cfg, f)

    print(f"\n  Extracting hidden states  dataset={dataset}  model={model_name}")
    cmd = [sys.executable, str(SRC_DIR / "extract_states.py"),
           "--config", str(cfg_path)]
    if force:
        cmd.append("--force")

    print(f"  $ {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"extract_states failed: {result.returncode}")

    sentinel.write_text(str(time.time()))
    elapsed = time.time() - t0
    print(f"  Extract done  elapsed={elapsed:.0f}s")

    # Compute GPT-only NLL from val split and save extraction summary
    gpt_nll = _compute_gpt_nll(output_dir)
    _save_extraction_summary(output_dir, dataset, model_name, ds_n, ct_n, val_n,
                             seed, gpt_nll)
    return output_dir


def _compute_gpt_nll(output_dir: Path) -> float:
    """Read val.pt and return mean GPT-only NLL for this dataset."""
    import torch
    val_path = output_dir / "states" / "val.pt"
    if not val_path.exists():
        return float("nan")
    try:
        val = torch.load(val_path, weights_only=False)
        return float(val["nll_gpt"].float().mean().item())
    except Exception as e:
        print(f"  [warn] could not compute GPT NLL: {e}")
        return float("nan")


def _save_extraction_summary(output_dir: Path, dataset: str, model_name: str,
                              ds_n: int, ct_n: int, val_n: int,
                              seed: int, gpt_nll: float):
    summary = {
        "dataset": dataset,
        "model_name": model_name,
        "seq_len": 128,
        "datastore_size": ds_n,
        "controller_train_size": ct_n,
        "val_size": val_n,
        "seed": seed,
        "gpt_nll": round(gpt_nll, 6) if math.isfinite(gpt_nll) else None,
    }
    json_path = output_dir / "extraction_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    md_path = output_dir / "extraction_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Extraction Summary: {dataset}\n\n")
        f.write("| Field | Value |\n|---|---|\n")
        for k, v in summary.items():
            f.write(f"| {k} | {v} |\n")
    print(f"  Extraction summary: {json_path}")


def run_stage(name: str, cmd: list, sentinel_path: Path, force: bool) -> bool:
    """Run a subprocess stage with sentinel caching. Returns True on success."""
    if sentinel_path.exists() and not force:
        print(f"  [cached] {name}")
        return True

    print(f"\n  [{name}]")
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    t0 = time.time()
    result = subprocess.run([str(c) for c in cmd], check=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  [FAILED] {name}  exit={result.returncode}  elapsed={elapsed:.0f}s")
        return False
    sentinel_path.parent.mkdir(parents=True, exist_ok=True)
    sentinel_path.write_text(str(time.time()))
    print(f"  [OK] {name}  elapsed={elapsed:.0f}s")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",        default="wikitext2_raw",
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--model_name",     default="gpt2",
                        choices=["gpt2", "gpt2-medium", "gpt2-large",
                                 "EleutherAI/pythia-160m", "EleutherAI/pythia-410m"])
    parser.add_argument("--output",         required=True)
    parser.add_argument("--datastore_size", type=int, default=100_000)
    parser.add_argument("--seeds",          nargs="+", type=int, default=[42])
    parser.add_argument("--budgets",        nargs="+", type=int,
                        default=[5000, 10000, 25000])
    parser.add_argument("--methods",        nargs="+",
                        default=["minibatch_kmeans", "utility_weighted"])
    parser.add_argument("--raw_baselines",  nargs="+",
                        default=["raw_random", "cluster_medoids"])
    parser.add_argument("--max_q_samples",  type=int, default=500_000)
    parser.add_argument("--fast_grid",      action="store_true", default=True)
    parser.add_argument("--no_fast_grid",   action="store_false", dest="fast_grid")
    parser.add_argument("--run_knn_lm",    action="store_true",
                        help="Also run kNN-LM full-datastore baseline for efficiency comparison")
    parser.add_argument("--run_full_raw",  action="store_true",
                        help="Run full N-position raw datastore Q-read for efficiency comparison")
    parser.add_argument("--full_raw_only", action="store_true",
                        help="Skip extraction/build/equal-budget stages; only run full-raw + aggregate")
    parser.add_argument("--device",         default="cuda")
    parser.add_argument("--force",          action="store_true")
    args = parser.parse_args()

    out  = Path(args.output)
    sent = out / "sentinels"
    sent.mkdir(parents=True, exist_ok=True)

    print(f"\nrun_dataset_replication")
    print(f"  Dataset:  {args.dataset}")
    print(f"  Model:    {args.model_name}")
    print(f"  Output:   {out}")
    print(f"  DS size:  {args.datastore_size:,}")

    results = []

    for seed in args.seeds:
        seed_out = out / f"seed{seed}"
        seed_out.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"  Seed {seed}")
        print(f"{'='*60}")

        # Stages 1-4 can be skipped when --full_raw_only (data already on disk)
        if args.full_raw_only:
            src = seed_out
        else:
            # Stage 1: extract
            try:
                src = extract_states_for_dataset(
                    args.dataset, args.model_name, seed_out,
                    args.datastore_size, seed, args.device, args.force,
                )
            except Exception as e:
                print(f"  [FAILED] extract: {e}")
                continue

            # Stage 2: build state codebooks
            ok = run_stage(
                "build_state_codebooks",
                [sys.executable, SCRIPT_DIR / "build_state_codebooks.py",
                 "--source", src, "--output", seed_out,
                 "--methods", *args.methods,
                 "--budgets", *map(str, args.budgets),
                 "--device", args.device, "--seed", str(seed),
                 *(["--force"] if args.force else [])],
                sent / f"seed{seed}_states.done", args.force,
            )
            if not ok:
                continue

            # Stage 3: build raw baselines
            run_stage(
                "build_raw_memory_baselines",
                [sys.executable, SCRIPT_DIR / "build_raw_memory_baselines.py",
                 "--source", src, "--output", seed_out,
                 "--methods", *args.raw_baselines,
                 "--budgets", *map(str, args.budgets),
                 "--device", args.device, "--seed", str(seed),
                 *(["--force"] if args.force else [])],
                sent / f"seed{seed}_raw.done", args.force,
            )

            # Stage 4: Q-read for states
            run_stage(
                "evaluate_q_read_states",
                [sys.executable, SCRIPT_DIR / "evaluate_q_read.py",
                 "--source", src,
                 "--states_dir", seed_out / "states",
                 "--output", seed_out / "q_read" / "states",
                 "--max_q_samples", str(args.max_q_samples),
                 "--device", args.device,
                 *(["--fast_grid"] if args.fast_grid else []),
                 *(["--force"] if args.force else [])],
                sent / f"seed{seed}_qread_states.done", args.force,
            )

            # Stage 4b: Q-read for raw baselines
            run_stage(
                "evaluate_q_read_raw",
                [sys.executable, SCRIPT_DIR / "evaluate_q_read.py",
                 "--source", src,
                 "--states_dir", seed_out / "raw_baselines",
                 "--output", seed_out / "q_read" / "raw_baselines",
                 "--max_q_samples", str(args.max_q_samples),
                 "--device", args.device,
                 *(["--fast_grid"] if args.fast_grid else []),
                 *(["--force"] if args.force else [])],
                sent / f"seed{seed}_qread_raw.done", args.force,
            )

        # Stage 4c: full raw datastore Q-read (optional)
        full_raw_result = None
        if args.run_full_raw or args.full_raw_only:
            from run_full_raw_q_read import run_full_raw
            run_stage(
                "full_raw_q_read",
                [sys.executable, SCRIPT_DIR / "run_full_raw_q_read.py",
                 "--source",        seed_out,
                 "--output",        seed_out / "q_read" / "full_raw",
                 "--max_q_samples", str(args.max_q_samples),
                 "--device",        args.device,
                 *(["--fast_grid"] if args.fast_grid else []),
                 *(["--force"]     if args.force     else [])],
                sent / f"seed{seed}_full_raw.done", args.force,
            )
            fr_summary = seed_out / "q_read" / "full_raw" / "evaluate_q_read_summary.json"
            if fr_summary.exists():
                with open(fr_summary) as f:
                    fr = json.load(f)
                rr = fr.get("results", [{}])[0] if fr.get("results") else {}
                full_raw_result = {
                    "full_raw_fixed_nll":  rr.get("best_fixed_nll"),
                    "full_raw_q_nll":      rr.get("q_state_nll"),
                    "full_raw_oracle_nll": rr.get("oracle_nll"),
                    "full_raw_n":          rr.get("budget"),
                }

        # Stage 4d: kNN-LM full-datastore baseline (optional)
        knn_lm_result = None
        if args.run_knn_lm:
            run_stage(
                "knn_lm_baseline",
                [sys.executable, SCRIPT_DIR / "run_knn_lm_baseline.py",
                 "--datastore", src / "states" / "datastore.pt",
                 "--val",       src / "states" / "val.pt",
                 "--output",    seed_out / "knn_lm",
                 "--device",    args.device],
                sent / f"seed{seed}_knn_lm.done", args.force,
            )
            best_json = seed_out / "knn_lm" / "knn_lm_best.json"
            if best_json.exists():
                with open(best_json) as f:
                    knn_lm_result = json.load(f)

        # Collect GPT NLL — from summary if available, else compute directly from val.pt
        gpt_nll = float("nan")
        extr_summary = seed_out / "extraction_summary.json"
        if extr_summary.exists():
            with open(extr_summary) as f:
                gpt_nll = json.load(f).get("gpt_nll") or float("nan")
        if not math.isfinite(gpt_nll):
            gpt_nll = _compute_gpt_nll(seed_out)
            if math.isfinite(gpt_nll):
                _save_extraction_summary(seed_out, args.dataset, args.model_name,
                                         args.datastore_size,
                                         max(1000, int(args.datastore_size * 0.2)),
                                         max(500,  int(args.datastore_size * 0.05)),
                                         seed, gpt_nll)

        # Stage 5: aggregate (pass dataset-specific GPT NLL)
        agg_cmd = [sys.executable, SCRIPT_DIR / "aggregate_paper_results.py",
                   "--output", seed_out,
                   "--dataset", args.dataset,
                   "--model", args.model_name,
                   *(["--force"] if args.force else [])]
        if math.isfinite(gpt_nll):
            agg_cmd += ["--gpt_nll", str(round(gpt_nll, 6))]
        run_stage("aggregate", agg_cmd, sent / f"seed{seed}_aggregate.done", args.force)

        # Collect verdict
        verdict_path = seed_out / "reports" / "verdicts.json"
        if verdict_path.exists():
            with open(verdict_path) as f:
                v = json.load(f)
            results.append({"seed": seed, "gpt_nll": gpt_nll,
                            "knn_lm": knn_lm_result,
                            "full_raw": full_raw_result,
                            **v})
            print(f"\n  Seed {seed} verdict: {v.get('overall')}  ({v.get('reason','')})")
            if full_raw_result:
                print(f"  Full-raw Q NLL:  {full_raw_result['full_raw_q_nll']:.4f}  "
                      f"fixed={full_raw_result['full_raw_fixed_nll']:.4f}  "
                      f"oracle={full_raw_result['full_raw_oracle_nll']:.4f}")
            if knn_lm_result:
                print(f"  kNN-LM NLL:  {knn_lm_result['best_nll']:.4f}  "
                      f"(k={knn_lm_result['best_k']}, λ={knn_lm_result['best_lambda']})")

    # ── summary ───────────────────────────────────────────────────────────────
    if results:
        print(f"\n{'='*60}")
        print(f"  Dataset replication summary: {args.dataset}  {args.model_name}")
        print(f"{'='*60}")
        for r in results:
            print(f"  Seed {r['seed']}: {r.get('overall')}  "
                  f"best_q={r.get('best_q_nll', float('nan')):.4f}  "
                  f"{r.get('reason','')}")

        with open(out / "replication_summary.json", "w") as f:
            json.dump({
                "dataset": args.dataset, "model": args.model_name,
                "datastore_size": args.datastore_size,
                "results_by_seed": results,
            }, f, indent=2)

        _print_wikitext2_summary(results, args.dataset)
        _write_wikitext2_verdict(out, results, args.dataset, args.model_name,
                                 args.datastore_size)

    print("\nDone.")


# ── WikiText-2 specific reporting ─────────────────────────────────────────────

def _wikitext2_verdict(results: list[dict]) -> str:
    """
    Compute verdict for cross-dataset replication.

    Tiers (in order):
      STRONG_GREEN  beats GPT, 100% raw win rate, matches/beats full raw Q (≤+0.02)
      GREEN         beats GPT, ≥80% raw win rate, close to full raw Q (≤+0.08)
      GREEN_EQUAL_BUDGET_FULL_RAW_PENDING  beats GPT, ≥80% raw, full raw not yet run
      YELLOW        beats GPT, ≥50% raw, but full raw clearly better
      RED           does not beat GPT or <50% raw win rate
    """
    if not results:
        return "RED"

    # Recompute beats_gpt from actual NLL values — don't trust cached verdicts.json
    # which may have been produced with wrong constants.
    def _beats_gpt(r: dict) -> bool:
        gn = r.get("gpt_nll", float("nan"))
        bq = r.get("best_q_nll", float("inf"))
        if math.isfinite(gn) and math.isfinite(bq):
            return bq < gn
        return bool(r.get("beats_gpt_all", False))

    beats_gpt_all = all(_beats_gpt(r) for r in results)
    raw_fracs     = [r.get("beats_raw_frac", float("nan")) for r in results]
    valid_fracs   = [f for f in raw_fracs if math.isfinite(f)]
    mean_raw_frac = sum(valid_fracs) / len(valid_fracs) if valid_fracs else 0.0

    if not beats_gpt_all:
        return "RED"
    if mean_raw_frac < 0.5:
        return "RED"

    # Best state Q NLL across seeds
    best_qs = [r.get("best_q_nll", float("inf")) for r in results]
    best_q  = min(q for q in best_qs if math.isfinite(q)) if any(math.isfinite(q) for q in best_qs) else float("inf")

    # Full raw Q NLL — average across seeds that have it
    fr_nlls = [r["full_raw"]["full_raw_q_nll"]
                for r in results
                if r.get("full_raw") and r["full_raw"].get("full_raw_q_nll") is not None]
    has_full_raw = len(fr_nlls) > 0
    mean_fr_q    = sum(fr_nlls) / len(fr_nlls) if fr_nlls else float("nan")

    delta_vs_full_raw = best_q - mean_fr_q if has_full_raw else float("nan")

    if mean_raw_frac >= 0.8:
        if not has_full_raw:
            return "GREEN_EQUAL_BUDGET_FULL_RAW_PENDING"
        if delta_vs_full_raw <= 0.02:          # matches or beats full raw
            return "STRONG_GREEN"
        if delta_vs_full_raw <= 0.08:          # close to full raw
            return "GREEN"
        return "YELLOW"                        # clearly worse than full raw

    return "YELLOW"


def _print_wikitext2_summary(results: list[dict], dataset: str):
    verdict    = _wikitext2_verdict(results)
    best_q     = min((r.get("best_q_nll", float("inf")) for r in results), default=float("nan"))
    gpt_nlls   = [r.get("gpt_nll", float("nan")) for r in results]
    gpt_nll    = next((v for v in gpt_nlls if math.isfinite(v)), float("nan"))
    raw_fracs  = [r.get("beats_raw_frac", float("nan")) for r in results]
    valid_fracs = [f for f in raw_fracs if math.isfinite(f)]
    mean_frac  = sum(valid_fracs) / len(valid_fracs) if valid_fracs else float("nan")

    # Full raw stats
    fr_results = [r["full_raw"] for r in results if r.get("full_raw")]
    fr_q_nlls  = [r["full_raw_q_nll"]    for r in fr_results if r.get("full_raw_q_nll") is not None]
    fr_fix_nlls= [r["full_raw_fixed_nll"] for r in fr_results if r.get("full_raw_fixed_nll") is not None]
    fr_q_nll   = sum(fr_q_nlls)   / len(fr_q_nlls)   if fr_q_nlls  else float("nan")
    fr_fix_nll = sum(fr_fix_nlls) / len(fr_fix_nlls) if fr_fix_nlls else float("nan")
    delta_fr   = best_q - fr_q_nll if math.isfinite(fr_q_nll) and math.isfinite(best_q) else float("nan")

    # Best equal-budget raw Q (lowest across all seeds / methods / budgets)
    eb_nlls = [r.get("best_raw_q_nll", float("nan")) for r in results]
    eb_q    = min((v for v in eb_nlls if math.isfinite(v)), default=float("nan"))

    print(f"\n{'='*60}")
    print(f"  {dataset} Full Raw Comparison")
    print(f"  {'─'*40}")
    print(f"  GPT-only NLL:              {gpt_nll:.4f}" if math.isfinite(gpt_nll) else "  GPT-only NLL:              —")
    print(f"  Full raw fixed NLL:        {fr_fix_nll:.4f}" if math.isfinite(fr_fix_nll) else "  Full raw fixed NLL:        — (not run)")
    print(f"  Full raw Q NLL:            {fr_q_nll:.4f}"   if math.isfinite(fr_q_nll)   else "  Full raw Q NLL:            — (not run)")
    print(f"  Best state Q NLL:          {best_q:.4f}"     if math.isfinite(best_q)     else "  Best state Q NLL:          —")
    print(f"  Best equal-budget raw Q:   {eb_q:.4f}"       if math.isfinite(eb_q)       else "  Best equal-budget raw Q:   —")
    print(f"  State Δ vs GPT:            {best_q - gpt_nll:+.4f}" if math.isfinite(gpt_nll) and math.isfinite(best_q) else "  State Δ vs GPT:            —")
    print(f"  State Δ vs equal-bud raw:  {best_q - eb_q:+.4f}"    if math.isfinite(eb_q)    and math.isfinite(best_q) else "  State Δ vs equal-bud raw:  —")
    print(f"  State Δ vs full raw Q:     {delta_fr:+.4f}"          if math.isfinite(delta_fr)                          else "  State Δ vs full raw Q:     — (not run)")
    print(f"  Verdict:                   {verdict}")
    print(f"{'='*60}")

    if verdict == "STRONG_GREEN":
        print("  State matches/beats full raw Q at fraction of storage. Run 200k datastore next.")
    elif verdict in ("GREEN", "GREEN_EQUAL_BUDGET_FULL_RAW_PENDING"):
        print("  Next: run 200k datastore, then second model (GPT-2 medium).")
    elif verdict == "YELLOW":
        print("  Next: increase budget range or debug weak budgets before second model.")
    else:
        print("  Next: debug failure — check if state Q NLL > GPT NLL on this dataset.")


def _write_wikitext2_verdict(out: Path, results: list[dict], dataset: str,
                              model_name: str, datastore_size: int):
    verdict = _wikitext2_verdict(results)
    best_q  = min((r.get("best_q_nll", float("inf")) for r in results),
                  default=float("nan"))
    gpt_nlls = [r.get("gpt_nll", float("nan")) for r in results]
    gpt_nll  = next((v for v in gpt_nlls if math.isfinite(v)), float("nan"))
    raw_fracs = [r.get("beats_raw_frac", float("nan")) for r in results]
    valid_fracs = [f for f in raw_fracs if math.isfinite(f)]
    mean_frac = sum(valid_fracs) / len(valid_fracs) if valid_fracs else float("nan")

    # kNN-LM: best across seeds
    knn_results = [r["knn_lm"] for r in results if r.get("knn_lm")]
    knn_nll  = min(r["best_nll"] for r in knn_results) if knn_results else None
    knn_size = knn_results[0]["storage_mb"] if knn_results else None

    # Full raw: average across seeds
    fr_results = [r["full_raw"] for r in results if r.get("full_raw")]
    fr_q_vals  = [r["full_raw_q_nll"]    for r in fr_results if r.get("full_raw_q_nll") is not None]
    fr_fix_vals= [r["full_raw_fixed_nll"] for r in fr_results if r.get("full_raw_fixed_nll") is not None]
    fr_q_nll   = sum(fr_q_vals)  / len(fr_q_vals)  if fr_q_vals  else None
    fr_fix_nll = sum(fr_fix_vals) / len(fr_fix_vals) if fr_fix_vals else None

    # Object ratio: best state budget vs full datastore
    budgets_used = [25000, datastore_size // 4]
    best_budget  = max(b for b in budgets_used if b < datastore_size) if any(b < datastore_size for b in budgets_used) else datastore_size
    obj_ratio    = f"{best_budget:,} / {datastore_size:,} = {best_budget/datastore_size:.0%}"

    delta_st_fr = (best_q - fr_q_nll) if fr_q_nll is not None and math.isfinite(best_q) else None

    rep_dir = out / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset": dataset, "model": model_name,
        "datastore_size": datastore_size,
        "gpt_nll": round(gpt_nll, 6) if math.isfinite(gpt_nll) else None,
        "knn_lm_best_nll": round(knn_nll, 6) if knn_nll is not None else None,
        "knn_lm_storage_mb": knn_size,
        "full_raw_q_nll": round(fr_q_nll, 6) if fr_q_nll is not None else None,
        "full_raw_fixed_nll": round(fr_fix_nll, 6) if fr_fix_nll is not None else None,
        "best_state_q_nll": round(best_q, 6) if math.isfinite(best_q) else None,
        "state_win_rate_vs_raw": round(mean_frac, 4) if math.isfinite(mean_frac) else None,
        "verdict": verdict,
        "results_by_seed": results,
    }
    with open(rep_dir / "replication_verdict.json", "w") as f:
        json.dump(summary, f, indent=2)

    _fmt = lambda v: f"{v:.4f}" if isinstance(v, float) and math.isfinite(v) else "—"

    ts_gpt   = 2.5533
    ts_fr_q  = 2.2977
    ts_best  = 2.2873
    ts_ratio = "25%"

    sep = "-" * (len(dataset) + 2)
    lines = [
        f"# TinyStories vs {dataset} Comparison\n",
        f"Model: {model_name}  |  Datastore: {datastore_size:,}\n",
        "| Metric | TinyStories | " + dataset + " |",
        "|--------|------------|" + sep + "|",
        f"| GPT-only NLL | {ts_gpt:.4f} | {_fmt(gpt_nll)} |",
        f"| Full raw fixed NLL | — | " + (f"{fr_fix_nll:.4f}" if fr_fix_nll is not None else "— (not run)") + " |",
        f"| Full raw Q NLL | {ts_fr_q:.4f} | " + (f"{fr_q_nll:.4f}" if fr_q_nll is not None else "— (not run)") + " |",
        f"| Best state Q NLL | {ts_best:.4f} | {_fmt(best_q)} |",
        f"| Δ state vs full raw Q | {ts_best - ts_fr_q:+.4f} | " + (f"{delta_st_fr:+.4f}" if delta_st_fr is not None else "— (not run)") + " |",
        f"| State win rate vs equal-budget raw | 100% | {_fmt(mean_frac * 100) + '%' if math.isfinite(mean_frac) else '—'} |",
        f"| Object ratio (best state / full raw) | {ts_ratio} | {obj_ratio} |",
        f"| Cross-dataset verdict | GREEN | {verdict} |",
        "",
    ]

    if verdict == "STRONG_GREEN":
        lines.append(
            "**Conclusion:** The predictive state codebook result generalises to general-domain "
            f"text. At ≤{best_budget/datastore_size:.0%} of full-datastore storage, state "
            f"codebooks match or beat the full {datastore_size:,}-position Q-read on {dataset}.")
    elif verdict == "GREEN":
        lines.append(
            "**Conclusion:** The predictive state codebook result is not limited to TinyStories. "
            f"State codebooks beat GPT-only and equal-budget raw baselines on {dataset} with "
            f"{best_budget/datastore_size:.0%} of full-datastore storage.")
    elif verdict == "GREEN_EQUAL_BUDGET_FULL_RAW_PENDING":
        lines.append(
            "**Conclusion (pending):** State codebooks beat GPT-only and equal-budget raw "
            f"baselines on {dataset}. Full-datastore comparison not yet run — "
            "re-run with --run_full_raw to complete the efficiency claim.")
    elif verdict == "YELLOW":
        lines.append(
            "**Conclusion:** The method works strongly on TinyStories and partially transfers "
            f"to {dataset}, but general-domain robustness remains unresolved.")
    else:
        lines.append(
            "**Conclusion:** The current method may exploit the repetitive/predictable structure "
            "of TinyStories. The paper should be framed as controlled-domain evidence unless "
            "improved.")

    md_path = rep_dir / "TINYSTORIES_VS_WIKITEXT2_COMPARISON.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Comparison report: {md_path}")


if __name__ == "__main__":
    main()

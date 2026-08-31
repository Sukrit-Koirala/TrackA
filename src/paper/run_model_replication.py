"""
run_model_replication.py  --  Track A Paper  (Experiment 8)

Tests whether the result depends on GPT-2 small's hidden-state geometry
by running the full pipeline on a second backbone model.

Supported models:
  gpt2-medium          (1024-dim hidden)
  EleutherAI/pythia-160m  (768-dim hidden)
  EleutherAI/pythia-410m  (1024-dim hidden)

Usage:
  python run_model_replication.py \\
    --model_name gpt2-medium \\
    --dataset TinyStories \\
    --output outputs_track_a_offline_paper_gpt2_medium \\
    --datastore_size 100000 \\
    --seeds 42 123 999 \\
    --budgets 5000 10000 25000 \\
    --methods minibatch_kmeans utility_weighted query_kmeans \\
    --raw_baselines raw_random cluster_medoids raw_high_gpt_entropy raw_high_gpt_loss raw_coverage \\
    --run_full_raw \\
    --fast_grid \\
    --max_q_samples 500000 \\
    --device cuda
"""

import sys
import math
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse, time

from paper.run_dataset_replication import (
    run_stage, extract_states_for_dataset, SCRIPT_DIR,
    _compute_gpt_nll, _save_extraction_summary,
)

# TinyStories GPT-2 small reference (for comparison table)
_TS_GPT2_SMALL = {
    "gpt_nll":        2.5533,
    "full_raw_q_nll": 2.2977,
    "best_state_q":   2.2873,
    "datastore_size": 200_000,
}


# ── verdict logic (same tier system as _wikitext2_verdict) ────────────────────

def _second_model_verdict(results: list[dict]) -> str:
    """
    STRONG_GREEN  state beats GPT + 100% raw + Δ ≤ +0.02 vs full_raw Q
    GREEN         state beats GPT + ≥80% raw + Δ ≤ +0.08 vs full_raw Q
    GREEN_FULL_RAW_PENDING  beats GPT + ≥80% raw + full_raw not yet run
    YELLOW        beats GPT but raw or full_raw gap is large
    RED           does not beat GPT or <50% raw win rate
    """
    if not results:
        return "RED"

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

    if not beats_gpt_all or mean_raw_frac < 0.5:
        return "RED"

    best_qs = [r.get("best_q_nll", float("inf")) for r in results]
    best_q  = min((q for q in best_qs if math.isfinite(q)), default=float("inf"))

    fr_nlls = [r["full_raw"]["full_raw_q_nll"]
               for r in results
               if r.get("full_raw") and r["full_raw"].get("full_raw_q_nll") is not None]
    has_fr   = len(fr_nlls) > 0
    mean_fr  = sum(fr_nlls) / len(fr_nlls) if fr_nlls else float("nan")
    delta_fr = best_q - mean_fr if has_fr else float("nan")

    if mean_raw_frac >= 0.8:
        if not has_fr:
            return "GREEN_FULL_RAW_PENDING"
        if delta_fr <= 0.02:
            return "STRONG_GREEN"
        if delta_fr <= 0.08:
            return "GREEN"
        return "YELLOW"

    return "YELLOW"


def _print_second_model_summary(results: list[dict], model_name: str, dataset: str,
                                 datastore_size: int):
    verdict    = _second_model_verdict(results)
    best_q     = min((r.get("best_q_nll", float("inf")) for r in results), default=float("nan"))
    gpt_nlls   = [r.get("gpt_nll", float("nan")) for r in results]
    gpt_nll    = next((v for v in gpt_nlls if math.isfinite(v)), float("nan"))
    raw_fracs  = [r.get("beats_raw_frac", float("nan")) for r in results]
    valid_f    = [f for f in raw_fracs if math.isfinite(f)]
    mean_frac  = sum(valid_f) / len(valid_f) if valid_f else float("nan")

    # Best equal-budget raw Q
    eb_nlls = [r.get("best_raw_q_nll", float("nan")) for r in results]
    eb_q    = min((v for v in eb_nlls if math.isfinite(v)), default=float("nan"))

    # Full raw stats
    fr_results  = [r["full_raw"] for r in results if r.get("full_raw")]
    fr_q_vals   = [r["full_raw_q_nll"]    for r in fr_results if r.get("full_raw_q_nll")    is not None]
    fr_fix_vals = [r["full_raw_fixed_nll"] for r in fr_results if r.get("full_raw_fixed_nll") is not None]
    fr_q_nll    = sum(fr_q_vals)   / len(fr_q_vals)   if fr_q_vals   else float("nan")
    fr_fix_nll  = sum(fr_fix_vals) / len(fr_fix_vals) if fr_fix_vals else float("nan")

    # Best method/budget
    best_seed = min(results, key=lambda r: r.get("best_q_nll", float("inf")), default={})
    best_method = best_seed.get("best_method", "—")
    best_budget = best_seed.get("best_budget", "—")

    def _f(v): return f"{v:.4f}" if math.isfinite(v) else "—"

    print(f"\n{'='*60}")
    print(f"  Second Model Track A Result")
    print(f"  {'─'*40}")
    print(f"  Dataset:                   {dataset}")
    print(f"  Model:                     {model_name}")
    print(f"  Datastore size:            {datastore_size:,}")
    print(f"  GPT-only NLL:              {_f(gpt_nll)}")
    print(f"  Full raw fixed NLL:        " + (_f(fr_fix_nll) if math.isfinite(fr_fix_nll) else "— (not run)"))
    print(f"  Full raw Q NLL:            " + (_f(fr_q_nll)   if math.isfinite(fr_q_nll)   else "— (not run)"))
    print(f"  Best state Q NLL:          {_f(best_q)}")
    print(f"  Best state method/budget:  {best_method} / {best_budget}")
    print(f"  Best equal-budget raw Q:   {_f(eb_q)}")
    print(f"  State win rate vs raw:     {mean_frac:.0%}" if math.isfinite(mean_frac) else "  State win rate vs raw:     —")
    print(f"  State Δ vs GPT:            {best_q - gpt_nll:+.4f}" if math.isfinite(gpt_nll) and math.isfinite(best_q) else "  State Δ vs GPT:            —")
    print(f"  State Δ vs equal-bud raw:  {best_q - eb_q:+.4f}"    if math.isfinite(eb_q)    and math.isfinite(best_q) else "  State Δ vs equal-bud raw:  —")
    print(f"  State Δ vs full raw Q:     {best_q - fr_q_nll:+.4f}" if math.isfinite(fr_q_nll) and math.isfinite(best_q) else "  State Δ vs full raw Q:     — (not run)")
    obj_ratio = f"{best_budget:,} / {datastore_size:,} = {best_budget/datastore_size:.0%}" if isinstance(best_budget, int) else "—"
    print(f"  Object ratio:              {obj_ratio}")
    print(f"  Verdict:                   {verdict}")
    print(f"{'='*60}")

    print(f"\n  Interpretation:")
    gpt2_small_specific = "no" if verdict in ("STRONG_GREEN", "GREEN") else "yes"
    second_model_pass   = "yes" if verdict in ("STRONG_GREEN", "GREEN") else ("mixed" if verdict == "YELLOW" else "no")
    print(f"    GOOD:    result is GPT-2-small-specific? {gpt2_small_specific}")
    print(f"    SERIOUS: second-model replication passed? {second_model_pass}")

    if verdict == "STRONG_GREEN":
        print(f"    Next:    Paper upgrades to serious. Run WikiText-2 + {model_name} next.")
    elif verdict == "GREEN":
        print(f"    Next:    Model robustness supported. Frame full-raw matching carefully.")
    elif verdict == "GREEN_FULL_RAW_PENDING":
        print(f"    Next:    Re-run with --run_full_raw to complete efficiency claim.")
    elif verdict == "YELLOW":
        print(f"    Next:    Core effect survives but weaker. May be tuned to GPT-2 small geometry.")
    else:
        print(f"    Next:    Result is model-specific. Keep paper framed around GPT-2 small.")


def _write_comparison_report(out: Path, results: list[dict], model_name: str,
                               dataset: str, datastore_size: int):
    verdict   = _second_model_verdict(results)
    best_q    = min((r.get("best_q_nll", float("inf")) for r in results), default=float("nan"))
    gpt_nlls  = [r.get("gpt_nll", float("nan")) for r in results]
    gpt_nll   = next((v for v in gpt_nlls if math.isfinite(v)), float("nan"))
    raw_fracs = [r.get("beats_raw_frac", float("nan")) for r in results]
    valid_f   = [f for f in raw_fracs if math.isfinite(f)]
    mean_frac = sum(valid_f) / len(valid_f) if valid_f else float("nan")

    fr_results  = [r["full_raw"] for r in results if r.get("full_raw")]
    fr_q_vals   = [r["full_raw_q_nll"] for r in fr_results if r.get("full_raw_q_nll") is not None]
    fr_q_nll    = sum(fr_q_vals) / len(fr_q_vals) if fr_q_vals else None

    best_budget = min(max(b for b in [datastore_size // 4, 25000] if b < datastore_size),
                      default=datastore_size)
    obj_ratio_m = f"{best_budget:,} / {datastore_size:,} = {best_budget/datastore_size:.0%}"

    _fmt = lambda v: f"{v:.4f}" if isinstance(v, float) and math.isfinite(v) else "—"

    ref  = _TS_GPT2_SMALL
    sep  = "-" * (len(model_name) + 2)
    lines = [
        f"# GPT-2 small vs {model_name} Comparison\n",
        f"Dataset: {dataset}\n",
        f"| Metric | GPT-2 small | {model_name} |",
        f"|--------|-------------|{sep}|",
        f"| Dataset | {dataset} | {dataset} |",
        f"| Datastore size | {ref['datastore_size']:,} | {datastore_size:,} |",
        f"| GPT-only NLL | {ref['gpt_nll']:.4f} | {_fmt(gpt_nll)} |",
        f"| Full raw Q NLL | {ref['full_raw_q_nll']:.4f} | " + (f"{fr_q_nll:.4f}" if fr_q_nll is not None else "— (not run)") + " |",
        f"| Best state Q NLL | {ref['best_state_q']:.4f} | {_fmt(best_q)} |",
        f"| Δ state vs full raw Q | {ref['best_state_q'] - ref['full_raw_q_nll']:+.4f} | "
            + (f"{best_q - fr_q_nll:+.4f}" if fr_q_nll is not None and math.isfinite(best_q) else "— (not run)") + " |",
        f"| State win rate vs equal-budget raw | 100% | {mean_frac:.0%}" + " |",
        f"| Object ratio | 25% | {obj_ratio_m} |",
        f"| Verdict | STRONG_GREEN | {verdict} |",
        "",
    ]

    if verdict == "STRONG_GREEN":
        lines.append(
            f"**Conclusion:** The predictive state codebook result is not GPT-2-small-specific. "
            f"{model_name} achieves the same compression benefit: {best_budget/datastore_size:.0%} "
            f"of full-datastore storage matches full raw Q-read.")
    elif verdict == "GREEN":
        lines.append(
            f"**Conclusion:** The result survives the model change. {model_name} state codebooks "
            f"beat GPT-only and equal-budget raw baselines, approaching (not fully matching) "
            f"full raw Q-read.")
    elif verdict == "GREEN_FULL_RAW_PENDING":
        lines.append(
            f"**Conclusion (pending):** {model_name} state codebooks beat GPT-only and equal-budget "
            f"raw. Full-datastore comparison pending — re-run with --run_full_raw.")
    elif verdict == "YELLOW":
        lines.append(
            f"**Conclusion:** Core compression effect partly survives on {model_name}, but "
            f"performance gap vs full raw Q is larger than on GPT-2 small. The method may be "
            f"partially tuned to GPT-2 small geometry.")
    else:
        lines.append(
            f"**Conclusion:** Result does not transfer to {model_name}. "
            f"The paper should stay framed around GPT-2 small until improved.")

    rep_dir = out / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    md_path = rep_dir / "GPT2_SMALL_VS_GPT2_MEDIUM_COMPARISON.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Comparison report: {md_path}")

    # Also write verdict JSON
    summary = {
        "model": model_name, "dataset": dataset,
        "datastore_size": datastore_size,
        "gpt_nll": round(gpt_nll, 6) if math.isfinite(gpt_nll) else None,
        "full_raw_q_nll": round(fr_q_nll, 6) if fr_q_nll is not None else None,
        "best_state_q_nll": round(best_q, 6) if math.isfinite(best_q) else None,
        "state_win_rate_vs_raw": round(mean_frac, 4) if math.isfinite(mean_frac) else None,
        "verdict": verdict,
        "results_by_seed": results,
    }
    with open(rep_dir / "second_model_verdict.json", "w") as f:
        json.dump(summary, f, indent=2)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",     required=True,
                        choices=["gpt2-medium", "gpt2-large",
                                 "EleutherAI/pythia-160m",
                                 "EleutherAI/pythia-410m"])
    parser.add_argument("--dataset",        default="TinyStories",
                        choices=["TinyStories", "wikitext2_raw"])
    parser.add_argument("--output",         required=True)
    parser.add_argument("--datastore_size", type=int, default=100_000)
    parser.add_argument("--seeds",          nargs="+", type=int, default=[42])
    parser.add_argument("--budgets",        nargs="+", type=int,
                        default=[5000, 10000, 25000])
    parser.add_argument("--methods",        nargs="+",
                        default=["minibatch_kmeans", "utility_weighted", "query_kmeans"])
    parser.add_argument("--raw_baselines",  nargs="+",
                        default=["raw_random", "cluster_medoids",
                                 "raw_high_gpt_entropy", "raw_high_gpt_loss", "raw_coverage"])
    parser.add_argument("--max_q_samples",  type=int, default=500_000)
    parser.add_argument("--n_epochs",       type=int, default=30)
    parser.add_argument("--fast_grid",      action="store_true", default=True)
    parser.add_argument("--no_fast_grid",   action="store_false", dest="fast_grid")
    parser.add_argument("--run_full_raw",   action="store_true")
    parser.add_argument("--device",         default="cuda")
    parser.add_argument("--force",          action="store_true")
    args = parser.parse_args()

    out  = Path(args.output)
    sent = out / "sentinels"
    sent.mkdir(parents=True, exist_ok=True)

    print(f"\nrun_model_replication")
    print(f"  Model:    {args.model_name}")
    print(f"  Dataset:  {args.dataset}")
    print(f"  Output:   {out}")
    print(f"  DS size:  {args.datastore_size:,}")
    print(f"  Seeds:    {args.seeds}")
    print(f"  Budgets:  {args.budgets}")
    print(f"  Methods:  {args.methods}")
    print(f"  Raw:      {args.raw_baselines}")
    print(f"  full_raw: {args.run_full_raw}")

    results = []

    for seed in args.seeds:
        seed_out = out / f"seed{seed}"
        seed_out.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"  Seed {seed}  model={args.model_name}")
        print(f"{'='*60}")

        # Stage 1: Extract hidden states
        try:
            src = extract_states_for_dataset(
                args.dataset, args.model_name, seed_out,
                args.datastore_size, seed, args.device, args.force,
            )
        except Exception as e:
            print(f"  [FAILED] extract: {e}")
            continue

        # Stage 2: Build state codebooks
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

        # Stage 3: Build raw memory baselines
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

        # Stage 4: Q-read for state codebooks
        run_stage(
            "evaluate_q_read_states",
            [sys.executable, SCRIPT_DIR / "evaluate_q_read.py",
             "--source", src,
             "--states_dir", seed_out / "states",
             "--output", seed_out / "q_read" / "states",
             "--max_q_samples", str(args.max_q_samples),
             "--n_epochs", str(args.n_epochs),
             "--device", args.device,
             *(["--fast_grid"] if args.fast_grid else []),
             *(["--force"] if args.force else [])],
            sent / f"seed{seed}_qread_states.done", args.force,
        )

        # Stage 5: Q-read for raw baselines
        run_stage(
            "evaluate_q_read_raw",
            [sys.executable, SCRIPT_DIR / "evaluate_q_read.py",
             "--source", src,
             "--states_dir", seed_out / "raw_baselines",
             "--output", seed_out / "q_read" / "raw_baselines",
             "--max_q_samples", str(args.max_q_samples),
             "--n_epochs", str(args.n_epochs),
             "--device", args.device,
             *(["--fast_grid"] if args.fast_grid else []),
             *(["--force"] if args.force else [])],
            sent / f"seed{seed}_qread_raw.done", args.force,
        )

        # Stage 6: Full raw datastore Q-read (optional)
        full_raw_result = None
        if args.run_full_raw:
            run_stage(
                "full_raw_q_read",
                [sys.executable, SCRIPT_DIR / "run_full_raw_q_read.py",
                 "--source",        seed_out,
                 "--output",        seed_out / "q_read" / "full_raw",
                 "--max_q_samples", str(args.max_q_samples),
                 "--n_epochs",      str(args.n_epochs),
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

        # Collect GPT NLL from extraction summary or val.pt
        gpt_nll = float("nan")
        extr_summary = seed_out / "extraction_summary.json"
        if extr_summary.exists():
            with open(extr_summary) as f:
                gpt_nll = json.load(f).get("gpt_nll") or float("nan")
        if not math.isfinite(gpt_nll):
            gpt_nll = _compute_gpt_nll(seed_out)
            if math.isfinite(gpt_nll):
                _save_extraction_summary(
                    seed_out, args.dataset, args.model_name,
                    args.datastore_size,
                    max(1000, int(args.datastore_size * 0.2)),
                    max(500,  int(args.datastore_size * 0.05)),
                    seed, gpt_nll,
                )

        # Stage 7: Aggregate
        agg_cmd = [sys.executable, SCRIPT_DIR / "aggregate_paper_results.py",
                   "--output", seed_out,
                   "--dataset", args.dataset,
                   "--model", args.model_name,
                   *(["--force"] if args.force else [])]
        if math.isfinite(gpt_nll):
            agg_cmd += ["--gpt_nll", str(round(gpt_nll, 6))]
        run_stage("aggregate", agg_cmd, sent / f"seed{seed}_aggregate.done", args.force)

        # Collect per-seed verdict
        verdict_path = seed_out / "reports" / "verdicts.json"
        if verdict_path.exists():
            with open(verdict_path) as f:
                v = json.load(f)
            results.append({
                "seed": seed, "model": args.model_name,
                "gpt_nll": gpt_nll,
                "full_raw": full_raw_result,
                **v,
            })
            print(f"\n  Seed {seed} verdict: {v.get('overall')}  "
                  f"best_q={v.get('best_q_nll','nan'):.4f}  "
                  f"gpt_nll={gpt_nll:.4f}")
            if full_raw_result:
                print(f"  Full-raw Q NLL:  {full_raw_result['full_raw_q_nll']:.4f}  "
                      f"fixed={full_raw_result['full_raw_fixed_nll']:.4f}")

    if not results:
        print("\n[FAILED] No seeds completed successfully.")
        sys.exit(1)

    # ── Summary across seeds ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Model replication summary: {args.model_name}  {args.dataset}")
    print(f"{'='*60}")
    for r in results:
        print(f"  Seed {r['seed']}: {r.get('overall')}  "
              f"best_q={r.get('best_q_nll', float('nan')):.4f}  "
              f"gpt_nll={r.get('gpt_nll', float('nan')):.4f}")

    # Save replication summary
    with open(out / "model_replication_summary.json", "w") as f:
        json.dump({
            "model": args.model_name, "dataset": args.dataset,
            "datastore_size": args.datastore_size,
            "results_by_seed": results,
        }, f, indent=2, default=lambda v: None if not math.isfinite(v) else v)

    # ── Final printed summary (spec section 13) ───────────────────────────────
    _print_second_model_summary(results, args.model_name, args.dataset, args.datastore_size)

    # ── Write comparison report ───────────────────────────────────────────────
    _write_comparison_report(out, results, args.model_name, args.dataset, args.datastore_size)

    print("\nDone.")


if __name__ == "__main__":
    main()

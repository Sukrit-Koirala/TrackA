"""
aggregate_seed_results.py

Loads per-seed results and produces:
  outputs_seed_sweep/seed_summary.csv
  outputs_seed_sweep/seed_summary.json
  outputs_seed_sweep/SEED_SWEEP_REPORT.md

Usage:
  python src/aggregate_seed_results.py --seeds 42 123 999
"""

import argparse
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path

# Methods tracked in the summary (must match method names in final_metrics_v2.json)
TRACKED_METHODS = [
    "GPT only",
    "Best fixed kNN (CT-selected)",
    "Best threshold heuristic (extended)",
    "Q-MLP-full",
    "Q-MLP-A",
    "Q-GradientBoosting",
    "[DIAG] Oracle per-example (val)",
]

NaN = float("nan")


# ── loaders ───────────────────────────────────────────────────────────────────

def load_metrics(seed: int) -> list[dict] | None:
    path = Path(f"outputs_seed_sweep/seed_{seed}/reports/final_metrics_v2.json")
    if not path.exists():
        print(f"  WARNING: {path} not found -- seed {seed} has no metrics yet")
        return None
    with open(path) as f:
        return json.load(f)


def load_audit(seed: int) -> dict:
    path = Path(f"outputs_seed_sweep/seed_{seed}/audit/split_overlap_report.json")
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def max_story_overlap(audit: dict) -> int:
    try:
        pairs = audit["state_overlaps"]["story_overlap"]
        return max(int(p["overlap"]) for p in pairs)
    except (KeyError, ValueError):
        return -1


def val_same_story_nbr(audit: dict) -> int:
    try:
        return int(audit["neighbor_sources"]["val"]["queries_with_any_same_story_neighbor"])
    except KeyError:
        return -1


def snippet_overlap_ds_val(audit: dict) -> int:
    """Datastore ∩ val context-snippet overlap count."""
    try:
        pairs = audit["state_overlaps"]["snippet_overlap"]
        entry = next(p for p in pairs
                     if "datastore" in p["pair"] and "val" in p["pair"])
        return int(entry["overlap"])
    except (KeyError, StopIteration):
        return -1


# ── format helpers ────────────────────────────────────────────────────────────

def _n(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.4f}"


def _signed(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:+.4f}"


def _pm(mean, std) -> str:
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "n/a"
    return f"{mean:.4f} +/- {std:.4f}"


def _ret(mean, std) -> str:
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "n/a"
    return f"{mean*100:.1f}% +/- {std*100:.1f}%"


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Aggregate seed sweep results")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 999])
    args = parser.parse_args()

    out_dir = Path("outputs_seed_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── collect per-seed data ──────────────────────────────────────────────────
    per_seed_rows: list[dict] = []
    seed_data: dict[int, list[dict]] = {}   # seed -> [method_dict, ...]

    for seed in args.seeds:
        metrics = load_metrics(seed)
        audit   = load_audit(seed)

        so     = max_story_overlap(audit)
        ssn    = val_same_story_nbr(audit)
        snip   = snippet_overlap_ds_val(audit)
        so_ok  = bool(so == 0) if so >= 0 else None

        if metrics is None:
            continue

        seed_data[seed] = metrics
        by_method = {r["method"]: r for r in metrics}

        gpt_nll   = by_method.get("GPT only", {}).get("val_nll", NaN)
        fixed_nll = by_method.get("Best fixed kNN (CT-selected)", {}).get("val_nll", NaN)

        for method in TRACKED_METHODS:
            r    = by_method.get(method, {})
            nll  = r.get("val_nll", NaN)
            dgpt = nll - gpt_nll   if not (math.isnan(nll) or math.isnan(gpt_nll))   else NaN
            dfix = nll - fixed_nll if not (math.isnan(nll) or math.isnan(fixed_nll)) else NaN

            per_seed_rows.append({
                "seed":                      seed,
                "method":                    method,
                "val_nll":                   nll,
                "val_ppl":                   r.get("val_ppl", NaN),
                "avg_k":                     r.get("avg_k", NaN),
                "retrieval_usage":           r.get("retrieval_usage", NaN),
                "delta_vs_gpt":              dgpt,
                "delta_vs_best_fixed":       dfix,
                "story_overlap_ok":          so_ok,
                "story_overlap_count":       so,
                "same_story_neighbor_count": ssn,
                "snippet_overlap_ds_val":    snip,
            })

    if not per_seed_rows:
        print("No seed results found. Run run_seed_sweep.py first.")
        return

    completed_seeds = sorted(seed_data.keys())
    n_seeds = len(completed_seeds)

    # ── save per-seed CSV/JSON ─────────────────────────────────────────────────
    df = pd.DataFrame(per_seed_rows)
    df.to_csv(out_dir / "seed_summary.csv", index=False)
    with open(out_dir / "seed_summary.json", "w", encoding="utf-8") as f:
        json.dump(per_seed_rows, f, indent=2)
    print(f"Saved: {out_dir / 'seed_summary.csv'}")
    print(f"Saved: {out_dir / 'seed_summary.json'}")

    # ── aggregate per-method ───────────────────────────────────────────────────
    agg_rows: list[dict] = []
    for method in TRACKED_METHODS:
        sub   = df[df["method"] == method]
        nlls  = sub["val_nll"].dropna().values
        ks    = sub["avg_k"].dropna().values
        rets  = sub["retrieval_usage"].dropna().values
        dfixs = sub["delta_vs_best_fixed"].dropna().values
        wins  = int((sub["delta_vs_best_fixed"] < 0).sum())

        agg_rows.append({
            "method":                   method,
            "mean_val_nll":             float(np.mean(nlls))  if len(nlls)  else NaN,
            "std_val_nll":              float(np.std(nlls))   if len(nlls)  else NaN,
            "mean_avg_k":               float(np.mean(ks))    if len(ks)    else NaN,
            "std_avg_k":                float(np.std(ks))     if len(ks)    else NaN,
            "mean_retrieval_usage":     float(np.mean(rets))  if len(rets)  else NaN,
            "std_retrieval_usage":      float(np.std(rets))   if len(rets)  else NaN,
            "mean_delta_vs_best_fixed": float(np.mean(dfixs)) if len(dfixs) else NaN,
            "std_delta_vs_best_fixed":  float(np.std(dfixs))  if len(dfixs) else NaN,
            "wins_vs_best_fixed_count": wins,
            "num_seeds":                n_seeds,
        })

    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(out_dir / "seed_aggregate.csv", index=False)
    with open(out_dir / "seed_aggregate.json", "w", encoding="utf-8") as f:
        json.dump(agg_rows, f, indent=2)
    print(f"Saved: {out_dir / 'seed_aggregate.csv'}")
    print(f"Saved: {out_dir / 'seed_aggregate.json'}")

    # ── helper lookups ─────────────────────────────────────────────────────────
    def agg(method: str) -> dict:
        return next((r for r in agg_rows if r["method"] == method), {})

    qf_agg   = agg("Q-MLP-full")
    qa_agg   = agg("Q-MLP-A")
    heur_agg = agg("Best threshold heuristic (extended)")
    gb_agg   = agg("Q-GradientBoosting")

    qf_wins = qf_agg.get("wins_vs_best_fixed_count", 0)
    qa_wins = qa_agg.get("wins_vs_best_fixed_count", 0)

    # ── verdict ────────────────────────────────────────────────────────────────
    if qf_wins == n_seeds and n_seeds > 0:
        verdict_tag  = "STRONG SUCCESS"
        verdict_body = f"Q-MLP-full beats best fixed kNN on all {n_seeds}/{n_seeds} seeds."
        verdict_interp = (
            "The improvement is robust to random seed variation. "
            "Dynamic gate control is consistently better than any fixed (k, tau, alpha) setting."
        )
    elif qf_wins >= max(1, (n_seeds * 2) // 3) and n_seeds > 0:
        verdict_tag  = "PARTIAL SUCCESS"
        verdict_body = f"Q-MLP-full beats best fixed kNN on {qf_wins}/{n_seeds} seeds."
        verdict_interp = (
            "The result is promising but somewhat seed-sensitive. "
            "More seeds would clarify whether this is a reliable effect."
        )
    elif qf_wins == 0 or n_seeds == 0:
        verdict_tag  = "FAILURE"
        verdict_body = f"Q-MLP-full did not beat best fixed kNN on any tested seed."
        verdict_interp = (
            "The original improvement appears unstable. "
            "The gate-chain controller does not reliably outperform fixed retrieval."
        )
    else:
        verdict_tag  = "WEAK SUCCESS"
        verdict_body = f"Q-MLP-full beats best fixed kNN on {qf_wins}/{n_seeds} seeds."
        verdict_interp = (
            "The result is seed-sensitive. "
            "Interpret cautiously; more seeds are needed."
        )

    # ── build SEED_SWEEP_REPORT.md ─────────────────────────────────────────────
    lines: list[str] = [
        "# Branch A Gate-Chain MVP -- Seed Sweep Report",
        "",
        f"Seeds evaluated: {completed_seeds}  ({n_seeds} total)",
        "",
        "> **Scope**: Tests whether learned dynamic control over the",
        "> MATCH -> SELECT -> PREDICT -> MIX gate chain consistently",
        "> outperforms a fixed (k, tau, alpha) retrieval gate setting,",
        "> across multiple random seeds with strict story-level splits.",
        "> This does **not** test emergent abstractions, WRITE/PRUNE gates,",
        "> regions, modules, or Branch B.",
        "",
        "---",
        "",
        "## Section 1 -- Summary Verdict",
        "",
        f"**[{verdict_tag}]** {verdict_body}",
        "",
        verdict_interp,
        "",
    ]

    qf_mean_dfix = qf_agg.get("mean_delta_vs_best_fixed", NaN)
    qf_std_dfix  = qf_agg.get("std_delta_vs_best_fixed", NaN)
    if not math.isnan(qf_mean_dfix):
        lines.append(
            f"Mean delta-NLL (Q-MLP-full vs best fixed): "
            f"**{qf_mean_dfix:+.4f}** +/- {qf_std_dfix:.4f}"
        )
        lines.append("")

    lines += [
        "---",
        "",
        "## Section 2 -- Per-Seed Table",
        "",
        "| Seed | GPT NLL | Fixed NLL | Heuristic NLL | Q-GB NLL | Q-MLP-full NLL | Q-MLP-A NLL | full vs fixed | A vs fixed |",
        "|------|---------|-----------|---------------|----------|----------------|-------------|---------------|------------|",
    ]

    for seed in completed_seeds:
        sub  = df[df["seed"] == seed]
        m2v  = sub.set_index("method")["val_nll"].to_dict()

        gpt  = m2v.get("GPT only", NaN)
        fix  = m2v.get("Best fixed kNN (CT-selected)", NaN)
        heur = m2v.get("Best threshold heuristic (extended)", NaN)
        gb   = m2v.get("Q-GradientBoosting", NaN)
        qf   = m2v.get("Q-MLP-full", NaN)
        qa   = m2v.get("Q-MLP-A", NaN)

        df_qf = (qf - fix) if not (math.isnan(qf) or math.isnan(fix)) else NaN
        df_qa = (qa - fix) if not (math.isnan(qa) or math.isnan(fix)) else NaN

        lines.append(
            f"| {seed} | {_n(gpt)} | {_n(fix)} | {_n(heur)} | {_n(gb)} | "
            f"{_n(qf)} | {_n(qa)} | {_signed(df_qf)} | {_signed(df_qa)} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Section 3 -- Mean +/- Std Table",
        "",
        "| Method | mean NLL +/- std | mean avg_k +/- std | mean ret% +/- std | mean delta-vs-fixed +/- std |",
        "|--------|------------------|--------------------|-------------------|----------------------------|",
    ]

    for r in agg_rows:
        if r["method"].startswith("[DIAG]"):
            continue
        lines.append(
            f"| {r['method']} "
            f"| {_pm(r['mean_val_nll'], r['std_val_nll'])} "
            f"| {_pm(r['mean_avg_k'], r['std_avg_k'])} "
            f"| {_ret(r['mean_retrieval_usage'], r['std_retrieval_usage'])} "
            f"| {_pm(r['mean_delta_vs_best_fixed'], r['std_delta_vs_best_fixed'])} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Section 4 -- Win Counts vs Best Fixed kNN",
        "",
    ]

    for r in agg_rows:
        if r["method"] in (
            "GPT only",
            "Best fixed kNN (CT-selected)",
            "[DIAG] Oracle per-example (val)",
        ):
            continue
        w = r["wins_vs_best_fixed_count"]
        n = r["num_seeds"]
        lines.append(f"- **{r['method']}**: {w} / {n} seeds beat best fixed kNN")

    lines += [
        "",
        "---",
        "",
        "## Section 5 -- Audit Confirmation",
        "",
        "| Seed | Story overlap | Same-story neighbors (val) | Snippet overlap (DS cap Val) | Audit OK |",
        "|------|---------------|---------------------------|------------------------------|----------|",
    ]

    for seed in completed_seeds:
        audit  = load_audit(seed)
        so     = max_story_overlap(audit)
        ssn    = val_same_story_nbr(audit)
        snip   = snippet_overlap_ds_val(audit)
        ok_str = "YES" if (so == 0 and ssn == 0) else "NO"
        lines.append(
            f"| {seed} | {so if so >= 0 else 'n/a'} "
            f"| {ssn if ssn >= 0 else 'n/a'} "
            f"| {snip if snip >= 0 else 'n/a'} "
            f"| {ok_str} |"
        )

    lines += [
        "",
        "> **Target**: story overlap = 0, same-story neighbors (val) = 0.",
        "> Context-snippet overlap is expected to be small but non-zero even with",
        "> story-level splitting (identical short contexts occur across stories).",
        "> It does not indicate leakage as long as story-level disjointness holds.",
        "",
        "---",
        "",
        "## Section 6 -- Interpretation",
        "",
        "### What this experiment tests",
        "",
        "A small Q-MLP (2 hidden layers, 128 units) learned to select gate parameters",
        "(k, tau, alpha) per query from 13 observation features derived from frozen",
        "GPT-2 hidden states and their nearest-neighbor similarity structure.",
        "The gate chain itself is fixed: MATCH -> SELECT -> PREDICT -> MIX.",
        "",
        "### What this does NOT test",
        "",
        "- Emergent abstraction or symbol binding",
        "- WRITE or PRUNE gates",
        "- Region or module discovery",
        "- Semantic clustering or hierarchy",
        "- Any form of structure learning (Branch B)",
        "- Transformer fine-tuning",
        "",
        "### Honest interpretation",
        "",
    ]

    if qf_wins == n_seeds and n_seeds >= 3:
        lines += [
            f"Q-MLP-full beat best fixed kNN on all {n_seeds} seeds.",
            "",
            f"Mean ΔNLL = {qf_mean_dfix:+.4f} +/- {qf_std_dfix:.4f}.",
            "The controller reliably identifies when and how to configure retrieval",
            "better than any single fixed (k, tau, alpha) setting can.",
            "",
            "This is a narrow claim about dynamic gate configuration, not about",
            "abstract reasoning, memory, or generalization beyond TinyStories.",
        ]
    elif qf_wins >= max(1, (n_seeds * 2) // 3) and n_seeds > 0:
        lines += [
            f"Q-MLP-full beat best fixed kNN on {qf_wins}/{n_seeds} seeds.",
            "",
            "The result is promising but not fully consistent across seeds.",
            "The mean improvement is "
            f"{qf_mean_dfix:+.4f} +/- {qf_std_dfix:.4f} NLL.",
            "More seeds or a larger validation set would be needed to confirm stability.",
        ]
    else:
        lines += [
            f"Q-MLP-full beat best fixed kNN on only {qf_wins}/{n_seeds} seeds.",
            "",
            "The original result (seed 42) does not generalize robustly.",
            "The gate-chain controller approach requires further investigation.",
        ]

    lines += [
        "",
        "---",
    ]

    report_path = out_dir / "SEED_SWEEP_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {report_path}")

    # ── terminal summary ───────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("SEED SWEEP AGGREGATE")
    print("=" * 72)
    print(f"{'Method':<44}  {'mean NLL':>8}  {'std NLL':>7}  {'wins':>6}")
    print("-" * 72)
    for r in agg_rows:
        if math.isnan(r["mean_val_nll"]):
            continue
        w = r["wins_vs_best_fixed_count"]
        n = r["num_seeds"]
        print(
            f"  {r['method']:<42}  {r['mean_val_nll']:>8.4f}  "
            f"{r['std_val_nll']:>7.4f}  {w}/{n}"
        )
    print("=" * 72)
    print(f"\n[{verdict_tag}] {verdict_body}")
    print(f"SEED_SWEEP_REPORT.md -> {report_path}")
    print("Done.")


if __name__ == "__main__":
    main()

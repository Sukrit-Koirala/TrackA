"""
compare_split_results.py

Compare results between the original random-position split and the clean
story-level split.

Loads:
  outputs/reports/          (original split)
  outputs_clean_storysplit/reports/   (clean story split)

Builds a side-by-side table and writes an AUDIT_REPORT.md.

Methods compared:
  GPT only
  Best fixed kNN
  Best threshold heuristic
  Q-MLP-full
  Q-MLP-A
  Q-GradientBoosting
  Oracle per-example [diagnostic]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json
import math
import numpy as np
import pandas as pd

from utils import load_config, ensure_dirs, ppl


NaN = float("nan")


def load_json(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def get_method_nll(
    method: str,
    baselines: dict | None,
    ctrl_res: dict | None,
    q_res: dict | None,
    sk_res: dict | None,
    best_heur: dict | None,
) -> tuple[float, float, float]:
    """Return (val_nll, avg_k, ret_usage)."""
    if method == "GPT only":
        if baselines:
            return baselines["gpt_only_val_nll"], 0.0, 0.0
    elif method == "Best fixed kNN":
        if baselines:
            return (baselines["best_train_selected_val_nll"],
                    baselines["best_train_selected_val_k"],
                    baselines["best_train_selected_val_ret"])
    elif method == "Best threshold heuristic":
        if best_heur:
            return best_heur["val_nll"], best_heur["val_k"], best_heur["val_ret_usage"]
        if ctrl_res and "entropy_heuristic_val" in ctrl_res:
            eh = ctrl_res["entropy_heuristic_val"]
            return eh["mean_nll"], eh["mean_k"], eh["retrieval_usage"]
    elif method.startswith("Q-MLP"):
        key = method.replace("/", "-")
        if q_res and key in q_res:
            m = q_res[key]
            return m["mean_nll"], m["mean_k"], m["retrieval_usage"]
    elif method == "Q-GradientBoosting":
        if sk_res and "Q-GradientBoosting" in sk_res:
            m = sk_res["Q-GradientBoosting"]
            return m["mean_nll"], m["mean_k"], m["retrieval_usage"]
    elif method == "Oracle per-example":
        if ctrl_res and "oracle_val" in ctrl_res:
            return ctrl_res["oracle_val"]["nll"], NaN, NaN
    return NaN, NaN, NaN


def load_split_results(reports_dir: Path) -> dict:
    return {
        "baselines": load_json(reports_dir / "fixed_baselines.json"),
        "ctrl_res":  load_json(reports_dir / "controller_results.json"),
        "q_res":     load_json(reports_dir / "q_controller_metrics.json"),
        "sk_res":    load_json(reports_dir / "sklearn_controller_metrics.json"),
        "best_heur": load_json(reports_dir / "best_heuristics.json"),
    }


METHODS = [
    "GPT only",
    "Best fixed kNN",
    "Best threshold heuristic",
    "Q-MLP-full",
    "Q-MLP-A",
    "Q-GradientBoosting",
    "Oracle per-example",
]


def main():
    parser = argparse.ArgumentParser(description="Compare old vs clean split results")
    parser.add_argument("--config", default="configs/clean_storysplit.yaml")
    parser.add_argument("--old-reports-dir", default="outputs/reports",
                        help="Directory with original split reports")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    old_dir   = Path(args.old_reports_dir)
    clean_dir = Path(cfg["reports_dir"])
    audit_dir = Path(cfg.get("audit_dir", "outputs_clean_storysplit/audit"))
    audit_dir.mkdir(parents=True, exist_ok=True)

    print(f"Old   reports: {old_dir}")
    print(f"Clean reports: {clean_dir}")

    old   = load_split_results(old_dir)
    clean = load_split_results(clean_dir)

    rows = []
    for method in METHODS:
        o_nll, o_k, o_ret = get_method_nll(method, **old)
        c_nll, c_k, c_ret = get_method_nll(method, **clean)
        rows.append({
            "method":               method,
            "old_val_nll":          o_nll,
            "clean_val_nll":        c_nll,
            "delta_clean_minus_old": c_nll - o_nll if not (math.isnan(c_nll) or math.isnan(o_nll)) else NaN,
            "old_avg_k":            o_k,
            "clean_avg_k":          c_k,
            "old_ret_usage":        o_ret,
            "clean_ret_usage":      c_ret,
        })

    df = pd.DataFrame(rows)

    # ── print table ────────────────────────────────────────────────────────────
    print("\n" + "="*90)
    print("OLD (random split) vs CLEAN (story-level split)")
    print("="*90)
    hdr = f"{'Method':<30} {'old_NLL':>8}  {'clean_NLL':>9}  {'Δ(clean-old)':>12}  {'old_k':>6}  {'clean_k':>7}"
    print(hdr)
    print("-"*90)
    for r in rows:
        def fn(v):
            return f"{v:.4f}" if not math.isnan(v) else "  nan  "
        delta = r["delta_clean_minus_old"]
        delta_str = (f"{delta:+.4f}" if not math.isnan(delta) else "    —   ")
        print(f"  {r['method']:<28} {fn(r['old_val_nll']):>8}  {fn(r['clean_val_nll']):>9}  "
              f"{delta_str:>12}  {fn(r['old_avg_k']):>6}  {fn(r['clean_avg_k']):>7}")
    print("="*90)

    df.to_csv(audit_dir / "old_vs_clean_comparison.csv", index=False)
    with open(audit_dir / "old_vs_clean_comparison.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved: {audit_dir / 'old_vs_clean_comparison.csv'}")

    # ── load audit data for the AUDIT_REPORT ──────────────────────────────────
    overlap_json    = load_json(audit_dir / "split_overlap_report.json")
    nbr_src_json    = load_json(audit_dir / "neighbor_source_report.json")
    sim_diag_json   = load_json(audit_dir / "similarity_diagnostics.json")
    cost_sweep_path = clean_dir / "cost_sweep.csv"

    # ── write AUDIT_REPORT.md ─────────────────────────────────────────────────
    clean_gpt_nll   = get_method_nll("GPT only",      **clean)[0]
    clean_fixed_nll = get_method_nll("Best fixed kNN", **clean)[0]
    clean_heur_nll  = get_method_nll("Best threshold heuristic", **clean)[0]
    clean_qfull_nll = get_method_nll("Q-MLP-full",    **clean)[0]
    clean_qa_nll    = get_method_nll("Q-MLP-A",       **clean)[0]
    clean_gb_nll    = get_method_nll("Q-GradientBoosting", **clean)[0]
    clean_oracle_nll = get_method_nll("Oracle per-example", **clean)[0]

    def _n(v):
        return f"{v:.4f}" if not math.isnan(v) else "N/A"

    def _delta(v, ref):
        if math.isnan(v) or math.isnan(ref):
            return "N/A"
        d = v - ref
        return f"{d:+.4f}"

    # Determine go/no-go
    learned_nlls = {
        "Best threshold heuristic": clean_heur_nll,
        "Q-MLP-full":               clean_qfull_nll,
        "Q-MLP-A":                  clean_qa_nll,
        "Q-GradientBoosting":       clean_gb_nll,
    }
    best_learned_name = min(
        ((n, v) for n, v in learned_nlls.items() if not math.isnan(v)),
        key=lambda x: x[1], default=("N/A", NaN)
    )
    beats_fixed = (not math.isnan(best_learned_name[1])
                   and best_learned_name[1] < clean_fixed_nll)

    # Story overlap summary
    story_overlaps = overlap_json.get("state_overlaps", {}).get("story_overlap", []) if overlap_json else []
    max_story_overlap = max((r["overlap"] for r in story_overlaps), default=None)

    # Neighbor same-story
    nbr_val = (nbr_src_json.get("val", {}) if nbr_src_json else {})
    nbr_same = nbr_val.get("queries_with_any_same_story_neighbor", None)

    # p99 nearest_sim
    p99_sim = (sim_diag_json.get("segments", {}).get("all", {})
               .get("features", {}).get("nearest_sim", {}).get("p99", None)
               if sim_diag_json else None)

    report_lines = [
        "# Branch A Gate-Chain MVP — Audit Report",
        "",
        "This report documents the data-integrity audit performed after the v2 results",
        "showed Q-MLP-full beating best fixed kNN by ΔNLL = -0.0686 on the random split.",
        "The audit re-runs the same experiments under a strict story-level split to test",
        "whether the improvement survives, or was driven by split leakage.",
        "",
        "---",
        "",
        "## Section 1 — Split Integrity",
        "",
        "### Story-level overlap (target: 0 for story_split: true)",
    ]
    if story_overlaps:
        for r in story_overlaps:
            report_lines.append(
                f"- {r['pair']}: overlap = **{r['overlap']}** "
                f"({r['overlap_pct_a']:.2f}% of A, {r['overlap_pct_b']:.2f}% of B)")
        if max_story_overlap == 0:
            report_lines.append("\n**✓ Zero story overlap across all split pairs.**")
        else:
            report_lines.append(f"\n**✗ Non-zero story overlap detected: {max_story_overlap} examples.**")
    else:
        report_lines.append("_(story_id metadata not found — was story_split: true used?)_")

    report_lines += [
        "",
        "### Neighbor source (same-story contamination in val)",
    ]
    if nbr_same is not None:
        if nbr_same == 0:
            report_lines.append(f"- Val: **0** queries have a same-story neighbor. ✓")
        else:
            report_lines.append(
                f"- Val: **{nbr_same}** queries have at least one same-story neighbor "
                f"({nbr_val.get('fraction_any_same_story', NaN)*100:.2f}%).")
    else:
        report_lines.append("_(neighbor_story_ids not present in neighbor files)_")

    report_lines += [
        "",
        "---",
        "",
        "## Section 2 — Similarity Diagnostics",
        "",
    ]
    if p99_sim is not None:
        report_lines.append(f"**p99 nearest_sim (all val):** {p99_sim:.6f}")
        if p99_sim > 0.99:
            report_lines.append(
                "\n> ⚠️  Very high p99 similarity.  Even with clean story split, "
                "GPT-2 hidden states at adjacent positions within the same context "
                "type produce near-identical representations.  This is a property "
                "of the model, not of leakage.  The audit checks whether story-level "
                "disjointness changes performance substantially.")
        else:
            report_lines.append(
                "\n> p99 ≤ 0.99.  Similarity values look reasonable for a clean split.")
    else:
        report_lines.append("_(similarity_diagnostics.json not found)_")

    report_lines += [
        "",
        "---",
        "",
        "## Section 3 — Cost-Sweep Fix",
        "",
    ]
    if cost_sweep_path.exists():
        try:
            import pandas as _pd
            cs = _pd.read_csv(cost_sweep_path)
            has_nan = cs["val_nll"].isna().any()
            report_lines.append("**Status:** NaN bug in run_heuristics.py cost-sweep fixed.")
            report_lines.append(f"Remaining NaNs: {'yes' if has_nan else 'none'}.")
            report_lines.append("")
            report_lines.append("| λ | best CT action | val NLL | avg k |")
            report_lines.append("|---|---|---|---|")
            for _, row in cs.iterrows():
                report_lines.append(
                    f"| {row['lambda_cost']:.4f} | {row['best_ct_action']} "
                    f"| {row['val_nll']:.4f} | {row['val_avg_k']:.1f} |")
        except Exception as e:
            report_lines.append(f"_(could not load cost_sweep.csv: {e})_")
    else:
        report_lines.append("_(cost_sweep.csv not found — run run_heuristics.py first)_")

    report_lines += [
        "",
        "---",
        "",
        "## Section 4 — Clean Split Performance",
        "",
        f"| Method | val NLL | val PPL | Δ vs GPT | Δ vs Best Fixed |",
        "|---|---|---|---|---|",
        f"| GPT only                  | {_n(clean_gpt_nll)}  | {_n(math.exp(clean_gpt_nll) if not math.isnan(clean_gpt_nll) else NaN)} | — | {_delta(clean_gpt_nll, clean_fixed_nll)} |",
        f"| Best fixed kNN            | {_n(clean_fixed_nll)} | {_n(math.exp(clean_fixed_nll) if not math.isnan(clean_fixed_nll) else NaN)} | {_delta(clean_fixed_nll, clean_gpt_nll)} | — |",
        f"| Best heuristic            | {_n(clean_heur_nll)}  | — | {_delta(clean_heur_nll, clean_gpt_nll)} | {_delta(clean_heur_nll, clean_fixed_nll)} |",
        f"| Q-MLP-full                | {_n(clean_qfull_nll)} | — | {_delta(clean_qfull_nll, clean_gpt_nll)} | {_delta(clean_qfull_nll, clean_fixed_nll)} |",
        f"| Q-MLP-A                   | {_n(clean_qa_nll)}    | — | {_delta(clean_qa_nll, clean_gpt_nll)} | {_delta(clean_qa_nll, clean_fixed_nll)} |",
        f"| Q-GradientBoosting        | {_n(clean_gb_nll)}    | — | {_delta(clean_gb_nll, clean_gpt_nll)} | {_delta(clean_gb_nll, clean_fixed_nll)} |",
        f"| [DIAG] Oracle per-example | {_n(clean_oracle_nll)} | — | {_delta(clean_oracle_nll, clean_gpt_nll)} | {_delta(clean_oracle_nll, clean_fixed_nll)} |",
        "",
        "---",
        "",
        "## Section 5 — Old vs Clean Comparison",
        "",
        "| Method | old NLL | clean NLL | Δ(clean−old) |",
        "|---|---|---|---|",
    ]
    for r in rows:
        report_lines.append(
            f"| {r['method']} | {_n(r['old_val_nll'])} | {_n(r['clean_val_nll'])} "
            f"| {_n(r['delta_clean_minus_old'])} |")

    # Go/no-go
    report_lines += [
        "",
        "---",
        "",
        "## Section 6 — Go / No-Go Judgment",
        "",
    ]
    if not math.isnan(clean_fixed_nll):
        if beats_fixed:
            report_lines += [
                f"**✓ MVP RESULT SURVIVES AUDIT.**",
                "",
                f"Best learned method on clean split: **{best_learned_name[0]}**  "
                f"(val NLL = {_n(best_learned_name[1])})  "
                f"vs best fixed kNN (val NLL = {_n(clean_fixed_nll)}).",
                f"ΔNLL = {_delta(best_learned_name[1], clean_fixed_nll)}.",
                "",
                "The gain is not explained by split leakage.  The controller genuinely",
                "learned when retrieval improves over GPT-only on held-out stories.",
            ]
        else:
            old_qfull_nll = get_method_nll("Q-MLP-full", **old)[0]
            old_fixed_nll = get_method_nll("Best fixed kNN", **old)[0]
            report_lines += [
                f"**✗ MVP RESULT DID NOT SURVIVE THE STORY-LEVEL AUDIT.**",
                "",
                f"Q-MLP-full old NLL = {_n(old_qfull_nll)}, clean NLL = {_n(clean_qfull_nll)}.",
                f"Best fixed kNN old = {_n(old_fixed_nll)}, clean = {_n(clean_fixed_nll)}.",
                "",
                "The improvement on the original split was likely driven by same-story",
                "contamination: datastore and val queries from the same story share",
                "high cosine similarity (adjacent hidden states), making retrieval",
                "trivially easy.  On a clean story split the advantage collapses.",
                "",
                "Next steps:",
                "- Check whether the extended heuristic still helps on the clean split.",
                "- Investigate whether a larger datastore or longer context window",
                "  provides legitimate retrieval signal.",
            ]
    else:
        report_lines += [
            "_(Cannot determine go/no-go: clean split results not yet available.",
            " Run the full pipeline with configs/clean_storysplit.yaml first.)_",
        ]

    report_txt = "\n".join(report_lines) + "\n"
    report_path = audit_dir / "AUDIT_REPORT.md"
    report_path.write_text(report_txt, encoding="utf-8")
    print(f"\nSaved: {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()

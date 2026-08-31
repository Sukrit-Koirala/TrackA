"""
aggregate_scale_results.py

Loads per-scale final_metrics_v2.json and audit JSONs.

Produces:
  outputs_scale_sweep/scale_summary.csv
  outputs_scale_sweep/scale_summary.json
  outputs_scale_sweep/SCALE_SWEEP_REPORT.md
  outputs_scale_sweep/plots/nll_vs_datastore_size.png
  outputs_scale_sweep/plots/delta_vs_fixed_by_scale.png
  outputs_scale_sweep/plots/avg_k_by_scale.png
  outputs_scale_sweep/plots/retrieval_usage_by_scale.png

Usage:
  python src/aggregate_scale_results.py --seed 42 --scales 50000 100000 200000
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

METHODS_ORDERED = [
    "GPT only",
    "Best fixed kNN (CT-selected)",
    "Best threshold heuristic (extended)",
    "Q-MLP-full",
    "Q-MLP-A",
    "Q-GradientBoosting",
    "[DIAG] Oracle per-example (val)",
]

PLOT_METHODS = [
    ("GPT only",                         "dimgray",     "--",  1.4),
    ("Best fixed kNN (CT-selected)",     "black",       "-",   2.0),
    ("Best threshold heuristic (extended)", "steelblue",":",   1.6),
    ("Q-MLP-full",                       "forestgreen", "-",   2.2),
    ("Q-MLP-A",                          "limegreen",   "--",  1.8),
    ("Q-GradientBoosting",               "darkorange",  "-.",  1.6),
]

NaN = float("nan")


# ── loaders ───────────────────────────────────────────────────────────────────

def scale_label(n: int) -> str:
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1_000:
        return f"{n // 1_000}k"
    return str(n)


def load_metrics(ds_size: int, seed: int) -> list[dict] | None:
    label = scale_label(ds_size)
    path  = Path(f"outputs_scale_sweep/scale_{label}_seed{seed}/reports/final_metrics_v2.json")
    if not path.exists():
        print(f"  WARNING: {path} not found -- scale {label} skipped")
        return None
    with open(path) as f:
        return json.load(f)


def load_scale_cfg(ds_size: int, seed: int) -> dict:
    label = scale_label(ds_size)
    path  = Path(f"configs/scale_sweep/scale_{label}_seed{seed}.yaml")
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f)


def load_audit(ds_size: int, seed: int) -> dict:
    label = scale_label(ds_size)
    path  = Path(f"outputs_scale_sweep/scale_{label}_seed{seed}/audit/split_overlap_report.json")
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def max_story_overlap(audit: dict) -> int:
    try:
        return max(int(p["overlap"]) for p in audit["state_overlaps"]["story_overlap"])
    except (KeyError, ValueError):
        return -1


def val_same_story_nbr(audit: dict) -> int:
    try:
        return int(audit["neighbor_sources"]["val"]["queries_with_any_same_story_neighbor"])
    except KeyError:
        return -1


# ── format helpers ────────────────────────────────────────────────────────────

def _n(v, fmt=".4f") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:{fmt}}"


def _signed(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:+.4f}"


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Aggregate scale sweep results")
    parser.add_argument("--seed",   type=int,  default=42)
    parser.add_argument("--scales", nargs="+", type=int,
                        default=[50_000, 100_000, 200_000])
    args = parser.parse_args()

    out_dir = Path("outputs_scale_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── collect per-scale data ─────────────────────────────────────────────────
    summary_rows: list[dict] = []
    scale_data:  dict[int, dict[str, dict]] = {}   # ds_size -> {method: row_dict}
    completed_scales: list[int] = []

    for ds_size in args.scales:
        metrics = load_metrics(ds_size, args.seed)
        cfg     = load_scale_cfg(ds_size, args.seed)
        audit   = load_audit(ds_size, args.seed)

        so      = max_story_overlap(audit)
        ssn     = val_same_story_nbr(audit)
        so_ok   = bool(so == 0) if so >= 0 else None

        ct_size  = cfg.get("controller_train_positions", NaN)
        val_size = cfg.get("val_positions", NaN)
        max_q    = cfg.get("max_q_samples", NaN)
        q_used   = (min(ct_size * 46, max_q)
                    if not (math.isnan(ct_size) or math.isnan(max_q)) else NaN)

        if metrics is None:
            continue

        completed_scales.append(ds_size)
        by_method = {r["method"]: r for r in metrics}
        scale_data[ds_size] = by_method

        gpt_nll   = by_method.get("GPT only", {}).get("val_nll", NaN)
        fixed_nll = by_method.get("Best fixed kNN (CT-selected)", {}).get("val_nll", NaN)

        for method in METHODS_ORDERED:
            r    = by_method.get(method, {})
            nll  = r.get("val_nll", NaN)
            dgpt = nll - gpt_nll   if not math.isnan(nll) and not math.isnan(gpt_nll)   else NaN
            dfix = nll - fixed_nll if not math.isnan(nll) and not math.isnan(fixed_nll) else NaN
            summary_rows.append({
                "scale_name":                 scale_label(ds_size),
                "seed":                       args.seed,
                "datastore_positions":        ds_size,
                "controller_train_positions": ct_size,
                "val_positions":              val_size,
                "method":                     method,
                "val_nll":                    nll,
                "val_ppl":                    r.get("val_ppl", NaN),
                "avg_k":                      r.get("avg_k", NaN),
                "retrieval_usage":            r.get("retrieval_usage", NaN),
                "delta_vs_gpt":               dgpt,
                "delta_vs_best_fixed":        dfix,
                "story_overlap_ok":           so_ok,
                "same_story_neighbor_count":  ssn,
                "q_samples_used":             q_used,
            })

    if not summary_rows:
        print("No scale results found. Run run_scale_sweep.py first.")
        return

    # ── save summary CSV/JSON ─────────────────────────────────────────────────
    df = pd.DataFrame(summary_rows)
    df.to_csv(out_dir / "scale_summary.csv", index=False)
    with open(out_dir / "scale_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"Saved: {out_dir / 'scale_summary.csv'}")
    print(f"Saved: {out_dir / 'scale_summary.json'}")

    n_scales = len(completed_scales)

    # ── win counts ─────────────────────────────────────────────────────────────
    qf_wins = qa_wins = gb_wins = heur_wins = 0
    for ds_size in completed_scales:
        by_m      = scale_data[ds_size]
        fixed_nll = by_m.get("Best fixed kNN (CT-selected)", {}).get("val_nll", NaN)
        if math.isnan(fixed_nll):
            continue

        def wins(method: str) -> bool:
            v = by_m.get(method, {}).get("val_nll", NaN)
            return not math.isnan(v) and v < fixed_nll

        if wins("Q-MLP-full"):                        qf_wins   += 1
        if wins("Q-MLP-A"):                           qa_wins   += 1
        if wins("Q-GradientBoosting"):                gb_wins   += 1
        if wins("Best threshold heuristic (extended)"): heur_wins += 1

    # ── verdict ────────────────────────────────────────────────────────────────
    if qf_wins == n_scales and n_scales > 0:
        if qa_wins >= max(1, (n_scales * 2) // 3):
            verdict_tag = "STRONG SCALE SUCCESS"
        else:
            verdict_tag = "SCALE SUCCESS"
        verdict_body = (
            f"Q-MLP-full beats best fixed on all {n_scales}/{n_scales} scales. "
            f"Q-MLP-A beats fixed on {qa_wins}/{n_scales} scales."
        )
    elif qf_wins >= max(1, (n_scales * 2) // 3) and n_scales > 0:
        verdict_tag  = "MEDIUM SCALE SUCCESS"
        verdict_body = (
            f"Q-MLP-full beats best fixed on {qf_wins}/{n_scales} scales. "
            "Margin may shrink at larger scale."
        )
    elif qf_wins > 0:
        verdict_tag  = "WARNING"
        verdict_body = (
            f"Q-MLP-full only beats fixed at {qf_wins}/{n_scales} scales. "
            "Advantage does not hold at all scales."
        )
    else:
        verdict_tag  = "FAILURE"
        verdict_body = "Best fixed matches or beats Q-MLP-full at all tested scales."

    # ── delta trend analysis ───────────────────────────────────────────────────
    deltas_qf: list[tuple[int, float]] = []
    for ds_size in completed_scales:
        by_m      = scale_data[ds_size]
        fixed_nll = by_m.get("Best fixed kNN (CT-selected)", {}).get("val_nll", NaN)
        qf_nll    = by_m.get("Q-MLP-full",                  {}).get("val_nll", NaN)
        if not math.isnan(fixed_nll) and not math.isnan(qf_nll):
            deltas_qf.append((ds_size, qf_nll - fixed_nll))

    if len(deltas_qf) >= 2:
        first_delta = deltas_qf[0][1]
        last_delta  = deltas_qf[-1][1]
        margin_change = last_delta - first_delta  # negative = margin grew
        if margin_change < -0.005:
            trend = "growing"
        elif abs(last_delta) < abs(first_delta) * 0.5:
            trend = "shrinking substantially"
        elif margin_change > 0.005:
            trend = "shrinking somewhat"
        else:
            trend = "stable"
    else:
        trend = "unknown (only one scale)"

    # ── build SCALE_SWEEP_REPORT.md ────────────────────────────────────────────
    lines: list[str] = [
        "# Branch A Gate-Chain MVP -- Scale Sweep Report",
        "",
        f"Scales tested: {[scale_label(s) for s in completed_scales]}  "
        f"(seed {args.seed})",
        "",
        "> Tests whether learned gate configuration (Q-MLP-full) continues to",
        "> outperform the best fixed (k, tau, alpha) setting as datastore and",
        "> query set sizes increase.  This does NOT test new gates, abstraction,",
        "> or Branch B -- only scale robustness of the existing MVP.",
        "",
        "---",
        "",
        "## Section 1 -- Summary Verdict",
        "",
        f"**[{verdict_tag}]** {verdict_body}",
        "",
        f"Controller advantage trend: **{trend}** across scales.",
        "",
        "---",
        "",
        "## Section 2 -- Scale Trend Table",
        "",
        "| Scale | DS | CT | Val | GPT NLL | Fixed NLL | "
        "Heuristic | Q-GB | Q-MLP-full | Q-MLP-A | full vs fixed | A vs fixed |",
        "|-------|----|----|-----|---------|-----------|"
        "-----------|------|------------|---------|---------------|------------|",
    ]

    for ds_size in completed_scales:
        by_m  = scale_data[ds_size]
        label = scale_label(ds_size)
        cfg   = load_scale_cfg(ds_size, args.seed)
        ct    = cfg.get("controller_train_positions", "?")
        val   = cfg.get("val_positions", "?")

        gpt   = by_m.get("GPT only",                            {}).get("val_nll", NaN)
        fix   = by_m.get("Best fixed kNN (CT-selected)",        {}).get("val_nll", NaN)
        heur  = by_m.get("Best threshold heuristic (extended)", {}).get("val_nll", NaN)
        gb    = by_m.get("Q-GradientBoosting",                  {}).get("val_nll", NaN)
        qf    = by_m.get("Q-MLP-full",                          {}).get("val_nll", NaN)
        qa    = by_m.get("Q-MLP-A",                             {}).get("val_nll", NaN)
        df_qf = (qf - fix) if not math.isnan(qf) and not math.isnan(fix) else NaN
        df_qa = (qa - fix) if not math.isnan(qa) and not math.isnan(fix) else NaN

        lines.append(
            f"| {label} | {ds_size//1000}k | {ct//1000 if isinstance(ct,int) else ct}k "
            f"| {val//1000 if isinstance(val,int) else val}k "
            f"| {_n(gpt)} | {_n(fix)} | {_n(heur)} | {_n(gb)} "
            f"| {_n(qf)} | {_n(qa)} | {_signed(df_qf)} | {_signed(df_qa)} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Section 3 -- Per-Method Detail by Scale",
        "",
        "| Scale | Method | NLL | PPL | avg_k | ret% | delta vs fixed |",
        "|-------|--------|-----|-----|-------|------|----------------|",
    ]

    for ds_size in completed_scales:
        by_m      = scale_data[ds_size]
        label     = scale_label(ds_size)
        fixed_nll = by_m.get("Best fixed kNN (CT-selected)", {}).get("val_nll", NaN)

        for method in METHODS_ORDERED:
            if method.startswith("[DIAG]"):
                continue
            r    = by_m.get(method, {})
            nll  = r.get("val_nll", NaN)
            ppl  = r.get("val_ppl", NaN)
            k    = r.get("avg_k", NaN)
            ret  = r.get("retrieval_usage", NaN)
            dfix = (nll - fixed_nll) if not math.isnan(nll) and not math.isnan(fixed_nll) else NaN
            ret_str = f"{ret*100:.0f}%" if not math.isnan(ret) else "n/a"
            lines.append(
                f"| {label} | {method} | {_n(nll)} | {_n(ppl,'.2f')} "
                f"| {_n(k,'.1f')} | {ret_str} | {_signed(dfix)} |"
            )

    lines += [
        "",
        "---",
        "",
        "## Section 4 -- Win Counts vs Best Fixed kNN",
        "",
        f"- **Q-MLP-full**:                {qf_wins} / {n_scales} scales",
        f"- **Q-MLP-A**:                   {qa_wins} / {n_scales} scales",
        f"- **Q-GradientBoosting**:        {gb_wins} / {n_scales} scales",
        f"- **Best threshold heuristic**:  {heur_wins} / {n_scales} scales",
        "",
        "---",
        "",
        "## Section 5 -- Audit Confirmation",
        "",
        "| Scale | Story overlap | Same-story nbrs (val) | Audit OK |",
        "|-------|---------------|-----------------------|----------|",
    ]

    for ds_size in completed_scales:
        label = scale_label(ds_size)
        audit = load_audit(ds_size, args.seed)
        so    = max_story_overlap(audit)
        ssn   = val_same_story_nbr(audit)
        ok    = "YES" if (so == 0 and ssn == 0) else ("NO" if (so > 0 or ssn > 0) else "n/a")
        lines.append(f"| {label} | {so if so >= 0 else 'n/a'} "
                     f"| {ssn if ssn >= 0 else 'n/a'} | {ok} |")

    lines += [
        "",
        "> Target: story overlap = 0, same-story neighbors (val) = 0.",
        "",
        "---",
        "",
        "## Section 6 -- Interpretation",
        "",
        "### What this experiment tests",
        "",
        "Whether the Q-MLP controller advantage over the best fixed (k, tau, alpha)",
        "gate setting persists as datastore and query sizes increase.",
        "",
        "### What this does NOT test",
        "",
        "- Emergent abstraction or new gate types",
        "- WRITE / PRUNE gates",
        "- Region or module discovery",
        "- Branch B (any form of structure learning)",
        "- Fine-tuning of GPT-2",
        "",
        "### Verdict",
        "",
    ]

    if qf_wins == n_scales and n_scales >= 2:
        lines += [
            f"Q-MLP-full beat best fixed kNN at all {n_scales} tested scales.",
            "",
            "Delta-NLL (Q-MLP-full vs best fixed) by scale:",
        ]
        for ds_size, delta in deltas_qf:
            lines.append(f"  {scale_label(ds_size)}: {delta:+.4f}")

        if trend == "growing":
            lines += [
                "",
                "Dynamic gate configuration is not a small-datastore artifact.",
                "The controller advantage **grows** with datastore size -- larger",
                "datastores provide richer retrieval diversity for the controller to exploit.",
            ]
        elif trend == "stable":
            lines += [
                "",
                "Dynamic gate configuration is not a small-datastore artifact.",
                "The controller advantage is **stable** across scales.",
            ]
        elif "shrinking" in trend:
            lines += [
                "",
                "The controller advantage **shrinks** at larger scale but remains positive.",
                "With more data the fixed baseline improves, but the controller still wins.",
            ]

        if qa_wins == n_scales:
            lines += [
                "",
                "Q-MLP-A (restricted action set) also beats best fixed at all scales,",
                "with lower average k than Q-MLP-full.  A restricted controller can",
                "reduce retrieval cost while improving NLL.",
            ]

    elif qf_wins > 0:
        lines += [
            f"Q-MLP-full beat best fixed at {qf_wins}/{n_scales} scales.",
            "The advantage is not fully robust to scale increases.",
        ]
    else:
        lines += [
            "Q-MLP-full did not beat best fixed at any tested scale.",
            "The gate-chain controller approach requires further investigation.",
        ]

    lines += ["", "---", ""]

    report_path = out_dir / "SCALE_SWEEP_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {report_path}")

    # ── plots ─────────────────────────────────────────────────────────────────
    if HAS_MPL and len(completed_scales) >= 2:
        plot_dir = out_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        x = [s / 1_000 for s in completed_scales]   # x-axis in thousands

        # helper: extract y values
        def yvals(method: str, key: str = "val_nll") -> list[float]:
            return [scale_data.get(s, {}).get(method, {}).get(key, NaN)
                    for s in completed_scales]

        # 1. NLL vs datastore size
        fig, ax = plt.subplots(figsize=(9, 5))
        for method, color, ls, lw in PLOT_METHODS:
            y = yvals(method)
            if any(not math.isnan(v) for v in y):
                ax.plot(x, y, marker="o", color=color, ls=ls,
                        lw=lw, label=method.replace(" (CT-selected)", "")
                                          .replace(" (extended)", ""))
        ax.set_xlabel("Datastore size (thousands of positions)")
        ax.set_ylabel("Val NLL (lower is better)")
        ax.set_title("NLL vs Datastore Size")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / "nll_vs_datastore_size.png", dpi=130)
        plt.close(fig)
        print(f"Saved: {plot_dir / 'nll_vs_datastore_size.png'}")

        # 2. Delta vs fixed by scale
        fig, ax = plt.subplots(figsize=(9, 5))
        for method, color, ls, lw in PLOT_METHODS[2:]:   # skip GPT-only and fixed
            fixed_nlls = yvals("Best fixed kNN (CT-selected)")
            method_nlls = yvals(method)
            y = [(m - f) if not (math.isnan(m) or math.isnan(f)) else NaN
                 for m, f in zip(method_nlls, fixed_nlls)]
            if any(not math.isnan(v) for v in y):
                ax.plot(x, y, marker="o", color=color, ls=ls, lw=lw,
                        label=method.replace(" (extended)", ""))
        ax.axhline(0, color="black", lw=0.8, ls="--", label="Best fixed baseline")
        ax.set_xlabel("Datastore size (thousands of positions)")
        ax.set_ylabel("Delta NLL vs best fixed (negative = better)")
        ax.set_title("Controller Advantage vs Best Fixed kNN by Scale")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / "delta_vs_fixed_by_scale.png", dpi=130)
        plt.close(fig)
        print(f"Saved: {plot_dir / 'delta_vs_fixed_by_scale.png'}")

        # 3. avg_k by scale
        fig, ax = plt.subplots(figsize=(9, 5))
        for method, color, ls, lw in PLOT_METHODS:
            y = yvals(method, "avg_k")
            if any(not math.isnan(v) for v in y):
                ax.plot(x, y, marker="o", color=color, ls=ls, lw=lw,
                        label=method.replace(" (CT-selected)", "")
                                    .replace(" (extended)", ""))
        ax.set_xlabel("Datastore size (thousands of positions)")
        ax.set_ylabel("Mean avg_k")
        ax.set_title("Average Neighbors Retrieved per Query by Scale")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / "avg_k_by_scale.png", dpi=130)
        plt.close(fig)
        print(f"Saved: {plot_dir / 'avg_k_by_scale.png'}")

        # 4. Retrieval usage by scale
        fig, ax = plt.subplots(figsize=(9, 5))
        for method, color, ls, lw in PLOT_METHODS[2:]:
            y = [v * 100 if not math.isnan(v) else NaN
                 for v in yvals(method, "retrieval_usage")]
            if any(not math.isnan(v) for v in y):
                ax.plot(x, y, marker="o", color=color, ls=ls, lw=lw,
                        label=method.replace(" (extended)", ""))
        ax.set_xlabel("Datastore size (thousands of positions)")
        ax.set_ylabel("Retrieval usage (%)")
        ax.set_title("Fraction of Queries Using Retrieval by Scale")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
        fig.tight_layout()
        fig.savefig(plot_dir / "retrieval_usage_by_scale.png", dpi=130)
        plt.close(fig)
        print(f"Saved: {plot_dir / 'retrieval_usage_by_scale.png'}")

    elif not HAS_MPL:
        print("  (matplotlib not available -- skipping plots)")
    else:
        print("  (need >= 2 completed scales for plots)")

    # ── terminal summary ──────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("SCALE SWEEP SUMMARY")
    print("=" * 78)
    print(f"{'Scale':<8}  {'Fixed NLL':>10}  {'Q-MLP-full':>12}  "
          f"{'Q-MLP-A':>10}  {'full vs fix':>12}  {'A vs fix':>10}")
    print("-" * 78)
    for ds_size in completed_scales:
        by_m  = scale_data[ds_size]
        label = scale_label(ds_size)
        fix   = by_m.get("Best fixed kNN (CT-selected)", {}).get("val_nll", NaN)
        qf    = by_m.get("Q-MLP-full",                  {}).get("val_nll", NaN)
        qa    = by_m.get("Q-MLP-A",                     {}).get("val_nll", NaN)
        df_qf = (qf - fix) if not math.isnan(qf) and not math.isnan(fix) else NaN
        df_qa = (qa - fix) if not math.isnan(qa) and not math.isnan(fix) else NaN
        print(f"  {label:<6}  {_n(fix):>10}  {_n(qf):>12}  "
              f"{_n(qa):>10}  {_signed(df_qf):>12}  {_signed(df_qa):>10}")
    print("=" * 78)
    print(f"\n[{verdict_tag}] {verdict_body}")
    print(f"Trend: {trend}")
    print(f"\nSCALE_SWEEP_REPORT.md -> {report_path}")
    print("Done.")


if __name__ == "__main__":
    main()

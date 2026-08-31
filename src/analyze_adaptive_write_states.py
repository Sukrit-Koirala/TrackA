"""
analyze_adaptive_write_states.py  --  MVP 3a analysis

Aggregates q_metrics.json files from evaluate_adaptive_write_states.py,
compares against MVP 2c baselines, and writes ADAPTIVE_WRITE_REPORT.md.

MVP 2c baselines used for comparison:
  minibatch_kmeans B=25k   Q=2.2937
  minibatch_kmeans B=50k   Q=2.2852   (best)
  utility_weighted B=10k   Q=2.2963
  GPT-only (correct)        =2.5533
  full_ds fixed kNN         =2.3873
  full_ds Q-MLP             =2.2977

Usage:
  python branch_a_gate_mvp/src/analyze_adaptive_write_states.py \\
    --output  branch_a_gate_mvp/outputs_mvp3a_adaptive_write \\
    --states_dir branch_a_gate_mvp/outputs_mvp3a_adaptive_write/states
"""

import sys, re, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import numpy as np
import torch
import pandas as pd

# Canonical baselines (never propagate the typo: GPT-only = 2.5533)
GPT_ONLY_NLL   = 2.5533
FULL_DS_FIXED  = 2.3873
FULL_DS_QMLP   = 2.2977

# MVP 2c baselines by budget
MVP2C_BY_BUDGET = {
    10000: {"minibatch_kmeans": float("nan"), "utility_weighted": 2.2963},
    25000: {"minibatch_kmeans": 2.2937},
    50000: {"minibatch_kmeans": 2.2852},
}
MVP2C_BEST_OVERALL = 2.2852   # minibatch_kmeans B=50k


def parse_tag(tag: str) -> dict:
    """Extract method, budget, and hyperparam info from a state file tag."""
    info = {"tag": tag, "method": "unknown", "budget": 0}

    # Budget
    m = re.search(r"_B(\d+)$", tag)
    if m:
        info["budget"] = int(m.group(1))

    # Method
    for meth in ["adaptive_threshold", "budget_filling", "uncertainty_aware",
                  "token_conflict", "split_conflict", "prune_low"]:
        if tag.startswith(meth):
            info["method"] = meth
            break

    # Hyperparams
    for pattern, key in [
        (r"thr([\d.]+)", "threshold"),
        (r"max(\d+)",    "max_state_count"),
        (r"ema(\d+)",    "ema_cap"),
        (r"wf([\d.]+)",  "warmup_fraction"),
        (r"lt([\d.]+)",  "loose_threshold"),
        (r"ct([\d.]+)",  "conflict_threshold"),
        (r"lp([\d.]+)",  "low_prob_threshold"),
    ]:
        mp = re.search(pattern, tag)
        if mp:
            try:
                info[key] = float(mp.group(1))
            except ValueError:
                pass

    return info


def load_all_metrics(q_read_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(q_read_dir.glob("*/q_metrics.json")):
        try:
            with open(p) as f:
                m = json.load(f)
            info = parse_tag(m.get("tag", p.parent.name))
            m.update(info)
            rows.append(m)
        except Exception as e:
            print(f"  [warn] failed to load {p}: {e}")
    return rows


def load_state_stats(states_dir: Path, tag: str) -> dict:
    """Load a .pt state file and return basic stats."""
    p = states_dir / f"{tag}.pt"
    if not p.exists():
        return {}
    try:
        s = torch.load(p, weights_only=False)
        B         = len(s["prototype_h"])
        tc        = s["total_counts"].numpy()
        ent       = s["state_entropy"].numpy()
        pur       = s["state_purity"].numpy()
        ac        = s.get("assigned_count", s["total_counts"]).numpy()
        active    = float((ac > 0).mean())
        return {
            "actual_num_states":    B,
            "mean_state_entropy":   float(ent.mean()),
            "median_state_purity":  float(np.median(pur)),
            "mean_state_count":     float(tc.mean()),
            "active_state_fraction": active,
        }
    except Exception:
        return {}


def build_comparison_df(metrics: list[dict], states_dir: Path | None = None) -> pd.DataFrame:
    rows = []
    for m in metrics:
        tag    = m.get("tag", "")
        budget = m.get("budget", 0)
        method = m.get("method", "unknown")
        q_nll  = m.get("q_state_nll", float("nan"))
        fixed  = m.get("best_fixed_nll", float("nan"))
        oracle = m.get("oracle_nll", float("nan"))
        n_st   = m.get("num_states", 0)

        # MVP 2c comparison at same budget
        mvp2c_same = MVP2C_BY_BUDGET.get(budget, {})
        best_mvp2c_same = min(mvp2c_same.values()) if mvp2c_same else float("nan")

        # State file stats
        st_stats = {}
        if states_dir:
            st_stats = load_state_stats(states_dir, tag)

        rows.append({
            "tag":              tag,
            "method":           method,
            "budget":           budget,
            "actual_num_states": st_stats.get("actual_num_states", n_st),
            "budget_usage_fraction": st_stats.get("actual_num_states", n_st) / max(budget, 1),
            "threshold":        m.get("threshold", float("nan")),
            "max_state_count":  m.get("max_state_count", float("nan")),
            "ema_cap":          m.get("ema_cap", 128),
            "loose_threshold":  m.get("loose_threshold", float("nan")),
            "conflict_threshold": m.get("conflict_threshold", float("nan")),
            "low_prob_threshold": m.get("low_prob_threshold", float("nan")),
            "fixed_state_nll":  fixed,
            "q_state_nll":      q_nll,
            "oracle_nll":       oracle,
            "gpt_only_nll":     GPT_ONLY_NLL,
            "full_ds_fixed_nll": FULL_DS_FIXED,
            "full_ds_qmlp_nll": FULL_DS_QMLP,
            "delta_q_vs_full_ds_qmlp":    q_nll - FULL_DS_QMLP,
            "delta_q_vs_best_mvp2c_overall": q_nll - MVP2C_BEST_OVERALL,
            "delta_q_vs_mvp2c_same_budget": q_nll - best_mvp2c_same,
            "delta_q_vs_gpt":   q_nll - GPT_ONLY_NLL,
            "active_state_fraction": st_stats.get("active_state_fraction", float("nan")),
            "mean_state_entropy":    st_stats.get("mean_state_entropy", float("nan")),
            "median_state_purity":   st_stats.get("median_state_purity", float("nan")),
            "mean_state_count":      st_stats.get("mean_state_count", float("nan")),
            "retrieval_usage":  m.get("retrieval_usage", float("nan")),
            "avg_k_states":     m.get("avg_k_states", float("nan")),
            "oracle_gap_vs_q":  oracle - q_nll,
            "top_action_1":     m.get("top_actions", [{}])[0].get("name", "") if m.get("top_actions") else "",
            "top_action_1_frac": m.get("top_actions", [{}])[0].get("frac", float("nan")) if m.get("top_actions") else float("nan"),
            "top_action_2":     m.get("top_actions", [{},{}])[1].get("name", "") if len(m.get("top_actions", [])) > 1 else "",
            "top_action_2_frac": m.get("top_actions", [{},{}])[1].get("frac", float("nan")) if len(m.get("top_actions", [])) > 1 else float("nan"),
        })

    return pd.DataFrame(rows)


def generate_report(df: pd.DataFrame, out: Path) -> str:
    lines = [
        "# MVP 3a Adaptive Write States — Analysis Report",
        "",
        "## Baselines (canonical — do not change)",
        f"| Baseline | NLL |",
        f"|----------|-----|",
        f"| GPT-only | {GPT_ONLY_NLL:.4f} |",
        f"| full_ds fixed kNN | {FULL_DS_FIXED:.4f} |",
        f"| full_ds Q-MLP | {FULL_DS_QMLP:.4f} |",
        f"| MVP 2c best (minibatch B=50k) | {MVP2C_BEST_OVERALL:.4f} |",
        "",
        "## Summary: All Adaptive-Write Variants",
        "",
    ]

    if df.empty:
        lines.append("No results found.")
    else:
        # Sort by Q NLL
        df_sorted = df.sort_values("q_state_nll")

        # Per-method best
        lines += [
            "### Best per Method",
            "",
            "| method | budget | actual_B | q_nll | delta_vs_qmlp | delta_vs_mvp2c_best | retrieval% | avg_k |",
            "|--------|--------|----------|-------|---------------|---------------------|------------|-------|",
        ]
        for meth, grp in df_sorted.groupby("method"):
            r = grp.loc[grp["q_state_nll"].idxmin()]
            lines.append(
                f"| {r['method']} | {int(r['budget'])} "
                f"| {int(r['actual_num_states']) if not np.isnan(r['actual_num_states']) else '?'} "
                f"| {r['q_state_nll']:.4f} "
                f"| {r['delta_q_vs_full_ds_qmlp']:+.4f} "
                f"| {r['delta_q_vs_best_mvp2c_overall']:+.4f} "
                f"| {r['retrieval_usage']:.1%} "
                f"| {r['avg_k_states']:.1f} |"
            )

        lines += [
            "",
            "### Full Table (sorted by Q NLL)",
            "",
            "| tag | B | q_nll | fixed | oracle | d_qmlp | d_mvp2c |",
            "|-----|---|-------|-------|--------|--------|---------|",
        ]
        for _, r in df_sorted.iterrows():
            B_ = int(r["actual_num_states"]) if not np.isnan(r["actual_num_states"]) else r["budget"]
            lines.append(
                f"| {r['tag'][:50]} | {B_} "
                f"| {r['q_state_nll']:.4f} | {r['fixed_state_nll']:.4f} "
                f"| {r['oracle_nll']:.4f} "
                f"| {r['delta_q_vs_full_ds_qmlp']:+.4f} "
                f"| {r['delta_q_vs_best_mvp2c_overall']:+.4f} |"
            )

        # Per-budget analysis
        lines += ["", "## Per-Budget Analysis", ""]
        for budget, grp in df_sorted.groupby("budget"):
            lines.append(f"### Budget = {int(budget):,}")
            lines.append("")
            mvp2c_same = min(MVP2C_BY_BUDGET.get(int(budget), {float("nan")}).values(),
                             default=float("nan"))
            if not np.isnan(mvp2c_same):
                lines.append(f"MVP 2c best at this budget: {mvp2c_same:.4f}")
            n_beat_qmlp = int((grp["q_state_nll"] < FULL_DS_QMLP).sum())
            n_beat_mvp2c = int((grp["q_state_nll"] < MVP2C_BEST_OVERALL).sum())
            lines.append(f"Variants beating full_ds_qmlp ({FULL_DS_QMLP:.4f}): {n_beat_qmlp}/{len(grp)}")
            lines.append(f"Variants beating MVP 2c best ({MVP2C_BEST_OVERALL:.4f}): {n_beat_mvp2c}/{len(grp)}")
            lines.append("")

            lines += [
                "| tag | q_nll | delta_vs_qmlp | budget_usage | active_frac | top_action |",
                "|-----|-------|---------------|--------------|-------------|------------|",
            ]
            for _, r in grp.iterrows():
                lines.append(
                    f"| {r['tag'][:45]} | {r['q_state_nll']:.4f} "
                    f"| {r['delta_q_vs_full_ds_qmlp']:+.4f} "
                    f"| {r['budget_usage_fraction']:.2f} "
                    f"| {r['active_state_fraction']:.2f} "
                    f"| {str(r['top_action_1'])[:25]} |"
                )
            lines.append("")

        # Method deep-dive
        lines += ["## Method Deep-Dive", ""]

        for meth, grp in df.groupby("method"):
            lines.append(f"### {meth}")
            lines.append(f"n_variants={len(grp)}")
            if len(grp) > 0:
                best_r = grp.loc[grp["q_state_nll"].idxmin()]
                lines.append(f"best_q_nll={best_r['q_state_nll']:.4f}  "
                             f"budget={int(best_r['budget'])}  "
                             f"tag={best_r['tag']}")
                lines.append(f"mean_retrieval_usage={grp['retrieval_usage'].mean():.2%}  "
                             f"mean_active_frac={grp['active_state_fraction'].mean():.2%}")
                lines.append(f"mean_oracle_gap_vs_q={grp['oracle_gap_vs_q'].mean():.4f}  "
                             f"(room for improvement)")
            lines.append("")

        # Conclusions
        if not df.empty:
            best_r = df.loc[df["q_state_nll"].idxmin()]
            lines += [
                "## Conclusions",
                "",
                f"**Best adaptive-write result: {best_r['q_state_nll']:.4f}**  "
                f"({best_r['tag']})",
                "",
                f"- Delta vs GPT-only ({GPT_ONLY_NLL:.4f}): "
                f"{best_r['q_state_nll'] - GPT_ONLY_NLL:+.4f}",
                f"- Delta vs full_ds fixed ({FULL_DS_FIXED:.4f}): "
                f"{best_r['q_state_nll'] - FULL_DS_FIXED:+.4f}",
                f"- Delta vs full_ds Q-MLP ({FULL_DS_QMLP:.4f}): "
                f"{best_r['q_state_nll'] - FULL_DS_QMLP:+.4f}",
                f"- Delta vs MVP 2c best ({MVP2C_BEST_OVERALL:.4f}): "
                f"{best_r['q_state_nll'] - MVP2C_BEST_OVERALL:+.4f}",
                "",
                f"Budget usage: {best_r['budget_usage_fraction']:.1%}  "
                f"Active states: {best_r['active_state_fraction']:.1%}  "
                f"Retrieval usage: {best_r['retrieval_usage']:.1%}",
            ]

    report = "\n".join(lines) + "\n"
    rpath  = out / "ADAPTIVE_WRITE_REPORT.md"
    with open(rpath, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report: {rpath}")
    return report


def try_plots(df: pd.DataFrame, out: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if df.empty or len(df) < 2:
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 1. Q NLL by method and budget
        ax = axes[0]
        for meth, grp in df.groupby("method"):
            ax.scatter(grp["budget"], grp["q_state_nll"], label=meth, alpha=0.7, s=60)
        ax.axhline(FULL_DS_QMLP, ls="--", color="r", label=f"full_ds_qmlp={FULL_DS_QMLP:.4f}")
        ax.axhline(MVP2C_BEST_OVERALL, ls=":", color="g", label=f"mvp2c_best={MVP2C_BEST_OVERALL:.4f}")
        ax.set_xlabel("Budget"); ax.set_ylabel("Q NLL")
        ax.set_title("Q NLL by Method and Budget"); ax.legend(fontsize=7)

        # 2. Budget usage vs Q NLL
        ax = axes[1]
        ax.scatter(df["budget_usage_fraction"], df["q_state_nll"],
                   c=df["budget"].rank(), cmap="viridis", alpha=0.8, s=60)
        ax.axhline(FULL_DS_QMLP, ls="--", color="r", alpha=0.5)
        ax.set_xlabel("Budget Usage Fraction"); ax.set_ylabel("Q NLL")
        ax.set_title("Budget Usage vs Q NLL")

        # 3. Active fraction vs Q NLL
        ax = axes[2]
        ax.scatter(df["active_state_fraction"], df["q_state_nll"], alpha=0.8, s=60)
        ax.axhline(FULL_DS_QMLP, ls="--", color="r", alpha=0.5)
        ax.set_xlabel("Active State Fraction"); ax.set_ylabel("Q NLL")
        ax.set_title("Active States vs Q NLL")

        plt.tight_layout()
        fig.savefig(out / "adaptive_write_comparison.png", dpi=120)
        plt.close(fig)
        print(f"  Plot: {out / 'adaptive_write_comparison.png'}")
    except Exception as e:
        print(f"  [plots skipped: {e}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",     required=True)
    parser.add_argument("--states_dir", default=None,
                        help="Path to states/ dir for extra state stats")
    parser.add_argument("--force",      action="store_true")
    args = parser.parse_args()

    out      = Path(args.output)
    qrd      = out / "q_read"
    sdir     = Path(args.states_dir) if args.states_dir else None

    csv_path = out / "adaptive_write_results.csv"
    if csv_path.exists() and not args.force:
        print(f"Loading cached results from {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print(f"\nLoading q_metrics.json files from {qrd} ...")
        metrics = load_all_metrics(qrd)
        print(f"  Found {len(metrics)} result files")
        df = build_comparison_df(metrics, sdir)
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

    if df.empty:
        print("No results to analyze.")
        return

    print(f"\n{len(df)} variants loaded")
    print(df[["method", "budget", "q_state_nll",
              "delta_q_vs_full_ds_qmlp", "retrieval_usage"]].to_string(index=False))

    generate_report(df, out)
    try_plots(df, out)

    best = df.loc[df["q_state_nll"].idxmin()]
    print(f"\nBest: {best['tag']}")
    print(f"  Q NLL = {best['q_state_nll']:.4f}  "
          f"delta_vs_qmlp = {best['delta_q_vs_full_ds_qmlp']:+.4f}  "
          f"delta_vs_mvp2c = {best['delta_q_vs_best_mvp2c_overall']:+.4f}")


if __name__ == "__main__":
    main()

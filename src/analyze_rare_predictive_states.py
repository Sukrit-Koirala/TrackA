"""
analyze_rare_predictive_states.py  --  MVP 4c-0 Patch: Quadrant / Rare-Gem Analysis

Performs the frequency-vs-utility quadrant analysis, oracle vs bandit normalized
comparison, false negative extension, and support/frequency correlation analysis.

Requires normalized_object_utility.parquet from analyze_normalized_object_utility.py.

Answers Q11-Q18 and writes a report addendum.

Outputs (in {output}/normalized_analysis/):
  oracle_vs_bandit_normalized.csv
  support_vs_utility.csv
  frequency_vs_utility.csv
  immediate_vs_normalized_future.csv   (if bandit events contain immediate reward)
  false_negative_by_utility_type.csv
  rare_gems.csv
  workhorse_states.csv
  rare_low_value.csv
  common_low_value.csv
  plots/*.png
  normalized_analysis_summary.json
  report_addendum.md

Usage:
  python src/analyze_rare_predictive_states.py \\
    --output outputs_mvp4c0_delayed_credit_audit_fast \\
    --trajectories oracle bandit \\
    --lambda_primary 10 \\
    --k_primary 8 \\
    --imm_reward_col reward_local \\
    --force
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _spearman(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return float("nan"), int(mask.sum())
    try:
        return float(spearmanr(x[mask], y[mask]).statistic), int(mask.sum())
    except Exception:
        return float("nan"), int(mask.sum())


def _scatter_with_hist(ax, x, y, xlabel, ylabel, title, log_x=False, alpha=0.25):
    mask = np.isfinite(x) & np.isfinite(y)
    xm, ym = x[mask], y[mask]
    if log_x and (xm > 0).any():
        xm = np.log1p(xm)
        xlabel = f"log(1 + {xlabel})"
    ax.scatter(xm, ym, alpha=alpha, s=8, linewidths=0)
    try:
        z = np.polyfit(xm, ym, 1)
        lx = np.linspace(xm.min(), xm.max(), 100)
        ax.plot(lx, np.polyval(z, lx), "r-", lw=1.5, label="trend")
    except Exception:
        pass
    rho, _ = _spearman(x, y)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(f"{title}\nSpearman ρ={rho:.3f}", fontsize=9)
    ax.grid(True, alpha=0.3)


def _hist(ax, values, xlabel, title, bins=40, color="steelblue"):
    v = values[np.isfinite(values)]
    ax.hist(v, bins=bins, color=color, edgecolor="white", linewidth=0.3)
    ax.axvline(float(np.median(v)), color="r", linestyle="--", lw=1.5,
               label=f"median={np.median(v):.4f}")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("count", fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


# ── Quadrant assignment ───────────────────────────────────────────────────────

def assign_quadrant(df: pd.DataFrame, freq_col: str, util_col: str,
                    split: str = "quartile") -> pd.DataFrame:
    """
    Assigns quadrant labels using Q25/Q75 thresholds computed from nonzero-frequency
    objects only. Objects in the middle are labeled 'middle' (not forced into a quadrant).
    Zero-frequency objects are labeled 'zero_freq' and handled separately.

    Quadrant rule:
      rare_gem:        freq <= Q25 AND util >= Q75  (of nonzero-freq objects)
      workhorse:       freq >= Q75 AND util >= Q75
      rare_low_value:  freq <= Q25 AND util <= Q25
      common_low_value:freq >= Q75 AND util <= Q25
      middle:          everything else among nonzero-freq objects
      zero_freq:       freq == 0 (always excluded from threshold computation)

    Validation: rare + workhorse + rare_low + common_low + middle + zero_freq
                == total objects.
    """
    df = df.copy()
    freq_vals = df[freq_col].fillna(0)
    util_vals = df[util_col].fillna(0)

    # Separate zero-frequency objects — never classified into named quadrants
    zero_mask = freq_vals == 0

    # Thresholds computed ONLY from nonzero-frequency objects
    nonzero = freq_vals[~zero_mask]
    nonzero_util = util_vals[~zero_mask]

    if len(nonzero) < 4:
        # Too few nonzero objects to compute meaningful thresholds
        df["quadrant"]  = np.where(zero_mask, "zero_freq", "middle")
        df["freq_thr_lo"] = float("nan")
        df["freq_thr_hi"] = float("nan")
        df["util_thr_lo"] = float("nan")
        df["util_thr_hi"] = float("nan")
        return df

    freq_lo = float(np.percentile(nonzero, 25))        # Q25 frequency
    freq_hi = float(np.percentile(nonzero, 75))        # Q75 frequency
    util_lo = float(np.percentile(nonzero_util, 25))   # Q25 utility
    util_hi = float(np.percentile(nonzero_util, 75))   # Q75 utility

    def label(f, u, is_zero):
        if is_zero:
            return "zero_freq"
        if f <= freq_lo and u >= util_hi:
            return "rare_gem"
        if f >= freq_hi and u >= util_hi:
            return "workhorse"
        if f <= freq_lo and u <= util_lo:
            return "rare_low_value"
        if f >= freq_hi and u <= util_lo:
            return "common_low_value"
        return "middle"

    df["quadrant"]    = [label(f, u, z) for f, u, z in
                         zip(freq_vals, util_vals, zero_mask)]
    df["freq_thr_lo"] = freq_lo
    df["freq_thr_hi"] = freq_hi
    df["util_thr_lo"] = util_lo
    df["util_thr_hi"] = util_hi

    # Validation: counts must sum to total
    expected  = len(df)
    quad_cats = ["rare_gem", "workhorse", "rare_low_value",
                 "common_low_value", "middle", "zero_freq"]
    actual    = sum((df["quadrant"] == q).sum() for q in quad_cats)
    assert actual == expected, (
        f"Quadrant count mismatch: {actual} != {expected}. "
        f"Counts: {dict((q, int((df['quadrant']==q).sum())) for q in quad_cats)}"
    )

    return df


# ── Oracle vs bandit comparison ───────────────────────────────────────────────

def compare_trajectories(all_df: pd.DataFrame, lambdas: list[int]) -> pd.DataFrame:
    lam = lambdas[1] if len(lambdas) > 1 else lambdas[0]
    metrics = (
        ["utility_total", "mean_gain_per_retrieval", "median_gain_per_retrieval",
         "positive_use_fraction", "negative_use_fraction",
         f"shrunk_mean_lambda{lam}", "utility_per_support",
         "unique_counterfactual_gain", "replaceability_ratio",
         "mean_gain_per_opportunity"]
        + [f"shrunk_opp_lambda{lam}_at_8" if f"shrunk_opp_lambda{lam}_at_8" in all_df.columns
           else f"shrunk_opportunity_lambda{lam}"]
    )
    rows = []
    for traj in all_df["trajectory"].unique():
        sub = all_df[all_df["trajectory"] == traj]
        _fc = "relevant_opportunity_count" if "relevant_opportunity_count" in sub.columns else "retrieval_count"
        sel = sub[sub[_fc].fillna(0) > 0]
        for m in metrics:
            if m not in sub.columns:
                continue
            vals = sub[m].dropna()
            sel_vals = sel[m].dropna() if len(sel) > 0 else vals[:0]
            rows.append({
                "trajectory": traj,
                "metric":     m,
                "n_all":      len(vals),
                "n_selected": len(sel_vals),
                "mean_all":   float(vals.mean()) if len(vals) > 0 else float("nan"),
                "mean_selected": float(sel_vals.mean()) if len(sel_vals) > 0 else float("nan"),
                "median_all": float(vals.median()) if len(vals) > 0 else float("nan"),
                "p75_all":    float(vals.quantile(0.75)) if len(vals) > 0 else float("nan"),
                "p25_all":    float(vals.quantile(0.25)) if len(vals) > 0 else float("nan"),
            })
    return pd.DataFrame(rows)


# ── False negative analysis ───────────────────────────────────────────────────

def false_negative_analysis(df: pd.DataFrame, events_df: pd.DataFrame,
                             lambdas: list[int], pct_thresholds: list[float]) -> pd.DataFrame:
    """
    False negative = immediate write reward <= 0 BUT normalized future utility is high.
    Requires immediate_reward column in events_df (optional, skipped if missing).
    """
    if events_df is None or len(events_df) == 0:
        return pd.DataFrame()
    if "immediate_reward" not in events_df.columns and "reward_local" not in events_df.columns:
        return pd.DataFrame()

    imm_col = "immediate_reward" if "immediate_reward" in events_df.columns else "reward_local"

    # Aggregate immediate reward per object (from first CREATE event)
    create_ev = events_df[events_df["action_type"].str.upper().isin(
        {"CREATE", "CREATE_BUFFER"})].copy()
    imm_per_obj = create_ev.groupby("object_id")[imm_col].mean().rename("imm_reward")
    merged = df.merge(imm_per_obj, on="object_id", how="inner")
    if len(merged) == 0:
        return pd.DataFrame()

    imm_vals = merged["imm_reward"].values
    imm_neg  = imm_vals <= 0

    lam = lambdas[1] if len(lambdas) > 1 else lambdas[0]
    util_cols = {
        "total":       "utility_total",
        "shrunk_mean": f"shrunk_mean_lambda{lam}",
        "shrunk_opp":  f"shrunk_opportunity_lambda{lam}",
        "unique":      "unique_counterfactual_gain",
    }

    rows = []
    for pct in pct_thresholds:
        for util_name, util_col in util_cols.items():
            if util_col not in merged.columns:
                continue
            thr  = float(np.percentile(merged[util_col].fillna(0), 100 - pct))
            high = merged[util_col].fillna(0) >= thr
            fn   = imm_neg & high
            rows.append({
                "trajectory":   merged["trajectory"].iloc[0] if "trajectory" in merged.columns else "?",
                "utility_type": util_name,
                "pct_threshold": pct,
                "high_util_thr": thr,
                "n_total":       len(merged),
                "n_imm_neg":     int(imm_neg.sum()),
                "n_high_util":   int(high.sum()),
                "n_fn":          int(fn.sum()),
                "fn_rate":       float(fn.sum() / max(imm_neg.sum(), 1)),
                "fn_of_high":    float(fn.sum() / max(high.sum(), 1)),
            })
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output",          required=True)
    ap.add_argument("--trajectories",    nargs="+", default=["oracle", "bandit"])
    ap.add_argument("--lambda_primary",  type=int, default=10)
    ap.add_argument("--k_primary",       type=int, default=8)
    ap.add_argument("--imm_reward_col",  default="reward_local",
                    help="Column in events parquet holding immediate reward")
    ap.add_argument("--force",           action="store_true")
    args = ap.parse_args()

    out_dir  = Path(args.output)
    norm_dir = out_dir / "normalized_analysis"
    plot_dir = norm_dir / "plots"
    norm_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    sentinel = norm_dir / "normalized_analysis_summary.json"
    if sentinel.exists() and not args.force:
        print(f"[cached] {sentinel}")
        return

    util_path = out_dir / "object_utility" / "normalized_object_utility.parquet"
    cfg_path  = out_dir / "object_utility" / "norm_config.json"

    if not util_path.exists():
        print("ERROR: normalized_object_utility.parquet not found")
        print("  Run analyze_normalized_object_utility.py first")
        return

    all_df = pd.read_parquet(util_path)
    print(f"Loaded {len(all_df):,} object utility records")

    config     = json.load(open(cfg_path)) if cfg_path.exists() else {}
    lambdas    = list(config.get(args.trajectories[0], {}).get("lambdas", [5, 10, 20]))
    lam        = args.lambda_primary
    lam_col    = f"shrunk_mean_lambda{lam}"
    # Primary frequency axis = opportunity@k (how often this object was in the top-k
    # retrieval pool), NOT raw retrieval_count. Per spec Part G.
    freq_col   = "relevant_opportunity_count"

    # Load events for FN analysis
    all_events = {}
    for traj in args.trajectories:
        ev_path = out_dir / "provenance" / f"{traj}_events.parquet"
        if ev_path.exists():
            ev = pd.read_parquet(ev_path)
            if args.imm_reward_col in ev.columns:
                ev = ev.rename(columns={args.imm_reward_col: "immediate_reward"})
            all_events[traj] = ev

    # ── Oracle vs Bandit normalized comparison ────────────────────────────────
    ovb_df = compare_trajectories(all_df, lambdas)
    ovb_df.to_csv(norm_dir / "oracle_vs_bandit_normalized.csv", index=False)
    print(f"\nOracle vs Bandit (shrunk_mean_lambda{lam}):")
    sub = ovb_df[ovb_df["metric"] == lam_col]
    if not sub.empty:
        print(sub[["trajectory", "mean_all", "median_all", "p25_all", "p75_all"]].to_string(index=False))

    # ── Quadrant analysis (Q25/Q75 thresholds from nonzero-freq objects) ────────
    quad_frames   = {}
    rare_gem_dfs  = []
    workhorse_dfs = []
    rare_lv_dfs   = []
    common_lv_dfs = []

    for traj in args.trajectories:
        sub = all_df[all_df["trajectory"] == traj].copy()
        if sub.empty or lam_col not in sub.columns:
            continue
        sub_q = assign_quadrant(sub, freq_col, lam_col)
        quad_frames[traj] = sub_q

        rare_gem_dfs.append(sub_q[sub_q["quadrant"] == "rare_gem"])
        workhorse_dfs.append(sub_q[sub_q["quadrant"] == "workhorse"])
        rare_lv_dfs.append(sub_q[sub_q["quadrant"] == "rare_low_value"])
        common_lv_dfs.append(sub_q[sub_q["quadrant"] == "common_low_value"])

        all_quad_cats = ["rare_gem", "workhorse", "rare_low_value",
                         "common_low_value", "middle", "zero_freq"]
        print(f"\n[{traj}] Quadrant counts (Q25/Q75 on nonzero-freq objects, {freq_col} / {lam_col}):")
        for qname in all_quad_cats:
            n_q = (sub_q["quadrant"] == qname).sum()
            print(f"  {qname:<25} {n_q:,}  ({n_q/len(sub_q)*100:.1f}%)")

    def _save_combined(frames, name):
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(norm_dir / f"{name}.csv", index=False)

    _save_combined(rare_gem_dfs,  "rare_gems")
    _save_combined(workhorse_dfs, "workhorse_states")
    _save_combined(rare_lv_dfs,   "rare_low_value")
    _save_combined(common_lv_dfs, "common_low_value")

    # ── Correlation tables ────────────────────────────────────────────────────
    def corr_table(df_sub, x_col, y_cols, label):
        rows = []
        for yc in y_cols:
            if yc not in df_sub.columns:
                continue
            x = df_sub[x_col].fillna(0).values.astype(float)
            y = df_sub[yc].fillna(0).values.astype(float)
            rho, n = _spearman(x, y)
            rows.append({"x": x_col, "y": yc, "spearman": rho, "n": n})
        out_df = pd.DataFrame(rows)
        out_df.to_csv(norm_dir / f"{label}.csv", index=False)
        return out_df

    util_targets = (
        ["utility_total", "mean_gain_per_retrieval", lam_col,
         "utility_per_support", "unique_counterfactual_gain",
         "mean_gain_per_opportunity"]
    )

    for traj in args.trajectories:
        sub = all_df[all_df["trajectory"] == traj]
        if sub.empty:
            continue

        corr_table(sub, "final_support",     util_targets, f"support_vs_utility_{traj}")
        corr_table(sub, freq_col,            util_targets, f"frequency_vs_utility_{traj}")

    # Combined
    corr_table(all_df, "final_support", util_targets, "support_vs_utility")
    corr_table(all_df, freq_col,        util_targets, "frequency_vs_utility")

    # ── False negative analysis ───────────────────────────────────────────────
    fn_frames = []
    for traj in args.trajectories:
        sub = all_df[all_df["trajectory"] == traj]
        ev  = all_events.get(traj)
        if ev is not None and len(sub) > 0:
            fn_df = false_negative_analysis(sub, ev, lambdas, [10.0, 25.0])
            if len(fn_df) > 0:
                fn_df["trajectory"] = traj
                fn_frames.append(fn_df)

    if fn_frames:
        fn_all = pd.concat(fn_frames, ignore_index=True)
        fn_all.to_csv(norm_dir / "false_negative_by_utility_type.csv", index=False)
        print(f"\nFalse negative rates:")
        print(fn_all[["trajectory","utility_type","pct_threshold","fn_rate"]].to_string(index=False))
    else:
        fn_all = pd.DataFrame()
        print("\nFalse negative analysis: no immediate reward column found in events")

    # ── Immediate reward vs normalized future (if available) ─────────────────
    imm_vs_fut_rows = []
    for traj in args.trajectories:
        sub = all_df[all_df["trajectory"] == traj]
        ev  = all_events.get(traj)
        if ev is None or "immediate_reward" not in ev.columns:
            continue
        create_ev    = ev[ev["action_type"].str.upper().isin({"CREATE","CREATE_BUFFER"})]
        imm_per_obj  = create_ev.groupby("object_id")["immediate_reward"].mean().rename("imm")
        merged       = sub.merge(imm_per_obj, on="object_id", how="inner")
        if len(merged) == 0:
            continue
        for ycol in util_targets:
            if ycol not in merged.columns:
                continue
            rho, n = _spearman(merged["imm"].values.astype(float),
                               merged[ycol].fillna(0).values.astype(float))
            imm_vs_fut_rows.append({
                "trajectory": traj, "future_metric": ycol,
                "spearman_with_imm": rho, "n": n,
            })

    if imm_vs_fut_rows:
        imm_fut_df = pd.DataFrame(imm_vs_fut_rows)
        imm_fut_df.to_csv(norm_dir / "immediate_vs_normalized_future.csv", index=False)
        print(f"\nImmediate reward vs future utility (Spearman ρ):")
        print(imm_fut_df.to_string(index=False))

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots ...")
    for traj in args.trajectories:
        sub = all_df[all_df["trajectory"] == traj]
        if sub.empty or lam_col not in sub.columns:
            continue
        sel = sub[sub[freq_col if freq_col in sub.columns else "retrieval_count"].fillna(0) > 0]

        # 1. Frequency vs utility (4 y-axes)
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        y_options = [
            ("utility_total",                "total utility"),
            ("mean_gain_per_retrieval",       "mean gain / retrieval"),
            (lam_col,                         f"shrunk mean (λ={lam})"),
            ("unique_counterfactual_gain",    "unique counterfactual gain"),
        ]
        for ax, (yc, ylabel) in zip(axes.flat, y_options):
            if yc not in sel.columns:
                continue
            _scatter_with_hist(ax, sel[freq_col].values.astype(float),
                               sel[yc].fillna(0).values.astype(float),
                               "retrieval count", ylabel,
                               f"[{traj}] freq vs {ylabel}", log_x=True)
        fig.suptitle(f"Frequency vs Utility Definitions  [{traj}]", fontsize=11)
        fig.tight_layout()
        fig.savefig(plot_dir / f"frequency_vs_utility_{traj}.png", dpi=120)
        plt.close(fig)

        # 2. Support vs utility
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        for ax, (yc, ylabel) in zip(axes.flat, y_options):
            if yc not in sel.columns:
                continue
            _scatter_with_hist(ax, sel["final_support"].values.astype(float),
                               sel[yc].fillna(0).values.astype(float),
                               "final support", ylabel,
                               f"[{traj}] support vs {ylabel}", log_x=True)
        fig.suptitle(f"Support vs Utility Definitions  [{traj}]", fontsize=11)
        fig.tight_layout()
        fig.savefig(plot_dir / f"support_vs_utility_{traj}.png", dpi=120)
        plt.close(fig)

        # 3. Distribution histograms
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        hist_cols = [
            ("utility_total",                "total utility (all)"),
            (lam_col,                         f"shrunk mean λ={lam} (selected)"),
            ("mean_gain_per_retrieval",       "mean gain / retrieval"),
            ("positive_use_fraction",         "positive use fraction"),
            ("utility_per_support",           "utility / support"),
            ("replaceability_ratio",          "replaceability ratio"),
        ]
        for ax, (col, label) in zip(axes.flat, hist_cols):
            src = sel if col in ["mean_gain_per_retrieval", lam_col,
                                  "positive_use_fraction"] else sub
            if col not in src.columns:
                continue
            _hist(ax, src[col].fillna(0).values.astype(float), col,
                  f"[{traj}] {label}")
        fig.suptitle(f"Utility Metric Distributions  [{traj}]", fontsize=11)
        fig.tight_layout()
        fig.savefig(plot_dir / f"distributions_{traj}.png", dpi=120)
        plt.close(fig)

    # Oracle vs Bandit distribution overlay
    if len(quad_frames) >= 2:
        trajs_present = list(quad_frames.keys())
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        colors = ["steelblue", "coral"]
        for col_idx, col in enumerate([lam_col, "mean_gain_per_retrieval", "utility_per_support"]):
            ax = axes[col_idx]
            for tidx, (traj, df_t) in enumerate(quad_frames.items()):
                _fcc = freq_col if freq_col in df_t.columns else "retrieval_count"
                sel_t = df_t[df_t[_fcc].fillna(0) > 0]
                if col not in sel_t.columns:
                    continue
                vals = sel_t[col].fillna(0).values
                ax.hist(vals, bins=40, alpha=0.6, color=colors[tidx % 2],
                        label=traj, density=True, edgecolor="none")
            ax.set_xlabel(col, fontsize=9)
            ax.set_ylabel("density", fontsize=9)
            ax.set_title(f"Oracle vs Bandit: {col}", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        fig.suptitle("Oracle vs Bandit Utility Distributions (selected objects)", fontsize=11)
        fig.tight_layout()
        fig.savefig(plot_dir / "oracle_vs_bandit_distributions.png", dpi=120)
        plt.close(fig)

    # Rare gem count bar chart
    if quad_frames:
        fig, ax = plt.subplots(figsize=(11, 5))
        quad_names  = ["rare_gem", "workhorse", "rare_low_value",
                        "common_low_value", "middle", "zero_freq"]
        quad_labels = ["Rare gems", "Workhorse", "Rare low-val",
                       "Common low-val", "Middle", "Zero-freq"]
        xs = np.arange(len(quad_names))
        width = 0.35
        for tidx, (traj, df_t) in enumerate(quad_frames.items()):
            counts = [int((df_t["quadrant"] == q).sum()) for q in quad_names]
            ax.bar(xs + tidx * width, counts, width, label=traj,
                   color=["steelblue", "coral"][tidx % 2], alpha=0.8)
        ax.set_xticks(xs + width / 2)
        ax.set_xticklabels(quad_labels, fontsize=9)
        ax.set_ylabel("object count", fontsize=9)
        ax.set_title("Quadrant Object Counts (Q25/Q75, nonzero-freq thresholds)", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(plot_dir / "rare_gem_counts.png", dpi=120)
        plt.close(fig)

    print(f"  Plots saved to {plot_dir}/")

    # ── Q11-Q18 verdicts ──────────────────────────────────────────────────────
    q_verdicts = {}

    # Q11: rare states with high conditional utility?
    n_rare_gems = sum((df_t["quadrant"] == "rare_gem").sum()
                      for df_t in quad_frames.values()) if quad_frames else 0
    q_verdicts["Q11"] = {
        "question": "Are there states with low retrieval frequency but high conditional utility?",
        "n_rare_gems": int(n_rare_gems),
        "answer": "YES" if n_rare_gems > 0 else "NO",
    }

    # Q12: does raw sum undervalue rare states?
    q12_rows = []
    for traj, df_t in quad_frames.items():
        rare = df_t[df_t["quadrant"] == "rare_gem"]
        if len(rare) == 0 or "utility_total" not in rare.columns or lam_col not in rare.columns:
            continue
        # Rank by total vs shrunk: do rare gems rank lower by total?
        rank_total  = df_t["utility_total"].rank(ascending=False)
        rank_shrunk = df_t[lam_col].rank(ascending=False)
        rare_idx    = rare.index
        mean_rank_total  = float(rank_total[rare_idx].mean())
        mean_rank_shrunk = float(rank_shrunk[rare_idx].mean())
        q12_rows.append({
            "traj": traj,
            "mean_rank_by_total":  mean_rank_total,
            "mean_rank_by_shrunk": mean_rank_shrunk,
            "undervalued": mean_rank_total > mean_rank_shrunk,
        })
    answer_q12 = "YES" if any(r["undervalued"] for r in q12_rows) else \
                 "INCONCLUSIVE" if not q12_rows else "NO"
    q_verdicts["Q12"] = {"question": "Does raw summed return systematically undervalue rare states?",
                          "details": q12_rows, "answer": answer_q12}

    # Q13: does shrinkage prevent one-shot objects from appearing artificially valuable?
    q13_rows = []
    for traj in args.trajectories:
        sub = all_df[all_df["trajectory"] == traj]
        if lam_col not in sub.columns or "mean_gain_per_retrieval" not in sub.columns:
            continue
        _fcc2 = freq_col if freq_col in sub.columns else "retrieval_count"
        one_shot = sub[sub[_fcc2].fillna(0) == 1]
        if len(one_shot) == 0:
            continue
        top_q75_raw    = float(sub["mean_gain_per_retrieval"].quantile(0.75))
        top_q75_shrunk = float(sub[lam_col].quantile(0.75))
        frac_os_top_raw    = float((one_shot["mean_gain_per_retrieval"] >= top_q75_raw).mean())
        frac_os_top_shrunk = float((one_shot[lam_col] >= top_q75_shrunk).mean())
        q13_rows.append({
            "traj": traj, "n_one_shot": len(one_shot),
            "frac_top25_by_raw": frac_os_top_raw,
            "frac_top25_by_shrunk": frac_os_top_shrunk,
            "shrinkage_reduces_inflated": frac_os_top_raw > frac_os_top_shrunk,
        })
    answer_q13 = "YES" if any(r["shrinkage_reduces_inflated"] for r in q13_rows) else "INCONCLUSIVE"
    q_verdicts["Q13"] = {"question": "Does shrinkage prevent one-shot rare states from appearing artificially valuable?",
                          "details": q13_rows, "answer": answer_q13}

    # Q14: oracle has more rare gems than bandit?
    gem_by_traj = {}
    for traj, df_t in quad_frames.items():
        frac = float((df_t["quadrant"] == "rare_gem").mean()) if len(df_t) > 0 else 0.0
        gem_by_traj[traj] = frac
    oracle_frac = gem_by_traj.get("oracle", 0)
    bandit_frac = gem_by_traj.get("bandit", 0)
    answer_q14  = "YES" if oracle_frac > bandit_frac else "NO" if bandit_frac > oracle_frac else "INCONCLUSIVE"
    q_verdicts["Q14"] = {"question": "Do oracle states contain more rare gems than Bandit states?",
                          "fracs": gem_by_traj, "answer": answer_q14}

    # Q15: support positively associated with utility after frequency normalization?
    sup_corr_rows = []
    for traj in args.trajectories:
        sub = all_df[all_df["trajectory"] == traj]
        _fcc3 = freq_col if freq_col in sub.columns else "retrieval_count"
        sel = sub[sub[_fcc3].fillna(0) > 0]
        for ycol in [lam_col, "mean_gain_per_retrieval"]:
            if ycol not in sel.columns:
                continue
            rho, n = _spearman(sel["final_support"].values.astype(float),
                               sel[ycol].fillna(0).values.astype(float))
            sup_corr_rows.append({"traj": traj, "metric": ycol, "spearman": rho, "n": n})
    pos_sup = [r["spearman"] > 0.05 for r in sup_corr_rows if r["metric"] == lam_col and np.isfinite(r["spearman"])]
    answer_q15 = "YES" if any(pos_sup) else "NO" if pos_sup else "INCONCLUSIVE"
    q_verdicts["Q15"] = {"question": "Does support remain positively associated with utility after frequency normalization?",
                          "details": sup_corr_rows, "answer": answer_q15}

    # Q16-Q18 (from FN data and predictor results)
    if len(fn_all) > 0:
        fn_shrunk = fn_all[(fn_all["utility_type"] == "shrunk_mean") &
                           (fn_all["pct_threshold"] == 10.0)]
        high_fn = any(fn_shrunk["fn_rate"] > 0.15)
        answer_q16 = "YES" if high_fn else "NO"
    else:
        answer_q16 = "INCONCLUSIVE"
    q_verdicts["Q16"] = {"question": "Does immediate Bandit reward miss rare high-value objects?",
                          "answer": answer_q16}
    q_verdicts["Q17"] = {"question": "Which utility metric is most predictable using WRITE-time features?",
                          "answer": "see utility_prediction/ from predict_future_object_utility.py"}
    q_verdicts["Q18"] = {"question": "Which utility metric best distinguishes useful oracle objects from weak Bandit objects?",
                          "answer": "INCONCLUSIVE" if answer_q14 == "INCONCLUSIVE" else
                                    (f"shrunk_mean_lambda{lam} (oracle has "
                                     f"{oracle_frac:.1%} rare gems vs bandit {bandit_frac:.1%})"
                                     if answer_q14 == "YES" else "NO CLEAR WINNER")}

    print("\n--- Q11–Q18 Verdicts ---")
    for q, v in q_verdicts.items():
        print(f"  {q}: {v['answer']}")

    # ── Report addendum ───────────────────────────────────────────────────────
    report_path = out_dir / "report_addendum.md"
    with open(report_path, "w") as f:
        f.write("# Frequency-Normalized Object Utility — Report Addendum\n\n")
        f.write("## 1. Why Raw Sum Is Insufficient\n\n")
        f.write(
            "Raw total return (G_raw) accumulates across all queries where an object is selected. "
            "A high-frequency state will automatically have a large G_raw even with mediocre per-use "
            "quality. A state retrieved 5 times with mean gain 0.20 may outscore a state retrieved "
            "1000 times with mean gain 0.002 on a per-use basis, yet G_raw says the opposite. "
            "This biases both Q-learning targets and RL credit assignment toward common states.\n\n"
        )
        f.write("## 2. Utility Definitions\n\n")
        for name, desc in [
            ("utility_total",                 "Summed gain across all retrievals (frequency-biased)"),
            ("mean_gain_per_retrieval",       "Mean gain when selected (unbiased, noisy for rare objects)"),
            (f"shrunk_mean_lambda{lam}",       f"James–Stein shrinkage toward global prior (λ={lam}); regularizes rare objects"),
            ("utility_per_support",           "Total gain / final_support: memory efficiency metric"),
            ("utility_per_write_event",       "Total gain / number of write events: write effort efficiency"),
            ("mean_gain_per_opportunity",     f"Mean gain when in top-{args.k_primary}: opportunity-adjusted utility"),
            (f"shrunk_opp_lambda{lam}",        f"Opportunity utility with λ={lam} shrinkage"),
            ("unique_counterfactual_gain",    "LOO NLL gain: how much worse would prediction be without this object?"),
            ("replaceability_ratio",          "Similarity of best alternative / self; high = easily replaced"),
        ]:
            f.write(f"- **{name}**: {desc}\n")
        f.write("\n")

        f.write("## 3. Rare-State Analysis (Q11, Q12, Q13)\n\n")
        q11 = q_verdicts["Q11"]
        f.write(f"**Q11** ({q11['question']}): **{q11['answer']}** — {q11['n_rare_gems']:,} objects identified as rare gems.\n\n")
        q12 = q_verdicts["Q12"]
        f.write(f"**Q12** ({q12['question']}): **{q12['answer']}**\n\n")
        q13 = q_verdicts["Q13"]
        f.write(f"**Q13** ({q13['question']}): **{q13['answer']}**\n\n")

        f.write("## 4. Oracle vs Bandit (Q14, Q18)\n\n")
        f.write(f"**Q14**: {q_verdicts['Q14']['answer']}. ")
        for t, fr in gem_by_traj.items():
            f.write(f"{t}: {fr:.1%} rare gems. ")
        f.write("\n\n")
        f.write(f"**Q18**: {q_verdicts['Q18']['answer']}\n\n")

        f.write("## 5. Support as a Proxy (Q15)\n\n")
        f.write(f"**Q15**: {answer_q15}. ")
        for r in sup_corr_rows:
            f.write(f"[{r['traj']} / {r['metric']}] Spearman ρ={r['spearman']:.3f}. ")
        f.write("\n\n")

        f.write("## 6. Immediate vs Delayed Utility (Q16)\n\n")
        f.write(f"**Q16**: {answer_q16}\n\n")

        f.write("## 7. Utility Predictability (Q17)\n\n")
        f.write(f"**Q17**: See `utility_prediction/` subdirectory. Each target "
                f"(utility_total, shrunk_mean, shrunk_opportunity, unique_gain) has a separate model.\n\n")

        f.write("## 8. Implications for Future Credit Assignment\n\n")
        f.write(
            "- **Monte Carlo targets**: If delayed credit is significant (Q3=YES) and shrunk mean is "
            "more predictable than total utility (Q17), MC returns should normalize by retrieval "
            "frequency before training the Q-WRITE function.\n"
            "- **TD targets**: TD bootstrapping naturally averages over future steps; use shrunk "
            "opportunity utility as the learning target to avoid frequency bias.\n"
            "- **RUDDER-style redistribution**: Unique counterfactual gain provides the per-object "
            "credit signal. Replaceability ratio can modulate credit: irreplaceable objects should "
            "receive proportionally higher redistributed reward.\n"
            "- **No algorithm is chosen here.** This audit documents which utility definition best "
            "separates signal from frequency confounds. The choice of algorithm follows from that result.\n"
        )

        f.write("\n## 9. Research Question Summary\n\n")
        for q, v in q_verdicts.items():
            f.write(f"- **{q}**: {v['answer']} — {v['question']}\n")

    # ── Summary JSON ─────────────────────────────────────────────────────────
    summary = {
        "n_objects_total":    int(len(all_df)),
        "trajectories":       args.trajectories,
        "lambda_primary":     lam,
        "quadrant_counts":    {traj: {q: int((df_t["quadrant"] == q).sum())
                                      for q in ["rare_gem","workhorse","rare_low_value",
                                                "common_low_value","middle","zero_freq"]}
                               for traj, df_t in quad_frames.items()},
        "primary_freq_col":   freq_col,
        "quadrant_method":    "Q25/Q75 from nonzero-frequency objects; middle/zero_freq unclassified",
        "q_verdicts":         {k: v.get("answer", "?") for k, v in q_verdicts.items()},
        "report_addendum":    str(report_path),
    }
    with open(sentinel, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {sentinel}")
    print(f"Report addendum: {report_path}")


if __name__ == "__main__":
    main()

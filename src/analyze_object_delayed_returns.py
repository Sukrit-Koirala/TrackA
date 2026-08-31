"""
analyze_object_delayed_returns.py  --  MVP 4c-0 Stage 4

Joins object provenance with future utility returns and produces:
  - Delay-to-utility distributions (Part 4 of spec)
  - Oracle vs Bandit object utility comparison (Part 8)
  - State-health vs actual-utility correlations (Part 7)
  - Summary CSV files

Answers:
  Q1: Do predictive objects exhibit measurable future READ utility?
  Q2: How long after CREATE/UPDATE/PROMOTE does utility appear?
  Q6: Do Bandit-WRITE objects have worse utility than oracle objects?
  Q7: Does support/entropy/purity predict future utility?

Outputs (in {output}/analysis/):
  oracle_vs_bandit.csv
  state_health_vs_utility.csv
  correlation_tables.csv
  delay_distributions.csv

Usage:
  python src/analyze_object_delayed_returns.py \\
    --output outputs_mvp4c0_delayed_credit_audit_fast \\
    --trajectories oracle bandit \\
    --force
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


POSITIVE_THRESHOLD = 0.001
NEGATIVE_THRESHOLD = -0.001


def _corr(x, y, name_x="x", name_y="y") -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return {"pearson": float("nan"), "spearman": float("nan"), "n": int(mask.sum())}
    xm, ym = x[mask], y[mask]
    try:
        pr = pearsonr(xm, ym).statistic
    except Exception:
        pr = float("nan")
    try:
        sr = spearmanr(xm, ym).statistic
    except Exception:
        sr = float("nan")
    return {"pearson": float(pr), "spearman": float(sr), "n": int(mask.sum())}


def load_and_join(traj: str, out_dir: Path) -> pd.DataFrame:
    prov_path    = out_dir / "provenance"    / f"{traj}_objects.parquet"
    returns_path = out_dir / "object_utility" / f"{traj}_object_returns.parquet"
    if not prov_path.exists() or not returns_path.exists():
        print(f"  WARNING: missing data for {traj}, skipping")
        return pd.DataFrame()

    prov    = pd.read_parquet(prov_path)
    returns = pd.read_parquet(returns_path)
    merged  = prov.merge(returns, on="object_id", how="left",
                          suffixes=("", "_ret"))
    # Fill unselected objects
    merged["G_raw"]           = merged["G_raw"].fillna(0.0)
    merged["n_times_selected"] = merged["n_times_selected"].fillna(0).astype(int)
    merged["is_net_positive"] = merged["is_net_positive"].fillna(False)
    merged["is_net_negative"] = merged["is_net_negative"].fillna(False)
    merged["never_helped"]    = merged["never_helped"].fillna(True)
    merged["trajectory"]      = traj
    return merged


def compute_delay_stats(df: pd.DataFrame, col: str) -> dict:
    vals = df[col].dropna()
    if len(vals) == 0:
        return {}
    return {
        "n":    int(len(vals)),
        "mean": float(vals.mean()),
        "std":  float(vals.std()),
        "p25":  float(vals.quantile(0.25)),
        "p50":  float(vals.quantile(0.50)),
        "p75":  float(vals.quantile(0.75)),
        "p90":  float(vals.quantile(0.90)),
        "p95":  float(vals.quantile(0.95)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output",       required=True)
    ap.add_argument("--trajectories", nargs="+", default=["oracle", "bandit"])
    ap.add_argument("--force",        action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output)
    ana_dir = out_dir / "analysis"
    ana_dir.mkdir(parents=True, exist_ok=True)

    sentinel = ana_dir / "delayed_returns_done.json"
    if sentinel.exists() and not args.force:
        print(f"[cached] {sentinel}")
        return

    all_dfs = {}
    for traj in args.trajectories:
        print(f"\nLoading {traj} ...")
        df = load_and_join(traj, out_dir)
        if not df.empty:
            all_dfs[traj] = df
            print(f"  {len(df):,} objects loaded")

    if not all_dfs:
        print("ERROR: no data available")
        return

    # ── Delay distributions ───────────────────────────────────────────────────
    delay_rows = []
    for traj, df in all_dfs.items():
        sel_df = df[df["n_times_selected"] > 0]
        for col, label in [
            ("delay_to_first_pos",    "create_to_first_pos"),
            ("delay_to_first_subst",  "create_to_first_substantial"),
            ("delay_from_promo_to_pos", "promo_to_first_pos"),
        ]:
            if col not in df.columns:
                continue
            stats = compute_delay_stats(sel_df, col)
            delay_rows.append({"trajectory": traj, "delay_type": label, **stats})

    delay_df = pd.DataFrame(delay_rows)
    delay_df.to_csv(ana_dir / "delay_distributions.csv", index=False)
    print(f"\nDelay distributions:")
    print(delay_df.to_string(index=False))

    # ── Oracle vs Bandit comparison ───────────────────────────────────────────
    ovb_rows = []
    for traj, df in all_dfs.items():
        sel   = df[df["n_times_selected"] > 0]
        all_n = len(df)
        sel_n = len(sel)
        pos_n = int(df["is_net_positive"].sum())
        neg_n = int(df["is_net_negative"].sum())
        neu_n = all_n - pos_n - neg_n
        never_n = int(df["never_helped"].sum()) if "never_helped" in df.columns else 0
        never_sel_n = int((df["n_times_selected"] == 0).sum())

        ovb_rows.append({
            "trajectory":              traj,
            "n_objects":               all_n,
            "n_selected":              sel_n,
            "selection_rate":          sel_n / max(all_n, 1),
            "frac_net_positive":       pos_n / max(all_n, 1),
            "frac_net_negative":       neg_n / max(all_n, 1),
            "frac_neutral":            neu_n / max(all_n, 1),
            "frac_never_selected":     never_sel_n / max(all_n, 1),
            "frac_never_helped":       never_n / max(all_n, 1),
            "median_G_raw":            float(df["G_raw"].median()),
            "mean_G_raw":              float(df["G_raw"].mean()),
            "mean_G_raw_selected":     float(sel["G_raw"].mean()) if len(sel) > 0 else 0.0,
            "mean_n_times_selected":   float(df["n_times_selected"].mean()),
            "median_delay_to_first_pos":
                float(df["delay_to_first_pos"].dropna().median()) if "delay_to_first_pos" in df.columns else float("nan"),
            "mean_final_support":      float(df["final_support"].mean()) if "final_support" in df.columns else float("nan"),
        })

    ovb_df = pd.DataFrame(ovb_rows)
    ovb_df.to_csv(ana_dir / "oracle_vs_bandit.csv", index=False)
    print(f"\nOracle vs Bandit:")
    print(ovb_df.to_string(index=False))

    # ── State health vs future utility correlations ───────────────────────────
    health_cols = ["final_support", "final_entropy", "final_purity",
                   "n_times_selected", "is_persistent", "n_update_buf", "n_update_sta"]
    target_cols = ["G_raw", "mean_gain", "positive_use_frac"]

    corr_rows = []
    for traj, df in all_dfs.items():
        for hc in health_cols:
            if hc not in df.columns:
                continue
            for tc in target_cols:
                if tc not in df.columns:
                    continue
                x = df[hc].astype(float).values
                y = df[tc].astype(float).values
                c = _corr(x, y, hc, tc)
                corr_rows.append({
                    "trajectory": traj,
                    "feature":    hc,
                    "target":     tc,
                    **c,
                })

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(ana_dir / "state_health_vs_utility.csv", index=False)
    print(f"\nState health correlations (Spearman):")
    for traj in args.trajectories:
        sub = corr_df[(corr_df["trajectory"] == traj) & (corr_df["target"] == "G_raw")]
        if not sub.empty:
            print(f"  [{traj}]")
            for _, row in sub.iterrows():
                print(f"    {row['feature']:<30}  rho={row['spearman']:.3f}  n={row['n']}")

    # ── Q1: do objects exhibit measurable utility? ────────────────────────────
    q1_results = {}
    for traj, df in all_dfs.items():
        sel = df[df["n_times_selected"] > 0]
        if len(sel) == 0:
            q1_results[traj] = "INCONCLUSIVE (no selections)"
            continue
        pos_frac = float(sel["is_net_positive"].mean())
        mean_g   = float(sel["G_raw"].mean())
        q1_results[traj] = {
            "pos_frac":       pos_frac,
            "mean_G_raw":     mean_g,
            "n_selected":     len(sel),
            "answer":         "YES" if pos_frac > 0.3 and mean_g > 0 else "INCONCLUSIVE",
        }
    print(f"\nQ1 (measurable future utility): {q1_results}")

    # ── Summary JSON ─────────────────────────────────────────────────────────
    summary = {
        "q1_measurable_utility":  q1_results,
        "oracle_vs_bandit":       ovb_df.to_dict("records"),
        "delay_distributions":    delay_df.to_dict("records"),
        "n_trajectories":         len(all_dfs),
    }
    with open(sentinel, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {sentinel}")


if __name__ == "__main__":
    main()

"""
analyze_defer_with_controls.py  --  MVP 4c-1 Stage 8 (Repaired DEFER analysis)

Extends analyze_defer_shadow_objects.py with MATCHED CONTROLS.

For 500 sampled DEFER decisions we compare against:
  Control A: random stream events matched on (stream_quartile, gpt_nll_bucket,
             memory_occupancy_bucket)
  Control B: actual CREATE decisions matched on the same features

This tests whether DEFER decisions are distinctively bad, or whether the
apparent opportunity cost simply reflects the stream position and memory context.

Table D (per-DEFER row):
  defer_step | shadow_G | ctrl_A_G | ctrl_B_G | diff_A | diff_B | verdict

Aggregate Table D′:
  mean/median/std of (defer shadow - control) across sample

Outputs in {output}/analysis/:
  defer_with_controls.parquet
  defer_with_controls_summary.json  (replaces defer_shadow_done.json)
  shadow_objects.csv                 (backward-compat, same as before)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-8

STREAM_Q_BINS = 4      # quartiles
NLL_BINS      = 4
OCC_BINS      = 4      # memory occupancy


# ── Stratification helpers ────────────────────────────────────────────────────

def _bin_col(series: pd.Series, n_bins: int) -> pd.Series:
    try:
        return pd.qcut(series, n_bins, labels=False, duplicates="drop")
    except Exception:
        return pd.cut(series, n_bins, labels=False)


def _build_strat_key(df: pd.DataFrame,
                     stream_col: str, nll_col: str, occ_col: str,
                     n_stream: int = STREAM_Q_BINS,
                     n_nll:    int = NLL_BINS,
                     n_occ:    int = OCC_BINS) -> pd.Series:
    """Build a discrete strata string key for matching."""
    s_bin  = _bin_col(df[stream_col].fillna(0), n_stream).fillna(0).astype(int)
    nll_bin = _bin_col(df[nll_col].fillna(0),   n_nll).fillna(0).astype(int)
    occ_bin = _bin_col(df[occ_col].fillna(0),   n_occ).fillna(0).astype(int)
    return (s_bin.astype(str) + "_" + nll_bin.astype(str) + "_" + occ_bin.astype(str))


# ── Core analysis ─────────────────────────────────────────────────────────────

def _shadow_utility_for_defer(
    defer_sample: pd.DataFrame,
    oracle_merged: pd.DataFrame,
    args_min_sim: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    For each DEFER row, find the nearest oracle proxy and estimate shadow utility.
    Returns one row per DEFER with shadow_G, proxy_sim, valid_match.
    """
    rows = []
    # Use creation_step proximity if no proto embedding available
    has_stream = "stream_step" in defer_sample.columns

    for _, ev in defer_sample.iterrows():
        defer_step = float(ev.get("stream_step", 0))

        time_diffs = (oracle_merged["creation_step"] - defer_step).abs()
        if len(time_diffs) == 0:
            rows.append({"defer_stream_step": defer_step, "shadow_G": float("nan"),
                         "proxy_sim": float("nan"), "valid_match": False,
                         "proxy_object_id": -1})
            continue
        best_iloc  = int(time_diffs.values.argmin())
        proxy      = oracle_merged.iloc[best_iloc]
        best_sim   = float("nan")

        rows.append({
            "defer_stream_step": defer_step,
            "shadow_G":           float(proxy.get("G_raw", float("nan"))),
            "proxy_sim":          best_sim,
            "valid_match":        True,
            "proxy_object_id":    int(proxy["object_id"]),
            "proxy_n_selected":   int(proxy.get("n_times_selected", 0)),
            "proxy_net_positive": bool(proxy.get("is_net_positive", False)),
        })
    return pd.DataFrame(rows)


def _matched_mean(
    sample_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    strat_col: str,
    utility_col: str,
    rng: np.random.Generator,
    n_per_stratum: int = 5,
) -> pd.Series:
    """
    For each row in sample_df, draw up to n_per_stratum matches from pool_df
    with the same strata key. Return the mean utility across matches (one per sample row).
    """
    strata_groups = pool_df.groupby(strat_col)

    results = []
    for _, row in sample_df.iterrows():
        key = row.get(strat_col, None)
        if key is None or key not in strata_groups.groups:
            results.append(float("nan"))
            continue
        pool_sub = pool_df.loc[strata_groups.groups[key]]
        if len(pool_sub) == 0:
            results.append(float("nan"))
            continue
        n = min(n_per_stratum, len(pool_sub))
        chosen = pool_sub.sample(n=n, random_state=int(rng.integers(0, 2**31)))
        results.append(float(chosen[utility_col].mean()))
    return pd.Series(results, index=sample_df.index)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output",   required=True)
    ap.add_argument("--n_sample", type=int,   default=500)
    ap.add_argument("--min_sim",  type=float, default=0.70)
    ap.add_argument("--seed",     type=int,   default=42)
    ap.add_argument("--force",    action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output)
    ana_dir = out_dir / "analysis"
    ana_dir.mkdir(parents=True, exist_ok=True)

    sentinel = ana_dir / "defer_with_controls_summary.json"
    if sentinel.exists() and not args.force:
        print(f"[cached] {sentinel}")
        return

    rng = np.random.default_rng(args.seed)

    # ── Load data ─────────────────────────────────────────────────────────────
    bandit_ev_path  = out_dir / "provenance" / "bandit_events.parquet"
    oracle_obj_path = out_dir / "provenance" / "oracle_objects.parquet"
    oracle_ret_path = out_dir / "object_utility" / "oracle_object_returns.parquet"

    missing = [p for p in [bandit_ev_path, oracle_obj_path] if not p.exists()]
    if missing:
        print(f"ERROR: missing files: {[str(m) for m in missing]}")
        print("  Run build_write_object_provenance.py first.")
        return

    bandit_events  = pd.read_parquet(bandit_ev_path)
    oracle_objects = pd.read_parquet(oracle_obj_path)
    oracle_returns = pd.DataFrame()
    if oracle_ret_path.exists():
        oracle_returns = pd.read_parquet(oracle_ret_path)

    # Try alternate utility source
    if len(oracle_returns) == 0:
        for cand in [
            out_dir / "retrospective_utility" / "oracle_object_utility.parquet",
            out_dir / "object_utility" / "oracle_object_returns.parquet",
        ]:
            if cand.exists():
                oracle_returns = pd.read_parquet(cand)
                print(f"  Using utility from: {cand.name}")
                break

    if len(oracle_returns) == 0:
        print("WARNING: No oracle utility file found. Shadow G will be NaN.")
        oracle_returns = pd.DataFrame({"object_id": oracle_objects["object_id"],
                                        "G_raw": 0.0,
                                        "n_times_selected": 0,
                                        "is_net_positive": False,
                                        "positive_use_frac": 0.0})

    # Merge oracle utility onto oracle objects
    merge_cols = ["object_id", "G_raw", "n_times_selected",
                  "is_net_positive", "positive_use_frac"]
    merge_cols = [c for c in merge_cols if c in oracle_returns.columns]
    oracle_merged = oracle_objects.merge(
        oracle_returns[merge_cols], on="object_id", how="left")
    oracle_merged["G_raw"] = oracle_merged.get("G_raw", pd.Series(0.0)).fillna(0.0)

    # ── Separate event types ──────────────────────────────────────────────────
    act_col = "action_type"
    bandit_events["_atype"] = bandit_events[act_col].str.upper().fillna("")

    defer_all  = bandit_events[bandit_events["_atype"] == "DEFER"].copy()
    create_all = bandit_events[bandit_events["_atype"].isin(
        ["CREATE_BUFFER", "CREATE", "CREATE_STATE"])].copy()
    other_all  = bandit_events[~bandit_events["_atype"].isin(
        ["DEFER", "CREATE_BUFFER", "CREATE", "CREATE_STATE"])].copy()

    print(f"Bandit events:")
    print(f"  DEFER:       {len(defer_all):,}")
    print(f"  CREATE*:     {len(create_all):,}")
    print(f"  Other (UPDATE/PROMOTE): {len(other_all):,}")
    print(f"  Total:       {len(bandit_events):,}")

    if len(defer_all) == 0:
        print("WARNING: No DEFER events in bandit trajectory.")
        summary = {"n_defer_total": 0, "n_sampled": 0,
                   "verdict": "NO_DEFER",
                   "msg": "No DEFER events found in bandit_events.parquet"}
        with open(sentinel, "w") as f:
            json.dump(summary, f, indent=2)
        return

    # ── Sample DEFER ──────────────────────────────────────────────────────────
    n_sample   = min(args.n_sample, len(defer_all))
    defer_samp = defer_all.sample(n=n_sample, random_state=42).reset_index(drop=True)
    print(f"\nSampled {n_sample:,} DEFER events for matched control analysis")

    # ── Stratification keys ───────────────────────────────────────────────────
    # Need: stream_quartile, gpt_nll bucket, memory_occupancy bucket
    # gpt_nll from events (if present), else fill 0
    nll_col = "gpt_nll" if "gpt_nll" in bandit_events.columns else \
              "nll_gpt" if "nll_gpt" in bandit_events.columns else None
    occ_col = "n_objects_at_write" if "n_objects_at_write" in bandit_events.columns else \
              "memory_size" if "memory_size" in bandit_events.columns else None

    if nll_col is None:
        for _df in [bandit_events, defer_samp, create_all, other_all]:
            _df["_gpt_nll"] = 0.0
        nll_col = "_gpt_nll"
    if occ_col is None:
        for _df in [bandit_events, defer_samp, create_all, other_all]:
            _df["_occ"] = 0.0
        occ_col = "_occ"

    # Build strata on full event pool (for quantile bins)
    def _add_strata(df_target: pd.DataFrame,
                    df_ref: pd.DataFrame = bandit_events) -> pd.DataFrame:
        df = df_target.copy()
        # stream quartile
        step_q = pd.qcut(df_ref["stream_step"].fillna(0), STREAM_Q_BINS,
                         labels=False, duplicates="drop")
        step_edges = pd.qcut(df_ref["stream_step"].fillna(0), STREAM_Q_BINS,
                              duplicates="drop", retbins=True)[1]
        nll_edges  = pd.qcut(df_ref[nll_col].fillna(0),  NLL_BINS,
                              duplicates="drop", retbins=True)[1]
        occ_edges  = pd.qcut(df_ref[occ_col].fillna(0),  OCC_BINS,
                              duplicates="drop", retbins=True)[1]

        df["_sq"]  = pd.cut(df["stream_step"].fillna(0), bins=step_edges,
                            labels=False, include_lowest=True).fillna(0).astype(int)
        df["_nll"] = pd.cut(df[nll_col].fillna(0),       bins=nll_edges,
                            labels=False, include_lowest=True).fillna(0).astype(int)
        df["_occ"] = pd.cut(df[occ_col].fillna(0),       bins=occ_edges,
                            labels=False, include_lowest=True).fillna(0).astype(int)
        df["strata_key"] = (df["_sq"].astype(str) + "_" +
                            df["_nll"].astype(str) + "_" +
                            df["_occ"].astype(str))
        return df

    defer_samp_s = _add_strata(defer_samp)
    create_all_s = _add_strata(create_all)
    other_all_s  = _add_strata(other_all)

    # ── Shadow utility (oracle proxy) ─────────────────────────────────────────
    shadow_df = _shadow_utility_for_defer(defer_samp, oracle_merged, args.min_sim, rng)
    shadow_df.to_csv(ana_dir / "shadow_objects.csv", index=False)
    print(f"Shadow objects computed: {len(shadow_df):,}")

    defer_samp_s["shadow_G"] = shadow_df["shadow_G"].values
    defer_samp_s["proxy_valid"] = shadow_df["valid_match"].values

    # ── Control A: random stream events matched on strata ────────────────────
    # Pool = everything EXCEPT the sampled DEFER events
    sampled_steps = set(defer_samp_s["stream_step"].values)
    pool_A = bandit_events[~bandit_events["stream_step"].isin(sampled_steps)].copy()
    # Give pool_A a proxy G via oracle temporal proximity
    pool_A_shadow = _shadow_utility_for_defer(pool_A.head(5000), oracle_merged,
                                               args.min_sim, rng)
    pool_A_sub = pool_A.head(5000).copy()
    pool_A_sub["proxy_G"] = pool_A_shadow["shadow_G"].values
    pool_A_sub = _add_strata(pool_A_sub)

    ctrl_A_G = _matched_mean(
        defer_samp_s, pool_A_sub, "strata_key", "proxy_G", rng)
    defer_samp_s["ctrl_A_G"] = ctrl_A_G.values

    # ── Control B: actual CREATE events matched on strata ────────────────────
    create_shadow = _shadow_utility_for_defer(
        create_all.head(5000), oracle_merged, args.min_sim, rng)
    create_pool = create_all.head(5000).copy()
    create_pool["proxy_G"] = create_shadow["shadow_G"].values
    create_pool_s = _add_strata(create_pool)

    ctrl_B_G = _matched_mean(
        defer_samp_s, create_pool_s, "strata_key", "proxy_G", rng)
    defer_samp_s["ctrl_B_G"] = ctrl_B_G.values

    # ── Table D: per-row comparison ───────────────────────────────────────────
    defer_samp_s["diff_vs_ctrl_A"] = defer_samp_s["shadow_G"] - defer_samp_s["ctrl_A_G"]
    defer_samp_s["diff_vs_ctrl_B"] = defer_samp_s["shadow_G"] - defer_samp_s["ctrl_B_G"]
    defer_samp_s["verdict"] = defer_samp_s.apply(
        lambda r: ("DEFER_MISSES_SUBSTANTIALLY"
                   if (r["shadow_G"] > 0.01 and
                       r["diff_vs_ctrl_A"] > 0.005 and
                       r["diff_vs_ctrl_B"] > 0.005)
                   else "DEFER_COMPARABLE_TO_CONTROLS"
                   if (abs(r["diff_vs_ctrl_A"]) < 0.005 or
                       abs(r["diff_vs_ctrl_B"]) < 0.005)
                   else "INCONCLUSIVE"),
        axis=1,
    )

    out_path = ana_dir / "defer_with_controls.parquet"
    defer_samp_s.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(defer_samp_s):,} rows)")

    # ── Table D′: aggregates ──────────────────────────────────────────────────
    valid = defer_samp_s[defer_samp_s["proxy_valid"].fillna(True)].copy()
    n_valid = len(valid)

    def _agg(col: str) -> dict:
        v = valid[col].dropna()
        if len(v) == 0:
            return {"mean": float("nan"), "median": float("nan"),
                    "std": float("nan"), "n": 0}
        return {"mean": float(v.mean()), "median": float(v.median()),
                "std": float(v.std()), "n": len(v)}

    shadow_agg = _agg("shadow_G")
    ctrl_a_agg = _agg("ctrl_A_G")
    ctrl_b_agg = _agg("ctrl_B_G")
    diff_a_agg = _agg("diff_vs_ctrl_A")
    diff_b_agg = _agg("diff_vs_ctrl_B")

    # How many DEFERs distinctively miss utility vs controls?
    n_miss_sub = int((valid["verdict"] == "DEFER_MISSES_SUBSTANTIALLY").sum())
    n_comparable = int((valid["verdict"] == "DEFER_COMPARABLE_TO_CONTROLS").sum())

    print(f"\n=== Table D (DEFER vs Controls) ===")
    print(f"  Valid shadow matches:     {n_valid:,} / {n_sample:,}")
    print(f"  Shadow G (proxy):         mean={shadow_agg['mean']:.4f}  "
          f"med={shadow_agg['median']:.4f}  std={shadow_agg['std']:.4f}")
    print(f"  Control A G (random):     mean={ctrl_a_agg['mean']:.4f}  "
          f"med={ctrl_a_agg['median']:.4f}  std={ctrl_a_agg['std']:.4f}")
    print(f"  Control B G (CREATE):     mean={ctrl_b_agg['mean']:.4f}  "
          f"med={ctrl_b_agg['median']:.4f}  std={ctrl_b_agg['std']:.4f}")
    print(f"  Diff (shadow - ctrl A):   mean={diff_a_agg['mean']:.4f}  "
          f"med={diff_a_agg['median']:.4f}")
    print(f"  Diff (shadow - ctrl B):   mean={diff_b_agg['mean']:.4f}  "
          f"med={diff_b_agg['median']:.4f}")
    print(f"  DEFER misses substantially: {n_miss_sub:,} / {n_valid:,} "
          f"({n_miss_sub/max(n_valid,1):.1%})")
    print(f"  DEFER comparable:           {n_comparable:,} / {n_valid:,}")

    # Conclusion
    miss_frac = float(n_miss_sub / max(n_valid, 1))
    if np.isnan(shadow_agg["mean"]):
        q4_answer = "INCONCLUSIVE"
    elif miss_frac > 0.3 and diff_a_agg["mean"] > 0.005:
        q4_answer = "YES_SIGNIFICANT"
    elif miss_frac > 0.15 or diff_b_agg["mean"] > 0.002:
        q4_answer = "YES_MODERATE"
    elif diff_a_agg["mean"] < 0 and diff_b_agg["mean"] < 0:
        q4_answer = "NO_DEFER_AVOIDS_LOW_UTILITY"
    else:
        q4_answer = "INCONCLUSIVE"

    print(f"\n  Q4 verdict: {q4_answer}")

    summary = {
        "n_defer_total":    int(len(defer_all)),
        "n_sampled":        int(n_sample),
        "n_valid_matches":  int(n_valid),
        "min_sim":          args.min_sim,
        "shadow_G":         shadow_agg,
        "ctrl_A_G_random":  ctrl_a_agg,
        "ctrl_B_G_create":  ctrl_b_agg,
        "diff_vs_ctrl_A":   diff_a_agg,
        "diff_vs_ctrl_B":   diff_b_agg,
        "n_miss_substantially":  n_miss_sub,
        "n_comparable_to_ctrl":  n_comparable,
        "miss_frac":             float(miss_frac),
        "q4_defer_misses_utility": {
            "mean_shadow_G":       shadow_agg["mean"],
            "diff_vs_random_ctrl": diff_a_agg["mean"],
            "diff_vs_create_ctrl": diff_b_agg["mean"],
            "miss_frac":           float(miss_frac),
            "answer":              q4_answer,
        },
        # Backward-compat key used by run_mvp4c1_repaired_credit_audit.py
        "q4_answer": q4_answer,
    }

    with open(sentinel, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Also write the legacy sentinel for backward compat with orchestrator
    legacy = out_dir / "analysis" / "defer_shadow_done.json"
    with open(legacy, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nSaved: {sentinel}")


if __name__ == "__main__":
    main()

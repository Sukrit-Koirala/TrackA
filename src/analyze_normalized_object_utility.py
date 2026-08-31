"""
analyze_normalized_object_utility.py  --  MVP 4c-0 Patch: Frequency-Normalized Utility

Computes the full per-object utility vector, separating:
  - total usefulness   (G_raw, discounted)
  - conditional usefulness (mean/median per retrieval)
  - confidence (shrunk means, bootstrap CIs)
  - efficiency (per support, per write event)
  - opportunity-awareness (opportunity@K, gain per opportunity)
  - uniqueness (replaceability proxy from sim data)

Inputs (all from prior MVP 4c-0 stages):
  read_usage/{traj}_object_usage.parquet     -- per (query, object) gains, rank, sim
  provenance/{traj}_objects.parquet          -- object metadata
  provenance/{traj}_events.parquet           -- write event counts
  object_utility/discounted_returns.parquet  -- discounted G per gamma

Outputs:
  object_utility/normalized_object_utility.parquet  -- full utility vector per object
  object_utility/opportunity_utility.parquet         -- opportunity@K breakdown
  object_utility/utility_metric_summary.csv
  object_utility/norm_config.json                    -- epsilon and lambda values used

Usage:
  python src/analyze_normalized_object_utility.py \\
    --trajectory oracle \\
    --output outputs_mvp4c0_delayed_credit_audit_fast \\
    --lambdas 5 10 20 \\
    --k_list 1 4 8 16 \\
    --n_bootstrap 200 \\
    --force
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _bootstrap_ci(gains: np.ndarray, n: int = 200, ci: float = 0.95) -> tuple[float, float]:
    if len(gains) < 5:
        return float("nan"), float("nan")
    rng   = np.random.default_rng(0)
    means = np.array([gains[rng.integers(0, len(gains), len(gains))].mean()
                      for _ in range(n)])
    lo = float(np.percentile(means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(means, (1 + ci) / 2 * 100))
    return lo, hi


def _shrunk(mean_j: float, n_j: int, prior: float, lam: float) -> float:
    w = n_j / (n_j + lam)
    return w * mean_j + (1.0 - w) * prior


def compute_utility_vector(
    usage_df: pd.DataFrame,
    objects_df: pd.DataFrame,
    events_df: pd.DataFrame,
    disc_df: pd.DataFrame,
    traj: str,
    lambdas: list[int],
    k_list: list[int],
    n_bootstrap: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Returns (norm_utility_df, opportunity_df, config_dict).
    """
    # ── Filter to primary read mode ───────────────────────────────────────────
    if "read_mode" in usage_df.columns:
        modes   = usage_df["read_mode"].unique()
        primary = "fixed" if "fixed" in modes else modes[0]
        df      = usage_df[usage_df["read_mode"] == primary].copy()
    else:
        df = usage_df.copy()

    # Map state_row → object_id using objects_df
    row_to_obj = dict(zip(objects_df["state_row_index"].astype(int),
                          objects_df["object_id"].astype(int)))
    if "object_id" not in df.columns:
        df["object_id"] = df["state_row"].map(row_to_obj).fillna(-1).astype(int)
    df = df[df["object_id"] >= 0].copy()

    # ── Write event counts ────────────────────────────────────────────────────
    write_types = {"CREATE", "CREATE_BUFFER", "UPDATE_BUFFER", "UPDATE_STATE", "PROMOTE"}
    if len(events_df) > 0 and "action_type" in events_df.columns:
        ev_counts = (events_df[events_df["action_type"].str.upper().isin(write_types)]
                     .groupby("object_id").size().rename("n_write_events"))
    else:
        ev_counts = pd.Series(dtype=int, name="n_write_events")

    # ── Global prior and epsilon ──────────────────────────────────────────────
    all_gains  = df["gain_j"].values
    nonzero    = np.abs(all_gains[all_gains != 0])
    epsilon    = float(np.percentile(nonzero, 10)) if len(nonzero) > 0 else 1e-4
    epsilon    = max(epsilon, 1e-5)
    prior_mean = float(all_gains.mean()) if len(all_gains) > 0 else 0.0

    config = {
        "epsilon":        epsilon,
        "prior_mean":     prior_mean,
        "lambdas":        lambdas,
        "k_list":         k_list,
        "n_bootstrap":    n_bootstrap,
        "trajectory":     traj,
        "primary_mode":   primary if "read_mode" in usage_df.columns else "unknown",
        "n_usage_records": len(df),
    }

    # ── Per-query best-alternative sim (for replaceability) ───────────────────
    # For each query, sort objects by sim; best_alt_sim for object j = sim of
    # highest-sim object that is NOT j, among those with rank <= max(k_list).
    max_k = max(k_list)
    pool  = df[df["rank"] <= max_k][["query_id", "object_id", "sim", "rank"]].copy()
    # For each query: create a lookup of (object_id → best_alternative_sim)
    alt_sim_records = []
    for qid, qgrp in pool.groupby("query_id"):
        sims = qgrp.sort_values("sim", ascending=False)
        for _, row in sims.iterrows():
            others     = sims[sims["object_id"] != row["object_id"]]
            best_alt   = float(others["sim"].iloc[0]) if len(others) > 0 else float("nan")
            alt_sim_records.append({
                "query_id":    qid,
                "object_id":   int(row["object_id"]),
                "sim_self":    float(row["sim"]),
                "best_alt_sim": best_alt,
            })
    alt_sim_df = pd.DataFrame(alt_sim_records)

    df_aug = df.merge(alt_sim_df[["query_id", "object_id", "best_alt_sim"]],
                      on=["query_id", "object_id"], how="left")

    # ── Per-object aggregation ────────────────────────────────────────────────
    obj_rows = []
    for obj_id, grp in df_aug.groupby("object_id"):
        gains      = grp["gain_j"].values
        sims       = grp["sim"].values if "sim" in grp.columns else np.ones(len(grp))
        alt_sims   = grp["best_alt_sim"].values if "best_alt_sim" in grp.columns else np.full(len(grp), float("nan"))
        n_ret      = len(gains)

        # Support and write events
        obj_meta   = objects_df[objects_df["object_id"] == obj_id]
        final_sup  = float(obj_meta["final_support"].iloc[0]) if len(obj_meta) > 0 else 0.0
        cre_step   = float(obj_meta["creation_step"].iloc[0]) if len(obj_meta) > 0 else 0.0
        prm_step   = float(obj_meta["promotion_step"].iloc[0]) if len(obj_meta) > 0 else -1.0
        is_persist = bool(obj_meta["is_persistent"].iloc[0]) if "is_persistent" in obj_meta.columns and len(obj_meta) > 0 else prm_step >= 0
        n_write    = int(ev_counts.get(obj_id, 1))

        # Pos / neg / neutral counts
        n_pos  = int((gains >  epsilon).sum())
        n_neg  = int((gains < -epsilon).sum())
        n_neu  = n_ret - n_pos - n_neg

        # Per retrieval stats
        mean_g   = float(gains.mean())
        median_g = float(np.median(gains))
        std_g    = float(gains.std()) if n_ret > 1 else 0.0
        p25_g    = float(np.percentile(gains, 25))
        p75_g    = float(np.percentile(gains, 75))
        p90_g    = float(np.percentile(gains, 90))
        g_max    = float(gains.max())
        g_min    = float(gains.min())

        # Shrunk means (both priors)
        shrunk = {}
        for lam in lambdas:
            shrunk[f"shrunk_mean_lambda{lam}"] = _shrunk(mean_g, n_ret, prior_mean, lam)
            shrunk[f"shrunk_mean_lambda{lam}_prior0"] = _shrunk(mean_g, n_ret, 0.0, lam)

        # Bootstrap CI
        ci_lo, ci_hi = _bootstrap_ci(gains, n_bootstrap)

        # Replaceability
        valid_alt     = alt_sims[np.isfinite(alt_sims)]
        mean_alt_sim  = float(valid_alt.mean()) if len(valid_alt) > 0 else float("nan")
        mean_self_sim = float(sims.mean())
        repl_ratio    = float(mean_alt_sim / (mean_self_sim + 1e-8)) if np.isfinite(mean_alt_sim) else float("nan")
        frac_good_sub = float((valid_alt > 0.9 * sims[np.isfinite(alt_sims)]).mean()) \
                        if len(valid_alt) > 0 else float("nan")

        # Discounted totals (from disc_df)
        disc_vals = {}
        if len(disc_df) > 0 and "object_id" in disc_df.columns:
            obj_disc = disc_df[(disc_df["object_id"] == obj_id) &
                               (disc_df["trajectory"] == traj)]
            for _, dr in obj_disc.iterrows():
                gval  = float(dr.get("gamma", 1.0))
                gname = str(gval).replace(".", "")
                disc_vals[f"G_disc_{gname}"] = float(dr.get("G_discounted", float("nan")))

        row = {
            "object_id":               int(obj_id),
            "trajectory":              traj,
            "creation_step":           cre_step,
            "promotion_step":          prm_step,
            "is_promoted":             is_persist,
            "final_support":           final_sup,
            "n_write_events":          n_write,
            # Retrieval counts
            "retrieval_count":         n_ret,
            # Total utility
            "utility_total":           float(gains.sum()),
            # Per-retrieval distribution
            "mean_gain_per_retrieval": mean_g,
            "median_gain_per_retrieval": median_g,
            "std_gain_per_retrieval":  std_g,
            "p25_gain_per_retrieval":  p25_g,
            "p75_gain_per_retrieval":  p75_g,
            "p90_gain_per_retrieval":  p90_g,
            "max_gain":                g_max,
            "min_gain":                g_min,
            # Use rates
            "n_positive_uses":         n_pos,
            "n_negative_uses":         n_neg,
            "n_neutral_uses":          n_neu,
            "positive_use_fraction":   n_pos / max(n_ret, 1),
            "negative_use_fraction":   n_neg / max(n_ret, 1),
            "near_zero_use_fraction":  n_neu / max(n_ret, 1),
            # Shrunk means
            **shrunk,
            # Bootstrap CI
            "mean_gain_ci_lo":         ci_lo,
            "mean_gain_ci_hi":         ci_hi,
            # Efficiency
            "utility_per_support":     float(gains.sum()) / max(final_sup, 1),
            "utility_per_write_event": float(gains.sum()) / max(n_write, 1),
            # Replaceability
            "mean_self_sim":           mean_self_sim,
            "mean_best_alt_sim":       mean_alt_sim,
            "replaceability_ratio":    repl_ratio,
            "frac_queries_good_substitute": frac_good_sub,
            # Unique counterfactual = same as total in LOO setup
            "unique_counterfactual_gain":    float(gains.sum()),
            "unique_gain_after_replacement": float(gains.sum()),
            **disc_vals,
        }
        obj_rows.append(row)

    # ── Objects never retrieved — add zero rows ────────────────────────────────
    retrieved_ids = set(df_aug["object_id"].unique())
    for _, obj_row in objects_df.iterrows():
        oid = int(obj_row["object_id"])
        if oid in retrieved_ids:
            continue
        n_write = int(ev_counts.get(oid, 1))
        row = {
            "object_id":               oid,
            "trajectory":              traj,
            "creation_step":           float(obj_row.get("creation_step", 0)),
            "promotion_step":          float(obj_row.get("promotion_step", -1)),
            "is_promoted":             float(obj_row.get("promotion_step", -1)) >= 0,
            "final_support":           float(obj_row.get("final_support", 0)),
            "n_write_events":          n_write,
            "retrieval_count":         0,
            "utility_total":           0.0,
            "mean_gain_per_retrieval": 0.0,
            "median_gain_per_retrieval": 0.0,
            "std_gain_per_retrieval":  0.0,
            "p25_gain_per_retrieval":  0.0,
            "p75_gain_per_retrieval":  0.0,
            "p90_gain_per_retrieval":  0.0,
            "max_gain":                0.0,
            "min_gain":                0.0,
            "n_positive_uses":         0,
            "n_negative_uses":         0,
            "n_neutral_uses":          0,
            "positive_use_fraction":   0.0,
            "negative_use_fraction":   0.0,
            "near_zero_use_fraction":  1.0,
            **{f"shrunk_mean_lambda{lam}":       _shrunk(0.0, 0, prior_mean, lam) for lam in lambdas},
            **{f"shrunk_mean_lambda{lam}_prior0": 0.0 for lam in lambdas},
            "mean_gain_ci_lo":         float("nan"),
            "mean_gain_ci_hi":         float("nan"),
            "utility_per_support":     0.0,
            "utility_per_write_event": 0.0,
            "mean_self_sim":           float("nan"),
            "mean_best_alt_sim":       float("nan"),
            "replaceability_ratio":    float("nan"),
            "frac_queries_good_substitute": float("nan"),
            "unique_counterfactual_gain":    0.0,
            "unique_gain_after_replacement": 0.0,
        }
        obj_rows.append(row)

    norm_df = pd.DataFrame(obj_rows)

    # ── Opportunity@K ─────────────────────────────────────────────────────────
    n_queries = df["query_id"].nunique()
    opp_rows  = []
    for obj_id, grp in df.groupby("object_id"):
        orow = {"object_id": int(obj_id), "trajectory": traj}
        for k in k_list:
            in_topk      = grp[grp["rank"] <= k]
            n_opp        = len(in_topk)
            total_opp_g  = float(in_topk["gain_j"].sum()) if n_opp > 0 else 0.0
            mean_opp_g   = float(in_topk["gain_j"].mean()) if n_opp > 0 else 0.0
            orow[f"opp_at_{k}"]           = n_opp
            orow[f"opp_frac_at_{k}"]      = n_opp / max(n_queries, 1)
            orow[f"utility_per_opp_at_{k}"] = mean_opp_g
            # Shrunk opportunity utility
            for lam in lambdas:
                orow[f"shrunk_opp_lambda{lam}_at_{k}"] = _shrunk(
                    mean_opp_g, n_opp, prior_mean, lam)

        # Merge from norm_df for join
        orow["retrieval_count"] = int(len(grp))
        opp_rows.append(orow)

    opp_df = pd.DataFrame(opp_rows) if opp_rows else pd.DataFrame()
    # Add to norm_df: opportunity columns at primary K=8
    primary_k = 8 if 8 in k_list else k_list[-1]
    if len(opp_df) > 0 and "object_id" in opp_df.columns:
        merge_cols = (
            ["object_id", "trajectory", f"opp_at_{primary_k}",
             f"opp_frac_at_{primary_k}", f"utility_per_opp_at_{primary_k}"]
            + [f"shrunk_opp_lambda{lam}_at_{primary_k}" for lam in lambdas]
        )
        existing_cols = [c for c in merge_cols if c in opp_df.columns]
        if len(existing_cols) > 2:   # need more than just merge keys
            rename_map = {
                f"opp_at_{primary_k}":             "relevant_opportunity_count",
                f"opp_frac_at_{primary_k}":        "opportunity_fraction",
                f"utility_per_opp_at_{primary_k}": "mean_gain_per_opportunity",
                **{f"shrunk_opp_lambda{lam}_at_{primary_k}":
                   f"shrunk_opportunity_lambda{lam}" for lam in lambdas},
            }
            norm_df = norm_df.merge(
                opp_df[existing_cols].rename(columns=rename_map),
                on=["object_id", "trajectory"], how="left",
            )

    # Ensure opportunity columns always exist (filled with 0 when absent or NaN)
    for col, fill in [("relevant_opportunity_count", 0),
                      ("mean_gain_per_opportunity",  0.0)]:
        if col not in norm_df.columns:
            norm_df[col] = fill
        else:
            norm_df[col] = norm_df[col].fillna(fill)

    return norm_df, opp_df, config


def make_summary(norm_df: pd.DataFrame, traj: str, lambdas: list[int]) -> pd.DataFrame:
    rows = []
    metrics = (
        ["utility_total", "mean_gain_per_retrieval", "median_gain_per_retrieval",
         "positive_use_fraction", "negative_use_fraction",
         "utility_per_support", "utility_per_write_event",
         "unique_counterfactual_gain", "replaceability_ratio"]
        + [f"shrunk_mean_lambda{lam}" for lam in lambdas]
        + ["mean_gain_per_opportunity"]
    )
    for m in metrics:
        if m not in norm_df.columns:
            continue
        vals = norm_df[m].dropna()
        rows.append({
            "trajectory": traj,
            "metric":     m,
            "n":          int(len(vals)),
            "mean":       float(vals.mean()),
            "std":        float(vals.std()),
            "p25":        float(vals.quantile(0.25)),
            "p50":        float(vals.quantile(0.50)),
            "p75":        float(vals.quantile(0.75)),
            "p90":        float(vals.quantile(0.90)),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory",   required=True, choices=["oracle", "bandit"])
    ap.add_argument("--output",       required=True)
    ap.add_argument("--lambdas",      nargs="+", type=int, default=[5, 10, 20])
    ap.add_argument("--k_list",       nargs="+", type=int, default=[1, 4, 8, 16])
    ap.add_argument("--n_bootstrap",  type=int, default=200)
    ap.add_argument("--force",        action="store_true")
    args = ap.parse_args()

    traj    = args.trajectory
    out_dir = Path(args.output)
    util_dir = out_dir / "object_utility"
    util_dir.mkdir(parents=True, exist_ok=True)

    norm_path = util_dir / "normalized_object_utility.parquet"
    opp_path  = util_dir / "opportunity_utility.parquet"
    cfg_path  = util_dir / "norm_config.json"

    # Check if done for this traj
    if not args.force and norm_path.exists():
        existing = pd.read_parquet(norm_path)
        if traj in existing.get("trajectory", pd.Series()).values:
            print(f"[cached] normalized utility for '{traj}'")
            return

    # Load inputs
    use_path  = out_dir / "read_usage"    / f"{traj}_object_usage.parquet"
    obj_path  = out_dir / "provenance"    / f"{traj}_objects.parquet"
    ev_path   = out_dir / "provenance"    / f"{traj}_events.parquet"
    disc_path = out_dir / "object_utility" / "discounted_returns.parquet"

    if not use_path.exists():
        raise FileNotFoundError(f"{use_path}  (run log_read_object_usage.py first)")
    if not obj_path.exists():
        raise FileNotFoundError(f"{obj_path}  (run build_write_object_provenance.py first)")

    print(f"[{traj}] Loading usage ({use_path.name}) ...")
    usage_df   = pd.read_parquet(use_path)
    objects_df = pd.read_parquet(obj_path)
    events_df  = pd.read_parquet(ev_path) if ev_path.exists() else pd.DataFrame()
    disc_df    = pd.read_parquet(disc_path) if disc_path.exists() else pd.DataFrame()

    print(f"  {len(usage_df):,} usage records  |  {len(objects_df):,} objects")

    norm_df, opp_df, config = compute_utility_vector(
        usage_df, objects_df, events_df, disc_df,
        traj, args.lambdas, args.k_list, args.n_bootstrap,
    )

    # Merge / append across trajectories
    for path, df_new in [(norm_path, norm_df), (opp_path, opp_df)]:
        if path.exists() and not args.force:
            old = pd.read_parquet(path)
            old = old[old["trajectory"] != traj]
            df_new = pd.concat([old, df_new], ignore_index=True)
        df_new.to_parquet(path, index=False)

    # Summary
    summary_df = make_summary(norm_df, traj, args.lambdas)
    summ_path  = util_dir / "utility_metric_summary.csv"
    if summ_path.exists() and not args.force:
        old_summ = pd.read_csv(summ_path)
        old_summ = old_summ[old_summ["trajectory"] != traj]
        summary_df = pd.concat([old_summ, summary_df], ignore_index=True)
    summary_df.to_csv(summ_path, index=False)

    # Config
    existing_cfg = {}
    if cfg_path.exists():
        with open(cfg_path) as f:
            existing_cfg = json.load(f)
    existing_cfg[traj] = config
    with open(cfg_path, "w") as f:
        json.dump(existing_cfg, f, indent=2, default=str)

    # ── Replaceability degeneracy check ──────────────────────────────────────
    repl_col  = "replaceability_ratio"
    repl_vals = norm_df[repl_col].dropna()
    n_repl    = len(repl_vals)
    frac_near_1 = float((repl_vals.between(0.99, 1.01)).mean()) if n_repl > 0 else float("nan")
    repl_degenerate = n_repl > 0 and frac_near_1 > 0.95

    if repl_degenerate:
        print(f"\n  WARNING: replaceability_ratio is DEGENERATE "
              f"({frac_near_1:.1%} of values in [0.99, 1.01]). "
              f"Metric will be NaN'd — do not interpret replaceability conclusions.")
        norm_df[repl_col] = float("nan")
        norm_df["frac_queries_good_substitute"] = float("nan")

    # Save replaceability analysis summary (for validate_audit_v2.py check_9)
    repl_analysis = pd.DataFrame([{
        "trajectory":         traj,
        "n_objects":          n_repl,
        "frac_near_1":        frac_near_1,
        "degenerate":         repl_degenerate,
        "verdict":            "DEGENERATE_FLAGGED" if repl_degenerate else "OK",
        "replaceability_ratio": float(repl_vals.mean()) if n_repl > 0 else float("nan"),
    }])
    repl_path = util_dir / "replaceability_analysis.csv"
    if repl_path.exists() and not args.force:
        old_repl = pd.read_csv(repl_path)
        old_repl = old_repl[old_repl["trajectory"] != traj]
        repl_analysis = pd.concat([old_repl, repl_analysis], ignore_index=True)
    repl_analysis.to_csv(repl_path, index=False)
    print(f"  Replaceability: frac_near_1={frac_near_1:.3f}  "
          f"degenerate={repl_degenerate}  n={n_repl}")

    n = len(norm_df)
    sel = norm_df[norm_df["retrieval_count"] > 0]
    lam_primary = args.lambdas[1] if len(args.lambdas) > 1 else args.lambdas[0]
    print(f"\n[{traj}] Normalized utility summary:")
    print(f"  Total objects:       {n:,}")
    print(f"  Selected by READ:    {len(sel):,}  ({len(sel)/max(n,1)*100:.1f}%)")
    if len(sel) > 0:
        col = f"shrunk_mean_lambda{lam_primary}"
        if col in sel.columns:
            print(f"  Shrunk mean (λ={lam_primary}):")
            print(f"    p25={sel[col].quantile(0.25):.4f}  "
                  f"p50={sel[col].quantile(0.50):.4f}  "
                  f"p75={sel[col].quantile(0.75):.4f}")
        print(f"  Positive use frac:   {sel['positive_use_fraction'].mean():.3f}")
        if not repl_degenerate and "replaceability_ratio" in sel.columns:
            print(f"  Replaceability ratio: {sel['replaceability_ratio'].mean():.3f}")
        else:
            print(f"  Replaceability ratio: DEGENERATE (NaN'd)")
    print(f"\nSaved: {norm_path}")


if __name__ == "__main__":
    main()

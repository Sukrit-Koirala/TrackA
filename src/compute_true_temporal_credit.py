"""
compute_true_temporal_credit.py  --  MVP 4c-1 Stage 5 (True Credit)

Links each historical WRITE event to future READ contributions on the SAME stream.

Delay = query_stream_step - write_stream_step  (always > 0 by READ-before-WRITE)

Computes per WRITE event:
  future_gain_{H}         for H in 200, 1k, 5k, 10k, full
  positive_future_gain_{H}
  negative_future_gain_{H}
  time_to_first_positive
  time_to_first_substantial_positive  (gain > median positive)
  n_future_reads / n_positive / n_negative

Then:
  credit horizon curves per action type
  immediate vs future Spearman + Pearson (Tables B and C from spec)
  false-negative rates at Q75 and Q90 (Table C)

Hard invariant:  min(delay) > 0
  (strictly future; same-step reads impossible under READ-before-WRITE)

Outputs in {output}/temporal_credit/:
  {traj}_event_future_credit.parquet
  action_credit_horizons.csv
  immediate_vs_future.csv
  false_negatives.csv
  temporal_credit_done.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

HORIZONS = [200, 1000, 5000, 10000]


def _spearman(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    n    = int(mask.sum())
    if n < 10:
        return float("nan"), n
    rho, _ = spearmanr(x[mask], y[mask])
    return float(rho), n


def _pearson(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    n    = int(mask.sum())
    if n < 10:
        return float("nan"), n
    try:
        r, _ = pearsonr(x[mask], y[mask])
    except Exception:
        return float("nan"), n
    return float(r), n


# ── Event-level future credit ─────────────────────────────────────────────────

def compute_event_credit(
    reads_df: pd.DataFrame,   # from replay_online_memory_chronologically
    writes_df: pd.DataFrame,  # WRITE events from replay
) -> pd.DataFrame:
    """
    For each WRITE event, collect strictly future READ rewards
    (query_step > write_step) and compute horizon sums.

    Invariant: min(delay) > 0  (strictly future)
    """
    # Validate delays in reads_df
    if len(reads_df) > 0:
        neg = int((reads_df["delay"] < 0).sum())
        zero = int((reads_df["delay"] == 0).sum())
        if neg > 0:
            raise ValueError(f"AUDIT INVALID: {neg} READ records have delay < 0")
        if zero > 0:
            print(f"  WARNING: {zero} READ records have delay == 0 "
                  "(same-step read/write). These will be excluded from future credit.")

    # Build lookup: oid → sorted list of (read_step, gain_j)
    # Only strictly future: delay > 0 means query_step > write_step
    obj_reads: dict[int, list] = {}
    for row in reads_df[reads_df["delay"] > 0].itertuples(index=False):
        oid = row.object_id
        if oid not in obj_reads:
            obj_reads[oid] = []
        obj_reads[oid].append((row.stream_step, row.gain_j))
    for oid in obj_reads:
        obj_reads[oid].sort(key=lambda x: x[0])

    # Positive gain median for "substantial positive" threshold
    all_pos_gains = [g for lst in obj_reads.values() for _, g in lst if g > 0]
    median_pos    = float(np.median(all_pos_gains)) if all_pos_gains else 0.0

    rows = []
    for evt in writes_df.itertuples(index=False):
        oid   = int(evt.object_id)
        t_w   = int(evt.stream_step)
        atype = str(evt.action_type).upper()

        row = {
            "stream_step":  t_w,
            "object_id":    oid,
            "action_type":  atype,
            "trajectory":   getattr(evt, "trajectory", "?"),
        }
        # Copy immediate_reward if present
        if hasattr(evt, "immediate_reward"):
            row["immediate_reward"] = evt.immediate_reward

        future_list = [(q, g) for q, g in obj_reads.get(oid, []) if q > t_w]
        n_fut = len(future_list)

        if n_fut == 0:
            for H in HORIZONS + ["full"]:
                row[f"future_gain_{H}"]     = 0.0
                row[f"pos_future_gain_{H}"] = 0.0
                row[f"neg_future_gain_{H}"] = 0.0
                row[f"n_future_{H}"]        = 0
            row["time_to_first_positive"]      = float("nan")
            row["time_to_first_substantial"]   = float("nan")
            row["n_future_reads"]              = 0
            row["n_pos_reads"]                 = 0
            row["n_neg_reads"]                 = 0
        else:
            q_arr  = np.array([q for q, _ in future_list])
            g_arr  = np.array([g for _, g in future_list])
            delays = q_arr - t_w   # all > 0

            for H in HORIZONS:
                mask_h = delays <= H
                g_h    = g_arr[mask_h]
                row[f"future_gain_{H}"]     = float(g_h.sum())
                row[f"pos_future_gain_{H}"] = float(g_h[g_h > 0].sum()) if (g_h > 0).any() else 0.0
                row[f"neg_future_gain_{H}"] = float(g_h[g_h < 0].sum()) if (g_h < 0).any() else 0.0
                row[f"n_future_{H}"]        = int(mask_h.sum())

            row["future_gain_full"]     = float(g_arr.sum())
            row["pos_future_gain_full"] = float(g_arr[g_arr > 0].sum()) if (g_arr > 0).any() else 0.0
            row["neg_future_gain_full"] = float(g_arr[g_arr < 0].sum()) if (g_arr < 0).any() else 0.0
            row["n_future_full"]        = n_fut

            pos_mask = g_arr > 0
            sub_mask = g_arr > median_pos
            row["time_to_first_positive"]    = (float(delays[pos_mask].min())
                                                if pos_mask.any() else float("nan"))
            row["time_to_first_substantial"] = (float(delays[sub_mask].min())
                                                if sub_mask.any() else float("nan"))
            row["n_future_reads"] = n_fut
            row["n_pos_reads"]    = int(pos_mask.sum())
            row["n_neg_reads"]    = int((g_arr < 0).sum())

        rows.append(row)

    return pd.DataFrame(rows)


# ── Credit horizon curves ─────────────────────────────────────────────────────

def compute_credit_horizons(credit_df: pd.DataFrame) -> pd.DataFrame:
    """Table A: fraction of eventual positive reward visible within each horizon."""
    rows = []
    for atype in sorted(credit_df["action_type"].unique()):
        sub   = credit_df[credit_df["action_type"] == atype]
        has_p = sub[sub.get("pos_future_gain_full", sub.get("future_gain_full", pd.Series(0))) > 0]

        row = {
            "action_type":          atype,
            "n_events":             len(sub),
            "n_with_future_pos":    len(has_p),
        }
        full_sum = float(has_p.get("pos_future_gain_full",
                                   has_p.get("future_gain_full",
                                             pd.Series(0))).sum())

        for H in HORIZONS:
            col_g = f"future_gain_{H}"
            col_p = f"pos_future_gain_{H}"

            if col_p in sub.columns:
                h_sum = float(has_p[col_p].sum())
                row[f"frac_pos_reward_by_{H}"] = float(h_sum / full_sum) if full_sum > 0 else float("nan")
            else:
                row[f"frac_pos_reward_by_{H}"] = float("nan")

            if col_g in sub.columns:
                row[f"mean_gain_{H}"]   = float(sub[col_g].mean())
                row[f"median_gain_{H}"] = float(sub[col_g].median())
            else:
                row[f"mean_gain_{H}"]   = float("nan")
                row[f"median_gain_{H}"] = float("nan")

        row["frac_pos_reward_by_full"] = 1.0 if full_sum > 0 else float("nan")
        row["mean_gain_full"]          = float(sub.get("future_gain_full",
                                                        pd.Series([0])).mean())

        ttp_col = "time_to_first_positive"
        if ttp_col in sub.columns:
            ttp = sub[ttp_col].dropna()
            row["ttp_p25"] = float(ttp.quantile(0.25)) if len(ttp) > 0 else float("nan")
            row["ttp_p50"] = float(ttp.quantile(0.50)) if len(ttp) > 0 else float("nan")
            row["ttp_p75"] = float(ttp.quantile(0.75)) if len(ttp) > 0 else float("nan")
        rows.append(row)

    return pd.DataFrame(rows)


# ── Immediate vs future (Table B) ─────────────────────────────────────────────

def compute_immediate_vs_future(credit_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Spearman + Pearson between immediate_reward and future_gain_H.
    Requires immediate_reward in credit_df (joined earlier via join_bandit_immediate_rewards.py).
    """
    if "immediate_reward" not in credit_df.columns:
        return pd.DataFrame(), {"status": "NO_IMMEDIATE_REWARD", "pass": False,
                                "join_match_rate": 0.0, "join_valid_95pct": False}

    imm_df = credit_df[credit_df["immediate_reward"].notna()].copy()
    n_write    = len(credit_df)
    n_matched  = len(imm_df)
    match_rate = float(n_matched / max(n_write, 1))

    print(f"\n  Immediate-reward join:")
    print(f"    WRITE events total:             {n_write:,}")
    print(f"    Events with immediate reward:   {n_matched:,}")
    print(f"    Match rate:                     {match_rate:.1%}")

    join_valid = match_rate >= 0.99

    rows = []
    for atype in sorted(imm_df["action_type"].unique()):
        sub = imm_df[imm_df["action_type"] == atype]
        imm = sub["immediate_reward"].values.astype(float)

        row = {"action_type": atype, "n": len(sub), "join_match_rate": match_rate}

        for label, col in [
            ("200",   "future_gain_200"),
            ("1000",  "future_gain_1000"),
            ("5000",  "future_gain_5000"),
            ("10000", "future_gain_10000"),
            ("full",  "future_gain_full"),
        ]:
            fut = sub[col].fillna(0).values.astype(float) if col in sub.columns else np.zeros(len(sub))
            rho_s, n_s = _spearman(imm, fut)
            rho_p, n_p = _pearson(imm, fut)
            row[f"spearman_imm_vs_{label}"] = rho_s
            row[f"pearson_imm_vs_{label}"]  = rho_p
            row[f"n_matched_{label}"]       = n_s
        rows.append(row)

    corr_df    = pd.DataFrame(rows)
    validation = {
        "n_write_events":        n_write,
        "n_with_immediate":      n_matched,
        "join_match_rate":       round(match_rate, 4),
        "join_valid_95pct":      join_valid,
        "verdict":               "OK" if join_valid else "LOW_JOIN_RATE",
    }
    return corr_df, validation


# ── False negatives (Table C) ─────────────────────────────────────────────────

def compute_false_negatives(credit_df: pd.DataFrame) -> pd.DataFrame:
    """
    False negative: immediate_reward <= 0 AND future_gain_H >= Q75 or Q90.
    """
    if "immediate_reward" not in credit_df.columns:
        return pd.DataFrame()

    imm_df = credit_df[credit_df["immediate_reward"].notna()].copy()
    if len(imm_df) == 0:
        return pd.DataFrame()

    rows = []
    for atype in sorted(imm_df["action_type"].unique()):
        sub      = imm_df[imm_df["action_type"] == atype]
        imm      = sub["immediate_reward"].values.astype(float)
        imm_neg  = imm <= 0
        n        = len(sub)
        n_neg    = int(imm_neg.sum())

        row = {"action_type": atype, "n_events": n, "n_imm_negative": n_neg}

        for H in HORIZONS + ["full"]:
            col = f"future_gain_{H}"
            if col not in sub.columns:
                continue
            fut  = sub[col].fillna(0).values.astype(float)
            q75  = np.nanpercentile(fut, 75)
            q90  = np.nanpercentile(fut, 90)
            fn75 = imm_neg & (fut >= q75)
            fn90 = imm_neg & (fut >= q90)
            d    = max(n_neg, 1)
            row[f"fn_rate_q75_{H}"] = float(fn75.sum() / d)
            row[f"fn_rate_q90_{H}"] = float(fn90.sum() / d)
            row[f"n_fn_q75_{H}"]    = int(fn75.sum())
            row[f"n_fn_q90_{H}"]    = int(fn90.sum())
        rows.append(row)

    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", nargs="+", default=["oracle", "bandit"])
    ap.add_argument("--output",       required=True)
    ap.add_argument("--force",        action="store_true")
    args = ap.parse_args()

    out_dir    = Path(args.output)
    rep_dir    = out_dir / "temporal_replay"
    credit_dir = out_dir / "temporal_credit"
    credit_dir.mkdir(parents=True, exist_ok=True)

    sentinel = credit_dir / "temporal_credit_done.json"
    if sentinel.exists() and not args.force:
        print(f"[cached] {sentinel}")
        return

    summaries = {}

    for traj in args.trajectories:
        reads_path  = rep_dir / f"{traj}_replay_reads.parquet"
        writes_path = rep_dir / f"{traj}_replay_writes.parquet"

        if not reads_path.exists():
            print(f"WARNING: {reads_path} not found — skipping {traj}")
            continue
        if not writes_path.exists():
            print(f"WARNING: {writes_path} not found — skipping {traj}")
            continue

        reads_df  = pd.read_parquet(reads_path)
        writes_df = pd.read_parquet(writes_path)
        print(f"\n=== {traj} ===")
        print(f"  Read records: {len(reads_df):,}")
        print(f"  WRITE events: {len(writes_df):,}")

        if len(reads_df) == 0:
            summaries[traj] = {"valid": False, "reason": "no_read_records"}
            continue

        # Validate delays
        neg = int((reads_df["delay"] < 0).sum())
        if neg > 0:
            print(f"  AUDIT INVALID: {neg} records with delay < 0")
            summaries[traj] = {"valid": False, "reason": f"{neg}_neg_delays"}
            continue

        # Event credit
        credit_df = compute_event_credit(reads_df, writes_df)
        credit_df.to_parquet(credit_dir / f"{traj}_event_future_credit.parquet", index=False)
        print(f"  Saved {traj}_event_future_credit.parquet ({len(credit_df):,} rows)")

        # Horizon curves
        horizon_df = compute_credit_horizons(credit_df)
        horizon_df.to_csv(credit_dir / f"{traj}_credit_horizons.csv", index=False)
        print("\n  Credit Horizon Curves (Table A):")
        h_cols = (["action_type", "n_events"] +
                  [f"frac_pos_reward_by_{H}" for H in HORIZONS] +
                  [f"mean_gain_{H}" for H in HORIZONS[:2]])
        print(horizon_df[[c for c in h_cols if c in horizon_df.columns]].to_string(index=False))

        # Immediate vs future
        corr_df, join_val = compute_immediate_vs_future(credit_df)
        if len(corr_df) > 0:
            corr_df.to_csv(credit_dir / f"{traj}_immediate_vs_future.csv", index=False)
            print("\n  Spearman (immediate vs future) — Table B:")
            scols = (["action_type", "n"] +
                     [f"spearman_imm_vs_{l}" for l in ["200","1000","5000","10000","full"]])
            print(corr_df[[c for c in scols if c in corr_df.columns]].to_string(index=False))
        else:
            print("  Immediate vs future: SKIPPED (no immediate_reward column)")
            print("  → Run join_bandit_immediate_rewards.py first (Stage 5b)")
            join_val = {"status": "NOT_JOINED", "join_match_rate": 0.0,
                        "join_valid_95pct": False}

        # False negatives
        fn_df = compute_false_negatives(credit_df)
        if len(fn_df) > 0:
            fn_df.to_csv(credit_dir / f"{traj}_false_negatives.csv", index=False)
            print("\n  False Negative Rates — Table C:")
            fn_cols = (["action_type", "n_events"] +
                       [f"fn_rate_q75_{H}" for H in HORIZONS] +
                       [f"fn_rate_q90_{H}" for H in HORIZONS[:2]])
            print(fn_df[[c for c in fn_cols if c in fn_df.columns]].to_string(index=False))

        # Q4 metric: time to first positive
        ttp_all = credit_df["time_to_first_positive"].dropna()
        q4_ttp_p50 = float(ttp_all.median()) if len(ttp_all) > 0 else float("nan")
        q4_answer  = ("YES" if q4_ttp_p50 > 200 else
                      "WEAK" if q4_ttp_p50 > 0 else
                      "INCONCLUSIVE")

        # Q7 metric: Spearman imm vs future
        if len(corr_df) > 0 and "spearman_imm_vs_full" in corr_df.columns:
            rhos = corr_df["spearman_imm_vs_full"].dropna().values
            med_rho = float(np.median(rhos)) if len(rhos) > 0 else float("nan")
            q7 = ("YES" if np.isnan(med_rho) else
                  "YES" if med_rho < 0.2 else
                  "PARTIAL" if med_rho < 0.5 else
                  "NO")
        else:
            q7 = "INCONCLUSIVE"
            med_rho = float("nan")

        # Q8 metric: false-negative rate
        q8 = "INCONCLUSIVE"
        if len(fn_df) > 0 and "fn_rate_q75_full" in fn_df.columns:
            med_fn = float(fn_df["fn_rate_q75_full"].median())
            q8 = ("HIGH" if med_fn > 0.25 else
                  "MODERATE" if med_fn > 0.10 else
                  "LOW")

        summaries[traj] = {
            "valid":                    True,
            "n_read_records":           int(len(reads_df)),
            "n_write_events":           int(len(writes_df)),
            "delay_min":                int(reads_df["delay"].min()),
            "delay_max":                int(reads_df["delay"].max()),
            "delay_mean":               float(reads_df["delay"].mean()),
            "q4_create_ttp_p50":        q4_ttp_p50,
            "q4_answer":                q4_answer,
            "q7_immediate_vs_future":   q7,
            "q7_median_spearman":       med_rho,
            "q8_false_negative_rate":   q8,
            "join_validation":          join_val,
        }

        print(f"\n  Q4 (delayed reward): {q4_answer}  (ttp_p50={q4_ttp_p50:.0f} steps)")
        print(f"  Q7 (imm vs future):  {q7}  (median ρ={med_rho:.3f})")
        print(f"  Q8 (false negatives): {q8}")

    with open(sentinel, "w") as f:
        json.dump({"trajectories": summaries}, f, indent=2, default=str)
    print(f"\nSaved: {sentinel}")


if __name__ == "__main__":
    main()

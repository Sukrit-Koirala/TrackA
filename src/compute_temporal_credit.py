"""
compute_temporal_credit.py  --  MVP 4c-1 Stage: True Temporal Credit

Links WRITE events (at val_step V_e) to READ rewards (at val_step V > V_e).
Delay = V - V_e is guaranteed >= 0 by build_chronological_replay.py.

Computes:
  - Per-event cumulative future reward at horizons 200, 1k, 5k, 10k, full
  - Credit horizon curves per action type
  - Immediate vs future Spearman correlations (with join validation)
  - Short-term false negative rates
  - Validation table: any negative delay = INVALID AUDIT

Outputs (in {output}/temporal_credit/):
  {traj}_event_future_credit.parquet
  action_credit_horizons.csv
  immediate_vs_future.csv
  short_term_false_negatives.csv
  temporal_credit_summary.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HORIZONS = [200, 1000, 5000, 10000]   # in val steps


def _spearman_safe(x, y):
    """Returns (rho, n) or (nan, n) if too few matched pairs."""
    mask = np.isfinite(x) & np.isfinite(y)
    n = mask.sum()
    if n < 10:
        return float("nan"), int(n)
    rho, _ = spearmanr(x[mask], y[mask])
    return float(rho), int(n)


def compute_event_credit(
    rewards_df: pd.DataFrame,    # from build_chronological_replay
    chron_events: pd.DataFrame,  # WRITE events with val_step
) -> pd.DataFrame:
    """
    For each WRITE event, compute future rewards at each horizon.
    Hard assertion: all delays in rewards_df must be >= 0.
    """
    # ── Validate no negative delays ───────────────────────────────────────────
    neg = (rewards_df["delay"] < 0).sum()
    if neg > 0:
        raise ValueError(
            f"AUDIT INVALID: {neg} reward records have delay < 0. "
            "This means READ preceded WRITE — chronological invariant violated."
        )
    print(f"  Delay validation passed: min_delay={rewards_df['delay'].min():.0f} >= 0")

    # Build fast lookup: object_id → sorted list of (val_step, gain_j)
    obj_rewards: dict[int, list] = {}
    for row in rewards_df.itertuples(index=False):
        oid = row.object_id
        if oid not in obj_rewards:
            obj_rewards[oid] = []
        obj_rewards[oid].append((row.val_step, row.gain_j))
    # Sort by val_step
    for oid in obj_rewards:
        obj_rewards[oid].sort(key=lambda x: x[0])

    # ── Per-event future credit ───────────────────────────────────────────────
    write_events = chron_events[chron_events["action_type"] != "DEFER"].copy()

    rows = []
    for evt in write_events.itertuples(index=False):
        oid   = evt.object_id
        v_e   = evt.val_step
        atype = evt.action_type

        if oid == -1 or oid not in obj_rewards:
            # No future READ records for this object
            row = {
                "event_id":    getattr(evt, "event_id", -1),
                "object_id":   oid,
                "stream_step": evt.stream_step,
                "val_step":    v_e,
                "action_type": atype,
                "trajectory":  evt.trajectory,
                "n_future_reads": 0,
            }
            for H in HORIZONS + ["full"]:
                row[f"future_gain_{H}"] = 0.0
                row[f"future_pos_gain_{H}"] = 0.0
                row[f"n_future_reads_{H}"] = 0
            row["time_to_first_positive"] = float("nan")
            row["time_to_first_negative"] = float("nan")
            rows.append(row)
            continue

        # All READ records for this object with val_step > v_e (strictly future)
        future = [(v, g) for (v, g) in obj_rewards[oid] if v > v_e]
        n_future = len(future)

        row = {
            "event_id":       getattr(evt, "event_id", -1),
            "object_id":      oid,
            "stream_step":    evt.stream_step,
            "val_step":       v_e,
            "action_type":    atype,
            "trajectory":     evt.trajectory,
            "n_future_reads": n_future,
        }

        if n_future == 0:
            for H in HORIZONS + ["full"]:
                row[f"future_gain_{H}"] = 0.0
                row[f"future_pos_gain_{H}"] = 0.0
                row[f"n_future_reads_{H}"] = 0
            row["time_to_first_positive"] = float("nan")
            row["time_to_first_negative"] = float("nan")
            rows.append(row)
            continue

        vals   = np.array([v for v, _ in future])
        gains  = np.array([g for _, g in future])
        delays = vals - v_e  # all >= 0

        # Cumulative gain up to each horizon
        full_gain = float(gains.sum())
        full_pos  = float(gains[gains > 0].sum()) if (gains > 0).any() else 0.0

        for H in HORIZONS:
            mask = delays <= H
            row[f"future_gain_{H}"]     = float(gains[mask].sum())
            row[f"future_pos_gain_{H}"] = float(gains[mask & (gains > 0)].sum())
            row[f"n_future_reads_{H}"]  = int(mask.sum())

        row["future_gain_full"]     = full_gain
        row["future_pos_gain_full"] = full_pos
        row["n_future_reads_full"]  = n_future

        # Time to first positive / negative
        pos_mask = gains > 0
        neg_mask = gains < 0
        row["time_to_first_positive"] = (float(delays[pos_mask].min())
                                         if pos_mask.any() else float("nan"))
        row["time_to_first_negative"] = (float(delays[neg_mask].min())
                                         if neg_mask.any() else float("nan"))

        rows.append(row)

    credit_df = pd.DataFrame(rows)
    return credit_df


def compute_credit_horizons(credit_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each action type, compute what fraction of eventual positive reward
    is visible within each horizon window.
    """
    rows = []
    for atype in sorted(credit_df["action_type"].unique()):
        sub = credit_df[credit_df["action_type"] == atype]
        # Only events with some future positive reward
        has_pos = sub[sub["future_pos_gain_full"] > 0]
        n_total = len(sub)
        n_with_pos = len(has_pos)

        row = {
            "action_type": atype,
            "n_events": n_total,
            "n_with_future_pos_reward": n_with_pos,
        }

        if n_with_pos == 0:
            for H in HORIZONS + ["full"]:
                row[f"frac_pos_reward_by_{H}"] = float("nan")
                row[f"mean_gain_{H}"]           = 0.0
            rows.append(row)
            continue

        full_sum = has_pos["future_pos_gain_full"].sum()
        for H in HORIZONS:
            h_sum = has_pos[f"future_pos_gain_{H}"].sum()
            row[f"frac_pos_reward_by_{H}"] = float(h_sum / full_sum) if full_sum > 0 else float("nan")
            row[f"mean_gain_{H}"]           = float(sub[f"future_gain_{H}"].mean())

        row[f"frac_pos_reward_by_full"] = 1.0
        row[f"mean_gain_full"]           = float(sub["future_gain_full"].mean())

        # Delay stats (val steps)
        ttp = sub["time_to_first_positive"].dropna()
        row["ttp_p25"] = float(ttp.quantile(0.25)) if len(ttp) > 0 else float("nan")
        row["ttp_p50"] = float(ttp.quantile(0.50)) if len(ttp) > 0 else float("nan")
        row["ttp_p75"] = float(ttp.quantile(0.75)) if len(ttp) > 0 else float("nan")

        rows.append(row)

    return pd.DataFrame(rows)


def compute_immediate_vs_future(
    credit_df: pd.DataFrame,
    events_df: pd.DataFrame,   # original WRITE events WITH immediate_reward
) -> tuple[pd.DataFrame, dict]:
    """
    Join WRITE events (with immediate_reward) to future credit.
    Report join match rate and Spearman correlations.
    """
    # Only events where immediate_reward was actually recorded
    imm_df = events_df[events_df["immediate_reward"].notna()].copy()
    imm_df = imm_df[imm_df["action_type"] != "DEFER"]

    join_keys = ["stream_step", "action_type", "trajectory"]
    available_keys = [k for k in join_keys if k in credit_df.columns and k in imm_df.columns]

    merged = imm_df.merge(
        credit_df[available_keys + ["event_id",
                                    "future_gain_200",  "future_gain_1000",
                                    "future_gain_5000", "future_gain_10000",
                                    "future_gain_full"]],
        on=available_keys, how="left"
    )

    n_write      = len(imm_df)
    n_matched    = merged[f"future_gain_200"].notna().sum()
    match_rate   = float(n_matched / n_write) if n_write > 0 else 0.0

    print(f"\n  Immediate-reward join:")
    print(f"    WRITE events with immediate reward: {n_write:,}")
    print(f"    Matched to future credit:           {n_matched:,}")
    print(f"    Match rate:                         {match_rate:.1%}")

    join_valid = match_rate >= 0.95

    rows = []
    for atype in sorted(merged["action_type"].unique()):
        sub = merged[merged["action_type"] == atype]
        imm = sub["immediate_reward"].values.astype(float)

        row = {"action_type": atype, "n": len(sub), "join_match_rate": match_rate}
        for label, col in [("200",  "future_gain_200"),
                             ("1000", "future_gain_1000"),
                             ("5000", "future_gain_5000"),
                             ("10000","future_gain_10000"),
                             ("full", "future_gain_full")]:
            fut = sub[col].fillna(0).values.astype(float) if col in sub.columns else np.zeros(len(sub))
            rho, n_matched_pair = _spearman_safe(imm, fut)
            row[f"spearman_imm_vs_{label}"] = rho
            row[f"n_matched_{label}"]        = n_matched_pair
        rows.append(row)

    corr_df = pd.DataFrame(rows)

    validation = {
        "n_write_with_immediate_reward": int(n_write),
        "n_matched_to_future_credit":    int(n_matched),
        "join_match_rate":               round(match_rate, 4),
        "join_valid_95pct":              bool(join_valid),
        "q_verdict_if_invalid":          "INCONCLUSIVE" if not join_valid else "see_correlations",
    }
    return corr_df, validation


def compute_false_negatives(
    credit_df: pd.DataFrame,
    events_df: pd.DataFrame,
    horizons: list = None,
) -> pd.DataFrame:
    """
    False negative: immediate_reward <= 0 AND future_gain_H is in top quartile.
    """
    if horizons is None:
        horizons = [200, 1000, 5000, 10000]

    imm_df = events_df[events_df["immediate_reward"].notna()].copy()
    imm_df = imm_df[imm_df["action_type"] != "DEFER"]

    join_keys = ["stream_step", "action_type", "trajectory"]
    available_keys = [k for k in join_keys if k in credit_df.columns and k in imm_df.columns]
    merged = imm_df.merge(
        credit_df[available_keys + [f"future_gain_{H}" for H in horizons]],
        on=available_keys, how="inner"
    )

    rows = []
    for atype in sorted(merged["action_type"].unique()):
        sub = merged[merged["action_type"] == atype]
        imm = sub["immediate_reward"].values.astype(float)
        imm_neg = imm <= 0
        n = len(sub)

        row = {"action_type": atype, "n_events": n,
               "n_immediate_negative": int(imm_neg.sum())}
        for H in horizons:
            col = f"future_gain_{H}"
            if col not in sub.columns:
                continue
            fut = sub[col].fillna(0).values.astype(float)
            q75 = np.nanpercentile(fut, 75)
            q90 = np.nanpercentile(fut, 90)
            fn_q75 = imm_neg & (fut >= q75)
            fn_q90 = imm_neg & (fut >= q90)
            denom  = max(imm_neg.sum(), 1)
            row[f"fn_rate_q75_H{H}"]  = float(fn_q75.sum() / denom)
            row[f"fn_rate_q90_H{H}"]  = float(fn_q90.sum() / denom)
            row[f"n_fn_q75_H{H}"]     = int(fn_q75.sum())
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", nargs="+", default=["oracle", "bandit"])
    ap.add_argument("--output",       required=True)
    ap.add_argument("--force",        action="store_true")
    args = ap.parse_args()

    out_dir    = Path(args.output)
    rep_dir    = out_dir / "temporal_replay"
    prov_dir   = out_dir / "provenance"
    credit_dir = out_dir / "temporal_credit"
    credit_dir.mkdir(parents=True, exist_ok=True)

    sentinel = credit_dir / "temporal_credit_done.json"
    if sentinel.exists() and not args.force:
        print(f"[cached] temporal_credit_done.json")
        return

    all_credit_dfs = {}
    summaries      = {}

    for traj in args.trajectories:
        rw_path = rep_dir / f"{traj}_query_rewards.parquet"
        ce_path = rep_dir / f"{traj}_chronological_events.parquet"
        ev_path = prov_dir / f"{traj}_events.parquet"

        if not rw_path.exists():
            print(f"WARNING: {rw_path} not found, skipping {traj}")
            continue
        if not ce_path.exists():
            print(f"WARNING: {ce_path} not found, skipping {traj}")
            continue

        rewards_df    = pd.read_parquet(rw_path)
        chron_events  = pd.read_parquet(ce_path)
        events_df     = pd.read_parquet(ev_path) if ev_path.exists() else pd.DataFrame()

        print(f"\n=== {traj} ===")
        print(f"  Reward records: {len(rewards_df):,}")
        print(f"  WRITE events:   {len(chron_events):,}")

        # ── Validate delays ───────────────────────────────────────────────────
        if len(rewards_df) == 0:
            print(f"  WARNING: No reward records for {traj}")
            summaries[traj] = {"valid": False, "reason": "no_reward_records"}
            continue

        neg_delays = (rewards_df["delay"] < 0).sum()
        if neg_delays > 0:
            print(f"  AUDIT INVALID: {neg_delays} records have delay < 0!")
            summaries[traj] = {"valid": False, "reason": f"{neg_delays}_negative_delays"}
            continue

        # ── Event-level future credit ─────────────────────────────────────────
        credit_df = compute_event_credit(rewards_df, chron_events)
        credit_df.to_parquet(credit_dir / f"{traj}_event_future_credit.parquet", index=False)
        print(f"  Saved {traj}_event_future_credit.parquet  ({len(credit_df):,} events)")

        all_credit_dfs[traj] = credit_df

        # ── Credit horizon curves ─────────────────────────────────────────────
        horizon_df = compute_credit_horizons(credit_df)
        print("\n  Credit Horizon Curves:")
        print(horizon_df[["action_type", "n_events"] +
                          [f"frac_pos_reward_by_{H}" for H in HORIZONS]
                         ].to_string(index=False))

        # ── Immediate vs future ───────────────────────────────────────────────
        corr_df, join_validation = compute_immediate_vs_future(credit_df, events_df)
        print("\n  Immediate vs Future Spearman:")
        corr_cols = ["action_type", "n"] + [f"spearman_imm_vs_{H}"
                                             for H in ["200","1000","5000","10000","full"]]
        print(corr_df[[c for c in corr_cols if c in corr_df.columns]].to_string(index=False))

        # Q7 verdict: does immediate reward correlate poorly with long-horizon future?
        q7_rhos = corr_df.get("spearman_imm_vs_full", pd.Series([float("nan")])).values
        q7_rhos_valid = q7_rhos[np.isfinite(q7_rhos)]
        if len(q7_rhos_valid) == 0:
            q7 = "INCONCLUSIVE"
        elif np.median(q7_rhos_valid) < 0.2:
            q7 = "YES"   # immediate reward is a poor predictor of long-horizon value
        elif np.median(q7_rhos_valid) < 0.5:
            q7 = "PARTIAL"
        else:
            q7 = "NO"

        # ── False negatives ───────────────────────────────────────────────────
        fn_df = compute_false_negatives(credit_df, events_df)

        # Q8 verdict
        fn_rates = []
        for col in [f"fn_rate_q75_H1000", f"fn_rate_q75_H5000"]:
            if col in fn_df.columns:
                fn_rates.extend(fn_df[col].dropna().tolist())
        q8 = ("HIGH" if fn_rates and np.median(fn_rates) > 0.25
              else "MODERATE" if fn_rates and np.median(fn_rates) > 0.1
              else "LOW" if fn_rates else "INCONCLUSIVE")

        # ── Save tables ───────────────────────────────────────────────────────
        horizon_df.to_csv(credit_dir / "action_credit_horizons.csv", index=False)
        corr_df.to_csv(credit_dir / "immediate_vs_future.csv", index=False)
        fn_df.to_csv(credit_dir / "short_term_false_negatives.csv", index=False)

        # ── Summary ───────────────────────────────────────────────────────────
        has_future = (credit_df["n_future_reads"] > 0).mean()
        mean_delay = float(rewards_df["delay"].mean())
        median_delay = float(rewards_df["delay"].median())

        summaries[traj] = {
            "valid":               True,
            "n_write_events":      len(credit_df),
            "n_with_future_reads": int((credit_df["n_future_reads"] > 0).sum()),
            "frac_with_future":    float(has_future),
            "mean_delay_val_steps": mean_delay,
            "median_delay_val_steps": median_delay,
            "join_validation":     join_validation,
            "q7_immediate_vs_future": q7,
            "q8_false_negative_rate": q8,
        }

        # Q4: do future READ rewards appear many steps after CREATE?
        create_df = credit_df[credit_df["action_type"] == "CREATE_BUFFER"]
        if len(create_df) > 0:
            ttp_create = create_df["time_to_first_positive"].dropna()
            summaries[traj]["q4_create_ttp_p50"]  = float(ttp_create.quantile(0.50)) if len(ttp_create) > 0 else float("nan")
            summaries[traj]["q4_create_ttp_p75"]  = float(ttp_create.quantile(0.75)) if len(ttp_create) > 0 else float("nan")
            summaries[traj]["q4_answer"] = (
                "YES" if summaries[traj].get("q4_create_ttp_p50", 0) > 200 else
                "WEAK" if summaries[traj].get("q4_create_ttp_p50", 0) > 0 else
                "INCONCLUSIVE"
            )

    # Final summary
    with open(sentinel, "w") as f:
        json.dump({"trajectories": summaries}, f, indent=2, default=str)
    print(f"\nSaved temporal_credit_done.json")
    for traj, s in summaries.items():
        print(f"\n  [{traj}]")
        for k, v in s.items():
            if not isinstance(v, dict):
                print(f"    {k}: {v}")


if __name__ == "__main__":
    main()

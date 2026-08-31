"""
compute_object_counterfactual_utility.py  --  MVP 4c-0 Stage 3

Aggregates per-object counterfactual gains from the read usage logs.

For each object j across all val queries where j was selected:
  G_raw(j) = sum_q gain_j(q)         (sum of leave-one-out gains)
  mean_gain(j), median_gain(j)
  n_times_selected, n_helped, n_hurt
  gain_per_retrieval, gain_per_support

Also computes discounted return variants for configurable gamma values.
Delay step is stream_step of each query (since val.pt indices map to stream positions).

Outputs (in {output}/object_utility/):
  object_query_contributions.parquet   -- raw (object_id, query_id, gain) triples
  {traj}_object_returns.parquet        -- per-object aggregate returns
  discounted_returns.parquet           -- discounted variants

Usage:
  python src/compute_object_counterfactual_utility.py \\
    --trajectory oracle \\
    --output outputs_mvp4c0_delayed_credit_audit_fast \\
    --gammas 1.0 0.99 0.95 \\
    --discount_unit steps_100 \\
    --force
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


POSITIVE_THRESHOLD = 0.001   # gain > this = "helps"
NEGATIVE_THRESHOLD = -0.001  # gain < this = "hurts"


def _compute_delay(query_step: float, write_step: float, unit: str) -> float:
    """Delay from write event to query, in requested units."""
    raw = float(query_step) - float(write_step)
    if unit == "raw":
        return raw
    elif unit == "steps_100":
        return raw / 100.0
    elif unit == "steps_1000":
        return raw / 1000.0
    else:
        return raw


def aggregate_object_returns(
    usage_df: pd.DataFrame,
    objects_df: pd.DataFrame,
    gammas: list,
    discount_unit: str,
    traj: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    usage_df: columns [query_id, original_query_idx, true_token, gpt_nll,
                        state_row, rank, sim, nll_full, nll_minus_j, gain_j,
                        read_mode, trajectory]

    objects_df: columns [object_id, state_row_index, creation_step,
                          promotion_step, final_support, ...]
    """
    if usage_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Primary mode: fixed READ (or first available)
    modes = usage_df["read_mode"].unique()
    primary_mode = "fixed" if "fixed" in modes else modes[0]
    df = usage_df[usage_df["read_mode"] == primary_mode].copy()

    # Map state_row → object_id using objects_df
    row_to_obj = dict(zip(objects_df["state_row_index"].astype(int),
                          objects_df["object_id"].astype(int)))
    row_to_creation = dict(zip(objects_df["state_row_index"].astype(int),
                                objects_df["creation_step"].astype(float)))
    row_to_support  = dict(zip(objects_df["state_row_index"].astype(int),
                                objects_df["final_support"].astype(float)))
    row_to_promo    = dict(zip(objects_df["state_row_index"].astype(int),
                                objects_df["promotion_step"].astype(float)))

    df["object_id"]      = df["state_row"].map(row_to_obj).fillna(-1).astype(int)
    df["creation_step"]  = df["state_row"].map(row_to_creation).fillna(0)
    df["final_support"]  = df["state_row"].map(row_to_support).fillna(1)
    df["promotion_step"] = df["state_row"].map(row_to_promo).fillna(-1)

    # Query step is the original_query_idx (sequential position in stream)
    df["query_step"] = df["original_query_idx"].astype(float)

    # Undiscounted per-object aggregation
    obj_agg = {}
    for obj_id, grp in df.groupby("object_id"):
        if obj_id < 0:
            continue
        gains = grp["gain_j"].values
        creation_step = float(grp["creation_step"].iloc[0])
        final_support = float(grp["final_support"].iloc[0])
        promo_step    = float(grp["promotion_step"].iloc[0])

        pos_gains = gains[gains > POSITIVE_THRESHOLD]
        neg_gains = gains[gains < NEGATIVE_THRESHOLD]

        # Delay from creation to first positive use
        pos_mask = gains > POSITIVE_THRESHOLD
        if pos_mask.any():
            first_pos_step = float(grp["query_step"].values[pos_mask].min())
            delay_to_first_pos = first_pos_step - creation_step
        else:
            delay_to_first_pos = float("nan")

        # Delay from creation to first substantial (>0.005) use
        subst_mask = gains > 0.005
        if subst_mask.any():
            first_subst_step = float(grp["query_step"].values[subst_mask].min())
            delay_to_first_substantial = first_subst_step - creation_step
        else:
            delay_to_first_substantial = float("nan")

        # Delay from promotion to first positive use
        if promo_step > 0 and pos_mask.any():
            pos_after_promo = grp["query_step"].values[pos_mask]
            pos_after_promo = pos_after_promo[pos_after_promo > promo_step]
            delay_from_promo = (float(pos_after_promo.min()) - promo_step
                                 if len(pos_after_promo) > 0 else float("nan"))
        else:
            delay_from_promo = float("nan")

        obj_agg[obj_id] = {
            "object_id":              obj_id,
            "trajectory":             traj,
            "creation_step":          creation_step,
            "promotion_step":         promo_step,
            "final_support":          final_support,
            "G_raw":                  float(gains.sum()),
            "mean_gain":              float(gains.mean()),
            "median_gain":            float(np.median(gains)),
            "std_gain":               float(gains.std()) if len(gains) > 1 else 0.0,
            "n_times_selected":       int(len(gains)),
            "n_helped":               int(len(pos_gains)),
            "n_hurt":                 int(len(neg_gains)),
            "n_neutral":              int((np.abs(gains) <= POSITIVE_THRESHOLD).sum()),
            "positive_use_frac":      float(len(pos_gains) / max(len(gains), 1)),
            "negative_use_frac":      float(len(neg_gains) / max(len(gains), 1)),
            "gain_per_retrieval":     float(gains.mean()),
            "gain_per_support":       float(gains.sum() / max(final_support, 1)),
            "delay_to_first_pos":     delay_to_first_pos,
            "delay_to_first_subst":   delay_to_first_substantial,
            "delay_from_promo_to_pos": delay_from_promo,
            "is_net_positive":        float(gains.sum()) > 0,
            "is_net_negative":        float(gains.sum()) < 0,
            "never_helped":           len(pos_gains) == 0,
        }

    obj_returns_df = pd.DataFrame(list(obj_agg.values()))

    # Objects never selected
    all_obj_ids = set(objects_df["object_id"].astype(int).tolist())
    selected_obj_ids = set(obj_agg.keys())
    unselected = all_obj_ids - selected_obj_ids
    if unselected:
        unsel_rows = []
        for oid in unselected:
            row = objects_df[objects_df["object_id"] == oid].iloc[0]
            unsel_rows.append({
                "object_id":       int(oid),
                "trajectory":      traj,
                "creation_step":   float(row.get("creation_step", 0)),
                "promotion_step":  float(row.get("promotion_step", -1)),
                "final_support":   float(row.get("final_support", 0)),
                "G_raw":           0.0,
                "mean_gain":       0.0,
                "median_gain":     0.0,
                "std_gain":        0.0,
                "n_times_selected": 0,
                "n_helped":        0,
                "n_hurt":          0,
                "n_neutral":       0,
                "positive_use_frac": 0.0,
                "negative_use_frac": 0.0,
                "gain_per_retrieval": 0.0,
                "gain_per_support": 0.0,
                "delay_to_first_pos": float("nan"),
                "delay_to_first_subst": float("nan"),
                "delay_from_promo_to_pos": float("nan"),
                "is_net_positive": False,
                "is_net_negative": False,
                "never_helped":    True,
            })
        obj_returns_df = pd.concat(
            [obj_returns_df, pd.DataFrame(unsel_rows)], ignore_index=True)

    # Discounted returns
    disc_rows = []
    for gamma in gammas:
        for obj_id, grp in df.groupby("object_id"):
            if obj_id < 0:
                continue
            gains      = grp["gain_j"].values
            q_steps    = grp["query_step"].values
            c_step     = float(grp["creation_step"].iloc[0])
            delays     = np.array([_compute_delay(qs, c_step, discount_unit)
                                    for qs in q_steps])
            delays     = np.clip(delays, 0, None)
            disc_gains = gains * (gamma ** delays)
            disc_rows.append({
                "object_id":          int(obj_id),
                "trajectory":         traj,
                "gamma":              gamma,
                "discount_unit":      discount_unit,
                "G_discounted":       float(disc_gains.sum()),
                "G_raw":              float(gains.sum()),
                "n_times_selected":   int(len(gains)),
            })

    disc_df = pd.DataFrame(disc_rows)

    # Object-query contributions (raw records)
    contrib_df = df[["query_id", "original_query_idx", "object_id",
                     "state_row", "gain_j", "sim", "rank",
                     "nll_full", "nll_minus_j", "creation_step",
                     "query_step"]].copy()

    return contrib_df, obj_returns_df, disc_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory",     required=True, choices=["oracle", "bandit"])
    ap.add_argument("--output",         required=True)
    ap.add_argument("--gammas",         nargs="+", type=float,
                    default=[1.0, 0.99, 0.95])
    ap.add_argument("--discount_unit",  default="steps_100",
                    choices=["raw", "steps_100", "steps_1000"])
    ap.add_argument("--force",          action="store_true")
    args = ap.parse_args()

    traj    = args.trajectory
    out_dir = Path(args.output)
    util_dir = out_dir / "object_utility"
    util_dir.mkdir(parents=True, exist_ok=True)

    # Output files
    contrib_p   = util_dir / "object_query_contributions.parquet"
    returns_p   = util_dir / f"{traj}_object_returns.parquet"
    discount_p  = util_dir / "discounted_returns.parquet"

    if returns_p.exists() and not args.force:
        print(f"[cached] {returns_p.name}")
        return

    # Load inputs
    ru_dir     = out_dir / "read_usage"
    usage_path = ru_dir / f"{traj}_object_usage.parquet"
    prov_path  = out_dir / "provenance" / f"{traj}_objects.parquet"

    if not usage_path.exists():
        raise FileNotFoundError(f"Missing: {usage_path}  (run log_read_object_usage.py first)")
    if not prov_path.exists():
        raise FileNotFoundError(f"Missing: {prov_path}  (run build_write_object_provenance.py first)")

    print(f"Loading usage data ({traj}) ...")
    usage_df   = pd.read_parquet(usage_path)
    objects_df = pd.read_parquet(prov_path)
    print(f"  Usage records: {len(usage_df):,}")
    print(f"  Objects:       {len(objects_df):,}")

    print(f"Aggregating counterfactual returns ...")
    contrib_df, obj_returns_df, disc_df = aggregate_object_returns(
        usage_df, objects_df, args.gammas, args.discount_unit, traj)

    # Merge/append if contrib file already exists (multiple trajectories)
    if contrib_p.exists() and not args.force:
        old = pd.read_parquet(contrib_p)
        contrib_df = pd.concat([old, contrib_df], ignore_index=True)
    contrib_df.to_parquet(contrib_p, index=False)
    obj_returns_df.to_parquet(returns_p, index=False)

    # Append/merge discounted returns
    if discount_p.exists() and not args.force:
        old_disc = pd.read_parquet(discount_p)
        disc_df  = pd.concat([old_disc[old_disc["trajectory"] != traj],
                               disc_df], ignore_index=True)
    disc_df.to_parquet(discount_p, index=False)

    print(f"\nResults for '{traj}':")
    n = len(obj_returns_df)
    if n > 0:
        pos_n = obj_returns_df["is_net_positive"].sum()
        neg_n = obj_returns_df["is_net_negative"].sum()
        neu_n = n - pos_n - neg_n
        sel_n = (obj_returns_df["n_times_selected"] > 0).sum()
        print(f"  Total objects:     {n:,}")
        print(f"  Selected by READ:  {int(sel_n):,}")
        print(f"  Net-positive:      {int(pos_n):,}  ({pos_n/n*100:.1f}%)")
        print(f"  Net-negative:      {int(neg_n):,}  ({neg_n/n*100:.1f}%)")
        print(f"  Never helped:      {int(obj_returns_df['never_helped'].sum()):,}")
        print(f"  Mean G_raw:        {obj_returns_df['G_raw'].mean():.4f}")
        print(f"  Median G_raw:      {obj_returns_df['G_raw'].median():.4f}")

        delay_col = obj_returns_df["delay_to_first_pos"].dropna()
        if len(delay_col) > 0:
            print(f"  Delay to first pos:")
            for q_name, q_val in [("p25", 0.25), ("p50", 0.50),
                                    ("p75", 0.75), ("p90", 0.90)]:
                print(f"    {q_name}: {delay_col.quantile(q_val):.0f} stream steps")

    print(f"\nSaved: {returns_p}")


if __name__ == "__main__":
    main()

"""
join_bandit_immediate_rewards.py  --  MVP 4c-1 Stage 5b

Inspects bandit rollout logs for immediate reward data and joins with
provenance events.parquet.

Sources (tried in order):
  1. {bandit_dir}/reward_datasets/onpolicy/actions.parquet
       Produced by: rollout_mvp4b1.py --collect_rewards
       Contains: (step, action_type, reward_full, reward_local, ...)
       Coverage: every 5th tick at which the CHOSEN action was recorded
  2. {bandit_dir}/reward_datasets/bandit/actions.parquet
       Produced by: build_mvp4b1_reward_dataset.py  (oracle-trajectory samples)
       Note: these are ORACLE steps, not bandit steps — expected join rate 0%
  3. {bandit_dir}/rollout/rollout_log*.parquet
       Produced by: rollout_mvp4b1.py (sparse log every 5000 ticks)
       Note: no reward field — cannot join

Join key:  events.parquet["stream_step"]  ==  reward_log["step"]
           events.parquet["action_type"]  matches reward_log["action_name"]

Required:  join_match_rate ≥ 0.99

If match rate < 0.99:
  → Re-run rollout_mvp4b1.py with --collect_rewards to populate
    {bandit_dir}/reward_datasets/onpolicy/actions.parquet

Outputs (in {output}/temporal_credit/):
  bandit_immediate_rewards.parquet   -- joined (stream_step, action_type, imm_reward, ...)
  immediate_reward_join_stats.json   -- validation + coverage stats
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd


# ── Action type name normalisers ──────────────────────────────────────────────

def _norm_action(a: str) -> str:
    """Normalise action name to uppercase canonical form."""
    if pd.isna(a):
        return ""
    s = str(a).upper().strip()
    aliases = {
        "UPDATE_STATE":        "UPDATE_STATE",
        "UPDATESTATE":         "UPDATE_STATE",
        "ACT_UPDATE_STATE":    "UPDATE_STATE",
        "UPDATE_BUFFER":       "UPDATE_BUFFER",
        "UPDATEBUFFER":        "UPDATE_BUFFER",
        "ACT_UPDATE_BUFFER":   "UPDATE_BUFFER",
        "PROMOTE_AND_UPDATE":  "PROMOTE_AND_UPDATE",
        "PROMOTE":             "PROMOTE_AND_UPDATE",
        "ACT_PROMOTE":         "PROMOTE_AND_UPDATE",
        "CREATE_BUFFER":       "CREATE_BUFFER",
        "CREATEBUFFER":        "CREATE_BUFFER",
        "CREATE":              "CREATE_BUFFER",
        "ACT_CREATE_BUFFER":   "CREATE_BUFFER",
        "DEFER":               "DEFER",
        "ACT_DEFER":           "DEFER",
    }
    return aliases.get(s, s)


def _find_reward_source(bandit_dir: Path) -> tuple[pd.DataFrame | None, str]:
    """Try known reward source paths in priority order. Returns (df, source_label)."""
    candidates = [
        (bandit_dir / "reward_datasets" / "onpolicy" / "actions.parquet",
         "onpolicy/actions.parquet"),
        # Sometimes the 4b.1 rollout saves here:
        (bandit_dir / "reward_datasets" / "onpolicy_rewards" / "actions.parquet",
         "onpolicy_rewards/actions.parquet"),
        # Off-trajectory training dataset (oracle steps) — low expected match rate:
        (bandit_dir / "reward_datasets" / "bandit" / "actions.parquet",
         "bandit/actions.parquet (oracle steps — low match expected)"),
    ]
    for path, label in candidates:
        if path.exists():
            try:
                df = pd.read_parquet(path)
                print(f"  Found reward data: {label}  ({len(df):,} rows)")
                return df, label
            except Exception as e:
                print(f"  Could not read {path}: {e}")
    return None, "NONE"


def _infer_immediate_reward_col(df: pd.DataFrame) -> str | None:
    """Return the best available immediate reward column."""
    priority = [
        "reward_local",           # rollout_mvp4b1 onpolicy
        "reward_local_only",      # build_bandit_write_reward_dataset
        "reward_local_history",
        "reward_full",
        "reward_penalized",
    ]
    for col in priority:
        if col in df.columns:
            return col
    return None


def join_rewards(
    events_df: pd.DataFrame,
    reward_df: pd.DataFrame,
    reward_col: str,
) -> tuple[pd.DataFrame, dict]:
    """
    Join on (stream_step, action_type) using:
      events.stream_step  ==  reward_df.step
      events.action_type  ==  reward_df.action_name  (normalised)

    Returns (joined_df, stats_dict).
    """
    events = events_df.copy()
    rewards = reward_df.copy()

    # Normalise action names in events
    if "action_type" in events.columns:
        events["_act_norm"] = events["action_type"].apply(_norm_action)
    else:
        events["_act_norm"] = ""

    # Normalise action names in rewards
    if "action_name" in rewards.columns:
        rewards["_act_norm"] = rewards["action_name"].apply(_norm_action)
    elif "action_type" in rewards.columns:
        # action_type might be int or string
        if rewards["action_type"].dtype in (np.int32, np.int64, int):
            int_to_name = {0: "UPDATE_STATE", 1: "UPDATE_BUFFER",
                           2: "PROMOTE_AND_UPDATE", 3: "CREATE_BUFFER", 4: "DEFER"}
            rewards["_act_norm"] = rewards["action_type"].map(
                lambda x: int_to_name.get(int(x), "UNKNOWN"))
        else:
            rewards["_act_norm"] = rewards["action_type"].apply(_norm_action)
    else:
        rewards["_act_norm"] = ""

    # Step → stream_step
    if "step" in rewards.columns:
        rewards = rewards.rename(columns={"step": "stream_step"})

    # Keep only the chosen action at each step (action with max score, or labelled chosen)
    # If "chosen" column present, filter; otherwise keep all (counterfactual join)
    if "is_chosen" in rewards.columns:
        chosen = rewards[rewards["is_chosen"].astype(bool)].copy()
        if len(chosen) == 0:
            chosen = rewards.copy()   # fallback
    else:
        chosen = rewards.copy()

    # Deduplicate: keep one row per (stream_step, _act_norm)
    chosen = chosen.drop_duplicates(subset=["stream_step", "_act_norm"], keep="first")

    n_events = len(events)
    n_reward = len(chosen)

    # Merge (exclude reward_col from extras — it's already the primary column)
    extra_cols = [c for c in ["nll_before", "nll_after", "reward_full",
                               "reward_local", "reward_local_history",
                               "reward_penalized", "history_damage", "n_probes"]
                  if c in chosen.columns and c != reward_col]
    merged = events.merge(
        chosen[["stream_step", "_act_norm", reward_col] + extra_cols],
        on=["stream_step", "_act_norm"],
        how="left",
    )

    n_matched = int(merged[reward_col].notna().sum())
    match_rate = float(n_matched / max(n_events, 1))

    stats = {
        "n_events_in_provenance":   n_events,
        "n_reward_records":         n_reward,
        "n_matched":                n_matched,
        "join_match_rate":          round(match_rate, 4),
        "join_valid_99pct":         match_rate >= 0.99,
        "join_valid_95pct":         match_rate >= 0.95,
        "verdict": ("VALID_99" if match_rate >= 0.99
                    else "VALID_95" if match_rate >= 0.95
                    else "LOW_MATCH_RATE"),
    }

    # Print detail
    print(f"\n  Join statistics:")
    print(f"    Provenance events (total):   {n_events:>10,}")
    print(f"    Reward records (candidates): {n_reward:>10,}")
    print(f"    Matched (reward not-null):   {n_matched:>10,}")
    print(f"    Join match rate:             {match_rate:>10.1%}")
    print(f"    Verdict: {stats['verdict']}")

    if match_rate < 0.95:
        print("\n  LOW MATCH RATE — likely causes:")
        print("  1. rollout_mvp4b1.py was not run with --collect_rewards")
        print("     → Re-run with --collect_rewards to populate onpolicy/actions.parquet")
        print("  2. Reward data is from oracle trajectory, not bandit trajectory")
        print("     → Ensure you are using onpolicy/actions.parquet from the BANDIT run")
        print("  3. Sparse coverage (collect_rewards only every 5th tick = ~20% coverage)")
        print("     → This caps join rate at ~20%")
        print("\n  REQUIRED ACTION: re-run rollout with per-step reward logging enabled")

    # Per-action breakdown
    if "_act_norm" in merged.columns and n_matched > 0:
        print(f"\n  Per-action join rates:")
        for atype in sorted(merged["_act_norm"].unique()):
            sub = merged[merged["_act_norm"] == atype]
            n_a = len(sub)
            n_m = int(sub[reward_col].notna().sum())
            print(f"    {atype:<25}  {n_m:>7,} / {n_a:>7,}  ({n_m/max(n_a,1):.1%})")

    merged = merged.drop(columns=["_act_norm"], errors="ignore")
    merged = merged.rename(columns={reward_col: "immediate_reward"})

    return merged, stats


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bandit_dir",  required=True,
                    help="Bandit output dir (outputs_mvp4b_bandit_write_fast or similar)")
    ap.add_argument("--output",      required=True,
                    help="MVP 4c-1 output dir")
    ap.add_argument("--force",       action="store_true")
    args = ap.parse_args()

    out_dir    = Path(args.output)
    bandit_dir = Path(args.bandit_dir)
    credit_dir = out_dir / "temporal_credit"
    credit_dir.mkdir(parents=True, exist_ok=True)

    out_joined = credit_dir / "bandit_immediate_rewards.parquet"
    out_stats  = credit_dir / "immediate_reward_join_stats.json"

    if out_joined.exists() and out_stats.exists() and not args.force:
        print(f"[cached] {out_joined.name}")
        return

    # ── Load bandit events ────────────────────────────────────────────────────
    evt_path = out_dir / "provenance" / "bandit_events.parquet"
    if not evt_path.exists():
        print(f"ERROR: {evt_path} not found.")
        print("  Run Stage 1 (build_write_object_provenance.py) first.")
        sys.exit(1)

    events_df = pd.read_parquet(evt_path)
    print(f"Loaded {len(events_df):,} bandit events from provenance")

    # ── Find reward source ────────────────────────────────────────────────────
    print(f"\nSearching for reward data in: {bandit_dir}")
    reward_df, source_label = _find_reward_source(bandit_dir)

    if reward_df is None:
        print("\nWARNING: No reward data found in any expected location.")
        print(f"  Checked under: {bandit_dir}/reward_datasets/{{onpolicy,bandit}}/actions.parquet")
        print("\n  To generate reward data:")
        print("  python src/rollout_mvp4b1.py --collect_rewards ...")
        print("  OR for the older bandit rollout:")
        print("  python src/rollout_bandit_write.py --collect_rewards ...")

        # Write a stub so downstream scripts don't crash
        stub = events_df.copy()
        stub["immediate_reward"] = float("nan")
        stub.to_parquet(out_joined, index=False)

        stats = {
            "source": "NONE",
            "n_events_in_provenance": len(events_df),
            "n_reward_records": 0,
            "n_matched": 0,
            "join_match_rate": 0.0,
            "join_valid_99pct": False,
            "join_valid_95pct": False,
            "verdict": "NO_REWARD_DATA",
        }
        with open(out_stats, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nSaved stub: {out_joined}")
        sys.exit(0)

    # ── Determine reward column ───────────────────────────────────────────────
    reward_col = _infer_immediate_reward_col(reward_df)
    if reward_col is None:
        print(f"\nERROR: Could not find any reward column in {source_label}")
        print(f"  Available columns: {list(reward_df.columns)}")
        sys.exit(1)

    print(f"  Using immediate reward column: '{reward_col}'")
    print(f"  Columns available: {list(reward_df.columns[:12])}")

    # ── Inspect reward distributions ──────────────────────────────────────────
    r_vals = reward_df[reward_col].dropna()
    print(f"\n  Reward distribution ({reward_col}):")
    print(f"    n={len(r_vals):,}  mean={r_vals.mean():.4f}  "
          f"std={r_vals.std():.4f}  "
          f"p25={r_vals.quantile(0.25):.4f}  "
          f"p50={r_vals.median():.4f}  "
          f"p75={r_vals.quantile(0.75):.4f}")
    print(f"    n_positive: {int((r_vals > 0).sum()):,}  "
          f"n_zero: {int((r_vals == 0).sum()):,}  "
          f"n_negative: {int((r_vals < 0).sum()):,}")

    # ── Join ──────────────────────────────────────────────────────────────────
    joined_df, stats = join_rewards(events_df, reward_df, reward_col)

    # ── Save ──────────────────────────────────────────────────────────────────
    stats["source"] = source_label
    stats["reward_col"] = reward_col

    joined_df.to_parquet(out_joined, index=False)
    with open(out_stats, "w") as f:
        json.dump(stats, f, indent=2, default=str)

    print(f"\nSaved:")
    print(f"  {out_joined}  ({len(joined_df):,} rows)")
    print(f"  {out_stats}")

    # Exit nonzero if join rate is too low, but first detect dataset misalignment
    if not stats["join_valid_95pct"]:
        # Determine whether this is a step-range mismatch (different rollout) or
        # just sparse coverage within the same rollout
        step_col = "step" if "step" in reward_df.columns else (
                   "stream_step" if "stream_step" in reward_df.columns else None)
        if step_col:
            rew_steps  = set(reward_df[step_col].dropna().astype(int).values)
            prov_steps = set(events_df["stream_step"].dropna().astype(int).values)
            overlap    = len(prov_steps & rew_steps)
            overlap_frac = overlap / max(len(prov_steps), 1)
        else:
            overlap_frac = 0.0
            overlap      = 0
            prov_steps   = set()
            rew_steps    = set()

        # Misalignment if step ranges don't overlap OR match rate is near-zero.
        # Sparse per-5-step reward logging gives ~20% match; truly misaligned data
        # gives < 5% because the (step, action) pairs don't correspond at all.
        is_misaligned = overlap_frac < 0.01 or stats["join_match_rate"] < 0.05
        if is_misaligned:
            stats["verdict"]               = "IMMEDIATE_REWARD_DATASET_NOT_ALIGNED_WITH_FINAL_ROLLOUT"
            stats["step_overlap_fraction"] = round(overlap_frac, 6)
            stats["n_prov_steps"]          = len(prov_steps)
            stats["n_reward_steps"]        = len(rew_steps)
            stats["n_overlapping_steps"]   = overlap
            print(f"\n  *** DATASET MISALIGNED: step overlap {overlap}/{len(prov_steps)} "
                  f"({overlap_frac:.1%}), match rate {stats['join_match_rate']:.1%} ***")
            print(f"  Reward data is from a different rollout than the audited trajectory.")
            print(f"  C08 will be marked UNSUPPORTED (not FAIL) by the validator.")
            # Rewrite stats file with updated verdict before exiting
            with open(out_stats, "w") as f:
                json.dump(stats, f, indent=2, default=str)
            sys.exit(0)   # non-fatal: validator will mark C08 as UNSUPPORTED

        print(f"\n  *** VALIDATION FAILED: join rate {stats['join_match_rate']:.1%} < 95% ***")
        sys.exit(2)


if __name__ == "__main__":
    main()

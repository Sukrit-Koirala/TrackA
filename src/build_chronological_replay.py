"""
build_chronological_replay.py  --  MVP 4c-1 Stage: Chronological Temporal Replay

Fixes the fundamental timeline mismatch in 4c-0.

Design (avoids embedding-space mismatch):
  The raw training datastore (outputs_track_a_offline_paper_gpt2_medium) stores
  1024-dim GPT-2-medium hidden states.  The val.pt "h" field is 768-dim (base
  GPT-2 student model).  Attempting prototype @ h_q would crash with a size
  mismatch — and even if padded, the two embedding spaces are incompatible.

  Instead we re-use the already-correct retrieval from log_read_object_usage.py
  (Stage 2), which works in the right embedding space.  Stage 2 saves
  read_usage/{traj}_object_usage.parquet with the field original_query_idx —
  the actual val-set index for each query.  Sorting by that field gives
  sequential / chronological ordering.

Timeline fix:
  - WRITE events carry stream_step (index into N_train training steps).
  - We map each WRITE event to the equivalent val step:
        V_e = round(stream_step / N_train * N_val)
  - An object is "born" at birth_val_step = min(V_e) over all its WRITE events.
  - For every (original_query_idx=V, object_id, gain_j) tuple from Stage 2:
        delay = V - birth_val_step   >=  0  always  (pre-birth rows dropped)
  - Sorting by V gives true chronological order.

Outputs (in {output}/temporal_replay/):
  {traj}_query_rewards.parquet        -- (val_step, object_id, gain_j, delay, ...)
  {traj}_chronological_events.parquet -- WRITE events with mapped val_step
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch

from utils import set_seed


WRITE_TYPES = {"CREATE", "CREATE_BUFFER", "UPDATE_BUFFER", "UPDATE_STATE", "PROMOTE"}


def process_trajectory(
    traj: str,
    events_df: pd.DataFrame,
    objects_df: pd.DataFrame,
    usage_df: pd.DataFrame,
    N_train: int,
    N_val: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Maps pre-computed READ retrievals onto the chronological val-step timeline.

    Returns
    -------
    rewards_df   : (val_step, object_id, gain_j, delay, ...)  — sorted by val_step
    chron_events : WRITE events annotated with val_step
    """
    print(f"\n  === Chronological replay: {traj} ===")
    print(f"  N_train={N_train:,}  N_val={N_val:,}")

    # ── Step 1: Map WRITE events → val timeline ───────────────────────────────
    write_ev = events_df[
        events_df["action_type"].str.upper().isin(WRITE_TYPES)
    ].copy()
    write_ev["val_step"] = (
        write_ev["stream_step"] / N_train * N_val
    ).round().astype(int).clip(0, N_val - 1)

    # Object birth = earliest WRITE val_step for each object
    obj_born = (
        write_ev.groupby("object_id")["val_step"]
        .min()
        .rename("born_val_step")
        .reset_index()
    )

    print(f"  WRITE events: {len(write_ev):,}  |  objects born: {len(obj_born):,}")
    if len(write_ev) > 0:
        print(f"  val_step range for births: "
              f"[{int(write_ev['val_step'].min())}, {int(write_ev['val_step'].max())}]")

    # ── Step 2: Build state_row → object_id mapping ───────────────────────────
    row_to_obj: dict = {}
    if "state_row_index" in objects_df.columns and "object_id" in objects_df.columns:
        row_to_obj = dict(zip(
            objects_df["state_row_index"].astype(int),
            objects_df["object_id"].astype(int),
        ))
    else:
        print("  WARNING: objects.parquet missing 'state_row_index' column — "
              "will try using 'state_row' as object_id directly")

    # ── Step 3: Prepare usage records ─────────────────────────────────────────
    if "read_mode" in usage_df.columns:
        modes   = usage_df["read_mode"].unique()
        primary = "fixed" if "fixed" in modes else modes[0]
        usage   = usage_df[usage_df["read_mode"] == primary].copy()
        print(f"  Using read_mode='{primary}' ({len(usage):,} records)")
    else:
        usage = usage_df.copy()
        print(f"  {len(usage):,} usage records (no read_mode column)")

    # Rename original_query_idx → val_step
    if "original_query_idx" in usage.columns:
        usage = usage.rename(columns={"original_query_idx": "val_step"})
    elif "original_idx" in usage.columns:
        usage = usage.rename(columns={"original_idx": "val_step"})
    else:
        print("  ERROR: usage parquet has neither 'original_query_idx' "
              "nor 'original_idx'. Cannot determine val_step.")
        sys.exit(1)

    # Rename nll columns for consistency
    if "gpt_nll" in usage.columns and "nll_gpt" not in usage.columns:
        usage = usage.rename(columns={"gpt_nll": "nll_gpt"})
    if "true_token" not in usage.columns and "true_token" in events_df.columns:
        pass  # not needed here

    # Map state_row → object_id
    if "object_id" not in usage.columns:
        if row_to_obj:
            usage["object_id"] = usage["state_row"].map(row_to_obj).fillna(-1).astype(int)
        else:
            usage["object_id"] = usage["state_row"].astype(int)
    usage = usage[usage["object_id"] >= 0].copy()

    # ── Step 4: Join with birth step and compute delay ────────────────────────
    merged = usage.merge(obj_born, on="object_id", how="left")
    merged["born_val_step"] = merged["born_val_step"].fillna(0).astype(int)
    merged["delay"] = merged["val_step"].astype(int) - merged["born_val_step"]
    merged["obj_creation_val_step"] = merged["born_val_step"]

    # ── Step 5: Drop pre-birth queries (delay < 0) ────────────────────────────
    n_total   = len(merged)
    valid     = merged[merged["delay"] >= 0].copy()
    n_valid   = len(valid)
    n_dropped = n_total - n_valid
    print(f"  {n_valid:,} / {n_total:,} records causally valid (delay >= 0); "
          f"{n_dropped:,} dropped (query before object born)")

    if n_valid == 0:
        print("  WARNING: Zero valid records. "
              "Check that log_read_object_usage.py (Stage 2) ran successfully.")
        return pd.DataFrame(), write_ev

    # ── Step 6: Sort chronologically, select output columns ───────────────────
    valid = valid.sort_values(["val_step", "object_id"]).reset_index(drop=True)

    keep = [
        "val_step", "object_id", "gain_j", "delay",
        "born_val_step", "obj_creation_val_step",
        "sim", "rank", "nll_full", "nll_gpt", "true_token",
    ]
    out_cols   = [c for c in keep if c in valid.columns]
    rewards_df = valid[out_cols].copy()

    # Hard invariant
    neg = int((rewards_df["delay"] < 0).sum())
    assert neg == 0, f"INVARIANT VIOLATED: {neg} records have delay < 0"

    print(f"  Delay range: "
          f"[{int(rewards_df['delay'].min())}, {int(rewards_df['delay'].max())}] val steps")
    print(f"  Objects with credit: {rewards_df['object_id'].nunique():,}")
    print(f"  Val steps covered:   {rewards_df['val_step'].nunique():,}")

    return rewards_df, write_ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory",    required=True, choices=["oracle", "bandit"])
    ap.add_argument("--source",        required=True,
                    help="Raw data dir with states/val.pt (for N_val count only)")
    ap.add_argument("--datastore_dir", required=True,
                    help="Dir with states/datastore.pt (for N_train count only)")
    ap.add_argument("--oracle_dir",    required=True)
    ap.add_argument("--bandit_dir",    default=None)
    ap.add_argument("--output",        required=True)
    ap.add_argument("--max_temporal",  type=int, default=None,
                    help="Unused — kept for CLI compatibility with run_mvp4c1")
    ap.add_argument("--device",        default="cuda",
                    help="Unused — no GPU needed; kept for CLI compatibility")
    ap.add_argument("--seed",          type=int, default=42)
    ap.add_argument("--force",         action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    traj    = args.trajectory
    source  = Path(args.source)
    ds_dir  = Path(args.datastore_dir)
    out_dir = Path(args.output)
    rep_dir = out_dir / "temporal_replay"
    rep_dir.mkdir(parents=True, exist_ok=True)

    rw_path = rep_dir / f"{traj}_query_rewards.parquet"
    ce_path = rep_dir / f"{traj}_chronological_events.parquet"
    if rw_path.exists() and ce_path.exists() and not args.force:
        print(f"[cached] {rw_path.name}, {ce_path.name}")
        return

    # ── Load provenance ───────────────────────────────────────────────────────
    prov_dir = out_dir / "provenance"
    evt_path = prov_dir / f"{traj}_events.parquet"
    obj_path = prov_dir / f"{traj}_objects.parquet"
    for p in [evt_path, obj_path]:
        if not p.exists():
            print(f"ERROR: {p} not found. Run build_write_object_provenance.py first.")
            sys.exit(1)

    events_df  = pd.read_parquet(evt_path)
    objects_df = pd.read_parquet(obj_path)
    print(f"Loaded {len(events_df):,} events, {len(objects_df):,} objects for {traj}")
    print(f"  DEFER count: {(events_df['action_type'] == 'DEFER').sum():,}")

    # ── Load pre-computed read usage (Stage 2) ────────────────────────────────
    usage_path = out_dir / "read_usage" / f"{traj}_object_usage.parquet"
    if not usage_path.exists():
        print(f"ERROR: {usage_path} not found. "
              "Run log_read_object_usage.py (Stage 2) first.")
        sys.exit(1)
    usage_df = pd.read_parquet(usage_path)
    print(f"Loaded {len(usage_df):,} read usage records from Stage 2")

    # ── Get N_train and N_val (sizes only — h vectors NOT used) ──────────────
    ds_path = ds_dir / "states" / "datastore.pt"
    print(f"Loading datastore size from {ds_path} ...")
    ds_data = torch.load(ds_path, weights_only=False, map_location="cpu")
    N_train = int(len(ds_data["y"]))
    del ds_data

    val_path = source / "states" / "val.pt"
    print(f"Loading val size from {val_path} ...")
    val_data = torch.load(val_path, weights_only=False, map_location="cpu")
    N_val    = int(len(val_data["y"]))
    del val_data

    print(f"N_train={N_train:,}  N_val={N_val:,}")

    # ── Run ───────────────────────────────────────────────────────────────────
    rewards_df, chron_events = process_trajectory(
        traj, events_df, objects_df, usage_df, N_train, N_val
    )

    if len(rewards_df) == 0:
        print("ERROR: No valid chronological records produced.")
        sys.exit(1)

    rewards_df.to_parquet(rw_path, index=False)
    chron_events.to_parquet(ce_path, index=False)
    print(f"\nSaved:")
    print(f"  {rw_path}  ({len(rewards_df):,} rows)")
    print(f"  {ce_path}  ({len(chron_events):,} events)")


if __name__ == "__main__":
    main()

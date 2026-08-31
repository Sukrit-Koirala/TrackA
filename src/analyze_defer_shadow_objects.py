"""
analyze_defer_shadow_objects.py  --  MVP 4c-0 Stage 6

Shadow object analysis for sampled DEFER events.

A "shadow" object is a hypothetical object that would have been created/updated
at a DEFER event. To estimate its utility, we:
  1. Sample N_SAMPLE DEFER events from the bandit trajectory
  2. For each, find the closest oracle object (by prototype cosine similarity)
     that covers the same token region
  3. Use that oracle object's actual READ utility as a proxy for the shadow utility
  4. Compare: shadow utility vs. what the bandit chose to do instead

Answers:
  Q4: Are DEFER events missing high-utility write opportunities?

Outputs (in {output}/analysis/):
  shadow_objects.csv          -- per-sampled-DEFER: closest oracle proxy utility
  defer_shadow_summary.json

Usage:
  python src/analyze_defer_shadow_objects.py \\
    --output outputs_mvp4c0_delayed_credit_audit_fast \\
    --n_sample 500 \\
    --min_sim 0.70 \\
    --seed 42 \\
    --force
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _cosine_sim_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a: [N, D]  b: [M, D]
    Returns: [N, M] cosine similarities
    """
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return an @ bn.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output",   required=True)
    ap.add_argument("--n_sample", type=int,   default=500,
                    help="Number of DEFER events to sample for shadow analysis")
    ap.add_argument("--min_sim",  type=float, default=0.70,
                    help="Min cosine sim to oracle prototype to count as a valid proxy")
    ap.add_argument("--seed",     type=int,   default=42)
    ap.add_argument("--force",    action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output)
    ana_dir = out_dir / "analysis"
    ana_dir.mkdir(parents=True, exist_ok=True)

    sentinel = ana_dir / "defer_shadow_done.json"
    if sentinel.exists() and not args.force:
        print(f"[cached] {sentinel}")
        return

    rng = np.random.default_rng(args.seed)

    # ── Load bandit provenance ────────────────────────────────────────────────
    bandit_ev_path  = out_dir / "provenance" / "bandit_events.parquet"
    bandit_obj_path = out_dir / "provenance" / "bandit_objects.parquet"
    oracle_obj_path = out_dir / "provenance" / "oracle_objects.parquet"
    oracle_ret_path = out_dir / "object_utility" / "oracle_object_returns.parquet"

    missing = [p for p in [bandit_ev_path, bandit_obj_path, oracle_obj_path, oracle_ret_path]
               if not p.exists()]
    if missing:
        print(f"ERROR: missing files: {[str(m) for m in missing]}")
        print("  Run build_write_object_provenance.py and compute_object_counterfactual_utility.py first")
        return

    bandit_events  = pd.read_parquet(bandit_ev_path)
    bandit_objects = pd.read_parquet(bandit_obj_path)
    oracle_objects = pd.read_parquet(oracle_obj_path)
    oracle_returns = pd.read_parquet(oracle_ret_path)

    # Filter to DEFER events from bandit events log
    defer_events = bandit_events[bandit_events["action_type"].str.upper() == "DEFER"].copy()
    if len(defer_events) == 0:
        print("WARNING: no DEFER events found in bandit_events.parquet")
        summary = {
            "n_defer_total": 0, "n_sampled": 0,
            "msg": "No DEFER events found",
        }
        with open(sentinel, "w") as f:
            json.dump(summary, f, indent=2)
        return

    print(f"Total DEFER events: {len(defer_events):,}")

    # Sample
    n_sample = min(args.n_sample, len(defer_events))
    sample_idx = rng.choice(len(defer_events), size=n_sample, replace=False)
    defer_sample = defer_events.iloc[sample_idx].reset_index(drop=True)
    print(f"Sampled: {n_sample:,}")

    # ── Load oracle state prototypes ──────────────────────────────────────────
    # Oracle state file: reconstructions/sequential_running_mean/states/
    state_dirs = list((out_dir.parent).glob(
        "*oracle*reconstruction*/states/*.npz")) + \
        list(out_dir.glob("**/states/oracle_seq_*.npz"))

    if not state_dirs:
        # Try default path pattern
        oracle_state_candidates = list(
            out_dir.parent.glob(
                "outputs_mvp4a0_oracle_reconstruction/states/oracle_seq_*.npz"))
        if oracle_state_candidates:
            state_dirs = oracle_state_candidates

    oracle_proto_h = None
    if state_dirs:
        state_path = sorted(state_dirs)[-1]  # latest
        print(f"Loading oracle state from: {state_path.name}")
        try:
            data = np.load(state_path)
            oracle_proto_h = data["prototype_h"]  # [B, D]
            print(f"  Oracle prototypes shape: {oracle_proto_h.shape}")
        except Exception as e:
            print(f"  WARNING: could not load oracle state: {e}")

    # ── Merge oracle returns with oracle objects ──────────────────────────────
    oracle_merged = oracle_objects.merge(
        oracle_returns[["object_id", "G_raw", "n_times_selected",
                        "is_net_positive", "positive_use_frac"]],
        on="object_id", how="left")
    oracle_merged["G_raw"] = oracle_merged["G_raw"].fillna(0.0)

    # ── For each DEFER event: find the query embedding and nearest oracle obj ─
    shadow_rows = []

    # If we have prototype_h, do embedding-based matching; otherwise use stream_step
    has_proto = oracle_proto_h is not None and "query_h" in defer_events.columns

    for i, ev in defer_sample.iterrows():
        defer_step = float(ev.get("stream_step", -1))

        # Strategy 1: embedding-based matching (if available)
        if has_proto:
            q_h = np.array(ev["query_h"]).reshape(1, -1)
            sims_to_oracle = _cosine_sim_batch(q_h, oracle_proto_h)[0]
            best_idx = int(np.argmax(sims_to_oracle))
            best_sim = float(sims_to_oracle[best_idx])
            proxy_row = oracle_merged[oracle_merged["state_row_index"] == best_idx]
        else:
            # Strategy 2: temporal proximity — oracle object created closest to defer_step
            time_diffs = (oracle_merged["creation_step"] - defer_step).abs()
            best_idx_df = time_diffs.idxmin()
            proxy_row   = oracle_merged.loc[[best_idx_df]]
            sims_to_oracle = np.array([-1.0])  # unknown sim
            best_sim = float("nan")

        if proxy_row.empty:
            continue
        proxy = proxy_row.iloc[0]
        valid_match = np.isnan(best_sim) or best_sim >= args.min_sim

        shadow_rows.append({
            "defer_stream_step":   defer_step,
            "proxy_object_id":     int(proxy["object_id"]),
            "proxy_sim":           best_sim,
            "valid_match":         bool(valid_match),
            "proxy_G_raw":         float(proxy["G_raw"]),
            "proxy_net_positive":  bool(proxy.get("is_net_positive", False)),
            "proxy_pos_use_frac":  float(proxy.get("positive_use_frac", 0.0)),
            "proxy_n_selected":    int(proxy.get("n_times_selected", 0)),
            "proxy_creation_step": float(proxy.get("creation_step", -1)),
        })

    shadow_df = pd.DataFrame(shadow_rows)
    shadow_df.to_csv(ana_dir / "shadow_objects.csv", index=False)
    print(f"\nShadow objects computed: {len(shadow_df):,}")

    # ── Q4: Do DEFER events miss high-utility write opportunities? ───────────
    valid_shadow = shadow_df[shadow_df["valid_match"]] if len(shadow_df) > 0 else shadow_df
    if len(valid_shadow) > 0:
        mean_proxy_G   = float(valid_shadow["proxy_G_raw"].mean())
        pos_frac       = float(valid_shadow["proxy_net_positive"].mean())
        high_util_frac = float((valid_shadow["proxy_G_raw"] > 0.01).mean())
        missed_high    = int((valid_shadow["proxy_G_raw"] > 0.01).sum())
        q4_answer = "YES_SIGNIFICANT" if (pos_frac > 0.3 and mean_proxy_G > 0.005) else \
                    "YES_MODERATE"    if (pos_frac > 0.2 or mean_proxy_G > 0.002) else \
                    "NO"
    else:
        mean_proxy_G   = float("nan")
        pos_frac       = float("nan")
        high_util_frac = float("nan")
        missed_high    = 0
        q4_answer      = "INCONCLUSIVE"

    print(f"\nQ4 (DEFER missing high-utility opportunities):")
    print(f"  Valid matches:       {len(valid_shadow):,} / {len(shadow_df):,}")
    print(f"  Mean proxy G_raw:    {mean_proxy_G:.4f}")
    print(f"  Frac net positive:   {pos_frac:.3f}")
    print(f"  High-util frac:      {high_util_frac:.3f}")
    print(f"  Answer:              {q4_answer}")

    summary = {
        "n_defer_total":   int(len(defer_events)),
        "n_sampled":       int(n_sample),
        "n_valid_matches": int(len(valid_shadow)),
        "min_sim":         args.min_sim,
        "q4_defer_misses_utility": {
            "mean_proxy_G_raw":   mean_proxy_G,
            "frac_net_positive":  pos_frac,
            "high_util_frac":     high_util_frac,
            "n_missed_high_util": missed_high,
            "answer":             q4_answer,
        },
    }
    with open(sentinel, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {sentinel}")


if __name__ == "__main__":
    main()

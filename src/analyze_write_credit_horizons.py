"""
analyze_write_credit_horizons.py  --  MVP 4c-0 Stage 5

Per-action-type credit horizon analysis and immediate-vs-future reward correlations.

Answers:
  Q2: How long does credit take to arrive for each action type?
  Q3: Is there significant delayed credit (beyond immediate window)?
  Q5: Are false-positive/false-negative rates for 0-step credit meaningful?

Per-action-type metrics:
  - CREATE: delay from creation to first positive READ
  - UPDATE_BUFFER: delay from update event to next positive READ
  - UPDATE_STATE:  same
  - PROMOTE:       delay from promotion to first positive READ after promotion

Immediate vs future correlation:
  - "immediate": gain in first 200 steps after write event
  - "future":    gain in steps 200-5000 after write event
  - Pearson / Spearman between immediate and future gains per action type
  - False positive rate: immediate gain > 0 but future gain <= 0
  - False negative rate: immediate gain <= 0 but future gain > 0

Outputs (in {output}/analysis/):
  credit_horizons.csv        -- per-action-type delay quantiles
  immediate_vs_future.csv    -- per-action-type immediate/future correlation + FP/FN rates
  action_type_summary.json

Usage:
  python src/analyze_write_credit_horizons.py \\
    --output outputs_mvp4c0_delayed_credit_audit_fast \\
    --trajectories oracle bandit \\
    --immediate_window 200 \\
    --future_start 200 \\
    --force
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def _corr(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    n = mask.sum()
    if n < 5:
        return float("nan"), float("nan"), int(n)
    xm, ym = x[mask], y[mask]
    try:
        pr = pearsonr(xm, ym).statistic
    except Exception:
        pr = float("nan")
    try:
        sr = spearmanr(xm, ym).statistic
    except Exception:
        sr = float("nan")
    return float(pr), float(sr), int(n)


def _delay_quantiles(vals):
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {k: float("nan") for k in ["n", "mean", "std", "p25", "p50", "p75", "p90", "p95"]}
    return {
        "n":    int(len(vals)),
        "mean": float(vals.mean()),
        "std":  float(vals.std()),
        "p25":  float(np.percentile(vals, 25)),
        "p50":  float(np.percentile(vals, 50)),
        "p75":  float(np.percentile(vals, 75)),
        "p90":  float(np.percentile(vals, 90)),
        "p95":  float(np.percentile(vals, 95)),
    }


def load_all(traj: str, out_dir: Path):
    """
    Returns (events_df, usage_df, objects_df) or (None, None, None) if missing.
    events_df: from provenance/{traj}_events.parquet
    usage_df:  from read_usage/{traj}_object_usage.parquet
    objects_df: from provenance/{traj}_objects.parquet
    """
    ev_path  = out_dir / "provenance"  / f"{traj}_events.parquet"
    use_path = out_dir / "read_usage"  / f"{traj}_object_usage.parquet"
    obj_path = out_dir / "provenance"  / f"{traj}_objects.parquet"

    missing = [p for p in [ev_path, use_path, obj_path] if not p.exists()]
    if missing:
        print(f"  WARNING [{traj}]: missing {[str(m) for m in missing]}")
        return None, None, None

    return (pd.read_parquet(ev_path),
            pd.read_parquet(use_path),
            pd.read_parquet(obj_path))


def compute_credit_horizons(events_df, usage_df, objects_df, traj, imm_window, future_start):
    """
    For each write event, find all queries for that object that came AFTER the event.
    Split queries into immediate (< imm_window steps) and future (>= future_start steps).
    Returns:
      credit_rows: one row per (write_event, first_positive_query)
      iv_rows:     per-event immediate_gain / future_gain pair
    """
    # Build object_id → row mapping
    obj_id_to_row = dict(zip(objects_df["object_id"].astype(int),
                              objects_df["state_row_index"].astype(int)))

    # Filter to fixed-READ usage; add state_row
    modes = usage_df["read_mode"].unique() if "read_mode" in usage_df.columns else ["fixed"]
    primary = "fixed" if "fixed" in modes else modes[0]
    usage = usage_df[usage_df["read_mode"] == primary].copy() if "read_mode" in usage_df.columns else usage_df.copy()

    # Map object_id from state_row
    row_to_obj = {v: k for k, v in obj_id_to_row.items()}
    if "object_id" not in usage.columns:
        usage["object_id"] = usage["state_row"].map(row_to_obj).fillna(-1).astype(int)
    usage = usage[usage["object_id"] >= 0].copy()

    # query_step = original_query_idx (stream position)
    if "query_step" not in usage.columns:
        usage["query_step"] = usage["original_query_idx"].astype(float)

    # Build per-object gain lookup: {object_id → sorted (query_step, gain_j) array}
    obj_queries = {}
    for obj_id, grp in usage.groupby("object_id"):
        steps  = grp["query_step"].values
        gains  = grp["gain_j"].values
        order  = np.argsort(steps)
        obj_queries[int(obj_id)] = (steps[order], gains[order])

    credit_rows = []
    iv_rows     = []

    for _, ev in events_df.iterrows():
        obj_id    = int(ev["object_id"])
        ev_step   = float(ev["stream_step"])
        act_type  = str(ev.get("action_type", "UNKNOWN"))

        if obj_id not in obj_queries:
            continue

        steps, gains = obj_queries[obj_id]
        # Only queries after this event
        after_mask = steps > ev_step
        if not after_mask.any():
            continue

        aft_steps = steps[after_mask]
        aft_gains = gains[after_mask]
        delays    = aft_steps - ev_step

        # --- Credit horizon: delay to first positive query ---
        pos_mask = aft_gains > 0.001
        if pos_mask.any():
            delay_first_pos = float(delays[pos_mask].min())
        else:
            delay_first_pos = float("nan")

        credit_rows.append({
            "trajectory":      traj,
            "action_type":     act_type,
            "object_id":       obj_id,
            "event_id":        ev.get("event_id", -1),
            "stream_step":     ev_step,
            "delay_first_pos": delay_first_pos,
            "n_queries_after": int(after_mask.sum()),
            "sum_gain_after":  float(aft_gains.sum()),
        })

        # --- Immediate vs future gains ---
        imm_mask  = delays < imm_window
        fut_mask  = delays >= future_start

        imm_gain = float(aft_gains[imm_mask].sum()) if imm_mask.any() else 0.0
        fut_gain = float(aft_gains[fut_mask].sum()) if fut_mask.any() else 0.0

        iv_rows.append({
            "trajectory":    traj,
            "action_type":   act_type,
            "object_id":     obj_id,
            "event_id":      ev.get("event_id", -1),
            "immediate_gain": imm_gain,
            "future_gain":   fut_gain,
            "n_imm_queries": int(imm_mask.sum()),
            "n_fut_queries": int(fut_mask.sum()),
        })

    return pd.DataFrame(credit_rows), pd.DataFrame(iv_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output",           required=True)
    ap.add_argument("--trajectories",     nargs="+", default=["oracle", "bandit"])
    ap.add_argument("--immediate_window", type=float, default=200.0,
                    help="Steps after write considered 'immediate' credit")
    ap.add_argument("--future_start",     type=float, default=200.0,
                    help="Steps after write considered 'future' credit (may overlap immediate)")
    ap.add_argument("--force",            action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output)
    ana_dir = out_dir / "analysis"
    ana_dir.mkdir(parents=True, exist_ok=True)

    sentinel = ana_dir / "credit_horizons_done.json"
    if sentinel.exists() and not args.force:
        print(f"[cached] {sentinel}")
        return

    all_credit = []
    all_iv     = []

    for traj in args.trajectories:
        print(f"\n[{traj}] Loading ...")
        events_df, usage_df, objects_df = load_all(traj, out_dir)
        if events_df is None:
            continue
        print(f"  {len(events_df):,} write events, {len(usage_df):,} usage records")

        cred_df, iv_df = compute_credit_horizons(
            events_df, usage_df, objects_df, traj,
            args.immediate_window, args.future_start)

        print(f"  {len(cred_df):,} event-credit records, {len(iv_df):,} IV records")
        all_credit.append(cred_df)
        all_iv.append(iv_df)

    if not all_credit:
        print("ERROR: no data")
        return

    credit_df = pd.concat(all_credit, ignore_index=True)
    iv_df     = pd.concat(all_iv, ignore_index=True)

    # ── Per-action-type credit horizon quantiles ─────────────────────────────
    horizon_rows = []
    for (traj, act), grp in credit_df.groupby(["trajectory", "action_type"]):
        delays = grp["delay_first_pos"].values
        q = _delay_quantiles(delays)
        horizon_rows.append({
            "trajectory":  traj,
            "action_type": act,
            "n_events":    len(grp),
            **q,
        })
    horizon_df = pd.DataFrame(horizon_rows)
    horizon_df.to_csv(ana_dir / "credit_horizons.csv", index=False)
    print("\nCredit horizons (p50 delay to first positive READ):")
    print(horizon_df[["trajectory", "action_type", "n_events", "p50", "p90"]].to_string(index=False))

    # ── Immediate vs future correlation ──────────────────────────────────────
    ivf_rows = []
    for (traj, act), grp in iv_df.groupby(["trajectory", "action_type"]):
        x = grp["immediate_gain"].values
        y = grp["future_gain"].values
        pr, sr, n = _corr(x, y)

        # FP: immediate > 0 but future <= 0
        fp_mask = (x > 0.001) & (y <= 0.001)
        fn_mask = (x <= 0.001) & (y > 0.001)
        n_pos_imm = int((x > 0.001).sum())
        n_neg_imm = int((x <= 0.001).sum())
        fp_rate = float(fp_mask.sum() / max(n_pos_imm, 1))
        fn_rate = float(fn_mask.sum() / max(n_neg_imm, 1))

        ivf_rows.append({
            "trajectory":    traj,
            "action_type":   act,
            "n_events":      len(grp),
            "pearson":       pr,
            "spearman":      sr,
            "n_corr":        n,
            "mean_imm":      float(x.mean()),
            "mean_fut":      float(y.mean()),
            "fp_rate":       fp_rate,
            "fn_rate":       fn_rate,
            "n_pos_imm":     n_pos_imm,
            "n_fp":          int(fp_mask.sum()),
            "n_fn":          int(fn_mask.sum()),
        })

    ivf_df = pd.DataFrame(ivf_rows)
    ivf_df.to_csv(ana_dir / "immediate_vs_future.csv", index=False)
    print("\nImmediate vs Future credit correlation:")
    print(ivf_df[["trajectory", "action_type", "spearman", "fp_rate", "fn_rate"]].to_string(index=False))

    # ── Q3: delayed credit beyond immediate window? ───────────────────────────
    q3_results = {}
    for traj in args.trajectories:
        sub = iv_df[iv_df["trajectory"] == traj]
        if sub.empty:
            continue
        total_imm = float(sub["immediate_gain"].sum())
        total_fut = float(sub["future_gain"].sum())
        pct_future = (total_fut / (abs(total_imm) + abs(total_fut) + 1e-9)) * 100
        q3_results[traj] = {
            "total_immediate": total_imm,
            "total_future":    total_fut,
            "pct_future":      pct_future,
            "significant_delayed_credit": pct_future > 20.0,
        }
    print(f"\nQ3 (delayed credit beyond immediate window):")
    for traj, res in q3_results.items():
        print(f"  [{traj}] {res['pct_future']:.1f}% of credit is future → "
              f"{'YES' if res['significant_delayed_credit'] else 'NO'}")

    # ── Summary JSON ─────────────────────────────────────────────────────────
    summary = {
        "immediate_window": args.immediate_window,
        "future_start":     args.future_start,
        "q3_delayed_credit": q3_results,
        "action_type_horizons": horizon_df.to_dict("records"),
        "immediate_vs_future":  ivf_df.to_dict("records"),
    }
    with open(sentinel, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {sentinel}")


if __name__ == "__main__":
    main()

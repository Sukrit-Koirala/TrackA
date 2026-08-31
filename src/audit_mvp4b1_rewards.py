"""
audit_mvp4b1_rewards.py  --  MVP 4b.1 Stage 2

Enhanced reward audit with softer gates.  Gates that fail print a WARNING
but do not abort the pipeline (orchestrator passes check=False).

5 Gates:
  G1: finite_rewards    -- no NaN/Inf in reward columns
  G2: coverage_20pct    -- >=20% of decision points have |margin| > median(|gain|)
  G3: action_balance    -- no single action dominates >85%
  G4: info_beyond_sim   -- reward carries signal beyond top-1 sim
  G5: distinct_rewards  -- reward_local_history != reward_full (not all-same)

Usage:
  python src/audit_mvp4b1_rewards.py \\
    --output outputs_mvp4b1_onpolicy_bandit_write_clean_fast \\
    --trajectory_types oracle low_memory \\
    --reward_col reward_penalized \\
    --force
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mvp4b1_common import (
    ARTIFACT_VERSION, FEATURE_NAMES, REWARD_COLS,
    ACT_NAMES, N_ACT_TYPES,
    atomic_write_json, validate_json_cache,
)


def _load_actions(reward_dir: Path) -> pd.DataFrame:
    p = reward_dir / "actions.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    return pd.read_parquet(p)


def _gate_finite(df: pd.DataFrame, rc: str) -> tuple[bool, dict]:
    bad = {col: int((~np.isfinite(df[col].values)).sum())
           for col in REWARD_COLS if col in df.columns}
    n_bad = sum(bad.values())
    return n_bad == 0, {"non_finite_counts": bad, "total_bad": n_bad}


def _gate_coverage(df: pd.DataFrame, rc: str) -> tuple[bool, dict]:
    """≥20% of decision points have |margin| > median(|gain|)."""
    gains   = df[rc].values
    median_abs = float(np.median(np.abs(gains)))
    margins    = []
    for step, grp in df.groupby("step"):
        g = grp[rc].values
        if len(g) < 2:
            continue
        best = g.max(); worst = g.min()
        margins.append(best - worst)
    if not margins:
        return False, {"n_dp": 0}
    margins_arr = np.array(margins)
    frac_wide   = float((margins_arr > median_abs).mean())
    ok = frac_wide >= 0.20
    return ok, {"frac_wide": frac_wide, "threshold": median_abs,
                "n_dp": len(margins)}


def _gate_action_balance(df: pd.DataFrame, rc: str) -> tuple[bool, dict]:
    """No single action dominates >85% of oracle-optimal choices."""
    best_rows = []
    for step, grp in df.groupby("step"):
        best_rows.append(grp.loc[grp[rc].idxmax()])
    if not best_rows:
        return False, {"n_dp": 0}
    best_df = pd.DataFrame(best_rows)
    cnts  = best_df["action_type"].value_counts()
    total = len(best_df)
    fracs = {ACT_NAMES[int(k)]: float(v / total) for k, v in cnts.items()}
    max_frac = float(max(fracs.values())) if fracs else 1.0
    ok = max_frac <= 0.85
    return ok, {"oracle_action_fracs": fracs, "max_frac": max_frac}


def _gate_info_beyond_sim(df: pd.DataFrame, rc: str) -> tuple[bool, dict]:
    """Correlation between top-1 sim and reward should be < 0.90."""
    if "state_top1_sim" not in df.columns:
        return True, {"skipped": True}
    sims = df["state_top1_sim"].values
    rews = df[rc].values
    if len(sims) < 10:
        return True, {"skipped": True, "n": len(sims)}
    corr = float(np.corrcoef(sims, rews)[0, 1])
    ok = abs(corr) < 0.90
    return ok, {"sim_reward_corr": corr}


def _gate_distinct_rewards(df: pd.DataFrame, rc: str) -> tuple[bool, dict]:
    """reward_local_history must not equal reward_full everywhere."""
    if "reward_local_history" not in df.columns or "reward_full" not in df.columns:
        return False, {"error": "columns missing"}
    same = np.allclose(df["reward_local_history"].values,
                       df["reward_full"].values, atol=1e-7)
    frac_diff = float(
        (~np.isclose(df["reward_local_history"].values,
                     df["reward_full"].values, atol=1e-7)).mean()
    )
    return not same, {"frac_different": frac_diff}


GATES = [
    ("finite_rewards",     _gate_finite),
    ("coverage_20pct",     _gate_coverage),
    ("action_balance_85",  _gate_action_balance),
    ("info_beyond_sim",    _gate_info_beyond_sim),
    ("distinct_rewards",   _gate_distinct_rewards),
]


def audit_one(df: pd.DataFrame, rc: str, source_tag: str) -> dict:
    results = {}
    passed  = {}
    for name, fn in GATES:
        ok, info = fn(df, rc)
        status   = "PASS" if ok else "FAIL"
        results[name] = {"passed": ok, "status": status, **info}
        passed[name]  = ok
        print(f"  [{status}] {name}  {info}")

    n_pass    = sum(passed.values())
    n_gates   = len(passed)
    # Verdict: G1 (finite) is a hard gate; rest are soft warnings
    hard_gate = passed.get("finite_rewards", False)
    verdict   = "PASS" if hard_gate and n_pass >= n_gates - 1 else (
                "WARN" if hard_gate else "FAIL")

    return {
        "artifact_version": ARTIFACT_VERSION,
        "source_tag":       source_tag,
        "reward_col":       rc,
        "n_records":        len(df),
        "n_decision_points": df["step"].nunique() if "step" in df.columns else 0,
        "gates":   {k: v["passed"] for k, v in results.items()},
        "details": results,
        "n_pass":  n_pass,
        "verdict": verdict,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output",            required=True)
    ap.add_argument("--trajectory_types",  nargs="+",
                    default=["oracle", "low_memory"])
    ap.add_argument("--reward_col",        default="reward_penalized")
    ap.add_argument("--force",             action="store_true")
    args = ap.parse_args()

    out_dir   = Path(args.output)
    audit_dir = out_dir / "reward_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    sentinel  = audit_dir / "audit_report.json"

    cached = validate_json_cache(sentinel, ["artifact_version", "verdict"])
    if cached and not args.force:
        print(f"[cached] {sentinel}")
        return

    rc = args.reward_col
    combined_records = []
    per_source = {}

    for ttype in args.trajectory_types:
        rd = out_dir / "reward_datasets" / ttype
        print(f"\n-- Auditing {ttype} --")
        try:
            df = _load_actions(rd)
        except FileNotFoundError as e:
            print(f"  WARNING: {e}, skipping")
            continue
        combined_records.append(df)
        per_source[ttype] = audit_one(df, rc, ttype)

    # Audit combined
    if not combined_records:
        print("ERROR: no reward datasets found!")
        report = {
            "artifact_version": ARTIFACT_VERSION,
            "verdict": "FAIL",
            "error": "no datasets",
        }
    else:
        all_df = pd.concat(combined_records, ignore_index=True)
        print(f"\n-- Auditing combined ({len(all_df):,} records) --")
        combined_audit = audit_one(all_df, rc, "combined")

        # Summary stats
        gains = all_df[rc].values
        action_dist = {
            ACT_NAMES[k]: int(v)
            for k, v in all_df["action_type"].value_counts().items()
            if isinstance(k, (int, np.integer))
        }

        report = {
            "artifact_version": ARTIFACT_VERSION,
            "reward_col":       rc,
            "n_records":        int(len(all_df)),
            "n_decision_points": int(all_df["step"].nunique()
                                    if "step" in all_df.columns else 0),
            "trajectory_types": args.trajectory_types,
            "per_source":       per_source,
            "combined":         combined_audit,
            "gates":            combined_audit["gates"],
            "verdict":          combined_audit["verdict"],
            "gain_stats": {
                "mean":   float(np.nanmean(gains)),
                "std":    float(np.nanstd(gains)),
                "p10":    float(np.nanpercentile(gains, 10)),
                "p50":    float(np.nanmedian(gains)),
                "p90":    float(np.nanpercentile(gains, 90)),
            },
            "action_distribution": action_dist,
        }

    atomic_write_json(sentinel, report)
    print(f"\nAudit complete: verdict={report['verdict']}")
    print(f"Saved: {sentinel}")

    if report["verdict"] == "FAIL":
        print("WARNING: audit FAIL — check gates before proceeding")


if __name__ == "__main__":
    main()

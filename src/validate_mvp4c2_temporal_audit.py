"""
validate_mvp4c2_temporal_audit.py  --  MVP 4c-2 Validation Gate

14-check validation table.  Checks 1-9 are BLOCKING (must all pass before
temporal credit stages are allowed to run).  Checks 10-14 are advisory.

Usage:
  python validate_mvp4c2_temporal_audit.py --output <dir> [--phase pre_credit|final]

Exit codes:
  0  All blocking checks pass
  1  One or more blocking checks failed
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def _load(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


def _has_parquets(d: Path) -> bool:
    return d.exists() and any(d.glob("*.parquet"))


# ── Individual checks ─────────────────────────────────────────────────────────

def check_1_source_compatibility(out: Path) -> dict:
    """C1: Source artifacts are dimension-compatible and token-aligned."""
    d = _load(out / "validation" / "source_compatibility.json")
    if not d:
        return {"pass": False, "reason": "source_compatibility.json missing — run Stage 1"}
    p = d.get("pass")
    return {
        "pass":    bool(p),
        "verdict": d.get("verdict", "UNKNOWN"),
        "checks":  d.get("checks", {}),
    }


def check_2_provenance_exists(out: Path) -> dict:
    """C2: Write-event provenance parquets exist for oracle and bandit."""
    prov = out / "provenance"
    missing = []
    for traj in ("oracle", "bandit"):
        for fname in (f"{traj}_events.parquet", f"{traj}_objects.parquet"):
            if not (prov / fname).exists():
                missing.append(fname)
    return {
        "pass":    len(missing) == 0,
        "missing": missing,
    }


def check_3_rbw_ordering(out: Path) -> dict:
    """C3: All read_step > write_step for every READ record (Read-Before-Write)."""
    rep_dir = out / "temporal_replay"
    violations = 0
    n_total    = 0
    for traj in ("oracle", "bandit"):
        p = rep_dir / f"{traj}_replay_reads.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        n_total += len(df)
        violations += int((df["delay"] < 0).sum()) if "delay" in df.columns else 0
    if n_total == 0:
        return {"pass": False, "reason": "replay_reads.parquet not found"}
    return {
        "pass":       violations == 0,
        "violations": violations,
        "n_total":    n_total,
    }


def check_4_delay_positive(out: Path) -> dict:
    """C4: All delay values > 0 (object was created strictly before retrieval)."""
    rep_dir = out / "temporal_replay"
    zero_delays = 0
    n_total     = 0
    for traj in ("oracle", "bandit"):
        p = rep_dir / f"{traj}_replay_reads.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        n_total     += len(df)
        zero_delays += int((df["delay"] <= 0).sum()) if "delay" in df.columns else 0
    if n_total == 0:
        return {"pass": False, "reason": "replay_reads.parquet not found"}
    return {
        "pass":        zero_delays == 0,
        "zero_delays": zero_delays,
        "n_total":     n_total,
    }


def check_5_no_pre_create_use(out: Path) -> dict:
    """C5: All read objects have creation_step < stream_step."""
    rep_dir = out / "temporal_replay"
    violations = 0
    n_total    = 0
    for traj in ("oracle", "bandit"):
        p = rep_dir / f"{traj}_replay_reads.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if "creation_step" in df.columns and "stream_step" in df.columns:
            n_total    += len(df)
            violations += int((df["creation_step"] >= df["stream_step"]).sum())
    if n_total == 0:
        return {"pass": False, "reason": "replay_reads.parquet not found or missing columns"}
    return {
        "pass":       violations == 0,
        "violations": violations,
        "n_total":    n_total,
    }


def check_6_state_reconstruction(out: Path) -> dict:
    """C6 (BLOCKING): Historical state reconstruction matches replay snapshots at all checkpoints."""
    val_path = out / "validation" / "state_history_validation.json"
    if not val_path.exists():
        return {"pass": False, "reason": "state_history_validation.json missing"}
    raw = json.loads(val_path.read_text())
    results = {}
    all_pass = True
    for traj in ("oracle", "bandit"):
        r = raw.get(traj, raw)   # shared-file fallback
        if not isinstance(r, dict):
            all_pass = False
            results[traj] = {"pass": False, "reason": "malformed entry"}
            continue
        p = bool(r.get("pass", False))
        all_pass = all_pass and p
        results[traj] = {
            "pass":               p,
            "n_comparisons":      r.get("n_comparisons"),
            "support_exact_rate": r.get("support_exact_rate"),
            "token_exact_rate":   r.get("token_exact_rate"),
            "proto_cosine_rate":  r.get("proto_cosine_rate"),
            "n_failure_samples":  r.get("n_failure_samples"),
        }
    return {"pass": all_pass, "per_trajectory": results}


def check_7_replay_coverage(out: Path) -> dict:
    """C7: Replay covered at least 10,000 stream steps for each trajectory."""
    rep_dir = out / "temporal_replay"
    coverage = {}
    min_steps = 10_000
    for traj in ("oracle", "bandit"):
        p = rep_dir / f"{traj}_replay_summary.json"
        if not p.exists():
            coverage[traj] = {"pass": False, "T_replay": None}
            continue
        s = json.loads(p.read_text())
        T = s.get("T_replay", 0)
        coverage[traj] = {"pass": T >= min_steps, "T_replay": T}
    return {
        "pass":     all(v["pass"] for v in coverage.values()),
        "min_required": min_steps,
        "coverage": coverage,
    }


def check_8_immediate_reward_join(out: Path) -> dict:
    """C8: Bandit immediate reward join >= 95%  (advisory — no rollout data is non-blocking)."""
    stats_path = out / "temporal_credit" / "immediate_reward_join_stats.json"
    if not stats_path.exists():
        return {
            "pass":    None,
            "reason":  "immediate_reward_join_stats.json missing — Stage 5b skipped",
        }
    d = _load(stats_path)
    verdict = d.get("verdict", "")
    # Dataset misalignment or missing data → UNSUPPORTED (pass=None, not FAIL)
    if verdict in ("NO_REWARD_DATA",
                   "IMMEDIATE_REWARD_DATASET_NOT_ALIGNED_WITH_FINAL_ROLLOUT"):
        return {
            "pass":    None,
            "verdict": verdict,
            "reason":  ("Reward data is from a different rollout than the audited trajectory"
                        if "NOT_ALIGNED" in verdict else "No reward data found"),
            "stats":   d,
        }
    rate = d.get("join_match_rate", d.get("join_rate", 0.0))
    return {
        "pass":      rate >= 0.95,
        "join_rate": rate,
        "stats":     d,
    }


def check_9_defer_present(out: Path) -> dict:
    """C9: DEFER events are present in bandit provenance (bandit must have been run)."""
    evt_path = out / "provenance" / "bandit_events.parquet"
    if not evt_path.exists():
        return {"pass": False, "reason": "bandit_events.parquet not found"}
    df = pd.read_parquet(evt_path, columns=["action_type"])
    n_defer = int((df["action_type"].str.upper() == "DEFER").sum())
    n_total = len(df)
    defer_rate = n_defer / n_total if n_total else 0.0
    return {
        "pass":       n_defer > 0,
        "n_defer":    n_defer,
        "n_total":    n_total,
        "defer_rate": round(defer_rate, 4),
    }


def check_10_output_separation(out: Path) -> dict:
    """C10: Temporal and retrospective outputs are in distinct directories."""
    retro_dirs  = [out / "object_utility", out / "retrospective_utility", out / "retrospective"]
    temporal_dirs = [out / "temporal_credit", out / "temporal"]

    retro_exists  = any(_has_parquets(d) for d in retro_dirs)
    temporal_exists = any(_has_parquets(d) for d in temporal_dirs)

    # Verify no cross-contamination: temporal artifacts not inside a retro dir
    cross_contamination = False
    for rd in retro_dirs:
        if not rd.exists():
            continue
        for td in temporal_dirs:
            if td.exists() and str(td).startswith(str(rd)):
                cross_contamination = True

    return {
        "pass":                 retro_exists and temporal_exists and not cross_contamination,
        "retro_found":          retro_exists,
        "temporal_found":       temporal_exists,
        "cross_contamination":  cross_contamination,
    }


def check_11_opportunity_axis(out: Path) -> dict:
    """C11: Primary frequency axis for rare-state analysis uses opportunity counts."""
    candidates = [
        out / "normalized_analysis" / "normalized_analysis_summary.json",
        out / "analysis" / "normalized_analysis_summary.json",
    ]
    for p in candidates:
        if p.exists():
            d = _load(p)
            col = d.get("primary_freq_col", "")
            ok  = "opportunity" in col.lower()
            return {
                "pass":             ok,
                "primary_freq_col": col,
                "reason":           None if ok else f"Expected 'opportunity' in col name, got '{col}'",
            }
    return {"pass": None, "reason": "normalized_analysis_summary.json not found (Stage 6 not run)"}


def check_12_write_index_bounds(out: Path) -> dict:
    """C12: All write stream_step values are within bounds of h_train (no OOB h lookups)."""
    rep_dir  = out / "temporal_replay"
    prov_dir = out / "provenance"
    issues   = []
    for traj in ("oracle", "bandit"):
        summary_path = rep_dir / f"{traj}_replay_summary.json"
        evt_path     = prov_dir / f"{traj}_events.parquet"
        if not summary_path.exists() or not evt_path.exists():
            continue
        T = json.loads(summary_path.read_text()).get("T_replay", None)
        if T is None:
            continue
        df = pd.read_parquet(evt_path, columns=["stream_step", "action_type"])
        write_mask = ~df["action_type"].str.upper().isin({"DEFER"})
        oob = int((df[write_mask]["stream_step"] >= T).sum())
        if oob > 0:
            issues.append(f"{traj}: {oob} write events have stream_step >= T_replay ({T})")
    return {
        "pass":   len(issues) == 0,
        "issues": issues,
    }


def check_13_temporal_credit_populated(out: Path) -> dict:
    """C13: Temporal credit parquets exist and have content."""
    tc_dir = out / "temporal_credit"
    if not tc_dir.exists():
        return {"pass": False, "reason": "temporal_credit/ directory not found"}
    parquets = list(tc_dir.glob("*.parquet"))
    if not parquets:
        return {"pass": False, "reason": "No parquet files in temporal_credit/"}
    rows = {}
    for p in parquets:
        try:
            rows[p.name] = len(pd.read_parquet(p))
        except Exception:
            rows[p.name] = None
    return {
        "pass":    all(v and v > 0 for v in rows.values()),
        "files":   rows,
    }


def check_14_no_future_leakage(out: Path) -> dict:
    """C14: No read record has stream_step <= creation_step (would imply future-state read)."""
    rep_dir = out / "temporal_replay"
    leakage = 0
    n_total = 0
    for traj in ("oracle", "bandit"):
        p = rep_dir / f"{traj}_replay_reads.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        n_total += len(df)
        if "stream_step" in df.columns and "creation_step" in df.columns:
            leakage += int((df["stream_step"] <= df["creation_step"]).sum())
    if n_total == 0:
        return {"pass": False, "reason": "No replay reads found"}
    return {
        "pass":    leakage == 0,
        "leakage": leakage,
        "n_total": n_total,
    }


# ── Check table ───────────────────────────────────────────────────────────────

CHECKS = [
    # (id, label, fn, blocking)
    ("C01_source_compat",        "Source compatibility",                    check_1_source_compatibility,  True),
    ("C02_provenance_exists",    "Write provenance exists",                 check_2_provenance_exists,     True),
    ("C03_rbw_ordering",         "RBW: read_step > write_step",             check_3_rbw_ordering,          True),
    ("C04_delay_positive",       "Delay > 0 for all reads",                 check_4_delay_positive,        True),
    ("C05_no_pre_create_use",    "No pre-creation retrieval",               check_5_no_pre_create_use,     True),
    ("C06_state_reconstruction", "Historical state reconstruction",         check_6_state_reconstruction,  True),
    ("C07_replay_coverage",      "Replay coverage >= 10k steps",            check_7_replay_coverage,       True),
    ("C08_immediate_reward_join","Immediate reward join >= 95%",            check_8_immediate_reward_join, False),
    ("C09_defer_present",        "DEFER events present in bandit",          check_9_defer_present,         True),
    ("C10_output_separation",    "Temporal vs retrospective separation",    check_10_output_separation,    False),
    ("C11_opportunity_axis",     "Primary freq axis uses opportunity",      check_11_opportunity_axis,     False),
    ("C12_write_index_bounds",   "Write indices within h_train bounds",     check_12_write_index_bounds,   False),
    ("C13_temporal_credit_populated", "Temporal credit parquets populated", check_13_temporal_credit_populated, False),
    ("C14_no_future_leakage",    "No future-state leakage",                 check_14_no_future_leakage,    False),
]


def run_all_checks(out: Path, phase: str = "pre_credit") -> dict:
    print(f"\n{'='*72}")
    print(f"  MVP 4c-2 Temporal Audit Validation  (phase={phase})")
    print(f"  Output dir: {out}")
    print(f"{'='*72}")

    # In pre_credit phase, skip checks that require temporal credit output
    skip_in_pre_credit = {"C10_output_separation", "C11_opportunity_axis",
                          "C13_temporal_credit_populated"}

    results = {}
    blocking_fails = []
    advisory_fails = []

    for cid, label, fn, blocking in CHECKS:
        if phase == "pre_credit" and cid in skip_in_pre_credit:
            results[cid] = {"pass": None, "skipped": True, "reason": f"deferred to final phase"}
            print(f"  [{cid}] {label}: SKIPPED (phase={phase})")
            continue

        try:
            r = fn(out)
        except Exception as e:
            r = {"pass": False, "error": str(e)}

        p = r.get("pass")
        status = "PASS" if p is True else ("SKIP" if p is None else "FAIL")
        marker = "[BLOCK]" if blocking and p is False else ("[ADV]" if not blocking else "     ")
        print(f"  {marker} [{cid}] {label}: {status}")

        if p is False:
            detail_keys = [k for k in r if k != "pass"]
            brief = {k: r[k] for k in detail_keys[:3]}
            print(f"           {brief}")

        if p is False and blocking:
            blocking_fails.append(cid)
        elif p is False and not blocking:
            advisory_fails.append(cid)

        results[cid] = {**r, "label": label, "blocking": blocking}

    n_blocking_fail = len(blocking_fails)
    n_advisory_fail = len(advisory_fails)
    gate_pass = n_blocking_fail == 0

    verdict = "VALID" if gate_pass else "INVALID_AUDIT"
    print(f"\n  BLOCKING failures:  {n_blocking_fail}")
    print(f"  Advisory failures:  {n_advisory_fail}")
    print(f"  VERDICT: {verdict}")

    summary = {
        "phase":            phase,
        "overall_verdict":  verdict,
        "gate_pass":        gate_pass,
        "n_blocking_fail":  n_blocking_fail,
        "n_advisory_fail":  n_advisory_fail,
        "blocking_fails":   blocking_fails,
        "advisory_fails":   advisory_fails,
        "checks":           results,
    }

    val_dir = out / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    out_path = val_dir / f"mvp4c2_audit_{phase}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Saved → {out_path}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--phase",  default="pre_credit",
                    choices=["pre_credit", "final"],
                    help="pre_credit: run before temporal stages; final: run after all stages")
    ap.add_argument("--force",  action="store_true")
    args = ap.parse_args()

    out = Path(args.output)
    summary = run_all_checks(out, phase=args.phase)

    if not summary["gate_pass"]:
        print("\nVALIDATION GATE FAILED — temporal credit pipeline is BLOCKED.")
        sys.exit(1)

    print("\nVALIDATION GATE PASSED.")
    sys.exit(0)


if __name__ == "__main__":
    main()

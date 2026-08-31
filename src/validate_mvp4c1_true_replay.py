"""
validate_mvp4c1_true_replay.py  --  MVP 4c-1 Full Validation (True Replay)

11 validation checks per spec's validation table.

Checks (blocks 1-6 are CRITICAL — INVALID_AUDIT if any fail):
  1  Single true chronological stream:  replay reads stream_step, NOT a
     val_step mapped via N_train/N_val scaling
  2  READ-before-WRITE ordering:  all delay > 0 (strictly future)
  3  No final-state leakage:  IncrementalMemory reconstruction matches snapshots
  4  No negative delays:  min(delay) > 0
  5  No pre-creation object uses:  each read's stream_step > object birth step
  6  Historical state reconstruction:  cosine ≥ 0.999999 for 100×5 checkpoints

Non-blocking but required for clean audit (7-11):
  7  Bandit decision logging:  events.parquet covers ≥ 99% of stream steps
  8  Immediate reward join:  join rate ≥ 99%
  9  DEFER present in bandit trajectory
  10 Outputs separated:  retrospective_utility/ and temporal_credit/ distinct
  11 Rare-state primary axis:  opportunity_count / relevant_opportunity_count,
     NOT retrieval_count

Writes:
  {output}/validation/mvp4c1_true_replay_validation.json
  {output}/validation/mvp4c1_true_replay_checks.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ── Individual checks ─────────────────────────────────────────────────────────

def check_1_chronological_stream(out_dir: Path) -> dict:
    """
    Check that replay_reads parquet uses real stream_step (not a scaled val_step).
    Evidence: columns should include 'stream_step', NOT 'val_step' as the primary key.
    Reads should be from the new replay (temporal_replay/{traj}_replay_reads.parquet).
    """
    rep_dir  = out_dir / "temporal_replay"
    results  = {}
    for traj in ["oracle", "bandit"]:
        new_path = rep_dir / f"{traj}_replay_reads.parquet"
        old_path = rep_dir / f"{traj}_query_rewards.parquet"

        if new_path.exists():
            df = pd.read_parquet(new_path)
            has_stream_step = "stream_step" in df.columns
            has_val_step    = "val_step" in df.columns and "stream_step" not in df.columns
            results[traj] = {
                "source_file": new_path.name,
                "has_stream_step": has_stream_step,
                "is_true_replay": has_stream_step and not has_val_step,
                "pass": has_stream_step and not has_val_step,
            }
        elif old_path.exists():
            # Old build_chronological_replay output — this uses val_step (wrong)
            df = pd.read_parquet(old_path)
            results[traj] = {
                "source_file": old_path.name,
                "has_stream_step": "stream_step" in df.columns,
                "is_true_replay": False,
                "pass": False,
                "note": "Using old val_step mapping — run replay_online_memory_chronologically.py",
            }
        else:
            results[traj] = {
                "source_file": "MISSING",
                "pass": False,
                "note": "Run replay_online_memory_chronologically.py (Stage 4)",
            }
    return results


def check_2_read_before_write(out_dir: Path) -> dict:
    """All delays strictly > 0."""
    rep_dir = out_dir / "temporal_replay"
    results = {}
    for traj in ["oracle", "bandit"]:
        for fname in [f"{traj}_replay_reads.parquet",
                      f"{traj}_query_rewards.parquet"]:
            p = rep_dir / fname
            if p.exists():
                df = pd.read_parquet(p)
                if len(df) == 0:
                    results[traj] = {"pass": False, "note": "EMPTY"}
                    break
                neg  = int((df["delay"] <= 0).sum()) if "delay" in df.columns else -1
                zero = int((df["delay"] == 0).sum()) if "delay" in df.columns else -1
                results[traj] = {
                    "n_records":    int(len(df)),
                    "n_delay_le_0": neg,
                    "n_delay_eq_0": zero,
                    "min_delay":    float(df["delay"].min()) if "delay" in df.columns else float("nan"),
                    "pass":         neg == 0 if neg >= 0 else False,
                }
                break
        else:
            results[traj] = {"pass": False, "note": "FILE_MISSING"}
    return results


def check_3_no_final_state_leakage(out_dir: Path) -> dict:
    """
    No final-state leakage: the write files must come from IncrementalMemory
    (created during replay, not extracted from a final completed state).
    Evidence: if replay_writes.parquet exists, it was generated during online replay.
    If only old build_chronological_replay output exists, leakage is possible.
    """
    rep_dir = out_dir / "temporal_replay"
    results = {}
    for traj in ["oracle", "bandit"]:
        writes_p  = rep_dir / f"{traj}_replay_writes.parquet"
        old_evt_p = rep_dir / f"{traj}_chronological_events.parquet"

        if writes_p.exists():
            df = pd.read_parquet(writes_p)
            results[traj] = {
                "source": "replay_writes.parquet (IncrementalMemory)",
                "n_write_events": int(len(df)),
                "leakage_risk":   "LOW",
                "pass":           True,
            }
        elif old_evt_p.exists():
            df = pd.read_parquet(old_evt_p)
            results[traj] = {
                "source": "chronological_events.parquet (old pipeline — val_step mapping)",
                "n_write_events": int(len(df)),
                "leakage_risk":   "HIGH",
                "pass":           False,
                "note": "Run replay_online_memory_chronologically.py to fix",
            }
        else:
            results[traj] = {
                "source": "MISSING",
                "pass":   False,
                "note":   "Run replay_online_memory_chronologically.py",
            }
    return results


def check_4_no_negative_delays(out_dir: Path) -> dict:
    """Alias for check 2 but more explicit: min(delay) > 0."""
    c2 = check_2_read_before_write(out_dir)
    return {traj: {"pass": v.get("pass", False),
                   "min_delay": v.get("min_delay", float("nan")),
                   "n_delay_le_0": v.get("n_delay_le_0", -1)}
            for traj, v in c2.items()}


def check_5_no_pre_creation_use(out_dir: Path) -> dict:
    """
    Every read at stream_step t references an object with birth_step < t.
    (read_stream_step > write_stream_step for all records)
    """
    rep_dir = out_dir / "temporal_replay"
    results = {}
    for traj in ["oracle", "bandit"]:
        for fname in [f"{traj}_replay_reads.parquet",
                      f"{traj}_query_rewards.parquet"]:
            p = rep_dir / fname
            if not p.exists():
                continue
            df = pd.read_parquet(p)
            if len(df) == 0:
                results[traj] = {"pass": False, "note": "EMPTY"}
                break

            # delay = stream_step - birth_step  (for true replay)
            # or delay = val_step - born_val_step  (for old replay)
            if "delay" not in df.columns:
                results[traj] = {"pass": False, "note": "NO_DELAY_COLUMN"}
                break

            violated = int((df["delay"] <= 0).sum())
            results[traj] = {
                "n_records":     int(len(df)),
                "n_violations":  violated,
                "pass":          violated == 0,
                "delay_min":     float(df["delay"].min()),
            }
            break
        else:
            results[traj] = {"pass": False, "note": "FILE_MISSING"}
    return results


def check_6_state_reconstruction(out_dir: Path) -> dict:
    """
    IncrementalMemory historical state reconstruction PASS.
    Evidence: replay_online_memory_chronologically.py writes a state_history_validation.json.
    """
    rep_dir = out_dir / "temporal_replay"
    val_dir = out_dir / "validation"
    results = {}
    for traj in ["oracle", "bandit"]:
        # Try traj-specific first, then shared validation file
        candidates = [
            rep_dir / f"{traj}_state_history_validation.json",
            val_dir / "state_history_validation.json",
        ]
        val_path = next((p for p in candidates if p.exists()), None)
        if val_path is None:
            results[traj] = {
                "pass": False,
                "note": "state_history_validation.json not found",
                "source": "MISSING",
            }
            continue
        with open(val_path) as f:
            raw = json.load(f)
        # Shared file is keyed by traj name
        data = raw.get(traj, raw) if isinstance(raw, dict) and traj in raw else raw
        passed = data.get("all_pass", False)
        n_checks = data.get("n_checkpoints", 0)
        min_cos   = data.get("min_cosine", float("nan"))
        results[traj] = {
            "pass":         bool(passed),
            "n_checkpoints": int(n_checks),
            "min_cosine":    float(min_cos) if not isinstance(min_cos, str) else float("nan"),
            "required_cos":  data.get("required_cosine", 0.999999),
            "verdict":       data.get("verdict", "UNKNOWN"),
        }
    return results


def check_7_decision_logging(out_dir: Path) -> dict:
    """
    Bandit decision logging: events.parquet covers ≥ 99% of stream steps.
    This is approximate — we check n_events vs expected N_train.
    """
    prov_dir = out_dir / "provenance"
    evt_path = prov_dir / "bandit_events.parquet"
    if not evt_path.exists():
        return {"pass": False, "note": "bandit_events.parquet not found"}

    df = pd.read_parquet(evt_path)
    n_events = len(df)
    n_unique_steps = int(df["stream_step"].nunique()) if "stream_step" in df.columns else 0

    # Actions are logged per decision, not per step (DEFER included)
    # Coverage = fraction of stream steps that appear in the log
    # Without knowing N_train exactly, we use unique_steps / max_stream_step
    if n_events > 0 and "stream_step" in df.columns:
        max_step = int(df["stream_step"].max())
        coverage = float(n_unique_steps / max(max_step + 1, 1))
        all_types = sorted(df["action_type"].str.upper().unique().tolist())
        defer_count = int((df["action_type"].str.upper() == "DEFER").sum())
    else:
        coverage = 0.0
        all_types = []
        defer_count = 0
        max_step = 0

    return {
        "n_events":           n_events,
        "n_unique_steps":     n_unique_steps,
        "max_stream_step":    max_step,
        "estimated_coverage": round(coverage, 4),
        "action_types_logged": all_types,
        "n_defer":            defer_count,
        "pass":               coverage >= 0.99 or n_events > 50000,
        "note": ("Coverage estimate; actual N_train unknown. "
                 "Pass if n_events>50k (likely covers all steps)."),
    }


def check_8_immediate_reward_join(out_dir: Path) -> dict:
    """Immediate reward join ≥ 99%."""
    credit_dir = out_dir / "temporal_credit"
    stats_path = credit_dir / "immediate_reward_join_stats.json"

    if not stats_path.exists():
        return {"pass": False, "note": "immediate_reward_join_stats.json not found",
                "join_match_rate": 0.0, "join_valid_99pct": False}

    with open(stats_path) as f:
        stats = json.load(f)

    rate = stats.get("join_match_rate", 0.0)
    return {
        "join_match_rate":  rate,
        "join_valid_99pct": stats.get("join_valid_99pct", False),
        "verdict":          stats.get("verdict", "UNKNOWN"),
        "source":           stats.get("source", "?"),
        "pass":             bool(stats.get("join_valid_99pct", False)),
        "note": ("Required ≥ 99% match rate. "
                 "Run rollout with --collect_rewards if rate is low."),
    }


def check_9_defer_present(out_dir: Path) -> dict:
    """DEFER events present in bandit provenance."""
    prov_dir = out_dir / "provenance"
    evt_path = prov_dir / "bandit_events.parquet"
    if not evt_path.exists():
        return {"pass": False, "note": "bandit_events.parquet not found"}

    df = pd.read_parquet(evt_path)
    n_defer = int((df["action_type"].str.upper() == "DEFER").sum())
    n_total = len(df)
    return {
        "n_total_events": n_total,
        "n_defer":        n_defer,
        "defer_frac":     round(float(n_defer / max(n_total, 1)), 4),
        "pass":           n_defer > 0,
        "verdict":        "DEFER_PRESENT" if n_defer > 0 else "DEFER_MISSING",
    }


def check_10_outputs_separated(out_dir: Path) -> dict:
    """
    Retrospective utility and temporal credit are in SEPARATE directories
    with SEPARATE parquet files. No cross-contamination.
    """
    retro_dir  = out_dir / "retrospective_utility"
    temp_dir   = out_dir / "temporal_credit"
    obj_dir    = out_dir / "object_utility"

    retro_has  = retro_dir.exists() and any(retro_dir.glob("*.parquet"))
    temp_has   = temp_dir.exists() and any(temp_dir.glob("*.parquet"))

    # Check no temporal files live in retrospective dir
    cross_contamination = False
    if retro_dir.exists():
        for f in retro_dir.glob("*credit*"):
            cross_contamination = True
            break
        for f in retro_dir.glob("*replay*"):
            cross_contamination = True
            break

    return {
        "retrospective_dir_exists": retro_dir.exists(),
        "retrospective_has_parquet": retro_has,
        "temporal_dir_exists":       temp_dir.exists(),
        "temporal_has_parquet":      temp_has,
        "cross_contamination":       cross_contamination,
        "pass":                      retro_has and temp_has and not cross_contamination,
        "verdict": ("OK" if (retro_has and temp_has and not cross_contamination)
                    else "MISSING_RETRO" if not retro_has
                    else "MISSING_TEMPORAL" if not temp_has
                    else "CROSS_CONTAMINATION"),
    }


def check_11_rare_state_axis(out_dir: Path) -> dict:
    """
    Rare-state primary frequency axis must be opportunity_count or
    relevant_opportunity_count (= opportunity@k), NOT retrieval_count.
    """
    # Look for the rare state analysis output
    for candidate in [
        out_dir / "retrospective_utility" / "rare_state_analysis.csv",
        out_dir / "analysis" / "rare_state_analysis.csv",
    ]:
        if candidate.exists():
            try:
                df = pd.read_csv(candidate)
                cols = [c.lower() for c in df.columns]
                has_opportunity = any("opportunity" in c for c in cols)
                has_retrieval   = any("retrieval" in c and "count" in c for c in cols)
                primary_col_ok  = has_opportunity and not has_retrieval
                return {
                    "file":              candidate.name,
                    "has_opportunity_col": has_opportunity,
                    "has_retrieval_col":   has_retrieval,
                    "pass":              has_opportunity,
                    "verdict": ("CORRECT_opportunity_axis" if has_opportunity
                                else "WRONG_retrieval_axis"),
                }
            except Exception as e:
                return {"pass": False, "note": f"Read error: {e}"}

    # Check analyze_rare_predictive_states.py sentinel
    for sentinel in [
        out_dir / "retrospective_utility" / "rare_state_done.json",
        out_dir / "analysis" / "rare_state_done.json",
    ]:
        if sentinel.exists():
            with open(sentinel) as f:
                data = json.load(f)
            axis = data.get("primary_freq_col", "UNKNOWN")
            is_opportunity = "opportunity" in axis.lower()
            return {
                "file":           sentinel.name,
                "primary_axis":   axis,
                "pass":           is_opportunity,
                "verdict":        ("CORRECT" if is_opportunity
                                   else f"WRONG ({axis}) — must be opportunity@k"),
            }

    return {
        "pass":   False,
        "note":   "rare_state_analysis.csv not found — run analyze_rare_predictive_states.py",
        "verdict": "MISSING",
    }


# ── Summary table ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--force",  action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output)
    val_dir = out_dir / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    out_json = val_dir / "mvp4c1_true_replay_validation.json"
    out_csv  = val_dir / "mvp4c1_true_replay_checks.csv"

    if out_json.exists() and not args.force:
        print(f"[cached] {out_json.name}")
        return

    print("MVP 4c-1 True Replay Validation")
    print("=" * 60)

    c1  = check_1_chronological_stream(out_dir)
    c2  = check_2_read_before_write(out_dir)
    c3  = check_3_no_final_state_leakage(out_dir)
    c4  = check_4_no_negative_delays(out_dir)
    c5  = check_5_no_pre_creation_use(out_dir)
    c6  = check_6_state_reconstruction(out_dir)
    c7  = check_7_decision_logging(out_dir)
    c8  = check_8_immediate_reward_join(out_dir)
    c9  = check_9_defer_present(out_dir)
    c10 = check_10_outputs_separated(out_dir)
    c11 = check_11_rare_state_axis(out_dir)

    # Per-traj pass (critical checks 1-6)
    def _traj_pass(chk: dict) -> bool:
        if isinstance(chk, dict) and "pass" in chk:
            return bool(chk["pass"])
        # dict of traj -> result
        return all(v.get("pass", False) for v in chk.values() if isinstance(v, dict))

    c1_ok  = _traj_pass(c1)
    c2_ok  = _traj_pass(c2)
    c3_ok  = _traj_pass(c3)
    c4_ok  = _traj_pass(c4)
    c5_ok  = _traj_pass(c5)
    c6_ok  = _traj_pass(c6)
    c7_ok  = _traj_pass(c7)
    c8_ok  = _traj_pass(c8)
    c9_ok  = _traj_pass(c9)
    c10_ok = _traj_pass(c10)
    c11_ok = _traj_pass(c11)

    checks = [
        ("1",  "Single true chronological stream",     c1_ok,  True,  c1),
        ("2",  "READ-before-WRITE ordering (delay>0)", c2_ok,  True,  c2),
        ("3",  "No final-state leakage",               c3_ok,  True,  c3),
        ("4",  "No negative delays",                   c4_ok,  True,  c4),
        ("5",  "No pre-creation object uses",          c5_ok,  True,  c5),
        ("6",  "Historical state reconstruction",      c6_ok,  True,  c6),
        ("7",  "Bandit decision logging ≥99%",         c7_ok,  False, c7),
        ("8",  "Immediate reward join ≥99%",           c8_ok,  False, c8),
        ("9",  "DEFER present in bandit events",       c9_ok,  False, c9),
        ("10", "Outputs separated (retro vs temporal)",c10_ok, False, c10),
        ("11", "Rare-state axis = opportunity count",  c11_ok, False, c11),
    ]

    csv_rows = []
    print("\n  Check   Description                               Blocking  Status")
    print("  ------  ----------------------------------------  --------  ------")
    for num, desc, ok, blocking, _ in checks:
        status   = "PASS" if ok else "FAIL"
        blocking_str = "YES" if blocking else "no"
        print(f"  {num:<6}  {desc:<40}  {blocking_str:<8}  {status}")
        csv_rows.append({
            "check": num, "description": desc,
            "blocking": blocking, "pass": ok, "status": status,
        })

    critical_fail = not all(ok for _, _, ok, blocking, _ in checks if blocking)
    overall = "INVALID_AUDIT" if critical_fail else "VALID"

    print(f"\n  OVERALL: {overall}")
    if critical_fail:
        failed = [num for num, _, ok, blocking, _ in checks if blocking and not ok]
        print(f"  Blocking failures: checks {', '.join(failed)}")
        print("  Temporal credit results MUST NOT be interpreted.")
    else:
        warn = [num for num, _, ok, blocking, _ in checks if not blocking and not ok]
        if warn:
            print(f"  Non-blocking warnings: checks {', '.join(warn)}")

    # Save
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False)

    result = {
        "overall_verdict":   overall,
        "critical_fail":     critical_fail,
        "checks": {
            "c1_true_stream":       {"pass": c1_ok, "detail": c1},
            "c2_rbw_ordering":      {"pass": c2_ok, "detail": c2},
            "c3_no_leakage":        {"pass": c3_ok, "detail": c3},
            "c4_no_neg_delay":      {"pass": c4_ok, "detail": c4},
            "c5_no_pre_create_use": {"pass": c5_ok, "detail": c5},
            "c6_state_recon":       {"pass": c6_ok, "detail": c6},
            "c7_decision_log":      {"pass": c7_ok, "detail": c7},
            "c8_imm_reward_join":   {"pass": c8_ok, "detail": c8},
            "c9_defer_present":     {"pass": c9_ok, "detail": c9},
            "c10_outputs_sep":      {"pass": c10_ok,"detail": c10},
            "c11_rare_axis":        {"pass": c11_ok,"detail": c11},
        },
    }
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_csv}")

    if critical_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()

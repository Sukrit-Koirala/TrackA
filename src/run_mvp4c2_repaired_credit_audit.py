"""
run_mvp4c2_repaired_credit_audit.py  --  MVP 4c-2 Orchestrator

Repaired temporal credit audit with blocking validation gate.

Stage layout:
  Stage 01:  inspect_source_compatibility         (blocks on dim / token mismatch)
  Stage 02a: build_write_object_provenance        (oracle)
  Stage 02b: build_write_object_provenance        (bandit)
  Stage 03a: replay_online_memory_chronologically  (oracle, student h, fixed bugs)
  Stage 03b: replay_online_memory_chronologically  (bandit)
  Stage 04a: log_read_object_usage                (oracle, retrospective)
  Stage 04b: log_read_object_usage                (bandit, retrospective)
  Stage 05a: compute_object_counterfactual_utility (oracle)
  Stage 05b: compute_object_counterfactual_utility (bandit)
  Stage 06a: analyze_normalized_object_utility     (oracle)
  Stage 06b: analyze_normalized_object_utility     (bandit)
  Stage 07:  analyze_rare_predictive_states
  Stage 08:  analyze_defer_with_controls
  Stage 09:  join_bandit_immediate_rewards         (non-blocking)
  Stage 10:  *** BLOCKING VALIDATION GATE ***      (validate_mvp4c2_temporal_audit --phase pre_credit)
  Stage 11:  compute_true_temporal_credit
  Stage 12:  predict_future_object_utility
  Stage 13:  validate_mvp4c2_temporal_audit        (final pass)
  Stage 14:  synthesis / report

Output: outputs_mvp4c2_repaired_credit_audit_fast/
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent


def run_stage(label: str, cmd: list, skip: bool = False, check: bool = True) -> int:
    """Run a stage; return exit code.  Exits process if check=True and rc != 0."""
    if skip:
        print(f"\n[SKIP] {label}")
        return 0
    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"{'='*72}")
    print(f"  CMD: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run([sys.executable] + [str(c) for c in cmd])
    rc = result.returncode
    if rc != 0:
        print(f"\nERROR: {label} failed (exit {rc})")
        if check:
            sys.exit(rc)
    return rc


def load_json_safe(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",        required=True,
                    help="Student source dir (scale_200k_seed42, contains states/datastore.pt)")
    ap.add_argument("--datastore_dir", default=None,
                    help="Student KMeans/offline-states dir (outputs_track_a_offline_paper/seed42). "
                         "Defaults to --source if not set. Do NOT use the gpt2_medium path here.")
    ap.add_argument("--oracle_dir",    required=True)
    ap.add_argument("--bandit_dir",    default=None)
    ap.add_argument("--bandit_rollout_dir", default=None,
                    help="Bandit rollout dir for immediate reward join (Stage 09)")
    ap.add_argument("--output",        required=True,
                    help="Base output dir (outputs_mvp4c2_repaired_credit_audit_fast)")
    ap.add_argument("--device",        default="cuda")
    ap.add_argument("--seed",          type=int, default=42)
    ap.add_argument("--max_queries",   type=int, default=5000)
    ap.add_argument("--max_defer_shadows", type=int, default=500)
    ap.add_argument("--strict_temporal", action="store_true",
                    help="Abort if blocking gate fails (default: always abort on gate fail)")
    ap.add_argument("--force",         action="store_true")

    # Per-stage skip flags
    for tag in ["01", "02", "03", "04", "05", "06", "07", "08", "09",
                "10", "11", "12", "13"]:
        ap.add_argument(f"--skip_stage_{tag}", action="store_true")
    args = ap.parse_args()

    out          = Path(args.output)
    oracle       = Path(args.oracle_dir)
    bandit       = Path(args.bandit_dir) if args.bandit_dir else oracle
    bandit_ro    = Path(args.bandit_rollout_dir) if args.bandit_rollout_dir else bandit
    source       = Path(args.source)
    # datastore_dir = student offline-states dir, NOT the gpt2_medium teacher dir
    ds_dir       = Path(args.datastore_dir) if args.datastore_dir else source
    force        = ["--force"] if args.force else []
    out.mkdir(parents=True, exist_ok=True)

    skip = lambda tag: getattr(args, f"skip_stage_{tag}")

    # ── Stage 01: Source compatibility ───────────────────────────────────────
    run_stage(
        "01 — source compatibility (student datastore vs provenance events)",
        [SCRIPT_DIR / "inspect_source_compatibility.py",
         "--source",  source,
         "--output",  out] + force,
        skip=skip("01"),
        check=True,   # blocking: wrong file invalidates all h lookups
    )

    # ── Stage 02a/b: Write provenance ────────────────────────────────────────
    for traj, traj_dir in [("oracle", oracle), ("bandit", bandit)]:
        run_stage(
            f"02/{traj} — write provenance (+ DEFER)",
            [SCRIPT_DIR / "build_write_object_provenance.py",
             "--trajectory",    traj,
             "--oracle_dir",    oracle,
             "--bandit_dir",    bandit,
             "--source",        source,
             "--datastore_dir", ds_dir,
             "--output",        out,
             "--device",        args.device] + force,
            skip=skip("02"),
        )

    # ── Stage 03a/b: True online chronological replay (FIXED bugs) ───────────
    for traj in ("oracle", "bandit"):
        run_stage(
            f"03/{traj} — true online chronological replay (repaired)",
            [SCRIPT_DIR / "replay_online_memory_chronologically.py",
             "--trajectory",    traj,
             "--source",        source,
             "--oracle_dir",    oracle,
             "--bandit_dir",    bandit,
             "--output",        out,
             "--device",        args.device,
             "--seed",          args.seed] + force,
            skip=skip("03"),
        )

    # ── Stage 04a/b: Retrospective read usage ────────────────────────────────
    for traj in ("oracle", "bandit"):
        run_stage(
            f"04/{traj} — retrospective read usage",
            [SCRIPT_DIR / "log_read_object_usage.py",
             "--trajectory",    traj,
             "--oracle_dir",    oracle,
             "--bandit_dir",    bandit,
             "--source",        source,
             "--datastore_dir", ds_dir,
             "--output",        out,
             "--device",        args.device,
             "--max_queries",   args.max_queries,
             "--seed",          args.seed] + force,
            skip=skip("04"),
        )

    # ── Stage 05a/b: Counterfactual utility ──────────────────────────────────
    for traj in ("oracle", "bandit"):
        run_stage(
            f"05/{traj} — retrospective counterfactual utility",
            [SCRIPT_DIR / "compute_object_counterfactual_utility.py",
             "--trajectory",    traj,
             "--output",        out,
             "--gammas",        "1.0",
             "--discount_unit", "steps_100"] + force,
            skip=skip("05"),
        )

    # ── Stage 06a/b: Normalized retrospective utility ─────────────────────────
    for traj in ("oracle", "bandit"):
        run_stage(
            f"06/{traj} — normalized retrospective utility",
            [SCRIPT_DIR / "analyze_normalized_object_utility.py",
             "--trajectory",  traj,
             "--output",      out,
             "--lambdas",     "5", "10", "20",
             "--k_list",      "1", "4", "8", "16",
             "--n_bootstrap", "200"] + force,
            skip=skip("06"),
        )

    # ── Stage 07: Rare-gem / quadrant analysis ────────────────────────────────
    run_stage(
        "07 — rare-gem and quadrant analysis (opportunity axis)",
        [SCRIPT_DIR / "analyze_rare_predictive_states.py",
         "--output",         out,
         "--trajectories",   "oracle", "bandit",
         "--lambda_primary", "10",
         "--k_primary",      "8"] + force,
        skip=skip("07"),
    )

    # ── Stage 08: DEFER opportunity cost with matched controls ────────────────
    run_stage(
        "08 — DEFER opportunity cost (matched controls A + B)",
        [SCRIPT_DIR / "analyze_defer_with_controls.py",
         "--output",    out,
         "--n_sample",  str(args.max_defer_shadows),
         "--min_sim",   "0.70",
         "--seed",      str(args.seed)] + force,
        skip=skip("08"),
    )

    # ── Stage 09: Immediate reward join (non-blocking) ────────────────────────
    run_stage(
        "09 — immediate reward join (bandit, non-blocking)",
        [SCRIPT_DIR / "join_bandit_immediate_rewards.py",
         "--bandit_dir",    bandit_ro,
         "--output",        out] + force,
        skip=skip("09"),
        check=False,   # exits 2 on low join rate; warned but not blocking
    )

    # ── Stage 10: BLOCKING VALIDATION GATE ───────────────────────────────────
    print(f"\n{'#'*72}")
    print("  STAGE 10: BLOCKING VALIDATION GATE")
    print(f"{'#'*72}")
    rc = run_stage(
        "10 — validation gate (pre_credit phase)",
        [SCRIPT_DIR / "validate_mvp4c2_temporal_audit.py",
         "--output", out,
         "--phase",  "pre_credit"],
        skip=skip("10"),
        check=False,
    )
    if rc != 0 and not skip("10"):
        print("\n" + "#"*72)
        print("  TEMPORAL CREDIT PIPELINE BLOCKED.")
        print("  Fix all blocking validation failures before proceeding.")
        print("  See: validation/mvp4c2_audit_pre_credit.json")
        print("#"*72)
        sys.exit(1)

    # ── Stage 11: True temporal credit (only runs after gate passes) ──────────
    run_stage(
        "11 — true temporal credit (horizon curves + false negatives)",
        [SCRIPT_DIR / "compute_true_temporal_credit.py",
         "--trajectories",  "oracle", "bandit",
         "--output",        out] + force,
        skip=skip("11"),
    )

    # ── Stage 12: Multi-target utility prediction ─────────────────────────────
    run_stage(
        "12 — multi-target utility prediction",
        [SCRIPT_DIR / "predict_future_object_utility.py",
         "--output",       out,
         "--trajectories", "oracle", "bandit",
         "--targets",
         "G_raw",
         "shrunk_mean_lambda10",
         "shrunk_opportunity_lambda10",
         "--test_frac",    "0.2",
         "--seed",         str(args.seed)] + force,
        skip=skip("12"),
    )

    # ── Stage 13: Final validation (all 14 checks) ───────────────────────────
    run_stage(
        "13 — final validation (all 14 checks)",
        [SCRIPT_DIR / "validate_mvp4c2_temporal_audit.py",
         "--output", out,
         "--phase",  "final"],
        skip=skip("13"),
        check=False,   # non-blocking for synthesis; result is in the JSON
    )

    # ── Stage 14: Synthesis ───────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  Stage 14: Research Question Synthesis")
    print(f"{'='*72}")

    gate_result  = load_json_safe(out / "validation" / "mvp4c2_audit_pre_credit.json")
    final_result = load_json_safe(out / "validation" / "mvp4c2_audit_final.json")
    credit_done  = load_json_safe(out / "temporal_credit" / "temporal_credit_done.json")
    norm_summary = load_json_safe(out / "normalized_analysis" / "normalized_analysis_summary.json")
    multi_pred   = load_json_safe(out / "utility_prediction" / "multi_target_summary.json")
    defer_done   = (load_json_safe(out / "analysis" / "defer_with_controls_summary.json") or
                    load_json_safe(out / "analysis" / "defer_shadow_done.json"))

    gate_pass     = gate_result.get("gate_pass", False)
    final_valid   = final_result.get("gate_pass", False)
    traj_credit   = credit_done.get("trajectories", {})

    def _credit(traj, key, default="INCONCLUSIVE"):
        v = traj_credit.get(traj, {}).get(key)
        if v is None or (isinstance(v, float) and v != v):
            return "INCONCLUSIVE"
        return v

    print(f"\n  Gate pass (pre_credit):  {gate_pass}")
    print(f"  Final audit valid:       {final_valid}")

    # Report blocking check status
    gate_checks = gate_result.get("checks", {})
    for cid in ["C01_source_compat", "C03_rbw_ordering", "C06_state_reconstruction",
                "C07_replay_coverage", "C09_defer_present"]:
        c = gate_checks.get(cid, {})
        status = "PASS" if c.get("pass") is True else ("SKIP" if c.get("pass") is None else "FAIL")
        print(f"    {cid}: {status}")

    if not gate_pass:
        verdict = "TEMPORAL_AUDIT_INVALID_FIX_PIPELINE"
        print(f"\n  OVERALL VERDICT: {verdict}")
    else:
        print("\n  === Research Questions ===")
        for traj in ("oracle", "bandit"):
            q4  = _credit(traj, "q4_answer")
            q7  = _credit(traj, "q7_immediate_vs_future")
            q8  = _credit(traj, "q8_false_negative_rate")
            p50 = _credit(traj, "q4_create_ttp_p50")
            print(f"\n  [{traj}]")
            print(f"    Q4 (future READ reward appears hundreds of steps later): {q4}")
            print(f"       CREATE ttp_p50 = {p50} val steps")
            print(f"    Q7 (immediate reward poorly predicts long-horizon value): {q7}")
            print(f"    Q8 (false negative rate — immediate rejects future top-Q): {q8}")

        q_verdicts = norm_summary.get("q_verdicts", {})
        for q, v in q_verdicts.items():
            print(f"    {q}: {v}")

        q4_oracle  = _credit("oracle", "q4_answer")
        q7_oracle  = _credit("oracle", "q7_immediate_vs_future")
        q4_bandit  = _credit("bandit", "q4_answer")
        deferred_bottleneck = (
            defer_done.get("q4_defer_misses_utility", {}).get("answer", "NO") != "NO"
        )

        if (q4_oracle in ("YES", "WEAK") and
                q7_oracle in ("YES", "PARTIAL") and
                q4_bandit in ("YES", "WEAK")):
            verdict = "DELAYED_CREDIT_CONFIRMED_COMPARE_MC_TD_RUDDER"
        elif deferred_bottleneck:
            verdict = "DEFER_OPPORTUNITY_COST_IS_PRIMARY_BOTTLENECK"
        elif q7_oracle == "NO":
            verdict = "IMMEDIATE_REWARD_MOSTLY_TRACKS_LONG_TERM_VALUE"
        else:
            verdict = "INCONCLUSIVE"

        print(f"\n  OVERALL VERDICT: {verdict}")

    final_summary = {
        "mvp_version":    "4c-2",
        "gate_pass":      gate_pass,
        "final_valid":    final_valid,
        "overall_verdict": verdict if gate_pass else "TEMPORAL_AUDIT_INVALID_FIX_PIPELINE",
        "gate_result":    gate_result,
        "final_result":   final_result,
        "temporal_credit": traj_credit,
        "q_verdicts_normalized": norm_summary.get("q_verdicts", {}),
        "multi_target_prediction": multi_pred,
    }
    out_path = out / "mvp4c2_final_summary.json"
    with open(out_path, "w") as f:
        json.dump(final_summary, f, indent=2, default=str)
    print(f"\nSaved → {out_path}")

    # Generate repair report
    run_stage(
        "14b — repair report",
        [SCRIPT_DIR / "write_repair_report.py", "--output", out],
        check=False,
    )

    print("\nMVP 4c-2 COMPLETE")


if __name__ == "__main__":
    main()

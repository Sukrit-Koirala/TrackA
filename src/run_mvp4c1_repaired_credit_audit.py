"""
run_mvp4c1_repaired_credit_audit.py  --  MVP 4c-1 Orchestrator (True Replay)

True Online Chronological Replay + Temporal Credit Audit.

Pipeline:
  Stage 1:  build_write_object_provenance       (oracle + bandit, WITH DEFER logging)
  Stage 2:  log_read_object_usage               (oracle + bandit, retrospective only)
  Stage 3:  compute_object_counterfactual_utility  (oracle + bandit)
  Stage 4:  replay_online_memory_chronologically  (true online replay, student h vectors)
  Stage 5:  compute_true_temporal_credit          (horizon curves, Tables A/B/C)
  Stage 5b: join_bandit_immediate_rewards         (immediate reward join ≥ 99%)
  Stage 6:  analyze_normalized_object_utility     (oracle + bandit, retrospective)
  Stage 7:  analyze_rare_predictive_states        (primary axis = opportunity@8)
  Stage 8:  analyze_defer_with_controls           (DEFER + matched controls A+B)
  Stage 9:  predict_future_object_utility         (multi-target)
  Stage 10: validate_mvp4c1_true_replay           (11 checks per spec)
  Stage 11: synthesis / report

Output: outputs_mvp4c1_repaired_credit_audit_fast/
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent


def run_stage(label: str, cmd: list, skip: bool = False, check: bool = True) -> bool:
    if skip:
        print(f"\n[SKIP] {label}")
        return True
    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"{'='*72}")
    print(f"  CMD: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run([sys.executable] + [str(c) for c in cmd])
    if result.returncode != 0:
        print(f"\nERROR: {label} failed (exit {result.returncode})")
        if check:
            sys.exit(result.returncode)
        return False
    return True


def load_json_safe(path: Path) -> dict:
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _nan_to_inconclusive(val, threshold_yes=0.3):
    """Convert NaN or zero-n correlation to INCONCLUSIVE verdict."""
    if val is None or (isinstance(val, float) and (val != val)):
        return "INCONCLUSIVE"
    return val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",        required=True)
    ap.add_argument("--datastore_dir", required=True)
    ap.add_argument("--oracle_dir",    required=True)
    ap.add_argument("--bandit_dir",    default=None)
    ap.add_argument("--output",        required=True)
    ap.add_argument("--device",        default="cuda")
    ap.add_argument("--seed",          type=int, default=42)
    ap.add_argument("--max_queries",   type=int, default=5000,
                    help="Held-out retrospective queries (Stage 2)")
    ap.add_argument("--max_temporal",  type=int, default=5000,
                    help="Chronological val steps for temporal credit (Stage 4)")
    ap.add_argument("--bandit_rollout_dir", default=None,
                    help="Bandit ROLLOUT dir (outputs_mvp4b*) for immediate reward join (Stage 5b)")
    # Skip flags
    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        ap.add_argument(f"--skip_stage_{i}", action="store_true")
    ap.add_argument("--skip_stage_5b", action="store_true",
                    help="Skip Stage 5b (immediate reward join)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out    = Path(args.output)
    oracle = Path(args.oracle_dir)
    bandit = Path(args.bandit_dir) if args.bandit_dir else oracle
    bandit_rollout = (Path(args.bandit_rollout_dir)
                      if args.bandit_rollout_dir else bandit)
    source = Path(args.source)
    ds_dir = Path(args.datastore_dir)
    force  = ["--force"] if args.force else []
    out.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: Write provenance (WITH DEFER logging) ───────────────────────
    for traj, traj_dir in [("oracle", oracle), ("bandit", bandit)]:
        run_stage(
            f"1/{traj} — write provenance (+ DEFER)",
            [SCRIPT_DIR / "build_write_object_provenance.py",
             "--trajectory",    traj,
             "--oracle_dir",    oracle,
             "--bandit_dir",    bandit,
             "--source",        source,
             "--datastore_dir", ds_dir,
             "--output",        out,
             "--device",        args.device] + force,
            skip=args.skip_stage_1,
        )

    # ── Stage 2: Retrospective read usage (held-out, no temporal claim) ──────
    for traj in ["oracle", "bandit"]:
        run_stage(
            f"2/{traj} — retrospective read usage",
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
            skip=args.skip_stage_2,
        )

    # ── Stage 3: Counterfactual utility (retrospective) ──────────────────────
    for traj in ["oracle", "bandit"]:
        run_stage(
            f"3/{traj} — retrospective counterfactual utility",
            [SCRIPT_DIR / "compute_object_counterfactual_utility.py",
             "--trajectory",    traj,
             "--output",        out,
             "--gammas",        "1.0",
             "--discount_unit", "steps_100"] + force,
            skip=args.skip_stage_3,
        )

    # ── Stage 4: TRUE online chronological replay (IncrementalMemory) ─────────
    # Uses student model h vectors from train.pt (768-dim), NOT teacher datastore.
    # Delay = query_stream_step - write_stream_step (same stream, always > 0).
    for traj in ["oracle", "bandit"]:
        run_stage(
            f"4/{traj} — true online chronological replay",
            [SCRIPT_DIR / "replay_online_memory_chronologically.py",
             "--trajectory",    traj,
             "--source",        source,
             "--oracle_dir",    oracle,
             "--bandit_dir",    bandit,
             "--output",        out,
             "--device",        args.device,
             "--seed",          args.seed] + force,
            skip=args.skip_stage_4,
        )

    # ── Stage 5: True temporal credit (horizon curves, Tables A/B/C) ─────────
    run_stage(
        "5 — true temporal credit (horizon curves + false negatives)",
        [SCRIPT_DIR / "compute_true_temporal_credit.py",
         "--trajectories",  "oracle", "bandit",
         "--output",        out] + force,
        skip=args.skip_stage_5,
    )

    # ── Stage 5b: Immediate reward join (required ≥ 99%) ─────────────────────
    run_stage(
        "5b — immediate reward join (bandit decisions)",
        [SCRIPT_DIR / "join_bandit_immediate_rewards.py",
         "--bandit_dir",    bandit_rollout,
         "--output",        out] + force,
        skip=args.skip_stage_5b,
        check=False,   # non-blocking: exits 2 on low join rate, not 1
    )

    # ── Stage 6: Normalized retrospective utility (labeled RETROSPECTIVE) ─────
    for traj in ["oracle", "bandit"]:
        run_stage(
            f"6/{traj} — normalized retrospective object utility",
            [SCRIPT_DIR / "analyze_normalized_object_utility.py",
             "--trajectory",  traj,
             "--output",      out,
             "--lambdas",     "5", "10", "20",
             "--k_list",      "1", "4", "8", "16",
             "--n_bootstrap", "200"] + force,
            skip=args.skip_stage_6,
        )

    # ── Stage 7: Rare-gem / quadrant analysis ────────────────────────────────
    run_stage(
        "7 — rare-gem and quadrant analysis",
        [SCRIPT_DIR / "analyze_rare_predictive_states.py",
         "--output",         out,
         "--trajectories",   "oracle", "bandit",
         "--lambda_primary", "10",
         "--k_primary",      "8"] + force,
        skip=args.skip_stage_7,
    )

    # ── Stage 8: DEFER opportunity cost WITH matched controls ────────────────
    run_stage(
        "8 — DEFER opportunity cost (matched controls A + B)",
        [SCRIPT_DIR / "analyze_defer_with_controls.py",
         "--output",    out,
         "--n_sample",  "500",
         "--min_sim",   "0.70",
         "--seed",      args.seed] + force,
        skip=args.skip_stage_8,
    )

    # ── Stage 9: Multi-target utility prediction ──────────────────────────────
    run_stage(
        "9 — multi-target utility prediction",
        [SCRIPT_DIR / "predict_future_object_utility.py",
         "--output",       out,
         "--trajectories", "oracle", "bandit",
         "--targets",
         "G_raw",
         "shrunk_mean_lambda10",
         "shrunk_opportunity_lambda10",
         "--test_frac",    "0.2",
         "--seed",         args.seed] + force,
        skip=args.skip_stage_9,
    )

    # ── Stage 10: True replay validation (11 checks per spec) ────────────────
    run_stage(
        "10 — true replay validation (spec 11-check table)",
        [SCRIPT_DIR / "validate_mvp4c1_true_replay.py",
         "--output",  out] + force,
        skip=args.skip_stage_10,
        check=False,   # validation failure exits 1 but is non-blocking for synthesis
    )

    # ── Stage 11: Synthesis ──────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  Stage 11: Research Question Synthesis")
    print(f"{'='*72}")

    # Prefer true replay validation; fall back to old audit summary
    val_summary    = (load_json_safe(out / "validation" / "mvp4c1_true_replay_validation.json") or
                      load_json_safe(out / "validation" / "audit_summary.json"))
    credit_done    = load_json_safe(out / "temporal_credit" / "temporal_credit_done.json")
    norm_summary   = load_json_safe(out / "normalized_analysis" / "normalized_analysis_summary.json")
    multi_pred     = load_json_safe(out / "utility_prediction" / "multi_target_summary.json")
    defer_done     = (load_json_safe(out / "analysis" / "defer_with_controls_summary.json") or
                      load_json_safe(out / "analysis" / "defer_shadow_done.json"))

    audit_valid    = val_summary.get("overall_verdict") == "VALID"
    traj_credit    = credit_done.get("trajectories", {})

    def _credit(traj, key, default="INCONCLUSIVE"):
        v = traj_credit.get(traj, {}).get(key)
        return _nan_to_inconclusive(v) if v is None else v

    def _val_check(key_new: str, key_old: str) -> str:
        """Read check result from either new or old validation format."""
        checks = val_summary.get("checks", {})
        if checks and key_new in checks:
            return "PASS" if checks[key_new].get("pass") else "FAIL"
        v = val_summary.get(key_old)
        if v is True:   return "PASS"
        if v is False:  return "FAIL"
        return str(v) if v is not None else "UNKNOWN"

    print("\n  === Audit Validity ===")
    print(f"  Overall audit valid: {audit_valid}")
    for new_key, old_key, label in [
        ("c2_rbw_ordering",  "check1_no_neg_delay",  "Check 1/2 (no neg delay / RBW)"),
        ("c5_no_pre_create_use", "check2_no_pre_create", "Check 5 (no pre-create use)"),
        ("c9_defer_present", "check4_defer_present",  "Check 9 (DEFER present)"),
        ("c8_imm_reward_join","check5_join_reported", "Check 8 (immediate reward join)"),
        ("c10_outputs_sep",  "check10_separated",     "Check 10 (outputs separated)"),
    ]:
        print(f"    {label}: {_val_check(new_key, old_key)}")

    if not audit_valid:
        print("\n  *** OVERALL VERDICT: INVALID_AUDIT ***")
        print("  Blocking validation checks failed. Do not interpret temporal credit results.")
    else:
        print("\n  === Research Questions ===")

        for traj in ["oracle", "bandit"]:
            print(f"\n  [{traj}]")
            q4  = _credit(traj, "q4_answer")
            q7  = _credit(traj, "q7_immediate_vs_future")
            q8  = _credit(traj, "q8_false_negative_rate")
            q4_p50 = _credit(traj, "q4_create_ttp_p50")
            print(f"    Q4 (future READ reward appears hundreds of steps later): {q4}")
            print(f"       CREATE ttp_p50 = {q4_p50} val steps")
            print(f"    Q7 (immediate reward poorly predicts long-horizon value): {q7}")
            print(f"    Q8 (false negative rate — immediate rejects future top-Q): {q8}")

        q_verdicts = norm_summary.get("q_verdicts", {})
        for q, v in q_verdicts.items():
            print(f"    {q}: {v}")

        # Overall decision
        q4_oracle = _credit("oracle", "q4_answer")
        q7_oracle = _credit("oracle", "q7_immediate_vs_future")
        q4_bandit = _credit("bandit", "q4_answer")
        q7_bandit = _credit("bandit", "q7_immediate_vs_future")

        delayed_confirmed = (
            q4_oracle in ("YES", "WEAK") and
            q7_oracle in ("YES", "PARTIAL") and
            q4_bandit in ("YES", "WEAK")
        )
        deferred_bottleneck = defer_done.get("q4_defer_misses_utility", {}).get("answer", "NO") != "NO"

        if not audit_valid:
            verdict = "TEMPORAL_AUDIT_INVALID_FIX_PIPELINE"
        elif delayed_confirmed:
            verdict = "DELAYED_CREDIT_CONFIRMED_COMPARE_MC_TD_RUDDER"
        elif deferred_bottleneck:
            verdict = "DEFER_OPPORTUNITY_COST_IS_PRIMARY_BOTTLENECK"
        elif q7_oracle in ("NO",):
            verdict = "IMMEDIATE_REWARD_MOSTLY_TRACKS_LONG_TERM_VALUE"
        else:
            verdict = "INCONCLUSIVE"

        print(f"\n  OVERALL VERDICT: {verdict}")

    # Save final summary
    final = {
        "audit_valid":    audit_valid,
        "overall_verdict": "INVALID_AUDIT" if not audit_valid else verdict,
        "validation":     val_summary,
        "temporal_credit": traj_credit,
        "q_verdicts_normalized": norm_summary.get("q_verdicts", {}),
        "multi_target_prediction": multi_pred,
    }
    with open(out / "mvp4c1_final_summary.json", "w") as f:
        json.dump(final, f, indent=2, default=str)
    print(f"\nSaved: {out / 'mvp4c1_final_summary.json'}")
    print("\nMVP 4c-1 COMPLETE")


if __name__ == "__main__":
    main()

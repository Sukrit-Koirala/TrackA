"""
run_mvp4c0_delayed_credit_audit.py  --  MVP 4c-0 Full Orchestrator

Runs the complete delayed credit audit pipeline in 9 stages:

  Stage 1: build_write_object_provenance     (oracle + bandit)
  Stage 2: log_read_object_usage             (oracle + bandit, fixed + q-read)
  Stage 3: compute_object_counterfactual_utility  (oracle + bandit)
  Stage 4: analyze_object_delayed_returns
  Stage 5: analyze_write_credit_horizons
  Stage 6: analyze_defer_shadow_objects
  Stage 7: predict_future_object_utility     (G_raw target only, for Q7)
  Stage 8: (synthesis / report) -- prints Q1-Q7 verdicts from saved JSONs

Frequency-normalization patch (Stages 9-11):
  Stage 9:  analyze_normalized_object_utility   (oracle + bandit)
  Stage 10: analyze_rare_predictive_states
  Stage 11: predict_future_object_utility       (multi-target: shrunk_mean, opportunity, unique)

Each stage can be skipped with --skip_stage_N.

Usage:
  python src/run_mvp4c0_delayed_credit_audit.py \\
    --source              /path/to/scale_200k_seed42 \\
    --oracle_dir          /path/to/outputs_mvp4a0_oracle_reconstruction \\
    --bandit_dir          /path/to/outputs_mvp4b_bandit_write \\
    --output              /path/to/outputs_mvp4c0_delayed_credit_audit_fast \\
    --device              cuda \\
    --seed                42 \\
    [--skip_stage_1] ... [--skip_stage_11]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent


def run_stage(label: str, cmd: list, skip: bool = False, check: bool = True) -> bool:
    if skip:
        print(f"\n[SKIP] Stage {label}")
        return True
    print(f"\n{'='*70}")
    print(f"  Stage {label}")
    print(f"{'='*70}")
    print(f"  CMD: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run([sys.executable] + [str(c) for c in cmd])
    if result.returncode != 0:
        print(f"\nERROR: Stage {label} failed (exit {result.returncode})")
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


def print_verdict(q: str, result: dict | str | None):
    if result is None or result == {}:
        print(f"  {q}: MISSING")
        return
    if isinstance(result, str):
        print(f"  {q}: {result}")
    elif isinstance(result, dict):
        ans = result.get("answer") or result.get("q4_defer_misses_utility", {}).get("answer", "?")
        print(f"  {q}: {ans}  |  {json.dumps(result, default=str)[:120]}")
    else:
        print(f"  {q}: {result}")


def main():
    ap = argparse.ArgumentParser()
    # Path args
    ap.add_argument("--source",        required=True,
                    help="Source data dir with train.pt / val.pt")
    ap.add_argument("--datastore_dir", required=True,
                    help="Dir containing states/datastore.pt (precomputed GPT-2 embeddings)")
    ap.add_argument("--oracle_dir",    required=True,
                    help="Oracle reconstruction output dir")
    ap.add_argument("--bandit_dir",    default=None,
                    help="Bandit rollout output dir (defaults to --oracle_dir)")
    ap.add_argument("--output",        required=True,
                    help="Output dir for all MVP 4c-0 artifacts")
    # Runtime
    ap.add_argument("--device",      default="cuda")
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--max_queries", type=int, default=5000,
                    help="Max val queries to process in Stage 2")
    # Skip flags
    ap.add_argument("--skip_stage_1", action="store_true")
    ap.add_argument("--skip_stage_2", action="store_true")
    ap.add_argument("--skip_stage_3", action="store_true")
    ap.add_argument("--skip_stage_4", action="store_true")
    ap.add_argument("--skip_stage_5", action="store_true")
    ap.add_argument("--skip_stage_6", action="store_true")
    ap.add_argument("--skip_stage_7",  action="store_true")
    ap.add_argument("--skip_stage_9",  action="store_true")
    ap.add_argument("--skip_stage_10", action="store_true")
    ap.add_argument("--skip_stage_11", action="store_true")
    # Forwarder flags
    ap.add_argument("--force",        action="store_true",
                    help="Pass --force to all stages")
    args = ap.parse_args()

    out      = Path(args.output)
    oracle   = Path(args.oracle_dir)
    bandit   = Path(args.bandit_dir) if args.bandit_dir else oracle
    source   = Path(args.source)
    ds_dir   = Path(args.datastore_dir)
    force    = ["--force"] if args.force else []

    out.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: provenance ──────────────────────────────────────────────────
    for traj, traj_dir in [("oracle", oracle), ("bandit", bandit)]:
        run_stage(
            f"1/{traj} (write provenance)",
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

    # ── Stage 2: read object usage ───────────────────────────────────────────
    for traj in ["oracle", "bandit"]:
        run_stage(
            f"2/{traj} (log read usage)",
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

    # ── Stage 3: counterfactual utility ──────────────────────────────────────
    for traj in ["oracle", "bandit"]:
        run_stage(
            f"3/{traj} (compute utility)",
            [SCRIPT_DIR / "compute_object_counterfactual_utility.py",
             "--trajectory",     traj,
             "--output",         out,
             "--gammas",         "1.0", "0.99", "0.95",
             "--discount_unit",  "steps_100"] + force,
            skip=args.skip_stage_3,
        )

    # ── Stage 4: delayed return analysis ─────────────────────────────────────
    run_stage(
        "4 (delayed returns analysis)",
        [SCRIPT_DIR / "analyze_object_delayed_returns.py",
         "--output",        out,
         "--trajectories",  "oracle", "bandit"] + force,
        skip=args.skip_stage_4,
    )

    # ── Stage 5: credit horizon analysis ─────────────────────────────────────
    run_stage(
        "5 (credit horizon analysis)",
        [SCRIPT_DIR / "analyze_write_credit_horizons.py",
         "--output",           out,
         "--trajectories",     "oracle", "bandit",
         "--immediate_window", "200",
         "--future_start",     "200"] + force,
        skip=args.skip_stage_5,
    )

    # ── Stage 6: defer shadow objects ────────────────────────────────────────
    run_stage(
        "6 (defer shadow objects)",
        [SCRIPT_DIR / "analyze_defer_shadow_objects.py",
         "--output",    out,
         "--n_sample",  "500",
         "--min_sim",   "0.70",
         "--seed",      args.seed] + force,
        skip=args.skip_stage_6,
    )

    # ── Stage 7: predictability (G_raw only, fast baseline) ──────────────────
    run_stage(
        "7 (predict future utility — G_raw)",
        [SCRIPT_DIR / "predict_future_object_utility.py",
         "--output",       out,
         "--trajectories", "oracle", "bandit",
         "--targets",      "G_raw",
         "--test_frac",    "0.2",
         "--seed",         args.seed] + force,
        skip=args.skip_stage_7,
    )

    # ── Stage 8: synthesis ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  Stage 8: Research Question Verdicts")
    print(f"{'='*70}")

    ana_dir  = out / "analysis"
    pred_dir = out / "predictors"

    dr = load_json_safe(ana_dir / "delayed_returns_done.json")
    ch = load_json_safe(ana_dir / "credit_horizons_done.json")
    ds = load_json_safe(ana_dir / "defer_shadow_done.json")
    pr = load_json_safe(pred_dir / "predictor_results.json")

    q1 = dr.get("q1_measurable_utility", {})
    q3 = ch.get("q3_delayed_credit", {})
    q4 = ds.get("q4_defer_misses_utility", {})
    q7_rho    = pr.get("best_spearman", "?")
    q7_answer = pr.get("q7_predictability", "?")

    # Q2: median delay
    delay_rows = dr.get("delay_distributions", [])
    q2_delays  = {r["trajectory"]: r.get("p50", "?")
                  for r in delay_rows if r.get("delay_type") == "create_to_first_pos"}

    print("\n  Research Questions:")
    print(f"  Q1 (measurable utility):")
    for traj, res in q1.items():
        ans = res.get("answer", str(res)) if isinstance(res, dict) else str(res)
        print(f"     [{traj}] {ans}")

    print(f"  Q2 (delay to first positive READ / p50 steps):")
    for traj, delay in q2_delays.items():
        print(f"     [{traj}] {delay}")

    print(f"  Q3 (significant delayed credit):")
    for traj, res in q3.items():
        ans = res.get("significant_delayed_credit", "?")
        pct = res.get("pct_future", "?")
        print(f"     [{traj}] {ans}  ({pct:.1f}% credit is future)" if isinstance(pct, float) else f"     [{traj}] {ans}")

    print(f"  Q4 (DEFER misses utility):")
    ans4 = q4.get("answer", "?")
    print(f"     {ans4}  |  mean proxy G_raw={q4.get('mean_proxy_G_raw', '?')}")

    print(f"  Q7 (write-time features predict utility):")
    print(f"     {q7_answer}  (best rho={q7_rho})")

    # Overall verdict
    has_signal   = any(v.get("answer") == "YES" for v in q1.values() if isinstance(v, dict))
    has_delay    = any(v.get("significant_delayed_credit") for v in q3.values() if isinstance(v, dict))
    defer_misses = q4.get("answer", "NO") not in ("NO", "INCONCLUSIVE")
    predictable  = q7_answer in ("STRONG", "MODERATE")

    if has_signal and has_delay and predictable:
        verdict = "PROCEED_TO_MC_OR_TD"
    elif has_signal and (has_delay or predictable):
        verdict = "PROCEED_WITH_CAUTION"
    elif has_signal:
        verdict = "SIGNAL_EXISTS_NO_DELAY"
    else:
        verdict = "INSUFFICIENT_SIGNAL"

    print(f"\n  OVERALL VERDICT: {verdict}")
    if verdict in ("PROCEED_TO_MC_OR_TD", "PROCEED_WITH_CAUTION"):
        print("  → Delayed credit is measurable. Proceed to Monte Carlo or TD learning.")
    elif verdict == "SIGNAL_EXISTS_NO_DELAY":
        print("  → Signal exists but credit is immediate. Bandit with immediate reward may suffice.")
    else:
        print("  → Insufficient signal. Revisit READ method or object selection before implementing RL.")

    # Save Stage 8 summary
    final = {
        "verdict":   verdict,
        "q1":        q1,
        "q2_delays": q2_delays,
        "q3":        q3,
        "q4":        q4,
        "q7_rho":    q7_rho,
        "q7":        q7_answer,
    }
    with open(out / "mvp4c0_final_summary.json", "w") as f:
        json.dump(final, f, indent=2, default=str)
    print(f"\n  Saved: {out / 'mvp4c0_final_summary.json'}")

    # ── Stage 9: normalized utility vectors ──────────────────────────────────
    for traj in ["oracle", "bandit"]:
        run_stage(
            f"9/{traj} (normalized object utility)",
            [SCRIPT_DIR / "analyze_normalized_object_utility.py",
             "--trajectory",  traj,
             "--output",      out,
             "--lambdas",     "5", "10", "20",
             "--k_list",      "1", "4", "8", "16",
             "--n_bootstrap", "200"] + force,
            skip=args.skip_stage_9,
        )

    # ── Stage 10: rare gem / quadrant analysis ────────────────────────────────
    run_stage(
        "10 (rare state analysis)",
        [SCRIPT_DIR / "analyze_rare_predictive_states.py",
         "--output",         out,
         "--trajectories",   "oracle", "bandit",
         "--lambda_primary", "10",
         "--k_primary",      "8"] + force,
        skip=args.skip_stage_10,
    )

    # ── Stage 11: multi-target utility prediction ─────────────────────────────
    run_stage(
        "11 (multi-target utility prediction)",
        [SCRIPT_DIR / "predict_future_object_utility.py",
         "--output",       out,
         "--trajectories", "oracle", "bandit",
         "--targets",
         "G_raw",
         "shrunk_mean_lambda10",
         "shrunk_opportunity_lambda10",
         "unique_gain_after_replacement",
         "--test_frac",    "0.2",
         "--seed",         args.seed] + force,
        skip=args.skip_stage_11,
    )

    # ── Final extended summary ────────────────────────────────────────────────
    norm_summary = load_json_safe(out / "normalized_analysis" / "normalized_analysis_summary.json")
    multi_pred   = load_json_safe(out / "utility_prediction"  / "multi_target_summary.json")

    q11_18 = norm_summary.get("q_verdicts", {})
    q17    = multi_pred.get("best_target", "?")
    q17_rho = multi_pred.get("best_spearman", "?")

    print(f"\n{'='*70}")
    print("  Stage 11-12: Frequency-Normalized Audit Verdicts")
    print(f"{'='*70}")
    for q, v in q11_18.items():
        print(f"  {q}: {v}")
    if q17 != "?":
        print(f"\n  Q17 (most predictable utility): '{q17}'  (Spearman ρ={q17_rho})")

    final["q11_18"] = q11_18
    final["q17_best_target"] = q17
    final["q17_best_spearman"] = q17_rho
    with open(out / "mvp4c0_final_summary.json", "w") as f:
        json.dump(final, f, indent=2, default=str)

    print("\nMVP 4c-0 COMPLETE")


if __name__ == "__main__":
    main()

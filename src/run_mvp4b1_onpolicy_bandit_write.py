"""
run_mvp4b1_onpolicy_bandit_write.py  --  MVP 4b.1 Orchestrator

15 stages:
  Stage 1: Build reward datasets
    1a. oracle reward dataset      (seed 42 stream)
    1b. low_memory reward dataset  (seed 42 stream)
  Stage 2: Audit
    2a. audit oracle + low_memory rewards
  Stage 3: First scorer training (oracle + low_memory only)
    3a. train scorer pass 1
    3b. evaluate scorer pass 1
  Stage 4: On-policy rollout + reward collection (seed 123, variant A)
    4a. rollout seed=123 variant=A --collect_rewards
  Stage 5: Second scorer training (oracle + low_memory + onpolicy)
    5a. train scorer pass 2 (includes onpolicy)
    5b. evaluate scorer pass 2
  Stage 6: Final four rollouts (no reward collection)
    6a. rollout seed=123, variant=A
    6b. rollout seed=123, variant=B
    6c. rollout seed=999, variant=A
    6d. rollout seed=999, variant=B
  Stage 7: Evaluate + Analyze
    7a. evaluate all four states
    7b. analyze

Skip flags available for each stage.

Usage:
  python src/run_mvp4b1_onpolicy_bandit_write.py \\
    --source              scale_200k_seed42 \\
    --offline_qread_dir   outputs_track_a_offline_paper/seed42/q_read/states \\
    --oracle_dir          outputs_mvp4a0_oracle_reconstruction \\
    --output              outputs_mvp4b1_onpolicy_bandit_write_clean_fast \\
    --object_budget       10000 \\
    --scorer              mlp \\
    --n_decision_points   10000 \\
    --seed                42 \\
    --device              cuda
"""

import sys
import json
import subprocess
import time
from pathlib import Path
import argparse


def run_step(name: str, cmd: list, check: bool = True) -> int:
    print(f"\n{'='*70}")
    print(f"STEP: {name}")
    print(f"CMD:  {' '.join(str(c) for c in cmd)}")
    print(f"{'='*70}")
    t0     = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
    print(f"\n[{name}] {status}  ({elapsed:.0f}s)")
    if check and result.returncode != 0:
        print(f"ERROR: step '{name}' failed — aborting.")
        sys.exit(result.returncode)
    return result.returncode


def _load_json(p: Path) -> dict:
    try:
        return json.load(open(p)) if p.exists() else {}
    except json.JSONDecodeError:
        return {}


def print_summary(out: Path, scorer: str, budget: int) -> None:
    eval_sum = _load_json(out / "evaluation" / "eval_summary.json")
    analysis = _load_json(out / "analysis" / "analysis_complete.json")
    audit    = _load_json(out / "reward_audit" / "audit_report.json")
    sc_eval  = _load_json(out / "scorer_eval" / "scorer_eval.json")

    print("\n" + "=" * 70)
    print("MVP 4b.1 ONPOLICY BANDIT-WRITE SUMMARY")
    print("=" * 70)
    print(f"Scorer: {scorer}  Budget: {budget}")

    print("\n--- Reward audit ---")
    print(f"  verdict: {audit.get('verdict','—')}")

    print("\n--- Scorer eval ---")
    for mname, mr in sc_eval.get("models", {}).items():
        rho  = mr.get("spearman_rho", "—")
        reg  = mr.get("regret", {})
        print(f"  {mname.upper()}: rho={rho:.3f}  "
              f"regret={reg.get('mean_regret','?'):.4f}  "
              f"action_match={reg.get('action_match_frac','?'):.3f}")

    print("\n--- Q-NLL results ---")
    results = eval_sum.get("results", {})
    for tag, m in results.items():
        q = m.get("q_state_nll")
        s = f"{q:.4f}" if q else "—"
        print(f"  {tag:<45}  {s}")
    print(f"  {'GPT-only':45}  2.5533")
    print(f"  {'MVP3b':45}  2.4204")

    verdict = analysis.get("verdict", "—")
    best_q  = analysis.get("best_q_nll")
    best_tag = analysis.get("best_tag")
    print(f"\n==> MVP 4b.1 verdict: {verdict}")
    if best_tag:
        q_str = f"{best_q:.4f}" if best_q else "?"
        print(f"    Best: {best_tag}  Q-NLL={q_str}")
    rp = out / "report.md"
    if rp.exists():
        print(f"    Full report: {rp}")


def main():
    ap = argparse.ArgumentParser()
    # ── data / output ──
    ap.add_argument("--source",              required=True)
    ap.add_argument("--offline_qread_dir",   required=True)
    ap.add_argument("--oracle_dir",          required=True)
    ap.add_argument("--output",              required=True)
    # ── config ──
    ap.add_argument("--object_budget",       type=int, default=10000)
    ap.add_argument("--scorer",              default="mlp",
                    choices=["mlp", "gbm", "both"])
    ap.add_argument("--n_decision_points",   type=int, default=10000)
    ap.add_argument("--n_lm_decision_points", type=int, default=5000)
    ap.add_argument("--k_states",            type=int, default=8)
    ap.add_argument("--k_buffers",           type=int, default=8)
    ap.add_argument("--n_local_probes",      type=int, default=16)
    ap.add_argument("--n_history_probes",    type=int, default=16)
    ap.add_argument("--n_replay_probes",     type=int, default=16)
    ap.add_argument("--max_exemplars",       type=int, default=16)
    ap.add_argument("--promotion_support",   type=int, default=8)
    ap.add_argument("--max_lm_objects",      type=int, default=500)
    ap.add_argument("--episode_length_lm",   type=int, default=2000)
    ap.add_argument("--mlp_epochs",          type=int, default=60)
    ap.add_argument("--gbm_n_estimators",    type=int, default=400)
    ap.add_argument("--reward_col",          default="reward_penalized")
    ap.add_argument("--max_q_samples",       type=int, default=800_000)
    ap.add_argument("--n_epochs_q",          type=int, default=30)
    ap.add_argument("--seed",                type=int, default=42)
    ap.add_argument("--device",              default="cuda")
    ap.add_argument("--fast",                action="store_true")
    ap.add_argument("--force",               action="store_true")
    # ── skip flags ──
    ap.add_argument("--skip_build_oracle",   action="store_true")
    ap.add_argument("--skip_build_lm",       action="store_true")
    ap.add_argument("--skip_audit",          action="store_true")
    ap.add_argument("--skip_train1",         action="store_true")
    ap.add_argument("--skip_eval1",          action="store_true")
    ap.add_argument("--skip_onpolicy",       action="store_true")
    ap.add_argument("--skip_train2",         action="store_true")
    ap.add_argument("--skip_eval2",          action="store_true")
    ap.add_argument("--skip_rollout",        action="store_true")
    ap.add_argument("--skip_eval_states",    action="store_true")
    ap.add_argument("--skip_analyze",        action="store_true")
    args = ap.parse_args()

    out    = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    src    = Path(__file__).parent

    # Save config
    cfg = vars(args)
    cfg["output"] = str(out)
    with open(out / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    force_flag   = ["--force"] if args.force else []
    scorers      = ["mlp", "gbm"] if args.scorer == "both" else [args.scorer]
    primary      = scorers[0]
    model_flags  = ["--model"] + scorers

    reward_base_flags = [
        "--source",            args.source,
        "--oracle_dir",        args.oracle_dir,
        "--output",            str(out),
        "--object_budget",     str(args.object_budget),
        "--k_states",          str(args.k_states),
        "--k_buffers",         str(args.k_buffers),
        "--n_local_probes",    str(args.n_local_probes),
        "--n_history_probes",  str(args.n_history_probes),
        "--n_replay_probes",   str(args.n_replay_probes),
        "--max_exemplars",     str(args.max_exemplars),
        "--promotion_support", str(args.promotion_support),
        "--seed",              str(args.seed),
        "--device",            args.device,
    ]

    # ── 1a: oracle reward dataset ─────────────────────────────────────────────
    if not args.skip_build_oracle:
        run_step("1a. build oracle rewards",
                 [python, str(src / "build_mvp4b1_reward_dataset.py"),
                  "--trajectory_type", "oracle",
                  "--n_decision_points", str(args.n_decision_points),
                  *reward_base_flags, *force_flag])
    else:
        print("\n[SKIP] build oracle rewards")

    # ── 1b: low_memory reward dataset ────────────────────────────────────────
    if not args.skip_build_lm:
        run_step("1b. build low_memory rewards",
                 [python, str(src / "build_mvp4b1_reward_dataset.py"),
                  "--trajectory_type", "low_memory",
                  "--n_decision_points", str(args.n_lm_decision_points),
                  "--max_lm_objects",   str(args.max_lm_objects),
                  "--episode_length_lm", str(args.episode_length_lm),
                  *reward_base_flags, *force_flag])
    else:
        print("\n[SKIP] build low_memory rewards")

    # ── 2a: audit ────────────────────────────────────────────────────────────
    if not args.skip_audit:
        run_step("2a. audit rewards",
                 [python, str(src / "audit_mvp4b1_rewards.py"),
                  "--output",            str(out),
                  "--trajectory_types",  "oracle", "low_memory",
                  "--reward_col",        args.reward_col,
                  *force_flag],
                 check=False)
    else:
        print("\n[SKIP] audit rewards")

    def _train_step(label, traj_types: list):
        run_step(label,
                 [python, str(src / "train_mvp4b1_scorer.py"),
                  "--output",            str(out),
                  "--reward_col",        args.reward_col,
                  "--trajectory_types",  *traj_types,
                  *model_flags,
                  "--mlp_epochs",        str(args.mlp_epochs),
                  "--gbm_n_estimators",  str(args.gbm_n_estimators),
                  "--seed",              str(args.seed),
                  "--device",            args.device,
                  *force_flag])

    def _eval_scorer_step(label, traj_types: list):
        run_step(label,
                 [python, str(src / "evaluate_mvp4b1_scorer.py"),
                  "--output",            str(out),
                  "--reward_col",        args.reward_col,
                  "--trajectory_types",  *traj_types,
                  *model_flags,
                  "--device",            args.device,
                  *force_flag])

    # ── 3a: train pass 1 ─────────────────────────────────────────────────────
    if not args.skip_train1:
        _train_step("3a. train scorer pass 1", ["oracle", "low_memory"])
    else:
        print("\n[SKIP] train scorer pass 1")

    # ── 3b: eval scorer pass 1 ───────────────────────────────────────────────
    if not args.skip_eval1:
        _eval_scorer_step("3b. eval scorer pass 1", ["oracle", "low_memory"])
    else:
        print("\n[SKIP] eval scorer pass 1")

    # ── 4a: on-policy rollout + collect rewards (seed 123, variant A) ─────────
    if not args.skip_onpolicy:
        run_step("4a. on-policy rollout + collect rewards",
                 [python, str(src / "rollout_mvp4b1.py"),
                  "--source",             args.source,
                  "--output",             str(out),
                  "--object_budget",      str(args.object_budget),
                  "--bootstrap_variant",  "A",
                  "--bootstrap_min",      "50",
                  "--seed",              "123",
                  "--scorer",             primary,
                  "--k_states",           str(args.k_states),
                  "--k_buffers",          str(args.k_buffers),
                  "--max_exemplars",      str(args.max_exemplars),
                  "--promotion_support",  str(args.promotion_support),
                  "--collect_rewards",
                  "--device",             args.device,
                  *force_flag])
    else:
        print("\n[SKIP] on-policy rollout")

    # ── 5a: train pass 2 (with onpolicy) ─────────────────────────────────────
    if not args.skip_train2:
        _train_step("5a. train scorer pass 2", ["oracle", "low_memory", "onpolicy"])
    else:
        print("\n[SKIP] train scorer pass 2")

    # ── 5b: eval scorer pass 2 ───────────────────────────────────────────────
    if not args.skip_eval2:
        _eval_scorer_step("5b. eval scorer pass 2",
                          ["oracle", "low_memory", "onpolicy"])
    else:
        print("\n[SKIP] eval scorer pass 2")

    # ── 6a-6d: final four rollouts ────────────────────────────────────────────
    rollout_cfgs = [
        ("A", 123), ("B", 123), ("A", 999), ("B", 999)
    ]
    for i, (var, rseed) in enumerate(rollout_cfgs, start=1):
        step_label = f"6{chr(96+i)}. rollout variant={var} seed={rseed}"
        if not args.skip_rollout:
            run_step(step_label,
                     [python, str(src / "rollout_mvp4b1.py"),
                      "--source",            args.source,
                      "--output",            str(out),
                      "--object_budget",     str(args.object_budget),
                      "--bootstrap_variant", var,
                      "--bootstrap_min",     "50",
                      "--seed",              str(rseed),
                      "--scorer",            primary,
                      "--k_states",          str(args.k_states),
                      "--k_buffers",         str(args.k_buffers),
                      "--max_exemplars",     str(args.max_exemplars),
                      "--promotion_support", str(args.promotion_support),
                      "--device",            args.device,
                      *force_flag])
        else:
            print(f"\n[SKIP] {step_label}")

    # ── 7a: evaluate all states ───────────────────────────────────────────────
    if not args.skip_eval_states:
        q_flags = [
            "--max_q_samples", str(args.max_q_samples),
            "--n_epochs",      str(args.n_epochs_q),
        ]
        if args.fast:
            q_flags = ["--max_q_samples", "100000", "--n_epochs", "10", "--fast"]
        run_step("7a. evaluate all rollout states",
                 [python, str(src / "evaluate_mvp4b1_states.py"),
                  "--source",            args.source,
                  "--offline_qread_dir", args.offline_qread_dir,
                  "--output",            str(out),
                  "--scorer",            primary,
                  "--object_budget",     str(args.object_budget),
                  "--seed",              str(args.seed),
                  "--device",            args.device,
                  *q_flags,
                  *force_flag])
    else:
        print("\n[SKIP] evaluate states")

    # ── 7b: analyze ──────────────────────────────────────────────────────────
    if not args.skip_analyze:
        run_step("7b. analyze",
                 [python, str(src / "analyze_mvp4b1.py"),
                  "--output",        str(out),
                  "--oracle_dir",    args.oracle_dir,
                  "--scorer",        primary,
                  "--object_budget", str(args.object_budget),
                  *force_flag])
    else:
        print("\n[SKIP] analyze")

    print_summary(out, primary, args.object_budget)


if __name__ == "__main__":
    main()

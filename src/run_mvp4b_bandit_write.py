"""
run_mvp4b_bandit_write.py  --  MVP 4b: Teacher-Free Predictive-Utility WRITE

Orchestrates all stages:
  Stage 1:
    1a. build_bandit_write_reward_dataset.py  (counterfactual rewards)
    1b. audit_bandit_write_rewards.py         (validation + go/no-go)
  Stage 2:
    2a. train_bandit_write_scorer.py          (MLP + GBM)
    2b. evaluate_bandit_write_scorer.py       (regression + regret metrics)
  Stage 3:
    3a. rollout_bandit_write.py               (teacher-free online rollout)
    3b. evaluate_bandit_write_states.py       (Q-read evaluation)
    3c. analyze_bandit_write_rollout.py       (analysis + report)

Usage:
  python src/run_mvp4b_bandit_write.py \\
    --source              scale_200k_seed42 \\
    --offline_qread_dir   outputs_track_a_offline_paper/seed42/q_read/states \\
    --oracle_dir          outputs_mvp4a0_oracle_reconstruction \\
    --output              outputs_mvp4b_bandit_write_fast \\
    --budget              10000 \\
    --scorer              mlp \\
    --n_decision_points   10000 \\
    --promotion_support   8 \\
    --seed                42 \\
    --device              cuda

Skip flags:
  --skip_build_rewards
  --skip_audit
  --skip_train
  --skip_eval_scorer
  --skip_rollout
  --skip_eval_states
  --skip_analyze
"""

import sys
import json
import subprocess
import time
from pathlib import Path
import argparse


def run_step(name: str, cmd: list[str], check: bool = True) -> int:
    print(f"\n{'='*70}")
    print(f"STEP: {name}")
    print(f"CMD:  {' '.join(str(c) for c in cmd)}")
    print(f"{'='*70}")
    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
    print(f"\n[{name}] {status}  ({elapsed:.0f}s)")
    if check and result.returncode != 0:
        print(f"ERROR: step '{name}' failed — aborting.")
        sys.exit(result.returncode)
    return result.returncode


def _load_json(p: Path) -> dict:
    return json.load(open(p)) if p.exists() else {}


def print_summary(out: Path, scorer: str, budget: int) -> None:
    method = f"bandit_write_{scorer}"

    roll_stats  = _load_json(out / "rollout" / "rollout_stats.json")
    bandit_m    = _load_json(out / "evaluation" / f"{method}_metrics.json")
    if not bandit_m:
        bandit_m = _load_json(out / "evaluation" / f"{method}_B{budget}" / "q_metrics.json")
    audit       = _load_json(out / "reward_audit"  / "audit_report.json")
    scorer_eval = _load_json(out / "scorer_eval"   / "scorer_eval.json")
    analysis    = _load_json(out / "analysis"      / "analysis_complete.json")

    def _q(m): return m.get("q_state_nll") if m else None
    def _fmt(v): return f"{v:.4f}" if isinstance(v, float) else str(v)

    print("\n" + "="*70)
    print("MVP 4b BANDIT-WRITE SUMMARY")
    print("="*70)
    print(f"Scorer:     {scorer}  Budget: {budget}")
    print(f"N objects:  {roll_stats.get('n_total_objects','—')}  "
          f"(persistent={roll_stats.get('n_persistent','—')}  "
          f"buffers={roll_stats.get('n_buffers','—')})")

    print("\n--- Reward audit ---")
    print(f"  verdict:       {audit.get('verdict','—')}")
    if audit.get("gates"):
        for g, v in audit["gates"].items():
            print(f"  {'PASS' if v else 'FAIL'}  {g}")

    print("\n--- Scorer evaluation ---")
    for mname, mres in scorer_eval.get("models", {}).items():
        reg = mres.get("regret", {})
        print(f"  {mname.upper()}:  "
              f"Spearman={_fmt(mres.get('spearman_rho'))}  "
              f"MAE={_fmt(mres.get('mae'))}  "
              f"regret={_fmt(reg.get('mean_regret'))}  "
              f"action_match={_fmt(reg.get('action_match_frac'))}")

    print("\n--- Q-read NLL ---")
    q_bandit = _q(bandit_m)
    rows = [
        (method,            q_bandit),
        ("oracle_teacher",  analysis.get("q_nll_gaps", {}).get("vs_oracle_teacher")),
        ("MVP3b_best",      2.4204),
        ("full_raw_200k",   2.2977),
    ]
    for name, val in rows:
        if name == "oracle_teacher":
            continue
        print(f"  {name:<30}  Q-NLL={_fmt(val)}")

    print("\n--- Gaps ---")
    for k, v in analysis.get("q_nll_gaps", {}).items():
        direction = "better" if isinstance(v, float) and v < 0 else "worse"
        print(f"  {k:<28}  {_fmt(v):>8}  ({direction})")

    verdict = analysis.get("verdict", "—")
    print(f"\n==> MVP 4b verdict: {verdict}")
    rp = out / "report.md"
    if rp.exists():
        print(f"Full report: {rp}")


def main():
    ap = argparse.ArgumentParser()
    # data / output
    ap.add_argument("--source",             required=True)
    ap.add_argument("--offline_qread_dir",  required=True)
    ap.add_argument("--oracle_dir",         required=True)
    ap.add_argument("--output",             required=True)
    # config
    ap.add_argument("--budget",             type=int, default=10000)
    ap.add_argument("--scorer",             default="mlp", choices=["mlp", "gbm", "both"])
    ap.add_argument("--n_decision_points",  type=int, default=10000)
    ap.add_argument("--k_states",           type=int, default=8)
    ap.add_argument("--k_buffers",          type=int, default=8)
    ap.add_argument("--n_local_probes",     type=int, default=16)
    ap.add_argument("--n_history_probes",   type=int, default=16)
    ap.add_argument("--n_replay_probes",    type=int, default=16)
    ap.add_argument("--max_exemplars",      type=int, default=16)
    ap.add_argument("--promotion_support",  type=int, default=8)
    ap.add_argument("--state_budget",       type=int, default=10000)
    ap.add_argument("--buffer_budget",      type=int, default=10000)
    ap.add_argument("--bootstrap_min",      type=int, default=50)
    ap.add_argument("--mlp_epochs",         type=int, default=50)
    ap.add_argument("--gbm_n_estimators",   type=int, default=400)
    ap.add_argument("--max_q_samples",      type=int, default=800_000)
    ap.add_argument("--n_epochs",           type=int, default=30)
    ap.add_argument("--reward_col",         default="reward_penalized")
    ap.add_argument("--seed",               type=int, default=42)
    ap.add_argument("--device",             default="cuda")
    ap.add_argument("--fast",               action="store_true")
    ap.add_argument("--force",              action="store_true")
    # skip flags
    ap.add_argument("--skip_build_rewards", action="store_true")
    ap.add_argument("--skip_audit",         action="store_true")
    ap.add_argument("--skip_train",         action="store_true")
    ap.add_argument("--skip_eval_scorer",   action="store_true")
    ap.add_argument("--skip_rollout",       action="store_true")
    ap.add_argument("--skip_eval_states",   action="store_true")
    ap.add_argument("--skip_analyze",       action="store_true")
    args = ap.parse_args()

    out    = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    src    = Path(__file__).parent

    cfg = vars(args)
    cfg["output"] = str(out)
    with open(out / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    force_flag = ["--force"] if args.force else []

    # Resolve scorer list
    scorers = (["mlp", "gbm"] if args.scorer == "both"
               else [args.scorer])
    primary_scorer = scorers[0]

    # ── 1a: build reward dataset ──────────────────────────────────────────────
    if not args.skip_build_rewards:
        run_step("1a. build_bandit_write_reward_dataset",
                 [python, str(src / "build_bandit_write_reward_dataset.py"),
                  "--source",            args.source,
                  "--oracle_dir",        args.oracle_dir,
                  "--output",            str(out),
                  "--budget",            str(args.budget),
                  "--n_decision_points", str(args.n_decision_points),
                  "--k_states",          str(args.k_states),
                  "--k_buffers",         str(args.k_buffers),
                  "--n_local_probes",    str(args.n_local_probes),
                  "--n_history_probes",  str(args.n_history_probes),
                  "--n_replay_probes",   str(args.n_replay_probes),
                  "--max_exemplars",     str(args.max_exemplars),
                  "--promotion_support", str(args.promotion_support),
                  "--state_budget",      str(args.state_budget),
                  "--buffer_budget",     str(args.buffer_budget),
                  "--seed",              str(args.seed),
                  "--device",            args.device,
                  *force_flag])
    else:
        print("\n[SKIP] build_bandit_write_reward_dataset")

    # ── 1b: audit rewards ─────────────────────────────────────────────────────
    if not args.skip_audit:
        run_step("1b. audit_bandit_write_rewards",
                 [python, str(src / "audit_bandit_write_rewards.py"),
                  "--output",     str(out),
                  "--reward_col", args.reward_col,
                  *force_flag],
                 check=False)   # audit failure prints warning, doesn't abort
    else:
        print("\n[SKIP] audit_bandit_write_rewards")

    # ── 2a: train scorer ──────────────────────────────────────────────────────
    if not args.skip_train:
        model_flags = ["--model"] + scorers
        run_step("2a. train_bandit_write_scorer",
                 [python, str(src / "train_bandit_write_scorer.py"),
                  "--output",           str(out),
                  "--reward_col",       args.reward_col,
                  *model_flags,
                  "--mlp_epochs",       str(args.mlp_epochs),
                  "--gbm_n_estimators", str(args.gbm_n_estimators),
                  "--seed",             str(args.seed),
                  "--device",           args.device,
                  *force_flag])
    else:
        print("\n[SKIP] train_bandit_write_scorer")

    # ── 2b: evaluate scorer ───────────────────────────────────────────────────
    if not args.skip_eval_scorer:
        model_flags = ["--model"] + scorers
        run_step("2b. evaluate_bandit_write_scorer",
                 [python, str(src / "evaluate_bandit_write_scorer.py"),
                  "--output",     str(out),
                  "--reward_col", args.reward_col,
                  *model_flags,
                  "--device",     args.device,
                  *force_flag])
    else:
        print("\n[SKIP] evaluate_bandit_write_scorer")

    # ── 3a: rollout ───────────────────────────────────────────────────────────
    if not args.skip_rollout:
        run_step("3a. rollout_bandit_write",
                 [python, str(src / "rollout_bandit_write.py"),
                  "--source",            args.source,
                  "--output",            str(out),
                  "--budget",            str(args.budget),
                  "--k_states",          str(args.k_states),
                  "--k_buffers",         str(args.k_buffers),
                  "--max_exemplars",     str(args.max_exemplars),
                  "--promotion_support", str(args.promotion_support),
                  "--state_budget",      str(args.state_budget),
                  "--buffer_budget",     str(args.buffer_budget),
                  "--bootstrap_min",     str(args.bootstrap_min),
                  "--scorer",            primary_scorer,
                  "--seed",              str(args.seed),
                  "--device",            args.device,
                  *force_flag])
    else:
        print("\n[SKIP] rollout_bandit_write")

    # ── 3b: evaluate states ───────────────────────────────────────────────────
    if not args.skip_eval_states:
        q_flags = [
            "--max_q_samples", str(args.max_q_samples),
            "--n_epochs",      str(args.n_epochs),
        ]
        if args.fast:
            q_flags.append("--fast")
        run_step("3b. evaluate_bandit_write_states",
                 [python, str(src / "evaluate_bandit_write_states.py"),
                  "--source",            args.source,
                  "--offline_qread_dir", args.offline_qread_dir,
                  "--output",            str(out),
                  "--scorer",            primary_scorer,
                  "--budget",            str(args.budget),
                  "--seed",              str(args.seed),
                  "--device",            args.device,
                  *q_flags,
                  *force_flag])
    else:
        print("\n[SKIP] evaluate_bandit_write_states")

    # ── 3c: analyze ───────────────────────────────────────────────────────────
    if not args.skip_analyze:
        run_step("3c. analyze_bandit_write_rollout",
                 [python, str(src / "analyze_bandit_write_rollout.py"),
                  "--output",     str(out),
                  "--oracle_dir", args.oracle_dir,
                  "--scorer",     primary_scorer,
                  "--budget",     str(args.budget),
                  *force_flag])
    else:
        print("\n[SKIP] analyze_bandit_write_rollout")

    # ── summary ───────────────────────────────────────────────────────────────
    print_summary(out, primary_scorer, args.budget)


if __name__ == "__main__":
    main()

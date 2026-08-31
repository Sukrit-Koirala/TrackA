"""
run_mvp4c2_end_to_end.py  --  Full pipeline from oracle → bandit → credit audit

Single-source orchestrator.  Everything derives from one directory:
  source/states/datastore.pt  (student GPT-2 base, 768-dim)

Phase A  (oracle reconstruction)   →  {output}/oracle/
Phase B  (bandit rollout)          →  {output}/bandit/
Phase C  (temporal credit audit)   →  {output}/audit/

Usage:
  python src/run_mvp4c2_end_to_end.py \\
    --source            scale_200k_seed42 \\
    --offline_states    outputs_track_a_offline_paper/seed42/states \\
    --offline_qread     outputs_track_a_offline_paper/seed42/q_read/states \\
    --output            outputs_mvp4c2_full \\
    --budget            10000 \\
    --seed              42 \\
    --device            cuda

Skip flags:
  --skip_oracle   skip Phase A (use existing oracle output)
  --skip_bandit   skip Phase B (use existing bandit output)
  --skip_audit    skip Phase C

  --oracle_dir <path>  use existing oracle dir instead of rebuilding
  --bandit_dir <path>  use existing bandit dir instead of rebuilding
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def run(label: str, cmd: list, check: bool = True) -> int:
    print(f"\n{'#'*72}")
    print(f"  {label}")
    print(f"{'#'*72}")
    print(f"  CMD: {' '.join(str(c) for c in cmd)}\n")
    rc = subprocess.run([sys.executable] + [str(c) for c in cmd]).returncode
    if rc != 0:
        print(f"\nERROR: '{label}' failed (exit {rc})")
        if check:
            sys.exit(rc)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",         required=True,
                    help="Student source dir (contains states/datastore.pt)")
    ap.add_argument("--offline_states", required=True,
                    help="Offline KMeans states dir (outputs_track_a_offline_paper/seed42/states)")
    ap.add_argument("--offline_qread",  required=True,
                    help="Offline Q-read dir (outputs_track_a_offline_paper/seed42/q_read/states)")
    ap.add_argument("--output",         required=True,
                    help="Root output dir; oracle/, bandit/, audit/ created inside")
    ap.add_argument("--budget",         type=int, default=10000)
    ap.add_argument("--promotion_support", type=int, default=8)
    ap.add_argument("--seed",           type=int, default=42)
    ap.add_argument("--device",         default="cuda")
    ap.add_argument("--max_queries",    type=int, default=5000)
    ap.add_argument("--max_defer_shadows", type=int, default=500)
    ap.add_argument("--n_epochs",       type=int, default=30)
    ap.add_argument("--max_q_samples",  type=int, default=800000)

    # Skip / override flags
    ap.add_argument("--skip_oracle",  action="store_true")
    ap.add_argument("--skip_bandit",  action="store_true")
    ap.add_argument("--skip_audit",   action="store_true")
    ap.add_argument("--oracle_dir",   default=None,
                    help="Use existing oracle dir (implies --skip_oracle)")
    ap.add_argument("--bandit_dir",   default=None,
                    help="Use existing bandit dir (implies --skip_bandit)")
    ap.add_argument("--force",        action="store_true")
    args = ap.parse_args()

    out    = Path(args.output)
    source = Path(args.source)
    force  = ["--force"] if args.force else []

    # Validate student datastore exists upfront
    ds = source / "states" / "datastore.pt"
    if not ds.exists():
        print(f"ERROR: Student datastore not found: {ds}")
        print(f"  The pipeline requires {source}/states/datastore.pt")
        sys.exit(1)
    print(f"Student datastore: {ds}")

    # ── Derived output dirs ───────────────────────────────────────────────────
    oracle_out = Path(args.oracle_dir) if args.oracle_dir else out / "oracle"
    bandit_out = Path(args.bandit_dir) if args.bandit_dir else out / "bandit"
    audit_out  = out / "audit"

    for d in (out, oracle_out, bandit_out, audit_out):
        d.mkdir(parents=True, exist_ok=True)

    # ── Phase A: Oracle reconstruction ───────────────────────────────────────
    skip_oracle = args.skip_oracle or (args.oracle_dir is not None)
    if skip_oracle:
        print(f"\n[SKIP Phase A] Using oracle: {oracle_out}")
    else:
        run(
            "Phase A — Oracle reconstruction",
            [SCRIPT_DIR / "run_mvp4a0_oracle_reconstruction.py",
             "--source",             source,
             "--offline_states_dir", args.offline_states,
             "--offline_qread_dir",  args.offline_qread,
             "--output",             oracle_out,
             "--teacher",            "minibatch_kmeans",
             "--budget",             args.budget,
             "--promotion_support",  args.promotion_support,
             "--seed",               args.seed,
             "--device",             args.device,
             "--max_q_samples",      args.max_q_samples,
             "--n_epochs",           args.n_epochs],
        )

    # ── Phase B: Bandit rollout ───────────────────────────────────────────────
    skip_bandit = args.skip_bandit or (args.bandit_dir is not None)
    if skip_bandit:
        print(f"\n[SKIP Phase B] Using bandit: {bandit_out}")
    else:
        run(
            "Phase B — Bandit rollout",
            [SCRIPT_DIR / "run_mvp4b1_onpolicy_bandit_write.py",
             "--source",             source,
             "--oracle_dir",         oracle_out,
             "--offline_qread_dir",  args.offline_qread,
             "--output",             bandit_out,
             "--object_budget",      args.budget,
             "--scorer",             "mlp",
             "--n_decision_points",  10000,
             "--n_lm_decision_points", 5000,
             "--max_lm_objects",     500,
             "--episode_length_lm",  2000,
             "--k_states",           8,
             "--k_buffers",          8,
             "--n_local_probes",     16,
             "--n_history_probes",   16,
             "--n_replay_probes",    16,
             "--max_exemplars",      16,
             "--promotion_support",  args.promotion_support,
             "--mlp_epochs",         60,
             "--gbm_n_estimators",   400,
             "--reward_col",         "reward_penalized",
             "--max_q_samples",      args.max_q_samples,
             "--n_epochs_q",         args.n_epochs,
             "--seed",               args.seed,
             "--device",             args.device],
        )

    # ── Phase C: Temporal credit audit ───────────────────────────────────────
    if args.skip_audit:
        print(f"\n[SKIP Phase C] Audit skipped.")
    else:
        run(
            "Phase C — Temporal credit audit (MVP 4c-2)",
            [SCRIPT_DIR / "run_mvp4c2_repaired_credit_audit.py",
             "--source",             source,
             "--oracle_dir",         oracle_out,
             "--bandit_dir",         bandit_out,
             "--bandit_rollout_dir", bandit_out,
             "--output",             audit_out,
             "--device",             args.device,
             "--seed",               args.seed,
             "--max_queries",        args.max_queries,
             "--max_defer_shadows",  args.max_defer_shadows] + force,
        )

    print(f"\n{'='*72}")
    print(f"  END-TO-END PIPELINE COMPLETE")
    print(f"  oracle:  {oracle_out}")
    print(f"  bandit:  {bandit_out}")
    print(f"  audit:   {audit_out}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()

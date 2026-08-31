#!/bin/bash
# slurm_mvp4c2_end_to_end.sh
#
# Submits the full pipeline as three chained SLURM jobs:
#   JOB A  oracle reconstruction   (~6h)
#   JOB B  bandit rollout          (~16h, runs after A)
#   JOB C  credit audit            (~24h, runs after B)
#
# Usage:
#   bash scripts/slurm_mvp4c2_end_to_end.sh
#
# Override output root:
#   OUTPUT_ROOT=outputs_mvp4c2_fresh bash scripts/slurm_mvp4c2_end_to_end.sh

set -euo pipefail

BASE=/home/sukrit.koirala/ondemand/upload_me/RegionTokenizer
RL=$BASE/scripts/RL

# ── Single source of truth ────────────────────────────────────────────────────
SOURCE=$RL/scale_200k_seed42
OFFLINE_STATES=$RL/outputs_track_a_offline_paper/seed42/states
OFFLINE_QREAD=$RL/outputs_track_a_offline_paper/seed42/q_read/states

# All outputs go under this root
OUTPUT_ROOT=${OUTPUT_ROOT:-$RL/outputs_mvp4c2_full}
ORACLE_OUT=$OUTPUT_ROOT/oracle
BANDIT_OUT=$OUTPUT_ROOT/bandit
AUDIT_OUT=$OUTPUT_ROOT/audit

mkdir -p $RL/logs $ORACLE_OUT $BANDIT_OUT $AUDIT_OUT

# ── Preflight: verify student datastore exists ────────────────────────────────
if [ ! -f "$SOURCE/states/datastore.pt" ]; then
    echo "ERROR: student datastore not found: $SOURCE/states/datastore.pt"
    echo "This file is required by all stages. Cannot proceed."
    exit 1
fi
echo "Student datastore OK: $SOURCE/states/datastore.pt"

if [ ! -d "$OFFLINE_STATES" ]; then
    echo "ERROR: offline states not found: $OFFLINE_STATES"
    exit 1
fi
if [ ! -d "$OFFLINE_QREAD" ]; then
    echo "ERROR: offline qread not found: $OFFLINE_QREAD"
    exit 1
fi

echo ""
echo "Pipeline output root: $OUTPUT_ROOT"
echo ""

# ── Job A: Oracle reconstruction ─────────────────────────────────────────────
JOB_A=$(sbatch --parsable \
  --job-name=mvp4c2_oracle \
  --partition=gpuGeneral \
  --nodes=1 --ntasks-per-node=1 \
  --gres=gpu:l40s:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=8:00:00 \
  --output=$RL/logs/mvp4c2_oracle-%j.out \
  --error=$RL/logs/mvp4c2_oracle-%j.err \
  --wrap="
    set -e
    cd $RL
    export PYTHONPATH=$RL:\$PYTHONPATH
    export CUDA_VISIBLE_DEVICES=0
    source venv/bin/activate 2>/dev/null || true

    echo 'Job A started: \$(date)'
    python src/run_mvp4a0_oracle_reconstruction.py \
      --source              $SOURCE \
      --offline_states_dir  $OFFLINE_STATES \
      --offline_qread_dir   $OFFLINE_QREAD \
      --output              $ORACLE_OUT \
      --teacher             minibatch_kmeans \
      --budget              10000 \
      --promotion_support   8 \
      --seed                42 \
      --device              cuda \
      --max_q_samples       800000 \
      --n_epochs            30
    echo 'Job A done: \$(date)'
  ")

echo "Submitted Job A (oracle): $JOB_A"

# ── Job B: Bandit rollout (runs after A) ──────────────────────────────────────
JOB_B=$(sbatch --parsable \
  --job-name=mvp4c2_bandit \
  --partition=gpuGeneral \
  --nodes=1 --ntasks-per-node=1 \
  --gres=gpu:l40s:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=18:00:00 \
  --dependency=afterok:$JOB_A \
  --output=$RL/logs/mvp4c2_bandit-%j.out \
  --error=$RL/logs/mvp4c2_bandit-%j.err \
  --wrap="
    set -e
    cd $RL
    export PYTHONPATH=$RL:\$PYTHONPATH
    export CUDA_VISIBLE_DEVICES=0
    source venv/bin/activate 2>/dev/null || true

    echo 'Job B started: \$(date)'
    python src/run_mvp4b1_onpolicy_bandit_write.py \
      --source              $SOURCE \
      --oracle_dir          $ORACLE_OUT \
      --offline_qread_dir   $OFFLINE_QREAD \
      --output              $BANDIT_OUT \
      --object_budget       10000 \
      --scorer              mlp \
      --n_decision_points   10000 \
      --n_lm_decision_points 5000 \
      --max_lm_objects      500 \
      --episode_length_lm   2000 \
      --k_states            8 \
      --k_buffers           8 \
      --n_local_probes      16 \
      --n_history_probes    16 \
      --n_replay_probes     16 \
      --max_exemplars       16 \
      --promotion_support   8 \
      --mlp_epochs          60 \
      --gbm_n_estimators    400 \
      --reward_col          reward_penalized \
      --max_q_samples       800000 \
      --n_epochs_q          30 \
      --seed                42 \
      --device              cuda
    echo 'Job B done: \$(date)'
  ")

echo "Submitted Job B (bandit): $JOB_B  [depends on $JOB_A]"

# ── Job C: Credit audit (runs after B) ────────────────────────────────────────
JOB_C=$(sbatch --parsable \
  --job-name=mvp4c2_audit \
  --partition=gpuGeneral \
  --nodes=1 --ntasks-per-node=1 \
  --gres=gpu:l40s:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=24:00:00 \
  --dependency=afterok:$JOB_B \
  --output=$RL/logs/mvp4c2_audit-%j.out \
  --error=$RL/logs/mvp4c2_audit-%j.err \
  --wrap="
    set -e
    cd $RL
    export PYTHONPATH=$RL:\$PYTHONPATH
    export CUDA_VISIBLE_DEVICES=0
    source venv/bin/activate 2>/dev/null || true

    python -c 'import sklearn, scipy, matplotlib' 2>/dev/null || \
        pip install scikit-learn scipy matplotlib --quiet

    echo 'Job C started: \$(date)'
    python src/run_mvp4c2_repaired_credit_audit.py \
      --source              $SOURCE \
      --oracle_dir          $ORACLE_OUT \
      --bandit_dir          $BANDIT_OUT \
      --bandit_rollout_dir  $BANDIT_OUT \
      --output              $AUDIT_OUT \
      --device              cuda \
      --seed                42 \
      --max_queries         5000 \
      --max_defer_shadows   500
    echo 'Job C done: \$(date)'
  ")

echo "Submitted Job C (audit):  $JOB_C  [depends on $JOB_B]"

echo ""
echo "Chain: $JOB_A → $JOB_B → $JOB_C"
echo "Output root: $OUTPUT_ROOT"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f $RL/logs/mvp4c2_oracle-${JOB_A}.out"
echo "  tail -f $RL/logs/mvp4c2_bandit-${JOB_B}.out"
echo "  tail -f $RL/logs/mvp4c2_audit-${JOB_C}.out"

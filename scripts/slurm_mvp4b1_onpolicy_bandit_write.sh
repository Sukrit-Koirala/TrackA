#!/bin/bash
#SBATCH --job-name=mvp4b1_onpolicy
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/RegionTokenizer/scripts/RL/logs/mvp4b1_onpolicy-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/RegionTokenizer/scripts/RL/logs/mvp4b1_onpolicy-%j.err

set -e
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

# ── Environment ───────────────────────────────────────────────────────────────
BASE=/home/sukrit.koirala/ondemand/upload_me/RegionTokenizer
cd $BASE/scripts/RL

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate region_tokenizer 2>/dev/null || conda activate base
elif [ -f "$HOME/miniconda3/bin/activate" ]; then
    source "$HOME/miniconda3/bin/activate"
    conda activate region_tokenizer 2>/dev/null || conda activate base
fi

export PYTHONPATH="$BASE:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

# ── Paths ─────────────────────────────────────────────────────────────────────
SOURCE=$BASE/scale_200k_seed42
ORACLE_DIR=$BASE/outputs_mvp4a0_oracle_reconstruction
OFFLINE_QREAD=$BASE/outputs_track_a_offline_paper/seed42/q_read/states
OUTPUT=$BASE/outputs_mvp4b1_onpolicy_bandit_write_clean_fast

mkdir -p $BASE/scripts/RL/logs
mkdir -p $OUTPUT

# ── Run ───────────────────────────────────────────────────────────────────────
python $BASE/scripts/RL/src/run_mvp4b1_onpolicy_bandit_write.py \
  --source              $SOURCE                 \
  --oracle_dir          $ORACLE_DIR             \
  --offline_qread_dir   $OFFLINE_QREAD          \
  --output              $OUTPUT                 \
  --object_budget       10000                   \
  --scorer              mlp                     \
  --n_decision_points   10000                   \
  --n_lm_decision_points 5000                   \
  --max_lm_objects      500                     \
  --episode_length_lm   2000                    \
  --k_states            8                       \
  --k_buffers           8                       \
  --n_local_probes      16                      \
  --n_history_probes    16                      \
  --n_replay_probes     16                      \
  --max_exemplars       16                      \
  --promotion_support   8                       \
  --mlp_epochs          60                      \
  --gbm_n_estimators    400                     \
  --reward_col          reward_penalized         \
  --max_q_samples       800000                  \
  --n_epochs_q          30                      \
  --seed                42                      \
  --device              cuda

echo "Job finished: $(date)"

#!/bin/bash
#SBATCH --job-name=mvp4c2_bandit
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/RegionTokenizer/scripts/RL/logs/mvp4c2_bandit-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/RegionTokenizer/scripts/RL/logs/mvp4c2_bandit-%j.err

set -e
echo "Job B (bandit) started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

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

RL_BASE=$BASE/scripts/RL
export PYTHONPATH="$RL_BASE:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

SOURCE=$RL_BASE/scale_200k_seed42
OFFLINE_QREAD=$RL_BASE/outputs_track_a_offline_paper/seed42/q_read/states
OUTPUT_ROOT=${OUTPUT_ROOT:-$RL_BASE/outputs_mvp4c2_full}
ORACLE_OUT=$OUTPUT_ROOT/oracle
BANDIT_OUT=$OUTPUT_ROOT/bandit

mkdir -p $BANDIT_OUT

python $RL_BASE/src/run_mvp4b1_onpolicy_bandit_write.py \
  --source                $SOURCE              \
  --oracle_dir            $ORACLE_OUT          \
  --offline_qread_dir     $OFFLINE_QREAD       \
  --output                $BANDIT_OUT          \
  --object_budget         10000                \
  --scorer                mlp                  \
  --n_decision_points     10000                \
  --n_lm_decision_points  5000                 \
  --max_lm_objects        500                  \
  --episode_length_lm     2000                 \
  --k_states              8                    \
  --k_buffers             8                    \
  --n_local_probes        16                   \
  --n_history_probes      16                   \
  --n_replay_probes       16                   \
  --max_exemplars         16                   \
  --promotion_support     8                    \
  --mlp_epochs            60                   \
  --gbm_n_estimators      400                  \
  --reward_col            reward_penalized      \
  --max_q_samples         800000              \
  --n_epochs_q            30                   \
  --seed                  42                   \
  --device                cuda

echo "Job B (bandit) done: $(date)"

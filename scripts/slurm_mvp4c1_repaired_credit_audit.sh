#!/bin/bash
#SBATCH --job-name=mvp4c1_true_replay
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/RegionTokenizer/scripts/RL/logs/mvp4c1_true_replay-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/RegionTokenizer/scripts/RL/logs/mvp4c1_true_replay-%j.err

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

python -c "import sklearn, scipy, matplotlib" 2>/dev/null || \
    pip install scikit-learn scipy matplotlib --quiet

RL_BASE=$BASE/scripts/RL
export PYTHONPATH="$RL_BASE:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

# ── Paths ─────────────────────────────────────────────────────────────────────
SOURCE=$RL_BASE/scale_200k_seed42
DATASTORE_DIR=$RL_BASE/outputs_track_a_offline_paper_gpt2_medium/seed42
ORACLE_DIR=$RL_BASE/outputs_mvp4a0_oracle_reconstruction
BANDIT_DIR=$RL_BASE/outputs_mvp4b_bandit_write_fast
# Bandit ROLLOUT dir — used by Stage 5b (immediate reward join)
# Must contain reward_datasets/onpolicy/actions.parquet
# Re-run rollout_mvp4b1.py --collect_rewards if this file is missing
BANDIT_ROLLOUT_DIR=$RL_BASE/outputs_mvp4b_bandit_write_fast
OUTPUT=$RL_BASE/outputs_mvp4c1_repaired_credit_audit_fast

mkdir -p $RL_BASE/logs
mkdir -p $OUTPUT

# ── Run ───────────────────────────────────────────────────────────────────────
# Stage 4 uses --force to invalidate cached val_step-mapped replay from old pipeline.
# Stage 5b is non-blocking (exits 2 on low join rate, warns but continues).
python $RL_BASE/src/run_mvp4c1_repaired_credit_audit.py \
  --source                $SOURCE              \
  --datastore_dir         $DATASTORE_DIR       \
  --oracle_dir            $ORACLE_DIR          \
  --bandit_dir            $BANDIT_DIR          \
  --bandit_rollout_dir    $BANDIT_ROLLOUT_DIR  \
  --output                $OUTPUT              \
  --device                cuda                 \
  --seed                  42                   \
  --max_queries           5000

echo "Job finished: $(date)"

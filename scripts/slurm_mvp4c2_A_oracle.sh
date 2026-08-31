#!/bin/bash
#SBATCH --job-name=mvp4c2_oracle
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/RegionTokenizer/scripts/RL/logs/mvp4c2_oracle-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/RegionTokenizer/scripts/RL/logs/mvp4c2_oracle-%j.err

set -e
echo "Job A (oracle) started: $(date)"
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
OFFLINE_STATES=$RL_BASE/outputs_track_a_offline_paper/seed42/states
OFFLINE_QREAD=$RL_BASE/outputs_track_a_offline_paper/seed42/q_read/states
OUTPUT_ROOT=${OUTPUT_ROOT:-$RL_BASE/outputs_mvp4c2_full}
ORACLE_OUT=$OUTPUT_ROOT/oracle

mkdir -p $RL_BASE/logs $ORACLE_OUT

python $RL_BASE/src/run_mvp4a0_oracle_reconstruction.py \
  --source              $SOURCE              \
  --offline_states_dir  $OFFLINE_STATES      \
  --offline_qread_dir   $OFFLINE_QREAD       \
  --output              $ORACLE_OUT          \
  --teacher             minibatch_kmeans     \
  --budget              10000                \
  --promotion_support   8                   \
  --seed                42                   \
  --device              cuda                 \
  --max_q_samples       800000              \
  --n_epochs            30

echo "Job A (oracle) done: $(date)"

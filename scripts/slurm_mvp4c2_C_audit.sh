#!/bin/bash
#SBATCH --job-name=mvp4c2_audit
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/RegionTokenizer/scripts/RL/logs/mvp4c2_audit-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/RegionTokenizer/scripts/RL/logs/mvp4c2_audit-%j.err

set -e
echo "Job C (audit) started: $(date)"
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

python -c "import sklearn, scipy, matplotlib" 2>/dev/null || \
    pip install scikit-learn scipy matplotlib --quiet

RL_BASE=$BASE/scripts/RL
export PYTHONPATH="$RL_BASE:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

SOURCE=$RL_BASE/scale_200k_seed42
OUTPUT_ROOT=${OUTPUT_ROOT:-$RL_BASE/outputs_mvp4c2_full}
ORACLE_OUT=$OUTPUT_ROOT/oracle
BANDIT_OUT=$OUTPUT_ROOT/bandit
AUDIT_OUT=$OUTPUT_ROOT/audit

mkdir -p $AUDIT_OUT

python $RL_BASE/src/run_mvp4c2_repaired_credit_audit.py \
  --source              $SOURCE              \
  --oracle_dir          $ORACLE_OUT          \
  --bandit_dir          $BANDIT_OUT          \
  --bandit_rollout_dir  $BANDIT_OUT          \
  --output              $AUDIT_OUT           \
  --device              cuda                 \
  --seed                42                   \
  --max_queries         5000                 \
  --max_defer_shadows   500

echo "Job C (audit) done: $(date)"

#!/bin/bash
#SBATCH --job-name=mvp4c2_repaired_audit
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/RegionTokenizer/scripts/RL/logs/mvp4c2_repaired_audit-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/RegionTokenizer/scripts/RL/logs/mvp4c2_repaired_audit-%j.err

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
# Single source of truth: student GPT-2 base, 768-dim.
# build_write_object_provenance.py reads from SOURCE/states/datastore.pt.
# replay_online_memory_chronologically.py now also reads from the same file.
SOURCE=$RL_BASE/scale_200k_seed42

# Student offline KMeans states (NOT the gpt2_medium teacher dir).
# This is where minibatch_kmeans_B10000.pt lives.
OFFLINE_STATES_DIR=$RL_BASE/outputs_track_a_offline_paper/seed42

ORACLE_DIR=$RL_BASE/outputs_mvp4a0_oracle_reconstruction

# Use the newer clean bandit rollout if available, else fall back to fast one.
if [ -d "$RL_BASE/outputs_mvp4b1_onpolicy_bandit_write_clean_fast" ]; then
    BANDIT_DIR=$RL_BASE/outputs_mvp4b1_onpolicy_bandit_write_clean_fast
else
    BANDIT_DIR=$RL_BASE/outputs_mvp4b_bandit_write_fast
fi
BANDIT_ROLLOUT_DIR=$BANDIT_DIR

# NEW output dir — do NOT reuse 4c-1 temporal cache.
OUTPUT=$RL_BASE/outputs_mvp4c2_repaired_credit_audit_fast

mkdir -p $RL_BASE/logs
mkdir -p $OUTPUT

# ── Run ───────────────────────────────────────────────────────────────────────
# No --force: cached stages skip automatically.
# Stage 10 is the blocking validation gate — the run aborts if it fails.
# --datastore_dir is intentionally omitted; defaults to --source inside the orchestrator.
python $RL_BASE/src/run_mvp4c2_repaired_credit_audit.py \
  --source                $SOURCE              \
  --oracle_dir            $ORACLE_DIR          \
  --bandit_dir            $BANDIT_DIR          \
  --bandit_rollout_dir    $BANDIT_ROLLOUT_DIR  \
  --output                $OUTPUT              \
  --device                cuda                 \
  --seed                  42                   \
  --max_queries           5000                 \
  --max_defer_shadows     500

echo "Job finished: $(date)"

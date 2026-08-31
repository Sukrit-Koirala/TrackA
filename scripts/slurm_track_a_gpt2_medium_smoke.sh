#!/bin/bash
#SBATCH --job-name=track_a_gpt2m_smoke
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/RegionTokenizer/scripts/RL/logs/track_a_gpt2m_smoke-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/RegionTokenizer/scripts/RL/logs/track_a_gpt2m_smoke-%j.err
#
# Track A: GPT-2 Medium Second-Model Replication — Smoke Test
# Seed 42, budgets 5k/10k, 200k Q samples, 100k datastore
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/ondemand/upload_me/RegionTokenizer/scripts/RL
export PYTHONPATH="$PWD"
mkdir -p logs

# ── Environment activation ────────────────────────────────────────────────────
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate learned_regions 2>/dev/null || conda activate base
elif [ -f "$HOME/miniconda3/bin/activate" ]; then
    source "$HOME/miniconda3/bin/activate"
    conda activate learned_regions 2>/dev/null || conda activate base
fi

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT="outputs_track_a_offline_paper_gpt2_medium_smoke"
SCRIPT="src/paper/run_model_replication.py"

echo "========================================================"
echo " Track A: GPT-2 Medium Second-Model Smoke Test"
echo " output:  ${OUTPUT}"
echo " $(date)"
echo "========================================================"
echo ""

# ── Preflight ─────────────────────────────────────────────────────────────────
echo "[preflight] Checking environment..."
[ ! -f "$SCRIPT" ] && { echo "ERROR: $SCRIPT not found"; exit 1; }

for f in "$SCRIPT" src/paper/run_full_raw_q_read.py src/paper/aggregate_paper_results.py; do
    python -m py_compile "$f" && echo "  [OK] $f compiles" || { echo "ERROR: $f fails to compile"; exit 1; }
done

python -c "
import torch, sys
from transformers import GPT2LMHeadModel
print(f'  PyTorch: {torch.__version__}')
if not torch.cuda.is_available():
    print('  ERROR: CUDA not available'); sys.exit(1)
print(f'  GPU:  {torch.cuda.get_device_name(0)}')
print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
print('  Checking gpt2-medium can be instantiated ...')
cfg = GPT2LMHeadModel.from_pretrained('gpt2-medium', torch_dtype=torch.float16).config
print(f'  gpt2-medium: hidden_dim={cfg.n_embd}  vocab={cfg.vocab_size}  layers={cfg.n_layer}')
"

echo ""
echo "[preflight] All checks passed."
echo ""

# ── Launch ────────────────────────────────────────────────────────────────────
echo "[launch] SMOKE TEST (seed 42, budgets 5k/10k, 200k Q samples, full_raw)"
echo ""

python "$SCRIPT" \
    --model_name     gpt2-medium \
    --dataset        TinyStories \
    --output         "$OUTPUT" \
    --datastore_size 100000 \
    --seeds          42 \
    --budgets        5000 10000 \
    --methods        minibatch_kmeans utility_weighted \
    --raw_baselines  raw_random cluster_medoids raw_high_gpt_entropy raw_high_gpt_loss \
    --run_full_raw \
    --fast_grid \
    --max_q_samples  200000 \
    --device         cuda

EXIT=$?
if [ $EXIT -ne 0 ]; then
    echo ""; echo "ERROR: pipeline exited with code $EXIT"; exit $EXIT
fi

echo ""
echo "========================================================"
echo " GPT-2 medium smoke test complete.  $(date)"
echo "========================================================"
echo ""

# ── Print final verdict ───────────────────────────────────────────────────────
python - <<PYEOF
import os, json

output = "${OUTPUT}"

vp = f"{output}/reports/second_model_verdict.json"
if os.path.isfile(vp):
    v = json.load(open(vp))
    print(f"[second model verdict]  {v.get('verdict')}")
    print(f"  Model:             {v.get('model')}")
    print(f"  GPT NLL:           {v.get('gpt_nll')}")
    print(f"  Best state Q NLL:  {v.get('best_state_q_nll')}")
    print(f"  Full raw Q NLL:    {v.get('full_raw_q_nll')}")
    print(f"  Win rate vs raw:   {v.get('state_win_rate_vs_raw')}")
else:
    print("[WARN] second_model_verdict.json not found")
    sum_path = f"{output}/model_replication_summary.json"
    if os.path.isfile(sum_path):
        s = json.load(open(sum_path))
        for r in s.get("results_by_seed", []):
            print(f"  Seed {r.get('seed')}: {r.get('overall')}  best_q={r.get('best_q_nll')}")

cmp = f"{output}/reports/GPT2_SMALL_VS_GPT2_MEDIUM_COMPARISON.md"
if os.path.isfile(cmp):
    print(f"\n[comparison report]")
    for line in open(cmp).readlines()[:30]:
        print(f"  {line}", end="")
PYEOF

echo ""
echo "[done] Outputs: ${OUTPUT}/"
echo "  Comparison: ${OUTPUT}/reports/GPT2_SMALL_VS_GPT2_MEDIUM_COMPARISON.md"
echo "  Verdict:    ${OUTPUT}/reports/second_model_verdict.json"

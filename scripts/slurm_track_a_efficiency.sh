#!/bin/bash
#SBATCH --job-name=track_a_efficiency
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/RegionTokenizer/scripts/RL/logs/track_a_efficiency-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/RegionTokenizer/scripts/RL/logs/track_a_efficiency-%j.err
#
# Track A: DIME Efficiency / Cheapness Analysis
# Loads existing experiment outputs — no re-extraction.
# Runs for: TinyStories (GPT-2 small) + WikiText-2 (GPT-2 small)
# Also runs for GPT-2 medium if outputs_track_a_offline_paper_gpt2_medium/ exists.
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
SCRIPT="src/paper/analyze_dime_efficiency.py"
SEEDS=(42 123 999)

TS_OUTPUT="outputs_track_a_offline_paper"
TS_SOURCE="scale_200k_seed42"
WT2_OUTPUT="outputs_track_a_offline_paper_wikitext2"
GPT2M_OUTPUT="outputs_track_a_offline_paper_gpt2_medium"

echo "========================================================"
echo " Track A: DIME Efficiency Analysis"
echo " TinyStories: ${TS_OUTPUT}"
echo " WikiText-2:  ${WT2_OUTPUT}"
echo " $(date)"
echo "========================================================"
echo ""

# ── Preflight ─────────────────────────────────────────────────────────────────
echo "[preflight] Checking environment..."
[ ! -f "$SCRIPT" ] && { echo "ERROR: $SCRIPT not found"; exit 1; }

python -m py_compile "$SCRIPT" && echo "  [OK] $SCRIPT compiles" \
    || { echo "ERROR: $SCRIPT fails to compile"; exit 1; }

python -c "
import torch, sys
print(f'  PyTorch: {torch.__version__}')
if not torch.cuda.is_available():
    print('  ERROR: CUDA not available'); sys.exit(1)
print(f'  GPU:  {torch.cuda.get_device_name(0)}')
print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
"

echo ""
echo "[preflight] All checks passed."
echo ""

# ── TinyStories ───────────────────────────────────────────────────────────────
echo "------------------------------------------------------------"
echo " TinyStories / GPT-2 small"
echo " $(date)"
echo "------------------------------------------------------------"

if [ -d "${TS_OUTPUT}" ]; then
    python "$SCRIPT" \
        --source   "${TS_SOURCE}" \
        --output   "${TS_OUTPUT}" \
        --dataset  TinyStories \
        --model    gpt2 \
        --seeds    "${SEEDS[@]}" \
        --device   cuda

    EXIT=$?
    if [ $EXIT -ne 0 ]; then
        echo "  WARNING: TinyStories efficiency exited with code ${EXIT}"
    else
        echo "  TinyStories efficiency complete."
    fi
else
    echo "  [SKIP] ${TS_OUTPUT} not found"
fi
echo ""

# ── WikiText-2 ────────────────────────────────────────────────────────────────
echo "------------------------------------------------------------"
echo " WikiText-2 / GPT-2 small"
echo " $(date)"
echo "------------------------------------------------------------"

if [ -d "${WT2_OUTPUT}" ]; then
    # WikiText-2: data source lives in each seed's own states/ dir (no shared source)
    python "$SCRIPT" \
        --output   "${WT2_OUTPUT}" \
        --dataset  WikiText-2 \
        --model    gpt2 \
        --seeds    "${SEEDS[@]}" \
        --device   cuda

    EXIT=$?
    if [ $EXIT -ne 0 ]; then
        echo "  WARNING: WikiText-2 efficiency exited with code ${EXIT}"
    else
        echo "  WikiText-2 efficiency complete."
    fi
else
    echo "  [SKIP] ${WT2_OUTPUT} not found"
fi
echo ""

# ── GPT-2 medium (if available) ───────────────────────────────────────────────
echo "------------------------------------------------------------"
echo " GPT-2 medium (if outputs exist)"
echo " $(date)"
echo "------------------------------------------------------------"

if [ -d "${GPT2M_OUTPUT}" ]; then
    python "$SCRIPT" \
        --output     "${GPT2M_OUTPUT}" \
        --dataset    TinyStories \
        --model      gpt2-medium \
        --seeds      "${SEEDS[@]}" \
        --vocab_size 50257 \
        --device     cuda

    EXIT=$?
    if [ $EXIT -ne 0 ]; then
        echo "  WARNING: GPT-2 medium efficiency exited with code ${EXIT}"
    else
        echo "  GPT-2 medium efficiency complete."
    fi
else
    echo "  [SKIP] ${GPT2M_OUTPUT} not found — GPT-2 medium analysis skipped"
fi
echo ""

echo "========================================================"
echo " Efficiency analysis done.  $(date)"
echo "========================================================"
echo ""

# ── Collect verdicts ──────────────────────────────────────────────────────────
echo "[summary] Efficiency verdicts:"
for vp in \
    "${TS_OUTPUT}/reports/dime_efficiency_verdict.json" \
    "${WT2_OUTPUT}/reports/dime_efficiency_verdict.json" \
    "${GPT2M_OUTPUT}/reports/dime_efficiency_verdict.json"
do
    if [ -f "$vp" ]; then
        python - <<PYEOF
import json
d = json.load(open("${vp}"))
print(f"  {d['dataset']} / {d['model']}:  {d['verdict']}")
print(f"    entry compression:   {d.get('entry_compression')}×")
print(f"    DIME Q NLL:          {d.get('best_dime_q_nll')}")
print(f"    full raw Q NLL:      {d.get('fr_q_nll')}")
top_m = d.get('top_m_nlls', {})
sparse_key = min((k for k, v in top_m.items() if v is not None),
                  key=lambda k: (top_m[k] or 999), default=None)
if sparse_key:
    print(f"    best sparse top-M:   top{sparse_key}  Q={top_m[sparse_key]:.4f}")
PYEOF
    fi
done

echo ""
echo "[done] Reports saved to:"
echo "  ${TS_OUTPUT}/reports/dime_efficiency_*.{csv,md,json}"
echo "  ${WT2_OUTPUT}/reports/dime_efficiency_*.{csv,md,json}"

#!/bin/bash
#SBATCH --job-name=track_a_wikitext2
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/RegionTokenizer/scripts/RL/logs/track_a_wikitext2-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/RegionTokenizer/scripts/RL/logs/track_a_wikitext2-%j.err
#
# Track A: Cross-Dataset Replication on WikiText-2
# Runs the full extract → build → Q-read pipeline on WikiText-2 raw.
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
MODE="${1:-smoke}"   # smoke | full | full200k
SCRIPT="src/paper/run_dataset_replication.py"

echo "========================================================"
echo " Track A: WikiText-2 Replication  [mode=$MODE]"
echo " $(date)"
echo "========================================================"
echo ""

# ── Preflight ─────────────────────────────────────────────────────────────────
echo "[preflight] Checking environment..."
[ ! -f "$SCRIPT" ] && { echo "ERROR: $SCRIPT not found"; exit 1; }
echo "  [OK] $SCRIPT"

python -m py_compile "$SCRIPT"
echo "  [OK] $SCRIPT compiles"

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

# ── Launch ────────────────────────────────────────────────────────────────────
if [ "$MODE" = "smoke" ]; then
    OUTPUT="outputs_track_a_offline_paper_wikitext2_smoke"
    echo "[launch] SMOKE TEST (seed=42, budgets 5k/10k, 200k Q samples)"
    echo ""
    python "$SCRIPT" \
        --dataset       wikitext2_raw \
        --model_name    gpt2 \
        --output        "$OUTPUT" \
        --datastore_size 100000 \
        --seeds         42 \
        --budgets       5000 10000 \
        --methods       minibatch_kmeans utility_weighted \
        --raw_baselines raw_random cluster_medoids raw_high_gpt_entropy raw_high_gpt_loss \
        --max_q_samples 200000 \
        --fast_grid \
        --device        cuda

elif [ "$MODE" = "full" ]; then
    OUTPUT="outputs_track_a_offline_paper_wikitext2"
    echo "[launch] FULL RUN (seeds 42/123/999, budgets 5k/10k/25k, 500k Q samples)"
    echo ""
    python "$SCRIPT" \
        --dataset       wikitext2_raw \
        --model_name    gpt2 \
        --output        "$OUTPUT" \
        --datastore_size 100000 \
        --seeds         42 123 999 \
        --budgets       5000 10000 25000 \
        --methods       minibatch_kmeans utility_weighted query_kmeans \
        --raw_baselines raw_random cluster_medoids raw_high_gpt_entropy raw_high_gpt_loss raw_coverage \
        --max_q_samples 500000 \
        --fast_grid \
        --device        cuda

elif [ "$MODE" = "full200k" ]; then
    OUTPUT="outputs_track_a_offline_paper_wikitext2_200k"
    echo "[launch] 200k DATASTORE RUN (seed=42, budgets 10k/25k/50k)"
    echo ""
    python "$SCRIPT" \
        --dataset       wikitext2_raw \
        --model_name    gpt2 \
        --output        "$OUTPUT" \
        --datastore_size 200000 \
        --seeds         42 \
        --budgets       10000 25000 50000 \
        --methods       minibatch_kmeans utility_weighted query_kmeans \
        --raw_baselines raw_random cluster_medoids raw_high_gpt_entropy raw_high_gpt_loss raw_coverage \
        --max_q_samples 500000 \
        --fast_grid \
        --device        cuda

else
    echo "ERROR: unknown MODE=$MODE  (use: smoke | full | full200k)"
    exit 1
fi

EXIT=$?
if [ $EXIT -ne 0 ]; then
    echo ""; echo "ERROR: pipeline exited with code $EXIT"; exit $EXIT
fi

echo ""
echo "========================================================"
echo " WikiText-2 complete  [mode=$MODE]  $(date)"
echo "========================================================"
echo ""

# ── Print final verdict ───────────────────────────────────────────────────────
python - <<PYEOF
import os, re, json, glob

output = "${OUTPUT}"
# Per-seed reports
reports = sorted(glob.glob(f"{output}/seed*/reports/TRACK_A_OFFLINE_PAPER_REPORT.md"))
if reports:
    rpt = reports[0]
    text = open(rpt).read()
    m = re.search(r"\*\*(GREEN|YELLOW|RED)[^*]*\*\*", text)
    if m:
        print(f"\n[per-seed verdict] {m.group(0)}")

# Cross-dataset comparison
cmp = f"{output}/reports/TINYSTORIES_VS_WIKITEXT2_COMPARISON.md"
if os.path.isfile(cmp):
    print(f"\n[comparison report]\n")
    for line in open(cmp).readlines()[:30]:
        print(f"  {line}", end="")

# Replication verdict
vp = f"{output}/reports/replication_verdict.json"
if os.path.isfile(vp):
    v = json.load(open(vp))
    print(f"\n[replication verdict] {v.get('verdict')}")
    print(f"  GPT NLL:          {v.get('gpt_nll')}")
    print(f"  Best state Q NLL: {v.get('best_state_q_nll')}")
    print(f"  Win rate vs raw:  {v.get('state_win_rate_vs_raw')}")
PYEOF

echo ""
echo "[done] Outputs: ${OUTPUT}/"
echo "  Per-seed:    ${OUTPUT}/seed*/reports/"
echo "  Comparison:  ${OUTPUT}/reports/TINYSTORIES_VS_WIKITEXT2_COMPARISON.md"
echo "  Verdict:     ${OUTPUT}/reports/replication_verdict.json"

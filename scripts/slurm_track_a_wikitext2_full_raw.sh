#!/bin/bash
#SBATCH --job-name=track_a_wt2_fullraw
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/RegionTokenizer/scripts/RL/logs/track_a_wt2_fullraw-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/RegionTokenizer/scripts/RL/logs/track_a_wt2_fullraw-%j.err
#
# Track A: WikiText-2 Full-Raw Datastore Q-read + Re-aggregate
# Skips all extraction/build/Q-read stages (already done).
# Runs: Stage 4c (full raw Q-read, N=100k) + aggregate per seed.
# Seeds 42/123/999, 500k Q samples.
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
OUTPUT="outputs_track_a_offline_paper_wikitext2"
SCRIPT="src/paper/run_dataset_replication.py"

echo "========================================================"
echo " Track A: WikiText-2 Full-Raw Baseline"
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
print(f'  PyTorch: {torch.__version__}')
if not torch.cuda.is_available():
    print('  ERROR: CUDA not available'); sys.exit(1)
print(f'  GPU:  {torch.cuda.get_device_name(0)}')
print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
"

# Confirm previous run's Q-read outputs exist
echo ""
echo "[preflight] Checking prior run outputs..."
for seed in 42 123 999; do
    DIR="${OUTPUT}/seed${seed}"
    if [ -d "$DIR/q_read/states" ]; then
        echo "  [OK] seed${seed} Q-read states found"
    else
        echo "  [WARN] seed${seed}: $DIR/q_read/states not found — full_raw_only mode will still proceed"
    fi
done

echo ""
echo "[preflight] All checks passed."
echo ""

# ── Clear stale sentinels so full_raw + aggregate re-run ──────────────────────
echo "[setup] Clearing full_raw state files, sentinels, and aggregate sentinels..."
for seed in 42 123 999; do
    SENT_DIR="${OUTPUT}/sentinels"
    SEED_DIR="${OUTPUT}/seed${seed}"
    # Remove the bad fp16 state file so it gets rebuilt as fp32
    rm -f "${SEED_DIR}/full_raw_states/full_raw_datastore_B"*.pt
    # Remove full_raw Q-read output dir so evaluate_q_read reruns
    rm -rf "${SEED_DIR}/q_read/full_raw"
    rm -f "${SENT_DIR}/seed${seed}_full_raw.done"
    rm -f "${SENT_DIR}/seed${seed}_aggregate.done"
    echo "  Cleared seed${seed}"
done
echo ""

# ── Launch ────────────────────────────────────────────────────────────────────
echo "[launch] Full-raw Q-read + re-aggregate (seeds 42/123/999)"
echo ""

python "$SCRIPT" \
    --dataset        wikitext2_raw \
    --model_name     gpt2 \
    --output         "$OUTPUT" \
    --datastore_size 100000 \
    --seeds          42 123 999 \
    --max_q_samples  500000 \
    --run_full_raw \
    --full_raw_only \
    --fast_grid \
    --device         cuda

EXIT=$?
if [ $EXIT -ne 0 ]; then
    echo ""; echo "ERROR: pipeline exited with code $EXIT"; exit $EXIT
fi

echo ""
echo "========================================================"
echo " Full-raw run complete.  $(date)"
echo "========================================================"
echo ""

# ── Print final verdict ───────────────────────────────────────────────────────
python - <<PYEOF
import os, json, glob

output = "${OUTPUT}"

# Replication verdict (cross-seed summary)
vp = f"{output}/reports/replication_verdict.json"
if os.path.isfile(vp):
    v = json.load(open(vp))
    print(f"[replication verdict]  {v.get('verdict')}")
    print(f"  GPT NLL:             {v.get('gpt_nll')}")
    print(f"  Best state Q NLL:    {v.get('best_state_q_nll')}")
    print(f"  Full raw Q NLL:      {v.get('full_raw_q_nll')}")
    print(f"  Full raw fixed NLL:  {v.get('full_raw_fixed_nll')}")
    print(f"  Win rate vs raw:     {v.get('state_win_rate_vs_raw')}")
else:
    print("[WARN] replication_verdict.json not found")

# Per-seed full_raw results
print("")
for seed in [42, 123, 999]:
    fr_path = f"{output}/seed{seed}/q_read/full_raw/evaluate_q_read_summary.json"
    if os.path.isfile(fr_path):
        fr = json.load(open(fr_path))
        rr = fr.get("results", [{}])[0] if fr.get("results") else {}
        print(f"  Seed {seed} full_raw:  Q={rr.get('q_state_nll')}  fixed={rr.get('best_fixed_nll')}  oracle={rr.get('oracle_nll')}")
    else:
        print(f"  Seed {seed} full_raw:  [not found]")

# Comparison report
cmp = f"{output}/reports/TINYSTORIES_VS_WIKITEXT2_COMPARISON.md"
if os.path.isfile(cmp):
    print(f"\n[comparison report]")
    for line in open(cmp).readlines()[:35]:
        print(f"  {line}", end="")
PYEOF

echo ""
echo "[done] Outputs: ${OUTPUT}/"
echo "  Comparison: ${OUTPUT}/reports/TINYSTORIES_VS_WIKITEXT2_COMPARISON.md"
echo "  Verdict:    ${OUTPUT}/reports/replication_verdict.json"
echo "  Full-raw:   ${OUTPUT}/seed{42,123,999}/q_read/full_raw/"

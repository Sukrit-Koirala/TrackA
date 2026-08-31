#!/bin/bash
#SBATCH --job-name=track_a_seed999
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/RegionTokenizer/scripts/RL/logs/track_a_seed999-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/RegionTokenizer/scripts/RL/logs/track_a_seed999-%j.err
#
# Track A: Offline Predictive State Codebook Paper
# Full experiment — seeds 42/123/999, budgets 1k-50k, full grid, 800k Q samples
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

# ── Path configuration ────────────────────────────────────────────────────────
SOURCE="scale_200k_seed42"
OUTPUT="outputs_track_a_offline_paper"
SCRIPT="src/paper/run_track_a_offline_paper.py"

echo "========================================================"
echo " Track A: Offline Predictive State Codebook Paper"
echo " source:  ${SOURCE}"
echo " output:  ${OUTPUT}"
echo " $(date)"
echo "========================================================"
echo ""

# ── Preflight checks ──────────────────────────────────────────────────────────
echo "[preflight] Checking required inputs..."

[ ! -f "$SCRIPT" ] && { echo "ERROR: $SCRIPT not found"; exit 1; }
echo "  [OK] $SCRIPT"

[ ! -f "${SOURCE}/states/datastore.pt" ] && { echo "ERROR: ${SOURCE}/states/datastore.pt not found"; exit 1; }
echo "  [OK] ${SOURCE}/states/datastore.pt"

[ ! -f "${SOURCE}/states/controller_train.pt" ] && { echo "ERROR: ${SOURCE}/states/controller_train.pt not found"; exit 1; }
echo "  [OK] ${SOURCE}/states/controller_train.pt"

[ ! -f "${SOURCE}/states/val.pt" ] && { echo "ERROR: ${SOURCE}/states/val.pt not found"; exit 1; }
echo "  [OK] ${SOURCE}/states/val.pt"

echo ""
echo "[preflight] Checking syntax..."
python -m py_compile "$SCRIPT"
echo "  [OK] $SCRIPT compiles"

echo ""
echo "[preflight] Checking CUDA..."
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

mkdir -p "$OUTPUT"

echo "[launch] Starting Track A offline paper experiment..."
echo ""

python "$SCRIPT" \
    --source         "$SOURCE" \
    --output         "$OUTPUT" \
    --seeds          42 123 999 \
    --budgets        1000 5000 10000 25000 50000 \
    --methods        minibatch_kmeans utility_weighted query_kmeans \
    --raw_baselines  raw_random raw_high_gpt_loss raw_high_gpt_entropy \
                     raw_token_rarity raw_coverage cluster_medoids \
    --run_seed_sweep \
    --run_equal_budget \
    --run_efficiency \
    --run_state_ablations \
    --run_q_ablation \
    --run_diagnostics \
    --max_q_samples  800000 \
    --device         cuda

EXIT=$?
if [ $EXIT -ne 0 ]; then
    echo ""; echo "ERROR: pipeline exited with code $EXIT"; exit $EXIT
fi

echo ""
echo "========================================================"
echo " Track A complete. $(date)"
echo "========================================================"
echo ""
echo "Key outputs (per-seed):"
for SEED in 42 123 999; do
    SREP="${OUTPUT}/seed${SEED}/reports"
    [ -d "$SREP" ] && echo "  ${SREP}/TRACK_A_OFFLINE_PAPER_REPORT.md"
done
echo ""

# ── Print final verdict ───────────────────────────────────────────────────────
python - <<PYEOF
import os, re, glob

# With multiple seeds, reports are per-seed; fall back to top-level if present
candidates = (
    glob.glob("${OUTPUT}/seed*/reports/TRACK_A_OFFLINE_PAPER_REPORT.md")
    + ["${OUTPUT}/reports/TRACK_A_OFFLINE_PAPER_REPORT.md"]
)
rpt = next((p for p in candidates if os.path.isfile(p)), None)
if not rpt:
    print("[WARN] No report found"); exit(0)

text = open(rpt).read()

m = re.search(r"\*\*(GREEN|YELLOW|RED)[^*]*\*\*", text)
if m: print(f"\n[result] VERDICT: {m.group(0)}\n")

for line in text.split("\n"):
    if any(k in line for k in ["beat GPT", "beat equal-budget", "Best Q NLL", "Paper readiness"]):
        print(f"  {line.strip()}")
PYEOF

echo ""
echo "[done] Reports: ${OUTPUT}/seed{42,123,999}/reports/"

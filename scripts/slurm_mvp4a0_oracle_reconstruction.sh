#!/bin/bash
#SBATCH --job-name=mvp4a0_oracle
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/RegionTokenizer/scripts/RL/logs/mvp4a0_oracle-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/RegionTokenizer/scripts/RL/logs/mvp4a0_oracle-%j.err

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
SOURCE_DIR="scale_200k_seed42"
STATES_DIR="outputs_track_a_offline_paper/seed42/states"
QREAD_DIR="outputs_track_a_offline_paper/seed42/q_read/states"
OUTPUT_DIR="outputs_mvp4a0_oracle_reconstruction"
BUDGET=10000
PROMOTION_SUPPORT=8
SEED=42
DEVICE=cuda

# Allow overrides from env
SOURCE_DIR="${ORACLE_SOURCE_DIR:-$SOURCE_DIR}"
STATES_DIR="${ORACLE_STATES_DIR:-$STATES_DIR}"
QREAD_DIR="${ORACLE_QREAD_DIR:-$QREAD_DIR}"
OUTPUT_DIR="${ORACLE_OUTPUT_DIR:-$OUTPUT_DIR}"
BUDGET="${ORACLE_BUDGET:-$BUDGET}"
PROMOTION_SUPPORT="${ORACLE_PROMOTION_SUPPORT:-$PROMOTION_SUPPORT}"
SEED="${ORACLE_SEED:-$SEED}"

echo "========================================================"
echo " MVP 4a-0: Oracle Sequential Reconstruction"
echo " output:  ${OUTPUT_DIR}"
echo " $(date)"
echo "========================================================"
echo ""

# ── Preflight ─────────────────────────────────────────────────────────────────
echo "[preflight] Checking environment..."

python -c "
import torch, sys
print(f'  PyTorch: {torch.__version__}')
if not torch.cuda.is_available():
    print('  ERROR: CUDA not available'); sys.exit(1)
print(f'  GPU:  {torch.cuda.get_device_name(0)}')
print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
"

[ ! -d "$SOURCE_DIR" ] && { echo "ERROR: source dir not found: $SOURCE_DIR"; exit 1; }
[ ! -d "$STATES_DIR" ] && { echo "ERROR: offline states dir not found: $STATES_DIR"; exit 1; }
[ ! -d "$QREAD_DIR"  ] && { echo "ERROR: offline Q-read dir not found: $QREAD_DIR"; exit 1; }
[ ! -f "$SOURCE_DIR/states/datastore.pt" ] && { echo "ERROR: datastore not found"; exit 1; }
[ ! -f "$STATES_DIR/minibatch_kmeans_B${BUDGET}.pt" ] && { echo "ERROR: teacher state not found"; exit 1; }

python -m py_compile src/run_mvp4a0_oracle_reconstruction.py \
    && echo "  [OK] orchestrator compiles" \
    || { echo "ERROR: orchestrator fails to compile"; exit 1; }

echo ""
echo "[preflight] All checks passed."
echo "  source:     $SOURCE_DIR"
echo "  states_dir: $STATES_DIR"
echo "  qread_dir:  $QREAD_DIR"
echo "  output:     $OUTPUT_DIR"
echo "  budget:     $BUDGET  promo_support: $PROMOTION_SUPPORT  seed: $SEED"
echo ""

# ── Run ───────────────────────────────────────────────────────────────────────
python src/run_mvp4a0_oracle_reconstruction.py \
    --source              "$SOURCE_DIR" \
    --offline_states_dir  "$STATES_DIR" \
    --offline_qread_dir   "$QREAD_DIR" \
    --output              "$OUTPUT_DIR" \
    --teacher             minibatch_kmeans \
    --budget              "$BUDGET" \
    --promotion_support   "$PROMOTION_SUPPORT" \
    --seed                "$SEED" \
    --device              "$DEVICE" \
    --max_q_samples       800000 \
    --n_epochs            30

echo ""
echo "--- Job complete: $(date) ---"

# ── Quick results ─────────────────────────────────────────────────────────────
echo ""
echo "=== QUICK RESULTS ==="
python - <<'PYEOF'
import json, sys
from pathlib import Path

out = Path("outputs_mvp4a0_oracle_reconstruction")

vp = out / "teacher_assignments" / "assignment_validation.json"
if vp.exists():
    val = json.load(open(vp))
    print(f"Assignment validation: {'PASS' if val.get('PASS') else 'FAIL'}")
else:
    print("Assignment validation: NOT FOUND")

agg = {}
ap = out / "analysis" / "aggregate_state_metrics.json"
if ap.exists():
    agg = json.load(open(ap))

ed = out / "evaluation"
for mode in ["exact_teacher_prototype", "sequential_running_mean"]:
    mp = ed / f"{mode}_metrics.json"
    if mp.exists():
        m  = json.load(open(mp))
        qn = m.get("q_state_nll")
        fn = m.get("best_fixed_nll")
        on = m.get("oracle_nll")
        a  = agg.get(mode, {})
        cs = (a.get("proto_cos_sim") or {}).get("mean")
        kl = (a.get("kl_teacher_recon") or {}).get("mean")
        print(f"\n{mode}:")
        print(f"  Q-NLL={qn}  fixed={fn}  oracle={on}")
        print(f"  proto_cos_sim={cs}  kl={kl}")
    else:
        print(f"\n{mode}: evaluation not found")

recall_p = out / "match_diagnostics" / "target_recall.json"
if recall_p.exists():
    rc   = json.load(open(recall_p))
    comb = rc.get("combined", {})
    print("\nMATCH recall (combined):")
    for k in [1, 4, 8, 16]:
        v = comb.get(f"recall@{k}")
        if v is not None:
            print(f"  recall@{k}: {v:.4f}")

ref_q = 2.3015
tm    = ed / "teacher_reference_metrics.json"
if tm.exists():
    ref_q = json.load(open(tm)).get("q_state_nll", ref_q)

print(f"\n--- go/no-go (ref Q-NLL={ref_q:.4f}) ---")
for mode in ["exact_teacher_prototype", "sequential_running_mean"]:
    mp = ed / f"{mode}_metrics.json"
    if mp.exists():
        q = json.load(open(mp)).get("q_state_nll")
        if q is not None:
            gap = q - ref_q
            v   = "PASS" if abs(gap) <= 0.02 else ("WARNING" if abs(gap) <= 0.05 else "FAIL")
            print(f"  {mode[:28]}: gap={gap:+.4f}  --> {v}")
PYEOF

echo ""
echo "[done] Outputs: ${OUTPUT_DIR}/"

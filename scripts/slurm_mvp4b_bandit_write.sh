#!/bin/bash
#SBATCH --job-name=mvp4b_bandit
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/RegionTokenizer/scripts/RL/logs/mvp4b_bandit-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/RegionTokenizer/scripts/RL/logs/mvp4b_bandit-%j.err

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
QREAD_DIR="outputs_track_a_offline_paper/seed42/q_read/states"
ORACLE_DIR="outputs_mvp4a0_oracle_reconstruction"
OUTPUT_DIR="outputs_mvp4b_bandit_write_fast"
BUDGET=10000
PROMOTION_SUPPORT=8
SEED=42
SCORER="${BANDIT_SCORER:-mlp}"
N_DP="${BANDIT_N_DP:-10000}"
DEVICE=cuda

# Allow env overrides
SOURCE_DIR="${BANDIT_SOURCE_DIR:-$SOURCE_DIR}"
QREAD_DIR="${BANDIT_QREAD_DIR:-$QREAD_DIR}"
ORACLE_DIR="${BANDIT_ORACLE_DIR:-$ORACLE_DIR}"
OUTPUT_DIR="${BANDIT_OUTPUT_DIR:-$OUTPUT_DIR}"
BUDGET="${BANDIT_BUDGET:-$BUDGET}"
SEED="${BANDIT_SEED:-$SEED}"

echo "========================================================"
echo " MVP 4b: Teacher-Free Bandit-WRITE"
echo " source:     ${SOURCE_DIR}"
echo " oracle_dir: ${ORACLE_DIR}"
echo " output:     ${OUTPUT_DIR}"
echo " scorer:     ${SCORER}  n_dp: ${N_DP}"
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

[ ! -d "$SOURCE_DIR" ]  && { echo "ERROR: source dir not found: $SOURCE_DIR"; exit 1; }
[ ! -d "$QREAD_DIR"  ]  && { echo "ERROR: Q-read dir not found: $QREAD_DIR"; exit 1; }
[ ! -d "$ORACLE_DIR" ]  && { echo "ERROR: oracle dir not found: $ORACLE_DIR"; exit 1; }
[ ! -f "$SOURCE_DIR/states/datastore.pt" ] && { echo "ERROR: datastore not found"; exit 1; }
[ ! -f "$ORACLE_DIR/teacher_assignments/minibatch_kmeans_B${BUDGET}_assignments.npz" ] \
    && { echo "ERROR: teacher assignments not found in oracle_dir"; exit 1; }

python -m py_compile src/run_mvp4b_bandit_write.py \
    && echo "  [OK] orchestrator compiles" \
    || { echo "ERROR: orchestrator fails to compile"; exit 1; }

echo ""
echo "[preflight] All checks passed."
echo ""

# ── Run ───────────────────────────────────────────────────────────────────────
python src/run_mvp4b_bandit_write.py \
    --source              "$SOURCE_DIR" \
    --offline_qread_dir   "$QREAD_DIR" \
    --oracle_dir          "$ORACLE_DIR" \
    --output              "$OUTPUT_DIR" \
    --budget              "$BUDGET" \
    --scorer              "$SCORER" \
    --n_decision_points   "$N_DP" \
    --k_states            8 \
    --k_buffers           8 \
    --n_local_probes      16 \
    --n_history_probes    16 \
    --n_replay_probes     16 \
    --max_exemplars       16 \
    --promotion_support   "$PROMOTION_SUPPORT" \
    --state_budget        "$BUDGET" \
    --buffer_budget       "$BUDGET" \
    --mlp_epochs          50 \
    --gbm_n_estimators    400 \
    --max_q_samples       800000 \
    --n_epochs            30 \
    --reward_col          reward_penalized \
    --seed                "$SEED" \
    --device              "$DEVICE"

echo ""
echo "--- Job complete: $(date) ---"

# ── Quick results ─────────────────────────────────────────────────────────────
echo ""
echo "=== QUICK RESULTS ==="
python - <<'PYEOF'
import json, sys
from pathlib import Path

out = Path("outputs_mvp4b_bandit_write_fast")

audit = {}
ap = out / "reward_audit" / "audit_report.json"
if ap.exists():
    audit = json.load(open(ap))
    print(f"Reward audit: {audit.get('verdict','—')}")
    for g, v in audit.get("gates", {}).items():
        print(f"  {'PASS' if v else 'FAIL'}  {g}")

scorer_eval = {}
sep = out / "scorer_eval" / "scorer_eval.json"
if sep.exists():
    scorer_eval = json.load(open(sep))
    print(f"\nScorer evaluation: {scorer_eval.get('verdict','—')}")
    for m, r in scorer_eval.get("models", {}).items():
        reg = r.get("regret", {})
        print(f"  {m.upper()}: "
              f"Spearman={r.get('spearman_rho','—'):.4f}  "
              f"MAE={r.get('mae','—'):.5f}  "
              f"regret={reg.get('mean_regret','—'):.5f}  "
              f"action_match={reg.get('action_match_frac','—'):.3f}")

roll = {}
rp = out / "rollout" / "rollout_stats.json"
if rp.exists():
    roll = json.load(open(rp))
    print(f"\nRollout:")
    print(f"  Objects: {roll.get('n_total_objects','—')}  "
          f"(persistent={roll.get('n_persistent','—')}  "
          f"buffers={roll.get('n_buffers','—')})")
    for aname, cnt in roll.get("action_counts", {}).items():
        frac = roll.get("action_fracs", {}).get(aname, 0)
        print(f"  {aname:<22} {cnt:>8,} ({frac*100:5.1f}%)")

scorer = roll.get("scorer", "mlp")
method = f"bandit_write_{scorer}"
budget = roll.get("budget", 10000)

# Try evaluation metrics
for path in [
    out / "evaluation" / f"{method}_metrics.json",
    out / "evaluation" / f"{method}_B{budget}" / "q_metrics.json",
]:
    if path.exists():
        m = json.load(open(path))
        print(f"\nQ-read results ({method}):")
        print(f"  Q-NLL={m.get('q_state_nll','—')}  "
              f"fixed={m.get('best_fixed_nll','—')}  "
              f"oracle={m.get('oracle_nll','—')}")
        print(f"  n_states={m.get('n_states','—')}")
        ref_q = 2.4204  # MVP3b baseline
        q = m.get("q_state_nll")
        if q is not None:
            gap = q - ref_q
            verdict = "PASS" if gap < 0 else "FAIL"
            print(f"\n  vs MVP3b (2.4204): gap={gap:+.4f}  --> {verdict}")
        break

analysis = {}
an_path = out / "analysis" / "analysis_complete.json"
if an_path.exists():
    analysis = json.load(open(an_path))
    print(f"\nFinal verdict: {analysis.get('verdict','—')}")
PYEOF

echo ""
echo "[done] Outputs: ${OUTPUT_DIR}/"

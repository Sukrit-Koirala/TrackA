#!/bin/bash
#SBATCH --job-name=track_a_abl_wt2
#SBATCH --partition=gpuGeneral
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=32:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=/home/sukrit.koirala/ondemand/upload_me/RegionTokenizer/scripts/RL/logs/track_a_abl_wt2-%j.out
#SBATCH --error=/home/sukrit.koirala/ondemand/upload_me/RegionTokenizer/scripts/RL/logs/track_a_abl_wt2-%j.err
#
# Track A: State-Object Ablations — WikiText-2 / GPT-2 small
# All 11 variants, seeds 42/123/999, budgets 5k/10k/25k, 200k Q samples
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
OUTPUT_BASE="outputs_track_a_offline_paper_wikitext2"
SCRIPT="src/paper/run_state_object_ablations.py"
SEEDS=(42 123 999)
METHODS="minibatch_kmeans utility_weighted query_kmeans"
BUDGETS="5000 10000 25000"
VARIANTS="original majority_token top4 top8 top16 top32 top64 shuffled_distribution shuffled_prototype global_unigram random_partition"
MAX_Q=200000

echo "========================================================"
echo " Track A: State-Object Ablations (WikiText-2 / GPT-2 small)"
echo " output:  ${OUTPUT_BASE}"
echo " seeds:   ${SEEDS[*]}"
echo " $(date)"
echo "========================================================"
echo ""

# ── Preflight ─────────────────────────────────────────────────────────────────
echo "[preflight] Checking environment..."
[ ! -f "$SCRIPT" ] && { echo "ERROR: $SCRIPT not found"; exit 1; }

python -m py_compile "$SCRIPT" && echo "  [OK] ablation script compiles" \
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

# ── Run ablations per seed ────────────────────────────────────────────────────
for SEED in "${SEEDS[@]}"; do
    SEED_DIR="${OUTPUT_BASE}/seed${SEED}"

    echo "------------------------------------------------------------"
    echo " Seed ${SEED}  →  ${SEED_DIR}"
    echo " $(date)"
    echo "------------------------------------------------------------"

    if [ ! -d "${SEED_DIR}/states" ]; then
        echo "  [WARN] ${SEED_DIR}/states not found — skipping seed ${SEED}"
        continue
    fi

    python "$SCRIPT" \
        --source       "${SEED_DIR}" \
        --output       "${SEED_DIR}" \
        --methods      $METHODS \
        --budgets      $BUDGETS \
        --variants     $VARIANTS \
        --dataset      WikiText-2 \
        --model        gpt2 \
        --max_q_samples "$MAX_Q" \
        --fast_grid \
        --device       cuda \
        --seed         "$SEED" \
        --allow_missing

    EXIT=$?
    if [ $EXIT -ne 0 ]; then
        echo "  WARNING: seed ${SEED} exited with code ${EXIT}"
    else
        echo "  Seed ${SEED} complete."
    fi
    echo ""
done

echo "========================================================"
echo " All seeds done.  $(date)"
echo "========================================================"
echo ""

# ── Aggregate across seeds ────────────────────────────────────────────────────
echo "[aggregate] Collecting per-seed results..."

python - <<'PYEOF'
import json, math, os, sys
from pathlib import Path
import pandas as pd

base = Path("outputs_track_a_offline_paper_wikitext2")
seeds = [42, 123, 999]
rows = []

for s in seeds:
    csv_path = base / f"seed{s}" / "ablations" / "state_object_ablation_results.csv"
    if csv_path.exists():
        df_s = pd.read_csv(csv_path)
        rows.append(df_s)
        print(f"  Loaded seed{s}: {len(df_s)} rows")
    else:
        print(f"  [WARN] missing: {csv_path}")

if not rows:
    print("  No per-seed results found."); sys.exit(0)

df = pd.concat(rows, ignore_index=True)

rep = base / "reports"
rep.mkdir(parents=True, exist_ok=True)
out_csv = rep / "ablations_wikitext2_all_seeds.csv"
df.to_csv(out_csv, index=False)
print(f"\n  Saved: {out_csv}  ({len(df)} rows total)")

agg = df.groupby("variant")["q_nll"].agg(["mean", "std", "count"]).reset_index()
agg = agg.sort_values("mean")
print("\n  Cross-seed variant Q NLL:")
print(f"  {'Variant':<28}  {'Mean Q NLL':>10}  {'Std':>8}  {'N':>5}")
print(f"  {'-'*55}")
for _, r in agg.iterrows():
    print(f"  {r['variant']:<28}  {r['mean']:>10.4f}  {r['std']:>8.4f}  {int(r['count']):>5}")

orig = agg[agg["variant"] == "original"]["mean"].values
if len(orig) > 0:
    orig_q = float(orig[0])
    def _q(v):
        r = agg[agg["variant"] == v]["mean"].values
        return float(r[0]) if len(r) > 0 else float("nan")

    checks = {
        "majority_worse":       _q("majority_token") > orig_q,
        "shuffled_dist_worse":  _q("shuffled_distribution") > orig_q,
        "shuffled_proto_worse": _q("shuffled_prototype") > orig_q,
        "top32_close":          (_q("top32") - orig_q) <= 0.02,
        "random_part_worse":    _q("random_partition") > orig_q,
    }
    n_pass = sum(v for v in checks.values() if isinstance(v, bool))
    passed = [k for k, v in checks.items() if v]
    failed = [k for k, v in checks.items() if not v]

    if not checks.get("shuffled_dist_worse") or not checks.get("majority_worse"):
        verdict = "FAIL"
    elif n_pass == 5:
        verdict = "STRONG_SUPPORT"
    elif n_pass >= 3:
        verdict = "SUPPORT"
    elif n_pass >= 2:
        verdict = "MIXED"
    else:
        verdict = "FAIL"

    print(f"\n  Checks passed: {passed}")
    print(f"  Checks failed: {failed}")
    print(f"\n  Cross-seed ablation verdict: {verdict}")

    with open(rep / "ablations_wikitext2_cross_seed_verdict.json", "w") as f:
        json.dump({
            "dataset": "WikiText-2",
            "model": "gpt2",
            "seeds": seeds,
            "verdict": verdict,
            "checks": {k: bool(v) for k, v in checks.items()},
            "original_q": orig_q,
            "majority_token_q": _q("majority_token"),
            "top32_q": _q("top32"),
            "shuffled_distribution_q": _q("shuffled_distribution"),
            "shuffled_prototype_q": _q("shuffled_prototype"),
            "random_partition_q": _q("random_partition"),
        }, f, indent=2)
    print(f"  Saved: {rep}/ablations_wikitext2_cross_seed_verdict.json")
PYEOF

echo ""
echo "[done] Outputs: ${OUTPUT_BASE}/"
echo "  Per-seed: ${OUTPUT_BASE}/seed{42,123,999}/ablations/"
echo "  Reports:  ${OUTPUT_BASE}/reports/ablations_wikitext2_*.{csv,json}"

"""
evaluate_mvp4b1_states.py  --  MVP 4b.1 Stage 6

Evaluates Q-read NLL for all four rollout states:
  mvp4b1_{scorer}_vA_s123_B{budget}.pt
  mvp4b1_{scorer}_vB_s123_B{budget}.pt
  mvp4b1_{scorer}_vA_s999_B{budget}.pt
  mvp4b1_{scorer}_vB_s999_B{budget}.pt

Uses run_method_budget from train_q_state_read.py.

Usage:
  python src/evaluate_mvp4b1_states.py \\
    --source scale_200k_seed42 \\
    --offline_qread_dir outputs_track_a_offline_paper/seed42/q_read/states \\
    --output outputs_mvp4b1_onpolicy_bandit_write_clean_fast \\
    --scorer mlp --object_budget 10000 \\
    --max_q_samples 800000 --n_epochs 30 \\
    --seed 42 --device cuda
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch

from mvp4b1_common import (
    ARTIFACT_VERSION,
    atomic_write_json, validate_json_cache,
)


ROLLOUT_SEEDS    = [123, 999]
ROLLOUT_VARIANTS = ["A", "B"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",             required=True)
    ap.add_argument("--offline_qread_dir",  required=True)
    ap.add_argument("--output",             required=True)
    ap.add_argument("--scorer",             default="mlp")
    ap.add_argument("--object_budget",      type=int, default=10000)
    ap.add_argument("--max_q_samples",      type=int, default=800_000)
    ap.add_argument("--n_epochs",           type=int, default=30)
    ap.add_argument("--fast",               action="store_true")
    ap.add_argument("--seed",               type=int, default=42)
    ap.add_argument("--device",             default="cuda")
    ap.add_argument("--force",              action="store_true")
    args = ap.parse_args()

    from train_q_state_read import run_method_budget

    out_dir    = Path(args.output)
    states_dir = out_dir / "rollout" / "states"
    eval_dir   = out_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    sentinel = eval_dir / "eval_summary.json"
    cached   = validate_json_cache(sentinel, ["artifact_version", "results"])
    if cached and not args.force:
        print(f"[cached] {sentinel}")
        return

    B      = args.object_budget
    source = Path(args.source)

    results = {}
    tags    = [
        f"mvp4b1_{args.scorer}_v{v}_s{s}"
        for s in ROLLOUT_SEEDS
        for v in ROLLOUT_VARIANTS
    ]

    for tag in tags:
        state_path = states_dir / f"{tag}_B{B}.pt"
        if not state_path.exists():
            print(f"  WARNING: missing {state_path}, skipping")
            results[tag] = {"q_state_nll": None, "error": "state file missing"}
            continue

        print(f"\nEvaluating {tag} ...")
        out_tag_dir = eval_dir / f"{tag}_B{B}"
        out_tag_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = out_tag_dir / "q_metrics.json"

        if metrics_path.exists() and not args.force:
            import json
            m = {}
            try:
                with open(metrics_path) as f:
                    m = json.load(f)
                print(f"  [cached] Q-NLL = {m.get('q_state_nll', '?'):.4f}")
                results[tag] = m
                continue
            except Exception:
                pass

        max_q   = 100_000 if args.fast else args.max_q_samples
        n_ep    = 10      if args.fast else args.n_epochs

        try:
            m = run_method_budget(
                method=tag,
                budget=B,
                src=source,
                states_dir=states_dir,
                out=eval_dir,
                device=torch.device(args.device),
                max_q_samples=max_q,
                n_epochs=n_ep,
            )
            results[tag] = m
            q_nll = m.get("q_state_nll")
            print(f"  Q-NLL = {q_nll:.4f}" if q_nll else "  Q-NLL = ?")
        except Exception as e:
            print(f"  ERROR: {e}")
            results[tag] = {"q_state_nll": None, "error": str(e)}

    # Print comparison table
    print("\n" + "=" * 60)
    print("MVP 4b.1 Rollout Q-NLL Comparison")
    print("=" * 60)
    print(f"{'Tag':<45}  {'Q-NLL':>8}")
    for tag, m in results.items():
        q = m.get("q_state_nll")
        s = f"{q:.4f}" if q else "—"
        print(f"  {tag:<43}  {s:>8}")
    print("-" * 60)
    print(f"  {'GPT-only baseline':43}  {'2.5533':>8}")
    print(f"  {'MVP3b best':43}  {'2.4204':>8}")
    print(f"  {'Oracle teacher':43}  {'2.3026':>8}")
    print(f"  {'Full-raw-200k':43}  {'2.2977':>8}")

    # Best result
    best_tag, best_q = None, float("inf")
    for tag, m in results.items():
        q = m.get("q_state_nll")
        if q and q < best_q:
            best_q, best_tag = q, tag
    if best_tag:
        print(f"\n  Best: {best_tag}  Q-NLL={best_q:.4f}")

    summary = {
        "artifact_version": ARTIFACT_VERSION,
        "scorer":           args.scorer,
        "object_budget":    B,
        "results":          results,
        "best_tag":         best_tag,
        "best_q_nll":       best_q if best_q < float("inf") else None,
    }
    atomic_write_json(sentinel, summary)
    print(f"\nSaved: {sentinel}")


if __name__ == "__main__":
    main()

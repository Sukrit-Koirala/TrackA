"""
evaluate_bandit_write_states.py  --  MVP 4b Stage 3 evaluation

Evaluates the bandit-write rollout state artifacts using the same Q-read
pipeline (run_method_budget) as all other Track A methods.

Usage:
  python src/evaluate_bandit_write_states.py \\
    --source             scale_200k_seed42 \\
    --offline_qread_dir  outputs_track_a_offline_paper/seed42/q_read/states \\
    --output             outputs_mvp4b_bandit_write_fast \\
    --scorer             mlp \\
    --budget             10000 \\
    --seed               42 \\
    --device             cuda
"""

import sys
import json
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import pandas as pd

from utils import set_seed, get_device
from train_q_state_read import run_method_budget


def _load_json(p: Path) -> dict:
    return json.load(open(p)) if p.exists() else {}


def _fmt(m: dict | None, key: str) -> str:
    if m is None:
        return "—"
    v = m.get(key)
    return f"{v:.4f}" if v is not None else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",             required=True)
    ap.add_argument("--offline_qread_dir",  required=True)
    ap.add_argument("--output",             required=True)
    ap.add_argument("--scorer",             default="mlp",
                    choices=["mlp", "gbm"])
    ap.add_argument("--budget",             type=int, default=10000)
    ap.add_argument("--seed",               type=int, default=42)
    ap.add_argument("--device",             default="cuda")
    ap.add_argument("--fast",               action="store_true")
    ap.add_argument("--max_q_samples",      type=int, default=800_000)
    ap.add_argument("--n_epochs",           type=int, default=30)
    ap.add_argument("--force",              action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device({"device": args.device})

    out_dir    = Path(args.output)
    roll_dir   = out_dir / "rollout"
    states_dir = roll_dir / "states"
    eval_dir   = out_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    method_name = f"bandit_write_{args.scorer}"
    B           = args.budget
    tag         = f"{method_name}_B{B}"

    state_path = states_dir / f"{tag}.pt"
    if not state_path.exists():
        print(f"ERROR: bandit-write state not found: {state_path}")
        print("Run rollout_bandit_write.py first.")
        sys.exit(1)

    # ── check cache ────────────────────────────────────────────────────────────
    sentinel = eval_dir / tag / "q_metrics.json"
    if sentinel.exists() and not args.force:
        print(f"[cached] {sentinel}")
        m = _load_json(sentinel)
        print(f"  Q-NLL={_fmt(m,'q_state_nll')}  "
              f"fixed={_fmt(m,'best_fixed_nll')}  "
              f"oracle={_fmt(m,'oracle_nll')}")
        return

    src = Path(args.source)
    fast_grid = args.fast
    max_q = min(args.max_q_samples, 300_000) if fast_grid else args.max_q_samples

    print(f"Evaluating {method_name} ...")
    print(f"  states_dir: {states_dir}")
    print(f"  state file: {state_path}")

    m = run_method_budget(
        method          = method_name,
        budget          = B,
        src             = src,
        states_dir      = states_dir,
        out             = eval_dir,
        device          = device,
        max_q_samples   = max_q,
        n_epochs        = args.n_epochs,
        fast_grid       = fast_grid,
        force_neighbors = args.force,
        force_rewards   = args.force,
        force_train     = args.force,
        force_eval      = args.force,
    )

    if m is None:
        print("ERROR: evaluation returned None")
        sys.exit(1)

    # ── load reference metrics for comparison ─────────────────────────────────
    refs = {}
    # Oracle teacher reference
    oracle_tag = f"minibatch_kmeans_B{B}"
    oracle_m   = _load_json(Path(args.offline_qread_dir) / oracle_tag / "q_metrics.json")
    if oracle_m:
        refs["oracle_teacher"] = oracle_m.get("q_state_nll")

    # Save evaluation metrics
    import json as _json
    out_path = eval_dir / f"{method_name}_metrics.json"
    with open(out_path, "w") as f:
        _json.dump(m, f, indent=2, default=str)

    # ── print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Bandit-Write Q-read Evaluation")
    print(f"{'='*60}")
    print(f"  Method:      {method_name}")
    print(f"  Budget:      {B}")
    print(f"  Q-NLL:       {_fmt(m,'q_state_nll')}")
    print(f"  fixed NLL:   {_fmt(m,'best_fixed_nll')}")
    print(f"  oracle NLL:  {_fmt(m,'oracle_nll')}")
    print(f"  n_states:    {m.get('n_states','—')}")
    print(f"  GPT NLL:     {_fmt(m,'gpt_nll')}")

    if refs.get("oracle_teacher"):
        ref_q = refs["oracle_teacher"]
        q     = m.get("q_state_nll")
        if q is not None:
            gap = q - ref_q
            verdict = "PASS" if abs(gap) <= 0.05 else ("WARN" if abs(gap) <= 0.10 else "FAIL")
            print(f"\n  vs oracle teacher: gap={gap:+.4f}  --> {verdict}")

    # Reference comparisons table
    ref_rows = [
        {"memory": method_name, "q_nll": m.get("q_state_nll"),
         "fixed_nll": m.get("best_fixed_nll"), "n_states": m.get("n_states")},
        {"memory": "oracle_teacher",     "q_nll": refs.get("oracle_teacher"),
         "fixed_nll": None, "n_states": B},
        {"memory": "MVP3b_best",         "q_nll": 2.4204,
         "fixed_nll": None, "n_states": 519},
        {"memory": "full_raw_datastore", "q_nll": 2.2977,
         "fixed_nll": None, "n_states": 200000},
    ]
    cmp_path = eval_dir / f"{method_name}_comparison.csv"
    pd.DataFrame(ref_rows).to_csv(cmp_path, index=False)
    print(f"\nSaved comparison: {cmp_path}")
    print(f"Saved metrics:    {out_path}")


if __name__ == "__main__":
    main()

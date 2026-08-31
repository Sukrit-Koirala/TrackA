"""
evaluate_oracle_write_reconstruction.py  --  MVP 4a-0: Oracle Reconstruction

Evaluates oracle-reconstructed state memories using the EXISTING Q-state-read
pipeline (run_method_budget from train_q_state_read.py).

Evaluates:
  1. Teacher reference metrics (loaded from existing MVP 2c Q-read outputs)
  2. exact_teacher_prototype reconstruction
  3. sequential_running_mean reconstruction

All evaluation uses the same controller_train / val splits, same action grid,
same smoothing, same Q-MLP architecture as the original experiment.

Usage:
  python src/evaluate_oracle_write_reconstruction.py \\
    --source            outputs_scale_sweep/scale_200k_seed42 \\
    --offline_qread_dir outputs_mvp2c_q_state_read \\
    --output            outputs_mvp4a0_oracle_reconstruction \\
    --teacher           minibatch_kmeans \\
    --budget            10000 \\
    --seed              42 \\
    --device            cuda
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import pandas as pd
import torch

from utils import set_seed, get_device
from train_q_state_read import run_method_budget


MODES = [
    ("exact_teacher_prototype", "oracle_exact_{teacher}"),
    ("sequential_running_mean", "oracle_seq_{teacher}"),
]


def _load_json(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _fmt(m: dict | None, key: str) -> str:
    if m is None:
        return "—"
    v = m.get(key)
    return f"{v:.4f}" if v is not None else "—"


def evaluate_mode(
    mode:          str,
    method_name:   str,
    budget:        int,
    src:           Path,
    recon_dir:     Path,
    eval_out:      Path,
    device:        torch.device,
    fast_grid:     bool,
    max_q_samples: int,
    n_epochs:      int,
    force:         bool,
) -> dict | None:
    states_dir  = recon_dir / mode / "states"
    state_path  = states_dir / f"{method_name}_B{budget}.pt"
    if not state_path.exists():
        print(f"  [SKIP] state file not found: {state_path}")
        return None

    mode_out = eval_out / mode
    sentinel = mode_out / f"{method_name}_B{budget}" / "q_metrics.json"
    if sentinel.exists() and not force:
        print(f"  [cached] {mode}")
        return _load_json(sentinel)

    print(f"\n  Evaluating {mode} ...")
    return run_method_budget(
        method          = method_name,
        budget          = budget,
        src             = src,
        states_dir      = states_dir,
        out             = mode_out,
        device          = device,
        max_q_samples   = max_q_samples,
        n_epochs        = n_epochs,
        fast_grid       = fast_grid,
        force_neighbors = force,
        force_rewards   = force,
        force_train     = force,
        force_eval      = force,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",              required=True)
    ap.add_argument("--offline_states_dir",  default=None)
    ap.add_argument("--offline_qread_dir",   required=True,
                    help="outputs_mvp2c_q_state_read (for teacher reference metrics)")
    ap.add_argument("--output",              required=True)
    ap.add_argument("--teacher",             default="minibatch_kmeans")
    ap.add_argument("--budget",              type=int, default=10000)
    ap.add_argument("--seed",                type=int, default=42)
    ap.add_argument("--device",              default="cuda")
    ap.add_argument("--fast",                action="store_true",
                    help="Use 80-action fast grid")
    ap.add_argument("--max_q_samples",       type=int, default=800_000)
    ap.add_argument("--n_epochs",            type=int, default=30)
    ap.add_argument("--force",               action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device    = get_device({"device": args.device})
    src       = Path(args.source)
    out       = Path(args.output)
    B         = args.budget
    tag       = f"{args.teacher}_B{B}"
    fast_grid = args.fast
    max_q     = args.max_q_samples if not args.fast else min(args.max_q_samples, 300_000)

    recon_dir = out / "reconstructions"
    eval_dir  = out / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, dict | None] = {}

    # ── teacher reference ─────────────────────────────────────────────────────
    teacher_qread = Path(args.offline_qread_dir) / tag / "q_metrics.json"
    if teacher_qread.exists():
        teacher_m = _load_json(teacher_qread)
        metrics["teacher_reference"] = teacher_m
        dst = eval_dir / "teacher_reference_metrics.json"
        with open(dst, "w") as f:
            json.dump(teacher_m, f, indent=2)
        print(f"Teacher reference: Q-NLL={_fmt(teacher_m,'q_state_nll')}  "
              f"fixed={_fmt(teacher_m,'best_fixed_nll')}  "
              f"oracle={_fmt(teacher_m,'oracle_nll')}")
    else:
        print(f"WARNING: teacher Q-read not found: {teacher_qread}")
        metrics["teacher_reference"] = None

    # ── evaluate each reconstruction mode ─────────────────────────────────────
    for mode, method_tmpl in MODES:
        method_name = method_tmpl.format(teacher=args.teacher)
        m = evaluate_mode(
            mode, method_name, B, src, recon_dir, eval_dir,
            device, fast_grid, max_q, args.n_epochs, args.force,
        )
        metrics[mode] = m
        if m:
            dst = eval_dir / f"{mode}_metrics.json"
            with open(dst, "w") as f:
                json.dump(m, f, indent=2, default=str)
            print(f"  {mode}: Q-NLL={_fmt(m,'q_state_nll')}  "
                  f"fixed={_fmt(m,'best_fixed_nll')}  "
                  f"oracle={_fmt(m,'oracle_nll')}")

    # ── comparison CSV ────────────────────────────────────────────────────────
    ref_q = {
        "MVP3b_best": 2.4204,
        "full_raw":   2.2977,
        "gpt_only":   2.5533,
    }

    def _row(name, n, m: dict | None) -> dict:
        base = {"memory": name, "n_states": n,
                "fixed_nll": None, "q_nll": None, "oracle_nll": None,
                "gpt_nll": None, "avg_k_states": None, "retrieval_usage": None}
        if m:
            base.update({
                "n_states":       m.get("n_states", n),
                "fixed_nll":      m.get("best_fixed_nll"),
                "q_nll":          m.get("q_state_nll"),
                "oracle_nll":     m.get("oracle_nll"),
                "gpt_nll":        m.get("gpt_nll"),
                "avg_k_states":   m.get("avg_k_states"),
                "retrieval_usage": m.get("retrieval_usage"),
            })
        return base

    rows = [
        _row("teacher_reference",          B,      metrics.get("teacher_reference")),
        _row("exact_teacher_prototype",    B,      metrics.get("exact_teacher_prototype")),
        _row("sequential_running_mean",    B,      metrics.get("sequential_running_mean")),
        _row("MVP3b_best_reference",       519,    None),
        _row("full_raw_datastore",         200000, None),
    ]
    # Fill in known reference values
    for r in rows:
        if r["memory"] == "MVP3b_best_reference":
            r["q_nll"] = 2.4204
        if r["memory"] == "full_raw_datastore":
            r["q_nll"] = 2.2977

    df = pd.DataFrame(rows)
    cmp_path = eval_dir / "comparison.csv"
    df.to_csv(cmp_path, index=False)
    print(f"\nSaved: {cmp_path}")
    print(df[["memory", "n_states", "fixed_nll", "q_nll", "oracle_nll"]].to_string(index=False))


if __name__ == "__main__":
    main()

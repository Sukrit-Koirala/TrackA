"""
evaluate_q_read.py  --  Track A Paper

Runs Q-state-read evaluation (train_q_state_read.run_method_budget) on all
.pt files in a given states directory, saving results under <out_dir>.

Used for both state codebooks and raw memory baselines (both are in MVP 2b
state format).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse, json, time
import torch

from utils import get_device, set_seed
from train_q_state_read import run_method_budget


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",        required=True,
                        help="scale_200k_seed42 dir (for controller_train/val)")
    parser.add_argument("--states_dir",    required=True,
                        help="dir containing *_B*.pt state/raw-baseline files")
    parser.add_argument("--output",        required=True,
                        help="output dir for Q-read results")
    parser.add_argument("--methods",       nargs="+", default=None,
                        help="filter by method name (default: all)")
    parser.add_argument("--budgets",       nargs="+", type=int, default=None,
                        help="filter by budget (default: all)")
    parser.add_argument("--max_q_samples", type=int, default=800_000)
    parser.add_argument("--n_epochs",      type=int, default=30)
    parser.add_argument("--fast_grid",     action="store_true")
    parser.add_argument("--device",        default="cuda")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--force_neighbors", action="store_true")
    parser.add_argument("--force_rewards",   action="store_true")
    parser.add_argument("--force_train",     action="store_true")
    parser.add_argument("--force_eval",      action="store_true")
    parser.add_argument("--force",           action="store_true",
                        help="force all stages")
    args = parser.parse_args()

    if args.force:
        args.force_neighbors = args.force_rewards = \
            args.force_train = args.force_eval = True

    set_seed(args.seed)
    device     = get_device({"device": args.device})
    src        = Path(args.source)
    states_dir = Path(args.states_dir)
    out        = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Discover state files
    state_files = sorted(states_dir.glob("*.pt"))
    to_run = []
    for sfp in state_files:
        parts = sfp.stem.split("_B")
        if len(parts) < 2:
            continue
        method = "_B".join(parts[:-1])
        try:
            budget = int(parts[-1])
        except ValueError:
            continue
        if args.methods and method not in args.methods:
            continue
        if args.budgets and budget not in args.budgets:
            continue
        to_run.append((method, budget))

    if not to_run:
        print(f"[warn] No matching state files in {states_dir}")
        return

    print(f"\nevaluate_q_read")
    print(f"  States:       {states_dir}")
    print(f"  Output:       {out}")
    print(f"  To evaluate:  {len(to_run)} configs")
    print(f"  fast_grid:    {args.fast_grid}")
    print(f"  max_q_samples:{args.max_q_samples:,}")

    all_metrics = []
    failed      = []
    t0_all      = time.time()

    for method, budget in to_run:
        tag = f"{method}_B{budget}"
        print(f"\n{'='*60}")
        print(f"  {tag}")
        print(f"{'='*60}")
        try:
            m = run_method_budget(
                method=method,
                budget=budget,
                src=src,
                states_dir=states_dir,
                out=out,
                device=device,
                max_q_samples=args.max_q_samples,
                n_epochs=args.n_epochs,
                fast_grid=args.fast_grid,
                force_neighbors=args.force_neighbors,
                force_rewards=args.force_rewards,
                force_train=args.force_train,
                force_eval=args.force_eval,
            )
            if m:
                all_metrics.append(m)
                print(f"  fixed={m['best_fixed_nll']:.4f}  "
                      f"Q={m['q_state_nll']:.4f}  "
                      f"oracle={m['oracle_nll']:.4f}")
        except Exception as e:
            import traceback
            print(f"  [FAILED] {tag}: {e}")
            traceback.print_exc()
            failed.append(tag)

    elapsed = time.time() - t0_all
    print(f"\nDone  {len(all_metrics)}/{len(to_run)} succeeded  "
          f"elapsed={elapsed:.0f}s")
    if failed:
        print(f"Failed: {failed}")

    # Save summary JSON
    summary_path = out / "evaluate_q_read_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "n_success": len(all_metrics),
            "n_failed":  len(failed),
            "failed":    failed,
            "results":   all_metrics,
        }, f, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

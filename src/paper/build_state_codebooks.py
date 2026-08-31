"""
build_state_codebooks.py  --  Track A Paper

Thin wrapper around build_predictive_states.py.
Builds offline predictive state codebooks for multiple methods and budgets,
saving results to <output>/states/.

This is the same as the MVP 2b state-build step, redirected to the
paper output directory.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse, subprocess, time

DEFAULT_METHODS = [
    "minibatch_kmeans",
    "utility_weighted",
    "query_kmeans",
]
DEFAULT_BUDGETS = [5000, 10000, 25000, 50000]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",    required=True,
                        help="scale_200k_seed42 directory")
    parser.add_argument("--output",    required=True,
                        help="paper output dir; states go to <output>/states/")
    parser.add_argument("--methods",   nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--budgets",   nargs="+", type=int, default=DEFAULT_BUDGETS)
    parser.add_argument("--pca_dim",   type=int, default=64)
    parser.add_argument("--device",    default="cuda")
    parser.add_argument("--seed",      type=int,  default=42)
    parser.add_argument("--force",     action="store_true")
    args = parser.parse_args()

    src_script = Path(__file__).parent.parent / "build_predictive_states.py"
    states_dir = Path(args.output) / "states"
    states_dir.mkdir(parents=True, exist_ok=True)

    # Check which states already exist
    missing_methods = []
    missing_budgets = set()
    for method in args.methods:
        for budget in args.budgets:
            sfp = states_dir / f"{method}_B{budget}.pt"
            if not sfp.exists() or args.force:
                missing_methods.append(method)
                missing_budgets.add(budget)
    missing_methods = sorted(set(missing_methods))
    missing_budgets = sorted(missing_budgets)

    if not missing_methods and not args.force:
        print(f"[cached] All state files exist in {states_dir}")
        return

    print(f"\nbuild_state_codebooks")
    print(f"  Source:   {args.source}")
    print(f"  States:   {states_dir}")
    print(f"  Methods:  {missing_methods or args.methods}")
    print(f"  Budgets:  {missing_budgets or args.budgets}")

    cmd = [
        sys.executable, str(src_script),
        "--source",  args.source,
        "--output",  str(Path(args.output)),   # build_predictive_states appends /states internally
        "--methods", *( missing_methods or args.methods),
        "--budgets", *(str(b) for b in (missing_budgets or args.budgets)),
        "--pca_dim", str(args.pca_dim),
    ]
    if args.force:
        cmd.append("--force")

    print(f"\n  $ {' '.join(cmd)}\n")
    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  [FAILED] build_predictive_states exit={result.returncode}")
        sys.exit(result.returncode)
    print(f"  Done  elapsed={elapsed:.0f}s")


if __name__ == "__main__":
    main()

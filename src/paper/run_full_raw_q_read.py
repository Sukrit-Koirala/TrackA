"""
run_full_raw_q_read.py  --  Full raw datastore Q-read baseline

Converts the full N-position datastore.pt into an MVP-2b state file where
every training position is its own single-token "prototype", then runs the
same Q-read evaluation pipeline used for state codebooks and equal-budget
raw baselines.

This is the key efficiency comparison:
    25k predictive state objects  vs  100k raw positions (full datastore)

Usage:
  python run_full_raw_q_read.py \\
    --source outputs_track_a_offline_paper_wikitext2/seed42 \\
    --output outputs_track_a_offline_paper_wikitext2/seed42/q_read/full_raw \\
    --max_q_samples 500000 \\
    --device cuda \\
    --fast_grid
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch


# ── build synthetic MVP-2b state file ─────────────────────────────────────────

def build_full_raw_state_file(source_dir: Path, force: bool = False) -> Path:
    """
    Convert datastore.pt into an MVP-2b state file.

    Each of the N datastore positions becomes its own prototype:
      prototype_h         [N, D]   — L2-normalised hidden states
      top_k_token_ids     [N, 1]   — the single next-token label
      top_k_token_counts  [N, 1]   — count = 1
      total_counts        [N]      — 1 per position
      state_entropy       [N]      — 0 (single token = zero entropy)
      state_purity        [N]      — 1 (single token = perfect purity)
    """
    states_dir = source_dir / "states"
    ds_path    = states_dir / "datastore.pt"
    if not ds_path.exists():
        raise FileNotFoundError(f"datastore.pt not found: {ds_path}")

    ds  = torch.load(ds_path, weights_only=False)
    h   = ds["h"].float()                        # [N, D]
    y   = ds["y"].long()                          # [N]
    N   = len(y)

    output_dir  = source_dir / "full_raw_states"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"full_raw_datastore_B{N}.pt"

    if out_path.exists() and not force:
        print(f"  [cached] full_raw state file  ({out_path.name})")
        return out_path

    print(f"  Building full-raw state file  N={N:,}  ...")
    h_norm = h / (h.norm(dim=-1, keepdim=True) + 1e-8)

    state = {
        "prototype_h":        h_norm.float(),                  # [N, D] fp32 — must match query dtype
        "top_k_token_ids":    y.unsqueeze(1),                  # [N, 1]
        "top_k_token_counts": torch.ones(N, 1, dtype=torch.float32),  # [N, 1]
        "total_counts":       torch.ones(N, dtype=torch.float32),      # [N]
        "state_entropy":      torch.zeros(N, dtype=torch.float32),     # [N]
        "state_purity":       torch.ones(N, dtype=torch.float32),      # [N]
    }
    torch.save(state, out_path)
    print(f"  Saved → {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


# ── main ──────────────────────────────────────────────────────────────────────

def run_full_raw(
    source_dir: Path,
    output_dir: Path,
    max_q_samples: int = 500_000,
    fast_grid: bool = True,
    device: str = "cuda",
    force: bool = False,
    n_epochs: int = 30,
) -> dict | None:
    """
    Build the synthetic state file and run Q-read evaluation.
    Returns the result dict (or None on failure).
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    SCRIPT_DIR = Path(__file__).parent

    # Build (or reuse) the synthetic state file
    state_path = build_full_raw_state_file(source_dir, force=force)
    states_dir = state_path.parent  # source_dir/full_raw_states/

    # Call evaluate_q_read.py on this state file
    cmd = [
        sys.executable, str(SCRIPT_DIR / "evaluate_q_read.py"),
        "--source",        str(source_dir),
        "--states_dir",    str(states_dir),
        "--output",        str(output_dir),
        "--max_q_samples", str(max_q_samples),
        "--n_epochs",      str(n_epochs),
        "--device",        device,
        *(["--fast_grid"] if fast_grid else []),
        *(["--force"]     if force     else []),
    ]

    summary_path = output_dir / "evaluate_q_read_summary.json"
    if summary_path.exists() and not force:
        print(f"  [cached] full_raw Q-read")
    else:
        print(f"  Running full-raw Q-read  (N=datastore, fast_grid={fast_grid}) ...")
        print(f"  $ {' '.join(cmd)}")
        t0     = time.time()
        result = subprocess.run(cmd)
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"  [FAILED] full_raw Q-read  exit={result.returncode}  elapsed={elapsed:.0f}s")
            return None
        print(f"  [OK] full_raw Q-read  elapsed={elapsed:.0f}s")

    if not summary_path.exists():
        print(f"  [WARN] summary not found: {summary_path}")
        return None

    with open(summary_path) as f:
        summary = json.load(f)

    results = summary.get("results", [])
    if not results:
        return None

    r = results[0]  # only one config: full_raw_datastore_B{N}
    return {
        "full_raw_fixed_nll":  r.get("best_fixed_nll"),
        "full_raw_q_nll":      r.get("q_state_nll"),
        "full_raw_oracle_nll": r.get("oracle_nll"),
        "full_raw_n":          r.get("budget"),
        "full_raw_method":     r.get("method"),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Full raw datastore Q-read baseline for Track A")
    ap.add_argument("--source",         required=True, type=Path,
                    help="Per-seed directory (contains states/)")
    ap.add_argument("--output",         required=True, type=Path,
                    help="Output directory for Q-read results")
    ap.add_argument("--max_q_samples",  type=int, default=500_000)
    ap.add_argument("--n_epochs",       type=int, default=30)
    ap.add_argument("--fast_grid",      action="store_true", default=True)
    ap.add_argument("--no_fast_grid",   action="store_false", dest="fast_grid")
    ap.add_argument("--device",         default="cuda")
    ap.add_argument("--force",          action="store_true")
    args = ap.parse_args()

    result = run_full_raw(
        source_dir    = args.source,
        output_dir    = args.output,
        max_q_samples = args.max_q_samples,
        fast_grid     = args.fast_grid,
        device        = args.device,
        force         = args.force,
        n_epochs      = args.n_epochs,
    )

    if result:
        print(f"\nFull Raw Datastore Results")
        print(f"  Fixed NLL:  {result['full_raw_fixed_nll']:.4f}")
        print(f"  Q NLL:      {result['full_raw_q_nll']:.4f}")
        print(f"  Oracle NLL: {result['full_raw_oracle_nll']:.4f}")
    else:
        print("[FAILED] full raw Q-read did not produce results")
        sys.exit(1)


if __name__ == "__main__":
    main()

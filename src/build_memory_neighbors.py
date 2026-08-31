"""
build_memory_neighbors.py

For each saved memory subset, builds nearest-neighbor lookups from
controller_train and val queries into the written memory.

Uses dual-chunked cosine search (reuses cosine_top_k from build_neighbors.py).

Saved files (neighbors/):
  <method>_B<B>_controller_train_top<k>.pt
  <method>_B<B>_val_top<k>.pt

Each file:
  neighbor_sims     FloatTensor  [N_query, k_eff]
  neighbor_y        LongTensor   [N_query, k_eff]   next-token labels from memory
  neighbor_indices  LongTensor   [N_query, k_eff]   memory-local (0..B-1)
  k_eff             int
  budget            int
  method            str
  memory_to_ds      LongTensor   [B]   maps memory-local -> global datastore idx

Usage:
  python src/build_memory_neighbors.py \\
    --source outputs_scale_sweep/scale_200k_seed42 \\
    --output outputs_mvp2_write_memory
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import torch
from tqdm import tqdm

from utils import get_device, set_seed
from build_neighbors import cosine_top_k


MAX_K              = 64
QUERY_CHUNK_SIZE   = 256
DS_CHUNK_SIZE      = 50_000


def build_for_memory(
    mem_path: Path,
    ds: dict,
    ds_norm: torch.Tensor,      # [N_ds, D]  L2-normalised CPU
    ct_data: dict,
    val_data: dict,
    nbrs_dir: Path,
    device: torch.device,
    force: bool = False,
):
    mem = torch.load(mem_path, weights_only=False)
    sel_idx = mem["selected_indices"]   # [B]
    B       = len(sel_idx)
    method  = mem["method"]
    budget  = mem["budget"]
    k_eff   = min(MAX_K, B)

    stem = f"{method}_B{budget}"

    for split_name, query_data in [("controller_train", ct_data), ("val", val_data)]:
        out_path = nbrs_dir / f"{stem}_{split_name}_top{k_eff}.pt"
        if out_path.exists() and not force:
            print(f"  [{stem}] {split_name}: exists, skipping")
            continue

        # Extract memory hidden states and labels
        mem_h    = ds_norm[sel_idx]            # [B, D]  already normalised
        mem_y    = ds["y"][sel_idx]            # [B]

        # Query hidden states (normalise)
        q_h      = query_data["h"].float()
        q_norm   = q_h / (q_h.norm(dim=-1, keepdim=True) + 1e-8)   # [N_q, D]
        N_q      = len(q_norm)

        print(f"  [{stem}] {split_name}: N_q={N_q:,}  B={B:,}  k_eff={k_eff}")

        mem_idx, mem_sims = cosine_top_k(
            q_norm, mem_h, k_eff,
            QUERY_CHUNK_SIZE, DS_CHUNK_SIZE, device,
        )   # [N_q, k_eff]

        neighbor_y = mem_y[mem_idx]   # [N_q, k_eff]

        torch.save({
            "neighbor_sims":    mem_sims,
            "neighbor_y":       neighbor_y,
            "neighbor_indices": mem_idx,
            "k_eff":            k_eff,
            "budget":           budget,
            "method":           method,
            "memory_to_ds":     sel_idx,
        }, out_path)
        print(f"    Saved -> {out_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",  required=True)
    parser.add_argument("--output",  required=True)
    parser.add_argument("--force",   action="store_true")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Filter by method name substring")
    args = parser.parse_args()

    src  = Path(args.source)
    out  = Path(args.output)
    nbrs_dir = out / "neighbors"
    nbrs_dir.mkdir(parents=True, exist_ok=True)

    device = get_device({"device": "cuda"})
    set_seed(42)

    print(f"\nbuild_memory_neighbors")
    print(f"Source: {src}  |  Output: {out}")
    print(f"Device: {device}")

    # ── load datastore once ───────────────────────────────────────────────────
    print("\nLoading datastore ...")
    ds    = torch.load(src / "states" / "datastore.pt", weights_only=False)
    N_ds  = len(ds["y"])
    ds_h  = ds["h"].float()
    ds_norm = ds_h / (ds_h.norm(dim=-1, keepdim=True) + 1e-8)
    del ds_h
    print(f"  N_ds={N_ds:,}")

    # ── load query data once ──────────────────────────────────────────────────
    print("Loading CT and val data ...")
    ct_data  = torch.load(src / "states" / "controller_train.pt", weights_only=False)
    val_data = torch.load(src / "states" / "val.pt",             weights_only=False)
    print(f"  N_ct={len(ct_data['y']):,}  N_val={len(val_data['y']):,}")

    # ── discover memory files ─────────────────────────────────────────────────
    mem_dir   = out / "memories"
    mem_files = sorted(mem_dir.glob("*.pt"))
    if not mem_files:
        print(f"\nNo memory files found in {mem_dir}")
        print("Run select_write_memories.py first.")
        return

    if args.methods:
        mem_files = [p for p in mem_files
                     if any(m in p.stem for m in args.methods)]

    print(f"\nProcessing {len(mem_files)} memory files ...")

    for mem_path in mem_files:
        try:
            build_for_memory(
                mem_path, ds, ds_norm, ct_data, val_data,
                nbrs_dir, device, force=args.force,
            )
        except Exception as e:
            print(f"  ERROR processing {mem_path.name}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()

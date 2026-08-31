"""
build_neighbors.py

For each query split (controller_train, val), compute the top-64 nearest
datastore hidden states by cosine similarity.

Similarity is computed with dual chunking: both queries and the datastore
are processed in chunks so this stays within GPU memory at any scale.

Saved files (outputs/neighbors/):
  controller_train_top64.pt
  val_top64.pt

Each file:
  neighbor_indices  LongTensor  [N_query, 64]
  neighbor_sims     FloatTensor [N_query, 64]
  neighbor_y        LongTensor  [N_query, 64]   next-token ids of neighbors
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import torch
from tqdm import tqdm

from utils import load_config, ensure_dirs, get_device, set_seed


def cosine_top_k(
    q_norm: torch.Tensor,   # [N_q, D]  L2-normalised, CPU
    ds_norm: torch.Tensor,  # [N_ds, D] L2-normalised, CPU
    k: int,
    query_chunk_size: int,
    ds_chunk_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dual-chunked cosine similarity -> top-k indices and similarities.

    Processes both queries and the datastore in chunks, so it is safe
    for datastores of any size without OOM.  For small datastores where
    ds_chunk_size >= N_ds, behaviour is equivalent to loading the full
    datastore onto the device and chunking only queries.
    """
    N_q   = len(q_norm)
    N_ds  = len(ds_norm)
    k_eff = min(k, N_ds)   # can't get more neighbors than the datastore has

    all_idx  = torch.zeros(N_q, k_eff, dtype=torch.long)
    all_sims = torch.zeros(N_q, k_eff)

    for q_start in tqdm(range(0, N_q, query_chunk_size), desc="  query chunks", leave=False):
        q_end   = min(q_start + query_chunk_size, N_q)
        q_chunk = q_norm[q_start:q_end].to(device)     # [Cq, D]
        Cq      = q_end - q_start

        # Running top-k accumulated across datastore chunks
        best_sims = torch.full((Cq, k_eff), float("-inf"), device=device)
        best_idx  = torch.zeros(Cq, k_eff, dtype=torch.long,  device=device)

        for ds_start in range(0, N_ds, ds_chunk_size):
            ds_end   = min(ds_start + ds_chunk_size, N_ds)
            ds_chunk = ds_norm[ds_start:ds_end].to(device)    # [Cds, D]
            Cds      = ds_end - ds_start

            sims    = q_chunk @ ds_chunk.T                     # [Cq, Cds]
            k_local = min(k_eff, Cds)
            chunk_top_sims, chunk_top_idx = sims.topk(k_local, dim=-1)  # [Cq, k_local]
            chunk_top_idx = chunk_top_idx + ds_start           # global datastore indices

            # Merge running best with this chunk's candidates
            combined_sims = torch.cat([best_sims,     chunk_top_sims], dim=-1)
            combined_idx  = torch.cat([best_idx,      chunk_top_idx],  dim=-1)

            new_top_sims, sel = combined_sims.topk(k_eff, dim=-1)
            best_sims = new_top_sims
            best_idx  = combined_idx.gather(-1, sel)

        all_sims[q_start:q_end] = best_sims.cpu()
        all_idx[q_start:q_end]  = best_idx.cpu()

    return all_idx, all_sims


def build_neighbors_for_split(
    cfg: dict,
    split_name: str,
    ds: dict,
    ds_norm: torch.Tensor,  # [N_ds, D] float32, CPU -- pre-computed once
    device: torch.device,
):
    k              = cfg["max_k"]
    q_chunk_size   = cfg.get("neighbor_query_chunk_size",
                             cfg.get("neighbor_chunk_size", 256))
    ds_chunk_size  = cfg.get("neighbor_ds_chunk_size", 50_000)
    states_dir     = Path(cfg["states_dir"])
    nbrs_dir       = Path(cfg["neighbors_dir"])

    print(f"\n  Processing split: {split_name}")
    query  = torch.load(states_dir / f"{split_name}.pt", weights_only=False)

    q_h    = query["h"].float()
    q_norm = q_h / (q_h.norm(dim=-1, keepdim=True) + 1e-8)

    print(f"    Query: {len(q_norm):,}  Datastore: {len(ds_norm):,}"
          f"  (q_chunk={q_chunk_size}, ds_chunk={ds_chunk_size})")

    indices, sims = cosine_top_k(q_norm, ds_norm, k, q_chunk_size, ds_chunk_size, device)

    ds_y       = ds["y"]
    neighbor_y = ds_y[indices]   # [N_q, k]

    out = {
        "neighbor_indices": indices,
        "neighbor_sims":    sims,
        "neighbor_y":       neighbor_y,
    }

    # story_id metadata for split-integrity audit
    ds_meta    = ds.get("metadata", [])
    query_meta = query.get("metadata", [])
    if ds_meta and "story_id" in ds_meta[0]:
        ds_story_ids            = torch.tensor([m["story_id"] for m in ds_meta], dtype=torch.long)
        out["neighbor_story_ids"] = ds_story_ids[indices]   # [N_q, k]
        if query_meta and "story_id" in query_meta[0]:
            out["query_story_ids"] = torch.tensor(
                [m["story_id"] for m in query_meta], dtype=torch.long)  # [N_q]

    out_path = nbrs_dir / f"{split_name}_top{k}.pt"
    torch.save(out, out_path)
    print(f"    Saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg)
    print(f"Device: {device}")

    states_dir = Path(cfg["states_dir"])
    print("\nLoading datastore ...")
    ds = torch.load(states_dir / "datastore.pt", weights_only=False)
    print(f"  Datastore size: {len(ds['y']):,}")

    # Normalise datastore hidden states once (stays on CPU; moved to GPU in chunks)
    ds_h    = ds["h"].float()
    ds_norm = ds_h / (ds_h.norm(dim=-1, keepdim=True) + 1e-8)
    del ds_h

    for split in ["controller_train", "val"]:
        build_neighbors_for_split(cfg, split, ds, ds_norm, device)

    print("\nDone.")


if __name__ == "__main__":
    main()

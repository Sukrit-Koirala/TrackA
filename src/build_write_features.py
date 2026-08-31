"""
build_write_features.py

Computes features for each datastore entry that are available at write time
(no validation leakage: uses only the datastore itself).

Features:
  gpt_entropy           GPT-2 entropy over vocabulary
  gpt_top1_prob         GPT-2 top-1 token probability
  gpt_margin            top1_prob - top2_prob
  gpt_nll_on_true       NLL of GPT-2 on the true next token
  token_rarity          -log(freq[y] / total)  where freq is datastore-internal
  next_token_frequency  freq[y] / total
  hidden_norm           L2 norm of the hidden state
  story_position_norm   pos_in_chunk / max(pos_in_chunk)
  token_position_norm   index in datastore / N_ds
  nearest_sim_to_ref    max cosine similarity to a random reference subset
  mean_top8_sim_to_ref  mean cosine similarity to top-8 in reference subset

Usage:
  python src/build_write_features.py \\
    --source outputs_scale_sweep/scale_200k_seed42 \\
    --output outputs_mvp2_write_memory
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import numpy as np
import torch

from utils import get_device, set_seed


REF_SIZE   = 500    # random reference subset for density features
CHUNK_SIZE = 10_000 # query chunk for density computation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",   required=True)
    parser.add_argument("--output",   required=True)
    parser.add_argument("--ref_size", type=int, default=REF_SIZE)
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    src = Path(args.source)
    out = Path(args.output)
    feat_dir = out / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    device = get_device({"device": "cuda"})
    set_seed(args.seed)

    print(f"\nbuild_write_features")
    print(f"Source: {src}")

    # ── load datastore ────────────────────────────────────────────────────────
    print("\nLoading datastore ...")
    ds   = torch.load(src / "states" / "datastore.pt", weights_only=False)
    h    = ds["h"].float()         # [N_ds, D]
    y    = ds["y"]                 # [N_ds]
    N_ds = len(y)
    D    = h.shape[1]
    print(f"  N_ds={N_ds:,}  D={D}")

    # ── scalar features from stored fields ────────────────────────────────────
    gpt_entropy = ds.get("gpt_entropy",   torch.zeros(N_ds)).float()
    gpt_top1    = ds.get("gpt_top1_prob", torch.zeros(N_ds)).float()
    gpt_top2    = ds.get("gpt_top2_prob", torch.zeros(N_ds)).float()
    nll_gpt     = ds.get("nll_gpt",       torch.zeros(N_ds)).float()
    gpt_margin  = gpt_top1 - gpt_top2

    # token frequency / rarity
    y_np    = y.numpy()
    max_tok = int(y_np.max()) + 1
    counts  = np.bincount(y_np, minlength=max_tok)
    freq    = counts[y_np].astype(np.float32) / N_ds
    token_rarity     = torch.from_numpy(-np.log(freq + 1e-10))
    next_token_freq  = torch.from_numpy(freq)

    # hidden norm
    hidden_norm = h.norm(dim=-1)   # [N_ds]

    # story position from metadata
    metadata = ds.get("metadata", [])
    if metadata and "pos_in_chunk" in metadata[0]:
        pos_raw = torch.tensor([m.get("pos_in_chunk", 0) for m in metadata],
                               dtype=torch.float32)
        story_position_norm = pos_raw / (pos_raw.max() + 1)
    else:
        story_position_norm = torch.zeros(N_ds)

    # token position (index-based)
    token_position_norm = torch.arange(N_ds, dtype=torch.float32) / N_ds

    # ── density features via random reference subset ───────────────────────────
    print(f"\nComputing density features (ref_size={args.ref_size}) ...")
    h_norm = h / (h.norm(dim=-1, keepdim=True) + 1e-8)   # [N_ds, D]

    ref_idx  = torch.randperm(N_ds)[:args.ref_size]
    h_ref    = h_norm[ref_idx].to(device)                 # [ref_size, D]

    nearest_sim = torch.zeros(N_ds)
    mean_top8   = torch.zeros(N_ds)
    top8_k      = min(8, args.ref_size)

    for start in range(0, N_ds, CHUNK_SIZE):
        end   = min(start + CHUNK_SIZE, N_ds)
        chunk = h_norm[start:end].to(device)              # [C, D]
        sims  = chunk @ h_ref.T                           # [C, ref_size]
        top8_sims, _ = sims.topk(top8_k, dim=-1)
        nearest_sim[start:end] = sims.max(dim=-1).values.cpu()
        mean_top8[start:end]   = top8_sims.mean(dim=-1).cpu()

    print("  Done.")

    # ── stack features ────────────────────────────────────────────────────────
    feature_names = [
        "gpt_entropy", "gpt_top1_prob", "gpt_margin", "gpt_nll_on_true",
        "token_rarity", "next_token_frequency", "hidden_norm",
        "story_position_norm", "token_position_norm",
        "nearest_sim_to_ref", "mean_top8_sim_to_ref",
    ]
    X = torch.stack([
        gpt_entropy, gpt_top1, gpt_margin, nll_gpt,
        token_rarity, next_token_freq, hidden_norm,
        story_position_norm, token_position_norm,
        nearest_sim, mean_top8,
    ], dim=-1)   # [N_ds, 11]

    # replace nan/inf
    X = torch.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)

    out_path = feat_dir / "write_features.pt"
    torch.save({
        "X":                 X,
        "feature_names":     feature_names,
        "datastore_indices": torch.arange(N_ds, dtype=torch.long),
    }, out_path)

    print(f"\nFeature matrix: {X.shape}  ({len(feature_names)} features)")
    for i, fn in enumerate(feature_names):
        col = X[:, i]
        print(f"  {fn:<30}  mean={col.mean():.4f}  std={col.std():.4f}")
    print(f"\nSaved -> {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()

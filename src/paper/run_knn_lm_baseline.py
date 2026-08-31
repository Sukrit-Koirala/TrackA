"""
run_knn_lm_baseline.py  --  kNN-LM efficiency comparison baseline

For each val query, retrieves k nearest neighbors from the full N-position
datastore by L2 distance in GPT hidden-state space, then interpolates:

    p(y|x) = (1-λ) * p_GPT(y|x) + λ * p_kNN(y|x)

Grid-searches over k ∈ {8,16,32,64} and λ ∈ {0.1,0.2,0.4,0.6,0.8}.
Saves knn_lm_results.csv (full grid) and knn_lm_best.json (best combo).

This gives the "oracle retrieval ceiling" to compare against the compressed
state codebook — same N-position datastore, but all positions retained.
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import pandas as pd


K_VALUES      = [8, 16, 32, 64]
LAMBDA_VALUES = [0.1, 0.2, 0.4, 0.6, 0.8]
TEMPERATURE   = 1.0
CHUNK_SIZE    = 512   # val queries per GPU chunk


def _l2_distances_chunked(queries: torch.Tensor, keys: torch.Tensor,
                           chunk_size: int = CHUNK_SIZE) -> torch.Tensor:
    """Compute L2 distance matrix [V, N] in row-chunks to stay inside GPU VRAM."""
    V  = queries.shape[0]
    N  = keys.shape[0]
    k_sq = (keys.float() ** 2).sum(-1)          # [N]
    dists = torch.empty(V, N, dtype=torch.float32, device=queries.device)

    for start in range(0, V, chunk_size):
        end  = min(start + chunk_size, V)
        q    = queries[start:end].float()        # [C, D]
        q_sq = (q ** 2).sum(-1, keepdim=True)   # [C, 1]
        qk   = q @ keys.float().T               # [C, N]
        dists[start:end] = (q_sq + k_sq.unsqueeze(0) - 2 * qk).clamp(min=0)

    return dists  # [V, N]


def run_knn_lm(
    datastore_pt: Path,
    val_pt: Path,
    output_dir: Path,
    k_values: list[int]   = K_VALUES,
    lambda_values: list[float] = LAMBDA_VALUES,
    temperature: float    = TEMPERATURE,
    device: str           = "cuda",
) -> dict:
    """
    Run kNN-LM grid search and save results.
    Returns the best-result dict (also written to knn_lm_best.json).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    print(f"  Loading datastore: {datastore_pt}")
    ds   = torch.load(datastore_pt, weights_only=False)
    ds_h = ds["h"].to(dev)           # [N, D]
    ds_y = ds["y"].to(dev)           # [N]

    print(f"  Loading val:       {val_pt}")
    val     = torch.load(val_pt, weights_only=False)
    val_h   = val["h"].to(dev)       # [V, D]
    val_y   = val["y"].to(dev)       # [V]
    nll_gpt = val["nll_gpt"].to(dev).float()   # [V]

    N, D = ds_h.shape
    V    = val_h.shape[0]
    gpt_nll_mean = nll_gpt.mean().item()

    # Storage size of the datastore in MB (fp16 hidden states + int32 labels)
    storage_mb = round((N * D * 2 + N * 4) / 1e6, 1)

    print(f"  Datastore: {N:,} pos  |  Val: {V:,} pos  |  dim={D}")
    print(f"  Storage:   {storage_mb} MB")
    print(f"  GPT-only baseline NLL: {gpt_nll_mean:.4f}")
    print()

    # Full L2 distance matrix [V, N]
    print(f"  Computing distances [{V} × {N}] …")
    dists = _l2_distances_chunked(val_h, ds_h)  # [V, N]

    # Top-max_k neighbors (sort once, slice per k)
    max_k = max(k_values)
    print(f"  Selecting top-{max_k} neighbors …")
    top_dists, top_idx = torch.topk(dists, max_k, dim=1, largest=False)  # [V, max_k]
    top_labels = ds_y[top_idx]   # [V, max_k]
    del dists

    p_gpt      = torch.exp(-nll_gpt)           # [V]
    val_y_col  = val_y.unsqueeze(1)             # [V, 1]

    rows = []
    for k in k_values:
        kd = top_dists[:, :k]    # [V, k]
        kl = top_labels[:, :k]   # [V, k]

        weights = F.softmax(-kd / temperature, dim=1)          # [V, k]
        match   = (kl == val_y_col).float()                    # [V, k]
        p_knn   = (weights * match).sum(dim=1)                 # [V]

        for lam in lambda_values:
            p_comb = ((1 - lam) * p_gpt + lam * p_knn).clamp(min=1e-9)
            nll    = -torch.log(p_comb).mean().item()
            rows.append({"k": k, "lambda": lam, "temperature": temperature,
                         "mean_nll": round(nll, 6)})

    df = pd.DataFrame(rows)
    csv_path = output_dir / "knn_lm_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Grid ({len(rows)} combos) saved → {csv_path}")

    best     = df.loc[df["mean_nll"].idxmin()]
    best_nll = float(best["mean_nll"])

    result = {
        "datastore_size": N,
        "storage_mb":     storage_mb,
        "val_size":       V,
        "gpt_nll":        round(gpt_nll_mean, 6),
        "best_k":         int(best["k"]),
        "best_lambda":    float(best["lambda"]),
        "best_nll":       round(best_nll, 6),
        "delta_vs_gpt":   round(best_nll - gpt_nll_mean, 6),
        "temperature":    temperature,
        "results_csv":    str(csv_path),
    }
    with open(output_dir / "knn_lm_best.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  kNN-LM best:  NLL={result['best_nll']:.4f}  "
          f"(k={result['best_k']}, λ={result['best_lambda']})  "
          f"Δ={result['delta_vs_gpt']:+.4f} vs GPT")

    return result


def main():
    ap = argparse.ArgumentParser(description="kNN-LM full-datastore baseline")
    ap.add_argument("--datastore",   required=True, type=Path,
                    help="Path to datastore.pt")
    ap.add_argument("--val",         required=True, type=Path,
                    help="Path to val.pt")
    ap.add_argument("--output",      required=True, type=Path,
                    help="Directory to write results")
    ap.add_argument("--k",           nargs="+", type=int, default=K_VALUES)
    ap.add_argument("--lambdas",     nargs="+", type=float, default=LAMBDA_VALUES)
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    ap.add_argument("--device",      default="cuda")
    args = ap.parse_args()

    result = run_knn_lm(
        datastore_pt  = args.datastore,
        val_pt        = args.val,
        output_dir    = args.output,
        k_values      = args.k,
        lambda_values = args.lambdas,
        temperature   = args.temperature,
        device        = args.device,
    )

    print(f"\nSummary")
    print(f"  GPT-only NLL:  {result['gpt_nll']:.4f}")
    print(f"  kNN-LM NLL:    {result['best_nll']:.4f}")
    print(f"  Improvement:   {result['delta_vs_gpt']:+.4f}")
    print(f"  Datastore:     {result['datastore_size']:,} positions  ({result['storage_mb']} MB)")


if __name__ == "__main__":
    main()

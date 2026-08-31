"""
build_raw_memory_baselines.py  --  Track A Paper

Select B examples from the raw datastore and pack them in the MVP 2b state
file format as "degenerate states".  Each selected example becomes a state
with prototype_h = its hidden vector and a point-mass token distribution.

This lets the existing Q-read infrastructure (train_q_state_read.run_method_budget)
evaluate raw-memory baselines without any code changes.

Selection methods:
  raw_random           -- uniform random B examples
  raw_high_gpt_loss    -- top-B by GPT NLL (high-uncertainty examples)
  raw_high_gpt_entropy -- top-B by GPT entropy
  raw_token_rarity     -- top-B by inverse token frequency (rare next-tokens)
  raw_coverage         -- greedy farthest-first in hidden-state space
  cluster_medoids      -- MiniBatchKMeans centroids + nearest actual example
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse, json, time
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

from utils import get_device, set_seed

TOP_K = 256          # match build_predictive_states.py
PCA_DIM = 64         # dimensionality for sklearn methods


# ── helpers ───────────────────────────────────────────────────────────────────

def normalize(h: torch.Tensor) -> torch.Tensor:
    return h / (h.norm(dim=-1, keepdim=True).clamp(min=1e-8))


def pack_as_state_file(
    h_keys: torch.Tensor,    # [B, D] normalised hidden vectors
    y_vals: np.ndarray,      # [B] int — next-token ids
    nll_vals: np.ndarray,    # [B] float
    ent_vals: np.ndarray,    # [B] float
    method: str,
    budget: int,
) -> dict:
    """Pack B raw examples into the MVP 2b state file format."""
    B = len(y_vals)

    top_k_ids  = torch.zeros(B, TOP_K, dtype=torch.int32)
    top_k_cnts = torch.zeros(B, TOP_K, dtype=torch.float32)
    for i in range(B):
        top_k_ids[i, 0]  = int(y_vals[i])
        top_k_cnts[i, 0] = 1.0

    return {
        "prototype_h":        h_keys.float(),
        "top_k_token_ids":    top_k_ids,
        "top_k_token_counts": top_k_cnts,
        "total_counts":       torch.ones(B,  dtype=torch.float32),
        "assigned_count":     torch.ones(B,  dtype=torch.int64),
        "state_entropy":      torch.zeros(B, dtype=torch.float32),
        "state_purity":       torch.ones(B,  dtype=torch.float32),
        "mean_nll":           torch.tensor(nll_vals, dtype=torch.float32),
        "mean_gpt_entropy":   torch.tensor(ent_vals, dtype=torch.float32),
        "method":  method,
        "budget":  budget,
        "config":  {"raw_baseline": True, "n_selected": B},
        "extra":   {},
    }


# ── selection methods ─────────────────────────────────────────────────────────

def select_random(N: int, B: int, rng) -> np.ndarray:
    return rng.choice(N, B, replace=False)


def select_high_gpt_loss(nll: np.ndarray, B: int) -> np.ndarray:
    return np.argsort(nll)[-B:]


def select_high_gpt_entropy(ent: np.ndarray, B: int) -> np.ndarray:
    return np.argsort(ent)[-B:]


def select_token_rarity(y: np.ndarray, B: int, vocab_size: int = 50257) -> np.ndarray:
    counts = np.bincount(y, minlength=vocab_size).astype(np.float32) + 1.0
    utility = 1.0 / counts[y]
    return np.argsort(utility)[-B:]


def select_coverage_greedy(
    h_norm: torch.Tensor,   # [N, D]
    B: int,
    device: torch.device,
    seed: int = 42,
) -> np.ndarray:
    """Exact greedy farthest-first coverage. O(B * N * D)."""
    N = h_norm.shape[0]
    rng = np.random.default_rng(seed)

    h_gpu = h_norm.float().to(device)     # [N, D]
    # max similarity from each point to any selected point (init to -inf)
    min_sim = torch.full((N,), -2.0)      # CPU

    selected = []
    not_sel  = np.ones(N, dtype=bool)

    first = int(rng.integers(N))
    selected.append(first)
    not_sel[first] = False

    # Init min_sim from first selected
    q   = h_gpu[first:first+1]             # [1, D]
    sim = (q @ h_gpu.T).squeeze(0).cpu()  # [N]
    min_sim = sim

    for b in range(1, B):
        if b % 2000 == 0:
            print(f"      coverage {b}/{B} ...")
        # Candidate = point with lowest max-sim to selected (= farthest)
        ms = min_sim.clone()
        ms[~torch.from_numpy(not_sel)] = 2.0   # exclude selected
        next_idx = int(ms.argmin())
        selected.append(next_idx)
        not_sel[next_idx] = False

        q   = h_gpu[next_idx:next_idx+1]
        sim = (q @ h_gpu.T).squeeze(0).cpu()
        min_sim = torch.maximum(min_sim, sim)

    return np.array(selected, dtype=np.int64)


def select_cluster_medoids(
    h_norm: torch.Tensor,   # [N, D]
    B: int,
    device: torch.device,
    pca_dim: int = PCA_DIM,
    seed: int = 42,
) -> np.ndarray:
    """MiniBatchKMeans in PCA space, then find nearest actual example per cluster."""
    N = h_norm.shape[0]
    h_np = h_norm.float().numpy()

    # PCA
    n_pca = min(pca_dim, N - 1, h_np.shape[1])
    pca   = PCA(n_components=n_pca, random_state=seed)
    h_pca = pca.fit_transform(h_np[:min(50000, N)])
    if N > 50000:
        h_pca = pca.transform(h_np)

    # K-means
    km = MiniBatchKMeans(n_clusters=B, random_state=seed,
                         batch_size=min(4096, N), n_init=3, max_iter=200)
    labels = km.fit_predict(h_pca)

    # For each cluster, find nearest example to centroid in ORIGINAL space
    centroids_pca = torch.tensor(km.cluster_centers_, dtype=torch.float32)
    centroids_h   = torch.tensor(pca.inverse_transform(centroids_pca.numpy()), dtype=torch.float32)
    centroids_h   = normalize(centroids_h)

    h_gpu     = h_norm.float().to(device)
    cent_gpu  = centroids_h.to(device)

    labels_t = torch.from_numpy(labels)
    selected = np.empty(B, dtype=np.int64)

    # Batch by cluster
    CHUNK = 512
    for c_start in range(0, B, CHUNK):
        c_end = min(c_start + CHUNK, B)
        for c in range(c_start, c_end):
            mask    = (labels_t == c).nonzero(as_tuple=True)[0]
            if len(mask) == 0:
                selected[c] = int(torch.randint(N, (1,)).item())
                continue
            h_cl = h_gpu[mask]                              # [Mc, D]
            sims = (cent_gpu[c:c+1] @ h_cl.T).squeeze(0)  # [Mc]
            best = int(mask[sims.argmax().item()].item())
            selected[c] = best

    return selected


# ── main ──────────────────────────────────────────────────────────────────────

ALL_METHODS = [
    "raw_random",
    "raw_high_gpt_loss",
    "raw_high_gpt_entropy",
    "raw_token_rarity",
    "raw_coverage",
    "cluster_medoids",
]

COVERAGE_MAX_B = 25_000   # exact greedy only for B <= this


def build_baseline(
    ds_data: dict,
    method: str,
    budget: int,
    device: torch.device,
    seed: int = 42,
    pca_dim: int = PCA_DIM,
) -> dict:
    t0  = time.time()
    N   = ds_data["h"].shape[0]
    rng = np.random.default_rng(seed)

    h_norm = normalize(ds_data["h"].float())  # [N, D]
    y      = ds_data["y"].numpy()
    nll    = ds_data["nll_gpt"].float().numpy()
    ent    = ds_data["gpt_entropy"].float().numpy()

    B = min(budget, N)

    print(f"    [{method} B={B}]  N={N:,}  seed={seed}")

    if method == "raw_random":
        idx = select_random(N, B, rng)

    elif method == "raw_high_gpt_loss":
        idx = select_high_gpt_loss(nll, B)

    elif method == "raw_high_gpt_entropy":
        idx = select_high_gpt_entropy(ent, B)

    elif method == "raw_token_rarity":
        idx = select_token_rarity(y, B)

    elif method == "raw_coverage":
        if B > COVERAGE_MAX_B:
            print(f"    [warn] raw_coverage: B={B} > {COVERAGE_MAX_B}; "
                  f"falling back to cluster_medoids for large budget")
            idx = select_cluster_medoids(h_norm, B, device, pca_dim, seed)
        else:
            idx = select_coverage_greedy(h_norm, B, device, seed)

    elif method == "cluster_medoids":
        idx = select_cluster_medoids(h_norm, B, device, pca_dim, seed)

    else:
        raise ValueError(f"Unknown method: {method}")

    idx = np.asarray(idx, dtype=np.int64)
    h_sel   = h_norm[idx]
    y_sel   = y[idx]
    nll_sel = nll[idx]
    ent_sel = ent[idx]

    sf = pack_as_state_file(h_sel, y_sel, nll_sel, ent_sel, method, B)
    print(f"    done  elapsed={time.time()-t0:.0f}s")
    return sf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",    required=True,
                        help="scale_200k_seed42 directory (contains states/datastore.pt)")
    parser.add_argument("--output",    required=True,
                        help="output directory; baselines saved to <output>/raw_baselines/")
    parser.add_argument("--methods",   nargs="+", default=ALL_METHODS)
    parser.add_argument("--budgets",   nargs="+", type=int,
                        default=[1000, 5000, 10000, 25000, 50000])
    parser.add_argument("--pca_dim",   type=int,  default=PCA_DIM)
    parser.add_argument("--device",    default="cuda")
    parser.add_argument("--seed",      type=int,  default=42)
    parser.add_argument("--force",     action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device({"device": args.device})
    src    = Path(args.source)
    out    = Path(args.output) / "raw_baselines"
    out.mkdir(parents=True, exist_ok=True)

    ds_path = src / "states" / "datastore.pt"
    if not ds_path.exists():
        raise FileNotFoundError(f"Datastore not found: {ds_path}")

    print(f"\nbuild_raw_memory_baselines")
    print(f"  Source:  {ds_path}")
    print(f"  Output:  {out}")
    print(f"  Methods: {args.methods}")
    print(f"  Budgets: {args.budgets}")
    print(f"  Device:  {device}")

    print(f"\n  Loading datastore ...")
    ds_data = torch.load(ds_path, weights_only=False)
    N = ds_data["h"].shape[0]
    print(f"  Datastore: N={N:,}  D={ds_data['h'].shape[1]}")

    results = []
    for method in args.methods:
        for budget in args.budgets:
            tag      = f"{method}_B{budget}"
            out_path = out / f"{tag}.pt"

            if out_path.exists() and not args.force:
                print(f"  [cached] {tag}")
                results.append({"method": method, "budget": budget,
                                 "path": str(out_path), "cached": True})
                continue

            try:
                sf = build_baseline(ds_data, method, budget, device,
                                    seed=args.seed, pca_dim=args.pca_dim)
                torch.save(sf, out_path)
                print(f"    Saved: {out_path}")
                results.append({"method": method, "budget": budget,
                                 "path": str(out_path), "cached": False})
            except Exception as e:
                import traceback
                print(f"  [FAILED] {tag}: {e}")
                traceback.print_exc()

    with open(out / "build_raw_baselines_log.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone.  {len(results)} baselines written to {out}")


if __name__ == "__main__":
    main()

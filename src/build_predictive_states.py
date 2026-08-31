"""
build_predictive_states.py

Builds persistent predictive states from datastore hidden states.
Each state aggregates many examples into a reusable prediction-bearing object.

Methods:
  minibatch_kmeans    MiniBatchKMeans on datastore h (scalable default)
  query_kmeans        Cluster CT query h; assign datastore entries to nearest
  balanced_kmeans     Capped assignment after minibatch_kmeans
  streaming_write     Online WRITE/UPDATE with cosine similarity threshold
  utility_weighted    minibatch_kmeans with GPT-NLL-weighted token counts
  random_partition    Random assignment sanity baseline

State file:
  method              str
  budget              int
  prototype_h         FloatTensor [B, D]   L2-normalised
  top_k_token_ids     LongTensor  [B, TOP_K]
  top_k_token_counts  FloatTensor [B, TOP_K]
  total_counts        FloatTensor [B]
  assigned_count      LongTensor  [B]
  state_entropy       FloatTensor [B]
  state_purity        FloatTensor [B]
  mean_nll            FloatTensor [B]
  mean_gpt_entropy    FloatTensor [B]
  config              dict

Usage:
  python src/build_predictive_states.py \\
    --source outputs_scale_sweep/scale_200k_seed42 \\
    --output outputs_mvp2b_state_write \\
    --budgets 1000 2000 5000 10000 25000 \\
    --methods minibatch_kmeans query_kmeans balanced_kmeans streaming_write random_partition
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import math
import numpy as np
import torch
from tqdm import tqdm

from utils import get_device, set_seed

TOP_K          = 256   # top tokens stored per state
PCA_DIM        = 64    # dims used for clustering (original space for prototypes)
MINIBATCH_SIZE = 4096
MAX_ITER       = 200
STREAM_CHUNK   = 8192
Q_CHUNK        = 512


# ── token count accumulation ──────────────────────────────────────────────────

def accumulate_token_counts(labels: np.ndarray, y: np.ndarray, B: int,
                             weights: np.ndarray | None = None):
    """Vectorised per-state token count using sorted arrays.
    Returns top_k_ids [B, TOP_K], top_k_cnts [B, TOP_K], per_state_total [B].
    """
    N = len(y)
    if weights is None:
        weights = np.ones(N, dtype=np.float32)

    # Sort by label
    order = np.argsort(labels, kind="stable")
    slabels = labels[order]
    sy      = y[order]
    sw      = weights[order]

    boundaries = np.searchsorted(slabels, np.arange(B + 1))

    top_k_ids  = np.zeros((B, TOP_K), dtype=np.int32)
    top_k_cnts = np.zeros((B, TOP_K), dtype=np.float32)
    total      = np.zeros(B, dtype=np.float32)

    for ci in range(B):
        lo, hi = int(boundaries[ci]), int(boundaries[ci + 1])
        if lo >= hi:
            continue
        ay = sy[lo:hi]
        aw = sw[lo:hi]
        total[ci] = float(aw.sum())
        # accumulate weighted counts per unique token
        unique_y, inv = np.unique(ay, return_inverse=True)
        cnts = np.zeros(len(unique_y), dtype=np.float64)
        np.add.at(cnts, inv, aw)
        # take top-k by count
        if len(unique_y) <= TOP_K:
            top_k_ids[ci,  :len(unique_y)] = unique_y
            top_k_cnts[ci, :len(unique_y)] = cnts
        else:
            ord2 = np.argpartition(-cnts, TOP_K)[:TOP_K]
            ord2 = ord2[np.argsort(-cnts[ord2])]
            top_k_ids[ci]  = unique_y[ord2]
            top_k_cnts[ci] = cnts[ord2]

    return top_k_ids, top_k_cnts, total


def compute_state_stats(h_orig: np.ndarray, labels: np.ndarray, B: int,
                         top_k_ids: np.ndarray, top_k_cnts: np.ndarray,
                         total: np.ndarray, nll_gpt: np.ndarray,
                         gpt_entropy: np.ndarray):
    """Compute prototypes, entropy, purity, mean nll from label assignment."""
    N, D = h_orig.shape

    # Prototype: normalized mean of assigned h in original space
    h_t  = torch.from_numpy(h_orig.astype(np.float32))
    lb_t = torch.from_numpy(labels.astype(np.int64))
    cnt_t = torch.zeros(B)
    sum_h = torch.zeros(B, D)
    cnt_t.scatter_add_(0, lb_t, torch.ones(N))
    sum_h.scatter_add_(0, lb_t.unsqueeze(1).expand(-1, D), h_t)
    norms = sum_h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    prototype_h = (sum_h / norms)                         # [B, D] normalised

    # mean_nll, mean_gpt_entropy
    nll_t  = torch.from_numpy(nll_gpt.astype(np.float32))
    ent_t  = torch.from_numpy(gpt_entropy.astype(np.float32))
    sum_nll = torch.zeros(B)
    sum_ent = torch.zeros(B)
    sum_nll.scatter_add_(0, lb_t, nll_t)
    sum_ent.scatter_add_(0, lb_t, ent_t)
    safe_cnt = cnt_t.clamp(min=1)
    mean_nll = sum_nll / safe_cnt
    mean_ent = sum_ent / safe_cnt

    # entropy & purity from top_k_cnts / total
    state_entropy = np.zeros(B, dtype=np.float32)
    state_purity  = np.zeros(B, dtype=np.float32)
    for ci in range(B):
        tot = float(total[ci])
        if tot <= 0:
            continue
        cnts = top_k_cnts[ci][top_k_cnts[ci] > 0]
        p = cnts / tot
        state_entropy[ci] = float(-np.sum(p * np.log(p + 1e-12)))
        state_purity[ci]  = float(cnts.max() / tot) if len(cnts) else 0.0

    return dict(
        prototype_h        = prototype_h,
        top_k_token_ids    = torch.from_numpy(top_k_ids),
        top_k_token_counts = torch.from_numpy(top_k_cnts),
        total_counts       = torch.from_numpy(total),
        assigned_count     = cnt_t.long(),
        state_entropy      = torch.from_numpy(state_entropy),
        state_purity       = torch.from_numpy(state_purity),
        mean_nll           = mean_nll,
        mean_gpt_entropy   = mean_ent,
    )


# ── PCA helper ────────────────────────────────────────────────────────────────

def fit_pca(h_norm: np.ndarray, n_components: int, seed: int = 42):
    from sklearn.decomposition import PCA
    sample = h_norm[:min(50_000, len(h_norm))]
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(sample)
    return pca


# ── Method 1: minibatch_kmeans ────────────────────────────────────────────────

def build_minibatch_kmeans(h_orig, h_norm, y, ds, B, config):
    from sklearn.cluster import MiniBatchKMeans
    seed  = config.get("seed", 42)
    n_pca = config.get("pca_dim", PCA_DIM)
    print(f"    PCA {h_norm.shape[1]}→{n_pca} ...")
    pca   = fit_pca(h_norm, n_pca, seed)
    h_pca = pca.transform(h_norm).astype(np.float32)

    print(f"    MiniBatchKMeans B={B} ...")
    km = MiniBatchKMeans(n_clusters=B, n_init=3, max_iter=MAX_ITER,
                         batch_size=MINIBATCH_SIZE, random_state=seed,
                         compute_labels=True)
    labels = km.fit_predict(h_pca).astype(np.int64)
    return labels, {}


# ── Method 2: query_kmeans ────────────────────────────────────────────────────

def build_query_kmeans(h_orig, h_norm, y, ds, ct_h_norm, B, config):
    """Cluster CT query hidden states; assign DS entries to nearest prototype."""
    from sklearn.cluster import MiniBatchKMeans
    seed  = config.get("seed", 42)
    n_pca = config.get("pca_dim", PCA_DIM)

    print(f"    PCA on CT queries {ct_h_norm.shape[1]}→{n_pca} ...")
    pca      = fit_pca(ct_h_norm, n_pca, seed)
    ct_pca   = pca.transform(ct_h_norm).astype(np.float32)

    print(f"    MiniBatchKMeans on CT (B={B}) ...")
    km = MiniBatchKMeans(n_clusters=min(B, len(ct_h_norm)), n_init=3,
                         max_iter=MAX_ITER, batch_size=MINIBATCH_SIZE,
                         random_state=seed, compute_labels=False)
    km.fit(ct_pca)

    # Assign DS entries to nearest CT-derived centroid
    print(f"    Assigning DS entries ...")
    ds_pca = pca.transform(h_norm).astype(np.float32)
    labels = km.predict(ds_pca).astype(np.int64)
    return labels, {"clustered_on": "controller_train"}


# ── Method 3: balanced_kmeans ─────────────────────────────────────────────────

def build_balanced_kmeans(h_orig, h_norm, y, ds, B, config):
    """MiniBatchKMeans + cap on assignments per cluster."""
    from sklearn.cluster import MiniBatchKMeans
    seed      = config.get("seed", 42)
    n_pca     = config.get("pca_dim", PCA_DIM)
    cap_factor = config.get("cap_factor", 3)

    print(f"    PCA + MiniBatchKMeans B={B} ...")
    pca   = fit_pca(h_norm, n_pca, seed)
    h_pca = pca.transform(h_norm).astype(np.float32)
    km    = MiniBatchKMeans(n_clusters=B, n_init=3, max_iter=MAX_ITER,
                            batch_size=MINIBATCH_SIZE, random_state=seed,
                            compute_labels=False)
    km.fit(h_pca)

    N = len(h_pca)
    max_cap = math.ceil(N / B * cap_factor)
    centroids = km.cluster_centers_  # [B, n_pca]
    labels    = np.full(N, -1, dtype=np.int64)
    counts    = np.zeros(B, dtype=np.int64)

    print(f"    Balanced assignment (cap={max_cap}) ...")
    # Process in chunks; greedily assign to nearest non-full cluster
    for start in tqdm(range(0, N, STREAM_CHUNK), leave=False):
        end  = min(start + STREAM_CHUNK, N)
        chunk = h_pca[start:end]  # [C, pca_dim]
        sims  = chunk @ centroids.T  # [C, B]
        order = np.argsort(-sims, axis=-1)  # [C, B] sorted by sim desc
        for j in range(end - start):
            for ci in order[j]:
                if counts[ci] < max_cap:
                    labels[start + j] = ci
                    counts[ci] += 1
                    break
            else:
                labels[start + j] = order[j, 0]  # fallback

    return labels, {"cap_factor": cap_factor}


# ── Method 4: streaming_write ─────────────────────────────────────────────────

def build_streaming_write(h_orig, h_norm, y, ds, B, threshold, config, device):
    """Online WRITE/UPDATE: create state or update nearest."""
    N, D = h_norm.shape
    seed  = config.get("seed", 42)
    rng   = np.random.default_rng(seed)
    order = rng.permutation(N)

    proto_norm   = np.zeros((B, D), dtype=np.float32)   # normalised prototypes
    proto_unnorm = np.zeros((B, D), dtype=np.float64)   # sum of h (for re-norm)
    assign_count = np.zeros(B, dtype=np.int64)
    labels       = np.zeros(N, dtype=np.int64)
    n_states     = 0

    h_t = torch.from_numpy(h_norm)  # [N, D] on CPU

    for chunk_start in tqdm(range(0, N, STREAM_CHUNK),
                             desc="  streaming", leave=False):
        chunk_end  = min(chunk_start + STREAM_CHUNK, N)
        idx_chunk  = order[chunk_start:chunk_end]  # [C]
        h_chunk    = h_t[idx_chunk].to(device)     # [C, D]
        C = len(idx_chunk)

        if n_states == 0:
            # Bootstrap with first entry
            i0 = int(idx_chunk[0])
            proto_norm[0]   = h_norm[i0]
            proto_unnorm[0] = h_norm[i0].astype(np.float64)
            assign_count[0] = 1
            labels[i0]      = 0
            n_states        = 1
            h_chunk = h_chunk[1:]
            idx_chunk = idx_chunk[1:]
            C -= 1
            if C == 0:
                continue

        # Batch similarity against current prototypes
        proto_t  = torch.from_numpy(proto_norm[:n_states]).to(device)  # [S, D]
        sims     = h_chunk @ proto_t.T        # [C, S]
        max_sims, nearest = sims.max(dim=-1)  # [C]
        max_sims = max_sims.cpu().numpy()
        nearest  = nearest.cpu().numpy()

        for j, gi in enumerate(idx_chunk):
            gi = int(gi)
            ms = float(max_sims[j])
            nn = int(nearest[j])

            if ms >= threshold:
                # UPDATE nearest state
                si = nn
                proto_unnorm[si] += h_norm[gi].astype(np.float64)
                norm = np.linalg.norm(proto_unnorm[si]) + 1e-8
                proto_norm[si] = (proto_unnorm[si] / norm).astype(np.float32)
            elif n_states < B:
                # CREATE new state
                si = n_states
                proto_norm[si]   = h_norm[gi]
                proto_unnorm[si] = h_norm[gi].astype(np.float64)
                n_states        += 1
            else:
                # Budget full: assign to nearest
                si = nn
                proto_unnorm[si] += h_norm[gi].astype(np.float64)
                norm = np.linalg.norm(proto_unnorm[si]) + 1e-8
                proto_norm[si] = (proto_unnorm[si] / norm).astype(np.float32)

            labels[gi] = si
            assign_count[si] += 1

    actual_B = n_states
    print(f"    Streaming result: {actual_B} states (threshold={threshold})")
    return labels, {"threshold": threshold, "actual_B": actual_B}


# ── Method 5: utility_weighted ────────────────────────────────────────────────

def build_utility_weighted(h_orig, h_norm, y, ds, B, config, util_path=None):
    """minibatch_kmeans with NLL-weighted token counts."""
    from sklearn.cluster import MiniBatchKMeans
    seed  = config.get("seed", 42)
    n_pca = config.get("pca_dim", PCA_DIM)

    print(f"    PCA + MiniBatchKMeans B={B} ...")
    pca   = fit_pca(h_norm, n_pca, seed)
    h_pca = pca.transform(h_norm).astype(np.float32)
    km    = MiniBatchKMeans(n_clusters=B, n_init=3, max_iter=MAX_ITER,
                            batch_size=MINIBATCH_SIZE, random_state=seed)
    labels = km.fit_predict(h_pca).astype(np.int64)

    N = len(y)
    nll_gpt = ds.get("nll_gpt", torch.zeros(N)).numpy().astype(np.float32)

    # Try to load utility scores
    weight_type = "nll_gpt"
    weights = nll_gpt.copy()
    if util_path is not None and Path(util_path).exists():
        try:
            u = torch.load(util_path, weights_only=False)
            raw = u.get("utility_positive", None)
            if raw is not None and len(raw) == N:
                weights = raw.numpy().astype(np.float32).clip(min=0)
                weight_type = "utility_positive"
                print(f"    Using utility_positive weights")
        except Exception:
            pass

    # Clip extreme weights
    p95 = float(np.percentile(weights[weights > 0], 95)) if (weights > 0).any() else 1.0
    weights = np.clip(weights, 0, p95)
    weights = weights + 0.1   # additive floor so every entry contributes

    return labels, {"weight_type": weight_type}


# ── Method 6: random_partition ────────────────────────────────────────────────

def build_random_partition(h_orig, h_norm, y, ds, B, config):
    seed   = config.get("seed", 42)
    rng    = np.random.default_rng(seed)
    N      = len(y)
    labels = rng.integers(0, B, size=N, dtype=np.int64)
    return labels, {}


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_state_file(state_dict: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, path)
    size_mb = path.stat().st_size / 1e6
    print(f"    Saved -> {path.name}  ({size_mb:.1f} MB)")


def build_and_save(method: str, budget: int, h_orig, h_norm, y_np, ds,
                   nll_gpt, gpt_entropy, ct_h_norm, config, out_dir: Path,
                   device, force: bool = False, util_path=None):
    # For streaming_write, budget is handled via threshold
    if method.startswith("streaming_write"):
        thresholds = config.get("stream_thresholds", [0.97, 0.98, 0.99, 0.995])
        for thr in thresholds:
            stem = f"streaming_write_thr{thr}_B{budget}"
            path = out_dir / f"{stem}.pt"
            if path.exists() and not force:
                print(f"  [{stem}] exists, skipping")
                continue
            print(f"  [{stem}]")
            labels, extra = build_streaming_write(h_orig, h_norm, y_np, ds,
                                                   budget, thr, config, device)
            weights = None
            ids, cnts, total = accumulate_token_counts(labels, y_np, budget, weights)
            stats = compute_state_stats(h_orig, labels, budget, ids, cnts, total,
                                         nll_gpt, gpt_entropy)
            stats.update({"method": stem, "budget": budget, "config": config,
                          "extra": extra})
            save_state_file(stats, path)
        return

    # Weighted method
    weights = None
    if method == "utility_weighted":
        labels, extra = build_utility_weighted(h_orig, h_norm, y_np, ds, budget,
                                                config, util_path)
        nll_gpt_np = nll_gpt.copy()
        weights = nll_gpt_np + 0.1
    elif method == "balanced_kmeans":
        labels, extra = build_balanced_kmeans(h_orig, h_norm, y_np, ds, budget, config)
    elif method == "query_kmeans":
        labels, extra = build_query_kmeans(h_orig, h_norm, y_np, ds, ct_h_norm, budget, config)
    elif method == "minibatch_kmeans":
        labels, extra = build_minibatch_kmeans(h_orig, h_norm, y_np, ds, budget, config)
    elif method == "random_partition":
        labels, extra = build_random_partition(h_orig, h_norm, y_np, ds, budget, config)
    else:
        print(f"  Unknown method: {method}, skipping")
        return

    stem = f"{method}_B{budget}"
    path = out_dir / f"{stem}.pt"
    if path.exists() and not force:
        print(f"  [{stem}] exists, skipping")
        return

    print(f"  [{stem}]")
    ids, cnts, total = accumulate_token_counts(labels, y_np, budget, weights)
    stats = compute_state_stats(h_orig, labels, budget, ids, cnts, total,
                                  nll_gpt, gpt_entropy)
    stats.update({"method": method, "budget": budget, "config": config, "extra": extra})
    save_state_file(stats, path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",      required=True)
    parser.add_argument("--output",      required=True)
    parser.add_argument("--budgets",     nargs="+", type=int,
                        default=[1000, 2000, 5000, 10000, 25000])
    parser.add_argument("--methods",     nargs="+",
                        default=["minibatch_kmeans", "query_kmeans",
                                 "balanced_kmeans", "streaming_write",
                                 "utility_weighted", "random_partition"])
    parser.add_argument("--pca_dim",     type=int, default=PCA_DIM)
    parser.add_argument("--stream_thresholds", nargs="+", type=float,
                        default=[0.97, 0.98, 0.99, 0.995])
    parser.add_argument("--force",       action="store_true")
    args = parser.parse_args()

    src     = Path(args.source)
    out     = Path(args.output)
    states_out = out / "states"
    states_out.mkdir(parents=True, exist_ok=True)

    device  = get_device({"device": "cuda"})
    set_seed(42)

    config = {
        "pca_dim":          args.pca_dim,
        "seed":             42,
        "stream_thresholds": args.stream_thresholds,
        "cap_factor":       3,
    }

    print(f"\nbuild_predictive_states")
    print(f"Source: {src}  |  Device: {device}")
    print(f"Budgets: {args.budgets}  |  Methods: {args.methods}")

    # ── load datastore ────────────────────────────────────────────────────────
    print("\nLoading datastore ...")
    ds   = torch.load(src / "states" / "datastore.pt", weights_only=False)
    N    = len(ds["y"])
    y_np = ds["y"].numpy().astype(np.int64)
    h_f  = ds["h"].float()
    h_orig = h_f.numpy()
    h_norm = (h_f / (h_f.norm(dim=-1, keepdim=True) + 1e-8)).numpy()
    nll_gpt     = ds.get("nll_gpt",     torch.zeros(N)).numpy().astype(np.float32)
    gpt_entropy = ds.get("gpt_entropy", torch.zeros(N)).numpy().astype(np.float32)
    print(f"  N_ds={N:,}  D={h_orig.shape[1]}")

    # ── load CT for query_kmeans ──────────────────────────────────────────────
    ct_h_norm = None
    if "query_kmeans" in args.methods:
        print("Loading CT for query_kmeans ...")
        ct  = torch.load(src / "states" / "controller_train.pt", weights_only=False)
        ct_h = ct["h"].float()
        ct_h_norm = (ct_h / (ct_h.norm(dim=-1, keepdim=True) + 1e-8)).numpy()
        print(f"  N_ct={len(ct_h_norm):,}")

    # ── locate utility file (optional) ────────────────────────────────────────
    util_path = None
    cand = Path(args.output).parent / "outputs_mvp2_write_memory" / "utility" / "write_utilities_modeA.pt"
    if cand.exists():
        util_path = str(cand)

    # ── build states ──────────────────────────────────────────────────────────
    for method in args.methods:
        print(f"\n{'='*60}  {method}")
        for B in args.budgets:
            if B > N:
                print(f"  Skipping B={B} (> N_ds)")
                continue
            build_and_save(method, B, h_orig, h_norm, y_np, ds,
                           nll_gpt, gpt_entropy, ct_h_norm, config,
                           states_out, device, force=args.force,
                           util_path=util_path)

    print("\nDone.")


if __name__ == "__main__":
    main()

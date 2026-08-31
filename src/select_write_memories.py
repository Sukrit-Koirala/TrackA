"""
select_write_memories.py

Selects B datastore entries for each memory selection method and budget.

Methods:
  random_seed{s}           random sample (3 seeds)
  uniform_story            ~equal entries per story
  high_gpt_loss            top-B by gpt NLL on true token
  high_gpt_entropy         top-B by GPT entropy
  token_rarity             top-B by -log(freq[y]/total)
  coverage_memory          k-means based diverse coverage
  oracle_utility_positive  top-B by utility_positive  [DIAG]
  oracle_utility_net       top-B by utility_net        [DIAG]
  learned_write_linear     top-B by linear scorer
  learned_write_gbr        top-B by GBR scorer
  learned_write_mlp        top-B by MLP scorer

Saves:
  memories/<method>_B<B>.pt
    selected_indices  LongTensor [B]
    method            str
    budget            int
    metadata          dict

Usage:
  python src/select_write_memories.py \\
    --source outputs_scale_sweep/scale_200k_seed42 \\
    --output outputs_mvp2_write_memory \\
    --budgets 1000 5000 10000 25000 50000
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import numpy as np
import torch

from utils import set_seed


RANDOM_SEEDS = [0, 1, 2]


def save_memory(mem_dir: Path, method: str, budget: int,
                selected: np.ndarray, meta: dict = None):
    fname = f"{method}_B{budget}.pt"
    torch.save({
        "selected_indices": torch.from_numpy(selected.astype(np.int64)),
        "method":           method,
        "budget":           budget,
        "metadata":         meta or {},
    }, mem_dir / fname)
    print(f"  Saved: {fname}  (n={len(selected)})")


def select_random(N_ds: int, B: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(N_ds, size=B, replace=False)


def select_uniform_story(metadata: list, N_ds: int, B: int) -> np.ndarray:
    if not metadata or "story_id" not in metadata[0]:
        # fallback: random
        return select_random(N_ds, B, 0)

    story_to_idx: dict[int, list] = {}
    for i, m in enumerate(metadata):
        sid = m.get("story_id", 0)
        story_to_idx.setdefault(sid, []).append(i)

    stories = sorted(story_to_idx.keys())
    n_stories = len(stories)
    per_story = max(1, B // n_stories)
    selected: list[int] = []
    rng = np.random.default_rng(0)
    for sid in stories:
        idxs = story_to_idx[sid]
        n    = min(per_story, len(idxs))
        selected.extend(rng.choice(idxs, size=n, replace=False).tolist())
        if len(selected) >= B:
            break
    # fill remainder if not enough
    if len(selected) < B:
        all_set = set(selected)
        pool    = [i for i in range(N_ds) if i not in all_set]
        extra   = rng.choice(pool, size=B - len(selected), replace=False)
        selected.extend(extra.tolist())

    return np.array(selected[:B], dtype=np.int64)


def select_by_score(scores: np.ndarray, B: int, highest: bool = True) -> np.ndarray:
    """Return top-B indices by score (highest or lowest)."""
    if highest:
        return np.argsort(scores)[-B:]
    else:
        return np.argsort(scores)[:B]


def select_coverage(h: np.ndarray, B: int, seed: int = 42) -> np.ndarray:
    """
    Diverse coverage selection via PCA + MiniBatchKMeans.
    Groups entries into n_clusters clusters, then selects the B entries
    that are nearest to their cluster centres (exemplars).
    """
    from sklearn.decomposition import PCA
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import pairwise_distances_argmin

    N_ds = len(h)
    n_clusters = min(B, 2000)   # cap cluster count for speed
    print(f"    PCA: fitting on {N_ds:,} entries ...")

    # PCA to 32 dims (fit on a sample if N_ds is large)
    sample_size = min(N_ds, 50_000)
    rng         = np.random.default_rng(seed)
    sample_idx  = rng.choice(N_ds, sample_size, replace=False)
    H_sample    = h[sample_idx]

    pca = PCA(n_components=32, random_state=seed)
    pca.fit(H_sample)
    H_pca = pca.transform(h).astype(np.float32)   # [N_ds, 32]

    print(f"    KMeans: n_clusters={n_clusters} ...")
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, n_init=3,
        random_state=seed, max_iter=300, batch_size=10_000,
    )
    labels  = kmeans.fit_predict(H_pca)
    centers = kmeans.cluster_centers_   # [n_clusters, 32]

    # For each cluster, find the nearest exemplar entry
    exemplars = pairwise_distances_argmin(centers, H_pca)   # [n_clusters]
    selected  = np.unique(exemplars)   # deduplicate

    # If n_clusters < B, fill with entries nearest to any cluster centre
    # (second-nearest exemplar per cluster, etc.)
    if len(selected) < B:
        needed = B - len(selected)
        already = set(selected.tolist())
        # Per cluster, find more entries sorted by distance to centre
        extras: list[int] = []
        # Compute per-entry distance to its assigned centre
        centre_per_entry = centers[labels]          # [N_ds, 32]
        dists = np.linalg.norm(H_pca - centre_per_entry, axis=-1)   # [N_ds]
        order = np.argsort(dists)
        for idx in order:
            if int(idx) not in already:
                extras.append(int(idx))
                already.add(int(idx))
                if len(extras) >= needed:
                    break
        selected = np.concatenate([selected, np.array(extras)])

    return selected[:B].astype(np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",  required=True)
    parser.add_argument("--output",  required=True)
    parser.add_argument("--budgets", nargs="+", type=int,
                        default=[1000, 5000, 10000, 25000, 50000])
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Subset of methods to run (default: all)")
    args = parser.parse_args()

    src     = Path(args.source)
    out     = Path(args.output)
    mem_dir = out / "memories"
    mem_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nselect_write_memories")
    print(f"Source: {src}  |  Budgets: {args.budgets}")

    # ── load datastore scalar fields ──────────────────────────────────────────
    print("\nLoading datastore ...")
    ds   = torch.load(src / "states" / "datastore.pt", weights_only=False)
    y    = ds["y"].numpy()
    N_ds = len(y)
    nll_gpt     = ds.get("nll_gpt",       torch.zeros(N_ds)).numpy()
    gpt_entropy = ds.get("gpt_entropy",   torch.zeros(N_ds)).numpy()
    metadata    = ds.get("metadata", [])
    print(f"  N_ds={N_ds:,}")

    # token rarity
    max_tok = int(y.max()) + 1
    counts  = np.bincount(y, minlength=max_tok)
    freq    = counts[y].astype(np.float64) / N_ds
    rarity  = -np.log(freq + 1e-10)

    # ── load utility scores (if available) ────────────────────────────────────
    util_a = util_b = None
    for mode, attr in [("A", "util_a"), ("B", "util_b")]:
        p = out / "utility" / f"write_utilities_mode{mode}.pt"
        if p.exists():
            u = torch.load(p, weights_only=False)
            if mode == "A":
                util_a = u
            else:
                util_b = u

    # ── load write scores (if available) ──────────────────────────────────────
    scores_dir = out / "write_scores"
    def load_scores(name: str) -> np.ndarray | None:
        p = scores_dir / f"{name}_scores.pt"
        if p.exists():
            return torch.load(p, weights_only=False)["scores"].numpy()
        return None

    linear_scores = load_scores("linear")
    gbr_scores    = load_scores("gbr")
    mlp_scores    = load_scores("mlp")

    # ── load hidden states for coverage (lazy, only if needed) ────────────────
    _h_cache: np.ndarray | None = None
    def get_h():
        nonlocal _h_cache
        if _h_cache is None:
            print("  Loading hidden states for coverage_memory ...")
            h = ds["h"].float()
            h = (h / (h.norm(dim=-1, keepdim=True) + 1e-8)).numpy()
            _h_cache = h
        return _h_cache

    all_methods = args.methods or [
        "random", "uniform_story", "high_gpt_loss", "high_gpt_entropy",
        "token_rarity", "coverage_memory",
        "oracle_utility_positive", "oracle_utility_net",
        "learned_write_linear", "learned_write_gbr", "learned_write_mlp",
    ]

    for B in args.budgets:
        if B > N_ds:
            print(f"\nSkipping B={B} (> N_ds={N_ds})")
            continue
        print(f"\n{'='*50}  B={B}")

        for method in all_methods:

            # -- random --
            if method == "random":
                for s in RANDOM_SEEDS:
                    sel = select_random(N_ds, B, s)
                    save_memory(mem_dir, f"random_seed{s}", B, sel)
                continue

            # -- uniform_story --
            if method == "uniform_story":
                sel = select_uniform_story(metadata, N_ds, B)
                save_memory(mem_dir, "uniform_story", B, sel)
                continue

            # -- high_gpt_loss --
            if method == "high_gpt_loss":
                sel = select_by_score(nll_gpt, B, highest=True)
                save_memory(mem_dir, "high_gpt_loss", B, sel)
                continue

            # -- high_gpt_entropy --
            if method == "high_gpt_entropy":
                sel = select_by_score(gpt_entropy, B, highest=True)
                save_memory(mem_dir, "high_gpt_entropy", B, sel)
                continue

            # -- token_rarity --
            if method == "token_rarity":
                sel = select_by_score(rarity, B, highest=True)
                save_memory(mem_dir, "token_rarity", B, sel)
                continue

            # -- coverage_memory --
            if method == "coverage_memory":
                try:
                    sel = select_coverage(get_h(), B)
                    save_memory(mem_dir, "coverage_memory", B, sel,
                                meta={"n_clusters": min(B, 2000)})
                except Exception as e:
                    print(f"  coverage_memory FAILED: {e}")
                continue

            # -- oracle_utility_positive [DIAG] --
            if method == "oracle_utility_positive":
                if util_a is None:
                    print("  oracle_utility_positive: skipped (utility not found)")
                    continue
                scores = util_a["utility_positive"].numpy()
                sel = select_by_score(scores, B, highest=True)
                save_memory(mem_dir, "oracle_utility_positive", B, sel,
                            meta={"is_diagnostic": True})
                continue

            # -- oracle_utility_net [DIAG] --
            if method == "oracle_utility_net":
                if util_a is None:
                    print("  oracle_utility_net: skipped (utility not found)")
                    continue
                scores = util_a["utility_net"].numpy()
                sel = select_by_score(scores, B, highest=True)
                save_memory(mem_dir, "oracle_utility_net", B, sel,
                            meta={"is_diagnostic": True})
                continue

            # -- learned methods --
            if method == "learned_write_linear":
                if linear_scores is None:
                    print("  learned_write_linear: skipped (scores not found)")
                    continue
                sel = select_by_score(linear_scores, B, highest=True)
                save_memory(mem_dir, "learned_write_linear", B, sel)
                continue

            if method == "learned_write_gbr":
                if gbr_scores is None:
                    print("  learned_write_gbr: skipped (scores not found)")
                    continue
                sel = select_by_score(gbr_scores, B, highest=True)
                save_memory(mem_dir, "learned_write_gbr", B, sel)
                continue

            if method == "learned_write_mlp":
                if mlp_scores is None:
                    print("  learned_write_mlp: skipped (scores not found)")
                    continue
                sel = select_by_score(mlp_scores, B, highest=True)
                save_memory(mem_dir, "learned_write_mlp", B, sel)
                continue

    print("\nDone.")


if __name__ == "__main__":
    main()

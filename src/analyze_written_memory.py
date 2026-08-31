"""
analyze_written_memory.py

Content analysis of each written memory (method x budget).

For each memory subset, reports:
  - Token frequency distribution (top-20 tokens and their counts)
  - Rare token coverage (fraction of datastore-rare tokens present)
  - GPT-loss distribution (mean, std, p25, p50, p75 of NLL values)
  - Story coverage (how many unique stories are represented)
  - Diversity estimate (mean pairwise cosine distance to random sample)

Saves:
  reports/memory_content_analysis.csv
  reports/memory_token_samples/<method>_B<B>_samples.txt

Usage:
  python src/analyze_written_memory.py \\
    --source outputs_scale_sweep/scale_200k_seed42 \\
    --output outputs_mvp2_write_memory
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import re
import numpy as np
import torch
import pandas as pd


SAMPLE_ESTIMATE_SIZE = 2_000   # entries sampled per memory for diversity estimate
RARE_PERCENTILE      = 90      # tokens in top-10% rarity are "rare"
TOP_K_TOKENS         = 20


def get_rare_token_ids(y: np.ndarray, percentile: float = 90) -> set:
    max_tok = int(y.max()) + 1
    counts  = np.bincount(y, minlength=max_tok)
    freq    = counts / len(y)
    # tokens with below-median frequency (rare = low freq)
    threshold = np.percentile(freq[freq > 0], 100 - percentile)
    rare = set(int(t) for t in np.where(freq > 0)[0] if freq[t] <= threshold)
    return rare


def diversity_estimate(h_mem: torch.Tensor, n_sample: int = SAMPLE_ESTIMATE_SIZE) -> float:
    """Mean cosine distance (1 - sim) between random pairs in memory."""
    B = len(h_mem)
    if B < 2:
        return 0.0
    n_sample = min(n_sample, B)
    idx   = torch.randperm(B)[:n_sample]
    h_sub = h_mem[idx].float()
    h_sub = h_sub / (h_sub.norm(dim=-1, keepdim=True) + 1e-8)
    # Compute pairwise similarities via matrix multiply
    # Use a subsample if too large
    max_n = min(n_sample, 500)   # avoid O(n^2) explosion
    h_sub2 = h_sub[:max_n]
    sims   = h_sub2 @ h_sub2.T   # [n, n]
    # Exclude diagonal
    mask   = ~torch.eye(max_n, dtype=torch.bool)
    mean_sim = sims[mask].mean().item()
    return 1.0 - mean_sim   # cosine distance


def token_freq_report(y_mem: np.ndarray, max_vocab: int) -> dict:
    counts  = np.bincount(y_mem, minlength=max_vocab)
    total   = len(y_mem)
    n_unique = int((counts > 0).sum())
    top_idx  = np.argsort(counts)[-TOP_K_TOKENS:][::-1]
    top_tokens = [(int(t), int(counts[t]), float(counts[t] / total))
                  for t in top_idx if counts[t] > 0]
    return {
        "n_unique_tokens": n_unique,
        "top_tokens":      top_tokens,   # list of (token_id, count, freq)
    }


def analyze_one_memory(
    mem_path: Path,
    ds: dict,
    rare_token_ids: set,
    max_vocab: int,
) -> dict:
    mem     = torch.load(mem_path, weights_only=False)
    sel_idx = mem["selected_indices"].numpy()
    B       = len(sel_idx)
    method  = mem["method"]
    budget  = mem["budget"]

    y_mem   = ds["y"].numpy()[sel_idx]
    h_mem   = ds["h"][sel_idx]

    nll_mem = None
    if "nll_gpt" in ds:
        nll_mem = ds["nll_gpt"].numpy()[sel_idx]

    metadata = ds.get("metadata", [])
    story_ids = None
    if metadata and len(metadata) == len(ds["y"]):
        story_ids = np.array([metadata[i].get("story_id", -1)
                              for i in sel_idx], dtype=np.int64)

    # Token rarity coverage
    mem_toks       = set(int(t) for t in y_mem)
    rare_in_mem    = mem_toks & rare_token_ids
    rare_coverage  = len(rare_in_mem) / len(rare_token_ids) if rare_token_ids else 0.0

    # Token distribution
    tok_report = token_freq_report(y_mem, max_vocab)

    # GPT-loss distribution
    nll_stats: dict = {}
    if nll_mem is not None and len(nll_mem) > 0:
        nll_stats = {
            "nll_mean": float(nll_mem.mean()),
            "nll_std":  float(nll_mem.std()),
            "nll_p25":  float(np.percentile(nll_mem, 25)),
            "nll_p50":  float(np.percentile(nll_mem, 50)),
            "nll_p75":  float(np.percentile(nll_mem, 75)),
        }

    # Story coverage
    n_stories = int(np.unique(story_ids).shape[0]) if story_ids is not None else -1

    # Diversity estimate
    diversity = diversity_estimate(h_mem)

    row = {
        "method":          method,
        "budget":          budget,
        "n_unique_tokens": tok_report["n_unique_tokens"],
        "rare_coverage":   rare_coverage,
        "n_stories":       n_stories,
        "diversity_estimate": diversity,
    }
    row.update(nll_stats)

    return row, tok_report["top_tokens"], method, budget


def write_token_sample(
    sample_dir: Path,
    method: str,
    budget: int,
    top_tokens: list,
    try_decode: bool = True,
):
    fname = sample_dir / f"{method}_B{budget}_samples.txt"
    lines = [f"Method: {method}  Budget: {budget}\n",
             f"Top-{TOP_K_TOKENS} tokens (id, count, freq):\n"]
    for tok_id, count, freq in top_tokens:
        tok_str = f"[{tok_id}]"
        if try_decode:
            try:
                from transformers import GPT2Tokenizer
                tok = GPT2Tokenizer.from_pretrained("gpt2")
                tok_str = repr(tok.decode([tok_id]))
            except Exception:
                pass
        lines.append(f"  {tok_str:<20}  count={count:>8,}  freq={freq:.5f}\n")
    with open(fname, "w", encoding="utf-8") as f:
        f.writelines(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",  required=True)
    parser.add_argument("--output",  required=True)
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Filter by method name substring")
    args = parser.parse_args()

    src = Path(args.source)
    out = Path(args.output)
    mem_dir    = out / "memories"
    report_dir = out / "reports"
    sample_dir = report_dir / "memory_token_samples"
    report_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(exist_ok=True)

    print(f"\nanalyze_written_memory")
    print(f"Source: {src}  |  Output: {out}")

    # ── load datastore ────────────────────────────────────────────────────────
    print("\nLoading datastore ...")
    ds   = torch.load(src / "states" / "datastore.pt", weights_only=False)
    N_ds = len(ds["y"])
    y_ds = ds["y"].numpy()
    print(f"  N_ds={N_ds:,}")

    max_vocab      = int(y_ds.max()) + 1
    rare_token_ids = get_rare_token_ids(y_ds, RARE_PERCENTILE)
    print(f"  Rare tokens (top-{RARE_PERCENTILE}% rarity): {len(rare_token_ids):,}")

    # ── discover memory files ─────────────────────────────────────────────────
    mem_files = sorted(mem_dir.glob("*.pt"))
    if not mem_files:
        print(f"\nNo memory files found in {mem_dir}")
        print("Run select_write_memories.py first.")
        return

    if args.methods:
        mem_files = [p for p in mem_files
                     if any(m in p.stem for m in args.methods)]

    print(f"\nAnalyzing {len(mem_files)} memory files ...")

    rows = []
    for mem_path in mem_files:
        try:
            print(f"  {mem_path.stem} ...", end=" ")
            row, top_tokens, method, budget = analyze_one_memory(
                mem_path, ds, rare_token_ids, max_vocab,
            )
            rows.append(row)
            write_token_sample(sample_dir, method, budget, top_tokens)
            print(f"stories={row['n_stories']:,}  "
                  f"rare_cov={row['rare_coverage']:.3f}  "
                  f"diversity={row['diversity_estimate']:.3f}")
        except Exception as e:
            print(f"ERROR: {e}")

    if not rows:
        print("No results. Exiting.")
        return

    df = pd.DataFrame(rows)
    # reorder columns sensibly
    front_cols = ["method", "budget", "n_unique_tokens", "rare_coverage",
                  "n_stories", "diversity_estimate"]
    nll_cols   = [c for c in df.columns if "nll" in c]
    df = df[front_cols + nll_cols]
    df.to_csv(report_dir / "memory_content_analysis.csv", index=False)
    print(f"\nSaved: reports/memory_content_analysis.csv  ({len(df)} rows)")

    # ── terminal summary table ────────────────────────────────────────────────
    print(f"\n{'Method':<35} {'Budget':>7} {'UniqueTokens':>12} {'RareCov':>8} "
          f"{'Stories':>8} {'Diversity':>10}")
    print("-" * 90)
    for _, r in df.sort_values(["method", "budget"]).iterrows():
        nll_str = f"{r['nll_p50']:.3f}" if "nll_p50" in r and pd.notna(r.get("nll_p50")) else "—"
        print(f"{r['method']:<35} {r['budget']:>7,} {r['n_unique_tokens']:>12,} "
              f"{r['rare_coverage']:>8.3f} {r['n_stories']:>8,} "
              f"{r['diversity_estimate']:>10.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()

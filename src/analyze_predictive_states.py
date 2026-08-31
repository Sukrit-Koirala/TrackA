"""
analyze_predictive_states.py

State quality diagnostics for persistent predictive states.

Computes per-state and per-method-budget aggregate stats:
  - entropy and purity distributions
  - count distribution (min/p25/median/p75/max)
  - rare token coverage (fraction of vocab 90th-pct tokens covered)
  - true-token-in-nearest-state hit rate across CT/val queries
  - inter-state prototype diversity (cosine distances)

Outputs:
  <output>/reports/state_quality_analysis.csv
  <output>/reports/state_method_summary.csv

Usage:
  python src/analyze_predictive_states.py \\
    --source outputs_scale_sweep/scale_200k_seed42 \\
    --output outputs_mvp2b_state_write
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import re
import numpy as np
import torch
import pandas as pd

from utils import get_device, set_seed
from build_neighbors import cosine_top_k

VOCAB_SIZE       = 50257
RARITY_PERCENTILE = 90
N_SAMPLE_DIVERSE  = 2000


# ── helpers ────────────────────────────────────────────────────────────────────

def parse_method_budget(stem: str):
    m = re.match(r"^(.+)_B(\d+)$", stem)
    if m:
        return m.group(1), int(m.group(2))
    return stem, 0


def get_rare_token_ids(y: torch.Tensor, percentile: float = RARITY_PERCENTILE) -> set:
    counts = torch.bincount(y.long(), minlength=VOCAB_SIZE).float()
    thresh = float(np.percentile(counts[counts > 0].numpy(), percentile))
    return set(int(t) for t in (counts <= thresh).nonzero(as_tuple=True)[0])


def diversity_estimate(proto_h: torch.Tensor, n_sample: int = N_SAMPLE_DIVERSE) -> float:
    B = len(proto_h)
    if B <= 1:
        return 0.0
    idx = torch.randperm(B)[:min(n_sample, B)]
    h   = proto_h[idx].float()
    h   = h / (h.norm(dim=-1, keepdim=True) + 1e-8)
    sim = h @ h.t()
    mask = torch.triu(torch.ones_like(sim, dtype=torch.bool), diagonal=1)
    sims = sim[mask]
    return float(1.0 - sims.mean())   # 1 - mean cosine = mean cosine distance


def rare_token_coverage(top_k_token_ids: torch.Tensor, rare_set: set) -> float:
    if not rare_set:
        return 0.0
    all_ids = set(int(t) for t in top_k_token_ids.flatten() if int(t) != 0)
    covered = all_ids & rare_set
    return len(covered) / len(rare_set)


def count_distribution(total_counts: torch.Tensor) -> dict:
    c = total_counts.float().numpy()
    return {
        "count_min":    float(c.min()),
        "count_p25":    float(np.percentile(c, 25)),
        "count_median": float(np.median(c)),
        "count_p75":    float(np.percentile(c, 75)),
        "count_max":    float(c.max()),
        "count_mean":   float(c.mean()),
        "count_std":    float(c.std()),
        "empty_states": int((c == 0).sum()),
    }


def true_token_hit_rate(h_norm: torch.Tensor,
                         y: torch.Tensor,
                         proto: torch.Tensor,
                         states: dict,
                         k: int,
                         device: torch.device,
                         chunk_size: int = 512) -> float:
    """Fraction of queries where y_true appears in any of top-k states."""
    from evaluate_predictive_states import compute_top_k_state_sims, precompute_lookup

    P_global = torch.ones(VOCAB_SIZE) / VOCAB_SIZE  # dummy; doesn't affect hit rate

    top_ids, top_sims = compute_top_k_state_sims(h_norm, proto, k, device)

    tok_ids   = states["top_k_token_ids"]  # [B, TOP_K]
    tok_cnts  = states["top_k_token_counts"]
    sel_total = states["total_counts"]
    TOP_K     = tok_ids.shape[1]
    N         = len(y)
    K_eff     = top_ids.shape[1]

    hits = 0
    for start in range(0, N, chunk_size):
        end    = min(start + chunk_size, N)
        ids_c  = top_ids[start:end]
        y_c    = y[start:end]
        C      = end - start

        sel_tids = tok_ids[ids_c]     # [C, K_eff, TOP_K]
        y_exp    = y_c.view(C, 1, 1).expand(C, K_eff, TOP_K)
        match    = (sel_tids.long() == y_exp.long()).any(-1).any(-1)  # [C]
        hits    += int(match.sum())

    return hits / N


# ── main analysis ─────────────────────────────────────────────────────────────

def analyze_state_file(sf: Path, ct_data: dict, val_data: dict,
                        rare_set: set, device: torch.device) -> dict:
    method, budget = parse_method_budget(sf.stem)
    states = torch.load(sf, weights_only=False)

    proto      = states["prototype_h"]        # [B, D]
    tok_ids    = states["top_k_token_ids"]    # [B, TOP_K]
    tok_cnts   = states["top_k_token_counts"] # [B, TOP_K]
    total_cnts = states["total_counts"]       # [B]
    B          = len(proto)

    ent   = states["state_entropy"].float()
    purity= states["state_purity"].float()

    def norm_h(data):
        h = data["h"].float()
        return h / (h.norm(dim=-1, keepdim=True) + 1e-8)

    ct_h_norm  = norm_h(ct_data)
    val_h_norm = norm_h(val_data)

    row = {
        "method": method,
        "budget": budget,
        "n_states": B,

        # Entropy stats
        "entropy_mean":   float(ent.mean()),
        "entropy_median": float(ent.median()),
        "entropy_p90":    float(torch.quantile(ent, 0.9)),

        # Purity stats
        "purity_mean":    float(purity.mean()),
        "purity_median":  float(purity.median()),
        "purity_p10":     float(torch.quantile(purity, 0.1)),

        # Diversity
        "prototype_diversity": diversity_estimate(proto),

        # Rare token coverage
        "rare_token_coverage": rare_token_coverage(tok_ids, rare_set),
    }

    # Count distribution
    row.update(count_distribution(total_cnts))

    # Hit rates at k=1, 4, 16
    for k in [1, 4, 16]:
        k_eff = min(k, B)
        ct_hr  = true_token_hit_rate(ct_h_norm,  ct_data["y"],  proto, states, k_eff, device)
        val_hr = true_token_hit_rate(val_h_norm, val_data["y"], proto, states, k_eff, device)
        row[f"ct_hit_rate_k{k}"]  = ct_hr
        row[f"val_hit_rate_k{k}"] = val_hr

    print(f"    {sf.stem}: B={B}  purity_med={row['purity_median']:.3f}  "
          f"div={row['prototype_diversity']:.3f}  "
          f"val_hit_k4={row.get('val_hit_rate_k4', 0):.3f}  "
          f"rare_cov={row['rare_token_coverage']:.3f}")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",  required=True)
    parser.add_argument("--output",  required=True)
    parser.add_argument("--force",   action="store_true")
    args = parser.parse_args()

    src = Path(args.source)
    out = Path(args.output)
    (out / "reports").mkdir(parents=True, exist_ok=True)

    device = get_device({"device": "cuda"})
    set_seed(42)

    print(f"\nanalyze_predictive_states")
    print(f"Source: {src}")

    ct_data  = torch.load(src / "states" / "controller_train.pt", weights_only=False)
    val_data = torch.load(src / "states" / "val.pt",              weights_only=False)
    ds_data  = torch.load(src / "states" / "datastore.pt",        weights_only=False)

    rare_set = get_rare_token_ids(ds_data["y"], RARITY_PERCENTILE)
    print(f"  Rare token set size: {len(rare_set):,}  (top {RARITY_PERCENTILE}th pct of freq distribution)")

    states_dir = out / "states"
    state_files = sorted(states_dir.glob("*.pt"))
    if not state_files:
        print(f"No state files in {states_dir}.")
        return
    print(f"  Analyzing {len(state_files)} state files ...")

    rows = []
    for sf in state_files:
        method, budget = parse_method_budget(sf.stem)
        if budget == 0:
            continue
        print(f"\n  [{sf.stem}]")
        row = analyze_state_file(sf, ct_data, val_data, rare_set, device)
        rows.append(row)

    if not rows:
        print("No results.")
        return

    df = pd.DataFrame(rows)
    path1 = out / "reports" / "state_quality_analysis.csv"
    df.to_csv(path1, index=False)
    print(f"\nSaved: {path1}  ({len(df)} rows)")

    # Method-level summary (best budget per method)
    summary = []
    for meth in sorted(df["method"].unique()):
        mdf = df[df["method"] == meth]
        best = mdf.loc[mdf["val_hit_rate_k4"].idxmax()] if "val_hit_rate_k4" in mdf else mdf.iloc[0]
        summary.append({
            "method":                meth,
            "best_budget":           int(best["budget"]),
            "best_val_hit_k4":       float(best.get("val_hit_rate_k4", 0)),
            "mean_purity_all_budgets": float(mdf["purity_median"].mean()),
            "mean_diversity":         float(mdf["prototype_diversity"].mean()),
            "mean_rare_coverage":     float(mdf["rare_token_coverage"].mean()),
        })
    sdf = pd.DataFrame(summary)
    path2 = out / "reports" / "state_method_summary.csv"
    sdf.to_csv(path2, index=False)
    print(f"Saved: {path2}")

    print(f"\n{'='*60}")
    print(f"  {'Method':<30} {'HitK4':>6} {'Purity':>7} {'Div':>6} {'RareCov':>8}")
    print(f"  {'-'*60}")
    for _, r in sdf.iterrows():
        print(f"  {r['method']:<30} {r['best_val_hit_k4']:>6.3f} "
              f"{r['mean_purity_all_budgets']:>7.3f} "
              f"{r['mean_diversity']:>6.3f} {r['mean_rare_coverage']:>8.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

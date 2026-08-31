"""
run_state_object_ablations.py  --  Track A Paper  (Experiment 4)

Proves that the useful memory entry is the PREDICTIVE STATE object
(prototype + empirical next-token distribution), not merely a cluster centroid.

Ablation variants:
  original             -- full state object as-built (reference)
  majority_token       -- one-hot on the most common token per state
  top4/8/16/32/64      -- sparse top-M token distribution
  shuffled_distribution-- shuffle token distributions across states (keep prototype)
  shuffled_prototype   -- shuffle prototypes across states (keep distribution)
  global_unigram       -- replace every state dist with global corpus unigram
  random_partition     -- random group assignment → new centroids + new distributions

Usage:
  python src/paper/run_state_object_ablations.py \\
    --source outputs_track_a_offline_paper/seed42 \\
    --output outputs_track_a_offline_paper/seed42 \\
    --methods minibatch_kmeans utility_weighted query_kmeans \\
    --budgets 5000 10000 25000 50000 \\
    --variants original majority_token top4 top8 top16 top32 top64 \\
               shuffled_distribution shuffled_prototype global_unigram random_partition \\
    --max_q_samples 500000 \\
    --device cuda --seed 42
"""

import sys
import math
import json
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import torch
import pandas as pd

from utils import get_device, set_seed
from train_q_state_read import run_method_budget

SCRIPT_DIR = Path(__file__).parent

# All supported variant names in display order
ALL_VARIANTS = [
    "original",
    "majority_token",
    "top4", "top8", "top16", "top32", "top64",
    "shuffled_distribution",
    "shuffled_prototype",
    "global_unigram",
    "random_partition",
]

TOP_M_MAP = {
    "top4": 4, "top8": 8, "top16": 16, "top32": 32, "top64": 64,
}


# ── ablation constructors ─────────────────────────────────────────────────────

def _clone(sf: dict) -> dict:
    return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in sf.items()}


def ablate_majority_token(sf: dict) -> dict:
    """One-hot on the most frequent token per state."""
    out  = _clone(sf)
    cnts = sf["top_k_token_counts"].clone()  # [B, TOP_K]
    new  = torch.zeros_like(cnts)
    new[:, 0] = cnts[:, 0]                  # top-1 already at index 0
    out["top_k_token_counts"] = new
    out["state_purity"]       = torch.ones(cnts.shape[0],  dtype=torch.float32)
    out["state_entropy"]      = torch.zeros(cnts.shape[0], dtype=torch.float32)
    return out


def ablate_top_m(sf: dict, M: int) -> dict:
    """Keep only the top-M tokens per state; zero the rest."""
    out  = _clone(sf)
    cnts = sf["top_k_token_counts"].clone()
    cnts[:, M:] = 0.0
    new_tot = cnts.sum(-1).clamp(min=1e-8)
    top1    = cnts[:, 0]
    out["top_k_token_counts"] = cnts
    out["total_counts"]       = new_tot
    out["state_purity"]       = top1 / new_tot
    p_norm  = cnts / new_tot.unsqueeze(1)
    ent     = -(p_norm * (p_norm + 1e-10).log()).sum(-1)
    out["state_entropy"] = ent
    return out


def ablate_shuffled_distribution(sf: dict, seed: int = 42) -> dict:
    """Keep real prototypes; permute token distributions across states."""
    rng  = np.random.default_rng(seed)
    B    = sf["top_k_token_ids"].shape[0]
    perm = torch.from_numpy(rng.permutation(B))
    out  = _clone(sf)
    out["top_k_token_ids"]    = sf["top_k_token_ids"][perm].clone()
    out["top_k_token_counts"] = sf["top_k_token_counts"][perm].clone()
    out["total_counts"]       = sf["total_counts"][perm].clone()
    out["state_entropy"]      = sf["state_entropy"][perm].clone()
    out["state_purity"]       = sf["state_purity"][perm].clone()
    return out


def ablate_shuffled_prototype(sf: dict, seed: int = 42) -> dict:
    """Permute prototype_h across states; keep distributions fixed."""
    rng  = np.random.default_rng(seed)
    B    = sf["prototype_h"].shape[0]
    perm = torch.from_numpy(rng.permutation(B))
    out  = _clone(sf)
    out["prototype_h"] = sf["prototype_h"][perm].clone()
    return out


def ablate_global_unigram(sf: dict, ds_data: dict) -> dict:
    """Replace every state's distribution with the corpus-wide unigram."""
    y          = ds_data["y"].numpy()
    vocab_size = max(int(y.max()) + 1, 50257)
    B          = sf["prototype_h"].shape[0]
    TOP_K      = sf["top_k_token_ids"].shape[1]

    global_cnt = np.bincount(y, minlength=vocab_size).astype(np.float32)
    total      = float(global_cnt.sum())
    k_use      = min(TOP_K, int((global_cnt > 0).sum()))
    topk_idx   = np.argsort(global_cnt)[::-1][:k_use].astype(np.int32)
    topk_cnt   = global_cnt[topk_idx]

    new_ids  = torch.zeros(B, TOP_K, dtype=torch.int32)
    new_cnts = torch.zeros(B, TOP_K, dtype=torch.float32)
    new_ids[:, :k_use]  = torch.from_numpy(topk_idx)
    new_cnts[:, :k_use] = torch.from_numpy(topk_cnt)

    new_tot = torch.full((B,), total)
    p       = torch.zeros(B, TOP_K)
    p[:, :k_use] = torch.from_numpy(topk_cnt / max(total, 1e-8)).unsqueeze(0).expand(B, -1)
    ent     = -(p * (p + 1e-10).log()).sum(-1)
    purity  = torch.full((B,), float(topk_cnt[0] / max(total, 1e-8)) if k_use > 0 else 0.0)

    out = _clone(sf)
    out["top_k_token_ids"]    = new_ids
    out["top_k_token_counts"] = new_cnts
    out["total_counts"]       = new_tot
    out["state_entropy"]      = ent
    out["state_purity"]       = purity
    return out


def ablate_random_partition(sf: dict, ds_data: dict, seed: int = 42) -> dict:
    """
    Random partition: randomly assign datastore examples to B groups,
    then recompute BOTH centroids (mean h) and distributions (empirical y).
    Tests whether the geometric+distributional structure of clustering matters.
    """
    rng        = np.random.default_rng(seed)
    B          = sf["prototype_h"].shape[0]
    TOP_K      = sf["top_k_token_ids"].shape[1]
    h          = ds_data["h"].float()   # [N, D]
    y          = ds_data["y"].numpy()   # [N]
    N          = len(y)
    vocab_size = max(int(y.max()) + 1, 50257)

    labels = rng.integers(0, B, size=N)

    # New centroids
    new_h   = torch.zeros(B, h.shape[1])
    for s in range(B):
        mask = labels == s
        if mask.any():
            new_h[s] = h[torch.from_numpy(np.where(mask)[0])].mean(0)
    new_h_norm = new_h / (new_h.norm(dim=-1, keepdim=True) + 1e-8)

    # New distributions
    new_ids  = torch.zeros(B, TOP_K, dtype=torch.int32)
    new_cnts = torch.zeros(B, TOP_K, dtype=torch.float32)
    new_tot  = torch.zeros(B, dtype=torch.float32)
    for s in range(B):
        mask = labels == s
        if not mask.any():
            continue
        cnt    = np.bincount(y[mask], minlength=vocab_size).astype(np.float32)
        total  = cnt.sum()
        k_use  = min(TOP_K, int((cnt > 0).sum()))
        idxs   = np.argsort(cnt)[::-1][:k_use].astype(np.int32)
        new_ids[s, :k_use]  = torch.from_numpy(idxs)
        new_cnts[s, :k_use] = torch.from_numpy(cnt[idxs])
        new_tot[s]           = float(total)

    p_norm = new_cnts / (new_tot.unsqueeze(1).clamp(min=1e-8))
    ent    = -(p_norm * (p_norm + 1e-10).log()).sum(-1)
    purity = new_cnts[:, 0] / new_tot.clamp(min=1e-8)

    out = _clone(sf)
    out["prototype_h"]        = new_h_norm
    out["top_k_token_ids"]    = new_ids
    out["top_k_token_counts"] = new_cnts
    out["total_counts"]       = new_tot.clamp(min=1.0)
    out["state_entropy"]      = ent
    out["state_purity"]       = purity
    return out


# ── summary / reporting ───────────────────────────────────────────────────────

def _fmt(v, decimals=4) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    return f"{float(v):.{decimals}f}"


def _fmtd(v, decimals=4) -> str:
    """Format a delta value with explicit sign."""
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    return f"{float(v):+.{decimals}f}"


def _compute_verdict(df: pd.DataFrame) -> str:
    """STRONG SUPPORT / SUPPORT / MIXED / FAIL."""
    if df.empty:
        return "NO_DATA"

    orig = df[df["variant"] == "original"]["q_nll"]
    orig_q = orig.mean() if not orig.empty else float("nan")

    def _mean_q(v: str) -> float:
        rows = df[df["variant"] == v]["q_nll"]
        return rows.mean() if not rows.empty else float("nan")

    maj_q    = _mean_q("majority_token")
    top32_q  = _mean_q("top32")
    shuf_d_q = _mean_q("shuffled_distribution")
    shuf_p_q = _mean_q("shuffled_prototype")
    rpart_q  = _mean_q("random_partition")

    if not math.isfinite(orig_q):
        return "NO_DATA"

    checks = {
        "majority_worse":          math.isfinite(maj_q)    and maj_q    > orig_q,
        "shuffled_dist_worse":     math.isfinite(shuf_d_q) and shuf_d_q > orig_q,
        "shuffled_proto_worse":    math.isfinite(shuf_p_q) and shuf_p_q > orig_q,
        "top32_close":             math.isfinite(top32_q)  and (top32_q - orig_q) <= 0.02,
        "random_partition_worse":  math.isfinite(rpart_q)  and rpart_q  > orig_q,
    }

    n_pass = sum(checks.values())

    # FAIL: shuffled distribution or majority matches/beats original
    if (math.isfinite(shuf_d_q) and shuf_d_q <= orig_q) or \
       (math.isfinite(maj_q) and maj_q <= orig_q):
        return "FAIL"

    if n_pass == len(checks):
        return "STRONG_SUPPORT"
    elif n_pass >= 3:
        return "SUPPORT"
    elif n_pass >= 2:
        return "MIXED"
    return "FAIL"


def _build_summary(df: pd.DataFrame, gpt_nll: float, fr_q_nll: float,
                   dataset: str, model: str) -> str:
    """Build the three tables + conclusion section."""
    lines = [
        f"# State-Object Ablation Results\n",
        f"Dataset: {dataset}  |  Model: {model}\n",
    ]

    orig_q = df[df["variant"] == "original"]["q_nll"].mean()

    # ── Table 1: Variant performance ─────────────────────────────────────────
    lines += [
        "## Table 1: Variant Performance\n",
        "| Variant | Mean Q NLL | Δ vs Original | Verdict |",
        "|---------|------------|---------------|---------|",
    ]
    for variant in ALL_VARIANTS:
        rows = df[df["variant"] == variant]["q_nll"]
        if rows.empty:
            lines.append(f"| {variant:<28} | — | — | — |")
            continue
        q      = rows.mean()
        delta  = q - orig_q if math.isfinite(orig_q) else float("nan")
        if variant == "original":
            verdict = "reference"
        elif not math.isfinite(delta):
            verdict = "—"
        elif delta < -0.005:
            verdict = "beats original"
        elif delta <= 0.005:
            verdict = "same"
        elif delta <= 0.02:
            verdict = "slightly worse"
        elif delta <= 0.08:
            verdict = "worse"
        else:
            verdict = "much worse"
        lines.append(f"| {variant:<28} | {_fmt(q)} | {_fmtd(delta):>8} | {verdict} |")

    lines.append("")

    # ── Table 2: Sparse distribution tradeoff ────────────────────────────────
    lines += [
        "## Table 2: Sparse Distribution Tradeoff\n",
        "| Top-M | Q NLL | Δ vs Full Dist | Verdict |",
        "|-------|-------|----------------|---------|",
    ]
    full_q = df[df["variant"] == "original"]["q_nll"].mean()
    for variant, M in [("top4", 4), ("top8", 8), ("top16", 16),
                        ("top32", 32), ("top64", 64), ("original", "full")]:
        rows = df[df["variant"] == variant]["q_nll"]
        if rows.empty:
            lines.append(f"| {str(M):<6} | — | — | — |")
            continue
        q     = rows.mean()
        delta = q - full_q if variant != "original" and math.isfinite(full_q) else 0.0
        pv    = "reference" if variant == "original" else (
                "≈ full" if abs(delta) <= 0.02 else
                "close"  if delta <= 0.05 else "worse")
        lines.append(f"| {str(M):<6} | {_fmt(q)} | {_fmtd(delta) if variant != 'original' else '+0.0000':>8} | {pv} |")

    lines.append("")

    # ── Table 3: Causal mechanism checks ────────────────────────────────────
    def _mean(v: str) -> float:
        rows = df[df["variant"] == v]["q_nll"]
        return rows.mean() if not rows.empty else float("nan")

    maj_q    = _mean("majority_token")
    shuf_d_q = _mean("shuffled_distribution")
    shuf_p_q = _mean("shuffled_prototype")
    top32_q  = _mean("top32")
    rpart_q  = _mean("random_partition")

    checks = [
        ("majority worse than original",        "yes", maj_q    > orig_q if math.isfinite(maj_q)    else None),
        ("shuffled distribution worse",          "yes", shuf_d_q > orig_q if math.isfinite(shuf_d_q) else None),
        ("shuffled prototype worse",             "yes", shuf_p_q > orig_q if math.isfinite(shuf_p_q) else None),
        ("top32 close to original (≤+0.02 NLL)","yes", (top32_q - orig_q) <= 0.02 if math.isfinite(top32_q) else None),
        ("random partition worse than original", "yes", rpart_q  > orig_q if math.isfinite(rpart_q)  else None),
    ]

    lines += [
        "## Table 3: Causal Mechanism Checks\n",
        "| Check | Expected | Observed | Passed? |",
        "|-------|----------|----------|---------|",
    ]
    for desc, expected, passed in checks:
        obs = "yes" if passed else ("no" if passed is not None else "—")
        ok  = "✓" if passed else ("✗" if passed is not None else "—")
        lines.append(f"| {desc:<48} | {expected} | {obs} | {ok} |")

    lines.append("")

    # ── Verdict ──────────────────────────────────────────────────────────────
    verdict = _compute_verdict(df)
    lines += [
        f"## Ablation Verdict: {verdict}\n",
    ]
    if verdict == "STRONG_SUPPORT":
        lines.append(
            "All mechanism checks pass. Empirical distributions and correct "
            "prototype-distribution pairing are necessary. Sparse top-32 distributions "
            "preserve most of the gain.")
    elif verdict == "SUPPORT":
        lines.append(
            "Most mechanism checks pass. The empirical distribution contributes to the gain.")
    elif verdict == "MIXED":
        lines.append(
            "Mixed evidence. Some ablations are close to original, "
            "suggesting the gain is not fully explained by distribution content.")
    elif verdict == "FAIL":
        lines.append(
            "FAIL: shuffled distribution or majority-token matches/beats original. "
            "The gain may not require state-specific prediction distributions.")

    return "\n".join(lines)


def _print_final_summary(df: pd.DataFrame, dataset: str, model: str,
                          seeds: list[int], gpt_nll: float, fr_q_nll: float):
    verdict = _compute_verdict(df)

    def _mean(v: str) -> float:
        rows = df[df["variant"] == v]["q_nll"]
        return rows.mean() if not rows.empty else float("nan")

    orig_q   = _mean("original")
    maj_q    = _mean("majority_token")
    top32_q  = _mean("top32")
    shuf_d_q = _mean("shuffled_distribution")
    shuf_p_q = _mean("shuffled_prototype")
    rpart_q  = _mean("random_partition")

    dist_matters  = "yes" if math.isfinite(shuf_d_q) and shuf_d_q > orig_q + 0.01 else ("no" if math.isfinite(shuf_d_q) else "—")
    pair_matters  = "yes" if math.isfinite(shuf_p_q) and shuf_p_q > orig_q + 0.01 else ("no" if math.isfinite(shuf_p_q) else "—")
    sparse_ok     = "yes" if math.isfinite(top32_q) and (top32_q - orig_q) <= 0.02 else ("no" if math.isfinite(top32_q) else "—")

    print(f"\n{'='*60}")
    print(f"  State-Object Ablation Result")
    print(f"  {'─'*40}")
    print(f"  Dataset:                          {dataset}")
    print(f"  Model:                            {model}")
    print(f"  Seeds:                            {seeds}")
    print(f"  Best original Q:                  {_fmt(orig_q)}")
    print(f"  Best majority-token Q:            {_fmt(maj_q)}")
    print(f"  Best top32 Q:                     {_fmt(top32_q)}")
    print(f"  Best shuffled-distribution Q:     {_fmt(shuf_d_q)}")
    print(f"  Best shuffled-prototype Q:        {_fmt(shuf_p_q)}")
    print(f"  Best random-partition Q:          {_fmt(rpart_q)}")
    print(f"  Does empirical distribution matter? {dist_matters}")
    print(f"  Does prototype-distribution pairing matter? {pair_matters}")
    print(f"  Can sparse top-M preserve performance? {sparse_ok}")
    print(f"  Verdict:                          {verdict}")
    print(f"{'='*60}")

    print(f"\n  Paper implication:")
    if verdict == "STRONG_SUPPORT":
        print(f"    The predictive state codebook advantage is not explained by clustering "
              f"alone; the empirical next-token distribution and correct "
              f"prototype-distribution pairing are necessary for the gain.")
    elif verdict == "SUPPORT":
        print(f"    Empirical distributions contribute to the gain; sparse top-M "
              f"distributions can preserve most of the benefit.")
    elif verdict == "MIXED":
        print(f"    Some gains come from prototype compression and adaptive reading, "
              f"while empirical distributions provide additional benefit in specific regimes.")
    else:
        print(f"    WARNING: ablation results are weaker than expected. "
              f"Review before making strong distribution-content claims.")


# ── data-source discovery ────────────────────────────────────────────────────

def _find_data_source(explicit: Path | None, src: Path) -> Path:
    """
    Locate the directory that contains states/controller_train.pt.

    Search order:
      1. --data_source if given
      2. src itself (WikiText-2: run_dataset_replication.py extracts here)
      3. Any sibling directory in CWD whose states/ has controller_train.pt
         (TinyStories: scale_200k_seedXX lives alongside the output dirs)
    """
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.append(src)

    for c in candidates:
        if (c / "states" / "controller_train.pt").exists():
            return c

    # Fallback: scan CWD siblings (covers scale_200k_seed42 next to outputs_…)
    cwd = Path.cwd()
    for sibling in sorted(cwd.iterdir()):
        if sibling.is_dir() and (sibling / "states" / "controller_train.pt").exists():
            print(f"  [auto] found controller_train.pt in sibling dir: {sibling}")
            return sibling

    # Nothing found — error clearly
    searched = [str(c) for c in candidates] + [f"{cwd}/<siblings>"]
    print(f"  [ERROR] controller_train.pt not found. Searched:")
    for p in searched:
        print(f"          {p}/states/controller_train.pt")
    print(f"          Pass --data_source <dir> explicitly.")
    sys.exit(1)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",         required=True,
                        help="Per-seed paper output dir (states/, ablations/ live here)")
    parser.add_argument("--data_source",    default=None,
                        help="Dir with states/controller_train.pt and val.pt. "
                             "Defaults to --source. Override for TinyStories where "
                             "extraction lives in the original scale_200k dir.")
    parser.add_argument("--output",         required=True,
                        help="Per-seed output directory (ablations go here)")
    parser.add_argument("--states_dir",     default=None,
                        help="State .pt files (default: <output>/states/)")
    parser.add_argument("--methods",        nargs="+",
                        default=["minibatch_kmeans", "utility_weighted", "query_kmeans"])
    parser.add_argument("--budgets",        nargs="+", type=int,
                        default=[5000, 10000, 25000])
    parser.add_argument("--variants",       nargs="+", default=ALL_VARIANTS,
                        choices=ALL_VARIANTS, metavar="VARIANT")
    parser.add_argument("--dataset",        default="TinyStories")
    parser.add_argument("--model",          default="gpt2")
    parser.add_argument("--max_q_samples",  type=int, default=500_000)
    parser.add_argument("--n_epochs",       type=int, default=30)
    parser.add_argument("--fast_grid",      action="store_true", default=True)
    parser.add_argument("--no_fast_grid",   action="store_false", dest="fast_grid")
    parser.add_argument("--device",         default="cuda")
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--force",          action="store_true")
    parser.add_argument("--allow_missing",  action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device     = get_device({"device": args.device})
    src        = Path(args.source)
    out        = Path(args.output)
    states_dir = Path(args.states_dir) if args.states_dir else out / "states"

    # data_src is the dir containing states/controller_train.pt + val.pt.
    # For WikiText-2 (run_dataset_replication.py) these live in src itself.
    # For TinyStories (scale_200k pipeline) they live in the original extraction dir.
    data_src = _find_data_source(
        explicit=Path(args.data_source) if args.data_source else None,
        src=src,
    )
    abl_dir    = out / "ablations"
    abl_states = abl_dir / "state_objects"
    q_out      = abl_dir / "q_read"
    rep_dir    = out / "reports"

    for d in (abl_dir, abl_states, q_out, rep_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"\nrun_state_object_ablations")
    print(f"  Source:     {src}")
    print(f"  Data src:   {data_src}")
    print(f"  States:     {states_dir}")
    print(f"  Output:     {abl_dir}")
    print(f"  Variants: {args.variants}")
    print(f"  Methods:  {args.methods}")
    print(f"  Budgets:  {args.budgets}")

    # Load context: GPT NLL and full-raw Q NLL for delta columns
    gpt_nll  = float("nan")
    fr_q_nll = float("nan")
    extr_s   = src / "extraction_summary.json"
    if extr_s.exists():
        with open(extr_s) as f:
            gpt_nll = json.load(f).get("gpt_nll") or float("nan")
    fr_sum = src / "q_read" / "full_raw" / "evaluate_q_read_summary.json"
    if fr_sum.exists():
        with open(fr_sum) as f:
            d = json.load(f)
        rr = d.get("results", [{}])[0] if d.get("results") else {}
        fr_q_nll = rr.get("q_state_nll", float("nan")) or float("nan")

    # Decide if we need the datastore
    needs_ds = any(v in args.variants
                   for v in ("global_unigram", "random_partition"))
    ds_data  = None
    if needs_ds:
        # Check src first, then data_src (scale_200k dir for TinyStories)
        for ds_candidate in [src / "states" / "datastore.pt",
                              data_src / "states" / "datastore.pt"]:
            if ds_candidate.exists():
                ds_path = ds_candidate
                break
        else:
            ds_path = None
        if ds_path:
            print(f"  Loading datastore from {ds_path} ...")
            t0 = time.time()
            ds_data = torch.load(ds_path, weights_only=False)
            print(f"  Datastore loaded  N={len(ds_data['y']):,}  "
                  f"D={ds_data['h'].shape[1]}  elapsed={time.time()-t0:.0f}s")
        else:
            print(f"  [WARN] datastore.pt not found in {src/'states'} or {data_src/'states'}")
            print(f"         global_unigram / random_partition will be skipped.")

    # Pre-flight: check required state files
    if not states_dir.exists():
        print(f"  [ERROR] states_dir not found: {states_dir}")
        sys.exit(1)

    missing = []
    for method in args.methods:
        for budget in args.budgets:
            fp = states_dir / f"{method}_B{budget}.pt"
            if not fp.exists():
                missing.append(str(fp))
    if missing and not args.allow_missing:
        for p in missing:
            print(f"  ERROR: missing: {p}")
        sys.exit("Missing state files. Use --allow_missing to skip.")

    all_rows = []

    for method in args.methods:
        for budget in args.budgets:
            fp = states_dir / f"{method}_B{budget}.pt"
            if not fp.exists():
                print(f"  [WARN] missing {fp.name}, skipping")
                continue

            print(f"\n{'='*60}")
            print(f"  {method}  B={budget}")
            print(f"{'='*60}")

            sf_orig = torch.load(fp, weights_only=False)

            for variant in args.variants:
                print(f"\n  -- variant: {variant}")
                t0 = time.time()

                # ── original: run Q-read on the unmodified state file ────────
                # run_method_budget has its own caching; if output exists it
                # returns the cached metrics without recomputing.
                if variant == "original":
                    m = run_method_budget(
                        method=method, budget=budget,
                        src=data_src, states_dir=states_dir,
                        out=q_out / "original",
                        device=device,
                        max_q_samples=args.max_q_samples,
                        n_epochs=args.n_epochs,
                        fast_grid=args.fast_grid,
                        force_neighbors=False, force_rewards=False,
                        force_train=False, force_eval=False,
                    )
                    if m:
                        all_rows.append({
                            "dataset": args.dataset, "model": args.model,
                            "method": method, "budget": budget,
                            "variant": "original",
                            "fixed_nll":  m.get("best_fixed_nll"),
                            "q_nll":      m.get("q_state_nll"),
                            "oracle_nll": m.get("oracle_nll"),
                            "delta_vs_gpt":      (m["q_state_nll"] - gpt_nll)  if math.isfinite(gpt_nll) and m.get("q_state_nll") else None,
                            "delta_vs_full_raw": (m["q_state_nll"] - fr_q_nll) if math.isfinite(fr_q_nll) and m.get("q_state_nll") else None,
                        })
                    continue

                # ── build ablated state dict ──────────────────────────────────
                sf_var = None
                if variant == "majority_token":
                    sf_var = ablate_majority_token(sf_orig)
                elif variant in TOP_M_MAP:
                    sf_var = ablate_top_m(sf_orig, TOP_M_MAP[variant])
                elif variant == "shuffled_distribution":
                    sf_var = ablate_shuffled_distribution(sf_orig, seed=args.seed)
                elif variant == "shuffled_prototype":
                    sf_var = ablate_shuffled_prototype(sf_orig, seed=args.seed)
                elif variant == "global_unigram":
                    if ds_data is None:
                        print(f"  [SKIP] global_unigram requires datastore.pt")
                        continue
                    sf_var = ablate_global_unigram(sf_orig, ds_data)
                elif variant == "random_partition":
                    if ds_data is None:
                        print(f"  [SKIP] random_partition requires datastore.pt")
                        continue
                    sf_var = ablate_random_partition(sf_orig, ds_data, seed=args.seed)
                else:
                    print(f"  [WARN] unknown variant: {variant}")
                    continue

                # Tag for file naming: {base}_abl_{variant}_B{budget}
                runner_method = f"{method}_abl_{variant}"
                sfp = abl_states / f"{runner_method}_B{budget}.pt"
                if not sfp.exists() or args.force:
                    torch.save(sf_var, sfp)

                # Check sentinel (q_metrics.json exists)
                qm_path = q_out / f"{runner_method}_B{budget}" / "q_metrics.json"
                if qm_path.exists() and not args.force:
                    print(f"  [cached] {runner_method} B={budget}")
                    with open(qm_path) as f:
                        m = json.load(f)
                else:
                    m = run_method_budget(
                        method=runner_method, budget=budget,
                        src=data_src, states_dir=abl_states, out=q_out,
                        device=device,
                        max_q_samples=args.max_q_samples,
                        n_epochs=args.n_epochs,
                        fast_grid=args.fast_grid,
                        force_neighbors=args.force, force_rewards=args.force,
                        force_train=args.force, force_eval=args.force,
                    )

                elapsed = time.time() - t0
                if m:
                    q_nll = m.get("q_state_nll")
                    all_rows.append({
                        "dataset": args.dataset, "model": args.model,
                        "method": method, "budget": budget,
                        "variant": variant,
                        "fixed_nll":  m.get("best_fixed_nll"),
                        "q_nll":      q_nll,
                        "oracle_nll": m.get("oracle_nll"),
                        "delta_vs_gpt":      (q_nll - gpt_nll)  if q_nll and math.isfinite(gpt_nll) else None,
                        "delta_vs_full_raw": (q_nll - fr_q_nll) if q_nll and math.isfinite(fr_q_nll) else None,
                    })
                    print(f"  [OK] {variant}  Q={q_nll:.4f}  elapsed={elapsed:.0f}s")
                else:
                    print(f"  [FAILED] {variant}  elapsed={elapsed:.0f}s")

    if not all_rows:
        print("\n  No results — exiting.")
        return

    df = pd.DataFrame(all_rows)

    # Save per-seed CSV
    csv_path = abl_dir / "state_object_ablation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}  ({len(df)} rows)")

    # Save per-seed summary markdown
    summary_md = _build_summary(df, gpt_nll, fr_q_nll, args.dataset, args.model)
    md_path = abl_dir / "state_object_ablation_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
    print(f"  Saved: {md_path}")

    # Save per-seed verdict JSON
    verdict = _compute_verdict(df)

    def _mean_q(v: str) -> float | None:
        rows = df[df["variant"] == v]["q_nll"]
        return float(rows.mean()) if not rows.empty and not rows.isna().all() else None

    verdict_json = {
        "dataset": args.dataset, "model": args.model, "seed": args.seed,
        "gpt_nll": gpt_nll if math.isfinite(gpt_nll) else None,
        "full_raw_q_nll": fr_q_nll if math.isfinite(fr_q_nll) else None,
        "original_q": _mean_q("original"),
        "majority_token_q": _mean_q("majority_token"),
        "top32_q": _mean_q("top32"),
        "shuffled_distribution_q": _mean_q("shuffled_distribution"),
        "shuffled_prototype_q": _mean_q("shuffled_prototype"),
        "random_partition_q": _mean_q("random_partition"),
        "verdict": verdict,
    }
    with open(abl_dir / "state_object_ablation_verdict.json", "w") as f:
        json.dump(verdict_json, f, indent=2)

    # Also copy to reports/
    df.to_csv(rep_dir / "state_object_ablation_results.csv", index=False)
    with open(rep_dir / "state_object_ablation_summary.md", "w", encoding="utf-8") as f:
        f.write(summary_md)

    _print_final_summary(df, args.dataset, args.model, [args.seed], gpt_nll, fr_q_nll)
    print("\nDone.")


if __name__ == "__main__":
    main()

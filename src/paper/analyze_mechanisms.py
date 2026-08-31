"""
analyze_mechanisms.py  --  Track A Paper  (Experiment 6)

For each state file, compute diagnostic metrics that explain WHY predictive
state codebooks work (or fail).

Metrics computed:
  - State count distribution (median, mean, frac ≤1, frac ≤5)
  - State entropy / purity (mean, median)
  - True-token hit@1, hit@4, hit@8
  - Mean/median p_state(true_token) at nearest state
  - Active state fraction (count > 0)
  - Oracle gap
  - Retrieval usage and avg_k from Q-model
  - Top helpful / harmful states

Outputs:
  diagnostics/<method>_B<budget>_diagnostics.json
  reports/mechanism_diagnostics.csv
  reports/mechanism_diagnostics_summary.md
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse, json, time
import numpy as np
import torch
import pandas as pd

from utils import get_device, set_seed
from evaluate_predictive_states import compute_top_k_state_sims, precompute_lookup, build_global_freq

VOCAB = 50257
TOP_K_EVAL = 8    # maximum k for hit@k computation


# ── diagnostics ───────────────────────────────────────────────────────────────

def compute_hit_at_k(
    val_data: dict,
    top_ids: torch.Tensor,    # [N, K]
    states: dict,
    k_vals: list = [1, 4, 8],
) -> dict:
    """Fraction of val queries where true token appears in top-k states' distributions."""
    y    = val_data["y"]          # [N]
    tk_ids  = states["top_k_token_ids"]   # [B, TOP_K]
    tk_cnts = states["top_k_token_counts"] # [B, TOP_K]
    N    = len(y)

    results = {}
    for k in k_vals:
        k_use = min(k, top_ids.shape[1])
        hits  = 0
        for q in range(N):
            yt = int(y[q])
            for s_rank in range(k_use):
                s_id = int(top_ids[q, s_rank])
                # Check if y_true appears in this state's token list
                token_list = tk_ids[s_id, :].tolist()
                cnt_list   = tk_cnts[s_id, :].tolist()
                found = any(tid == yt and c > 0
                            for tid, c in zip(token_list, cnt_list))
                if found:
                    hits += 1
                    break
        results[f"hit_at_{k}"] = hits / N if N > 0 else 0.0
    return results


def compute_hit_at_k_fast(
    val_data: dict,
    top_ids: torch.Tensor,    # [N, K]
    states: dict,
    k_vals: list = [1, 4, 8],
) -> dict:
    """Vectorised hit@k using precomputed raw_count from precompute_lookup."""
    y        = val_data["y"]                        # [N]
    tk_ids   = states["top_k_token_ids"]            # [B, TOP_K]
    tk_cnts  = states["top_k_token_counts"].float() # [B, TOP_K]
    N        = len(y)
    results  = {}

    # For each query and each top-k state, compute raw_count of y_true
    K    = top_ids.shape[1]
    # Build raw_count: [N, K]
    raw_count = torch.zeros(N, K, dtype=torch.float32)
    for s_rank in range(K):
        s_ids = top_ids[:, s_rank]         # [N]
        # For each query, find token_count[y_true] in state s_id
        cnts_s = tk_cnts[s_ids]            # [N, TOP_K]
        tids_s = tk_ids[s_ids]             # [N, TOP_K]
        y_exp  = y.unsqueeze(1).expand_as(tids_s)  # [N, TOP_K]
        match  = (tids_s == y_exp.int()).float() * cnts_s
        raw_count[:, s_rank] = match.sum(-1)

    for k in k_vals:
        k_use   = min(k, K)
        has_hit = (raw_count[:, :k_use] > 0).any(-1)  # [N]
        results[f"hit_at_{k}"] = float(has_hit.float().mean())
    return results


def p_state_true_stats(precomp: dict) -> dict:
    """Distribution of p_state(y_true) at the nearest state."""
    rc  = precomp["raw_count"][:, 0]  # [N] — count of y_true in nearest state
    tot = precomp["sel_total"][:, 0]  # [N] — total count of nearest state
    p   = rc / (tot + 1e-8)           # [N]
    return {
        "mean_p_state_true":   float(p.mean()),
        "median_p_state_true": float(p.median()),
        "frac_p_state_true_gt0":   float((p > 0).float().mean()),
        "frac_p_state_true_gt01":  float((p > 0.1).float().mean()),
    }


def state_count_stats(states: dict) -> dict:
    """Count distribution over states."""
    cnt = states["total_counts"].float()  # [B]
    return {
        "n_states":              int(len(cnt)),
        "mean_state_count":      float(cnt.mean()),
        "median_state_count":    float(cnt.median()),
        "frac_count_le_1":       float((cnt <= 1).float().mean()),
        "frac_count_le_5":       float((cnt <= 5).float().mean()),
        "frac_count_le_10":      float((cnt <= 10).float().mean()),
        "active_state_fraction": float((cnt > 0).float().mean()),
        "max_state_count":       float(cnt.max()),
    }


def state_quality_stats(states: dict) -> dict:
    """Entropy and purity distribution."""
    ent = states["state_entropy"].float()
    pur = states["state_purity"].float()
    return {
        "mean_state_entropy":    float(ent.mean()),
        "median_state_entropy":  float(ent.median()),
        "mean_state_purity":     float(pur.mean()),
        "median_state_purity":   float(pur.median()),
        "frac_purity_gt05":      float((pur > 0.5).float().mean()),
        "frac_purity_gt09":      float((pur > 0.9).float().mean()),
    }


def load_q_metrics(q_dir: Path, tag: str) -> dict:
    """Load Q-MLP result metrics if available."""
    candidates = [
        q_dir / tag / "q_metrics.json",
        q_dir / "states" / tag / "q_metrics.json",
        q_dir.parent / "q_read" / "states" / tag / "q_metrics.json",
    ]
    for p in candidates:
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return {}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",     required=True)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--states_dir", default=None)
    parser.add_argument("--q_dir",      default=None,
                        help="directory containing Q-MLP per-method subdirs")
    parser.add_argument("--methods",    nargs="+", default=None)
    parser.add_argument("--budgets",    nargs="+", type=int, default=None)
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--force",      action="store_true")
    args = parser.parse_args()

    set_seed(42)
    device     = get_device({"device": args.device})
    src        = Path(args.source)
    out        = Path(args.output)
    states_dir = Path(args.states_dir) if args.states_dir else out / "states"
    q_dir      = Path(args.q_dir) if args.q_dir else out / "q_read"
    diag_dir   = out / "diagnostics"
    rep_dir    = out / "reports"
    diag_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    csv_path = rep_dir / "mechanism_diagnostics.csv"
    if csv_path.exists() and not args.force:
        print(f"[cached] {csv_path}")
        return

    print(f"\nanalyze_mechanisms")
    print(f"  States: {states_dir}")
    print(f"  Device: {device}")

    # Load val data
    val_data = torch.load(src / "states" / "val.pt", weights_only=False)
    ds_data  = torch.load(src / "states" / "datastore.pt", weights_only=False)
    P_global = build_global_freq(ds_data["y"], VOCAB)
    val_h    = val_data["h"].float()
    val_h    = val_h / (val_h.norm(dim=-1, keepdim=True) + 1e-8)
    print(f"  Val: {len(val_data['y']):,} queries")

    # Discover state files
    if not states_dir.exists():
        print(f"  [WARN] states_dir not found: {states_dir}")
        return

    state_files = sorted(states_dir.glob("*.pt"))
    if args.methods or args.budgets:
        filtered = []
        for sfp in state_files:
            parts = sfp.stem.split("_B")
            if len(parts) != 2:
                continue
            meth, bud_s = parts[0], parts[1]
            try:
                bud = int(bud_s)
            except ValueError:
                continue
            if args.methods and meth not in args.methods:
                continue
            if args.budgets and bud not in args.budgets:
                continue
            filtered.append(sfp)
        state_files = filtered

    all_rows = []

    for sfp in state_files:
        tag    = sfp.stem
        parts  = tag.split("_B")
        method = parts[0] if len(parts) == 2 else tag
        budget = int(parts[1]) if len(parts) == 2 else 0

        diag_path = diag_dir / f"{tag}_diagnostics.json"
        if diag_path.exists() and not args.force:
            print(f"  [cached] {tag}")
            with open(diag_path) as f:
                d = json.load(f)
            all_rows.append(d)
            continue

        print(f"\n  [{tag}]")
        t0 = time.time()

        try:
            states = torch.load(sfp, weights_only=False)
            B      = states["prototype_h"].shape[0]

            # Top-K state neighbors for val queries
            K    = min(TOP_K_EVAL, B)
            print(f"    Computing top-{K} state neighbors ...")
            top_ids, top_sims = compute_top_k_state_sims(
                val_h, states["prototype_h"], K, device
            )

            # Token lookup
            precomp = precompute_lookup(val_data, top_ids, top_sims, states, P_global)

            # Metrics
            d = {"method": method, "budget": budget, "tag": tag}

            d.update(state_count_stats(states))
            d.update(state_quality_stats(states))

            print(f"    Computing hit@k ...")
            d.update(compute_hit_at_k_fast(val_data, top_ids, states, k_vals=[1, 4, 8]))

            d.update(p_state_true_stats(precomp))

            # Sim stats
            d["mean_nearest_sim"]   = float(top_sims[:, 0].mean())
            d["median_nearest_sim"] = float(top_sims[:, 0].median())

            # Load Q metrics if available
            qm = load_q_metrics(q_dir, tag)
            if qm:
                d["q_nll"]           = qm.get("q_state_nll",  float("nan"))
                d["fixed_nll"]       = qm.get("best_fixed_nll", float("nan"))
                d["oracle_nll"]      = qm.get("oracle_nll",   float("nan"))
                d["oracle_gap_vs_q"] = qm.get("oracle_gap_vs_q", float("nan"))
                d["retrieval_usage"] = qm.get("retrieval_usage",  float("nan"))
                d["avg_k_states"]    = qm.get("avg_k_states",     float("nan"))
            else:
                d["q_nll"] = d["fixed_nll"] = d["oracle_nll"] = float("nan")
                d["oracle_gap_vs_q"] = d["retrieval_usage"] = d["avg_k_states"] = float("nan")

            d["elapsed_s"] = round(time.time() - t0, 1)

            with open(diag_path, "w") as f:
                json.dump(d, f, indent=2)
            print(f"    Saved: {diag_path}  elapsed={d['elapsed_s']:.0f}s")
            all_rows.append(d)

        except Exception as e:
            import traceback
            print(f"  [FAILED] {tag}: {e}")
            traceback.print_exc()

    if not all_rows:
        print("  No results.")
        return

    df = pd.DataFrame(all_rows)
    float_cols = df.select_dtypes("float64").columns
    df[float_cols] = df[float_cols].round(4)
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}  ({len(df)} rows)")

    # ── summary markdown ──────────────────────────────────────────────────────
    key_cols = ["method", "budget", "n_states", "median_state_count",
                "mean_state_purity", "hit_at_1", "hit_at_4", "hit_at_8",
                "mean_p_state_true", "q_nll", "oracle_nll"]
    present = [c for c in key_cols if c in df.columns]

    lines = ["# Experiment 6: Mechanism Diagnostics\n"]
    lines.append("| " + " | ".join(f"{c}" for c in present) + " |")
    lines.append("|" + "|".join("-" * (len(c) + 2) for c in present) + "|")
    for _, r in df.sort_values(["method", "budget"]).iterrows():
        def fmt(v):
            if isinstance(v, float):
                return f"{v:.4f}" if not np.isnan(v) else "nan"
            return str(v)
        lines.append("| " + " | ".join(fmt(r[c]) for c in present) + " |")

    summary_path = rep_dir / "mechanism_diagnostics_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {summary_path}")


if __name__ == "__main__":
    main()

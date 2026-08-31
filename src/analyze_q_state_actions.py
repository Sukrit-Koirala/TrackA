"""
analyze_q_state_actions.py  —  MVP 2c action analysis

What does Q-state-read actually choose?

Loads val predictions (chosen action per query), val data, state neighbors.
Breaks down Q behavior by GPT confidence bins and state quality bins.

Outputs per (method, budget):
  <out>/action_analysis/<method>_B<budget>_action_distribution.csv
  <out>/action_analysis/<method>_B<budget>_behavior_by_bin.csv
  <out>/action_analysis/<method>_B<budget>_ACTION_SUMMARY.md
"""

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import numpy as np
import torch
import pandas as pd

EPS = 1e-10


def bin_stats(mask: np.ndarray, nll_gpt: np.ndarray, q_nlls: np.ndarray,
              fixed_nlls: np.ndarray, chosen: np.ndarray,
              actions: list) -> dict:
    n = int(mask.sum())
    if n == 0:
        return {"n": 0}
    k_arr  = np.array([actions[int(a)]["k_states"] for a in chosen[mask]])
    tau_arr = np.array([actions[int(a)]["tau"]   for a in chosen[mask]])
    alp_arr = np.array([actions[int(a)]["alpha"] for a in chosen[mask]])
    bet_arr = np.array([actions[int(a)]["beta"]  for a in chosen[mask]])
    ret_mask = k_arr > 0
    return {
        "n":              n,
        "q_nll_mean":     float(q_nlls[mask].mean()),
        "gpt_nll_mean":   float(nll_gpt[mask].mean()),
        "fixed_nll_mean": float(fixed_nlls[mask].mean()),
        "retrieval_usage":float(ret_mask.mean()),
        "avg_k_states":   float(k_arr.mean()),
        "mean_tau":       float(tau_arr.mean()),
        "mean_alpha":     float(alp_arr.mean()),
        "mean_beta":      float(bet_arr.mean()),
        "frac_gpt_only":  float((k_arr == 0).mean()),
        "frac_beta_0":    float((bet_arr == 0.0).mean()),
        "frac_beta_1":    float((bet_arr == 1.0).mean()),
        "frac_beta_5":    float((bet_arr == 5.0).mean()),
        "frac_beta_10":   float((bet_arr == 10.0).mean()),
    }


def analyze_method_budget(
    method: str, budget: int,
    src: Path, states_dir: Path, q_read_dir: Path, out_dir: Path,
) -> dict | None:
    tag        = f"{method}_B{budget}"
    mb_dir     = q_read_dir / tag
    nbr_path   = q_read_dir / "state_neighbors" / f"{tag}_val_top32.pt"
    pred_path  = mb_dir / "val_predictions.pt"
    model_path = mb_dir / "q_model.pt"

    missing = [p for p in [pred_path, model_path] if not p.exists()]
    if missing:
        print(f"  [skip] missing: {missing}")
        return None

    preds  = torch.load(pred_path,  weights_only=False)
    ckpt   = torch.load(model_path, weights_only=False)
    chosen = preds["chosen"].numpy()      # [N_val]
    q_nlls = preds["q_nlls"].numpy()      # [N_val]
    fixed_nlls = preds["fixed_nlls"].numpy()
    oracle_nlls = preds["oracle_nlls"].numpy() if "oracle_nlls" in preds else q_nlls.copy()
    actions = ckpt["actions"]
    A = len(actions)

    val_data = torch.load(src / "states" / "val.pt", weights_only=False)
    nll_gpt  = val_data["nll_gpt"].numpy()
    p_gpt_y  = val_data["p_gpt_true"].numpy()
    N        = len(chosen)

    # State similarity at nearest state
    nearest_sim = np.zeros(N)
    nearest_ent = np.zeros(N)
    nearest_pur = np.zeros(N)
    if nbr_path.exists() and (states_dir / f"{tag}.pt").exists():
        nbrs   = torch.load(nbr_path, weights_only=False)
        states = torch.load(states_dir / f"{tag}.pt", weights_only=False)
        nearest_sim = nbrs["sims"][:, 0].numpy()
        ids0        = nbrs["ids"][:, 0].numpy()
        nearest_ent = states["state_entropy"][ids0].numpy()
        nearest_pur = states["state_purity"][ids0].numpy()

    # ── Action distribution ───────────────────────────────────────────────────
    counts = np.bincount(chosen, minlength=A)
    act_rows = []
    for ai, act in enumerate(actions):
        if counts[ai] == 0:
            continue
        act_rows.append({
            **act,
            "count": int(counts[ai]),
            "frac":  float(counts[ai] / N),
        })
    act_df = pd.DataFrame(act_rows).sort_values("count", ascending=False)

    k_arr   = np.array([actions[int(a)]["k_states"] for a in chosen])
    tau_arr = np.array([actions[int(a)]["tau"]       for a in chosen])
    alp_arr = np.array([actions[int(a)]["alpha"]     for a in chosen])
    bet_arr = np.array([actions[int(a)]["beta"]      for a in chosen])
    ret_mask = k_arr > 0

    agg = {
        "method": method, "budget": budget,
        "n_val": N,
        "retrieval_usage":  float(ret_mask.mean()),
        "avg_k_states":     float(k_arr.mean()),
        "median_k_states":  float(np.median(k_arr)),
        "mean_tau":         float(tau_arr.mean()),
        "mean_alpha":       float(alp_arr.mean()),
        "mean_beta":        float(bet_arr.mean()),
        "frac_gpt_only":    float((k_arr == 0).mean()),
        "frac_beta_0":      float((bet_arr == 0.0).mean()),
        "frac_beta_1":      float((bet_arr == 1.0).mean()),
        "frac_beta_5":      float((bet_arr == 5.0).mean()),
        "frac_beta_10":     float((bet_arr == 10.0).mean()),
        "q_nll_mean":       float(q_nlls.mean()),
        "fixed_nll_mean":   float(fixed_nlls.mean()),
        "gpt_nll_mean":     float(nll_gpt.mean()),
        "oracle_nll_mean":  float(oracle_nlls.mean()),
    }

    # ── Behavior by bins ──────────────────────────────────────────────────────
    bin_rows = []
    for feat_name, feat_arr in [
        ("gpt_nll",      nll_gpt),
        ("nearest_sim",  nearest_sim),
        ("nearest_ent",  nearest_ent),
        ("nearest_pur",  nearest_pur),
    ]:
        qs = np.quantile(feat_arr, [0.25, 0.5, 0.75])
        bins = [
            (f"Q1_lo_{feat_name}", feat_arr <= qs[0]),
            (f"Q2_{feat_name}",    (feat_arr > qs[0]) & (feat_arr <= qs[1])),
            (f"Q3_{feat_name}",    (feat_arr > qs[1]) & (feat_arr <= qs[2])),
            (f"Q4_hi_{feat_name}", feat_arr > qs[2]),
        ]
        for bin_name, mask in bins:
            s = bin_stats(mask, nll_gpt, q_nlls, fixed_nlls, chosen, actions)
            s["feature"] = feat_name
            s["bin"]     = bin_name
            s["q25"]     = float(qs[0]); s["q50"] = float(qs[1]); s["q75"] = float(qs[2])
            bin_rows.append(s)

    bin_df = pd.DataFrame(bin_rows)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    act_df.to_csv(out_dir / f"{tag}_action_distribution.csv", index=False)
    bin_df.to_csv(out_dir / f"{tag}_behavior_by_bin.csv",      index=False)

    # ── Markdown summary ──────────────────────────────────────────────────────
    top3 = act_df.head(3)
    md_lines = [
        f"# Q-State-Read Action Analysis: {tag}",
        "",
        f"## Aggregate Behavior",
        f"N_val={N}  retrieval_usage={agg['retrieval_usage']:.2%}  "
        f"avg_k={agg['avg_k_states']:.1f}  frac_gpt_only={agg['frac_gpt_only']:.2%}",
        "",
        f"mean alpha={agg['mean_alpha']:.3f}  mean beta={agg['mean_beta']:.3f}",
        f"beta=0: {agg['frac_beta_0']:.2%}   beta=1: {agg['frac_beta_1']:.2%}   "
        f"beta=5: {agg['frac_beta_5']:.2%}   beta=10: {agg['frac_beta_10']:.2%}",
        "",
        f"q_nll={agg['q_nll_mean']:.4f}  fixed={agg['fixed_nll_mean']:.4f}  "
        f"gpt={agg['gpt_nll_mean']:.4f}  oracle={agg['oracle_nll_mean']:.4f}",
        "",
        f"## Top Actions",
        "",
        "| action | count | frac | k | tau | alpha | beta |",
        "|--------|-------|------|---|-----|-------|------|",
    ]
    for _, r in act_df.head(10).iterrows():
        md_lines.append(
            f"| {r['name']:<35} | {int(r['count']):>5} | {r['frac']:.3f} | "
            f"{int(r['k_states'])} | {r['tau']} | {r['alpha']} | {r['beta']} |"
        )
    md_lines += [
        "",
        f"## Behavior by GPT NLL Quartile (low=confident, high=uncertain)",
        "",
        "| bin | n | ret_usage | avg_k | mean_alpha | mean_beta | q_nll | gpt_nll |",
        "|-----|---|-----------|-------|------------|-----------|-------|---------|",
    ]
    for r in bin_rows:
        if r["feature"] == "gpt_nll":
            md_lines.append(
                f"| {r['bin']:<30} | {r['n']:>4} | "
                f"{r.get('retrieval_usage', 0):.2f} | {r.get('avg_k_states', 0):.1f} | "
                f"{r.get('mean_alpha', 0):.2f} | {r.get('mean_beta', 0):.2f} | "
                f"{r.get('q_nll_mean', 0):.4f} | {r.get('gpt_nll_mean', 0):.4f} |"
            )

    md_lines += [
        "",
        f"## Behavior by Nearest State Similarity Quartile",
        "",
        "| bin | n | ret_usage | avg_k | mean_alpha | q_nll | gpt_nll |",
        "|-----|---|-----------|-------|------------|-------|---------|",
    ]
    for r in bin_rows:
        if r["feature"] == "nearest_sim":
            md_lines.append(
                f"| {r['bin']:<35} | {r['n']:>4} | "
                f"{r.get('retrieval_usage', 0):.2f} | {r.get('avg_k_states', 0):.1f} | "
                f"{r.get('mean_alpha', 0):.2f} | "
                f"{r.get('q_nll_mean', 0):.4f} | {r.get('gpt_nll_mean', 0):.4f} |"
            )

    md_path = out_dir / f"{tag}_ACTION_SUMMARY.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"  [{tag}]  ret={agg['retrieval_usage']:.1%}  "
          f"avg_k={agg['avg_k_states']:.1f}  "
          f"gpt_only={agg['frac_gpt_only']:.1%}  "
          f"beta0={agg['frac_beta_0']:.1%}")
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",     required=True)
    parser.add_argument("--states_dir", required=True)
    parser.add_argument("--q_read_dir", required=True)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--methods",  nargs="+",
                        default=["minibatch_kmeans", "utility_weighted", "query_kmeans"])
    parser.add_argument("--budgets",  nargs="+", type=int,
                        default=[10000, 25000, 50000])
    parser.add_argument("--force",    action="store_true")
    args = parser.parse_args()

    src        = Path(args.source)
    states_dir = Path(args.states_dir)
    q_read_dir = Path(args.q_read_dir)
    out        = Path(args.output) / "action_analysis"

    print(f"\nanalyze_q_state_actions")
    all_agg = []
    for method in args.methods:
        for budget in args.budgets:
            tag = f"{method}_B{budget}"
            if (out / f"{tag}_action_distribution.csv").exists() and not args.force:
                print(f"  [cached] {tag}")
                continue
            agg = analyze_method_budget(method, budget, src, states_dir, q_read_dir, out)
            if agg:
                all_agg.append(agg)

    if all_agg:
        pd.DataFrame(all_agg).to_csv(out / "aggregate_action_stats.csv", index=False)
        print(f"  Saved aggregate_action_stats.csv")


if __name__ == "__main__":
    main()

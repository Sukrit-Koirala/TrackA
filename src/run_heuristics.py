"""
run_heuristics.py

Extended threshold heuristic search and cost-lambda sweep.

Heuristics:
  For each observation feature and each threshold direction (gt / lt),
  search the best threshold on controller_train.  Then evaluate the best
  config per (feature, direction) on val.  Report top-10 overall.

  Rule: if feature [op] threshold → use retrieval_action; else GPT-only.
  retrieval_action is also searched over all 45 non-gpt_only actions.

Cost-lambda sweep:
  Load pre-computed fixed-action CT/val metrics from fixed_baselines.json.
  For each lambda in cost_lambda_values, re-select best action on CT using
  cost-penalized criterion (ct_nll + lambda * k/max_k), then report val NLL
  and avg_k.  No retraining needed.

No regions, modules, hierarchy, WRITE/PRUNE gates.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json
import numpy as np
import pandas as pd
import torch

from utils import load_config, ensure_dirs, set_seed, build_action_grid, ppl
from train_controller import build_features
from train_q_controller import compute_q_base_data, build_all_action_features


# Feature index → name mapping (must match build_features in train_controller.py)
FEATURE_NAMES = [
    "gpt_entropy",           # f0  high = GPT uncertain → retrieve
    "gpt_top1_prob",         # f1  low  = GPT uncertain → retrieve
    "gpt_margin",            # f2  low  = GPT uncertain → retrieve
    "nearest_sim",           # f3  high = good match found → retrieve
    "mean_top4_sim",         # f4  high = good matches  → retrieve
    "mean_top8_sim",         # f5
    "mean_top16_sim",        # f6
    "mean_top32_sim",        # f7
    "std_top32_sim",         # f8  uncertain direction; try both
    "label_entropy_top8",    # f9  low  = neighbors agree → retrieve
    "label_entropy_top32",   # f10 low  = neighbors agree → retrieve
    "mode_match_top8",       # f11 uncertain direction; try both
    "mode_match_top32",      # f12
]


def heuristic_threshold_search(
    feat_vals: np.ndarray,    # [N] feature values on CT
    nll_gpt: np.ndarray,      # [N] GPT NLL on CT
    nll_ret: np.ndarray,      # [N] retrieval action NLL on CT
    n_thresholds: int = 50,
    direction: str = "gt",    # "gt" = retrieve if feat > thresh; "lt" = if feat < thresh
) -> tuple[float, float]:
    """Return (best_threshold, best_mean_nll) by searching percentile-based thresholds."""
    percentiles = np.linspace(5, 95, n_thresholds)
    thresholds  = np.percentile(feat_vals, percentiles)

    best_nll   = float("inf")
    best_thresh = thresholds[0]

    for t in thresholds:
        mask      = (feat_vals > t) if direction == "gt" else (feat_vals < t)
        nll_hyb   = np.where(mask, nll_ret, nll_gpt)
        mean_nll  = float(nll_hyb.mean())
        if mean_nll < best_nll:
            best_nll    = mean_nll
            best_thresh = float(t)

    return best_thresh, best_nll


def apply_heuristic(
    feat_vals: np.ndarray,    # [N]
    nll_gpt: np.ndarray,      # [N]
    nll_ret: np.ndarray,      # [N]
    threshold: float,
    direction: str,
    k_val: float,
) -> dict:
    """Evaluate a heuristic on a dataset and return metrics."""
    mask    = (feat_vals > threshold) if direction == "gt" else (feat_vals < threshold)
    nll_hyb = np.where(mask, nll_ret, nll_gpt)
    k_hyb   = np.where(mask, k_val, 0.0)
    return {
        "mean_nll":        float(nll_hyb.mean()),
        "mean_ppl":        ppl(float(nll_hyb.mean())),
        "mean_k":          float(k_hyb.mean()),
        "retrieval_usage": float(mask.mean()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extended heuristic search + cost-lambda sweep")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    set_seed(cfg.get("seed", 42))

    states_dir  = Path(cfg["states_dir"])
    nbrs_dir    = Path(cfg["neighbors_dir"])
    reports_dir = Path(cfg["reports_dir"])
    models_dir  = Path(cfg["models_dir"])
    k_max       = cfg["max_k"]
    n_thresh    = cfg.get("heuristic_threshold_steps", 50)

    print("Loading data …")
    ct_data  = torch.load(states_dir / "controller_train.pt", weights_only=False)
    val_data = torch.load(states_dir / "val.pt",             weights_only=False)
    ct_nbrs  = torch.load(nbrs_dir   / f"controller_train_top{k_max}.pt", weights_only=False)
    val_nbrs = torch.load(nbrs_dir   / f"val_top{k_max}.pt",              weights_only=False)

    actions         = build_action_grid(cfg)
    retrieval_acts  = [a for a in actions if a["k"] > 0]   # 45 actions
    print(f"  {len(retrieval_acts)} retrieval actions to sweep")

    # ── load or compute per-action NLLs on CT ─────────────────────────────────
    cache_path = models_dir / "q_base_data.pt"
    if cache_path.exists() and not args.force_recompute:
        print("Loading cached Q base data for per-action CT NLLs …")
        cache              = torch.load(cache_path, weights_only=False)
        X_obs_ct           = cache["X_obs"]
        per_action_rewards = cache["per_action_rewards"]   # [N, A]  reward = nll_gpt - nll_act
    else:
        print("Computing Q base data …")
        X_obs_ct, per_action_rewards = compute_q_base_data(ct_data, ct_nbrs, actions, cfg)
        torch.save({"X_obs": X_obs_ct, "per_action_rewards": per_action_rewards}, cache_path)

    nll_gpt_ct   = ct_data["nll_gpt"].float().numpy()          # [N]
    # per_action_rewards[:,ai] = nll_gpt - nll_action  (lambda=0)
    per_action_nlls_ct = (
        torch.from_numpy(nll_gpt_ct).unsqueeze(1) - per_action_rewards
    ).numpy()   # [N, A]

    # Val: need per-action NLLs too
    from gate_chain import evaluate_action_for_queries
    X_obs_val = build_features(val_data, val_nbrs)
    nll_gpt_val = val_data["nll_gpt"].float().numpy()

    # Pre-compute per-retrieval-action val NLLs (per-example, for applying heuristics on val)
    print("Pre-computing per-action val NLLs …")
    name_to_idx     = {a["name"]: i for i, a in enumerate(actions)}
    nll_val_per_act = {}   # action_name → np.ndarray [N_val]

    for action in retrieval_acts:
        ai = name_to_idx[action["name"]]
        m  = evaluate_action_for_queries(
            val_data, val_nbrs, action,
            max_k=k_max, lambda_cost=0.0, eps=cfg.get("eps", 1e-12),
        )
        nll_val_per_act[action["name"]] = m["per_example_nll"].numpy()

    # ── extended heuristic search ──────────────────────────────────────────────
    print("\n[1] Extended heuristic search …")
    feat_ct  = X_obs_ct.numpy()    # [N, 13]
    feat_val = X_obs_val.numpy()   # [N_val, 13]

    rows: list[dict] = []

    for fi, feat_name in enumerate(FEATURE_NAMES):
        f_ct  = feat_ct[:, fi]
        f_val = feat_val[:, fi]

        for direction in ("gt", "lt"):
            for ret_action in retrieval_acts:
                ai       = name_to_idx[ret_action["name"]]
                nll_r_ct = per_action_nlls_ct[:, ai]

                thresh, ct_nll = heuristic_threshold_search(
                    f_ct, nll_gpt_ct, nll_r_ct, n_thresh, direction
                )
                # Evaluate this threshold on val (no leakage: threshold chosen on CT)
                nll_r_val = nll_val_per_act[ret_action["name"]]
                val_m     = apply_heuristic(
                    f_val, nll_gpt_val, nll_r_val,
                    thresh, direction, float(ret_action["k"]),
                )
                rows.append({
                    "feature":          feat_name,
                    "feature_idx":      fi,
                    "direction":        direction,
                    "retrieval_action": ret_action["name"],
                    "k":                ret_action["k"],
                    "tau":              ret_action["tau"],
                    "alpha":            ret_action["alpha"],
                    "threshold":        thresh,
                    "ct_nll":           ct_nll,
                    "val_nll":          val_m["mean_nll"],
                    "val_ppl":          val_m["mean_ppl"],
                    "val_k":            val_m["mean_k"],
                    "val_ret_usage":    val_m["retrieval_usage"],
                })

    df = pd.DataFrame(rows).sort_values("val_nll")
    df.to_csv(reports_dir / "heuristic_baselines.csv", index=False)
    print(f"  Searched {len(rows)} (feature, direction, action) combos.")

    # ── report top-10 ─────────────────────────────────────────────────────────
    gpt_val_nll  = float(val_data["nll_gpt"].mean())
    best_val_nll_fixed = None
    try:
        with open(reports_dir / "fixed_baselines.json") as f:
            baselines = json.load(f)
        best_val_nll_fixed = baselines["best_train_selected_val_nll"]
    except FileNotFoundError:
        pass

    print("\nTop-10 heuristics by val NLL:")
    print(f"  (GPT-only val NLL baseline: {gpt_val_nll:.4f})")
    if best_val_nll_fixed:
        print(f"  (Best fixed action val NLL: {best_val_nll_fixed:.4f})")
    print(f"\n  {'Feature':<22} {'Dir':3} {'Action':<25} {'thresh':>7}  "
          f"{'ct_nll':>7}  {'val_nll':>7}  {'val_ppl':>8}  {'avg_k':>6}  {'ret%':>5}")
    print("  " + "-" * 110)
    for _, r in df.head(10).iterrows():
        print(f"  {r['feature']:<22} {r['direction']:3} {r['retrieval_action']:<25} "
              f"{r['threshold']:>7.3f}  {r['ct_nll']:>7.4f}  {r['val_nll']:>7.4f}  "
              f"{r['val_ppl']:>8.2f}  {r['val_k']:>6.1f}  {r['val_ret_usage']*100:>4.0f}%")

    best_row   = df.iloc[0]
    best_heur  = {
        "feature":          str(best_row["feature"]),
        "feature_idx":      int(best_row["feature_idx"]),
        "direction":        str(best_row["direction"]),
        "retrieval_action": str(best_row["retrieval_action"]),
        "threshold":        float(best_row["threshold"]),
        "ct_nll":           float(best_row["ct_nll"]),
        "val_nll":          float(best_row["val_nll"]),
        "val_ppl":          float(best_row["val_ppl"]),
        "val_k":            float(best_row["val_k"]),
        "val_ret_usage":    float(best_row["val_ret_usage"]),
        "top10":            df.head(10).to_dict(orient="records"),
    }
    with open(reports_dir / "best_heuristics.json", "w") as f:
        json.dump(best_heur, f, indent=2)
    print(f"\nSaved: {reports_dir / 'heuristic_baselines.csv'}")
    print(f"Saved: {reports_dir / 'best_heuristics.json'}")

    # ── cost-lambda sweep ─────────────────────────────────────────────────────
    print("\n[2] Cost-lambda sweep (fixed actions) …")
    cost_lambdas = cfg.get("cost_lambda_values", [0.0, 0.001, 0.005, 0.01, 0.05])

    try:
        with open(reports_dir / "fixed_baselines.json") as f:
            baselines = json.load(f)
        ct_results  = baselines["all_ct_results"]   # list of dicts
        val_results = {r["name"]: r for r in baselines["all_val_results"]}
    except (FileNotFoundError, KeyError):
        print("  fixed_baselines.json not found or missing all_ct_results. "
              "Run baselines.py first. Skipping cost sweep.")
        print("\nDone.")
        return

    sweep_rows: list[dict] = []
    for lam in cost_lambdas:
        # Re-select best action on CT using cost-penalized criterion.
        # all_ct_results keys: name, k, mean_nll, mean_ppl, mean_k, retrieval_usage
        best_r = min(ct_results,
                     key=lambda r: r["mean_nll"] + lam * r.get("k", 0) / k_max)
        bname = best_r["name"]
        bk    = best_r.get("k", 0)
        # all_val_results uses same key names: mean_nll, mean_ppl, retrieval_usage
        v     = val_results.get(bname, {})
        sweep_rows.append({
            "lambda_cost":           lam,
            "best_ct_action":        bname,
            "best_ct_k":             bk,
            "best_ct_nll":           best_r["mean_nll"],
            "best_ct_nll_penalized": best_r["mean_nll"] + lam * bk / k_max,
            "val_nll":               v.get("mean_nll",        float("nan")),
            "val_ppl":               v.get("mean_ppl",        float("nan")),
            "val_avg_k":             float(bk),
            "val_ret_usage":         v.get("retrieval_usage", float("nan")),
        })

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(reports_dir / "cost_sweep.csv", index=False)

    print(f"\n  {'lambda':>8}  {'action':<28}  {'val_nll':>7}  {'val_ppl':>9}  {'avg_k':>6}")
    print("  " + "-" * 70)
    for r in sweep_rows:
        k_str = f"{r['val_avg_k']:>6.1f}"
        n_str = f"{r['val_nll']:>7.4f}" if r["val_nll"] == r["val_nll"] else "    nan"
        p_str = f"{r['val_ppl']:>9.2f}" if r["val_ppl"] == r["val_ppl"] else "      nan"
        print(f"  {r['lambda_cost']:>8.4f}  {r['best_ct_action']:<28}  {n_str}  {p_str}  {k_str}")

    print(f"\nSaved: {reports_dir / 'cost_sweep.csv'}")
    print("\nDone.")


if __name__ == "__main__":
    main()

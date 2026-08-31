"""
similarity_diagnostics.py

Nearest-neighbor similarity distribution analysis.

The v1 heuristic found thresholds near nearest_sim > 0.999 which is suspicious.
This script reports the full quantile distribution of similarity features,
broken down by:
  - all examples
  - retrieval helps vs hurts (best fixed action as reference)
  - Q-MLP controller chose retrieval vs GPT-only  (if model available)

Outputs (in cfg["audit_dir"]):
  similarity_quantiles.csv
  similarity_diagnostics.json
  nearest_sim_hist_val.png   (if matplotlib available)
  nearest_sim_help_vs_hurt.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json
import numpy as np
import torch

from utils import load_config, ensure_dirs, set_seed, build_action_grid
from gate_chain import evaluate_action_for_queries
from train_controller import build_features

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

QUANTILES = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]

FEATURE_NAMES = [
    "gpt_entropy",
    "gpt_top1_prob",
    "gpt_margin",
    "nearest_sim",
    "mean_top4_sim",
    "mean_top8_sim",
    "mean_top16_sim",
    "mean_top32_sim",
    "std_top32_sim",
    "label_entropy_top8",
    "label_entropy_top32",
    "mode_match_top8",
    "mode_match_top32",
]


def quantile_row(name: str, vals: np.ndarray) -> dict:
    q = np.percentile(vals, QUANTILES)
    row = {"feature": name, "n": len(vals)}
    for pct, v in zip(QUANTILES, q):
        row[f"p{pct:03d}"] = float(v)
    row["mean"] = float(vals.mean())
    row["std"]  = float(vals.std())
    return row


def main():
    parser = argparse.ArgumentParser(description="Similarity distribution diagnostics")
    parser.add_argument("--config", default="configs/clean_storysplit.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    set_seed(cfg.get("seed", 42))

    states_dir = Path(cfg["states_dir"])
    nbrs_dir   = Path(cfg["neighbors_dir"])
    audit_dir  = Path(cfg.get("audit_dir", "outputs_clean_storysplit/audit"))
    audit_dir.mkdir(parents=True, exist_ok=True)
    k_max      = cfg["max_k"]

    print("Loading val data …")
    val_data = torch.load(states_dir / "val.pt",             weights_only=False)
    val_nbrs = torch.load(nbrs_dir   / f"val_top{k_max}.pt", weights_only=False)

    actions = build_action_grid(cfg)

    # Features [N, 13]
    print("Building observation features …")
    X_val = build_features(val_data, val_nbrs).numpy()
    N     = X_val.shape[0]

    # Best fixed action val NLLs (for help/hurt split)
    try:
        from pathlib import Path as _P
        import json as _json
        with open(Path(cfg["reports_dir"]) / "fixed_baselines.json") as f:
            baselines = _json.load(f)
        best_fixed_name = baselines["best_train_selected_name"]
        best_fixed_act  = next(a for a in actions if a["name"] == best_fixed_name)
        m_fixed = evaluate_action_for_queries(
            val_data, val_nbrs, best_fixed_act,
            max_k=k_max, lambda_cost=0.0, eps=cfg.get("eps", 1e-12),
        )
        nll_fixed = m_fixed["per_example_nll"].numpy()
        nll_gpt   = val_data["nll_gpt"].float().numpy()
        helps     = nll_gpt - nll_fixed   # positive = retrieval helped
        help_mask = helps > 0
        hurt_mask = helps < 0
        has_fixed = True
        print(f"  Best fixed action: {best_fixed_name}  "
              f"helps={help_mask.sum()}  hurts={hurt_mask.sum()}")
    except Exception as e:
        print(f"  Could not load fixed baselines: {e}")
        has_fixed = False
        help_mask = np.ones(N, dtype=bool)
        hurt_mask = np.zeros(N, dtype=bool)

    # Q-controller chosen actions (if available)
    q_mlp_path = Path(cfg["models_dir"]) / "q_mlp_full.pt"
    q_ret_mask = None
    if q_mlp_path.exists():
        try:
            from train_q_controller import load_q_model, apply_q_controller, build_all_action_features
            device  = torch.device("cpu")
            q_model, action_names = load_q_model(q_mlp_path, device)
            name_to_idx = {a["name"]: i for i, a in enumerate(actions)}
            actions_sub = [actions[name_to_idx[n]] for n in action_names if n in name_to_idx]
            act_feats   = build_all_action_features(actions_sub, k_max)
            X_t         = torch.from_numpy(X_val)
            chosen      = apply_q_controller(q_model, X_t, act_feats, device)
            q_ret_mask  = np.array([actions_sub[int(c)]["k"] > 0 for c in chosen])
            print(f"  Q-MLP-full: retrieval chosen for {q_ret_mask.sum()} / {N} examples")
        except Exception as e:
            print(f"  Could not load Q-MLP: {e}")

    # ── per-feature quantile tables ────────────────────────────────────────────
    print("\nComputing quantiles …")
    rows = []
    diag: dict = {"segments": {}}

    segments = {"all": np.ones(N, dtype=bool)}
    if has_fixed:
        segments["retrieval_helps"] = help_mask
        segments["retrieval_hurts"] = hurt_mask
    if q_ret_mask is not None:
        segments["q_chose_retrieval"] = q_ret_mask
        segments["q_chose_gpt_only"]  = ~q_ret_mask

    for seg_name, seg_mask in segments.items():
        seg_idx = np.where(seg_mask)[0]
        if len(seg_idx) == 0:
            continue
        X_seg = X_val[seg_idx]
        seg_rows = []
        for fi, feat_name in enumerate(FEATURE_NAMES):
            row = quantile_row(f"{feat_name} [{seg_name}]", X_seg[:, fi])
            row["segment"] = seg_name
            rows.append(row)
            seg_rows.append(row)
        diag["segments"][seg_name] = {
            "n": int(seg_mask.sum()),
            "features": {FEATURE_NAMES[fi]: {
                "mean":  float(X_seg[:, fi].mean()),
                "std":   float(X_seg[:, fi].std()),
                "p50":   float(np.median(X_seg[:, fi])),
                "p99":   float(np.percentile(X_seg[:, fi], 99)),
                "max":   float(X_seg[:, fi].max()),
            } for fi in range(len(FEATURE_NAMES))}
        }

    # Save CSV
    import pandas as pd
    df = pd.DataFrame(rows)
    out_csv = audit_dir / "similarity_quantiles.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    with open(audit_dir / "similarity_diagnostics.json", "w") as f:
        json.dump(diag, f, indent=2)
    print(f"Saved: {audit_dir / 'similarity_diagnostics.json'}")

    # ── print summary ──────────────────────────────────────────────────────────
    print("\nNearest-sim (f3) quantiles [all val]:")
    all_rows = [r for r in rows if "nearest_sim" in r["feature"] and "[all]" in r["feature"]]
    if all_rows:
        r = all_rows[0]
        for pct in [0, 1, 10, 25, 50, 75, 90, 95, 99, 100]:
            print(f"  p{pct:3d}: {r[f'p{pct:03d}']:.6f}")

    p99_all  = diag["segments"].get("all", {}).get("features", {}).get("nearest_sim", {}).get("p99", None)
    p99_help = diag["segments"].get("retrieval_helps", {}).get("features", {}).get("nearest_sim", {}).get("p99", None)
    p99_hurt = diag["segments"].get("retrieval_hurts", {}).get("features", {}).get("nearest_sim", {}).get("p99", None)

    print(f"\n  p99 nearest_sim — all:{p99_all}  helps:{p99_help}  hurts:{p99_hurt}")
    if p99_all is not None and p99_all > 0.99:
        print("  *** WARNING: p99 nearest_sim > 0.99 — investigate for near-duplicate leakage ***")
    elif p99_all is not None and p99_all > 0.95:
        print("  NOTE: p99 nearest_sim > 0.95 — some very close matches present")
    else:
        print("  OK: p99 nearest_sim ≤ 0.95 — no obvious near-duplicate issue")

    # ── optional histograms ────────────────────────────────────────────────────
    if HAS_MPL:
        sim_all = X_val[:, 3]   # nearest_sim

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(sim_all, bins=100, color="steelblue", edgecolor="none", alpha=0.8)
        ax.set_xlabel("Nearest-neighbor cosine similarity")
        ax.set_ylabel("Count")
        ax.set_title("Val: nearest_sim distribution")
        fig.tight_layout()
        fig.savefig(audit_dir / "nearest_sim_hist_val.png", dpi=120)
        plt.close(fig)
        print(f"\nSaved: {audit_dir / 'nearest_sim_hist_val.png'}")

        if has_fixed:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
            for ax, mask, label, col in [
                (axes[0], help_mask, "Retrieval Helps", "forestgreen"),
                (axes[1], hurt_mask, "Retrieval Hurts", "firebrick"),
            ]:
                vals = sim_all[mask]
                ax.hist(vals, bins=80, color=col, alpha=0.75)
                ax.set_xlabel("nearest_sim")
                ax.set_title(f"{label} (n={mask.sum()})")
            fig.suptitle("Nearest-sim distribution: help vs hurt")
            fig.tight_layout()
            fig.savefig(audit_dir / "nearest_sim_help_vs_hurt.png", dpi=120)
            plt.close(fig)
            print(f"Saved: {audit_dir / 'nearest_sim_help_vs_hurt.png'}")
    else:
        print("  (matplotlib not available — skipping histograms)")

    print("\nDone.")


if __name__ == "__main__":
    main()

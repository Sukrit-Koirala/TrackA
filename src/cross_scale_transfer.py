"""
cross_scale_transfer.py

Tests whether a Q-MLP controller trained at one datastore scale transfers
to other datastore scales without retraining.

For each source→target pair the source model's stored feat_mu/feat_std
(which covers the full 18-dim (obs || action) input) is used to normalise
target features — this is the correct transfer setup.

Usage:
  python src/cross_scale_transfer.py \\
    --root outputs_scale_sweep \\
    --scales 50000 100000 200000 \\
    --seed 42

Flags:
  --root         Scale sweep root dir  (default: outputs_scale_sweep)
  --scales       Datastore sizes to test
  --seed         Seed (default: 42)
  --force_train  Retrain source models even if checkpoints exist
  --force_eval   Re-evaluate even if transfer_full.json already saved
  --no_plots     Skip matplotlib heatmaps
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json
import math
import os
import subprocess
import numpy as np
import torch
import pandas as pd

from utils import get_device, set_seed, ppl
from train_controller import build_features
from train_q_controller import (
    QMLP, build_action_features, load_q_model,
    apply_q_controller, eval_chosen_on_val,
)


# ── constants ──────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "gpt_entropy", "gpt_top1_prob", "gpt_margin",
    "nearest_sim", "mean_top4_sim", "mean_top8_sim",
    "mean_top16_sim", "mean_top32_sim", "std_top32_sim",
    "label_entropy_top8", "label_entropy_top32",
    "mode_match_top8", "mode_match_top32",
]

CTRL_TYPES = [("Q-MLP-full", "q_mlp_full.pt"), ("Q-MLP-A", "q_mlp_A.pt")]


# ── path helpers ───────────────────────────────────────────────────────────────

def scale_label(n: int) -> str:
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1_000:
        return f"{n // 1_000}k"
    return str(n)


def scale_dir(root: Path, n: int, seed: int) -> Path:
    return root / f"scale_{scale_label(n)}_seed{seed}"


# ── action reconstruction ──────────────────────────────────────────────────────

def parse_action_name(name: str) -> dict:
    """Reconstruct action dict from the name stored in the checkpoint."""
    if name == "gpt_only":
        return {"name": "gpt_only", "k": 0, "tau": 0.1, "alpha": 1.0}
    parts = name.split("_")           # e.g. ["k16", "t0.05", "a0.75"]
    k     = int(parts[0][1:])
    tau   = float(parts[1][1:])
    alpha = float(parts[2][1:])
    return {"name": name, "k": k, "tau": tau, "alpha": alpha}


def load_json_safe(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ── reference baselines ────────────────────────────────────────────────────────

def load_target_baselines(sdir: Path) -> dict:
    out = {
        "gpt_nll":            float("nan"),
        "best_fixed_nll":     float("nan"),
        "best_heuristic_nll": float("nan"),
        "within_q_full_nll":  float("nan"),
        "within_q_A_nll":     float("nan"),
    }
    bl = load_json_safe(sdir / "reports" / "fixed_baselines.json")
    if bl:
        out["gpt_nll"]        = bl.get("gpt_only_val_nll",            float("nan"))
        out["best_fixed_nll"] = bl.get("best_train_selected_val_nll", float("nan"))

    bh = load_json_safe(sdir / "reports" / "best_heuristics.json")
    if bh:
        out["best_heuristic_nll"] = bh.get("val_nll", float("nan"))

    qm = load_json_safe(sdir / "reports" / "q_controller_metrics.json")
    if qm:
        if "Q-MLP-full" in qm:
            out["within_q_full_nll"] = qm["Q-MLP-full"].get("mean_nll", float("nan"))
        if "Q-MLP-A" in qm:
            out["within_q_A_nll"] = qm["Q-MLP-A"].get("mean_nll", float("nan"))
    return out


# ── training on demand ─────────────────────────────────────────────────────────

def train_source_if_needed(
    src_n: int, seed: int, root: Path, force_train: bool
) -> bool:
    """Train Q-MLP at source scale if checkpoint missing. Returns True if trained."""
    sdir = scale_dir(root, src_n, seed)
    ckpt = sdir / "models" / "q_mlp_full.pt"
    if ckpt.exists() and not force_train:
        return False

    label    = scale_label(src_n)
    cfg_path = Path("configs") / "scale_sweep" / f"scale_{label}_seed{seed}.yaml"
    if not cfg_path.exists():
        print(f"  WARNING: Config {cfg_path} not found — cannot train {label}")
        return False

    print(f"\n  Training Q-MLP at source {label} ...")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "train_q_controller.py"),
         "--config", str(cfg_path)],
        env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8",
    )
    if result.returncode != 0:
        print(f"  *** Training FAILED for {label} ***")
        print(result.stdout[-3000:])
        return False
    print(f"  Trained -> {ckpt}")
    return True


# ── transfer evaluation ────────────────────────────────────────────────────────

def transfer_eval_one(
    model: QMLP,
    actions_sub: list[dict],
    act_feats: torch.Tensor,       # [A_sub, 5]  CPU
    target_val_data: dict,
    target_val_nbrs: dict,
    device: torch.device,
) -> dict:
    """
    Apply source model to target val with source normalisation.
    Returns metrics dict plus 'target_obs' tensor for OOD analysis.
    """
    cfg = {"max_k": 64, "eps": 1e-12}

    target_obs = build_features(target_val_data, target_val_nbrs)  # [N, 13] CPU
    chosen     = apply_q_controller(model, target_obs, act_feats, device)  # [N] CPU

    metrics = eval_chosen_on_val(
        chosen, actions_sub, target_val_data, target_val_nbrs, cfg
    )
    metrics["target_obs"] = target_obs
    return metrics


# ── feature shift ──────────────────────────────────────────────────────────────

def compute_feature_shift(
    root: Path, scales: list[int], seed: int, max_k: int = 64
) -> pd.DataFrame:
    """Compute mean/std/quantiles of 13 obs features for each scale's val split."""
    rows = []
    for n in scales:
        sdir     = scale_dir(root, n, seed)
        sp, np_  = sdir / "states" / "val.pt", sdir / "neighbors" / f"val_top{max_k}.pt"
        if not sp.exists() or not np_.exists():
            print(f"  WARNING: val files missing for {scale_label(n)}, skipping feature shift")
            continue
        vd   = torch.load(sp,  weights_only=False)
        vn   = torch.load(np_, weights_only=False)
        obs  = build_features(vd, vn).numpy()   # [N, 13]
        for fi, fname in enumerate(FEATURE_NAMES):
            col = obs[:, fi]
            rows.append({
                "feature": fname,
                "scale":   scale_label(n),
                "mean":    float(col.mean()),
                "std":     float(col.std()),
                "p05":     float(np.percentile(col, 5)),
                "p50":     float(np.median(col)),
                "p95":     float(np.percentile(col, 95)),
            })
    return pd.DataFrame(rows)


# ── OOD analysis ───────────────────────────────────────────────────────────────

def compute_ood_report(pair_ood_data: list[dict]) -> pd.DataFrame:
    """
    For each source->target pair, check whether source-normalised target
    obs features are OOD.  Uses the first 13 dims of feat_mu / feat_std
    (the obs portion; action-feature dims 13-17 are scale-invariant).
    """
    rows = []
    for pr in pair_ood_data:
        mu13  = pr["obs_mu"].squeeze()[:13].cpu().numpy()   # [13]
        std13 = pr["obs_std"].squeeze()[:13].cpu().numpy()  # [13]
        obs   = pr["target_obs"].numpy()                    # [N, 13]
        z     = np.abs((obs - mu13) / std13)                # [N, 13]
        for fi, fname in enumerate(FEATURE_NAMES):
            col = z[:, fi]
            rows.append({
                "source_scale": pr["source_scale"],
                "target_scale": pr["target_scale"],
                "feature":      fname,
                "frac_z_gt3":   float((col > 3).mean()),
                "frac_z_gt5":   float((col > 5).mean()),
                "max_abs_z":    float(col.max()),
            })
    return pd.DataFrame(rows)


# ── matrix building ────────────────────────────────────────────────────────────

def build_matrix(
    results: list[dict],
    ctrl_type: str,
    metric: str,
    scales: list[int],
) -> pd.DataFrame:
    slabels = [scale_label(n) for n in scales]
    mat = pd.DataFrame(
        float("nan"),
        index=[f"train={l}" for l in slabels],
        columns=slabels,
        dtype=float,
    )
    for r in results:
        if r["controller_type"] != ctrl_type:
            continue
        row = f"train={r['source_scale']}"
        col = r["target_scale"]
        if row in mat.index and col in mat.columns:
            mat.loc[row, col] = r.get(metric, float("nan"))
    return mat


# ── plots ──────────────────────────────────────────────────────────────────────

def try_plots(results: list[dict], scales: list[int], out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        plot_dir = out_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        slabels = [scale_label(n) for n in scales]

        specs = [
            ("Q-MLP-full", "target_val_nll",             "Q-MLP-full Val NLL",              "q_mlp_full_transfer_nll"),
            ("Q-MLP-full", "delta_vs_target_best_fixed",  "Q-MLP-full delta vs target fixed", "q_mlp_full_delta_vs_fixed"),
            ("Q-MLP-A",    "target_val_nll",             "Q-MLP-A Val NLL",                 "q_mlp_A_transfer_nll"),
            ("Q-MLP-A",    "delta_vs_target_best_fixed",  "Q-MLP-A delta vs target fixed",   "q_mlp_A_delta_vs_fixed"),
        ]
        for ctrl_type, metric, title, fname in specs:
            mat  = build_matrix(results, ctrl_type, metric, scales)
            data = mat.values.astype(float)

            fig, ax = plt.subplots(figsize=(5, 4))
            if "nll" in metric:
                cmap  = "YlOrRd_r"
                im    = ax.imshow(data, cmap=cmap)
            else:
                vmin = min(float(np.nanmin(data)), -0.005)
                vmax = max(float(np.nanmax(data)),  0.005)
                norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
                im   = ax.imshow(data, cmap="RdYlGn_r", norm=norm)

            ax.set_xticks(range(len(slabels)));  ax.set_xticklabels(slabels)
            ax.set_yticks(range(len(slabels)));  ax.set_yticklabels([f"train {l}" for l in slabels])
            ax.set_xlabel("Eval scale"); ax.set_title(title)
            for i in range(len(slabels)):
                for j in range(len(slabels)):
                    v = data[i, j]
                    if not math.isnan(v):
                        ax.text(j, i, f"{v:.4f}", ha="center", va="center",
                                fontsize=8, color="black")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.savefig(plot_dir / f"{fname}.png", dpi=150)
            plt.close()

        print(f"  Plots -> {plot_dir}")
    except Exception as e:
        print(f"  Plots skipped: {e}")


# ── verdict ────────────────────────────────────────────────────────────────────

def verdict(results: list[dict]) -> str:
    off = [r for r in results
           if r["controller_type"] == "Q-MLP-full"
           and r["source_scale"] != r["target_scale"]]
    if not off:
        return "UNKNOWN"
    n_fix  = sum(1 for r in off
                 if not math.isnan(r.get("delta_vs_target_best_fixed", float("nan")))
                 and r["delta_vs_target_best_fixed"] < 0)
    frac   = n_fix / len(off)
    if frac >= 0.8:
        return "STRONG TRANSFER SUCCESS"
    if frac >= 0.5:
        return "MEDIUM TRANSFER SUCCESS"
    n_gpt  = sum(1 for r in off
                 if r.get("target_val_nll", float("nan")) < r.get("target_gpt_nll", float("inf")))
    if n_gpt == len(off):
        return "WEAK TRANSFER"
    return "TRANSFER FAILURE"


# ── report ─────────────────────────────────────────────────────────────────────

def generate_report(
    results: list[dict],
    feat_df: pd.DataFrame,
    ood_df: pd.DataFrame,
    scales: list[int],
    seed: int,
) -> str:
    v       = verdict(results)
    slabels = [scale_label(n) for n in scales]
    lines   = []

    def h(lvl, txt):  lines.append("#" * lvl + " " + txt); lines.append("")
    def p(txt=""):    lines.append(txt)

    h(1, "Cross-Scale Controller Transfer Report")
    p(f"Seed: {seed}  |  Scales: {', '.join(slabels)}")
    p()

    # -- 1. Goal --
    h(2, "1. Goal")
    p("Tests whether a Q-MLP controller trained at one datastore size transfers")
    p("to other datastore sizes without retraining.")
    p()
    p("**Key distinction from the prior scale sweep:**")
    p()
    p("- *Scale sweep*: Controllers retrained from scratch at each scale.")
    p("- *This experiment*: Source controller applied directly to target val,")
    p("  normalising target features with source-scale mu/std (stored in checkpoint).")
    p()
    p("If transfer works: the controller learned general gate-control signals,")
    p("not just memorised scale-specific statistics.")
    p()

    # -- 2. Q-MLP-full NLL matrix --
    h(2, "2. Q-MLP-full Transfer NLL Matrix")
    p("Val NLL (lower is better). Diagonal = within-scale (reproduces scale sweep).")
    p()
    mat = build_matrix(results, "Q-MLP-full", "target_val_nll", scales)
    p("```")
    p("Q-MLP-full Val NLL   (rows=train scale, cols=eval scale)")
    p()
    p(f"{'train \\ eval':<14}" + "  ".join(f"{l:>8}" for l in slabels))
    for sl in slabels:
        row = f"{'train ' + sl:<14}"
        for el in slabels:
            v2 = mat.loc[f"train={sl}", el]
            row += f"  {v2:>8.4f}" if not math.isnan(v2) else f"  {'—':>8}"
        p(row)
    p("```")
    p()

    # -- 3. Delta vs target fixed --
    h(2, "3. Q-MLP-full Delta vs Target Best Fixed kNN")
    p("Negative = transfer controller beats target fixed.  Positive = worse than target fixed.")
    p()
    mat_d = build_matrix(results, "Q-MLP-full", "delta_vs_target_best_fixed", scales)
    p("```")
    p("Q-MLP-full delta-NLL vs target best fixed kNN")
    p()
    p(f"{'train \\ eval':<14}" + "  ".join(f"{l:>9}" for l in slabels))
    for sl in slabels:
        row = f"{'train ' + sl:<14}"
        for el in slabels:
            v2 = mat_d.loc[f"train={sl}", el]
            if math.isnan(v2):
                row += f"  {'—':>9}"
            else:
                row += f"  {v2:>+9.4f}"
        p(row)
    p("```")
    p()

    # -- 4. Q-MLP-A --
    h(2, "4. Q-MLP-A Transfer Results")
    mat_A  = build_matrix(results, "Q-MLP-A", "target_val_nll", scales)
    mat_Ad = build_matrix(results, "Q-MLP-A", "delta_vs_target_best_fixed", scales)
    p("**Q-MLP-A Val NLL:**")
    p("```")
    p(f"{'train \\ eval':<14}" + "  ".join(f"{l:>8}" for l in slabels))
    for sl in slabels:
        row = f"{'train ' + sl:<14}"
        for el in slabels:
            v2 = mat_A.loc[f"train={sl}", el]
            row += f"  {v2:>8.4f}" if not math.isnan(v2) else f"  {'—':>8}"
        p(row)
    p("```")
    p()
    p("**Q-MLP-A delta vs target best fixed:**")
    p("```")
    p(f"{'train \\ eval':<14}" + "  ".join(f"{l:>9}" for l in slabels))
    for sl in slabels:
        row = f"{'train ' + sl:<14}"
        for el in slabels:
            v2 = mat_Ad.loc[f"train={sl}", el]
            if math.isnan(v2):
                row += f"  {'—':>9}"
            else:
                row += f"  {v2:>+9.4f}"
        p(row)
    p("```")
    p()

    # -- 5. Retrieval behavior --
    h(2, "5. Retrieval Behavior Under Transfer")
    p("Retrieval usage and avg k for Q-MLP-full across all source->target pairs.")
    p()
    p("| source -> target | ret% | avg_k | NLL | Δ vs fixed |")
    p("|-----------------|------|-------|-----|------------|")
    for r in sorted(results, key=lambda x: (x["controller_type"], x["source_scale"], x["target_scale"])):
        if r["controller_type"] != "Q-MLP-full":
            continue
        ret  = r.get("target_retrieval_usage", float("nan"))
        k2   = r.get("target_avg_k", float("nan"))
        nll  = r.get("target_val_nll", float("nan"))
        df2  = r.get("delta_vs_target_best_fixed", float("nan"))
        diag = " *" if r["source_scale"] == r["target_scale"] else ""
        p(f"| {r['source_scale']:>5} -> {r['target_scale']:<5}{diag} | "
          f"{ret*100:>4.1f}% | {k2:>5.1f} | {nll:.4f} | {df2:>+.4f} |")
    p()
    p("> `*` marks diagonal (within-scale) pairs.")
    p("> A 50k-trained controller may underuse retrieval at 200k; a 200k-trained")
    p("> controller may over-retrieve at 50k.")
    p()

    # -- 6. Feature shift summary --
    h(2, "6. Feature Distribution Shift")
    key_feats = ["nearest_sim", "mean_top8_sim", "label_entropy_top8",
                 "label_entropy_top32", "gpt_entropy", "gpt_top1_prob"]
    if not feat_df.empty:
        p("Mean of key observation features per scale's val split.")
        p()
        p("| feature | " + " | ".join(slabels) + " |")
        p("|---------|" + "|".join(["----------"] * len(slabels)) + "|")
        for fn in key_feats:
            sub = feat_df[feat_df["feature"] == fn]
            vals = []
            for sl in slabels:
                r2 = sub[sub["scale"] == sl]
                vals.append(f"{r2.iloc[0]['mean']:.4f}" if len(r2) else "—")
            p(f"| {fn:<28} | " + " | ".join(vals) + " |")
        p()

    if not ood_df.empty:
        p("**OOD summary** — max fraction of target features with |z| > 3 (per source->target pair):**")
        p()
        pair_sum = ood_df.groupby(["source_scale", "target_scale"]).agg(
            max_frac_gt3=("frac_z_gt3", "max"),
            max_frac_gt5=("frac_z_gt5", "max"),
            max_abs_z=("max_abs_z", "max"),
        ).reset_index()
        p("| source -> target | max frac |z|>3 | max frac |z|>5 | max |z| |")
        p("|------------------|------------|------------|---------|")
        for _, row in pair_sum.iterrows():
            p(f"| {row['source_scale']:>5} -> {row['target_scale']:<5} | "
              f"{row['max_frac_gt3']:>10.3f} | "
              f"{row['max_frac_gt5']:>10.3f} | "
              f"{row['max_abs_z']:>7.2f} |")
        p()

    # -- 7. Interpretation --
    h(2, "7. Interpretation")
    vstr = verdict(results)
    p(f"**Verdict: [{vstr}]**")
    p()

    off  = [r for r in results
            if r["controller_type"] == "Q-MLP-full" and r["source_scale"] != r["target_scale"]]
    n_b  = sum(1 for r in off
               if not math.isnan(r.get("delta_vs_target_best_fixed", float("nan")))
               and r["delta_vs_target_best_fixed"] < 0)
    p(f"Q-MLP-full beats target best fixed kNN on **{n_b}/{len(off)}** off-diagonal pairs.")
    p()

    if vstr == "STRONG TRANSFER SUCCESS":
        p("The learned gate-control policy captures scale-stable signals, not just")
        p("scale-specific tuning. Controllers trained at one scale generalise")
        p("effectively to other scales.")
    elif vstr == "MEDIUM TRANSFER SUCCESS":
        p("Dynamic gate control works, but the controller may need scale-aware")
        p("calibration. Transfer is asymmetric or partially successful.")
    elif vstr == "WEAK TRANSFER":
        p("Transferred controllers still beat GPT-only but fail to beat target")
        p("best fixed kNN. The controller learned something general but not enough")
        p("to displace scale-specific fixed tuning.")
    else:
        p("The current controller is scale-specific. The scale sweep still succeeded,")
        p("but cross-scale generalisation is not established.")
    p()

    h(2, "8. What This Does Not Test")
    for item in [
        "WRITE or PRUNE gates", "Emergent abstractions or symbol binding",
        "Region or module discovery", "Hierarchy or semantic clustering",
        "Reinforcement learning (PPO etc.)", "Transformer fine-tuning", "Branch B",
    ]:
        p(f"- {item}")
    p()
    p("It only tests: **Are learned gate-control policies scale-transferable?**")
    p()

    return "\n".join(lines)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cross-scale controller transfer for Branch A Gate-Chain MVP")
    parser.add_argument("--root",        default="outputs_scale_sweep")
    parser.add_argument("--scales",      nargs="+", type=int,
                        default=[50_000, 100_000, 200_000])
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--force_train", action="store_true")
    parser.add_argument("--force_eval",  action="store_true")
    parser.add_argument("--no_plots",    action="store_true")
    args = parser.parse_args()

    root   = Path(args.root)
    seed   = args.seed
    scales = args.scales
    device = get_device({"device": "cuda"})
    set_seed(seed)

    out_dir = root / "cross_scale_transfer"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nBranch A Gate-Chain MVP -- Cross-Scale Transfer")
    print(f"Seed: {seed}  |  Scales: {[scale_label(n) for n in scales]}")
    print(f"Output dir: {out_dir}")
    print(f"Device: {device}")

    # ── early exit if already done ────────────────────────────────────────────
    results_path = out_dir / "cross_scale_transfer_full.json"
    if results_path.exists() and not args.force_eval:
        print(f"\nResults already saved at {results_path}. Use --force_eval to rerun.")
        with open(results_path) as f:
            results = json.load(f)
        feat_df = pd.read_csv(out_dir / "feature_shift.csv") \
            if (out_dir / "feature_shift.csv").exists() else pd.DataFrame()
        ood_df  = pd.read_csv(out_dir / "ood_feature_report.csv") \
            if (out_dir / "ood_feature_report.csv").exists() else pd.DataFrame()
        _finish(results, feat_df, ood_df, scales, seed, out_dir, args.no_plots)
        return

    # ── 1. Ensure source checkpoints ──────────────────────────────────────────
    print("\n[1] Checking source checkpoints ...")
    for src_n in scales:
        trained = train_source_if_needed(src_n, seed, root, args.force_train)
        sdir    = scale_dir(root, src_n, seed)
        ok_full = (sdir / "models" / "q_mlp_full.pt").exists()
        ok_A    = (sdir / "models" / "q_mlp_A.pt").exists()
        tag     = "trained" if trained else ("exists" if ok_full else "MISSING")
        print(f"  {scale_label(src_n):>5}: {tag}  "
              f"(q_mlp_full: {ok_full}, q_mlp_A: {ok_A})")

    # ── 2. Load all target val data ───────────────────────────────────────────
    print("\n[2] Loading target val data ...")
    val_cache:  dict[int, tuple] = {}
    tgt_baselines: dict[int, dict] = {}

    for tgt_n in scales:
        tdir = scale_dir(root, tgt_n, seed)
        sp   = tdir / "states" / "val.pt"
        np_  = tdir / "neighbors" / "val_top64.pt"
        if not sp.exists() or not np_.exists():
            print(f"  WARNING: val files missing for {scale_label(tgt_n)}, skipping as target")
            continue
        vd = torch.load(sp,  weights_only=False)
        vn = torch.load(np_, weights_only=False)
        val_cache[tgt_n]    = (vd, vn)
        tgt_baselines[tgt_n] = load_target_baselines(tdir)
        bl = tgt_baselines[tgt_n]
        print(f"  {scale_label(tgt_n):>5}: N_val={len(vd['y']):,}  "
              f"best_fixed={bl['best_fixed_nll']:.4f}  "
              f"within_q_full={bl['within_q_full_nll']:.4f}")

    # ── 3. Run all source -> target pairs ─────────────────────────────────────
    print("\n[3] Running transfer evaluations ...")
    results: list[dict] = []
    pair_ood: list[dict] = []

    for src_n in scales:
        src_label = scale_label(src_n)
        sdir      = scale_dir(root, src_n, seed)

        for ctrl_type, ckpt_name in CTRL_TYPES:
            ckpt_path = sdir / "models" / ckpt_name
            if not ckpt_path.exists():
                print(f"  SKIP: {src_label} {ctrl_type} — checkpoint not found")
                continue

            print(f"\n  Source {src_label} | {ctrl_type}")
            model, action_names = load_q_model(ckpt_path, device)
            model.eval()

            actions_sub = [parse_action_name(n) for n in action_names]
            act_feats   = torch.stack([
                build_action_features(a, 64) for a in actions_sub
            ])  # [A_sub, 5]  CPU

            obs_mu  = model.get_buffer("feat_mu").cpu()   # [1, 18]
            obs_std = model.get_buffer("feat_std").cpu()  # [1, 18]

            for tgt_n in scales:
                tgt_label = scale_label(tgt_n)
                if tgt_n not in val_cache:
                    continue

                vd, vn   = val_cache[tgt_n]
                bl       = tgt_baselines[tgt_n]

                print(f"    -> {tgt_label}  (N={len(vd['y']):,})", end="  ", flush=True)

                m = transfer_eval_one(model, actions_sub, act_feats, vd, vn, device)
                target_obs = m.pop("target_obs")

                nll = m["mean_nll"]
                within_q = (bl["within_q_full_nll"] if ctrl_type == "Q-MLP-full"
                            else bl["within_q_A_nll"])

                row = {
                    "source_scale":              src_label,
                    "target_scale":              tgt_label,
                    "controller_type":           ctrl_type,
                    "target_val_nll":            nll,
                    "target_val_ppl":            ppl(nll),
                    "target_avg_k":              m["mean_k"],
                    "target_retrieval_usage":    m["retrieval_usage"],
                    "target_gpt_nll":            bl["gpt_nll"],
                    "target_best_fixed_nll":     bl["best_fixed_nll"],
                    "target_best_heuristic_nll": bl["best_heuristic_nll"],
                    "target_within_scale_q_nll": within_q,
                    "delta_vs_target_best_fixed":  nll - bl["best_fixed_nll"],
                    "delta_vs_target_heuristic":   nll - bl["best_heuristic_nll"],
                    "delta_vs_within_scale_q":     nll - within_q,
                }
                results.append(row)

                if ctrl_type == "Q-MLP-full":
                    pair_ood.append({
                        "source_scale": src_label,
                        "target_scale": tgt_label,
                        "obs_mu":       obs_mu,
                        "obs_std":      obs_std,
                        "target_obs":   target_obs,
                    })

                print(f"NLL={nll:.4f}  ret={m['retrieval_usage']*100:.0f}%  "
                      f"avg_k={m['mean_k']:.1f}  "
                      f"dfix={nll-bl['best_fixed_nll']:+.4f}  "
                      f"dwithin={nll-within_q:+.4f}")

    # ── 4. Feature shift ──────────────────────────────────────────────────────
    print("\n[4] Computing feature shift ...")
    feat_df = compute_feature_shift(root, scales, seed)

    # ── 5. OOD report ─────────────────────────────────────────────────────────
    print("[5] Computing OOD feature report ...")
    ood_df = compute_ood_report(pair_ood) if pair_ood else pd.DataFrame()

    _finish(results, feat_df, ood_df, scales, seed, out_dir, args.no_plots)


def _finish(
    results: list[dict],
    feat_df: pd.DataFrame,
    ood_df: pd.DataFrame,
    scales: list[int],
    seed: int,
    out_dir: Path,
    no_plots: bool,
):
    print("\n[6] Saving outputs ...")
    slabels = [scale_label(n) for n in scales]

    # Full results
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "cross_scale_transfer_full.csv", index=False)
    with open(out_dir / "cross_scale_transfer_full.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  cross_scale_transfer_full.csv  ({len(df)} rows)")

    # Per-controller matrices
    for ctrl_type, prefix in [("Q-MLP-full", "q_mlp_full"), ("Q-MLP-A", "q_mlp_A")]:
        build_matrix(results, ctrl_type, "target_val_nll",
                     scales).to_csv(out_dir / f"{prefix}_transfer_matrix.csv")
        build_matrix(results, ctrl_type, "delta_vs_target_best_fixed",
                     scales).to_csv(out_dir / f"{prefix}_delta_matrix.csv")
        print(f"  {prefix}_transfer_matrix.csv")

    # Feature shift
    if not feat_df.empty:
        feat_df.to_csv(out_dir / "feature_shift.csv", index=False)
        with open(out_dir / "feature_shift.json", "w") as f:
            json.dump(feat_df.to_dict(orient="records"), f, indent=2)
        print(f"  feature_shift.csv  ({len(feat_df)} rows)")

    # OOD
    if not ood_df.empty:
        ood_df.to_csv(out_dir / "ood_feature_report.csv", index=False)
        print(f"  ood_feature_report.csv  ({len(ood_df)} rows)")

    # Plots
    if not no_plots:
        print("\n[7] Generating plots ...")
        try_plots(results, scales, out_dir)

    # Report
    print("\n[8] Writing report ...")
    report = generate_report(results, feat_df, ood_df, scales, seed)
    rpath  = out_dir / "CROSS_SCALE_TRANSFER_REPORT.md"
    with open(rpath, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  {rpath}")

    # Terminal summary
    v = verdict(results)
    print(f"\n{'='*70}")
    print(f"  VERDICT: [{v}]")
    print(f"{'='*70}")
    full_results = sorted(
        [r for r in results if r["controller_type"] == "Q-MLP-full"],
        key=lambda x: (x["source_scale"], x["target_scale"]),
    )
    for r in full_results:
        diag = "(diag)" if r["source_scale"] == r["target_scale"] else "      "
        win  = "v" if r["delta_vs_target_best_fixed"] < 0 else "x"
        print(f"  [{win}] {r['source_scale']:>5} -> {r['target_scale']:<5} {diag}  "
              f"NLL={r['target_val_nll']:.4f}  "
              f"dfix={r['delta_vs_target_best_fixed']:>+.4f}  "
              f"dwithin={r['delta_vs_within_scale_q']:>+.4f}")
    print(f"{'='*70}")
    print(f"\nOutputs: {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()

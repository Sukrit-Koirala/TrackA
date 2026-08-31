"""
evaluate_write_memory.py

Evaluates each written memory (method × budget) using two read policies:
  A. Best fixed action (CT-selected) using written-memory neighbors
  B. Q-MLP-full transfer: apply the full-datastore Q-MLP to written-memory
     validation features (uses source-scale normalization stats)

Compares against full-datastore baselines:
  full_fixed_nll   best fixed kNN on full 200k datastore
  full_qmlp_nll    Q-MLP-full on full 200k datastore

Generates:
  reports/write_memory_results.csv
  reports/write_memory_results.json
  plots/nll_vs_budget.png
  plots/delta_vs_random_by_budget.png
  plots/memory_fraction_vs_nll.png
  MVP2_WRITE_MEMORY_REPORT.md

Usage:
  python src/evaluate_write_memory.py \\
    --source outputs_scale_sweep/scale_200k_seed42 \\
    --output outputs_mvp2_write_memory
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json
import math
import re
import numpy as np
import torch
import pandas as pd

from utils import build_action_grid, get_device, set_seed, ppl
from gate_chain import evaluate_action_for_queries
from train_controller import build_features


MAX_K = 64
DEFAULT_K_VALUES   = [4, 8, 16, 32, 64]
DEFAULT_TAU_VALUES = [0.05, 0.1, 0.2]
DEFAULT_ALPHA_VALUES = [0.25, 0.5, 0.75]
EPS = 1e-12


def build_default_actions():
    acts = [{"name": "gpt_only", "k": 0, "tau": 0.1, "alpha": 1.0}]
    for k in DEFAULT_K_VALUES:
        for tau in DEFAULT_TAU_VALUES:
            for alpha in DEFAULT_ALPHA_VALUES:
                acts.append({"name": f"k{k}_t{tau}_a{alpha}",
                              "k": k, "tau": tau, "alpha": alpha})
    return acts


def find_best_fixed_action(ct_data, ct_nbrs, actions):
    """Select action with lowest mean NLL on controller_train."""
    k_eff = ct_nbrs["neighbor_sims"].shape[1]
    best_nll, best_action = float("inf"), None
    for action in actions:
        if action["k"] > k_eff:
            continue
        m = evaluate_action_for_queries(ct_data, ct_nbrs, action,
                                         max_k=k_eff, lambda_cost=0.0, eps=EPS)
        if m["mean_nll"] < best_nll:
            best_nll, best_action = m["mean_nll"], action
    return best_action, best_nll


def eval_action_on_val(val_data, val_nbrs, action):
    k_eff = val_nbrs["neighbor_sims"].shape[1]
    return evaluate_action_for_queries(val_data, val_nbrs, action,
                                        max_k=k_eff, lambda_cost=0.0, eps=EPS)


def tensor_only_nbrs(nbrs: dict) -> dict:
    """Return only tensor-valued keys so eval_chosen_on_val can do v[mask] safely."""
    return {k: v for k, v in nbrs.items() if isinstance(v, torch.Tensor)}


def qmlp_transfer_eval(model, actions, act_feats, val_data, val_nbrs, device):
    """Apply full-datastore Q-MLP to written-memory val features."""
    from train_q_controller import apply_q_controller, eval_chosen_on_val
    k_eff = val_nbrs["neighbor_sims"].shape[1]
    # build_features indexes up to sims[:, :32]; pad columns with zeros if needed
    if k_eff < 64:
        pad = 64 - k_eff
        N_q = val_nbrs["neighbor_sims"].shape[0]
        val_nbrs = dict(val_nbrs)
        val_nbrs["neighbor_sims"] = torch.cat(
            [val_nbrs["neighbor_sims"], torch.zeros(N_q, pad)], dim=1)
        val_nbrs["neighbor_y"] = torch.cat(
            [val_nbrs["neighbor_y"], torch.zeros(N_q, pad, dtype=torch.long)], dim=1)
    obs   = build_features(val_data, val_nbrs)   # [N_val, 13]
    # clamp act_feats to valid k range
    valid = [i for i, a in enumerate(actions) if a["k"] <= k_eff]
    act_sub = [actions[i] for i in valid]
    feat_sub = act_feats[valid]

    chosen  = apply_q_controller(model, obs, feat_sub, device)
    metrics = eval_chosen_on_val(chosen, act_sub, val_data,
                                  tensor_only_nbrs(val_nbrs),
                                  {"max_k": k_eff, "eps": EPS})
    return metrics


def parse_stem(stem: str):
    """Extract method and budget from filename stem like 'high_gpt_loss_B10000'."""
    m = re.match(r"^(.+)_B(\d+)$", stem)
    if not m:
        return stem, 0
    return m.group(1), int(m.group(2))


def load_full_ds_baselines(src: Path) -> dict:
    out = {"gpt_nll": float("nan"), "fixed_nll": float("nan"), "qmlp_nll": float("nan")}
    bl = src / "reports" / "fixed_baselines.json"
    if bl.exists():
        with open(bl) as f:
            d = json.load(f)
        out["gpt_nll"]   = d.get("gpt_only_val_nll",            float("nan"))
        out["fixed_nll"] = d.get("best_train_selected_val_nll",  float("nan"))
    qm = src / "reports" / "q_controller_metrics.json"
    if qm.exists():
        with open(qm) as f:
            d = json.load(f)
        if "Q-MLP-full" in d:
            out["qmlp_nll"] = d["Q-MLP-full"].get("mean_nll", float("nan"))
    return out


def try_plots(df: pd.DataFrame, budgets: list, out_dir: Path, full_fixed: float, full_qmlp: float):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_dir = out_dir / "plots"
        plot_dir.mkdir(exist_ok=True)

        # Group by method × budget for best_fixed read policy
        sub = df[df["read_policy"] == "best_fixed"].copy()

        # ── NLL vs budget ──────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 5))
        methods = sub["method"].unique()
        for meth in sorted(methods):
            m_df = sub[sub["method"] == meth].sort_values("budget")
            if m_df.empty: continue
            style = "--" if "oracle" in meth else "-"
            ax.plot(m_df["budget"], m_df["val_nll"], style, marker="o",
                    label=meth, linewidth=1.5, markersize=4)
        ax.axhline(full_fixed, color="black",  linestyle=":", label="full_ds_fixed")
        ax.axhline(full_qmlp,  color="purple", linestyle=":", label="full_ds_Q-MLP")
        ax.set_xlabel("Memory budget (B)")
        ax.set_ylabel("Val NLL")
        ax.set_title("Val NLL vs memory budget (best-fixed read)")
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
        ax.set_xscale("log")
        plt.tight_layout()
        plt.savefig(plot_dir / "nll_vs_budget.png", dpi=150)
        plt.close()

        # ── delta vs random by budget ──────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 5))
        rand_mean = sub[sub["method"].str.startswith("random_")].groupby("budget")["val_nll"].mean()
        for meth in sorted(methods):
            if meth.startswith("random_"): continue
            m_df = sub[sub["method"] == meth].sort_values("budget")
            if m_df.empty: continue
            deltas = []
            bs = []
            for _, row in m_df.iterrows():
                b = row["budget"]
                if b in rand_mean.index:
                    deltas.append(row["val_nll"] - rand_mean[b])
                    bs.append(b)
            if not bs: continue
            style = "--" if "oracle" in meth else "-"
            ax.plot(bs, deltas, style, marker="o", label=meth, linewidth=1.5, markersize=4)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5, label="random baseline")
        ax.set_xlabel("Memory budget (B)")
        ax.set_ylabel("Delta NLL vs random (negative = better than random)")
        ax.set_title("Method advantage over random memory selection")
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
        ax.set_xscale("log")
        plt.tight_layout()
        plt.savefig(plot_dir / "delta_vs_random_by_budget.png", dpi=150)
        plt.close()

        # ── memory fraction vs NLL ─────────────────────────────────────────────
        N_ds = sub["n_ds"].iloc[0] if "n_ds" in sub.columns else 200_000
        fig, ax = plt.subplots(figsize=(7, 5))
        for meth in sorted(methods):
            m_df = sub[sub["method"] == meth].sort_values("budget")
            if m_df.empty: continue
            frac = m_df["budget"] / N_ds
            ax.plot(frac, m_df["val_nll"], "-o", label=meth, markersize=4)
        ax.axhline(full_fixed, color="black",  linestyle=":", label="full_ds_fixed")
        ax.axhline(full_qmlp,  color="purple", linestyle=":", label="full_ds_Q-MLP")
        ax.set_xlabel("Memory fraction (B / N_datastore)")
        ax.set_ylabel("Val NLL")
        ax.set_title("Memory efficiency: val NLL vs memory fraction")
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
        plt.tight_layout()
        plt.savefig(plot_dir / "memory_fraction_vs_nll.png", dpi=150)
        plt.close()

        print(f"  Plots -> {plot_dir}")
    except Exception as e:
        print(f"  Plots skipped: {e}")


def generate_report(df: pd.DataFrame, full: dict, budgets: list, out_dir: Path) -> str:
    lines = []
    def h(n, t): lines.append("#" * n + " " + t); lines.append("")
    def p(t=""): lines.append(t)

    h(1, "MVP 2 Write Memory Report")
    p(f"Full-datastore baselines: GPT={full['gpt_nll']:.4f}  "
      f"Fixed={full['fixed_nll']:.4f}  Q-MLP={full['qmlp_nll']:.4f}")
    p()

    h(2, "1. Goal")
    p("MVP 2 tests whether a learned WRITE gate can select which raw predictive")
    p("evidence to keep under a memory budget, so that the subsequent")
    p("MATCH -> SELECT -> PREDICT -> MIX gate chain performs better than naive selection.")
    p()
    p("**This is not abstraction discovery.** Each written item is a raw evidence")
    p("entry (hidden state + next token label). WRITE selects; it does not compress.")
    p()

    h(2, "2. Setup")
    p(f"- Dataset: TinyStories (GPT-2 hidden states)")
    p(f"- Full datastore: 200k entries (story-level split)")
    p(f"- Memory budgets: {budgets}")
    p(f"- Read policies: best fixed action (CT-selected), Q-MLP transfer")
    p(f"- Memory selection methods: random (3 seeds), uniform_story, high_gpt_loss,")
    p(f"  high_gpt_entropy, token_rarity, coverage_memory, oracle_utility (diag),")
    p(f"  learned_write_linear, learned_write_gbr, learned_write_mlp")
    p()

    h(2, "3. Utility Label Construction")
    p("For each controller_train query, the improvement (NLL_GPT - NLL_final) from")
    p("retrieval is distributed to selected neighbor datastore entries, weighted by")
    p("softmax-similarity. Only matching neighbors (label == y_true) receive positive")
    p("utility. All neighbors of queries where retrieval hurt receive negative utility.")
    p("utility_net = utility_positive - 0.25 * negative_utility.")
    p()

    h(2, "4. Main Results (best-fixed read policy)")
    sub = df[df["read_policy"] == "best_fixed"]
    p("Val NLL by method and budget (lower is better).")
    p(f"Reference: full_ds_fixed={full['fixed_nll']:.4f}  full_ds_Q-MLP={full['qmlp_nll']:.4f}  GPT={full['gpt_nll']:.4f}")
    p()

    all_methods = sorted(sub["method"].unique())
    p("| method | " + " | ".join(str(b) for b in budgets) + " |")
    p("|--------|" + "|".join(["-------"] * len(budgets)) + "|")
    for meth in all_methods:
        row = f"| {meth:<30} |"
        m_df = sub[sub["method"] == meth]
        for b in budgets:
            r2 = m_df[m_df["budget"] == b]
            val = r2["val_nll"].values[0] if len(r2) else float("nan")
            row += f" {val:.4f} |" if not math.isnan(val) else "   —    |"
        p(row)
    p()

    h(2, "5. Budget Curve")
    rand_df = sub[sub["method"].str.startswith("random_")]
    rand_mean = rand_df.groupby("budget")["val_nll"].mean()
    learned_methods = [m for m in all_methods
                       if m.startswith("learned_write")]
    if learned_methods and not rand_mean.empty:
        p("Mean random NLL vs best learned WRITE NLL per budget:")
        p()
        p("| budget | random (mean) | best learned | delta |")
        p("|--------|---------------|--------------|-------|")
        for b in budgets:
            r_nll = rand_mean.get(b, float("nan"))
            best_l = float("inf")
            for lm in learned_methods:
                r2 = sub[(sub["method"] == lm) & (sub["budget"] == b)]
                if len(r2):
                    best_l = min(best_l, r2["val_nll"].values[0])
            if best_l == float("inf"):
                best_l = float("nan")
            delta = best_l - r_nll if not (math.isnan(r_nll) or math.isnan(best_l)) else float("nan")
            r_str = f"{r_nll:.4f}" if not math.isnan(r_nll) else "—"
            l_str = f"{best_l:.4f}" if not math.isnan(best_l) else "—"
            d_str = f"{delta:+.4f}" if not math.isnan(delta) else "—"
            p(f"| {b:>6} | {r_str:>13} | {l_str:>12} | {d_str:>5} |")
        p()

    h(2, "6. Comparison Against Baselines")
    n_budgets = len(budgets)
    for lm in learned_methods:
        wins_over = {}
        baselines_to_check = [m for m in all_methods
                               if m not in learned_methods and "oracle" not in m]
        for bm in baselines_to_check:
            w = 0
            for b in budgets:
                rl = sub[(sub["method"] == lm) & (sub["budget"] == b)]
                rb = sub[(sub["method"] == bm) & (sub["budget"] == b)]
                if len(rl) and len(rb):
                    if rl["val_nll"].values[0] < rb["val_nll"].values[0]:
                        w += 1
            wins_over[bm] = w
        p(f"**{lm}** beats baseline at N/{n_budgets} budgets:")
        for bm, w in wins_over.items():
            p(f"  - vs {bm}: {w}/{n_budgets}")
        p()

    h(2, "7. Memory Efficiency")
    p("Q-MLP transfer evaluation (source: full-datastore Q-MLP applied to memory neighbors):")
    p()
    qsub = df[df["read_policy"] == "qmlp_transfer"]
    if not qsub.empty:
        p("| method | budget | val_nll | vs_full_fixed | vs_full_qmlp |")
        p("|--------|--------|---------|---------------|--------------|")
        for _, row in qsub.sort_values(["method", "budget"]).iterrows():
            p(f"| {row['method']:<30} | {row['budget']:>6} | "
              f"{row['val_nll']:.4f} | {row['val_nll']-full['fixed_nll']:>+.4f} | "
              f"{row['val_nll']-full['qmlp_nll']:>+.4f} |")
        p()
    else:
        p("Q-MLP transfer data not available.")
        p()

    h(2, "8. Verdict")
    # Determine verdict
    if not learned_methods or not rand_mean.size:
        verdict = "UNKNOWN"
    else:
        n_beats_random = 0
        n_total = 0
        for lm in learned_methods:
            for b in budgets:
                rl = sub[(sub["method"] == lm) & (sub["budget"] == b)]
                if b in rand_mean.index and len(rl):
                    n_total += 1
                    if rl["val_nll"].values[0] < rand_mean[b]:
                        n_beats_random += 1
        frac = n_beats_random / n_total if n_total else 0

        # Check oracle vs learned gap
        oracle_df = sub[sub["method"] == "oracle_utility_net"]

        # Check if learned approaches full-ds fixed
        best_learned_min = float("inf")
        for lm in learned_methods:
            ldf = sub[sub["method"] == lm]
            if not ldf.empty:
                best_learned_min = min(best_learned_min, ldf["val_nll"].min())

        if frac >= 0.8 and best_learned_min < full["fixed_nll"] + 0.005:
            verdict = "STRONG SUCCESS"
        elif frac >= 0.6:
            verdict = "SUCCESS"
        elif frac >= 0.4:
            verdict = "PARTIAL SUCCESS"
        else:
            verdict = "FAILURE — random/coverage performs as well as learned WRITE"

    p(f"**[{verdict}]**")
    p()
    if "SUCCESS" in verdict:
        p("Learned WRITE consistently selects more useful predictive evidence than")
        p("naive memory selection at the same budget.")
        if verdict == "STRONG SUCCESS":
            p("A small learned memory approaches full-datastore fixed kNN performance.")
    else:
        p("Learned WRITE does not consistently outperform random memory selection.")
        p("Recommendation: improve utility labels and write features before adding more gates.")
    p()

    h(2, "9. What This Does Not Test")
    for item in ["Abstraction creation or compression", "Region discovery",
                 "Semantic memory or module learning", "PPO or RL training",
                 "Transformer fine-tuning", "Branch B"]:
        p(f"- {item}")
    p()
    p("MVP 2 only tests: **can WRITE learn which raw predictive evidence to keep?**")
    p()

    h(2, "10. Next Step")
    if "SUCCESS" in verdict:
        p("Recommended: MVP 3 — WRITE + PRUNE or compressed reusable summaries.")
    else:
        p("Recommended: improve utility labels and write features, then retry.")
    p()

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",  required=True)
    parser.add_argument("--output",  required=True)
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()

    src = Path(args.source)
    out = Path(args.output)
    (out / "reports").mkdir(parents=True, exist_ok=True)

    device = get_device({"device": "cuda"})
    set_seed(42)

    print(f"\nevaluate_write_memory")
    print(f"Source: {src}  |  Output: {out}")
    print(f"Device: {device}")

    # ── full-datastore baselines ───────────────────────────────────────────────
    full = load_full_ds_baselines(src)
    print(f"\nFull-datastore baselines:")
    print(f"  GPT-only:   {full['gpt_nll']:.4f}")
    print(f"  Best fixed: {full['fixed_nll']:.4f}")
    print(f"  Q-MLP-full: {full['qmlp_nll']:.4f}")

    # ── build action grid ─────────────────────────────────────────────────────
    actions = build_default_actions()

    # ── load Q-MLP-full for transfer eval ────────────────────────────────────
    qmlp_model    = None
    qmlp_act_feats = None
    ckpt_path = src / "models" / "q_mlp_full.pt"
    if ckpt_path.exists():
        from train_q_controller import (
            load_q_model, build_action_features, build_all_action_features,
        )
        qmlp_model, action_names = load_q_model(ckpt_path, device)
        qmlp_model.eval()
        # Reconstruct action feats for the full action grid (not just subset)
        qmlp_act_feats = build_all_action_features(actions, MAX_K)
        print(f"  Q-MLP-full loaded from {ckpt_path}")
    else:
        print("  Q-MLP-full not found — skipping transfer eval")

    # ── load ct and val data once ─────────────────────────────────────────────
    print("\nLoading CT and val data ...")
    states_dir = src / "states"
    ct_data    = torch.load(states_dir / "controller_train.pt", weights_only=False)
    val_data   = torch.load(states_dir / "val.pt",             weights_only=False)
    N_ds       = len(torch.load(states_dir / "datastore.pt",
                                 weights_only=False)["y"])
    print(f"  N_ct={len(ct_data['y']):,}  N_val={len(val_data['y']):,}  N_ds={N_ds:,}")

    # ── discover neighbor files ───────────────────────────────────────────────
    nbrs_dir   = out / "neighbors"
    val_files  = sorted(nbrs_dir.glob("*_val_top*.pt"))

    if not val_files:
        print(f"\nNo val neighbor files found in {nbrs_dir}")
        print("Run build_memory_neighbors.py first.")
        return

    print(f"\nEvaluating {len(val_files)} memory × budget combinations ...")
    rows: list[dict] = []

    for val_path in val_files:
        # Corresponding CT file
        ct_path = Path(str(val_path).replace("_val_top", "_controller_train_top"))
        if not ct_path.exists():
            print(f"  Missing CT file for {val_path.name}, skipping")
            continue

        stem       = val_path.stem                    # method_B<B>_val_top64
        # extract method_B<B> part
        parts      = re.match(r"^(.+)_val_top\d+$", stem)
        if not parts:
            continue
        mem_stem   = parts.group(1)
        method, budget = parse_stem(mem_stem)

        if budget == 0:
            continue

        val_nbrs = torch.load(val_path, weights_only=False)
        ct_nbrs  = torch.load(ct_path,  weights_only=False)
        k_eff    = val_nbrs.get("k_eff", MAX_K)
        mem_frac = budget / N_ds

        print(f"\n  [{method} B={budget}]  k_eff={k_eff}", end="")

        base_row = {
            "method":      method,
            "budget":      budget,
            "n_ds":        N_ds,
            "memory_frac": mem_frac,
            "full_gpt_nll":   full["gpt_nll"],
            "full_fixed_nll": full["fixed_nll"],
            "full_qmlp_nll":  full["qmlp_nll"],
        }

        # ── A. Best fixed action ───────────────────────────────────────────────
        try:
            best_act, ct_nll = find_best_fixed_action(ct_data, ct_nbrs, actions)
            vm   = eval_action_on_val(val_data, val_nbrs, best_act)
            nll  = vm["mean_nll"]
            row  = dict(base_row)
            row.update({
                "read_policy":          "best_fixed",
                "best_fixed_action":    best_act["name"],
                "val_nll":              nll,
                "val_ppl":              ppl(nll),
                "avg_k":                vm["mean_k"],
                "retrieval_usage":      vm["retrieval_usage"],
                "delta_vs_gpt":         nll - full["gpt_nll"],
                "delta_vs_full_fixed":  nll - full["fixed_nll"],
                "delta_vs_full_qmlp":   nll - full["qmlp_nll"],
            })
            rows.append(row)
            print(f"  fixed NLL={nll:.4f}  ({best_act['name']})", end="")
        except Exception as e:
            print(f"  best_fixed FAILED: {e}", end="")

        # ── B. Q-MLP transfer eval ─────────────────────────────────────────────
        if qmlp_model is not None:
            try:
                qm = qmlp_transfer_eval(qmlp_model, actions, qmlp_act_feats,
                                         val_data, val_nbrs, device)
                nll = qm["mean_nll"]
                row = dict(base_row)
                row.update({
                    "read_policy":         "qmlp_transfer",
                    "val_nll":             nll,
                    "val_ppl":             ppl(nll),
                    "avg_k":               qm["mean_k"],
                    "retrieval_usage":     qm["retrieval_usage"],
                    "delta_vs_gpt":        nll - full["gpt_nll"],
                    "delta_vs_full_fixed": nll - full["fixed_nll"],
                    "delta_vs_full_qmlp":  nll - full["qmlp_nll"],
                })
                rows.append(row)
                print(f"  qmlp NLL={nll:.4f}", end="")
            except Exception as e:
                print(f"  qmlp FAILED: {e}", end="")

        print()

    if not rows:
        print("No results. Exiting.")
        return

    # ── save results ──────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(out / "reports" / "write_memory_results.csv", index=False)
    with open(out / "reports" / "write_memory_results.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved: reports/write_memory_results.csv  ({len(df)} rows)")

    budgets = sorted(df["budget"].unique().tolist())

    # ── plots ─────────────────────────────────────────────────────────────────
    if not args.no_plots:
        try_plots(df, budgets, out, full["fixed_nll"], full["qmlp_nll"])

    # ── report ────────────────────────────────────────────────────────────────
    report = generate_report(df, full, budgets, out)
    rpath  = out / "MVP2_WRITE_MEMORY_REPORT.md"
    with open(rpath, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved: {rpath}")

    # ── terminal summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    sub = df[df["read_policy"] == "best_fixed"].copy()
    rand_mean = sub[sub["method"].str.startswith("random_")].groupby("budget")["val_nll"].mean()
    learned   = [m for m in sub["method"].unique() if m.startswith("learned_write")]
    print("Best fixed read policy -- NLL by method (best budget):")
    for meth in sorted(sub["method"].unique()):
        m_df = sub[sub["method"] == meth]
        if m_df.empty: continue
        best_nll = m_df["val_nll"].min()
        tag = " [DIAG]" if "oracle" in meth else ""
        print(f"  {meth:<35} best NLL={best_nll:.4f}{tag}")
    print(f"\n  full_ds_fixed = {full['fixed_nll']:.4f}")
    print(f"  full_ds_qmlp  = {full['qmlp_nll']:.4f}")
    print(f"{'='*70}")
    print("Done.")


if __name__ == "__main__":
    main()

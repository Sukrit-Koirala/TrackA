"""
evaluate_predictive_states.py

Evaluates each persistent-predictive-state file (method × budget) using:
  A. Best fixed state-read action (CT-selected)
  B. Q-state-read controller (if trained)

Compares against:
  - random raw memory at same budget (from MVP2a results)
  - full-datastore fixed kNN baseline
  - full-datastore Q-MLP baseline
  - GPT-only

State-read action:
  k_states  number of states to retrieve
  tau       softmax temperature for state weights
  alpha     GPT mix weight  (P_final = alpha*P_GPT + (1-alpha)*P_local)
  beta      Laplace smoothing strength

Prediction:
  sim_i     = cosine(h_q, proto_h_i)
  weights   = softmax(sim[:k_states] / tau)
  P_state_i = (count_i(y) + beta * P_global(y)) / (total_i + beta)
  P_local   = sum_i weights_i * P_state_i(y_true)
  P_final   = alpha * P_GPT + (1-alpha) * P_local
  NLL       = -log P_final

Usage:
  python src/evaluate_predictive_states.py \\
    --source outputs_scale_sweep/scale_200k_seed42 \\
    --output outputs_mvp2b_state_write \\
    --raw_memory_csv outputs_mvp2_write_memory/reports/write_memory_results.csv
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
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

from utils import get_device, set_seed, ppl
from build_neighbors import cosine_top_k

MAX_K_STATES = 32
QUERY_CHUNK  = 256    # chunk size for top-k precomputation
DS_CHUNK     = 8192   # chunk for prototype search

K_VALS     = [1, 2, 4, 8, 16, 32]
TAU_VALS   = [0.05, 0.1, 0.2]
ALPHA_VALS = [0.25, 0.5, 0.75]
BETA_VALS  = [0.0, 1.0, 5.0, 10.0]


# ── action grid ───────────────────────────────────────────────────────────────

def build_state_action_grid(max_k_states: int = MAX_K_STATES) -> list[dict]:
    acts = [{"name": "gpt_only", "k_states": 0, "tau": 0.1,
              "alpha": 1.0, "beta": 0.0}]
    for k in K_VALS:
        if k > max_k_states:
            continue
        for tau in TAU_VALS:
            for alpha in ALPHA_VALS:
                for beta in BETA_VALS:
                    name = f"k{k}_t{tau}_a{alpha}_b{beta}"
                    acts.append({"name": name, "k_states": k, "tau": tau,
                                  "alpha": alpha, "beta": beta})
    return acts


# ── global frequency ──────────────────────────────────────────────────────────

def build_global_freq(y: torch.Tensor, vocab_size: int = 50257) -> torch.Tensor:
    counts = torch.bincount(y.long(), minlength=vocab_size).float()
    return (counts / counts.sum()).clamp(min=1e-10)


# ── top-k state similarity ────────────────────────────────────────────────────

def compute_top_k_state_sims(q_h_norm: torch.Tensor,
                               proto_h: torch.Tensor,
                               K: int, device: torch.device):
    """Returns (top_ids [N_q, K], top_sims [N_q, K]) using cosine_top_k."""
    B = len(proto_h)
    k_eff = min(K, B)
    idx, sims = cosine_top_k(q_h_norm, proto_h, k_eff,
                               QUERY_CHUNK, DS_CHUNK, device)
    if k_eff < K:
        pad_idx  = torch.zeros(len(q_h_norm), K - k_eff, dtype=torch.long)
        pad_sims = torch.zeros(len(q_h_norm), K - k_eff)
        idx  = torch.cat([idx,  pad_idx],  dim=1)
        sims = torch.cat([sims, pad_sims], dim=1)
    return idx, sims  # both [N_q, K]


# ── precompute per-query token lookup ─────────────────────────────────────────

def precompute_lookup(q_data: dict,
                       top_ids: torch.Tensor,   # [N_q, MAX_K]
                       top_sims: torch.Tensor,  # [N_q, MAX_K]
                       states: dict,
                       P_global: torch.Tensor,
                       chunk_size: int = 1024) -> dict:
    """
    For each query and each of its MAX_K nearest states, precompute:
      raw_count[q, s]  = top_k_token_counts[state_id[q,s]] where token == y_true[q]
      sel_total[q, s]  = total_counts[state_id[q,s]]
      p_global_y[q]    = P_global[y_true[q]]

    Stored in CPU tensors. Per-action eval then slices [:, :k_states] cheaply.
    """
    N_q    = len(q_data["y"])
    K      = top_ids.shape[1]
    y_true = q_data["y"]                          # [N_q]
    tok_ids  = states["top_k_token_ids"]           # [B, TOP_K]
    tok_cnts = states["top_k_token_counts"]        # [B, TOP_K]
    sel_total_all = states["total_counts"]         # [B]
    TOP_K  = tok_ids.shape[1]

    raw_count = torch.zeros(N_q, K)
    sel_total = torch.zeros(N_q, K)

    for start in range(0, N_q, chunk_size):
        end     = min(start + chunk_size, N_q)
        ids_c   = top_ids[start:end]               # [C, K]
        y_c     = y_true[start:end]                # [C]
        C       = end - start

        sel_tids = tok_ids[ids_c]                  # [C, K, TOP_K]
        sel_tcnts = tok_cnts[ids_c]                # [C, K, TOP_K]

        y_exp = y_c.view(C, 1, 1).expand(C, K, TOP_K)
        match = (sel_tids.long() == y_exp.long())  # [C, K, TOP_K]

        raw_count[start:end] = (sel_tcnts * match.float()).sum(-1)   # [C, K]
        sel_total[start:end] = sel_total_all[ids_c]                  # [C, K]

    p_global_y = P_global[y_true.long()]          # [N_q]

    return {
        "raw_count":  raw_count,   # [N_q, K]
        "sel_total":  sel_total,   # [N_q, K]
        "top_sims":   top_sims,    # [N_q, K]
        "p_global_y": p_global_y,  # [N_q]
        "p_gpt":      q_data["p_gpt_true"].float(),
        "nll_gpt":    q_data["nll_gpt"].float(),
        "y_true":     y_true,
    }


# ── fast per-action eval ──────────────────────────────────────────────────────

def eval_action(precomp: dict, action: dict) -> dict:
    k    = action["k_states"]
    tau  = action["tau"]
    alpha= action["alpha"]
    beta = action["beta"]
    p_gpt = precomp["p_gpt"]
    nll_gpt = precomp["nll_gpt"]

    if k == 0:
        nll = nll_gpt
        return {"mean_nll": float(nll.mean()), "per_example_nll": nll,
                "retrieval_usage": 0.0, "mean_k": 0.0}

    sims_k  = precomp["top_sims"][:, :k]      # [N, k]
    raw_k   = precomp["raw_count"][:, :k]     # [N, k]
    total_k = precomp["sel_total"][:, :k]     # [N, k]
    p_gy    = precomp["p_global_y"]           # [N]

    weights = F.softmax(sims_k / (tau + 1e-8), dim=-1)   # [N, k]
    p_state = (raw_k + beta * p_gy.unsqueeze(1)) / (total_k + beta + 1e-10)
    p_local = (weights * p_state).sum(-1).clamp(min=1e-10)

    p_final = (alpha * p_gpt + (1 - alpha) * p_local).clamp(min=1e-10)
    nll     = -p_final.log()

    return {
        "mean_nll":        float(nll.mean()),
        "per_example_nll": nll,
        "retrieval_usage": float((p_local > 1e-9).float().mean()),
        "mean_k":          float(k),
        "p_local":         p_local,
    }


def find_best_fixed_action(ct_precomp: dict, actions: list[dict]) -> dict:
    best_nll, best_act = float("inf"), actions[0]
    for act in actions:
        m = eval_action(ct_precomp, act)
        if m["mean_nll"] < best_nll:
            best_nll, best_act = m["mean_nll"], act
    return best_act, best_nll


# ── query routing diagnostics ─────────────────────────────────────────────────

def query_routing_stats(precomp: dict, top_ids: torch.Tensor,
                         states: dict, k: int = 4) -> dict:
    """Compute true-token-in-selected-state rate and similarity quantiles."""
    N = len(precomp["y_true"])
    k_eff = min(k, top_ids.shape[1])

    # True token in selected states: raw_count > 0 for any of top-k states
    any_match = (precomp["raw_count"][:, :k_eff] > 0).any(dim=-1)  # [N] bool
    hit_rate  = float(any_match.float().mean())

    # Nearest state similarity quantiles
    sims = precomp["top_sims"][:, 0]  # nearest state sim [N]
    qs   = [0.1, 0.25, 0.5, 0.75, 0.9]
    sim_quantiles = {f"p{int(q*100)}": float(torch.quantile(sims, q)) for q in qs}

    return {"true_token_hit_rate_k": hit_rate, "k_for_hit": k_eff,
            **sim_quantiles}


# ── main evaluation ───────────────────────────────────────────────────────────

def load_raw_memory_baselines(csv_path: str | None, budget: int) -> dict:
    out = {"random_best_fixed_nll": float("nan")}
    if csv_path is None or not Path(csv_path).exists():
        return out
    try:
        df = pd.read_csv(csv_path)
        r  = df[(df["method"].str.startswith("random_")) &
                (df["budget"] == budget) &
                (df["read_policy"] == "best_fixed")]
        if not r.empty:
            out["random_best_fixed_nll"] = float(r["val_nll"].mean())
    except Exception:
        pass
    return out


def load_full_ds_baselines(src: Path) -> dict:
    out = {"gpt_nll": 2.5533, "fixed_nll": 2.3873, "qmlp_nll": 2.2977}
    bl = src / "reports" / "fixed_baselines.json"
    if bl.exists():
        with open(bl) as f:
            d = json.load(f)
        out["gpt_nll"]   = d.get("gpt_only_val_nll",           out["gpt_nll"])
        out["fixed_nll"] = d.get("best_train_selected_val_nll", out["fixed_nll"])
    qm = src / "reports" / "q_controller_metrics.json"
    if qm.exists():
        with open(qm) as f:
            d = json.load(f)
        if "Q-MLP-full" in d:
            out["qmlp_nll"] = d["Q-MLP-full"].get("mean_nll", out["qmlp_nll"])
    return out


def parse_method_budget(stem: str):
    m = re.match(r"^(.+)_B(\d+)$", stem)
    if m:
        return m.group(1), int(m.group(2))
    return stem, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",           required=True)
    parser.add_argument("--output",           required=True)
    parser.add_argument("--raw_memory_csv",   default=None)
    parser.add_argument("--q_model_dir",      default=None,
                        help="Directory with trained Q-state-read models")
    parser.add_argument("--force",            action="store_true")
    args = parser.parse_args()

    src     = Path(args.source)
    out     = Path(args.output)
    (out / "reports").mkdir(parents=True, exist_ok=True)

    device = get_device({"device": "cuda"})
    set_seed(42)

    print(f"\nevaluate_predictive_states")
    print(f"Source: {src}  |  Device: {device}")

    # ── load data ─────────────────────────────────────────────────────────────
    print("\nLoading CT and val ...")
    ct_data  = torch.load(src / "states" / "controller_train.pt", weights_only=False)
    val_data = torch.load(src / "states" / "val.pt",              weights_only=False)
    ds_data  = torch.load(src / "states" / "datastore.pt",        weights_only=False)
    N_ct, N_val = len(ct_data["y"]), len(val_data["y"])
    vocab_size = 50257
    print(f"  N_ct={N_ct:,}  N_val={N_val:,}")

    P_global = build_global_freq(ds_data["y"], vocab_size)
    full_bl  = load_full_ds_baselines(src)
    print(f"  Baselines: GPT={full_bl['gpt_nll']:.4f}  "
          f"Fixed={full_bl['fixed_nll']:.4f}  Q-MLP={full_bl['qmlp_nll']:.4f}")

    def norm_h(data):
        h = data["h"].float()
        return h / (h.norm(dim=-1, keepdim=True) + 1e-8)

    ct_h_norm  = norm_h(ct_data)
    val_h_norm = norm_h(val_data)

    # ── discover state files ──────────────────────────────────────────────────
    states_dir = out / "states"
    state_files = sorted(states_dir.glob("*.pt"))
    if not state_files:
        print(f"\nNo state files in {states_dir}. Run build_predictive_states.py first.")
        return
    print(f"\nEvaluating {len(state_files)} state files ...")

    actions = build_state_action_grid(MAX_K_STATES)
    rows = []

    for sf in state_files:
        method, budget = parse_method_budget(sf.stem)
        if budget == 0:
            continue

        print(f"\n  [{sf.stem}]  budget={budget}")
        states = torch.load(sf, weights_only=False)
        proto  = states["prototype_h"]       # [B, D] normalised
        B      = len(proto)

        raw_bl = load_raw_memory_baselines(args.raw_memory_csv, budget)

        # State quality stats
        ent    = float(states["state_entropy"].mean())
        purity = float(states["state_purity"].mean())
        mean_count = float(states["total_counts"].mean())

        # Compute top-K state sims for CT and val
        print(f"    Computing CT top-{MAX_K_STATES} state sims ...")
        ct_idx, ct_sims = compute_top_k_state_sims(ct_h_norm, proto, MAX_K_STATES, device)
        print(f"    Computing val top-{MAX_K_STATES} state sims ...")
        val_idx, val_sims = compute_top_k_state_sims(val_h_norm, proto, MAX_K_STATES, device)

        # Precompute token lookups
        print(f"    Precomputing CT token lookups ...")
        ct_precomp  = precompute_lookup(ct_data,  ct_idx,  ct_sims,  states, P_global)
        print(f"    Precomputing val token lookups ...")
        val_precomp = precompute_lookup(val_data, val_idx, val_sims, states, P_global)

        # Best fixed action on CT
        best_act, ct_nll = find_best_fixed_action(ct_precomp, actions)
        print(f"    Best CT action: {best_act['name']}  CT NLL={ct_nll:.4f}")

        # Eval best fixed on val
        vm  = eval_action(val_precomp, best_act)
        rr  = query_routing_stats(val_precomp, val_idx, states, k=4)

        nll = vm["mean_nll"]
        base_row = dict(
            method=method, budget=budget,
            read_policy="best_fixed",
            best_action=best_act["name"],
            val_nll=nll, val_ppl=ppl(nll),
            avg_k_states=vm["mean_k"],
            retrieval_usage=vm["retrieval_usage"],
            mean_state_entropy=ent,
            median_state_purity=purity,
            mean_state_count=mean_count,
            true_token_hit_rate=rr["true_token_hit_rate_k"],
            nearest_state_sim_p50=rr.get("p50", float("nan")),
            delta_vs_gpt=nll - full_bl["gpt_nll"],
            delta_vs_full_fixed=nll - full_bl["fixed_nll"],
            delta_vs_full_qmlp=nll - full_bl["qmlp_nll"],
            delta_vs_raw_random=nll - raw_bl["random_best_fixed_nll"],
            full_gpt_nll=full_bl["gpt_nll"],
            full_fixed_nll=full_bl["fixed_nll"],
            full_qmlp_nll=full_bl["qmlp_nll"],
            raw_random_nll=raw_bl["random_best_fixed_nll"],
        )
        rows.append(base_row)
        print(f"    val NLL={nll:.4f}  hit_rate={rr['true_token_hit_rate_k']:.3f}  "
              f"vs_random={nll-raw_bl['random_best_fixed_nll']:+.4f}")

        # Q-controller eval (if model exists)
        if args.q_model_dir:
            q_path = Path(args.q_model_dir) / f"q_state_{method}_B{budget}.pt"
            if q_path.exists():
                try:
                    from train_state_read_controller import (
                        apply_q_state_controller, build_state_obs_features,
                        build_state_action_features_all,
                    )
                    q_ckpt = torch.load(q_path, weights_only=False)
                    q_actions = q_ckpt["actions"]
                    act_feats = build_state_action_features_all(q_actions, MAX_K_STATES)
                    obs = build_state_obs_features(val_data, val_precomp, states)
                    chosen = apply_q_state_controller(q_ckpt["model"], obs, act_feats, device)
                    # Eval chosen
                    from train_q_controller import eval_chosen_on_val as _ecv
                    # Build per-query val_precomp action
                    nlls = []
                    for qi, ai in enumerate(chosen.tolist()):
                        act = q_actions[ai]
                        m2 = eval_action({k: v[qi:qi+1] for k, v in val_precomp.items()
                                          if isinstance(v, torch.Tensor)}, act)
                        nlls.append(float(m2["per_example_nll"].mean()))
                    q_nll = float(np.mean(nlls))
                    qrow = dict(base_row, read_policy="q_state_controller",
                                val_nll=q_nll, val_ppl=ppl(q_nll),
                                delta_vs_gpt=q_nll - full_bl["gpt_nll"],
                                delta_vs_full_fixed=q_nll - full_bl["fixed_nll"],
                                delta_vs_full_qmlp=q_nll - full_bl["qmlp_nll"],
                                delta_vs_raw_random=q_nll - raw_bl["random_best_fixed_nll"])
                    rows.append(qrow)
                    print(f"    Q-controller val NLL={q_nll:.4f}")
                except Exception as e:
                    print(f"    Q-controller eval failed: {e}")

    # ── save CSV + JSON ───────────────────────────────────────────────────────
    if not rows:
        print("No results.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(out / "reports" / "state_write_results.csv", index=False)
    with open(out / "reports" / "state_write_results.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved: reports/state_write_results.csv  ({len(df)} rows)")

    # ── plots ─────────────────────────────────────────────────────────────────
    try_plots(df, out, full_bl)

    # ── generate report ───────────────────────────────────────────────────────
    report = generate_report(df, full_bl)
    rpath  = out / "MVP2B_STATE_WRITE_REPORT.md"
    with open(rpath, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved: {rpath}")

    # ── terminal summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    sub = df[df["read_policy"] == "best_fixed"]
    for meth in sorted(sub["method"].unique()):
        mdf = sub[sub["method"] == meth]
        best = mdf.loc[mdf["val_nll"].idxmin()]
        tag  = " [DIAG]" if "oracle" in meth else ""
        print(f"  {meth:<35} best NLL={best['val_nll']:.4f}  "
              f"B={int(best['budget']):>6}  "
              f"vs_random={best['delta_vs_raw_random']:+.4f}{tag}")
    print(f"\n  full_ds_fixed = {full_bl['fixed_nll']:.4f}")
    print(f"  full_ds_qmlp  = {full_bl['qmlp_nll']:.4f}")
    print(f"{'='*70}")


def try_plots(df: pd.DataFrame, out: Path, full_bl: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_dir = out / "plots"
        plot_dir.mkdir(exist_ok=True)
        sub = df[df["read_policy"] == "best_fixed"].copy()
        methods = sorted(sub["method"].unique())
        budgets = sorted(sub["budget"].unique())
        colors  = plt.cm.tab10(np.linspace(0, 1, max(len(methods), 1)))

        # NLL vs budget
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, meth in enumerate(methods):
            mdf = sub[sub["method"] == meth].sort_values("budget")
            if mdf.empty: continue
            ax.plot(mdf["budget"], mdf["val_nll"], "-o", label=meth,
                    color=colors[i], linewidth=1.5, markersize=4)
        ax.axhline(full_bl["fixed_nll"], color="black",  linestyle=":", label="full_ds_fixed")
        ax.axhline(full_bl["qmlp_nll"],  color="purple", linestyle=":", label="full_ds_Q-MLP")
        ax.set_xlabel("State budget (B)"); ax.set_ylabel("Val NLL")
        ax.set_title("Val NLL vs state budget"); ax.set_xscale("log")
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
        plt.tight_layout()
        plt.savefig(plot_dir / "nll_vs_state_budget.png", dpi=150); plt.close()

        # Delta vs raw random
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, meth in enumerate(methods):
            mdf = sub[sub["method"] == meth].sort_values("budget")
            if mdf.empty: continue
            d = mdf["delta_vs_raw_random"].values
            ax.plot(mdf["budget"], d, "-o", label=meth,
                    color=colors[i], linewidth=1.5, markersize=4)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("State budget (B)"); ax.set_ylabel("Delta NLL vs random (negative = better)")
        ax.set_title("State-forming WRITE vs random raw memory"); ax.set_xscale("log")
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
        plt.tight_layout()
        plt.savefig(plot_dir / "delta_vs_random_same_budget.png", dpi=150); plt.close()

        # Hit rate vs NLL
        if "true_token_hit_rate" in sub.columns:
            fig, ax = plt.subplots(figsize=(6, 5))
            for i, meth in enumerate(methods):
                mdf = sub[sub["method"] == meth]
                ax.scatter(mdf["true_token_hit_rate"], mdf["val_nll"],
                           label=meth, color=colors[i], s=30)
            ax.set_xlabel("True-token-in-selected-state rate")
            ax.set_ylabel("Val NLL")
            ax.set_title("Hit rate vs NLL")
            ax.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(plot_dir / "true_token_hit_rate_vs_nll.png", dpi=150); plt.close()

        print(f"  Plots -> {plot_dir}")
    except Exception as e:
        print(f"  Plots skipped: {e}")


def generate_report(df: pd.DataFrame, full_bl: dict) -> str:
    lines = []
    def h(n, t): lines.append("#"*n + " " + t); lines.append("")
    def p(t=""): lines.append(t)

    h(1, "MVP 2b: State-Forming WRITE — Report")
    p(f"Full-datastore baselines: GPT={full_bl['gpt_nll']:.4f}  "
      f"Fixed={full_bl['fixed_nll']:.4f}  Q-MLP={full_bl['qmlp_nll']:.4f}")
    p()

    h(2, "1. Why MVP 2a Failed")
    p("Naive WRITE as independent row selection failed because:")
    p("- Optimising individual entry importance destroyed coverage")
    p("- Oracle individual utility still lost to random memory selection")
    p("- High-utility entries cluster in popular embedding regions, leaving other queries unserved")
    p("- Token rarity / high GPT-loss methods select outlier entries that never match val queries")
    p()

    h(2, "2. MVP 2b Goal")
    p("Test whether WRITE can build **persistent predictive states** that aggregate")
    p("many raw examples into reusable prediction-bearing objects.")
    p()
    p("These are not manually defined abstractions or semantic regions.")
    p("They are gate-built predictive states. Interpretation follows evidence.")
    p()
    p("Key comparison: B persistent states vs B random raw memory entries (same budget).")
    p()

    h(2, "3. Methods")
    method_descs = {
        "minibatch_kmeans":   "MiniBatchKMeans on datastore hidden states (scalable default)",
        "query_kmeans":       "Cluster CT query h; assign DS entries to nearest centroid",
        "balanced_kmeans":    "minibatch_kmeans with capped assignment per state",
        "streaming_write":    "Online WRITE/UPDATE: create state or update nearest above threshold",
        "utility_weighted":   "minibatch_kmeans with NLL-weighted token counts",
        "random_partition":   "Random assignment baseline (sanity check)",
    }
    for meth, desc in method_descs.items():
        p(f"- **{meth}**: {desc}")
    p()

    h(2, "4. Main Results (best-fixed read policy)")
    sub = df[df["read_policy"] == "best_fixed"]
    methods = sorted(sub["method"].unique())
    budgets = sorted(sub["budget"].unique())

    p(f"Reference: full_ds_fixed={full_bl['fixed_nll']:.4f}  "
      f"full_ds_Q-MLP={full_bl['qmlp_nll']:.4f}  GPT={full_bl['gpt_nll']:.4f}")
    p()
    p("Val NLL (lower is better), delta_vs_random in parentheses:")
    p()
    header = "| method | " + " | ".join(str(b) for b in budgets) + " |"
    sep    = "|--------|" + "|".join(["------"] * len(budgets)) + "|"
    p(header); p(sep)
    for meth in methods:
        row = f"| {meth:<32} |"
        mdf = sub[sub["method"] == meth]
        for b in budgets:
            r2 = mdf[mdf["budget"] == b]
            if r2.empty:
                row += "  —   |"
            else:
                nll  = r2["val_nll"].values[0]
                dvr  = r2["delta_vs_raw_random"].values[0]
                sign = "+" if dvr >= 0 else ""
                row += f" {nll:.4f}({sign}{dvr:.3f}) |"
        p(row)
    p()
    p("Numbers in parentheses: delta vs random raw memory at same budget.")
    p("Negative = state method beats random. Positive = random wins.")
    p()

    h(2, "5. State Quality Diagnostics")
    quality_cols = ["method", "budget", "mean_state_entropy", "median_state_purity",
                    "mean_state_count"]
    qdf = sub[[c for c in quality_cols if c in sub.columns]].copy()
    if not qdf.empty:
        p("| method | budget | entropy | purity | mean_count |")
        p("|--------|--------|---------|--------|------------|")
        for _, r in qdf.sort_values(["method", "budget"]).iterrows():
            ent = f"{r.get('mean_state_entropy', float('nan')):.3f}"
            pur = f"{r.get('median_state_purity', float('nan')):.3f}"
            cnt = f"{r.get('mean_state_count', float('nan')):.1f}"
            p(f"| {r['method']:<32} | {int(r['budget']):>6} | {ent} | {pur} | {cnt} |")
        p()

    h(2, "6. Query Routing Diagnostics")
    p("True-token-in-selected-state rate (k=4): fraction of val queries where")
    p("at least one of the top-4 nearest states contains y_true.")
    p()
    if "true_token_hit_rate" in sub.columns:
        p("| method | budget | hit_rate_k4 | nearest_sim_p50 |")
        p("|--------|--------|-------------|-----------------|")
        for _, r in sub.sort_values(["method", "budget"]).iterrows():
            hr  = f"{r.get('true_token_hit_rate', float('nan')):.3f}"
            sim = f"{r.get('nearest_state_sim_p50', float('nan')):.3f}"
            p(f"| {r['method']:<32} | {int(r['budget']):>6} | {hr} | {sim} |")
        p()

    h(2, "7. Verdict")
    # Determine success
    beats_random = 0
    total_comparisons = 0
    for _, r in sub.iterrows():
        if "random_partition" not in r["method"] and not math.isnan(r.get("delta_vs_raw_random", float("nan"))):
            total_comparisons += 1
            if r["delta_vs_raw_random"] < 0:
                beats_random += 1

    best_nll = sub["val_nll"].min() if not sub.empty else float("nan")
    frac = beats_random / total_comparisons if total_comparisons > 0 else 0

    if frac >= 0.7 and best_nll < full_bl["fixed_nll"] + 0.01:
        verdict = "STRONG SUCCESS"
    elif frac >= 0.5:
        verdict = "SUCCESS"
    elif frac >= 0.3:
        verdict = "PARTIAL SUCCESS"
    else:
        verdict = "FAILURE — random raw memory still wins"

    p(f"**[{verdict}]**")
    p()
    p(f"State-forming methods beat random at {beats_random}/{total_comparisons} "
      f"(method × budget) comparisons ({frac:.0%}).")
    p()
    if "SUCCESS" in verdict:
        p("State-forming WRITE successfully aggregates raw examples into persistent")
        p("predictive states that preserve coverage while improving predictive quality.")
    else:
        p("State-forming WRITE does not consistently outperform random raw memory.")
        p("States may be too high-entropy, or the token distributions too noisy.")
    p()

    h(2, "8. Next Recommendation")
    if "SUCCESS" in verdict:
        p("Recommended: add online PRUNE or adaptive state merging to improve state quality.")
        p("Also: train Q-state-read controller to select optimal k_states / tau / alpha per query.")
    else:
        p("Recommended: rethink state objective.")
        p("Options:")
        p("- Supervised query-based optimisation (partition around query performance, not H geometry)")
        p("- Larger budgets (50k+ states)")
        p("- Discriminative state formation: penalise entropy, reward purity")
    p()

    h(2, "9. What This Does Not Test")
    for item in ["Semantic labeling or named abstractions", "Manual region definitions",
                 "Hierarchy or module learning", "PPO or RL training",
                 "Transformer fine-tuning", "Branch B"]:
        p(f"- {item}")
    p()
    p("MVP 2b only tests: **can WRITE form persistent predictive states that beat raw memory?**")
    p()

    return "\n".join(lines)


if __name__ == "__main__":
    main()

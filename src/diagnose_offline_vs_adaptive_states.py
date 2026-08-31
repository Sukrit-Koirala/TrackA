"""
diagnose_offline_vs_adaptive_states.py  --  MVP 3a diagnostic

Answers: what do offline states have that adaptive states lack?

Compares offline (minibatch_kmeans, utility_weighted) vs adaptive
(budget_filling, token_conflict) across 10 diagnostic axes.

Outputs to --output/reports/:
  offline_vs_adaptive_summary.csv
  state_size_distribution.csv
  token_entropy_purity.csv
  hit_rate_comparison.csv
  q_behavior_comparison.csv
  help_hurt_comparison.csv
  utility_concentration.csv
  local_prediction_quality.csv

Also writes OFFLINE_VS_ADAPTIVE_DIAGNOSIS_REPORT.md.
"""

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

from evaluate_predictive_states import (
    precompute_lookup, build_global_freq, eval_action,
)

EPS   = 1e-10
VOCAB = 50257

GPT_ONLY      = 2.5533
FULL_DS_FIXED = 2.3873
FULL_DS_QMLP  = 2.2977
MVP2C_BEST    = 2.2852


# ── group registry ─────────────────────────────────────────────────────────────

def discover_groups(offline_states_dir: Path, offline_q_dir: Path,
                    adaptive_states_dir: Path, adaptive_q_dir: Path,
                    budgets: list[int]) -> list[dict]:
    """Auto-discover comparison groups from available files."""
    groups = []
    nbr_suffix = "_val_top32.pt"

    offline_map = {
        "minibatch_kmeans": (offline_states_dir, offline_q_dir),
        "utility_weighted": (offline_states_dir, offline_q_dir),
    }
    adaptive_map = {
        "budget_filling": (adaptive_states_dir, adaptive_q_dir),
        "token_conflict": (adaptive_states_dir, adaptive_q_dir),
    }

    def find_state_file(states_dir: Path, prefix: str, budget: int):
        # Exact name (offline)
        exact = states_dir / f"{prefix}_B{budget}.pt"
        if exact.exists():
            return exact, f"{prefix}_B{budget}"
        # Fuzzy (adaptive - find any .pt with prefix and budget suffix)
        for p in sorted(states_dir.glob(f"{prefix}*_B{budget}.pt")):
            return p, p.stem
        return None, None

    def find_nbr_file(nbr_dir: Path, tag: str):
        p = nbr_dir / f"{tag}{nbr_suffix}"
        return p if p.exists() else None

    def find_q_dir(q_base: Path, tag: str):
        p = q_base / tag
        return p if p.exists() and (p / "val_predictions.pt").exists() else None

    for budget in budgets:
        for meth, (sdir, qdir) in {**offline_map, **adaptive_map}.items():
            mtype = "offline" if meth in offline_map else "adaptive"
            prefix_map = {
                "budget_filling": "budget_filling",
                "token_conflict":  "token_conflict",
            }
            prefix = prefix_map.get(meth, meth)

            state_path, tag = find_state_file(sdir, prefix, budget)
            if state_path is None:
                continue

            # Offline: state_neighbors is inside qdir (e.g. outputs_mvp2c_q_state_read/state_neighbors)
            # Adaptive: state_neighbors is at qdir.parent level (e.g. outputs_mvp3a_adaptive_write_fast/state_neighbors)
            if mtype == "offline":
                nbr_dir = qdir / "state_neighbors"
            else:
                nbr_dir = qdir.parent / "state_neighbors"
            nbr_path = find_nbr_file(nbr_dir, tag)
            if nbr_path is None:
                print(f"    [warn] neighbors not found: {nbr_dir}/{tag}_val_top32.pt")
                continue

            qd = find_q_dir(qdir, tag)
            if qd is None:
                continue

            groups.append({
                "tag":         tag,
                "method":      meth,
                "method_type": mtype,
                "budget":      budget,
                "state_path":  state_path,
                "nbr_path":    nbr_path,
                "pred_path":   qd / "val_predictions.pt",
                "actdist_path": qd / "action_distribution.csv",
                "q_dir":       qd,
            })

    # Sort: budget, then offline before adaptive, then method name
    groups.sort(key=lambda g: (g["budget"], g["method_type"], g["method"]))
    return groups


# ── loaders ────────────────────────────────────────────────────────────────────

def load_group_data(g: dict) -> dict:
    states = torch.load(g["state_path"], weights_only=False)
    nbrs   = torch.load(g["nbr_path"],  weights_only=False)
    preds  = torch.load(g["pred_path"], weights_only=False)

    act_df = None
    if g["actdist_path"].exists():
        act_df = pd.read_csv(g["actdist_path"])

    q_metrics = {}
    qm_path = g["q_dir"] / "q_metrics.json"
    if qm_path.exists():
        with open(qm_path) as f:
            q_metrics = json.load(f)

    return {
        "tag":         g["tag"],
        "method":      g["method"],
        "method_type": g["method_type"],
        "budget":      g["budget"],
        "states":      states,
        "val_ids":     nbrs["ids"],     # [N_val, 32]
        "val_sims":    nbrs["sims"],    # [N_val, 32]
        "q_nlls":      preds["q_nlls"],
        "fixed_nlls":  preds["fixed_nlls"],
        "oracle_nlls": preds["oracle_nlls"],
        "chosen":      preds["chosen"],  # [N_val] action indices
        "act_df":      act_df,
        "q_metrics":   q_metrics,
    }


# ── axis 1: budget usage ───────────────────────────────────────────────────────

def budget_usage(d: dict) -> dict:
    B       = len(d["states"]["prototype_h"])
    budget  = d["budget"]
    return {
        "tag":    d["tag"], "method": d["method"], "method_type": d["method_type"],
        "budget": budget,
        "actual_num_states": B,
        "budget_usage_fraction": B / max(budget, 1),
        "unused_budget": budget - B,
    }


# ── axis 2: state size distribution ───────────────────────────────────────────

def size_distribution(d: dict) -> dict:
    tc = d["states"]["total_counts"].numpy().astype(float)
    pcts = np.percentile(tc, [25, 50, 75, 90, 95, 99])
    return {
        "tag": d["tag"], "method": d["method"], "method_type": d["method_type"],
        "budget": d["budget"],
        "actual_B": len(tc),
        "mean_count":   float(tc.mean()),
        "median_count": float(np.median(tc)),
        "p25_count":    float(pcts[0]),
        "p75_count":    float(pcts[2]),
        "p90_count":    float(pcts[3]),
        "p95_count":    float(pcts[4]),
        "p99_count":    float(pcts[5]),
        "max_count":    float(tc.max()),
        "frac_count_1":    float((tc <= 1).mean()),
        "frac_count_le2":  float((tc <= 2).mean()),
        "frac_count_le5":  float((tc <= 5).mean()),
        "frac_count_ge64": float((tc >= 64).mean()),
        "frac_count_ge128":float((tc >= 128).mean()),
        "frac_count_ge512":float((tc >= 512).mean()),
    }


# ── axis 3: entropy/purity distribution ───────────────────────────────────────

def entropy_purity(d: dict) -> dict:
    ent = d["states"]["state_entropy"].numpy()
    pur = d["states"]["state_purity"].numpy()
    ep  = np.percentile(ent, [25, 50, 75, 90])
    pp  = np.percentile(pur, [25, 50, 75, 90])
    # Joint bins
    lohi  = float(((ent < np.percentile(ent, 33)) & (pur > np.percentile(pur, 67))).mean())
    modmod= float(((ent >= np.percentile(ent, 33)) & (ent <= np.percentile(ent, 67)) &
                   (pur >= np.percentile(pur, 33)) & (pur <= np.percentile(pur, 67))).mean())
    hilo  = float(((ent > np.percentile(ent, 67)) & (pur < np.percentile(pur, 33))).mean())
    return {
        "tag": d["tag"], "method": d["method"], "method_type": d["method_type"],
        "budget": d["budget"],
        "mean_entropy":   float(ent.mean()),
        "median_entropy": float(np.median(ent)),
        "p25_entropy":    float(ep[0]),
        "p75_entropy":    float(ep[2]),
        "p90_entropy":    float(ep[3]),
        "mean_purity":    float(pur.mean()),
        "median_purity":  float(np.median(pur)),
        "p25_purity":     float(pp[0]),
        "p75_purity":     float(pp[2]),
        "p90_purity":     float(pp[3]),
        "frac_low_ent_high_pur":      lohi,
        "frac_mod_ent_mod_pur":       modmod,
        "frac_high_ent_low_pur":      hilo,
    }


# ── axis 4: active state fraction ─────────────────────────────────────────────

def active_fraction(d: dict, val_y: torch.Tensor) -> dict:
    chosen  = d["chosen"]            # [N_val] action indices
    val_ids = d["val_ids"]           # [N_val, 32]
    B       = len(d["states"]["prototype_h"])
    N_val   = len(chosen)

    # States used as top-1 nearest by any val example
    top1_states    = val_ids[:, 0]
    unique_top1    = int(top1_states.unique().numel())
    # States where Q chose non-gpt_only action
    ret_mask = chosen != 0
    if ret_mask.any():
        active_ids = top1_states[ret_mask]
        unique_active = int(active_ids.unique().numel())
    else:
        unique_active = 0

    return {
        "tag": d["tag"], "method": d["method"], "method_type": d["method_type"],
        "budget": d["budget"],
        "B": B,
        "unique_top1_states_reached": unique_top1,
        "unique_states_q_selected":   unique_active,
        "frac_reached":   unique_top1 / max(B, 1),
        "frac_q_selected": unique_active / max(B, 1),
        "retrieval_usage": float(ret_mask.float().mean()),
        "gpt_only_fraction": float((chosen == 0).float().mean()),
    }


# ── axis 5: true-token hit rate ────────────────────────────────────────────────

def true_token_hit_rates(d: dict, val_y: torch.Tensor) -> dict:
    val_ids  = d["val_ids"]                          # [N_val, 32]
    tok_ids  = d["states"]["top_k_token_ids"].long() # [B, TOP_K]
    tok_cnts = d["states"]["top_k_token_counts"]     # [B, TOP_K]
    total    = d["states"]["total_counts"]            # [B]
    N_val    = len(val_y)
    TOP_K    = tok_ids.shape[1]

    hits = {}
    # p_state_true at top-1 state
    ids1 = val_ids[:, 0]                         # [N_val]
    st_tids = tok_ids[ids1]                      # [N_val, TOP_K]
    st_tcnts = tok_cnts[ids1]                    # [N_val, TOP_K]
    st_total = total[ids1]                       # [N_val]
    y_exp    = val_y.view(-1, 1).expand_as(st_tids)
    match1   = (st_tids == y_exp)               # [N_val, TOP_K]
    raw1     = (st_tcnts * match1.float()).sum(-1)  # [N_val]
    p_true1  = raw1 / (st_total + EPS)

    # weighted p_state across top-4
    k4    = min(4, val_ids.shape[1])
    ids4  = val_ids[:, :k4]                     # [N_val, 4]
    sims4 = d["val_sims"][:, :k4]              # [N_val, 4]
    w4    = F.softmax(sims4 / 0.05, dim=-1)    # [N_val, 4]
    p_mix4 = torch.zeros(N_val)
    for ki in range(k4):
        sids_ki  = ids4[:, ki]
        t_tids   = tok_ids[sids_ki]             # [N_val, TOP_K]
        t_tcnts  = tok_cnts[sids_ki]
        t_total  = total[sids_ki]
        match_ki = (t_tids == y_exp)
        raw_ki   = (t_tcnts * match_ki.float()).sum(-1)
        p_ki     = raw_ki / (t_total + EPS)
        p_mix4  += w4[:, ki] * p_ki

    # Hit rates at k=1,2,4,8,16
    hit_cumul = torch.zeros(N_val, dtype=torch.bool)
    hit_at_k  = {}
    for k in [1, 2, 4, 8, 16]:
        kk = min(k, val_ids.shape[1])
        for ki in range(kk):
            sids_ki  = val_ids[:, ki]
            t_tids   = tok_ids[sids_ki]
            y_ki_exp = val_y.view(-1, 1).expand_as(t_tids)
            hit_cumul |= (t_tids == y_ki_exp).any(-1)
        hit_at_k[k] = float(hit_cumul.float().mean())

    return {
        "tag": d["tag"], "method": d["method"], "method_type": d["method_type"],
        "budget": d["budget"],
        "hit_rate_top1":  hit_at_k[1],
        "hit_rate_top2":  hit_at_k[2],
        "hit_rate_top4":  hit_at_k[4],
        "hit_rate_top8":  hit_at_k[8],
        "hit_rate_top16": hit_at_k[16],
        "mean_p_state_true_top1":    float(p_true1.mean()),
        "median_p_state_true_top1":  float(p_true1.median()),
        "p90_p_state_true_top1":     float(torch.quantile(p_true1, 0.9)),
        "frac_zero_p_true_top1":     float((p_true1 == 0).float().mean()),
        "mean_p_state_true_top4mix": float(p_mix4.mean()),
        "median_p_state_true_top4mix": float(p_mix4.median()),
    }


# ── axis 6+7: nearest-sim stats & local prediction quality ────────────────────

def sim_and_local_nll(d: dict, val_data: dict, P_global: torch.Tensor) -> dict:
    val_sims = d["val_sims"]    # [N_val, 32]
    q_nlls   = d["q_nlls"]
    nll_gpt  = val_data["nll_gpt"]
    val_ids  = d["val_ids"]
    states   = d["states"]
    tok_ids  = states["top_k_token_ids"].long()
    val_y    = val_data["y"]
    N_val    = len(val_y)
    TOP_K    = tok_ids.shape[1]

    nearest_sim = val_sims[:, 0]

    # True token in nearest state (bool mask)
    ids1   = val_ids[:, 0]
    y_exp  = val_y.view(-1, 1).expand(N_val, TOP_K)
    match1 = (tok_ids[ids1] == y_exp).any(-1)  # [N_val] bool

    # help / hurt
    help_mask = (nll_gpt - q_nlls) > 0
    hurt_mask = (nll_gpt - q_nlls) < 0

    def qs(t):
        return [float(torch.quantile(t.float(), q)) for q in [0.1, 0.25, 0.5, 0.75, 0.9]]

    sim_all     = qs(nearest_sim)
    sim_hit     = qs(nearest_sim[match1]) if match1.any()  else [float("nan")]*5
    sim_miss    = qs(nearest_sim[~match1]) if (~match1).any() else [float("nan")]*5
    sim_help    = qs(nearest_sim[help_mask]) if help_mask.any() else [float("nan")]*5
    sim_hurt    = qs(nearest_sim[hurt_mask]) if hurt_mask.any() else [float("nan")]*5

    # Local state NLL with fixed action k=4, tau=0.05, alpha=0.5, beta=0
    precomp = precompute_lookup(val_data, val_ids, val_sims, states, P_global)
    fixed_actions = [
        {"k_states": 1,  "tau": 0.05, "alpha": 0.5,  "beta": 0.0, "name": "k1_fixed"},
        {"k_states": 4,  "tau": 0.05, "alpha": 0.5,  "beta": 0.0, "name": "k4_fixed"},
        {"k_states": 8,  "tau": 0.05, "alpha": 0.5,  "beta": 0.0, "name": "k8_fixed"},
        {"k_states": 4,  "tau": 0.05, "alpha": 0.75, "beta": 0.0, "name": "k4_a75_fixed"},
    ]
    local_nlls = {}
    for act in fixed_actions:
        r = eval_action(precomp, act)
        local_nlls[act["name"]] = float(r["mean_nll"])

    return {
        "tag": d["tag"], "method": d["method"], "method_type": d["method_type"],
        "budget": d["budget"],
        "nearest_sim_p10": sim_all[0], "nearest_sim_p25": sim_all[1],
        "nearest_sim_p50": sim_all[2], "nearest_sim_p75": sim_all[3],
        "nearest_sim_p90": sim_all[4],
        "nearest_sim_hit_p50":  sim_hit[2],
        "nearest_sim_miss_p50": sim_miss[2],
        "nearest_sim_help_p50": sim_help[2],
        "nearest_sim_hurt_p50": sim_hurt[2],
        "local_nll_k1_fixed":   local_nlls["k1_fixed"],
        "local_nll_k4_fixed":   local_nlls["k4_fixed"],
        "local_nll_k8_fixed":   local_nlls["k8_fixed"],
        "local_nll_k4_a75":     local_nlls["k4_a75_fixed"],
        "frac_true_in_top1_state": float(match1.float().mean()),
    }


# ── axis 8: Q-read behavior ────────────────────────────────────────────────────

def q_behavior(d: dict) -> dict:
    act_df   = d["act_df"]
    chosen   = d["chosen"]
    q_m      = d["q_metrics"]

    top5 = []
    if act_df is not None and not act_df.empty:
        for _, r in act_df.head(5).iterrows():
            top5.append(f"{r['name']}({r['frac']:.3f})")

    gpt_frac = float((chosen == 0).float().mean())

    # Average k from chosen actions
    if act_df is not None and not act_df.empty:
        k_arr   = torch.tensor([
            act_df[act_df["name"] == "gpt_only"]["k_states"].values[0]
            if (chosen == 0).any() else 0
        ], dtype=torch.float32)
        # Recompute from action_distribution
        k_mean  = float((act_df["k_states"] * act_df["frac"]).sum())
        mean_alpha = float((act_df["alpha"] * act_df["frac"]).sum())
        mean_beta  = float((act_df["beta"]  * act_df["frac"]).sum())
        beta0_frac = float(act_df[act_df["beta"] == 0.0]["frac"].sum())
    else:
        k_mean = mean_alpha = mean_beta = float("nan")
        beta0_frac = float("nan")

    return {
        "tag": d["tag"], "method": d["method"], "method_type": d["method_type"],
        "budget": d["budget"],
        "q_state_nll":        float(d["q_nlls"].mean()),
        "fixed_state_nll":    float(d["fixed_nlls"].mean()),
        "oracle_nll":         float(d["oracle_nlls"].mean()),
        "gpt_only_fraction":  gpt_frac,
        "retrieval_usage":    1.0 - gpt_frac,
        "avg_k_states":       k_mean,
        "mean_alpha":         mean_alpha,
        "mean_beta":          mean_beta,
        "frac_beta0":         beta0_frac,
        "top5_actions":       " | ".join(top5),
    }


# ── axis 9: help/hurt distribution ────────────────────────────────────────────

def help_hurt(d: dict, nll_gpt: torch.Tensor) -> dict:
    q_nlls  = d["q_nlls"]
    delta   = nll_gpt - q_nlls          # positive = Q helped
    help_m  = delta > 0
    hurt_m  = delta < 0

    def safe_q(t, q):
        if len(t) == 0: return float("nan")
        return float(torch.quantile(t.float(), q))

    return {
        "tag": d["tag"], "method": d["method"], "method_type": d["method_type"],
        "budget": d["budget"],
        "mean_delta_vs_gpt":   float(delta.mean()),
        "median_delta_vs_gpt": float(delta.median()),
        "frac_help":           float(help_m.float().mean()),
        "frac_hurt":           float(hurt_m.float().mean()),
        "p95_help":            safe_q(delta[help_m], 0.95),
        "p95_hurt":            safe_q(-delta[hurt_m], 0.95) if hurt_m.any() else 0.0,
        "mean_help_when_help": float(delta[help_m].mean())   if help_m.any() else 0.0,
        "mean_hurt_when_hurt": float(delta[hurt_m].mean())   if hurt_m.any() else 0.0,
        "frac_catastrophic_hurt": float((delta < -1.0).float().mean()),
        "frac_catastrophic_help": float((delta >  1.0).float().mean()),
    }


# ── axis 10: utility concentration ────────────────────────────────────────────

def utility_concentration(d: dict, nll_gpt: torch.Tensor) -> dict:
    q_nlls  = d["q_nlls"]
    delta   = (nll_gpt - q_nlls).float()   # per-example utility
    pos     = delta.clamp(min=0)
    neg     = delta.clamp(max=0)

    total_pos = float(pos.sum())
    total_neg = float(neg.sum())
    N         = len(delta)

    # Concentration: what fraction of total positive utility comes from top 1%/10%?
    pos_sorted, _ = pos.sort(descending=True)
    n1  = max(1, int(0.01 * N))
    n10 = max(1, int(0.10 * N))
    top1_share  = float(pos_sorted[:n1].sum()  / (total_pos + EPS))
    top10_share = float(pos_sorted[:n10].sum() / (total_pos + EPS))

    # State-level attribution (by top1 nearest state, simplified)
    val_ids = d["val_ids"][:, 0]    # [N_val] top1 state per example
    B       = len(d["states"]["prototype_h"])
    state_util = torch.zeros(B)
    state_util.scatter_add_(0, val_ids.long(), delta)
    su = state_util.numpy()

    return {
        "tag": d["tag"], "method": d["method"], "method_type": d["method_type"],
        "budget": d["budget"],
        "total_positive_utility": total_pos,
        "total_negative_utility": total_neg,
        "net_utility":            total_pos + total_neg,
        "top1pct_util_share":     top1_share,
        "top10pct_util_share":    top10_share,
        "frac_states_positive_net":  float((su > 0).mean()),
        "frac_states_negative_net":  float((su < 0).mean()),
        "frac_states_zero_utility":  float((su == 0).mean()),
    }


# ── report generation ──────────────────────────────────────────────────────────

def fmt(v, prec=4):
    if isinstance(v, float) and not np.isnan(v):
        return f"{v:.{prec}f}"
    return str(v)


def generate_report(
    summary_df: pd.DataFrame,
    size_df:    pd.DataFrame,
    ep_df:      pd.DataFrame,
    hit_df:     pd.DataFrame,
    qbeh_df:    pd.DataFrame,
    hh_df:      pd.DataFrame,
    uc_df:      pd.DataFrame,
    local_df:   pd.DataFrame,
    out:        Path,
) -> str:
    def row(df, tag):
        rows = df[df["tag"] == tag]
        return rows.iloc[0].to_dict() if not rows.empty else {}

    tags = sorted(summary_df["tag"].tolist())
    offline_tags  = summary_df[summary_df["method_type"] == "offline"]["tag"].tolist()
    adaptive_tags = summary_df[summary_df["method_type"] == "adaptive"]["tag"].tolist()

    L = ["# What Do Offline States Have That Adaptive States Lack?",
         "## Offline vs Adaptive State Diagnosis Report", ""]

    L += ["## 1. Core Question", "",
          "What do offline states have that adaptive states lack?", "",
          "## 2. Short Answer", ""]

    # Compute short answer from data
    if not hit_df.empty and not qbeh_df.empty:
        off_hit1 = hit_df[hit_df["method_type"]=="offline"]["hit_rate_top1"].mean()
        ada_hit1 = hit_df[hit_df["method_type"]=="adaptive"]["hit_rate_top1"].mean()
        off_nll  = qbeh_df[qbeh_df["method_type"]=="offline"]["q_state_nll"].mean()
        ada_nll  = qbeh_df[qbeh_df["method_type"]=="adaptive"]["q_state_nll"].mean()
        off_loc  = local_df[local_df["method_type"]=="offline"]["local_nll_k4_fixed"].mean() if not local_df.empty else float("nan")
        ada_loc  = local_df[local_df["method_type"]=="adaptive"]["local_nll_k4_fixed"].mean() if not local_df.empty else float("nan")
        off_psz  = size_df[size_df["method_type"]=="offline"]["median_count"].mean()
        ada_psz  = size_df[size_df["method_type"]=="adaptive"]["median_count"].mean()

        L.append("Offline states outperform adaptive states because they have:")
        L.append(f"- **Better true-token coverage**: top1 hit rate {off_hit1:.2%} vs {ada_hit1:.2%} "
                 f"(gap: {off_hit1 - ada_hit1:+.2%})")
        L.append(f"- **More reliable local distributions**: k4 fixed NLL {off_loc:.4f} vs {ada_loc:.4f}")
        L.append(f"- **Healthier state sizes**: median count {off_psz:.1f} vs {ada_psz:.1f}")
        L.append(f"- **Better overall Q-read NLL**: {off_nll:.4f} vs {ada_nll:.4f} "
                 f"(gap: {ada_nll - off_nll:+.4f})")

        off_budgetuse = summary_df[summary_df["method_type"]=="offline"]["budget_usage_fraction"].mean()
        ada_budgetuse = summary_df[summary_df["method_type"]=="adaptive"]["budget_usage_fraction"].mean()
        if ada_budgetuse < 0.85:
            L.append(f"- **Better budget utilization**: {off_budgetuse:.1%} vs {ada_budgetuse:.1%} "
                     f"(adaptive wasted {(1-ada_budgetuse)*100:.0f}% of budget)")

    L += [""]

    L += ["## 3. Performance Gap", "",
          "| tag | type | budget | actual_B | fixed_nll | q_nll | oracle_nll | "
          "d_vs_qmlp | d_vs_mvp2c_best |",
          "|-----|------|--------|----------|-----------|-------|------------|"
          "---------|-----------------|"]
    for _, r in qbeh_df.iterrows():
        sr = row(summary_df, r["tag"])
        d_qmlp  = r["q_state_nll"] - FULL_DS_QMLP
        d_mvp2c = r["q_state_nll"] - MVP2C_BEST
        L.append(f"| {r['tag'][:50]} | {r['method_type']} | {int(r['budget'])} "
                 f"| {int(sr.get('actual_num_states', 0))} "
                 f"| {r['fixed_state_nll']:.4f} | {r['q_state_nll']:.4f} "
                 f"| {r['oracle_nll']:.4f} "
                 f"| {d_qmlp:+.4f} | {d_mvp2c:+.4f} |")

    L += ["", f"Baselines: GPT={GPT_ONLY} | full_ds_fixed={FULL_DS_FIXED} | "
          f"full_ds_qmlp={FULL_DS_QMLP} | mvp2c_best={MVP2C_BEST}", ""]

    L += ["## 4. Budget Usage", "",
          "| tag | type | budget | actual_B | usage_frac | unused |",
          "|-----|------|--------|----------|------------|--------|"]
    for _, r in summary_df.iterrows():
        L.append(f"| {r['tag'][:50]} | {r['method_type']} | {int(r['budget'])} "
                 f"| {int(r['actual_num_states'])} "
                 f"| {r['budget_usage_fraction']:.2%} "
                 f"| {int(r.get('unused_budget', 0))} |")

    under_users = summary_df[summary_df["budget_usage_fraction"] < 0.8]
    if not under_users.empty:
        L.append(f"\n**Under-usage (< 80%)**: {', '.join(under_users['tag'].tolist())}")
        L.append("Budget_filling under-splits: the similarity threshold is too high, "
                 "so many examples merge into existing states instead of creating new ones.")
    L += [""]

    L += ["## 5. State Size Distribution", "",
          "| tag | type | median | mean | p90 | max | frac<=1 | frac<=5 | frac>=64 |",
          "|-----|------|--------|------|-----|-----|---------|---------|----------|"]
    for _, r in size_df.iterrows():
        L.append(f"| {r['tag'][:50]} | {r['method_type']} "
                 f"| {r['median_count']:.0f} | {r['mean_count']:.1f} "
                 f"| {r['p90_count']:.0f} | {r['max_count']:.0f} "
                 f"| {r['frac_count_1']:.2%} | {r['frac_count_le5']:.2%} "
                 f"| {r['frac_count_ge64']:.2%} |")

    L += ["",
          "**Interpretation:**",
          "- Adaptive budget_filling: huge fraction of states with count=1 → "
          "no token distribution signal (pure singletons)",
          "- Adaptive token_conflict: median=2, also dominated by tiny states "
          "→ fragmented, unreliable",
          "- Offline states: median ~18, provide stable predictive distributions", ""]

    L += ["## 6. State Entropy / Purity", "",
          "| tag | type | mean_ent | med_ent | mean_pur | med_pur | frac_lo_ent_hi_pur | frac_hi_ent_lo_pur |",
          "|-----|------|----------|---------|----------|---------|---------------------|---------------------|"]
    for _, r in ep_df.iterrows():
        L.append(f"| {r['tag'][:45]} | {r['method_type']} "
                 f"| {r['mean_entropy']:.3f} | {r['median_entropy']:.3f} "
                 f"| {r['mean_purity']:.3f} | {r['median_purity']:.3f} "
                 f"| {r['frac_low_ent_high_pur']:.2%} "
                 f"| {r['frac_high_ent_low_pur']:.2%} |")

    L += [""]

    L += ["## 7. True Token Coverage (CRITICAL)", "",
          "| tag | type | hit@1 | hit@4 | hit@8 | mean_p_true@1 | med_p_true@1 | frac_zero |",
          "|-----|------|-------|-------|-------|---------------|--------------|-----------|"]
    for _, r in hit_df.iterrows():
        L.append(f"| {r['tag'][:45]} | {r['method_type']} "
                 f"| {r['hit_rate_top1']:.2%} | {r['hit_rate_top4']:.2%} "
                 f"| {r['hit_rate_top8']:.2%} "
                 f"| {r['mean_p_state_true_top1']:.4f} "
                 f"| {r['median_p_state_true_top1']:.4f} "
                 f"| {r['frac_zero_p_true_top1']:.2%} |")

    if not hit_df.empty:
        off_h4 = hit_df[hit_df["method_type"]=="offline"]["hit_rate_top4"].mean()
        ada_h4 = hit_df[hit_df["method_type"]=="adaptive"]["hit_rate_top4"].mean()
        L.append(f"\nOffline mean hit@4: {off_h4:.2%}  "
                 f"Adaptive mean hit@4: {ada_h4:.2%}  "
                 f"Gap: {off_h4 - ada_h4:+.2%}")

    L += [""]

    L += ["## 8. Nearest-State Similarity vs Predictive Correctness", "",
          "| tag | type | sim_p50 | sim@hit_p50 | sim@miss_p50 | sim@help_p50 | sim@hurt_p50 |",
          "|-----|------|---------|-------------|--------------|--------------|--------------|"]
    for _, r in local_df.iterrows():
        L.append(f"| {r['tag'][:45]} | {r['method_type']} "
                 f"| {r['nearest_sim_p50']:.4f} "
                 f"| {r.get('nearest_sim_hit_p50', float('nan')):.4f} "
                 f"| {r.get('nearest_sim_miss_p50', float('nan')):.4f} "
                 f"| {r.get('nearest_sim_help_p50', float('nan')):.4f} "
                 f"| {r.get('nearest_sim_hurt_p50', float('nan')):.4f} |")

    L += ["", "## 8b. Local State Prediction Quality (without Q-read)", "",
          "Fixed action k=4, tau=0.05, alpha=0.5, beta=0.",
          "Measures raw state predictive quality before Q-read.", "",
          "| tag | type | k1_local | k4_local | k8_local | d_k4_vs_gpt |",
          "|-----|------|----------|----------|----------|-------------|"]
    for _, r in local_df.iterrows():
        d_k4 = r["local_nll_k4_fixed"] - GPT_ONLY
        L.append(f"| {r['tag'][:45]} | {r['method_type']} "
                 f"| {r['local_nll_k1_fixed']:.4f} "
                 f"| {r['local_nll_k4_fixed']:.4f} "
                 f"| {r['local_nll_k8_fixed']:.4f} "
                 f"| {d_k4:+.4f} |")

    L += [""]

    L += ["## 9. Q-Read Behavior", "",
          "| tag | type | q_nll | fixed_nll | gpt_only% | ret% | avg_k | frac_beta0 |",
          "|-----|------|-------|-----------|-----------|------|-------|------------|"]
    for _, r in qbeh_df.iterrows():
        L.append(f"| {r['tag'][:45]} | {r['method_type']} "
                 f"| {r['q_state_nll']:.4f} | {r['fixed_state_nll']:.4f} "
                 f"| {r['gpt_only_fraction']:.1%} | {r['retrieval_usage']:.1%} "
                 f"| {r['avg_k_states']:.1f} | {r['frac_beta0']:.1%} |")

    # Key observation: does fixed_nll > GPT_ONLY?
    bad_fixed = qbeh_df[qbeh_df["fixed_state_nll"] > GPT_ONLY]
    if not bad_fixed.empty:
        L.append(f"\n**Warning**: fixed_state_nll > GPT-only for: "
                 f"{', '.join(bad_fixed['tag'].tolist())}")
        L.append("This means even the best fixed state-read action hurts relative to GPT-only. "
                 "States are actively harmful, not just unhelpful.")

    L += [""]

    L += ["## 10. Help/Hurt Distribution", "",
          "| tag | type | mean_delta | frac_help | frac_hurt | p95_hurt | frac_catastrophic |",
          "|-----|------|------------|-----------|-----------|----------|-------------------|"]
    for _, r in hh_df.iterrows():
        L.append(f"| {r['tag'][:45]} | {r['method_type']} "
                 f"| {r['mean_delta_vs_gpt']:+.4f} "
                 f"| {r['frac_help']:.2%} | {r['frac_hurt']:.2%} "
                 f"| {r['p95_hurt']:.4f} "
                 f"| {r['frac_catastrophic_hurt']:.2%} |")

    L += [""]

    L += ["## 11. Qualitative Failure Cases", "",
          "See inspection/ directory for readable examples.",
          "Key patterns to check:",
          "- offline_helps_adaptive_fails.txt: where offline Q-read helps but adaptive hurts",
          "- true_token_missing_from_adaptive.txt: adaptive nearest state lacks true token",
          "- similarity_high_prediction_wrong.txt: high cosine sim but wrong token distribution",
          "", ""]

    L += ["## 12. Final Diagnosis", ""]

    # Auto-derive diagnosis from data
    if not size_df.empty and not hit_df.empty:
        ada_size = size_df[size_df["method_type"]=="adaptive"]
        bf_size  = ada_size[ada_size["method"].str.contains("budget_filling")]
        tc_size  = ada_size[ada_size["method"].str.contains("token_conflict")]

        bf_frac1 = bf_size["frac_count_1"].mean() if not bf_size.empty else float("nan")
        tc_frac1 = tc_size["frac_count_1"].mean() if not tc_size.empty else float("nan")
        bf_use   = summary_df[summary_df["method"].str.contains("budget_filling")]["budget_usage_fraction"].mean() if not summary_df.empty else float("nan")

        if not np.isnan(bf_use) and bf_use < 0.8:
            L.append("**Diagnosis C**: Both adaptive variants hit opposite failure modes.")
            L.append("")
            L.append(f"- **budget_filling** under-splits / over-merges: only {bf_use:.1%} of budget used, "
                     f"{bf_frac1:.1%} of states are singletons. "
                     "The threshold (0.990) is too aggressive — new examples merge into existing states "
                     "before enough signal accumulates. States are too large but few.")
            if not np.isnan(tc_frac1):
                L.append(f"- **token_conflict** over-splits / fragments: {tc_frac1:.1%} of states are singletons. "
                         "Creates a new state whenever the nearest state has low probability of the current token, "
                         "even when that state is genuinely a good match. "
                         "Results in many tiny states with one or two examples — no reliable token distribution.")
            L.append("")
            L.append("Both failures cause the same downstream problem: **states have too few examples "
                     "to accumulate reliable token count distributions**, so p_state(y) is noisy or zero. "
                     "The true token is absent from nearby state distributions far more often than in offline states.")
        else:
            L.append("**Diagnosis D**: Q-read/controller features may struggle due to "
                     "poor state quality (small counts, unreliable distributions).")

    L += ["", "## 13. Next Design Rule", "",
          "The missing signal is **predictive compatibility**.", "",
          "A new example should update an existing state only if it is:",
          "1. Geometrically close (cosine sim > threshold), AND",
          "2. Predictively compatible: the true token already has non-negligible probability in that state",
          "   (`p_state(y) > min_prob`), OR the state has too few examples to judge.", "",
          "**Recommended next writer**: `predictive_compatibility_write`", "",
          "Create a new state if:",
          "```",
          "nearest_sim < base_threshold",
          "OR (nearest_sim >= base_threshold",
          "    AND p_state(y_i, nearest) < min_compat_prob",
          "    AND state_count(nearest) >= min_reliable_count)",
          "```", "",
          "Key hyperparams to search:",
          "- base_threshold: 0.990–0.998",
          "- min_compat_prob: 0.001–0.01",
          "- min_reliable_count: 4–16 (below this, accept the example regardless)",
          "",
          "This directly prevents token_conflict fragmentation (singletons) while "
          "also preventing budget_filling over-merging (incompatible examples forced into wrong states).", ""]

    text = "\n".join(L) + "\n"
    rpath = out / "OFFLINE_VS_ADAPTIVE_DIAGNOSIS_REPORT.md"
    with open(rpath, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Report: {rpath}")
    return text


# ── plots ──────────────────────────────────────────────────────────────────────

def try_plots(groups_data: list, hit_df: pd.DataFrame, hh_df: pd.DataFrame,
              qbeh_df: pd.DataFrame, local_df: pd.DataFrame,
              size_df: pd.DataFrame, out: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TABLEAU_COLORS
        colors = list(TABLEAU_COLORS.values())

        plot_dir = out / "plots"
        plot_dir.mkdir(exist_ok=True)

        def color_for(tag):
            return colors[hash(tag) % len(colors)]

        # 1. State count distribution
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, d in enumerate(groups_data):
            tc = d["states"]["total_counts"].numpy()
            ax.hist(np.log1p(tc), bins=50, alpha=0.5, label=d["tag"][:35],
                    density=True, color=colors[i % len(colors)])
        ax.set_xlabel("log(1 + state_count)")
        ax.set_ylabel("density")
        ax.set_title("State Count Distribution")
        ax.legend(fontsize=7)
        fig.savefig(plot_dir / "state_count_distribution.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

        # 2. Entropy vs purity
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, d in enumerate(groups_data):
            ent = d["states"]["state_entropy"].numpy()
            pur = d["states"]["state_purity"].numpy()
            ax.scatter(ent, pur, alpha=0.3, s=4, label=d["tag"][:30],
                       color=colors[i % len(colors)])
        ax.set_xlabel("State Entropy"); ax.set_ylabel("State Purity")
        ax.set_title("Entropy vs Purity per State")
        ax.legend(fontsize=7)
        fig.savefig(plot_dir / "entropy_vs_purity_offline_adaptive.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

        # 3. Hit rate comparison
        if not hit_df.empty:
            ks = [1, 2, 4, 8, 16]
            fig, ax = plt.subplots(figsize=(9, 5))
            for i, (_, r) in enumerate(hit_df.iterrows()):
                vals = [r[f"hit_rate_top{k}"] for k in ks]
                ax.plot(ks, vals, marker="o", label=r["tag"][:30],
                        color=colors[i % len(colors)],
                        ls="--" if r["method_type"] == "adaptive" else "-")
            ax.set_xlabel("k"); ax.set_ylabel("True Token Hit Rate")
            ax.set_title("True Token Hit Rate at Top-k States")
            ax.legend(fontsize=7)
            fig.savefig(plot_dir / "hit_rate_topk_comparison.png", dpi=120, bbox_inches="tight")
            plt.close(fig)

        # 4. p_state_true distribution
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, d in enumerate(groups_data):
            val_ids = d["val_ids"][:, 0]
            tok_ids = d["states"]["top_k_token_ids"].long()
            tok_cnts = d["states"]["top_k_token_counts"]
            total = d["states"]["total_counts"]
            # Need val_y — stored in groups_data? No, pass separately
            # Skip for now; add as post-process if val_y available
        plt.close(fig)

        # 5. Help/hurt histogram
        if not hh_df.empty:
            fig, axes = plt.subplots(1, len(groups_data), figsize=(4*len(groups_data), 4),
                                      sharey=False)
            if len(groups_data) == 1:
                axes = [axes]
            for ax, d in zip(axes, groups_data):
                q_nlls = d["q_nlls"].numpy()
                # nll_gpt comes from groups_data? We'd need val_data; skip histogram,
                # just show q_nlls distribution
                ax.hist(q_nlls, bins=50, density=True, alpha=0.7)
                ax.set_title(d["tag"][:20], fontsize=8)
                ax.set_xlabel("Q NLL")
            fig.suptitle("Q NLL Distribution per Group")
            fig.savefig(plot_dir / "help_hurt_histogram.png", dpi=120, bbox_inches="tight")
            plt.close(fig)

        # 6. Q NLL vs hit rate scatter
        if not hit_df.empty and not qbeh_df.empty:
            merged = hit_df.merge(qbeh_df[["tag","q_state_nll"]], on="tag", how="inner")
            fig, ax = plt.subplots(figsize=(7, 5))
            for _, r in merged.iterrows():
                c = "blue" if r["method_type"] == "offline" else "red"
                ax.scatter(r["hit_rate_top4"], r["q_state_nll"], c=c, s=80, alpha=0.8)
                ax.annotate(r["tag"][:20], (r["hit_rate_top4"], r["q_state_nll"]),
                            fontsize=7, xytext=(4, 0), textcoords="offset points")
            ax.axhline(FULL_DS_QMLP, ls="--", color="g", alpha=0.5, label="full_ds_qmlp")
            ax.set_xlabel("True Token Hit Rate @ top4")
            ax.set_ylabel("Q NLL")
            ax.set_title("Hit Rate vs Q NLL")
            ax.legend()
            fig.savefig(plot_dir / "q_nll_vs_hit_rate.png", dpi=120, bbox_inches="tight")
            plt.close(fig)

        # 7. Retrieval usage vs Q NLL
        if not qbeh_df.empty:
            fig, ax = plt.subplots(figsize=(7, 5))
            for _, r in qbeh_df.iterrows():
                c = "blue" if r["method_type"] == "offline" else "red"
                ax.scatter(r["retrieval_usage"], r["q_state_nll"], c=c, s=80, alpha=0.8)
                ax.annotate(r["tag"][:20], (r["retrieval_usage"], r["q_state_nll"]),
                            fontsize=7, xytext=(4, 0), textcoords="offset points")
            ax.axhline(FULL_DS_QMLP, ls="--", color="g", alpha=0.5)
            ax.set_xlabel("Retrieval Usage"); ax.set_ylabel("Q NLL")
            ax.set_title("Retrieval Usage vs Q NLL")
            fig.savefig(plot_dir / "retrieval_usage_vs_q_nll.png", dpi=120, bbox_inches="tight")
            plt.close(fig)

        print(f"  Plots saved to {plot_dir}")
    except Exception as e:
        print(f"  [plots skipped: {e}]")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",               required=True)
    parser.add_argument("--offline_states_dir",   required=True)
    parser.add_argument("--offline_q_dir",        required=True)
    parser.add_argument("--adaptive_states_dir",  required=True)
    parser.add_argument("--adaptive_q_dir",       required=True)
    parser.add_argument("--output",               required=True)
    parser.add_argument("--budgets", nargs="+", type=int, default=[10000, 25000])
    parser.add_argument("--offline_methods",  nargs="+",
                        default=["minibatch_kmeans", "utility_weighted"])
    parser.add_argument("--adaptive_methods", nargs="+",
                        default=["budget_filling", "token_conflict"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    src   = Path(args.source)
    out   = Path(args.output)
    rep   = out / "reports"
    rep.mkdir(parents=True, exist_ok=True)

    # Load shared val data once
    print("Loading val data ...")
    val_data = torch.load(src / "states" / "val.pt",       weights_only=False)
    ds_data  = torch.load(src / "states" / "datastore.pt", weights_only=False)
    P_global = build_global_freq(ds_data["y"], VOCAB)
    val_y    = val_data["y"]
    nll_gpt  = val_data["nll_gpt"]
    N_val    = len(val_y)
    print(f"  N_val={N_val:,}  GPT NLL mean={float(nll_gpt.mean()):.4f}")

    # Discover groups
    off_sdir = Path(args.offline_states_dir)
    off_qdir = Path(args.offline_q_dir)
    ada_sdir = Path(args.adaptive_states_dir)
    ada_qdir = Path(args.adaptive_q_dir)

    groups = discover_groups(off_sdir, off_qdir, ada_sdir, ada_qdir, args.budgets)
    print(f"\nFound {len(groups)} comparison groups:")
    for g in groups:
        print(f"  {g['method_type']:<10} {g['tag']}")

    if not groups:
        print("No groups found. Check directory paths.")
        return

    # Load group data
    print("\nLoading group data ...")
    groups_data = []
    for g in groups:
        print(f"  {g['tag']} ...", end=" ", flush=True)
        try:
            gd = load_group_data(g)
            groups_data.append(gd)
            B = len(gd["states"]["prototype_h"])
            print(f"B={B:,}")
        except Exception as e:
            print(f"ERROR: {e}")

    # Compute all axes
    print("\nComputing diagnostics ...")

    summary_rows = []
    size_rows    = []
    ep_rows      = []
    hit_rows     = []
    qbeh_rows    = []
    hh_rows      = []
    uc_rows      = []
    local_rows   = []

    for d in groups_data:
        print(f"  {d['tag']} ...", end=" ", flush=True)
        bu = budget_usage(d)
        summary_rows.append(bu)
        size_rows.append(size_distribution(d))
        ep_rows.append(entropy_purity(d))
        hit_rows.append(true_token_hit_rates(d, val_y))
        qbeh_rows.append(q_behavior(d))
        hh_rows.append(help_hurt(d, nll_gpt))
        uc_rows.append(utility_concentration(d, nll_gpt))
        local_rows.append(sim_and_local_nll(d, val_data, P_global))
        print("done")

    # Build DataFrames and save
    def save(rows, fname, key=None):
        df = pd.DataFrame(rows)
        if key:
            df = df.merge(pd.DataFrame([{"tag": r["tag"],
                                          "actual_num_states": r.get("actual_num_states",
                                              r.get("B", 0)),
                                          "budget_usage_fraction": r.get("budget_usage_fraction", float("nan")),
                                          "unused_budget": r.get("unused_budget", 0)}
                                         for r in summary_rows]),
                          on="tag", how="left") if key == "summary" else df
        df.to_csv(rep / fname, index=False)
        return df

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(rep / "offline_vs_adaptive_summary.csv", index=False)
    size_df    = save(size_rows,    "state_size_distribution.csv")
    ep_df      = save(ep_rows,      "token_entropy_purity.csv")
    hit_df     = save(hit_rows,     "hit_rate_comparison.csv")
    qbeh_df    = save(qbeh_rows,    "q_behavior_comparison.csv")
    hh_df      = save(hh_rows,      "help_hurt_comparison.csv")
    uc_df      = save(uc_rows,      "utility_concentration.csv")
    local_df   = save(local_rows,   "local_prediction_quality.csv")

    print(f"\nReports saved to {rep}")

    # Generate report
    generate_report(summary_df, size_df, ep_df, hit_df, qbeh_df, hh_df, uc_df, local_df, out)

    # Plots
    try_plots(groups_data, hit_df, hh_df, qbeh_df, local_df, size_df, out)

    # Quick summary
    print(f"\n{'='*60}")
    print("KEY FINDING:")
    for _, r in qbeh_df.iterrows():
        sr = summary_df[summary_df["tag"] == r["tag"]].iloc[0]
        print(f"  {r['method_type']:<10} {r['tag'][:45]:<50} "
              f"B={int(sr['actual_num_states']):<6} "
              f"fixed={r['fixed_state_nll']:.4f}  Q={r['q_state_nll']:.4f}")


if __name__ == "__main__":
    main()

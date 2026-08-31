"""
log_read_object_usage.py  --  MVP 4c-0 Stage 2

For a given trajectory's final state file, evaluates all held-out val queries
and records:
  - Which state row indices (objects) participated in each query's prediction
  - The retrieval similarities and mixture weights
  - The NLL for each query under the selected READ action

Uses two READ modes:
  fixed  -- best fixed action (k=4, tau=0.05, alpha=0.75, beta=0.0)
  qread  -- Q-read controller (trained on CT data, applied to val)

Also computes leave-one-object-out NLL for each selected object in each query.
This is the core counterfactual utility measurement.

Outputs (in {output}/read_usage/):
  {traj}_queries.parquet           -- per-query summary
  {traj}_object_usage.parquet      -- (query_id, object_id) with gain measurements

Usage:
  python src/log_read_object_usage.py \\
    --trajectory oracle \\
    --source scale_200k_seed42 \\
    --oracle_dir outputs_mvp4a0_oracle_reconstruction \\
    --output outputs_mvp4c0_delayed_credit_audit_fast \\
    --teacher minibatch_kmeans --budget 10000 \\
    --read_modes fixed qread \\
    --max_queries 10000 \\
    --device cuda
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

from utils import get_device, set_seed
from evaluate_predictive_states import (
    compute_top_k_state_sims,
    precompute_lookup,
    build_global_freq,
)
from train_q_state_read import (
    build_action_grid, build_action_features,
    build_obs_features, compute_reward_matrix,
    find_best_fixed, train_q_mlp, apply_q_controller,
    QStateReadMLP, Q_DIM, OBS_DIM, ACT_DIM, MAX_K, VOCAB,
)

EPS = 1e-10

# Best known fixed action
FIXED_K     = 4
FIXED_TAU   = 0.05
FIXED_ALPHA = 0.75
FIXED_BETA  = 0.0


def _load_state(state_path: Path) -> dict:
    return torch.load(state_path, map_location="cpu", weights_only=False)


def _load_val(source: Path, max_queries: int, seed: int):
    val_data = torch.load(source / "states" / "val.pt", weights_only=False)
    N_val    = len(val_data["y"])
    if max_queries and max_queries < N_val:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N_val, size=max_queries, replace=False)
        idx = np.sort(idx)
        def _sub(d, idx):
            out = {}
            for k, v in d.items():
                out[k] = v[idx] if isinstance(v, torch.Tensor) else v
            return out
        val_data = _sub(val_data, idx)
        original_indices = idx
    else:
        original_indices = np.arange(N_val)
    return val_data, original_indices


def _load_ct(source: Path):
    return torch.load(source / "states" / "controller_train.pt", weights_only=False)


def _norm_h(data):
    h = data["h"].float()
    return h / (h.norm(dim=-1, keepdim=True) + EPS)


def _compute_nll_fixed(
    y_true: torch.Tensor,      # [N]
    p_gpt:  torch.Tensor,      # [N]
    nll_gpt: torch.Tensor,     # [N]
    top_ids:  torch.Tensor,    # [N, K]
    top_sims: torch.Tensor,    # [N, K]
    states: dict,
    k: int, tau: float, alpha: float, beta: float = 0.0,
    P_global: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (nll [N], p_local [N], weights [N, k])."""
    N   = len(y_true)
    if k == 0:
        return nll_gpt.clone(), torch.zeros(N), torch.zeros(N, 1)

    k_eff    = min(k, top_ids.shape[1], states["prototype_h"].shape[0])
    sims_k   = top_sims[:, :k_eff]          # [N, k_eff]
    ids_k    = top_ids[:, :k_eff]           # [N, k_eff]

    tok_ids  = states["top_k_token_ids"]    # [B, TOP_K]
    tok_cnts = states["top_k_token_counts"] # [B, TOP_K]
    tot_cnts = states["total_counts"]       # [B]
    TOP_K    = tok_ids.shape[1]

    # Per-query count of true token in each of the k states
    sel_tids = tok_ids[ids_k]               # [N, k, TOP_K]
    sel_tcnts = tok_cnts[ids_k]            # [N, k, TOP_K]
    sel_tot  = tot_cnts[ids_k]             # [N, k]

    y_exp    = y_true.view(N, 1, 1).expand(N, k_eff, TOP_K)
    match    = (sel_tids.long() == y_exp.long())
    raw_cnt  = (sel_tcnts * match.float()).sum(-1)  # [N, k]

    p_global_y = P_global[y_true.long()] if P_global is not None else torch.zeros(N)

    weights  = F.softmax(sims_k / (tau + EPS), dim=-1)
    p_state  = (raw_cnt + beta * p_global_y.unsqueeze(1)) / (sel_tot + beta + EPS)
    p_local  = (weights * p_state).sum(-1).clamp(EPS)
    p_final  = (alpha * p_gpt + (1 - alpha) * p_local).clamp(EPS)
    nll      = -p_final.log()

    return nll, p_local, weights


def _leave_one_out_nll(
    query_idx: int,
    remove_pos: int,          # position in top-k to remove
    y_true: torch.Tensor,     # [N]
    p_gpt:  torch.Tensor,     # [N]
    nll_gpt: torch.Tensor,    # [N]
    top_ids:  torch.Tensor,   # [N, K]
    top_sims: torch.Tensor,   # [N, K]
    states: dict,
    k: int, tau: float, alpha: float, beta: float,
    P_global: torch.Tensor,
) -> float:
    """Compute NLL for one query with one state removed from its top-k set."""
    if k <= 1:
        # Only one state; removing it means GPT-only
        return float(nll_gpt[query_idx])

    # Extract this query's top-k
    ids_k  = top_ids[query_idx, :k].clone()     # [k]
    sims_k = top_sims[query_idx, :k].clone()    # [k]

    # Remove position remove_pos
    mask = torch.ones(k, dtype=torch.bool)
    mask[remove_pos] = False
    ids_rem  = ids_k[mask]      # [k-1]
    sims_rem = sims_k[mask]     # [k-1]

    # Recompute NLL without that object
    tok_ids  = states["top_k_token_ids"]
    tok_cnts = states["top_k_token_counts"]
    tot_cnts = states["total_counts"]
    TOP_K    = tok_ids.shape[1]

    y_q = int(y_true[query_idx])
    sel_tids  = tok_ids[ids_rem]    # [k-1, TOP_K]
    sel_tcnts = tok_cnts[ids_rem]   # [k-1, TOP_K]
    sel_tot   = tot_cnts[ids_rem]   # [k-1]

    match    = (sel_tids.long() == y_q).float()
    raw_cnt  = (sel_tcnts * match).sum(-1)  # [k-1]

    p_global_y = float(P_global[y_q]) if P_global is not None else 0.0

    weights  = F.softmax(sims_rem / (tau + EPS), dim=0)
    p_state  = (raw_cnt + beta * p_global_y) / (sel_tot + beta + EPS)
    p_local  = float((weights * p_state).sum().clamp(EPS))
    p_gpt_q  = float(p_gpt[query_idx])
    p_final  = max(alpha * p_gpt_q + (1 - alpha) * p_local, EPS)
    return -float(torch.tensor(p_final).log())


def process_trajectory(
    traj: str,
    state_path: Path,
    source: Path,
    device: torch.device,
    read_modes: list,
    max_queries: int,
    seed: int,
    out_dir: Path,
    args,
    datastore_dir: Path = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    print(f"\n  Loading state: {state_path.name} ...")
    states   = _load_state(state_path)
    B        = states["prototype_h"].shape[0]
    print(f"  State has {B} objects")

    val_data, orig_idx = _load_val(source, max_queries, seed)
    N_val   = len(val_data["y"])
    y_true  = val_data["y"].long()
    p_gpt   = val_data["p_gpt_true"].float()
    nll_gpt = val_data["nll_gpt"].float()
    val_h   = _norm_h(val_data)

    ds_path = (datastore_dir if datastore_dir is not None else source) / "states" / "datastore.pt"
    P_global = build_global_freq(
        torch.load(ds_path, weights_only=False)["y"].long(), VOCAB)

    print(f"  Computing top-{MAX_K} state similarities for {N_val} val queries ...")
    proto = states["prototype_h"].float()
    top_ids, top_sims = compute_top_k_state_sims(val_h, proto, MAX_K, device)

    # Q-read: need CT data + train Q model
    q_model = None
    actions  = build_action_grid(fast=True)
    act_feats = build_action_features(actions)
    best_fixed_action = None

    if "qread" in read_modes:
        print(f"  Loading CT data for Q-read training ...")
        ct_data = _load_ct(source)
        ct_h    = _norm_h(ct_data)
        ct_ids, ct_sims = compute_top_k_state_sims(ct_h, proto, MAX_K, device)

        ct_precomp  = precompute_lookup(ct_data,  ct_ids,  ct_sims,  states, P_global)
        val_precomp = precompute_lookup(val_data, top_ids, top_sims, states, P_global)

        obs_ct  = build_obs_features(ct_data,  ct_ids,  ct_sims,  states)
        obs_val = build_obs_features(val_data, top_ids, top_sims, states)

        # Check for cached Q model
        cache_dir  = out_dir / "read_usage" / f"q_cache_{traj}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        q_path = cache_dir / "q_model.pt"
        if q_path.exists():
            print(f"  Loading cached Q model ...")
            ckpt = torch.load(q_path, weights_only=False)
            q_model = QStateReadMLP(Q_DIM)
            q_model.load_state_dict(ckpt["state_dict"])
            q_model.x_mu  = ckpt["x_mu"]
            q_model.x_std = ckpt["x_std"]
            q_model.y_mu  = ckpt["y_mu"]
            q_model.y_std = ckpt["y_std"]
        else:
            print(f"  Training Q model ...")
            ct_rewards = compute_reward_matrix(ct_precomp, actions)
            q_model, _ = train_q_mlp(obs_ct, act_feats, ct_rewards, device,
                                      max_q_samples=400_000, n_epochs=20, seed=seed)
            torch.save({
                "state_dict": q_model.state_dict(),
                "x_mu": q_model.x_mu, "x_std": q_model.x_std,
                "y_mu": q_model.y_mu, "y_std": q_model.y_std,
            }, q_path)

        # Choose actions per query
        chosen_actions = apply_q_controller(q_model, obs_val, act_feats, device)

        # Find best fixed for comparison
        ct_rewards_for_fixed = compute_reward_matrix(ct_precomp, actions)
        best_ai, best_fixed_action = find_best_fixed(ct_rewards_for_fixed, actions)

    query_rows  = []
    usage_rows  = []
    query_id    = 0

    print(f"  Computing per-query object usage and counterfactual gains ...")

    # Fixed READ params
    k_fixed   = FIXED_K
    tau_fixed = FIXED_TAU
    alp_fixed = FIXED_ALPHA
    bet_fixed = FIXED_BETA

    # Compute fixed NLL for all queries at once
    nll_fixed_all, p_loc_fixed, w_fixed = _compute_nll_fixed(
        y_true, p_gpt, nll_gpt, top_ids, top_sims, states,
        k_fixed, tau_fixed, alp_fixed, bet_fixed, P_global)

    # Q-read NLL per query
    if q_model is not None:
        nll_qread_all = torch.zeros(N_val)
        for qi in range(N_val):
            ai  = int(chosen_actions[qi])
            act = actions[ai]
            if act["k_states"] == 0:
                nll_qread_all[qi] = nll_gpt[qi]
            else:
                nll_q, _, _ = _compute_nll_fixed(
                    y_true[qi:qi+1], p_gpt[qi:qi+1], nll_gpt[qi:qi+1],
                    top_ids[qi:qi+1], top_sims[qi:qi+1], states,
                    act["k_states"], act["tau"], act["alpha"], act["beta"],
                    P_global)
                nll_qread_all[qi] = nll_q[0]

    for qi in range(N_val):
        y_q     = int(y_true[qi])
        gpt_nll = float(nll_gpt[qi])
        orig_qi = int(orig_idx[qi])

        row_base = {
            "query_id":      query_id,
            "original_idx":  orig_qi,
            "true_token":    y_q,
            "gpt_nll":       gpt_nll,
        }

        for mode in read_modes:
            if mode == "fixed":
                k, tau, alpha, beta = k_fixed, tau_fixed, alp_fixed, bet_fixed
                nll_full = float(nll_fixed_all[qi])
                k_eff    = min(k, top_ids.shape[1], B)
            elif mode == "qread":
                if q_model is None:
                    continue
                ai  = int(chosen_actions[qi])
                act = actions[ai]
                k, tau, alpha, beta = (act["k_states"], act["tau"],
                                       act["alpha"], act["beta"])
                nll_full = float(nll_qread_all[qi])
                k_eff    = min(k, top_ids.shape[1], B)
            else:
                continue

            row_base[f"nll_{mode}"] = nll_full
            row_base[f"retrieval_gain_{mode}"] = gpt_nll - nll_full

            if k_eff == 0:
                continue

            # Per-object leave-one-out gains
            for pos in range(k_eff):
                state_row  = int(top_ids[qi, pos])
                sim_val    = float(top_sims[qi, pos])
                nll_minus_j = _leave_one_out_nll(
                    qi, pos, y_true, p_gpt, nll_gpt, top_ids, top_sims,
                    states, k_eff, tau, alpha, beta, P_global)
                gain_j = nll_minus_j - nll_full  # positive = object helped

                usage_rows.append({
                    "query_id":   query_id,
                    "original_query_idx": orig_qi,
                    "true_token": y_q,
                    "gpt_nll":    gpt_nll,
                    "state_row":  state_row,
                    "rank":       pos,
                    "sim":        sim_val,
                    "nll_full":   nll_full,
                    "nll_minus_j": nll_minus_j,
                    "gain_j":     gain_j,   # positive = helpful
                    "read_mode":  mode,
                    "trajectory": traj,
                })

        query_rows.append(row_base)
        query_id += 1

        if query_id % 2000 == 0:
            print(f"    {query_id}/{N_val} queries ...")

    query_df = pd.DataFrame(query_rows)
    usage_df = pd.DataFrame(usage_rows)

    return query_df, usage_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory",       required=True, choices=["oracle", "bandit"])
    ap.add_argument("--source",           required=True,
                    help="Raw data dir with train.pt / val.pt")
    ap.add_argument("--datastore_dir",    default=None,
                    help="Dir containing states/datastore.pt. Defaults to --source.")
    ap.add_argument("--oracle_dir",       default=None)
    ap.add_argument("--bandit_dir",       default=None)
    ap.add_argument("--output",           required=True)
    ap.add_argument("--teacher",          default="minibatch_kmeans")
    ap.add_argument("--budget",           type=int, default=10000)
    ap.add_argument("--scorer",           default="mlp")
    ap.add_argument("--read_modes",       nargs="+", default=["fixed"],
                    choices=["fixed", "qread"])
    ap.add_argument("--max_queries",      type=int, default=10000)
    ap.add_argument("--seed",             type=int, default=42)
    ap.add_argument("--device",           default="cuda")
    ap.add_argument("--force",            action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device  = get_device({"device": args.device})
    src     = Path(args.source)
    ds_dir  = Path(args.datastore_dir) if args.datastore_dir else src
    out_dir = Path(args.output)
    ru_dir  = out_dir / "read_usage"
    ru_dir.mkdir(parents=True, exist_ok=True)

    traj = args.trajectory
    qp   = ru_dir / f"{traj}_queries.parquet"
    up   = ru_dir / f"{traj}_object_usage.parquet"

    if qp.exists() and up.exists() and not args.force:
        print(f"[cached] {qp.name}, {up.name}")
        return

    # Resolve state path
    if traj == "oracle":
        tag        = f"oracle_seq_{args.teacher}_B{args.budget}"
        state_path = (Path(args.oracle_dir) /
                      "reconstructions" / "sequential_running_mean" /
                      "states" / f"{tag}.pt")
        if not state_path.exists():
            # Try exact_teacher_prototype
            tag2 = f"oracle_exact_{args.teacher}_B{args.budget}"
            sp2  = (Path(args.oracle_dir) /
                    "reconstructions" / "exact_teacher_prototype" /
                    "states" / f"{tag2}.pt")
            if sp2.exists():
                state_path = sp2
                tag = tag2
        print(f"Oracle state: {state_path}")
    else:
        scorer = args.scorer
        tag    = f"bandit_write_{scorer}_B{args.budget}"
        states_dir = Path(args.bandit_dir) / "rollout" / "states"
        B = args.budget
        # MVP 4b.1 naming: prefer variant B (more objects), then A; seeds 123 then 999
        mvp4b1_candidates = [
            states_dir / f"mvp4b1_{scorer}_vB_s123_B{B}.pt",
            states_dir / f"mvp4b1_{scorer}_vB_s999_B{B}.pt",
            states_dir / f"mvp4b1_{scorer}_vA_s123_B{B}.pt",
            states_dir / f"mvp4b1_{scorer}_vA_s999_B{B}.pt",
        ]
        old_candidates = [
            states_dir / f"bandit_write_{scorer}_B{B}.pt",
            states_dir / f"bandit_write_B{B}.pt",
        ]
        for candidate in mvp4b1_candidates + old_candidates:
            if candidate.exists():
                state_path = candidate
                tag = candidate.stem
                break
        else:
            raise FileNotFoundError(f"Could not find bandit state in {args.bandit_dir}/rollout/states/")
        print(f"Bandit state: {state_path}")

    query_df, usage_df = process_trajectory(
        traj, state_path, src, device, args.read_modes,
        args.max_queries, args.seed, out_dir, args,
        datastore_dir=ds_dir)

    query_df.to_parquet(qp, index=False)
    usage_df.to_parquet(up, index=False)

    print(f"\nSaved:")
    print(f"  {qp}  ({len(query_df):,} queries)")
    print(f"  {up}  ({len(usage_df):,} usage records)")
    print(f"  Mean retrieval gain (fixed): "
          f"{float(query_df.get('retrieval_gain_fixed', pd.Series([0])).mean()):.4f}")


if __name__ == "__main__":
    main()

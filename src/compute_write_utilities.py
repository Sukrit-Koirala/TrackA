"""
compute_write_utilities.py

Assigns a utility score to each datastore entry based on how helpful it was
as retrieved evidence for controller_train queries.

For each ct query q and chosen action (k, tau, alpha):
  improvement = NLL_GPT[q] - NLL_final[q]
  weights = softmax(neighbor_sims[:k] / tau)

  For each neighbor i in top-k of q:
    if neighbor_y[i] == y_true[q]:
      utility_positive[ds_idx] += max(improvement, 0) * weights[i]
      positive_hits[ds_idx] += 1
    negative_utility[ds_idx] += max(-improvement, 0) * weights[i]
    retrieval_count[ds_idx] += 1

utility_net = utility_positive - beta * negative_utility  (beta=0.25)

Modes:
  A  best-action per query (argmax reward over all 46 actions)
  B  fixed action from fixed_baselines.json (best CT-selected)
  C  Q-MLP-full chosen action per query (if checkpoint available)

Usage:
  python src/compute_write_utilities.py \\
    --source outputs_scale_sweep/scale_200k_seed42 \\
    --output outputs_mvp2_write_memory
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F

from utils import build_action_grid, get_device, set_seed


BETA = 0.25   # negative-utility weight in utility_net
MAX_K = 64
DEFAULT_K_VALUES   = [4, 8, 16, 32, 64]
DEFAULT_TAU_VALUES = [0.05, 0.1, 0.2]
DEFAULT_ALPHA_VALUES = [0.25, 0.5, 0.75]


def build_default_actions():
    actions = [{"name": "gpt_only", "k": 0, "tau": 0.1, "alpha": 1.0}]
    for k in DEFAULT_K_VALUES:
        for tau in DEFAULT_TAU_VALUES:
            for alpha in DEFAULT_ALPHA_VALUES:
                actions.append({"name": f"k{k}_t{tau}_a{alpha}",
                                 "k": k, "tau": tau, "alpha": alpha})
    return actions


def compute_utility(
    per_action_rewards: torch.Tensor,   # [N_ct, A]
    chosen_action_idx: torch.Tensor,    # [N_ct]  which action each query uses
    actions: list[dict],
    ct_nbrs: dict,                      # neighbor_indices [N, 64], sims [N, 64], y [N, 64]
    ct_y: torch.Tensor,                 # [N_ct] true next tokens
    N_ds: int,
) -> dict:
    """
    Distribute utility to datastore entries.
    Vectorised: group queries by chosen action, process each group together.
    """
    utility_positive = torch.zeros(N_ds)
    negative_utility = torch.zeros(N_ds)
    positive_hits    = torch.zeros(N_ds)
    retrieval_count  = torch.zeros(N_ds, dtype=torch.long)

    nbr_idx  = ct_nbrs["neighbor_indices"]   # [N_ct, 64]
    nbr_sims = ct_nbrs["neighbor_sims"]      # [N_ct, 64]
    nbr_y    = ct_nbrs["neighbor_y"]         # [N_ct, 64]

    for ai, action in enumerate(actions):
        k = action["k"]
        if k == 0:
            continue   # GPT-only: no neighbors retrieved, no utility

        tau  = action["tau"]
        mask = (chosen_action_idx == ai)
        if mask.sum() == 0:
            continue

        q_rows = mask.nonzero(as_tuple=True)[0]   # query indices choosing this action

        # improvements for these queries (reward = nll_gpt - nll_final when lambda_cost=0)
        improvements = per_action_rewards[q_rows, ai]   # [Nq]

        sims_k  = nbr_sims[q_rows, :k]           # [Nq, k]
        weights = F.softmax(sims_k / tau, dim=-1) # [Nq, k]
        ds_idx  = nbr_idx[q_rows, :k]             # [Nq, k]
        nbr_y_k = nbr_y[q_rows, :k]               # [Nq, k]
        y_true_k = ct_y[q_rows].unsqueeze(1).expand_as(nbr_y_k)  # [Nq, k]

        match = (nbr_y_k == y_true_k).float()     # [Nq, k]

        # Positive utility: matching neighbors + positive improvement
        pos_imp    = improvements.clamp(min=0).unsqueeze(1)   # [Nq, 1]
        pos_contrib = (pos_imp * weights * match).reshape(-1)  # [Nq*k]
        ds_flat     = ds_idx.reshape(-1)

        utility_positive.scatter_add_(0, ds_flat, pos_contrib)
        positive_hits.scatter_add_(0, ds_flat, match.reshape(-1))

        # Negative utility: all neighbors + negative improvement (retrieval hurt)
        neg_imp    = (-improvements).clamp(min=0).unsqueeze(1)
        neg_contrib = (neg_imp * weights).reshape(-1)
        negative_utility.scatter_add_(0, ds_flat, neg_contrib)

        # Retrieval count (unweighted, how many times each entry was retrieved)
        ones = torch.ones(len(ds_flat), dtype=torch.long)
        retrieval_count.scatter_add_(0, ds_flat, ones)

    utility_net = utility_positive - BETA * negative_utility
    return {
        "utility_positive": utility_positive,
        "utility_net":      utility_net,
        "positive_hits":    positive_hits,
        "retrieval_count":  retrieval_count.float(),
        "negative_utility": negative_utility,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True,
                        help="Path to scale sweep output dir (e.g. outputs_scale_sweep/scale_200k_seed42)")
    parser.add_argument("--output", required=True,
                        help="MVP2 output root (e.g. outputs_mvp2_write_memory)")
    parser.add_argument("--modes", nargs="+", default=["A", "B", "C"],
                        help="Utility modes to compute (A B C)")
    args = parser.parse_args()

    src  = Path(args.source)
    out  = Path(args.output)
    util_dir = out / "utility"
    util_dir.mkdir(parents=True, exist_ok=True)

    device = get_device({"device": "cuda"})
    set_seed(42)

    print(f"\ncompute_write_utilities")
    print(f"Source: {src}")
    print(f"Output: {util_dir}")

    # ── load data ────────────────────────────────────────────────────────────
    print("\nLoading data ...")
    states_dir  = src / "states"
    nbrs_dir    = src / "neighbors"
    models_dir  = src / "models"
    reports_dir = src / "reports"

    ct_data = torch.load(states_dir / "controller_train.pt", weights_only=False)
    ct_nbrs = torch.load(nbrs_dir   / f"controller_train_top{MAX_K}.pt", weights_only=False)
    ct_y    = ct_data["y"]                     # [N_ct]
    N_ct    = len(ct_y)

    ds      = torch.load(states_dir / "datastore.pt", weights_only=False)
    N_ds    = len(ds["y"])

    print(f"  N_ct={N_ct:,}  N_ds={N_ds:,}")

    # Build or load action grid
    actions = build_default_actions()
    A = len(actions)
    print(f"  Actions: {A}")

    # ── load per-action rewards ───────────────────────────────────────────────
    cache_path = models_dir / "q_base_data.pt"
    if cache_path.exists():
        print("\nLoading cached q_base_data ...")
        cache = torch.load(cache_path, weights_only=False)
        per_action_rewards = cache["per_action_rewards"]   # [N_ct, A]
        print(f"  per_action_rewards: {per_action_rewards.shape}")
    else:
        print("\nq_base_data.pt not found — run train_q_controller.py first")
        print("Computing per-action rewards from scratch ...")
        from gate_chain import evaluate_action_for_queries
        per_action_rewards = torch.zeros(N_ct, A)
        for ai, action in enumerate(actions):
            m = evaluate_action_for_queries(
                ct_data, ct_nbrs, action, max_k=MAX_K, lambda_cost=0.0, eps=1e-12)
            per_action_rewards[:, ai] = m["per_example_reward"]
            if (ai + 1) % 10 == 0:
                print(f"  {ai+1}/{A} actions evaluated")

    # ── Mode A: best action per query ─────────────────────────────────────────
    if "A" in args.modes:
        print("\n[Mode A] Best-action utility ...")
        best_a = per_action_rewards.argmax(dim=-1)   # [N_ct]
        n_gpt_only = (best_a == 0).sum().item()
        print(f"  Queries where best=GPT-only: {n_gpt_only}/{N_ct} ({100*n_gpt_only/N_ct:.1f}%)")

        u = compute_utility(per_action_rewards, best_a, actions, ct_nbrs, ct_y, N_ds)
        u["utility_mode"] = "A"
        u["beta"]         = BETA
        torch.save(u, util_dir / "write_utilities_modeA.pt")
        n_useful = (u["utility_positive"] > 0).sum().item()
        print(f"  Entries with positive utility: {n_useful:,}/{N_ds:,} ({100*n_useful/N_ds:.1f}%)")
        print(f"  Saved -> utility/write_utilities_modeA.pt")

    # ── Mode B: fixed action ──────────────────────────────────────────────────
    if "B" in args.modes:
        print("\n[Mode B] Fixed-action utility ...")
        bl_path = reports_dir / "fixed_baselines.json"
        if bl_path.exists():
            with open(bl_path) as f:
                bl = json.load(f)
            best_name = bl["best_train_selected_name"]
        else:
            best_name = "k16_t0.05_a0.75"
            print(f"  WARNING: fixed_baselines.json not found, using default {best_name}")

        name_to_idx = {a["name"]: i for i, a in enumerate(actions)}
        if best_name not in name_to_idx:
            print(f"  WARNING: {best_name} not in action grid, skipping Mode B")
        else:
            fixed_idx  = name_to_idx[best_name]
            chosen_b   = torch.full((N_ct,), fixed_idx, dtype=torch.long)
            print(f"  Fixed action: {best_name} (idx {fixed_idx})")

            u = compute_utility(per_action_rewards, chosen_b, actions, ct_nbrs, ct_y, N_ds)
            u["utility_mode"]      = "B"
            u["beta"]              = BETA
            u["fixed_action_name"] = best_name
            torch.save(u, util_dir / "write_utilities_modeB.pt")
            n_useful = (u["utility_positive"] > 0).sum().item()
            print(f"  Entries with positive utility: {n_useful:,}/{N_ds:,} ({100*n_useful/N_ds:.1f}%)")
            print(f"  Saved -> utility/write_utilities_modeB.pt")

    # ── Mode C: Q-MLP-full chosen action ─────────────────────────────────────
    if "C" in args.modes:
        ckpt_path = models_dir / "q_mlp_full.pt"
        if not ckpt_path.exists():
            print("\n[Mode C] Skipping: q_mlp_full.pt not found")
        else:
            print("\n[Mode C] Q-MLP-full action utility ...")
            from train_q_controller import (
                load_q_model, build_action_features, apply_q_controller,
            )
            model, action_names = load_q_model(ckpt_path, device)
            model.eval()
            name_to_idx = {a["name"]: i for i, a in enumerate(actions)}
            act_feats   = torch.stack([
                build_action_features(a, MAX_K) for a in actions
            ])

            from train_controller import build_features
            X_obs_ct = build_features(ct_data, ct_nbrs)   # [N_ct, 13]

            # Map Q-MLP subset names back to full action grid indices
            sub_to_full = [name_to_idx[n] for n in action_names if n in name_to_idx]
            sub_feats   = act_feats[sub_to_full]
            sub_actions = [actions[i] for i in sub_to_full]

            chosen_sub  = apply_q_controller(model, X_obs_ct, sub_feats, device)   # [N_ct] into sub
            chosen_c    = torch.tensor([sub_to_full[i] for i in chosen_sub.tolist()], dtype=torch.long)

            u = compute_utility(per_action_rewards, chosen_c, actions, ct_nbrs, ct_y, N_ds)
            u["utility_mode"] = "C"
            u["beta"]         = BETA
            torch.save(u, util_dir / "write_utilities_modeC.pt")
            n_useful = (u["utility_positive"] > 0).sum().item()
            print(f"  Entries with positive utility: {n_useful:,}/{N_ds:,} ({100*n_useful/N_ds:.1f}%)")
            print(f"  Saved -> utility/write_utilities_modeC.pt")

    print("\nDone.")


if __name__ == "__main__":
    main()

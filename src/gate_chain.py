"""
gate_chain.py

Core MATCH → SELECT → PREDICT → MIX gate chain.

For each query hidden state h_t with true next token y_true:

  MATCH:   cosine similarity vs datastore (pre-computed, stored in neighbor_data)
  SELECT:  take top-k neighbor entries
  PREDICT: weights = softmax(sims / tau)
           P_local(y_true) = sum(weights_i  for i where y_i == y_true)
  MIX:     P_final(y_true) = alpha * P_GPT(y_true) + (1-alpha) * P_local(y_true)

  NLL_final = -log(P_final(y_true) + eps)
  reward    = NLL_GPT - NLL_final - lambda_cost * (k / max_k)

The controller learns to choose (k, tau, alpha) per query.
"""

import torch
import torch.nn.functional as F


def local_true_probability(
    neighbor_y: torch.Tensor,    # [N, k]
    neighbor_sims: torch.Tensor, # [N, k]
    y_true: torch.Tensor,        # [N]
    k: int,
    tau: float,
) -> torch.Tensor:
    """
    Compute P_local(y_true) for each query using the top-k neighbors.

    P_local(y_true) is the total softmax weight assigned to neighbors
    whose next-token label matches y_true.  This is NOT a full distribution
    over the vocabulary — it is only the probability mass on the true token.
    """
    sims_k = neighbor_sims[:, :k].float()                     # [N, k]
    y_k    = neighbor_y[:, :k]                                 # [N, k]

    # Softmax over top-k similarities, temperature-scaled
    weights = F.softmax(sims_k / tau, dim=-1)                  # [N, k]

    # Mask neighbors whose label matches the true next token
    y_expand = y_true.unsqueeze(1).expand_as(y_k)             # [N, k]
    match    = (y_k == y_expand).float()                       # [N, k]

    return (weights * match).sum(dim=-1)                       # [N]


def evaluate_action_for_queries(
    query_data: dict,
    neighbor_data: dict,
    action: dict,
    max_k: int = 64,
    lambda_cost: float = 0.0,
    eps: float = 1e-12,
) -> dict:
    """
    Run the gate chain for a given action on all queries.

    action = {"name": str, "k": int, "tau": float, "alpha": float}

    Returns a dict of aggregated and per-example metrics.
    """
    k     = action["k"]
    tau   = action["tau"]
    alpha = action["alpha"]

    p_gpt  = query_data["p_gpt_true"].float()   # [N]
    nll_gpt = query_data["nll_gpt"].float()      # [N]
    y_true  = query_data["y"]                    # [N]

    if k == 0:
        # GPT only: no retrieval
        nll_final = nll_gpt
        reward    = torch.zeros_like(nll_gpt)
        return {
            "mean_nll":         float(nll_final.mean()),
            "mean_reward":      float(reward.mean()),
            "mean_k":           0.0,
            "retrieval_usage":  0.0,
            "per_example_nll":  nll_final,
            "per_example_reward": reward,
        }

    # SELECT: top-k neighbor labels and similarities
    nbr_y    = neighbor_data["neighbor_y"]               # [N, max_k]
    nbr_sims = neighbor_data["neighbor_sims"]            # [N, max_k]

    # PREDICT
    p_local = local_true_probability(nbr_y, nbr_sims, y_true, k, tau)  # [N]

    # MIX
    p_final = alpha * p_gpt + (1.0 - alpha) * p_local   # [N]

    nll_final = -torch.log(p_final + eps)                # [N]

    # Reward (positive = retrieval helped reduce NLL)
    cost   = lambda_cost * (k / max_k)
    reward = nll_gpt - nll_final - cost                  # [N]

    return {
        "mean_nll":           float(nll_final.mean()),
        "mean_reward":        float(reward.mean()),
        "mean_k":             float(k),
        "retrieval_usage":    1.0,
        "per_example_nll":    nll_final,
        "per_example_reward": reward,
    }

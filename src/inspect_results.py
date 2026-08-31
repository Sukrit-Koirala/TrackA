"""
inspect_results.py

Save human-readable inspection cases for:
  1. Where controller used GPT-only
  2. Where controller used retrieval
  3. Where retrieval helped most (biggest NLL reduction)
  4. Where retrieval hurt most (biggest NLL increase)
  5. Where learned controller beat the best fixed action
  6. Where learned controller failed vs best fixed action

For retrieval cases, shows the nearest neighbor contexts and their next tokens.

Outputs:
  outputs/inspection/retrieval_used.txt
  outputs/inspection/gpt_only.txt
  outputs/inspection/biggest_help.txt
  outputs/inspection/biggest_hurt.txt
  outputs/inspection/controller_wins.txt
  outputs/inspection/controller_fails.txt
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json
import numpy as np
import torch

from utils import load_config, ensure_dirs, get_device, set_seed, build_action_grid, ppl
from gate_chain import evaluate_action_for_queries
from train_controller import ControllerMLP, build_features, apply_controller


def format_example(
    idx: int,
    val_meta: list,
    val_data: dict,
    val_nbrs: dict,
    ds_meta: list,
    chosen_action: dict,
    nll_ctrl: float,
    nll_gpt: float,
    nll_best_fixed: float,
    n_neighbor_snippets: int = 3,
) -> str:
    meta   = val_meta[idx]
    y_str  = meta["y_str"]
    ctx    = meta["context_snippet"]
    g_prob = float(val_data["p_gpt_true"][idx])
    g_nll  = float(val_data["nll_gpt"][idx])
    top1_id = int(val_data["gpt_top1_id"][idx])
    top1_p  = float(val_data["gpt_top1_prob"][idx])
    ent     = float(val_data["gpt_entropy"][idx])

    lines = [
        f"=== Example {idx} ===",
        f"Context:          …{ctx[-120:]}",
        f"True next token:  '{y_str}'",
        f"GPT top-1:        id={top1_id}  prob={top1_p:.4f}",
        f"GPT on true tok:  prob={g_prob:.4f}  NLL={g_nll:.4f}  entropy={ent:.3f}",
        f"Action chosen:    {chosen_action['name']}  "
              f"(k={chosen_action['k']}, tau={chosen_action['tau']}, alpha={chosen_action['alpha']})",
        f"Controller NLL:   {nll_ctrl:.4f}  (ΔNLL vs GPT: {g_nll - nll_ctrl:+.4f})",
        f"Best-fixed NLL:   {nll_best_fixed:.4f}",
    ]

    if chosen_action["k"] > 0:
        lines.append(f"\nNearest neighbors (top {n_neighbor_snippets}):")
        nbr_idx  = val_nbrs["neighbor_indices"][idx]
        nbr_sims = val_nbrs["neighbor_sims"][idx]
        nbr_y    = val_nbrs["neighbor_y"][idx]
        for ni in range(min(n_neighbor_snippets, chosen_action["k"])):
            ds_i    = int(nbr_idx[ni])
            sim     = float(nbr_sims[ni])
            ny_str  = ds_meta[ds_i]["y_str"] if ds_i < len(ds_meta) else "?"
            nctx    = ds_meta[ds_i]["context_snippet"][-80:] if ds_i < len(ds_meta) else "?"
            lines.append(f"  [{ni}] sim={sim:.4f}  next='{ny_str}'  ctx=…{nctx}")

    return "\n".join(lines) + "\n"


def save_cases(path: Path, cases: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {len(cases)} examples\n\n")
        f.write("\n".join(cases))
    print(f"  Saved {len(cases)} examples → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    device      = get_device(cfg)
    n_inspect   = cfg.get("n_inspect", 25)
    k           = cfg["max_k"]

    states_dir  = Path(cfg["states_dir"])
    nbrs_dir    = Path(cfg["neighbors_dir"])
    reports_dir = Path(cfg["reports_dir"])
    models_dir  = Path(cfg["models_dir"])
    insp_dir    = Path(cfg["inspection_dir"])

    print("Loading data …")
    val_data = torch.load(states_dir / "val.pt",         weights_only=False)
    ds_data  = torch.load(states_dir / "datastore.pt",   weights_only=False)
    val_nbrs = torch.load(nbrs_dir   / f"val_top{k}.pt", weights_only=False)
    val_meta = val_data["metadata"]
    ds_meta  = ds_data["metadata"]

    actions = build_action_grid(cfg)

    # ── load best fixed action ────────────────────────────────────────────────
    with open(reports_dir / "fixed_baselines.json") as f:
        baselines = json.load(f)
    best_fixed_name = baselines["best_train_selected_name"]
    best_fixed      = next(a for a in actions if a["name"] == best_fixed_name)

    # ── load controller ───────────────────────────────────────────────────────
    feat_dim = 13
    norm = torch.load(models_dir / "controller_norm.pt", map_location="cpu", weights_only=True)
    num_act = int(norm["num_actions"])
    ctrl_model = ControllerMLP(feat_dim, cfg["controller_hidden"], num_act)
    ctrl_model.register_buffer("feat_mu",  norm["feat_mu"])
    ctrl_model.register_buffer("feat_std", norm["feat_std"])
    ctrl_model.load_state_dict(
        torch.load(models_dir / "controller.pt", map_location="cpu", weights_only=True),
        strict=False,
    )
    ctrl_model = ctrl_model.to(device)

    X_val  = build_features(val_data, val_nbrs)
    chosen = apply_controller(ctrl_model, X_val, device)   # [N]

    # ── per-example NLL for controller and best-fixed ─────────────────────────
    print("Computing per-example NLLs …")
    nll_gpt = val_data["nll_gpt"].float()           # [N]

    # Controller per-example NLL
    nll_ctrl = torch.zeros(len(val_data["y"]))
    for ai, action in enumerate(actions):
        mask = (chosen == ai)
        if mask.sum() == 0:
            continue
        m = evaluate_action_for_queries(
            {k2: v[mask] if k2 != "metadata" else v for k2, v in val_data.items()},
            {k2: v[mask] for k2, v in val_nbrs.items()},
            action,
            max_k=cfg["max_k"], lambda_cost=cfg.get("lambda_cost", 0.0),
            eps=cfg.get("eps", 1e-12),
        )
        nll_ctrl[mask] = m["per_example_nll"]

    # Best-fixed per-example NLL
    m_fixed = evaluate_action_for_queries(
        val_data, val_nbrs, best_fixed,
        max_k=cfg["max_k"], lambda_cost=cfg.get("lambda_cost", 0.0),
        eps=cfg.get("eps", 1e-12),
    )
    nll_fixed = m_fixed["per_example_nll"]

    # ── helper: format + collect ───────────────────────────────────────────────
    def fmt(i):
        act = actions[int(chosen[i])]
        return format_example(i, val_meta, val_data, val_nbrs, ds_meta,
                               act, float(nll_ctrl[i]), float(nll_gpt[i]),
                               float(nll_fixed[i]))

    def top_n(scores, n, ascending=True):
        arr = scores.numpy()
        order = np.argsort(arr) if ascending else np.argsort(arr)[::-1]
        return order[:n].tolist()

    # ── retrieval used ────────────────────────────────────────────────────────
    ret_mask = torch.tensor([actions[int(c)]["k"] > 0 for c in chosen])
    ret_idx  = ret_mask.nonzero().squeeze(-1).tolist()[:n_inspect]
    save_cases(insp_dir / "retrieval_used.txt", [fmt(i) for i in ret_idx])

    # ── GPT-only ──────────────────────────────────────────────────────────────
    gpt_idx = (~ret_mask).nonzero().squeeze(-1).tolist()[:n_inspect]
    save_cases(insp_dir / "gpt_only.txt", [fmt(i) for i in gpt_idx])

    # ── biggest help (retrieval vs GPT-only) ─────────────────────────────────
    # "help" = GPT NLL - controller NLL (large positive = retrieval helped a lot)
    improvement = nll_gpt - nll_ctrl
    help_idx = top_n(-improvement, n_inspect)  # descending improvement
    save_cases(insp_dir / "biggest_help.txt", [fmt(i) for i in help_idx])

    # ── biggest hurt ──────────────────────────────────────────────────────────
    hurt_idx = top_n(improvement, n_inspect)   # ascending = most negative improvement
    save_cases(insp_dir / "biggest_hurt.txt", [fmt(i) for i in hurt_idx])

    # ── controller beats best fixed ───────────────────────────────────────────
    ctrl_vs_fixed = nll_fixed - nll_ctrl                # positive → controller wins
    win_idx = top_n(-ctrl_vs_fixed, n_inspect)
    save_cases(insp_dir / "controller_wins.txt", [fmt(i) for i in win_idx])

    # ── controller fails vs best fixed ───────────────────────────────────────
    fail_idx = top_n(ctrl_vs_fixed, n_inspect)
    save_cases(insp_dir / "controller_fails.txt", [fmt(i) for i in fail_idx])

    print(f"\nSummary:")
    print(f"  Retrieval used:    {ret_mask.sum().item()} / {len(ret_mask)} examples")
    print(f"  GPT-only:          {(~ret_mask).sum().item()} examples")
    print(f"  Ctrl beats fixed:  {(ctrl_vs_fixed > 0).sum().item()} examples")
    print(f"  Ctrl fails fixed:  {(ctrl_vs_fixed < 0).sum().item()} examples")
    print("\nDone.")


if __name__ == "__main__":
    main()

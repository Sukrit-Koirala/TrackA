"""
inspect_results_v2.py

Per-method inspection for all controllers.

For each method, produces outputs/inspection_v2/<method>/:
  retrieval_used.txt
  gpt_only.txt
  biggest_help_vs_gpt.txt
  biggest_hurt_vs_gpt.txt
  wins_vs_best_fixed.txt
  fails_vs_best_fixed.txt
  high_confidence_wrong.txt   (high GPT top1_prob yet high NLL — potential retrieval targets)

Each example entry shows:
  context snippet, true next token, GPT top-1, GPT prob on true token,
  chosen action, best fixed action, controller NLL, GPT NLL, best fixed NLL,
  ΔNLL vs GPT, ΔNLL vs best fixed, top-3 neighbor similarities + next tokens.
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
from train_controller import build_features, ControllerMLP, apply_controller
from train_q_controller import (
    build_all_action_features, apply_q_controller, load_q_model,
)


N_INSPECT = 25


# ── formatting ─────────────────────────────────────────────────────────────────

def format_example(
    idx: int,
    val_data: dict,
    val_nbrs: dict,
    ds_meta: list,
    chosen_action: dict,
    best_fixed_action: dict,
    nll_ctrl: float,
    nll_fixed: float,
    n_nbr: int = 3,
) -> str:
    meta    = val_data["metadata"][idx]
    y_str   = meta["y_str"]
    ctx     = meta["context_snippet"]
    g_prob  = float(val_data["p_gpt_true"][idx])
    g_nll   = float(val_data["nll_gpt"][idx])
    top1_id = int(val_data["gpt_top1_id"][idx])
    top1_p  = float(val_data["gpt_top1_prob"][idx])
    ent     = float(val_data["gpt_entropy"][idx])

    lines = [
        f"=== Example {idx} ===",
        f"Context:         …{ctx[-120:]}",
        f"True next token: '{y_str}'",
        f"GPT top-1:       id={top1_id}  prob={top1_p:.4f}",
        f"GPT on true tok: prob={g_prob:.4f}  NLL={g_nll:.4f}  entropy={ent:.3f}",
        f"Action chosen:   {chosen_action['name']}  "
              f"(k={chosen_action['k']}, tau={chosen_action['tau']}, alpha={chosen_action['alpha']})",
        f"Best fixed:      {best_fixed_action['name']}",
        f"NLL — ctrl:{nll_ctrl:.4f}  gpt:{g_nll:.4f}  fixed:{nll_fixed:.4f}  "
              f"Δgpt:{g_nll-nll_ctrl:+.4f}  Δfixed:{nll_fixed-nll_ctrl:+.4f}",
    ]

    if chosen_action["k"] > 0:
        lines.append(f"\nNearest neighbors (top {n_nbr}):")
        nbr_idx  = val_nbrs["neighbor_indices"][idx]
        nbr_sims = val_nbrs["neighbor_sims"][idx]
        nbr_y    = val_nbrs["neighbor_y"][idx]
        for ni in range(min(n_nbr, chosen_action["k"])):
            ds_i   = int(nbr_idx[ni])
            sim    = float(nbr_sims[ni])
            ny_str = ds_meta[ds_i]["y_str"]   if ds_i < len(ds_meta) else "?"
            nctx   = ds_meta[ds_i]["context_snippet"][-80:] if ds_i < len(ds_meta) else "?"
            lines.append(f"  [{ni}] sim={sim:.4f}  next='{ny_str}'  ctx=…{nctx}")

    return "\n".join(lines) + "\n"


def save_cases(path: Path, cases: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {len(cases)} examples\n\n")
        f.write("\n".join(cases))
    print(f"    {path.name}: {len(cases)} examples")


def top_n_indices(scores: np.ndarray, n: int, ascending: bool = True) -> list[int]:
    order = np.argsort(scores) if ascending else np.argsort(scores)[::-1]
    return order[:n].tolist()


# ── per-method inspection ──────────────────────────────────────────────────────

def run_inspection(
    method_name: str,
    chosen: torch.Tensor,           # [N_val] action indices
    actions_subset: list[dict],     # the action set this controller uses
    val_data: dict,
    val_nbrs: dict,
    ds_meta: list,
    best_fixed_action: dict,
    nll_fixed: torch.Tensor,        # [N_val] best-fixed per-example NLL
    cfg: dict,
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    N = len(val_data["y"])
    n = min(N_INSPECT, N)

    # Per-example controller NLL
    nll_ctrl = torch.zeros(N)
    for ai, action in enumerate(actions_subset):
        mask = (chosen == ai)
        if mask.sum() == 0:
            continue
        m = evaluate_action_for_queries(
            {k2: v[mask] if k2 != "metadata" else v for k2, v in val_data.items()},
            {k2: v[mask] for k2, v in val_nbrs.items()},
            action,
            max_k=cfg["max_k"], lambda_cost=0.0, eps=cfg.get("eps", 1e-12),
        )
        nll_ctrl[mask] = m["per_example_nll"]

    nll_gpt     = val_data["nll_gpt"].float()
    improvement = nll_gpt - nll_ctrl       # positive = ctrl better than GPT
    ctrl_vs_fix = nll_fixed - nll_ctrl     # positive = ctrl better than fixed

    def fmt(i: int) -> str:
        act = actions_subset[int(chosen[i])]
        return format_example(
            i, val_data, val_nbrs, ds_meta,
            chosen_action=act,
            best_fixed_action=best_fixed_action,
            nll_ctrl=float(nll_ctrl[i]),
            nll_fixed=float(nll_fixed[i]),
        )

    # retrieval used / gpt-only
    ret_mask = torch.tensor([actions_subset[int(c)]["k"] > 0 for c in chosen])
    save_cases(out_dir / "retrieval_used.txt",
               [fmt(i) for i in ret_mask.nonzero().squeeze(-1).tolist()[:n]])
    save_cases(out_dir / "gpt_only.txt",
               [fmt(i) for i in (~ret_mask).nonzero().squeeze(-1).tolist()[:n]])

    # biggest help/hurt vs GPT
    save_cases(out_dir / "biggest_help_vs_gpt.txt",
               [fmt(i) for i in top_n_indices(-improvement.numpy(), n)])
    save_cases(out_dir / "biggest_hurt_vs_gpt.txt",
               [fmt(i) for i in top_n_indices(improvement.numpy(), n)])

    # controller vs best fixed
    save_cases(out_dir / "wins_vs_best_fixed.txt",
               [fmt(i) for i in top_n_indices(-ctrl_vs_fix.numpy(), n)])
    save_cases(out_dir / "fails_vs_best_fixed.txt",
               [fmt(i) for i in top_n_indices(ctrl_vs_fix.numpy(), n)])

    # high-confidence wrong (top1_prob high but nll_gpt high — potential retrieval targets)
    top1_prob  = val_data["gpt_top1_prob"].float().numpy()
    # "wrong" = GPT NLL high; "confident" = GPT top1_prob high (on something else)
    confidence_wrong = top1_prob - nll_gpt.numpy()   # high when confident but wrong
    save_cases(out_dir / "high_confidence_wrong.txt",
               [fmt(i) for i in top_n_indices(-confidence_wrong, n)])

    mean_nll = float(nll_ctrl.mean())
    print(f"  {method_name}: val_NLL={mean_nll:.4f}  "
          f"ret={ret_mask.float().mean()*100:.0f}%  "
          f"ctrl_beats_fixed={int((ctrl_vs_fix>0).sum())}/{N}")


def main():
    parser = argparse.ArgumentParser(description="Inspection reports for all controllers v2")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    ensure_dirs(cfg)
    device = get_device(cfg)
    set_seed(cfg.get("seed", 42))

    states_dir  = Path(cfg["states_dir"])
    nbrs_dir    = Path(cfg["neighbors_dir"])
    reports_dir = Path(cfg["reports_dir"])
    models_dir  = Path(cfg["models_dir"])
    insp_dir    = Path(cfg.get("inspection_v2_dir", "outputs/inspection_v2"))
    k_max       = cfg["max_k"]

    print("Loading data …")
    val_data = torch.load(states_dir / "val.pt",           weights_only=False)
    ds_data  = torch.load(states_dir / "datastore.pt",     weights_only=False)
    val_nbrs = torch.load(nbrs_dir   / f"val_top{k_max}.pt", weights_only=False)
    ds_meta  = ds_data["metadata"]

    actions       = build_action_grid(cfg)
    all_act_feats = build_all_action_features(actions, k_max)
    name_to_idx   = {a["name"]: i for i, a in enumerate(actions)}

    # ── best fixed action ──────────────────────────────────────────────────────
    with open(reports_dir / "fixed_baselines.json") as f:
        baselines = json.load(f)
    best_fixed_name   = baselines["best_train_selected_name"]
    best_fixed_action = actions[name_to_idx[best_fixed_name]]

    # Pre-compute best-fixed per-example NLL on val (used by all methods)
    m_fixed  = evaluate_action_for_queries(
        val_data, val_nbrs, best_fixed_action,
        max_k=k_max, lambda_cost=0.0, eps=cfg.get("eps", 1e-12),
    )
    nll_fixed = m_fixed["per_example_nll"]   # [N_val]

    # Observation features for val
    X_val = build_features(val_data, val_nbrs)

    print("\nRunning inspection per method …")

    # ── best fixed (constant action) ───────────────────────────────────────────
    print("\n[best_fixed]")
    N_val = len(val_data["y"])
    chosen_fixed = torch.full((N_val,), fill_value=name_to_idx[best_fixed_name],
                              dtype=torch.long)
    run_inspection("best_fixed", chosen_fixed, actions, val_data, val_nbrs,
                   ds_meta, best_fixed_action, nll_fixed, cfg,
                   insp_dir / "best_fixed")

    # ── old classifier MLP ────────────────────────────────────────────────────
    ctrl_pt = models_dir / "controller.pt"
    if ctrl_pt.exists():
        print("\n[old_classifier_mlp]")
        norm     = torch.load(models_dir / "controller_norm.pt",
                              map_location="cpu", weights_only=False)
        num_act  = int(norm["num_actions"])
        old_ctrl = ControllerMLP(13, cfg["controller_hidden"], num_act)
        old_ctrl.register_buffer("feat_mu",  norm["feat_mu"])
        old_ctrl.register_buffer("feat_std", norm["feat_std"])
        old_ctrl.load_state_dict(
            torch.load(ctrl_pt, map_location="cpu", weights_only=True), strict=False)
        old_ctrl = old_ctrl.to(device)
        chosen_old  = apply_controller(old_ctrl, X_val, device)
        # Map indices back to full action list
        old_actions = [actions[i] for i in range(num_act)]
        run_inspection("old_classifier_mlp", chosen_old, old_actions,
                       val_data, val_nbrs, ds_meta, best_fixed_action, nll_fixed,
                       cfg, insp_dir / "old_classifier_mlp")
    else:
        print("  [old_classifier_mlp] controller.pt not found — skipping")

    # ── Q-MLP variants ────────────────────────────────────────────────────────
    for set_name in ("full", "A", "B", "C"):
        q_pt = models_dir / f"q_mlp_{set_name}.pt"
        if q_pt.exists():
            print(f"\n[Q-MLP-{set_name}]")
            q_model, action_names = load_q_model(q_pt, device)
            actions_sub  = [actions[name_to_idx[n]] for n in action_names
                            if n in name_to_idx]
            act_feats_sub = all_act_feats[[name_to_idx[n] for n in action_names
                                           if n in name_to_idx]]
            chosen_q = apply_q_controller(q_model, X_val, act_feats_sub, device)
            run_inspection(f"Q-MLP-{set_name}", chosen_q, actions_sub,
                           val_data, val_nbrs, ds_meta, best_fixed_action, nll_fixed,
                           cfg, insp_dir / f"q_mlp_{set_name}")
        else:
            print(f"  [Q-MLP-{set_name}] q_mlp_{set_name}.pt not found — "
                  f"run train_q_controller.py first")

    # ── best extended heuristic ────────────────────────────────────────────────
    heur_json = reports_dir / "best_heuristics.json"
    if heur_json.exists():
        print("\n[best_heuristic]")
        with open(heur_json) as f:
            bh = json.load(f)
        fi        = bh["feature_idx"]
        direction = bh["direction"]
        threshold = bh["threshold"]
        ret_name  = bh["retrieval_action"]
        ret_act   = actions[name_to_idx[ret_name]]
        ret_idx   = name_to_idx[ret_name]

        feat_val  = X_val[:, fi].numpy()
        mask      = (feat_val > threshold) if direction == "gt" else (feat_val < threshold)
        gpt_idx   = name_to_idx["gpt_only"]

        chosen_h  = torch.where(
            torch.from_numpy(mask),
            torch.full((N_val,), ret_idx,  dtype=torch.long),
            torch.full((N_val,), gpt_idx, dtype=torch.long),
        )
        run_inspection("best_heuristic", chosen_h, actions,
                       val_data, val_nbrs, ds_meta, best_fixed_action, nll_fixed,
                       cfg, insp_dir / "best_heuristic")
    else:
        print("  [best_heuristic] best_heuristics.json not found — "
              "run run_heuristics.py first")

    print("\nDone.  Inspection files in:", insp_dir)


if __name__ == "__main__":
    main()

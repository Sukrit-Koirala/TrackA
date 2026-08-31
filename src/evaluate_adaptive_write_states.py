"""
evaluate_adaptive_write_states.py  --  MVP 3a eval

Runs the MVP 2c Q-state-read pipeline on each adaptive-write state file.

Imports from train_q_state_read.py (no modification to existing scripts).

For each state .pt file in states_dir:
  1. Compute top-32 state neighbors for CT and val
  2. Precompute token lookups
  3. Build observation features
  4. Compute reward matrices (N x A)
  5. Train Q-MLP
  6. Eval fixed + Q-read on val
  7. Save to out/q_read/<tag>/

Usage:
  python branch_a_gate_mvp/src/evaluate_adaptive_write_states.py \\
    --source  branch_a_gate_mvp/outputs_scale_sweep/scale_200k_seed42 \\
    --states_dir branch_a_gate_mvp/outputs_mvp3a_adaptive_write/states \\
    --output  branch_a_gate_mvp/outputs_mvp3a_adaptive_write \\
    --fast_action_grid --max_q_samples 300000
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse, json, time
import torch

from utils import get_device, set_seed, ppl
from evaluate_predictive_states import (
    compute_top_k_state_sims,
    precompute_lookup,
    build_global_freq,
)
from train_q_state_read import (
    build_action_grid,
    build_action_features,
    build_obs_features,
    compute_reward_matrix,
    find_best_fixed,
    compute_oracle,
    QStateReadMLP, Q_DIM,
    train_q_mlp,
    apply_q_controller,
    eval_fixed_on_val,
    eval_q_on_val,
    action_distribution_df,
    write_inspection_files,
)

MAX_K  = 32
VOCAB  = 50257


def load_full_ds_baselines(src: Path) -> dict:
    out = {"gpt": 2.5533, "fixed": 2.3873, "qmlp": 2.2977}
    bl  = src / "reports" / "fixed_baselines.json"
    if bl.exists():
        with open(bl) as f:
            d = json.load(f)
        out["gpt"]   = d.get("gpt_only_val_nll",           out["gpt"])
        out["fixed"] = d.get("best_train_selected_val_nll", out["fixed"])
    qm = src / "reports" / "q_controller_metrics.json"
    if qm.exists():
        with open(qm) as f:
            d = json.load(f)
        if "Q-MLP-full" in d:
            out["qmlp"] = d["Q-MLP-full"].get("mean_nll", out["qmlp"])
    return out


def eval_state_file(
    state_path: Path,
    src: Path,
    out_base: Path,
    device,
    max_q_samples: int = 800_000,
    n_epochs: int = 30,
    fast_action_grid: bool = False,
    force_neighbors: bool = False,
    force_rewards: bool = False,
    force_train: bool = False,
    force_eval: bool = False,
    seed: int = 42,
) -> dict | None:
    tag     = state_path.stem
    mb_out  = out_base / "q_read" / tag
    nbr_dir = out_base / "state_neighbors"
    mb_out.mkdir(parents=True, exist_ok=True)
    nbr_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = mb_out / "q_metrics.json"
    if metrics_path.exists() and not force_eval and not force_train \
            and not force_rewards and not force_neighbors:
        with open(metrics_path) as f:
            m = json.load(f)
        print(f"  [cached] {tag}  q_nll={m.get('q_state_nll','?'):.4f}")
        return m

    print(f"\n  Evaluating: {tag}")
    t0 = time.time()

    # ── load data ────────────────────────────────────────────────────────────
    ct_data  = torch.load(src / "states" / "controller_train.pt", weights_only=False)
    val_data = torch.load(src / "states" / "val.pt",              weights_only=False)
    ds_data  = torch.load(src / "states" / "datastore.pt",        weights_only=False)
    states   = torch.load(state_path, weights_only=False)
    full_bl  = load_full_ds_baselines(src)
    P_global = build_global_freq(ds_data["y"], VOCAB)

    def norm_h(data):
        h = data["h"].float()
        return h / (h.norm(dim=-1, keepdim=True) + 1e-8)

    ct_h  = norm_h(ct_data)
    val_h = norm_h(val_data)
    proto = states["prototype_h"].float()   # [B, D]  already normalised
    B     = len(proto)
    print(f"    States: B={B:,}  (file: {state_path.name})")

    # ── neighbors ────────────────────────────────────────────────────────────
    ct_nbr_path  = nbr_dir / f"{tag}_ct_top{MAX_K}.pt"
    val_nbr_path = nbr_dir / f"{tag}_val_top{MAX_K}.pt"

    if ct_nbr_path.exists() and not force_neighbors:
        ct_nbrs = torch.load(ct_nbr_path, weights_only=False)
        ct_ids, ct_sims = ct_nbrs["ids"], ct_nbrs["sims"]
        print(f"    CT neighbors loaded from cache")
    else:
        print(f"    Computing CT top-{MAX_K} neighbors ...")
        ct_ids, ct_sims = compute_top_k_state_sims(ct_h, proto, MAX_K, device)
        torch.save({"ids": ct_ids, "sims": ct_sims}, ct_nbr_path)

    if val_nbr_path.exists() and not force_neighbors:
        val_nbrs = torch.load(val_nbr_path, weights_only=False)
        val_ids, val_sims = val_nbrs["ids"], val_nbrs["sims"]
        print(f"    Val neighbors loaded from cache")
    else:
        print(f"    Computing val top-{MAX_K} neighbors ...")
        val_ids, val_sims = compute_top_k_state_sims(val_h, proto, MAX_K, device)
        torch.save({"ids": val_ids, "sims": val_sims}, val_nbr_path)

    # ── precompute + obs features ─────────────────────────────────────────────
    ct_precomp  = precompute_lookup(ct_data,  ct_ids,  ct_sims,  states, P_global)
    val_precomp = precompute_lookup(val_data, val_ids, val_sims, states, P_global)
    obs_ct      = build_obs_features(ct_data,  ct_ids,  ct_sims,  states)
    obs_val     = build_obs_features(val_data, val_ids, val_sims, states)

    actions   = build_action_grid(fast=fast_action_grid)
    act_feats = build_action_features(actions)
    A         = len(actions)
    print(f"    Action grid: {A} actions (fast={fast_action_grid})")

    # ── reward matrices ───────────────────────────────────────────────────────
    ct_rew_path  = mb_out / "ct_rewards.pt"
    val_rew_path = mb_out / "val_rewards.pt"

    if ct_rew_path.exists() and not force_rewards:
        ct_rewards  = torch.load(ct_rew_path,  weights_only=False)
        val_rewards = torch.load(val_rew_path, weights_only=False)
        print(f"    Rewards loaded from cache")
    else:
        print(f"    Computing rewards ...")
        ct_rewards  = compute_reward_matrix(ct_precomp,  actions)
        val_rewards = compute_reward_matrix(val_precomp, actions)
        torch.save(ct_rewards,  ct_rew_path)
        torch.save(val_rewards, val_rew_path)

    # ── best fixed ────────────────────────────────────────────────────────────
    best_ai, best_act = find_best_fixed(ct_rewards, actions)
    fixed_m   = eval_fixed_on_val(best_ai, val_rewards, val_precomp)
    fixed_nll = fixed_m["mean_nll"]

    # ── oracle ────────────────────────────────────────────────────────────────
    oracle_nll, oracle_nlls = compute_oracle(val_precomp["nll_gpt"], val_rewards)

    # ── Q-MLP ─────────────────────────────────────────────────────────────────
    q_path = mb_out / "q_model.pt"
    if q_path.exists() and not force_train:
        print(f"    Loading Q model ...")
        ckpt  = torch.load(q_path, weights_only=False)
        model = QStateReadMLP(Q_DIM)
        model.load_state_dict(ckpt["state_dict"])
        model.x_mu  = ckpt["x_mu"]
        model.x_std = ckpt["x_std"]
        model.y_mu  = ckpt["y_mu"]
        model.y_std = ckpt["y_std"]
        q_train_m   = ckpt.get("train_metrics", {})
    else:
        print(f"    Training Q-MLP  samples={max_q_samples:,} ...")
        set_seed(seed)
        model, q_train_m = train_q_mlp(
            obs_ct, act_feats, ct_rewards, device,
            max_q_samples=max_q_samples, n_epochs=n_epochs,
        )
        torch.save({
            "state_dict":    model.state_dict(),
            "x_mu":          model.x_mu,
            "x_std":         model.x_std,
            "y_mu":          model.y_mu,
            "y_std":         model.y_std,
            "train_metrics": q_train_m,
            "actions":       actions,
            "tag":           tag,
        }, q_path)

    # ── Q eval ────────────────────────────────────────────────────────────────
    print(f"    Evaluating Q on val ...")
    q_m   = eval_q_on_val(model, obs_val, act_feats, val_rewards,
                           val_precomp, device, actions)
    q_nll = q_m["mean_nll"]
    print(f"    fixed={fixed_nll:.4f}  Q={q_nll:.4f}  oracle={oracle_nll:.4f}  "
          f"(GPT={float(val_precomp['nll_gpt'].mean()):.4f})")

    # ── save predictions ──────────────────────────────────────────────────────
    torch.save({
        "q_nlls":     q_m["per_example_nll"],
        "fixed_nlls": fixed_m["per_example_nll"],
        "oracle_nlls": oracle_nlls,
        "chosen":     q_m["chosen"],
    }, mb_out / "val_predictions.pt")

    act_df = action_distribution_df(q_m["chosen"], actions)
    act_df.to_csv(mb_out / "action_distribution.csv", index=False)
    top_acts = act_df.head(2)[["name", "frac"]].to_dict("records") if not act_df.empty else []

    # ── inspection ────────────────────────────────────────────────────────────
    insp_dir = out_base / "inspection" / tag
    write_inspection_files(
        insp_dir, val_data, val_precomp,
        val_ids, val_sims, states,
        q_m["per_example_nll"], fixed_m["per_example_nll"],
        q_m["chosen"], best_ai, actions,
    )

    # ── metrics ───────────────────────────────────────────────────────────────
    metrics = {
        "tag":              tag,
        "state_file":       str(state_path),
        "num_states":       B,
        "action_grid_size": A,
        "best_fixed_action": best_act["name"],
        "best_fixed_nll":   fixed_nll,
        "q_state_nll":      q_nll,
        "oracle_nll":       oracle_nll,
        "gpt_nll":          float(val_precomp["nll_gpt"].mean()),
        "full_ds_gpt_nll":  full_bl["gpt"],
        "full_ds_fixed_nll": full_bl["fixed"],
        "full_ds_qmlp_nll": full_bl["qmlp"],
        "delta_q_vs_fixed": q_nll - fixed_nll,
        "delta_q_vs_full_ds_qmlp": q_nll - full_bl["qmlp"],
        "delta_q_vs_full_ds_fixed": q_nll - full_bl["fixed"],
        "avg_k_states":     q_m["avg_k_states"],
        "retrieval_usage":  q_m["retrieval_usage"],
        "oracle_gap_vs_fixed": oracle_nll - fixed_nll,
        "oracle_gap_vs_q":  oracle_nll - q_nll,
        "top_actions":      top_acts,
        "q_train_metrics":  q_train_m,
        "elapsed_s":        time.time() - t0,
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",           required=True)
    parser.add_argument("--states_dir",       required=True)
    parser.add_argument("--output",           required=True)
    parser.add_argument("--max_q_samples",    type=int, default=800_000)
    parser.add_argument("--n_epochs",         type=int, default=30)
    parser.add_argument("--fast_action_grid", action="store_true")
    parser.add_argument("--force_neighbors",  action="store_true")
    parser.add_argument("--force_rewards",    action="store_true")
    parser.add_argument("--force_train",      action="store_true")
    parser.add_argument("--force_eval",       action="store_true")
    parser.add_argument("--force",            action="store_true",
                        help="Force all stages")
    parser.add_argument("--tags",   nargs="*", default=None,
                        help="Restrict to specific state file tags")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    device = get_device({"device": args.device})
    src    = Path(args.source)
    sdir   = Path(args.states_dir)
    out    = Path(args.output)

    if args.force:
        args.force_neighbors = args.force_rewards = args.force_train = args.force_eval = True

    # Find state files
    state_files = sorted(sdir.glob("*.pt"))
    if args.tags:
        state_files = [f for f in state_files if f.stem in args.tags]
    if not state_files:
        print(f"No state files found in {sdir}")
        return

    print(f"\nevaluate_adaptive_write_states")
    print(f"  Source:   {src}")
    print(f"  States:   {sdir}  ({len(state_files)} files)")
    print(f"  Output:   {out}")
    print(f"  Samples:  {args.max_q_samples:,}  fast_grid={args.fast_action_grid}")
    print(f"  Device:   {device}")

    all_metrics = []
    for i, sf in enumerate(state_files):
        print(f"\n[{i+1}/{len(state_files)}] {sf.stem}")
        m = eval_state_file(
            sf, src, out, device,
            max_q_samples   = args.max_q_samples,
            n_epochs        = args.n_epochs,
            fast_action_grid= args.fast_action_grid,
            force_neighbors = args.force_neighbors,
            force_rewards   = args.force_rewards,
            force_train     = args.force_train,
            force_eval      = args.force_eval,
            seed            = args.seed,
        )
        if m:
            all_metrics.append(m)

    print(f"\n{'='*55}")
    print(f"Evaluated {len(all_metrics)} state files")
    if all_metrics:
        best = min(all_metrics, key=lambda x: x["q_state_nll"])
        print(f"Best Q NLL: {best['q_state_nll']:.4f}  ({best['tag']})")


if __name__ == "__main__":
    main()

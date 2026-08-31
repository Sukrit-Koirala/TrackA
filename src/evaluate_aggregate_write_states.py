"""
evaluate_aggregate_write_states.py  --  MVP 3b eval

Stage 1: cheap health check before Q-training
Stage 2: Q-state-read evaluation (skipped if health fails unless --eval_all)

Reuses MVP 2c Q-state-read code unchanged.

Usage:
  python branch_a_gate_mvp/src/evaluate_aggregate_write_states.py \\
    --source  branch_a_gate_mvp/outputs_scale_sweep/scale_200k_seed42 \\
    --states_dir branch_a_gate_mvp/outputs_mvp3b_aggregate_write/states \\
    --output  branch_a_gate_mvp/outputs_mvp3b_aggregate_write \\
    --fast_action_grid --max_q_samples 300000
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse, json, time
import numpy as np
import torch
import pandas as pd

from utils import get_device, set_seed
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

MAX_K = 32
VOCAB = 50257
EPS   = 1e-8

# Health thresholds: configs failing these are skipped for Q-eval
HEALTH_MIN_HIT4  = 0.70     # true-token hit@4 must be >= 70%
HEALTH_MAX_SING  = 0.40     # singleton fraction must be < 40%
HEALTH_MIN_MED   = 2        # median state count must be >= 2
GPT_ONLY_NLL     = 2.5533
HEALTH_MAX_NLL_DELTA = 0.05  # fixed local NLL must not exceed GPT by more than 0.05


def _norm(h: torch.Tensor) -> torch.Tensor:
    return h / (h.norm(dim=-1, keepdim=True) + EPS)


def compute_state_health(state_path: Path, src: Path, device) -> dict:
    """Cheap per-state health check that does not require Q-training."""
    states   = torch.load(state_path, weights_only=False)
    val_data = torch.load(src / "states" / "val.pt", weights_only=False)

    B  = int(states.get("actual_num_states", len(states["prototype_h"])))
    tc = states["total_counts"].numpy()[:B]
    ent = states["state_entropy"].numpy()[:B]
    pur = states["state_purity"].numpy()[:B]
    budget = int(states.get("budget", B))

    # Basic state stats
    med_count  = float(np.median(tc))
    mean_count = float(np.mean(tc))
    frac_le1   = float((tc <= 1).mean())
    frac_le5   = float((tc <= 5).mean())
    mean_ent   = float(ent.mean())
    mean_pur   = float(pur.mean())

    # True-token hit rates via GPU matmul
    val_h = _norm(val_data["h"].float())
    val_y = val_data["y"].numpy().astype(np.int64)
    proto = states["prototype_h"].float()[:B].to(device)
    tok_ids_t = states["top_k_token_ids"][:B].to(device)    # [B, TOP_K]
    tok_cnt_t = states["top_k_token_counts"][:B].to(device)  # [B, TOP_K]
    N_val     = len(val_y)

    hit = {1: 0, 4: 0, 8: 0}
    p_true_sum = 0.0
    p_true_zero = 0
    local_nll_k4_sum = 0.0

    # Build global freq for mixing
    ds_data  = torch.load(src / "states" / "datastore.pt", weights_only=False)
    P_global = build_global_freq(ds_data["y"], VOCAB)
    P_global_t = torch.tensor(P_global, dtype=torch.float32, device=device)

    CHUNK = 512
    val_y_t = torch.tensor(val_y, dtype=torch.long, device=device)

    for start in range(0, N_val, CHUNK):
        end    = min(start + CHUNK, N_val)
        h_blk  = val_h[start:end].to(device)           # [C, D]
        y_blk  = val_y_t[start:end]                    # [C]
        C      = end - start

        sims = h_blk @ proto.T                         # [C, B]

        # hit@1
        top1_ids = sims.argmax(dim=1)                  # [C]
        top1_tok  = tok_ids_t[top1_ids]               # [C, TOP_K]
        top1_cnt  = tok_cnt_t[top1_ids]               # [C, TOP_K]
        top1_tot  = states["total_counts"][:B].to(device)[top1_ids]  # [C]
        y_exp     = y_blk.view(-1, 1).expand(-1, tok_ids_t.shape[1])
        hit[1]   += int((top1_tok == y_exp).any(dim=1).sum())

        # hit@4
        top4_ids = sims.topk(min(4, B), dim=1).indices   # [C, 4]
        top4_tok  = tok_ids_t[top4_ids.reshape(-1)].reshape(C, -1, tok_ids_t.shape[1])
        y_exp4    = y_blk.view(-1, 1, 1).expand_as(top4_tok)
        hit[4]   += int((top4_tok == y_exp4).any(dim=-1).any(dim=-1).sum())

        # hit@8
        top8_ids = sims.topk(min(8, B), dim=1).indices
        top8_tok  = tok_ids_t[top8_ids.reshape(-1)].reshape(C, -1, tok_ids_t.shape[1])
        y_exp8    = y_blk.view(-1, 1, 1).expand_as(top8_tok)
        hit[8]   += int((top8_tok == y_exp8).any(dim=-1).any(dim=-1).sum())

        # p_state(true_token) at nearest state
        raw_cnt = (top1_cnt * (top1_tok == y_exp).float()).sum(dim=1)   # [C]
        p_true  = raw_cnt / (top1_tot.float() + EPS)                    # [C]
        p_true_sum  += float(p_true.sum())
        p_true_zero += int((p_true == 0).sum())

        # Local NLL: k=4, alpha=0.5, beta=0, tau=0.05
        # For each of the k nearest states, find the count for the true token y,
        # then sum counts across states and divide by total count across states.
        k4 = min(4, B)
        topk4_ids = sims.topk(k4, dim=1).indices                             # [C, k4]
        k4_tok    = tok_ids_t[topk4_ids.reshape(-1)].reshape(C, k4, -1)     # [C, k4, TOP_K]
        k4_cnt    = tok_cnt_t[topk4_ids.reshape(-1)].reshape(C, k4, -1)     # [C, k4, TOP_K]
        k4_tot    = states["total_counts"][:B].to(device)[topk4_ids]        # [C, k4]
        # per-state count of true token, then sum across k4 states
        y_exp_k4   = y_blk.view(-1, 1, 1).expand(-1, k4, k4_tok.shape[-1]) # [C, k4, TOP_K]
        match_cnt  = (k4_cnt * (k4_tok == y_exp_k4).float()).sum(dim=-1)    # [C, k4]
        total_y    = match_cnt.sum(dim=1)                                    # [C]
        total_all  = k4_tot.sum(dim=1)                                       # [C]
        p_state_y  = total_y / (total_all + EPS)                            # [C]

        # Mix: alpha=0.5 * p_state + 0.5 * p_global
        alpha   = 0.5
        p_glo_y = P_global_t[y_blk]                                     # [C]
        p_mix   = alpha * p_state_y + (1.0 - alpha) * p_glo_y
        p_mix   = p_mix.clamp(min=1e-10)
        local_nll_k4_sum += float(-p_mix.log().sum())

    hit_at_1 = hit[1] / N_val
    hit_at_4 = hit[4] / N_val
    hit_at_8 = hit[8] / N_val
    mean_p_true  = p_true_sum / N_val
    frac_p_zero  = p_true_zero / N_val
    fixed_local_nll_k4 = local_nll_k4_sum / N_val

    # Buffer and commit stats from state file
    n_promoted = int(states.get("num_buffers_promoted", 0))
    n_dropped  = int(states.get("num_buffers_dropped",  0))
    n_final_buf= int(states.get("num_buffers_final",    0))
    n_direct   = int(states.get("num_direct_updates",   0))
    n_buf_upd  = int(states.get("num_buffer_updates",   0))
    n_buf_cre  = int(states.get("num_buffer_creates",   0))

    health = {
        "tag":                    state_path.stem,
        "budget":                 budget,
        "actual_num_states":      B,
        "budget_usage_fraction":  B / max(budget, 1),
        "median_state_count":     med_count,
        "mean_state_count":       mean_count,
        "frac_count_le_1":        frac_le1,
        "frac_count_le_5":        frac_le5,
        "mean_entropy":           mean_ent,
        "mean_purity":            mean_pur,
        "hit_at_1":               hit_at_1,
        "hit_at_4":               hit_at_4,
        "hit_at_8":               hit_at_8,
        "mean_p_state_true_top1": mean_p_true,
        "frac_zero_p_true":       frac_p_zero,
        "fixed_local_nll_k4":     fixed_local_nll_k4,
        "num_buffers_promoted":   n_promoted,
        "num_buffers_dropped":    n_dropped,
        "num_buffers_final":      n_final_buf,
        "num_direct_updates":     n_direct,
        "num_buffer_updates":     n_buf_upd,
        "num_buffer_creates":     n_buf_cre,
        "direct_fraction":        n_direct / max(n_direct + n_buf_upd + n_buf_cre, 1),
    }
    return health


def passes_health_check(h: dict) -> tuple:
    """Returns (passes: bool, reason: str)."""
    if h["hit_at_4"] < HEALTH_MIN_HIT4:
        return False, f"hit@4={h['hit_at_4']:.2%} < {HEALTH_MIN_HIT4:.0%}"
    if h["frac_count_le_1"] > HEALTH_MAX_SING:
        return False, f"singleton_frac={h['frac_count_le_1']:.2%} > {HEALTH_MAX_SING:.0%}"
    if h["median_state_count"] < HEALTH_MIN_MED:
        return False, f"median_count={h['median_state_count']:.1f} < {HEALTH_MIN_MED}"
    nll_delta = h["fixed_local_nll_k4"] - GPT_ONLY_NLL
    if nll_delta > HEALTH_MAX_NLL_DELTA:
        return False, f"local_nll_k4={h['fixed_local_nll_k4']:.4f} > GPT+{HEALTH_MAX_NLL_DELTA}"
    return True, "ok"


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
    if metrics_path.exists() and not any([force_eval, force_train, force_rewards, force_neighbors]):
        with open(metrics_path) as f:
            m = json.load(f)
        print(f"  [cached] {tag}  q_nll={m.get('q_state_nll','?'):.4f}")
        return m

    print(f"\n  Evaluating: {tag}")
    t0 = time.time()

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
    proto = states["prototype_h"].float()
    B     = len(proto)
    print(f"    States: B={B:,}  (file: {state_path.name})")

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

    ct_precomp  = precompute_lookup(ct_data,  ct_ids,  ct_sims,  states, P_global)
    val_precomp = precompute_lookup(val_data, val_ids, val_sims, states, P_global)
    obs_ct      = build_obs_features(ct_data,  ct_ids,  ct_sims,  states)
    obs_val     = build_obs_features(val_data, val_ids, val_sims, states)

    actions   = build_action_grid(fast=fast_action_grid)
    act_feats = build_action_features(actions)
    A         = len(actions)
    print(f"    Action grid: {A} actions (fast={fast_action_grid})")

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

    best_ai, best_act = find_best_fixed(ct_rewards, actions)
    fixed_m   = eval_fixed_on_val(best_ai, val_rewards, val_precomp)
    fixed_nll = fixed_m["mean_nll"]

    oracle_nll, oracle_nlls = compute_oracle(val_precomp["nll_gpt"], val_rewards)

    q_path = mb_out / "q_model.pt"
    if q_path.exists() and not force_train:
        print(f"    Loading Q model ...")
        ckpt  = torch.load(q_path, weights_only=False)
        model = QStateReadMLP(Q_DIM)
        model.load_state_dict(ckpt["state_dict"])
        model.x_mu  = ckpt["x_mu"];  model.x_std = ckpt["x_std"]
        model.y_mu  = ckpt["y_mu"];  model.y_std = ckpt["y_std"]
        q_train_m   = ckpt.get("train_metrics", {})
    else:
        print(f"    Training Q-MLP  samples={max_q_samples:,} ...")
        set_seed(seed)
        model, q_train_m = train_q_mlp(
            obs_ct, act_feats, ct_rewards, device,
            max_q_samples=max_q_samples, n_epochs=n_epochs,
        )
        torch.save({
            "state_dict": model.state_dict(),
            "x_mu": model.x_mu, "x_std": model.x_std,
            "y_mu": model.y_mu, "y_std": model.y_std,
            "train_metrics": q_train_m, "actions": actions, "tag": tag,
        }, q_path)

    print(f"    Evaluating Q on val ...")
    q_m   = eval_q_on_val(model, obs_val, act_feats, val_rewards,
                           val_precomp, device, actions)
    q_nll = q_m["mean_nll"]
    print(f"    fixed={fixed_nll:.4f}  Q={q_nll:.4f}  oracle={oracle_nll:.4f}  "
          f"(GPT={float(val_precomp['nll_gpt'].mean()):.4f})")

    torch.save({
        "q_nlls": q_m["per_example_nll"], "fixed_nlls": fixed_m["per_example_nll"],
        "oracle_nlls": oracle_nlls, "chosen": q_m["chosen"],
    }, mb_out / "val_predictions.pt")

    act_df   = action_distribution_df(q_m["chosen"], actions)
    act_df.to_csv(mb_out / "action_distribution.csv", index=False)
    top_acts = act_df.head(2)[["name", "frac"]].to_dict("records") if not act_df.empty else []

    insp_dir = out_base / "inspection" / tag
    write_inspection_files(
        insp_dir, val_data, val_precomp,
        val_ids, val_sims, states,
        q_m["per_example_nll"], fixed_m["per_example_nll"],
        q_m["chosen"], best_ai, actions,
    )

    metrics = {
        "tag":               tag,
        "state_file":        str(state_path),
        "num_states":        B,
        "action_grid_size":  A,
        "best_fixed_action": best_act["name"],
        "best_fixed_nll":    fixed_nll,
        "q_state_nll":       q_nll,
        "oracle_nll":        oracle_nll,
        "gpt_nll":           float(val_precomp["nll_gpt"].mean()),
        "full_ds_gpt_nll":   full_bl["gpt"],
        "full_ds_fixed_nll": full_bl["fixed"],
        "full_ds_qmlp_nll":  full_bl["qmlp"],
        "delta_q_vs_fixed":            q_nll - fixed_nll,
        "delta_q_vs_full_ds_qmlp":     q_nll - full_bl["qmlp"],
        "delta_q_vs_full_ds_fixed":    q_nll - full_bl["fixed"],
        "avg_k_states":      q_m["avg_k_states"],
        "retrieval_usage":   q_m["retrieval_usage"],
        "oracle_gap_vs_fixed": oracle_nll - fixed_nll,
        "oracle_gap_vs_q":     oracle_nll - q_nll,
        "top_actions":       top_acts,
        "q_train_metrics":   q_train_m,
        "elapsed_s":         time.time() - t0,
        # aggregate-specific metadata
        "num_buffers_promoted": int(states.get("num_buffers_promoted", 0)),
        "num_buffers_dropped":  int(states.get("num_buffers_dropped", 0)),
        "num_buffers_final":    int(states.get("num_buffers_final", 0)),
        "num_direct_updates":   int(states.get("num_direct_updates", 0)),
        "num_buffer_updates":   int(states.get("num_buffer_updates", 0)),
        "num_buffer_creates":   int(states.get("num_buffer_creates", 0)),
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
    parser.add_argument("--eval_all",         action="store_true",
                        help="Run Q-eval even on health-failing configs")
    parser.add_argument("--skip_health",      action="store_true")
    parser.add_argument("--force_neighbors",  action="store_true")
    parser.add_argument("--force_rewards",    action="store_true")
    parser.add_argument("--force_train",      action="store_true")
    parser.add_argument("--force_eval",       action="store_true")
    parser.add_argument("--force_health",     action="store_true")
    parser.add_argument("--force",            action="store_true")
    parser.add_argument("--tags",             nargs="*", default=None)
    parser.add_argument("--device",           default="cuda")
    parser.add_argument("--seed",             type=int, default=42)
    args = parser.parse_args()

    if args.force:
        args.force_neighbors = args.force_rewards = args.force_train = \
            args.force_eval = args.force_health = True

    device = get_device({"device": args.device})
    src    = Path(args.source)
    sdir   = Path(args.states_dir)
    out    = Path(args.output)

    state_files = sorted(sdir.glob("*.pt"))
    if args.tags:
        state_files = [f for f in state_files if f.stem in args.tags]
    if not state_files:
        print(f"No state files found in {sdir}"); return

    print(f"\nevaluate_aggregate_write_states")
    print(f"  Source:   {src}")
    print(f"  States:   {sdir}  ({len(state_files)} files)")
    print(f"  Output:   {out}")
    print(f"  Samples:  {args.max_q_samples:,}  fast_grid={args.fast_action_grid}")
    print(f"  eval_all: {args.eval_all}")

    # ── Stage 1: health checks ────────────────────────────────────────────────
    health_csv = out / "reports" / "aggregate_state_health.csv"
    health_csv.parent.mkdir(parents=True, exist_ok=True)

    health_results = {}
    if not args.skip_health:
        print(f"\n{'='*55}\nStage 1: health checks\n{'='*55}")
        health_rows = []

        for i, sf in enumerate(state_files):
            print(f"[{i+1}/{len(state_files)}] {sf.stem}")
            hpath = out / "reports" / f"health_{sf.stem}.json"

            if hpath.exists() and not args.force_health:
                with open(hpath) as f:
                    h = json.load(f)
                print(f"  [cached] hit@4={h['hit_at_4']:.2%}  "
                      f"med={h['median_state_count']:.1f}  "
                      f"sing={h['frac_count_le_1']:.1%}")
            else:
                t0 = time.time()
                try:
                    h = compute_state_health(sf, src, device)
                except Exception as e:
                    print(f"  [ERROR in health] {e}")
                    import traceback; traceback.print_exc()
                    continue
                with open(hpath, "w") as f:
                    json.dump(h, f, indent=2)
                elapsed = time.time() - t0
                print(f"  hit@4={h['hit_at_4']:.2%}  med={h['median_state_count']:.1f}  "
                      f"sing={h['frac_count_le_1']:.1%}  "
                      f"local_nll_k4={h['fixed_local_nll_k4']:.4f}  t={elapsed:.0f}s")

            ok, reason = passes_health_check(h)
            h["health_pass"] = ok
            h["health_reason"] = reason
            health_results[sf.stem] = h
            health_rows.append(h)
            status = "PASS" if ok else f"FAIL ({reason})"
            print(f"  Health: {status}")

        if health_rows:
            pd.DataFrame(health_rows).to_csv(health_csv, index=False)
            n_pass = sum(1 for h in health_rows if h.get("health_pass"))
            print(f"\nHealth summary: {n_pass}/{len(health_rows)} configs passed")
            for h in sorted(health_rows, key=lambda x: -x.get("hit_at_4", 0)):
                ok = h.get("health_pass", False)
                print(f"  {'PASS' if ok else 'FAIL'} {h['tag'][:60]:<62} "
                      f"hit@4={h['hit_at_4']:.2%}  "
                      f"med={h['median_state_count']:.1f}  "
                      f"sing={h['frac_count_le_1']:.1%}  "
                      f"local_nll_k4={h['fixed_local_nll_k4']:.4f}")

    # ── Stage 2: Q-eval ───────────────────────────────────────────────────────
    print(f"\n{'='*55}\nStage 2: Q-state-read evaluation\n{'='*55}")
    all_metrics = []

    for i, sf in enumerate(state_files):
        tag = sf.stem
        h   = health_results.get(tag, {})
        ok  = h.get("health_pass", True)   # if no health check, proceed

        if not args.skip_health and not ok and not args.eval_all:
            print(f"\n[{i+1}/{len(state_files)}] {tag}  [skipped: health FAIL]")
            continue

        print(f"\n[{i+1}/{len(state_files)}] {tag}")
        m = eval_state_file(
            sf, src, out, device,
            max_q_samples    = args.max_q_samples,
            n_epochs         = args.n_epochs,
            fast_action_grid = args.fast_action_grid,
            force_neighbors  = args.force_neighbors,
            force_rewards    = args.force_rewards,
            force_train      = args.force_train,
            force_eval       = args.force_eval,
            seed             = args.seed,
        )
        if m:
            if h:
                m.update({k: h[k] for k in
                          ["hit_at_4", "median_state_count", "frac_count_le_1",
                           "fixed_local_nll_k4", "health_pass", "health_reason"]
                          if k in h})
            all_metrics.append(m)

    print(f"\n{'='*55}")
    print(f"Evaluated {len(all_metrics)} state files")
    if all_metrics:
        best = min(all_metrics, key=lambda x: x["q_state_nll"])
        print(f"Best Q NLL: {best['q_state_nll']:.4f}  ({best['tag']})")


if __name__ == "__main__":
    main()

"""
rollout_bandit_write.py  --  MVP 4b Stage 3

Teacher-free online rollout: at each stream step the trained scorer
selects the WRITE action that maximises predicted reward.  No teacher IDs
are used; only the raw datastore (h, y, nll_gpt, gpt_entropy) is needed.

Saves:
  rollout/rollout_log.parquet    -- per-step action decision log
  rollout/states/                -- state files for Q-read evaluation
    bandit_write_B{budget}.pt    -- prototype matrix + token distributions
  rollout/rollout_stats.json

Usage:
  python src/rollout_bandit_write.py \\
    --source scale_200k_seed42 \\
    --output outputs_mvp4b_bandit_write_fast \\
    --budget 10000 \\
    --k_states 8 --k_buffers 8 \\
    --max_exemplars 16 \\
    --promotion_support 8 \\
    --state_budget 10000 --buffer_budget 10000 \\
    --scorer mlp \\
    --seed 42 --device cuda
"""

import sys, json, math, random, argparse, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch

from utils import get_device, set_seed
from build_predictive_states import TOP_K
from build_bandit_write_reward_dataset import (
    LiveObj, LiveMemory, _make_new_obj, _copy_obj, _update_inplace,
    _proto, _entropy_d, extract_features, FEATURE_NAMES, N_FEAT,
    ACT_UPDATE_STATE, ACT_UPDATE_BUFFER, ACT_PROMOTE_AND_UPDATE,
    ACT_CREATE_BUFFER, ACT_DEFER, ACT_NAMES, N_ACT_TYPES,
)
from train_bandit_write_scorer import ActionValueMLP


def _load_scorer(model_dir: Path, scorer_name: str, device: torch.device):
    if scorer_name == "mlp":
        p = model_dir / "mlp.pt"
        ckpt  = torch.load(p, map_location=device, weights_only=False)
        model = ActionValueMLP(ckpt["in_dim"], ckpt["hidden"])
        model.load_state_dict(ckpt["state_dict"])
        model.to(device).eval()
        return ("mlp", model)
    elif scorer_name == "gbm":
        p = model_dir / "gbm.pkl"
        with open(p, "rb") as f:
            return ("gbm", pickle.load(f))
    else:
        raise ValueError(f"Unknown scorer: {scorer_name}")


def _load_scaler(model_dir: Path):
    p = model_dir / "feature_scaler.pkl"
    with open(p, "rb") as f:
        return pickle.load(f)


def _score_actions(
    feature_matrix: np.ndarray,    # [n_actions, 25]
    scorer_name: str,
    scorer_model,
    scaler,
    device: torch.device,
) -> np.ndarray:
    X = scaler.transform(feature_matrix.astype(np.float32))
    if scorer_name == "mlp":
        with torch.no_grad():
            scores = scorer_model(
                torch.from_numpy(X).float().to(device)
            ).cpu().numpy()
    else:
        scores = scorer_model.predict(X)
    return scores.astype(np.float32)


def _save_state(mem: LiveMemory, budget: int, method: str,
                states_dir: Path):
    """Save bandit-write state in the canonical format expected by run_method_budget.

    Matches the schema from build_predictive_states.compute_state_stats:
      prototype_h        FloatTensor [K, D]
      top_k_token_ids    LongTensor  [K, TOP_K]
      top_k_token_counts FloatTensor [K, TOP_K]
      total_counts       FloatTensor [K]
      assigned_count     LongTensor  [K]
      state_entropy      FloatTensor [K]
      state_purity       FloatTensor [K]
      mean_nll           FloatTensor [K]
      mean_gpt_entropy   FloatTensor [K]
    """
    states_dir.mkdir(parents=True, exist_ok=True)
    objs = sorted(mem.objects.values(), key=lambda o: o.obj_id)
    K = len(objs)

    if K == 0:
        print("  WARNING: empty memory, no state saved.")
        return

    D = len(objs[0].sum_h)

    prototype_h        = np.zeros((K, D),    dtype=np.float32)
    top_k_token_ids    = np.zeros((K, TOP_K), dtype=np.int32)
    top_k_token_counts = np.zeros((K, TOP_K), dtype=np.float32)
    total_counts       = np.zeros(K,          dtype=np.float32)
    assigned_count     = np.zeros(K,          dtype=np.int64)
    state_entropy      = np.zeros(K,          dtype=np.float32)
    state_purity       = np.zeros(K,          dtype=np.float32)

    for i, obj in enumerate(objs):
        prototype_h[i] = _proto(obj)
        total = max(obj.count, 1)
        total_counts[i]   = float(obj.count)
        assigned_count[i] = obj.count

        # Top-K tokens by count
        if obj.token_counts:
            toks = sorted(obj.token_counts.items(), key=lambda x: -x[1])
            n    = min(len(toks), TOP_K)
            for j, (tok, cnt) in enumerate(toks[:n]):
                top_k_token_ids[i, j]    = tok
                top_k_token_counts[i, j] = cnt

        # Entropy and purity
        cnts_arr = np.array(list(obj.token_counts.values()), dtype=np.float32)
        if len(cnts_arr) > 0:
            p = cnts_arr / cnts_arr.sum()
            state_entropy[i] = float(-np.sum(p * np.log(p + 1e-12)))
            state_purity[i]  = float(p.max())

    state_dict = {
        "method":              method,
        "budget":              budget,
        "prototype_h":         torch.from_numpy(prototype_h),
        "top_k_token_ids":     torch.from_numpy(top_k_token_ids).long(),
        "top_k_token_counts":  torch.from_numpy(top_k_token_counts),
        "total_counts":        torch.from_numpy(total_counts),
        "assigned_count":      torch.from_numpy(assigned_count),
        "state_entropy":       torch.from_numpy(state_entropy),
        "state_purity":        torch.from_numpy(state_purity),
        "mean_nll":            torch.zeros(K),
        "mean_gpt_entropy":    torch.zeros(K),
        "config":              {"method": method, "budget": budget},
    }
    out_path = states_dir / f"{method}_B{budget}.pt"
    torch.save(state_dict, out_path)
    print(f"  Saved state: {out_path}  ({K} objects, {out_path.stat().st_size/1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",            required=True)
    ap.add_argument("--output",            required=True)
    ap.add_argument("--budget",            type=int, default=10000)
    ap.add_argument("--k_states",          type=int, default=8)
    ap.add_argument("--k_buffers",         type=int, default=8)
    ap.add_argument("--max_exemplars",     type=int, default=16)
    ap.add_argument("--promotion_support", type=int, default=8)
    ap.add_argument("--state_budget",      type=int, default=10000)
    ap.add_argument("--buffer_budget",     type=int, default=10000)
    ap.add_argument("--scorer",            default="mlp",
                    choices=["mlp", "gbm"])
    ap.add_argument("--bootstrap_min",     type=int, default=50,
                    help="force CREATE_BUFFER until this many objects exist; "
                         "avoids degenerate DEFER when memory is OOD-empty")
    ap.add_argument("--seed",    type=int,  default=42)
    ap.add_argument("--device",  default="cuda")
    ap.add_argument("--force",   action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    rng    = np.random.default_rng(args.seed)
    random.seed(args.seed)
    device = get_device({"device": args.device})

    out_dir   = Path(args.output)
    model_dir = out_dir / "scorer_models"
    roll_dir  = out_dir / "rollout"
    roll_dir.mkdir(parents=True, exist_ok=True)

    sentinel = roll_dir / "rollout_stats.json"
    if sentinel.exists() and not args.force:
        print(f"[cached] {sentinel}")
        return

    # ── load scorer ───────────────────────────────────────────────────────────
    print(f"Loading scorer ({args.scorer}) ...")
    scorer_name, scorer_model = _load_scorer(model_dir, args.scorer, device)
    scaler = _load_scaler(model_dir)
    schema = json.load(open(model_dir / "training_schema.json"))
    feat_cols = schema["feature_names"]
    print(f"  Features: {len(feat_cols)}")

    # ── load datastore ────────────────────────────────────────────────────────
    print("Loading datastore ...")
    ds_path = Path(args.source) / "states" / "datastore.pt"
    ds = torch.load(ds_path, weights_only=False)
    h_orig  = ds["h"].float().numpy()
    y_np    = ds["y"].numpy().astype(np.int64)
    nll_gpt     = ds.get("nll_gpt",     torch.zeros(len(y_np))).float().numpy()
    gpt_entropy = ds.get("gpt_entropy", torch.zeros(len(y_np))).float().numpy()
    N, D = h_orig.shape
    h_norm = (h_orig / (np.linalg.norm(h_orig, axis=1, keepdims=True) + 1e-8)
              ).astype(np.float32)
    p_gpt  = np.exp(-nll_gpt).astype(np.float32)
    print(f"  N={N:,}  D={D}")

    # ── initialize empty memory ───────────────────────────────────────────────
    mem = LiveMemory(args.state_budget, args.buffer_budget, D, device)

    log_rows = []
    n_step_logged = 0

    action_counts = {name: 0 for name in ACT_NAMES}

    print(f"\nRunning bandit rollout (scorer={args.scorer}) ...")

    for step in range(N):
        h_raw = h_orig[step]
        h_n   = h_norm[step]
        y_t   = int(y_np[step])

        n_pers = mem.n_persistent
        n_buf  = mem.n_buffers
        total_objs = n_pers + n_buf

        # ── build candidate features + score all valid actions ────────────────
        if total_objs > 0:
            state_cands, buf_cands, sim_info = mem.get_candidates(
                h_n, args.k_states, args.k_buffers)
        else:
            state_cands, buf_cands = [], []
            sim_info = (0.0, 0.0, 0.0, 0.0)

        # Enumerate valid actions and extract features
        action_specs = []   # (action_type, cand_obj_or_None, sim, rank)

        # DEFER always valid
        action_specs.append((ACT_DEFER, None, 0.0, 0))

        # CREATE_BUFFER: valid when budget not full
        if n_buf < args.buffer_budget:
            action_specs.append((ACT_CREATE_BUFFER, None, 0.0, 0))

        # UPDATE_STATE for each persistent candidate
        for rank, (sim, obj) in enumerate(state_cands):
            action_specs.append((ACT_UPDATE_STATE, obj, sim, rank))

        # UPDATE_BUFFER + PROMOTE_AND_UPDATE for each buffer candidate
        for rank, (sim, obj) in enumerate(buf_cands):
            action_specs.append((ACT_UPDATE_BUFFER, obj, sim, rank))
            action_specs.append((ACT_PROMOTE_AND_UPDATE, obj, sim, rank))

        if not action_specs:
            action_specs = [(ACT_DEFER, None, 0.0, 0)]

        # Bootstrap: training data never had empty-memory decision points, so
        # the scorer is OOD there and will always predict DEFER.  Force
        # CREATE_BUFFER until we have enough objects for the scorer to work.
        if total_objs < args.bootstrap_min and n_buf < args.buffer_budget:
            chosen_act, chosen_obj, chosen_sim, chosen_rank = (
                ACT_CREATE_BUFFER, None, 0.0, 0)
            chosen_score = 0.0
        else:
            # Build feature matrix [n_actions, 25]
            feat_matrix = np.zeros((len(action_specs), N_FEAT), dtype=np.float32)
            for ai, (act_type, cand_obj, cand_sim, cand_rank) in enumerate(action_specs):
                feat_matrix[ai] = extract_features(
                    step, N, y_t, float(nll_gpt[step]), float(gpt_entropy[step]),
                    n_pers, n_buf,
                    args.state_budget, args.buffer_budget,
                    sim_info, cand_obj, cand_sim, cand_rank, act_type,
                )

            # Score and pick best
            scores = _score_actions(feat_matrix, scorer_name, scorer_model,
                                    scaler, device)
            best_idx = int(np.argmax(scores))
            chosen_act, chosen_obj, chosen_sim, chosen_rank = action_specs[best_idx]
            chosen_score = float(scores[best_idx])

        # ── execute chosen action ─────────────────────────────────────────────
        if chosen_act == ACT_DEFER:
            pass  # no-op

        elif chosen_act == ACT_CREATE_BUFFER:
            if n_buf < args.buffer_budget:
                # Use a stable pseudo-ID: we don't have teacher IDs, use step
                pseudo_tid = step
                mem.create_buffer(pseudo_tid, h_raw, y_t, step)

        elif chosen_act == ACT_UPDATE_STATE:
            if chosen_obj is not None:
                _update_inplace(chosen_obj, h_raw, y_t, step, args.max_exemplars)
                mem._sync_proto(chosen_obj)

        elif chosen_act == ACT_UPDATE_BUFFER:
            if chosen_obj is not None:
                _update_inplace(chosen_obj, h_raw, y_t, step, args.max_exemplars)
                if (not chosen_obj.is_promoted
                        and chosen_obj.count >= args.promotion_support):
                    chosen_obj.is_promoted   = True
                    chosen_obj.promoted_step = step
                    row = mem.obj_to_row[chosen_obj.obj_id]
                    if row in mem.buffer_rows:
                        mem.buffer_rows.remove(row)
                        mem.persistent_rows.append(row)
                mem._sync_proto(chosen_obj)

        elif chosen_act == ACT_PROMOTE_AND_UPDATE:
            if chosen_obj is not None:
                _update_inplace(chosen_obj, h_raw, y_t, step, args.max_exemplars)
                if not chosen_obj.is_promoted:
                    chosen_obj.is_promoted   = True
                    chosen_obj.promoted_step = step
                    row = mem.obj_to_row[chosen_obj.obj_id]
                    if row in mem.buffer_rows:
                        mem.buffer_rows.remove(row)
                        mem.persistent_rows.append(row)
                mem._sync_proto(chosen_obj)

        action_counts[ACT_NAMES[chosen_act]] += 1

        # Log every 100th step for memory efficiency
        if step % 100 == 0:
            log_rows.append({
                "step":          step,
                "action_type":   chosen_act,
                "action_name":   ACT_NAMES[chosen_act],
                "score":         chosen_score,
                "n_persistent":  mem.n_persistent,
                "n_buffers":     mem.n_buffers,
                "cand_sim":      chosen_sim,
                "gpt_nll":       float(nll_gpt[step]),
            })
            n_step_logged += 1

        if step % 10000 == 0 and step > 0:
            print(f"  step={step:,}  n_pers={mem.n_persistent}  "
                  f"n_buf={mem.n_buffers}  "
                  f"last_act={ACT_NAMES[chosen_act]}")

    print(f"\nRollout complete. Memory: "
          f"{mem.n_persistent} persistent  {mem.n_buffers} buffers")
    print(f"Action counts:")
    for aname, cnt in action_counts.items():
        print(f"  {aname:<22}  {cnt:>8,}  ({cnt/N*100:.1f}%)")

    # ── save rollout log ──────────────────────────────────────────────────────
    log_df = pd.DataFrame(log_rows)
    log_df.to_parquet(roll_dir / "rollout_log.parquet", index=False)
    print(f"\nSaved: {roll_dir}/rollout_log.parquet")

    # ── save state artifacts ───────────────────────────────────────────────────
    method_name = f"bandit_write_{args.scorer}"
    states_dir  = roll_dir / "states"
    _save_state(mem, args.budget, method_name, states_dir)

    # ── save stats ────────────────────────────────────────────────────────────
    stats_dict = {
        "scorer":          args.scorer,
        "n_stream":        N,
        "n_persistent":    mem.n_persistent,
        "n_buffers":       mem.n_buffers,
        "n_total_objects": len(mem.objects),
        "budget":          args.budget,
        "action_counts":   action_counts,
        "action_fracs": {k: v / N for k, v in action_counts.items()},
        "method_name":     method_name,
        "states_path":     str(states_dir / f"{method_name}_B{args.budget}.pt"),
    }
    with open(sentinel, "w") as f:
        json.dump(stats_dict, f, indent=2)
    print(f"Saved: {sentinel}")


if __name__ == "__main__":
    main()

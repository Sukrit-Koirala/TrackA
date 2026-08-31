"""
rollout_mvp4b1.py  --  MVP 4b.1 Stage 5

Clean rollout with:
  - Two bootstrap variants: A (create only when n_total==0), B (create until n_total>=50)
  - Optional on-policy reward collection (--collect_rewards) for second training pass
  - Seed-controlled stream permutation (seed 123 = on-policy dev; 999 = final held-out)
  - Atomic state saves; canonical state schema

Produces four rollout states total (called by orchestrator):
  1. seed=123, variant=A
  2. seed=123, variant=B
  3. seed=999, variant=A
  4. seed=999, variant=B

Usage (single variant):
  python src/rollout_mvp4b1.py \\
    --source scale_200k_seed42 \\
    --output outputs_mvp4b1_onpolicy_bandit_write_clean_fast \\
    --object_budget 10000 \\
    --bootstrap_variant A \\
    --seed 123 \\
    --scorer mlp \\
    --device cuda

Usage (on-policy reward collection):
  python src/rollout_mvp4b1.py \\
    --source scale_200k_seed42 \\
    --output outputs_mvp4b1_onpolicy_bandit_write_clean_fast \\
    --collect_rewards \\
    --bootstrap_variant A --seed 123 --scorer mlp --device cuda
"""

import sys
import argparse
import pickle
import random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from utils import get_device, set_seed
from mvp4b1_common import (
    ARTIFACT_VERSION, FEATURE_NAMES, N_FEAT, REWARD_COLS,
    ACT_UPDATE_STATE, ACT_UPDATE_BUFFER, ACT_CREATE_BUFFER, ACT_DEFER,
    ACT_NAMES, N_ACT_TYPES,
    LiveObj, LiveMemory, _copy_obj, _update_inplace, _make_new_obj, _proto,
    probe_nll, extract_features,
    atomic_write_json, atomic_parquet_save, atomic_torch_save,
    validate_json_cache, save_and_verify_state,
)
from train_mvp4b1_scorer import BanditMLP


# ── Model loading ──────────────────────────────────────────────────────────────

def _load_scorer(model_dir: Path, scorer: str, device: torch.device):
    scaler_p = model_dir / "feature_scaler.pkl"
    if not scaler_p.exists():
        raise FileNotFoundError(f"No scaler at {scaler_p}")
    with open(scaler_p, "rb") as f:
        scaler = pickle.load(f)

    if scorer == "mlp":
        model = BanditMLP(N_FEAT).to(device)
        weights = torch.load(model_dir / "mlp_weights.pt",
                             map_location=device, weights_only=True)
        model.load_state_dict(weights)
        model.eval()
        return model, scaler
    elif scorer == "gbm":
        with open(model_dir / "gbm_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model, scaler
    else:
        raise ValueError(f"Unknown scorer: {scorer}")


def _score(model, scorer: str, X_n: np.ndarray, device: torch.device) -> np.ndarray:
    if scorer == "mlp":
        with torch.no_grad():
            return model(torch.from_numpy(X_n).float().to(device)
                         ).cpu().numpy().astype(np.float32)
    else:
        return model.predict(X_n).astype(np.float32)


# ── Datastore loading ─────────────────────────────────────────────────────────

def _load_datastore(source: Path):
    ds          = torch.load(source / "states" / "datastore.pt", weights_only=False)
    h_orig      = ds["h"].float().numpy()
    y_np        = ds["y"].numpy().astype(np.int64)
    nll_gpt     = ds.get("nll_gpt",     torch.zeros(len(y_np))).float().numpy()
    gpt_entropy = ds.get("gpt_entropy", torch.zeros(len(y_np))).float().numpy()
    p_gpt       = np.exp(-nll_gpt).astype(np.float32)
    N, D        = h_orig.shape
    h_norm      = (h_orig / (np.linalg.norm(h_orig, axis=1, keepdims=True) + 1e-8)
                   ).astype(np.float32)
    return h_orig, h_norm, y_np, nll_gpt, gpt_entropy, p_gpt, N, D


# ── Reward computation (for on-policy collection) ─────────────────────────────

def _compute_rewards_rollout(
    nll_base: np.ndarray, nll_after: np.ndarray,
    local_mask, hist_mask, lh_mask, rep_mask
) -> dict:
    def _gain(mask):
        if not mask.any(): return 0.0
        return float(nll_base[mask].mean()) - float(nll_after[mask].mean())
    def _dmg(mask):
        if not mask.any(): return 0.0
        return float(np.maximum(0, nll_after[mask] - nll_base[mask]).mean())
    return {
        "reward_local":         _gain(local_mask),
        "reward_local_history": _gain(lh_mask),
        "reward_full":          _gain(np.ones(len(nll_base), dtype=bool)),
        "nll_before":           float(nll_base.mean()),
        "nll_after":            float(nll_after.mean()),
        "history_damage":       _dmg(hist_mask),
    }


def _collect_probes(step, N, mem, sc, bc, rng, recent_replay,
                    n_lp=16, n_hp=16, n_rp=16):
    local_raw = rng.integers(0, N, size=n_lp * 3)
    local_raw = local_raw[local_raw != step][:n_lp]

    hist_pool = list({s for _, obj in (sc + bc) for s in obj.exemplars} - {step})
    if len(hist_pool) > n_hp:
        hist_pool = rng.choice(hist_pool, n_hp, replace=False).tolist()
    hist_raw  = np.array(hist_pool, dtype=np.int64)

    replay_pool = [r for r in recent_replay if r != step]
    rep_raw = (np.array(rng.choice(replay_pool, min(n_rp, len(replay_pool)),
                                   replace=False), dtype=np.int64)
               if replay_pool and n_rp > 0 else np.array([], dtype=np.int64))

    all_idx = np.unique(np.concatenate([local_raw.astype(np.int64),
                                         hist_raw, rep_raw]))
    all_idx = all_idx[all_idx != step]

    ls = set(local_raw.tolist()); hs = set(hist_pool); rs = set(rep_raw.tolist())
    lm  = np.array([i in ls for i in all_idx])
    hm  = np.array([i in hs for i in all_idx])
    rm  = np.array([i in rs for i in all_idx])
    lhm = lm | hm
    return all_idx, lm, hm, lhm, rm


# ── Main rollout ──────────────────────────────────────────────────────────────

def rollout(args, h_orig, h_norm, y_np, nll_gpt, gpt_entropy,
            p_gpt, N, D, device, model, scorer, scaler,
            method_name: str, collect_rewards: bool):

    rng = np.random.default_rng(args.seed)
    # Permute stream for this seed
    perm = rng.permutation(N)

    mem           = LiveMemory(args.object_budget, D, device)
    log_rows      = []
    action_counts = {n: 0 for n in ACT_NAMES}
    recent_replay = []
    MAX_REPLAY    = min(2048, N)
    reward_records = []
    lambda_hist    = getattr(args, "lambda_history", 1.0)
    lambda_create  = getattr(args, "lambda_create",  0.0)
    bootstrap_min  = args.bootstrap_min if args.bootstrap_variant == "B" else 0

    n_steps = N
    for tick in range(n_steps):
        step   = int(perm[tick])
        h_raw  = h_orig[step]
        h_n    = h_norm[step]
        y_t    = int(y_np[step])
        n_tot  = mem.n_total

        sc, bc, si = mem.get_candidates(h_n, args.k_states, args.k_buffers)

        # ── Bootstrap ──────────────────────────────────────────────────────────
        force_create = (n_tot == 0) or (n_tot < bootstrap_min and not mem.is_full)

        if force_create:
            chosen_act  = ACT_CREATE_BUFFER
            chosen_obj  = None
            chosen_sim  = 0.0
            chosen_rank = 0
        else:
            # Build feature matrix for candidate actions
            all_cands  = [(ACT_DEFER, None, 0.0, 0)]
            if not mem.is_full:
                all_cands.append((ACT_CREATE_BUFFER, None, 0.0, 0))
            for rank, (sim, obj) in enumerate(sc):
                all_cands.append((ACT_UPDATE_STATE, obj, sim, rank))
            for rank, (sim, obj) in enumerate(bc):
                all_cands.append((ACT_UPDATE_BUFFER, obj, sim, rank))

            n_cands   = len(all_cands)
            X_batch   = np.zeros((n_cands, N_FEAT), dtype=np.float32)
            for ci, (act, obj, sim, rank) in enumerate(all_cands):
                X_batch[ci] = extract_features(
                    step, N, y_t, float(nll_gpt[step]),
                    float(gpt_entropy[step]),
                    mem.n_persistent, mem.n_buffers, args.object_budget,
                    si, obj, sim, rank, act,
                )
            X_n = scaler.transform(X_batch).astype(np.float32)
            scores = _score(model, scorer, X_n, device)
            best_ci = int(np.argmax(scores))
            chosen_act, chosen_obj, chosen_sim, chosen_rank = all_cands[best_ci]

        # ── Execute action ─────────────────────────────────────────────────────
        if chosen_act == ACT_CREATE_BUFFER:
            pseudo_tid = mem._next_id + N * 1000   # unique pseudo teacher ID
            mem.create_buffer(pseudo_tid, h_raw, y_t, step)
        elif chosen_act == ACT_UPDATE_STATE:
            mem.update(chosen_obj.obj_id, h_raw, y_t, step,
                       args.max_exemplars, args.promotion_support)
        elif chosen_act == ACT_UPDATE_BUFFER:
            mem.update(chosen_obj.obj_id, h_raw, y_t, step,
                       args.max_exemplars, args.promotion_support)
        # DEFER: do nothing

        action_counts[ACT_NAMES[chosen_act]] += 1

        # ── On-policy reward collection ────────────────────────────────────────
        if collect_rewards and mem.n_total > 0 and tick % 5 == 0:
            sc2, bc2, si2 = mem.get_candidates(h_n, args.k_states, args.k_buffers)
            all_idx, lm, hm, lhm, rm = _collect_probes(
                step, N, mem, sc2, bc2, rng, recent_replay)
            if len(all_idx) > 0:
                hp   = h_norm[all_idx]
                yp   = y_np[all_idx].astype(np.int32)
                gp   = p_gpt[all_idx]
                pool = [o for _, o in (sc2 + bc2)]
                nll_base = probe_nll(hp, yp, gp, pool)

                for act_type in range(N_ACT_TYPES):
                    if act_type == ACT_CREATE_BUFFER and mem.is_full:
                        continue
                    if act_type == ACT_UPDATE_STATE and not sc2:
                        continue
                    if act_type == ACT_UPDATE_BUFFER and not bc2:
                        continue

                    if act_type == ACT_DEFER:
                        nll_aft  = nll_base.copy()
                        cobj, cs, cr = None, 0.0, 0
                    elif act_type == ACT_CREATE_BUFFER:
                        new_obj  = _make_new_obj(-1, -1, h_raw, y_t, step)
                        nll_aft  = probe_nll(hp, yp, gp, [new_obj] + pool)
                        cobj, cs, cr = None, 0.0, 0
                    elif act_type == ACT_UPDATE_STATE:
                        sim, obj = sc2[0]
                        mod = _copy_obj(obj)
                        _update_inplace(mod, h_raw, y_t, step, args.max_exemplars)
                        mod_pool = [mod if o.obj_id == obj.obj_id else o for o in pool]
                        nll_aft  = probe_nll(hp, yp, gp, mod_pool)
                        cobj, cs, cr = obj, sim, 0
                    else:  # ACT_UPDATE_BUFFER
                        sim, obj = bc2[0]
                        mod = _copy_obj(obj)
                        _update_inplace(mod, h_raw, y_t, step, args.max_exemplars)
                        if not mod.is_promoted and mod.count >= args.promotion_support:
                            mod.is_promoted = True; mod.promoted_step = step
                        mod_pool = [mod if o.obj_id == obj.obj_id else o for o in pool]
                        nll_aft  = probe_nll(hp, yp, gp, mod_pool)
                        cobj, cs, cr = obj, sim, 0

                    rw = _compute_rewards_rollout(nll_base, nll_aft, lm, hm, lhm, rm)
                    rw["reward_penalized"] = (
                        rw["reward_full"]
                        - lambda_hist * rw["history_damage"]
                        - lambda_create * (act_type == ACT_CREATE_BUFFER)
                    )
                    feat = extract_features(
                        step, N, y_t, float(nll_gpt[step]),
                        float(gpt_entropy[step]),
                        mem.n_persistent, mem.n_buffers, args.object_budget,
                        si2, cobj, cs, cr, act_type,
                    )
                    rec = {"step": step, "action_type": act_type,
                           "action_name": ACT_NAMES[act_type],
                           "cand_obj_id": cobj.obj_id if cobj else -1,
                           "n_probes": len(all_idx), **rw}
                    for fi, fn in enumerate(FEATURE_NAMES):
                        rec[fn] = float(feat[fi])
                    reward_records.append(rec)

        # ── Log row ────────────────────────────────────────────────────────────
        if tick % 5000 == 0:
            log_rows.append({
                "tick": tick, "step": step,
                "n_persistent": mem.n_persistent,
                "n_buffers":    mem.n_buffers,
                "n_total":      mem.n_total,
                "action":       ACT_NAMES[chosen_act],
            })

        recent_replay.append(step)
        if len(recent_replay) > MAX_REPLAY:
            recent_replay.pop(0)

    return mem, log_rows, action_counts, reward_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",             required=True)
    ap.add_argument("--output",             required=True)
    ap.add_argument("--object_budget",      type=int, default=10000)
    ap.add_argument("--bootstrap_variant",  default="A", choices=["A", "B"])
    ap.add_argument("--bootstrap_min",      type=int, default=50)
    ap.add_argument("--seed",               type=int, default=123)
    ap.add_argument("--scorer",             default="mlp", choices=["mlp", "gbm"])
    ap.add_argument("--k_states",           type=int, default=8)
    ap.add_argument("--k_buffers",          type=int, default=8)
    ap.add_argument("--max_exemplars",      type=int, default=16)
    ap.add_argument("--promotion_support",  type=int, default=8)
    ap.add_argument("--lambda_history",     type=float, default=1.0)
    ap.add_argument("--lambda_create",      type=float, default=0.0)
    ap.add_argument("--collect_rewards",    action="store_true",
                    help="Collect on-policy counterfactual rewards during rollout")
    ap.add_argument("--device",             default="cuda")
    ap.add_argument("--force",              action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    out_dir    = Path(args.output)
    model_dir  = out_dir / "scorer_models"
    roll_dir   = out_dir / "rollout"
    states_dir = roll_dir / "states"

    tag          = f"mvp4b1_{args.scorer}_v{args.bootstrap_variant}_s{args.seed}"
    stats_path   = roll_dir / f"rollout_stats_{tag}.json"
    op_actions_p = out_dir / "reward_datasets" / "onpolicy" / "actions.parquet"

    collect = args.collect_rewards

    # Check cache
    if not args.force and stats_path.exists():
        # also check state file exists
        state_path = states_dir / f"{tag}_B{args.object_budget}.pt"
        if state_path.exists():
            if not collect or op_actions_p.exists():
                print(f"[cached] {stats_path}")
                return

    print(f"Loading datastore from {args.source} ...")
    h_orig, h_norm, y_np, nll_gpt, gpt_entropy, p_gpt, N, D = \
        _load_datastore(Path(args.source))
    print(f"  N={N:,}  D={D}")

    print(f"Loading scorer ({args.scorer}) ...")
    model, scaler = _load_scorer(model_dir, args.scorer, device)

    print(f"\nRolling out: variant={args.bootstrap_variant}  seed={args.seed}  "
          f"budget={args.object_budget}")

    mem, log_rows, action_counts, reward_records = rollout(
        args, h_orig, h_norm, y_np, nll_gpt, gpt_entropy,
        p_gpt, N, D, device, model, args.scorer, scaler,
        tag, collect
    )

    n_stream = N
    action_fracs = {k: v / n_stream for k, v in action_counts.items()}

    print(f"\nRollout complete:")
    print(f"  Persistent: {mem.n_persistent}   Buffers: {mem.n_buffers}")
    for aname, cnt in action_counts.items():
        print(f"  {aname:<20}  {cnt:>8,}  ({action_fracs[aname]*100:.1f}%)")

    # Save rollout log
    roll_dir.mkdir(parents=True, exist_ok=True)
    log_df = pd.DataFrame(log_rows)
    atomic_parquet_save(log_df, roll_dir / f"rollout_log_{tag}.parquet")

    # Save stats
    stats = {
        "artifact_version": ARTIFACT_VERSION,
        "tag":              tag,
        "scorer":           args.scorer,
        "bootstrap_variant": args.bootstrap_variant,
        "seed":             args.seed,
        "object_budget":    args.object_budget,
        "n_stream":         n_stream,
        "n_persistent":     mem.n_persistent,
        "n_buffers":        mem.n_buffers,
        "n_total_objects":  mem.n_total,
        "action_counts":    action_counts,
        "action_fracs":     action_fracs,
    }
    atomic_write_json(stats_path, stats)

    # Save canonical state
    save_and_verify_state(mem, tag, args.object_budget, states_dir)

    # Save on-policy rewards
    if collect and reward_records:
        op_dir = out_dir / "reward_datasets" / "onpolicy"
        op_dir.mkdir(parents=True, exist_ok=True)
        df_op = pd.DataFrame(reward_records)
        atomic_parquet_save(df_op, op_actions_p)

        # feature_schema stub
        schema = {
            "artifact_version":   ARTIFACT_VERSION,
            "trajectory_type":    "onpolicy",
            "feature_names":      FEATURE_NAMES,
            "reward_cols":        REWARD_COLS,
            "n_features":         N_FEAT,
            "action_types":       ACT_NAMES,
            "n_records":          len(df_op),
            "n_decision_points":  int(df_op["step"].nunique()),
            "rollout_tag":        tag,
        }
        from mvp4b1_common import atomic_write_json as _awj
        _awj(op_dir / "feature_schema.json", schema)
        print(f"  On-policy rewards: {len(df_op):,} records saved to {op_actions_p}")

    print(f"\nSaved: {stats_path}")


if __name__ == "__main__":
    main()

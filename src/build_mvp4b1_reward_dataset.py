"""
build_mvp4b1_reward_dataset.py  --  MVP 4b.1 Stage 1

Builds counterfactual reward datasets from three trajectory types:
  oracle      -- replay teacher assignments from mvp4a0 (seed 42 stream order)
  low_memory  -- scripted policy, multiple episodes, n_total capped at max_lm_objects
                 generates decision points across 0-250 object count states

Key fixes vs MVP 4b:
  - reward_local / reward_local_history / reward_full are DISTINCT columns
  - Local+history uses local+hist probes ONLY (excludes replay)
  - reward_full uses local+hist+replay
  - Combined budget: n_persistent + n_buffers <= object_budget
  - PROMOTE_AND_UPDATE removed; promotion is automatic in UPDATE_BUFFER
  - Atomic writes; artifact versioning
  - Validated: reward_local_history != reward_full for at least some records

Usage (oracle):
  python src/build_mvp4b1_reward_dataset.py \\
    --source scale_200k_seed42 \\
    --oracle_dir outputs_mvp4a0_oracle_reconstruction \\
    --output outputs_mvp4b1_onpolicy_bandit_write_clean_fast \\
    --trajectory_type oracle \\
    --n_decision_points 10000 --seed 42 --device cuda

Usage (low_memory):
  python src/build_mvp4b1_reward_dataset.py \\
    --source scale_200k_seed42 \\
    --output outputs_mvp4b1_onpolicy_bandit_write_clean_fast \\
    --trajectory_type low_memory \\
    --n_decision_points 5000 --seed 42 --device cuda
"""

import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch

from utils import get_device, set_seed
from mvp4b1_common import (
    ARTIFACT_VERSION, FEATURE_NAMES, N_FEAT, REWARD_COLS,
    ACT_UPDATE_STATE, ACT_UPDATE_BUFFER, ACT_CREATE_BUFFER, ACT_DEFER,
    ACT_NAMES, N_ACT_TYPES,
    LiveObj, LiveMemory, _copy_obj, _update_inplace, _make_new_obj, _proto,
    probe_nll, extract_features, stratified_sample,
    atomic_write_json, atomic_parquet_save, validate_json_cache,
)


def _load_datastore(source: Path):
    ds = torch.load(source / "states" / "datastore.pt", weights_only=False)
    h_orig  = ds["h"].float().numpy()
    y_np    = ds["y"].numpy().astype(np.int64)
    nll_gpt     = ds.get("nll_gpt",     torch.zeros(len(y_np))).float().numpy()
    gpt_entropy = ds.get("gpt_entropy", torch.zeros(len(y_np))).float().numpy()
    N, D = h_orig.shape
    h_norm = (h_orig / (np.linalg.norm(h_orig, axis=1, keepdims=True) + 1e-8)
              ).astype(np.float32)
    p_gpt  = np.exp(-nll_gpt).astype(np.float32)
    return h_orig, h_norm, y_np, nll_gpt, gpt_entropy, p_gpt, N, D


def _collect_probe_indices(step: int, N: int, mem: LiveMemory,
                           state_cands, buf_cands,
                           rng, recent_replay: list,
                           n_local: int, n_history: int, n_replay: int):
    """Return (all_probe_idx, local_mask, hist_mask, lh_mask, rep_mask)."""
    # Local probes
    local_raw = rng.integers(0, N, size=n_local * 3)
    local_raw = local_raw[local_raw != step][:n_local]

    # History probes (candidate exemplars)
    hist_pool = []
    for _, obj in (state_cands + buf_cands):
        hist_pool.extend(obj.exemplars)
    hist_pool = list(set(hist_pool) - {step})
    if len(hist_pool) > n_history:
        hist_pool = rng.choice(hist_pool, n_history, replace=False).tolist()
    hist_raw = np.array(hist_pool[:n_history], dtype=np.int64)

    # Replay probes
    replay_pool = [r for r in recent_replay if r != step]
    if replay_pool and n_replay > 0:
        n_rp    = min(n_replay, len(replay_pool))
        rep_raw = np.array(rng.choice(replay_pool, n_rp, replace=False), dtype=np.int64)
    else:
        rep_raw = np.array([], dtype=np.int64)

    all_idx = np.unique(np.concatenate([
        local_raw.astype(np.int64), hist_raw, rep_raw
    ]))
    all_idx = all_idx[all_idx != step]

    local_set = set(local_raw.tolist())
    hist_set  = set(hist_pool[:n_history])
    rep_set   = set(rep_raw.tolist())

    local_mask = np.array([i in local_set  for i in all_idx])
    hist_mask  = np.array([i in hist_set   for i in all_idx])
    rep_mask   = np.array([i in rep_set    for i in all_idx])
    lh_mask    = local_mask | hist_mask    # local + history, NO replay

    return all_idx, local_mask, hist_mask, lh_mask, rep_mask


def _compute_rewards(nll_base: np.ndarray, nll_after: np.ndarray,
                     local_mask, hist_mask, lh_mask, rep_mask):
    """Extract distinct reward signals from per-probe arrays."""
    def _gain(mask):
        if not mask.any():
            return 0.0
        return float(nll_base[mask].mean()) - float(nll_after[mask].mean())

    def _damage(mask):
        if not mask.any():
            return 0.0
        return float(np.maximum(0, nll_after[mask] - nll_base[mask]).mean())

    all_mask = np.ones(len(nll_base), dtype=bool)
    return {
        "reward_local":         _gain(local_mask),
        "reward_local_history": _gain(lh_mask),    # local+hist ONLY
        "reward_full":          _gain(all_mask),   # local+hist+replay
        "nll_before":           float(nll_base.mean()),
        "nll_after":            float(nll_after.mean()),
        "history_damage":       _damage(hist_mask),
    }


def _make_record(step, act_type, cand_obj, cand_sim, cand_rank,
                 N, y_t, nll_gpt_t, gpt_entropy_t,
                 mem: LiveMemory, sim_info, object_budget,
                 rewards: dict) -> dict:
    feat = extract_features(
        step, N, y_t, float(nll_gpt_t), float(gpt_entropy_t),
        mem.n_persistent, mem.n_buffers, object_budget,
        sim_info, cand_obj, cand_sim, cand_rank, act_type,
    )
    rec = {
        "step":           step,
        "action_type":    act_type,
        "action_name":    ACT_NAMES[act_type],
        "cand_obj_id":    cand_obj.obj_id if cand_obj else -1,
        "cand_is_persistent": bool(cand_obj.is_promoted) if cand_obj else False,
        "n_probes":       0,   # filled later
        **rewards,
        "reward_penalized": 0.0,  # filled after calibration
    }
    for fi, fn in enumerate(FEATURE_NAMES):
        rec[fn] = float(feat[fi])
    return rec


# ── Oracle trajectory ─────────────────────────────────────────────────────────

def build_oracle_rewards(args, h_orig, h_norm, y_np, nll_gpt, gpt_entropy,
                          p_gpt, N, D, device, rng, out_dir: Path):
    tag     = f"{args.teacher}_B{args.object_budget}"
    npz_dir = Path(args.oracle_dir) / "teacher_assignments"
    npz     = np.load(npz_dir / f"{tag}_assignments.npz")
    teacher_ids = npz["teacher_ids"].astype(np.int64)

    sampled = stratified_sample(N, args.n_decision_points, rng)
    sampled_set = set(sampled.tolist())

    mem = LiveMemory(args.object_budget, D, device)
    recent_replay = []
    MAX_REPLAY = min(2048, N)
    records    = []
    n_dp       = 0

    print(f"  Oracle trajectory: {len(sampled_set)} decision points ...")

    for step in range(N):
        h_raw = h_orig[step]
        h_n   = h_norm[step]
        y_t   = int(y_np[step])
        z_t   = int(teacher_ids[step])

        if step in sampled_set and mem.n_total > 0:
            sc, bc, si = mem.get_candidates(h_n, args.k_states, args.k_buffers)
            all_idx, lm, hm, lhm, rm = _collect_probe_indices(
                step, N, mem, sc, bc, rng, recent_replay,
                args.n_local_probes, args.n_history_probes, args.n_replay_probes)
            if len(all_idx) == 0:
                pass
            else:
                hp = h_norm[all_idx]
                yp = y_np[all_idx].astype(np.int32)
                gp = p_gpt[all_idx]
                pool = [o for _, o in (sc + bc)]
                nll_base = probe_nll(hp, yp, gp, pool)

                def _eval_action(act_type, cand_obj, cand_sim, cand_rank, mod_pool):
                    nll_aft = probe_nll(hp, yp, gp, mod_pool)
                    rw = _compute_rewards(nll_base, nll_aft, lm, hm, lhm, rm)
                    rec = _make_record(step, act_type, cand_obj, cand_sim,
                                       cand_rank, N, y_t, nll_gpt[step],
                                       gpt_entropy[step], mem, si, args.object_budget, rw)
                    rec["n_probes"] = len(all_idx)
                    return rec

                # DEFER
                records.append(_eval_action(ACT_DEFER, None, 0.0, 0, pool))

                # CREATE_BUFFER
                if not mem.is_full:
                    new_obj = _make_new_obj(-1, -1, h_raw, y_t, step)
                    records.append(_eval_action(ACT_CREATE_BUFFER, None, 0.0, 0,
                                                [new_obj] + pool))

                # UPDATE_STATE
                for rank, (sim, obj) in enumerate(sc):
                    mod = _copy_obj(obj)
                    _update_inplace(mod, h_raw, y_t, step, args.max_exemplars)
                    mod_pool = [mod if o.obj_id == obj.obj_id else o for o in pool]
                    records.append(_eval_action(ACT_UPDATE_STATE, obj, sim, rank,
                                                mod_pool))

                # UPDATE_BUFFER (with auto-promotion in counterfactual copy)
                for rank, (sim, obj) in enumerate(bc):
                    mod = _copy_obj(obj)
                    _update_inplace(mod, h_raw, y_t, step, args.max_exemplars)
                    # simulate auto-promotion in counterfactual
                    if (not mod.is_promoted
                            and mod.count >= args.promotion_support):
                        mod.is_promoted   = True
                        mod.promoted_step = step
                    mod_pool = [mod if o.obj_id == obj.obj_id else o for o in pool]
                    records.append(_eval_action(ACT_UPDATE_BUFFER, obj, sim, rank,
                                                mod_pool))

                n_dp += 1
                if n_dp % 2000 == 0:
                    print(f"    {n_dp}/{len(sampled_set)} dp  step={step:,}  "
                          f"pers={mem.n_persistent}  buf={mem.n_buffers}  "
                          f"recs={len(records):,}")

        # Oracle update
        if not mem.has_teacher(z_t):
            mem.create_buffer(z_t, h_raw, y_t, step)
        else:
            oid = mem.teacher_to_obj[z_t]
            mem.update(oid, h_raw, y_t, step, args.max_exemplars,
                       args.promotion_support)

        recent_replay.append(step)
        if len(recent_replay) > MAX_REPLAY:
            recent_replay.pop(0)

    return records, n_dp


# ── Low-memory scripted trajectory ────────────────────────────────────────────

def build_low_memory_rewards(args, h_orig, h_norm, y_np, nll_gpt, gpt_entropy,
                              p_gpt, N, D, device, rng, out_dir: Path):
    """Multi-episode scripted policy keeping n_total small.

    Episodes of length `episode_length` start fresh each time.
    The scripted policy creates buffers aggressively when memory is sparse,
    then updates the nearest object. Caps total objects at max_lm_objects.
    """
    episode_length  = min(args.episode_length_lm, N)
    max_lm_objects  = args.max_lm_objects
    n_episodes      = max(1, N // episode_length)
    target_dp       = args.n_decision_points
    dp_per_episode  = max(1, target_dp // n_episodes)

    records = []
    n_dp    = 0
    print(f"  Low-memory trajectory: {n_episodes} episodes × "
          f"{episode_length} steps, max_objects={max_lm_objects} ...")

    for ep in range(n_episodes):
        if n_dp >= target_dp:
            break
        ep_start = ep * episode_length
        ep_end   = min(ep_start + episode_length, N)
        ep_steps = list(range(ep_start, ep_end))

        # Sample decision points within this episode
        ep_len = ep_end - ep_start
        ep_dp_count = min(dp_per_episode, ep_len)
        ep_dp_steps = set(
            (ep_start + rng.integers(0, ep_len, size=ep_dp_count * 2)[:ep_dp_count]
             ).tolist()
        )

        mem = LiveMemory(max_lm_objects, D, device)
        recent_replay = []
        MAX_REPLAY = min(512, ep_len)
        next_pseudo_tid = ep * episode_length  # unique pseudo-IDs per episode

        for step in ep_steps:
            h_raw = h_orig[step]
            h_n   = h_norm[step]
            y_t   = int(y_np[step])
            n_tot = mem.n_total

            if step in ep_dp_steps and n_tot > 0:
                sc, bc, si = mem.get_candidates(h_n, args.k_states, args.k_buffers)
                all_idx, lm, hm, lhm, rm = _collect_probe_indices(
                    step, N, mem, sc, bc, rng, recent_replay,
                    args.n_local_probes, args.n_history_probes,
                    args.n_replay_probes)
                if len(all_idx) > 0:
                    hp   = h_norm[all_idx]
                    yp   = y_np[all_idx].astype(np.int32)
                    gp   = p_gpt[all_idx]
                    pool = [o for _, o in (sc + bc)]
                    nll_base = probe_nll(hp, yp, gp, pool)

                    def _eval(act_type, cand_obj, cand_sim, cand_rank, mod_pool):
                        nll_aft = probe_nll(hp, yp, gp, mod_pool)
                        rw = _compute_rewards(nll_base, nll_aft, lm, hm, lhm, rm)
                        rec = _make_record(step, act_type, cand_obj, cand_sim,
                                           cand_rank, N, y_t, nll_gpt[step],
                                           gpt_entropy[step], mem, si,
                                           max_lm_objects, rw)
                        rec["n_probes"] = len(all_idx)
                        return rec

                    records.append(_eval(ACT_DEFER, None, 0.0, 0, pool))
                    if not mem.is_full:
                        new_obj = _make_new_obj(-1, -1, h_raw, y_t, step)
                        records.append(_eval(ACT_CREATE_BUFFER, None, 0.0, 0,
                                             [new_obj] + pool))
                    for rank, (sim, obj) in enumerate(sc):
                        mod = _copy_obj(obj)
                        _update_inplace(mod, h_raw, y_t, step, args.max_exemplars)
                        mod_pool = [mod if o.obj_id == obj.obj_id else o
                                    for o in pool]
                        records.append(_eval(ACT_UPDATE_STATE, obj, sim, rank,
                                             mod_pool))
                    for rank, (sim, obj) in enumerate(bc):
                        mod = _copy_obj(obj)
                        _update_inplace(mod, h_raw, y_t, step, args.max_exemplars)
                        if not mod.is_promoted and mod.count >= args.promotion_support:
                            mod.is_promoted = True; mod.promoted_step = step
                        mod_pool = [mod if o.obj_id == obj.obj_id else o
                                    for o in pool]
                        records.append(_eval(ACT_UPDATE_BUFFER, obj, sim, rank,
                                             mod_pool))
                    n_dp += 1

            # Scripted policy
            n_tot = mem.n_total
            if n_tot == 0:
                mem.create_buffer(next_pseudo_tid, h_raw, y_t, step)
                next_pseudo_tid += 1
            else:
                # Create with decreasing probability; never exceed cap
                p_create = max(0.0, 0.4 * (1 - n_tot / max_lm_objects))
                if not mem.is_full and rng.random() < p_create:
                    mem.create_buffer(next_pseudo_tid, h_raw, y_t, step)
                    next_pseudo_tid += 1
                else:
                    # Update nearest object
                    sc2, bc2, _ = mem.get_candidates(h_n, 1, 1)
                    if sc2:
                        oid = sc2[0][1].obj_id
                        mem.update(oid, h_raw, y_t, step, args.max_exemplars,
                                   args.promotion_support)
                    elif bc2:
                        oid = bc2[0][1].obj_id
                        mem.update(oid, h_raw, y_t, step, args.max_exemplars,
                                   args.promotion_support)

            recent_replay.append(step)
            if len(recent_replay) > MAX_REPLAY:
                recent_replay.pop(0)

        if ep % 10 == 0 or ep == n_episodes - 1:
            print(f"    episode {ep+1}/{n_episodes}  dp_so_far={n_dp}  "
                  f"recs={len(records):,}")

    return records, n_dp


# ── Calibrate and save ────────────────────────────────────────────────────────

def calibrate_and_save(records: list, args, out_dir: Path,
                       trajectory_type: str, n_dp: int):
    if not records:
        raise RuntimeError(f"No records generated for {trajectory_type}!")

    import pandas as pd
    df = pd.DataFrame(records)

    gains = df["reward_full"].values.astype(np.float64)
    gain_std = float(np.nanstd(gains))
    lam_create  = (args.lambda_create  if args.lambda_create  >= 0
                   else 0.10 * gain_std)
    lam_promote = (args.lambda_promote if args.lambda_promote >= 0
                   else 0.05 * gain_std)

    df["reward_penalized"] = (
        df["reward_full"]
        - args.lambda_history * df["history_damage"]
        - lam_create * (df["action_type"] == ACT_CREATE_BUFFER).astype(float)
    )

    # Validate: reward_local_history must differ from reward_full somewhere
    same = np.allclose(df["reward_local_history"].values,
                       df["reward_full"].values, atol=1e-7)
    if same:
        raise RuntimeError(
            "REWARD VALIDATION FAILED: reward_local_history == reward_full "
            "for all records. Replay probes were never used or all are empty.")

    # Validate: no NaN or inf
    for col in ["reward_local", "reward_local_history", "reward_full",
                "reward_penalized"]:
        bad = (~np.isfinite(df[col].values)).sum()
        if bad > 0:
            raise RuntimeError(
                f"REWARD VALIDATION FAILED: {bad} non-finite values in {col}")

    # Atomic save
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_parquet_save(df, out_dir / "actions.parquet")

    dp_df = pd.DataFrame({"step": df["step"].unique()})
    atomic_parquet_save(dp_df, out_dir / "decision_points.parquet")

    schema = {
        "artifact_version":   ARTIFACT_VERSION,
        "trajectory_type":    trajectory_type,
        "feature_names":      FEATURE_NAMES,
        "reward_cols":        REWARD_COLS,
        "n_features":         N_FEAT,
        "action_types":       ACT_NAMES,
        "n_decision_points":  n_dp,
        "n_records":          len(df),
        "lambda_create":      lam_create,
        "lambda_promote":     lam_promote,
        "lambda_history":     args.lambda_history,
        "gain_std":           gain_std,
        "read_params":        {"k": 4, "tau": 0.05, "alpha": 0.75},
    }
    atomic_write_json(out_dir / "feature_schema.json", schema)

    print(f"  Saved {len(df):,} records  ({n_dp} decision points)")
    print(f"  lambda_create={lam_create:.5f}  lambda_promote={lam_promote:.5f}"
          f"  gain_std={gain_std:.5f}")
    print(f"  reward_full vs reward_local_history differ: OK")
    return df, schema


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",              required=True)
    ap.add_argument("--oracle_dir",          default=None)
    ap.add_argument("--output",              required=True)
    ap.add_argument("--trajectory_type",     default="oracle",
                    choices=["oracle", "low_memory"])
    ap.add_argument("--teacher",             default="minibatch_kmeans")
    ap.add_argument("--object_budget",       type=int, default=10000)
    ap.add_argument("--promotion_support",   type=int, default=8)
    ap.add_argument("--n_decision_points",   type=int, default=10000)
    ap.add_argument("--k_states",            type=int, default=8)
    ap.add_argument("--k_buffers",           type=int, default=8)
    ap.add_argument("--n_local_probes",      type=int, default=16)
    ap.add_argument("--n_history_probes",    type=int, default=16)
    ap.add_argument("--n_replay_probes",     type=int, default=16)
    ap.add_argument("--max_exemplars",       type=int, default=16)
    ap.add_argument("--max_lm_objects",      type=int, default=500,
                    help="max objects per episode for low_memory trajectory")
    ap.add_argument("--episode_length_lm",   type=int, default=2000,
                    help="steps per episode for low_memory trajectory")
    ap.add_argument("--lambda_create",       type=float, default=-1.0)
    ap.add_argument("--lambda_promote",      type=float, default=-1.0)
    ap.add_argument("--lambda_history",      type=float, default=1.0)
    ap.add_argument("--seed",                type=int, default=42)
    ap.add_argument("--device",              default="cuda")
    ap.add_argument("--force",               action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    rng    = np.random.default_rng(args.seed)
    device = get_device({"device": args.device})

    ttype   = args.trajectory_type
    out_dir = Path(args.output) / "reward_datasets" / ttype
    sentinel = out_dir / "feature_schema.json"

    cached = validate_json_cache(sentinel, ["artifact_version", "n_records",
                                            "trajectory_type"])
    if cached and not args.force:
        print(f"[cached] {sentinel}")
        return

    print(f"Building {ttype} reward dataset ...")
    h_orig, h_norm, y_np, nll_gpt, gpt_entropy, p_gpt, N, D = \
        _load_datastore(Path(args.source))
    print(f"  N={N:,}  D={D}")

    if ttype == "oracle":
        if args.oracle_dir is None:
            raise ValueError("--oracle_dir required for oracle trajectory")
        records, n_dp = build_oracle_rewards(
            args, h_orig, h_norm, y_np, nll_gpt, gpt_entropy,
            p_gpt, N, D, device, rng, out_dir)
    elif ttype == "low_memory":
        records, n_dp = build_low_memory_rewards(
            args, h_orig, h_norm, y_np, nll_gpt, gpt_entropy,
            p_gpt, N, D, device, rng, out_dir)

    calibrate_and_save(records, args, out_dir, ttype, n_dp)
    print(f"\nDone. {ttype} reward dataset saved to {out_dir}")


if __name__ == "__main__":
    main()

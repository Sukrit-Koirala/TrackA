"""
build_write_object_provenance.py  --  MVP 4c-0 Stage 1

Replays a memory trajectory (oracle or bandit) from scratch, logging every
WRITE event with before/after object statistics. Builds a per-object lifecycle
record and a per-event record needed for delayed-credit attribution.

Trajectories:
  oracle  -- replay oracle_stream using teacher assignments (deterministic)
  bandit  -- re-run bandit rollout using saved scorer (deterministic)

Outputs (in {output}/provenance/{traj}_*):
  {traj}_objects.parquet  -- one row per object
  {traj}_events.parquet   -- one row per write event

Usage:
  python src/build_write_object_provenance.py \\
    --trajectory oracle \\
    --source scale_200k_seed42 \\
    --oracle_dir outputs_mvp4a0_oracle_reconstruction \\
    --output outputs_mvp4c0_delayed_credit_audit_fast \\
    --teacher minibatch_kmeans --budget 10000 \\
    --promotion_support 8 --seed 42 --device cuda

  python src/build_write_object_provenance.py \\
    --trajectory bandit \\
    --source scale_200k_seed42 \\
    --bandit_dir outputs_mvp4b_bandit_write_fast \\
    --output outputs_mvp4c0_delayed_credit_audit_fast \\
    --budget 10000 --scorer mlp \\
    --promotion_support 8 --seed 42 --device cuda
"""

import sys
import json
import math
import argparse
import pickle
import random
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch

from utils import get_device, set_seed


EPS = 1e-8


# ── Object stats helpers ──────────────────────────────────────────────────────

def _entropy(token_counts: dict, total: int) -> float:
    if total <= 0:
        return 0.0
    e = 0.0
    for c in token_counts.values():
        p = c / total
        if p > 0:
            e -= p * math.log(p + 1e-12)
    return e


def _purity(token_counts: dict, total: int) -> float:
    if total <= 0 or not token_counts:
        return 0.0
    return max(token_counts.values()) / total


def _p_y(token_counts: dict, y: int, total: int) -> float:
    return token_counts.get(y, 0) / max(total, 1)


def _proto_norm(sum_h: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(sum_h)
    return (sum_h / max(n, EPS)).astype(np.float32)


def _proto_cosine(sum_h_before: np.ndarray, sum_h_after: np.ndarray) -> float:
    p1 = _proto_norm(sum_h_before)
    p2 = _proto_norm(sum_h_after)
    return float(np.dot(p1, p2))


# ── Load datastore ────────────────────────────────────────────────────────────

def _load_datastore(source: Path):
    ds = torch.load(source / "states" / "datastore.pt", weights_only=False)
    h_orig  = ds["h"].float().numpy()
    y_np    = ds["y"].numpy().astype(np.int64)
    nll_gpt = ds.get("nll_gpt",     torch.zeros(len(y_np))).float().numpy()
    gpt_ent = ds.get("gpt_entropy", torch.zeros(len(y_np))).float().numpy()
    N, D    = h_orig.shape
    h_norm  = (h_orig / (np.linalg.norm(h_orig, axis=1, keepdims=True) + EPS)
               ).astype(np.float32)
    return h_orig, h_norm, y_np, nll_gpt, gpt_ent, N, D


# ══════════════════════════════════════════════════════════════════════════════
# Oracle trajectory
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OracleObjProv:
    object_id:         int
    teacher_id:        int
    sum_h:             np.ndarray
    token_counts:      dict
    count:             int
    created_step:      int
    last_update_step:  int
    is_promoted:       bool
    promoted_step:     int
    n_update_buf:      int = 0
    n_update_sta:      int = 0


def _replay_oracle(
    h_orig: np.ndarray, h_norm: np.ndarray, y_np: np.ndarray,
    teacher_ids: np.ndarray, promotion_support: int,
) -> tuple[list, list]:
    N, D = h_orig.shape
    objects: dict[int, OracleObjProv] = {}
    next_obj_id = 0
    event_id    = 0
    obj_rows    = []
    evt_rows    = []

    for step in range(N):
        h_raw = h_orig[step]
        y_t   = int(y_np[step])
        z_t   = int(teacher_ids[step])

        if z_t not in objects:
            # CREATE_BUFFER
            obj = OracleObjProv(
                object_id=next_obj_id, teacher_id=z_t,
                sum_h=h_raw.astype(np.float64),
                token_counts={y_t: 1}, count=1,
                created_step=step, last_update_step=step,
                is_promoted=False, promoted_step=-1,
            )
            objects[z_t] = obj
            next_obj_id += 1

            evt_rows.append({
                "event_id":          event_id,
                "object_id":         obj.object_id,
                "teacher_id":        z_t,
                "stream_step":       step,
                "action_type":       "CREATE_BUFFER",
                "true_token":        y_t,
                "support_before":    0,
                "support_after":     1,
                "entropy_before":    0.0,
                "entropy_after":     0.0,
                "purity_before":     0.0,
                "purity_after":      1.0,
                "p_y_true_before":   0.0,
                "p_y_true_after":    1.0,
                "proto_cosine_sim":  float("nan"),
                "immediate_reward":  float("nan"),
                "trajectory":        "oracle",
            })
            event_id += 1

        else:
            obj = objects[z_t]
            ent_bef = _entropy(obj.token_counts, obj.count)
            pur_bef = _purity(obj.token_counts, obj.count)
            py_bef  = _p_y(obj.token_counts, y_t, obj.count)
            sup_bef = obj.count
            sum_h_before = obj.sum_h.copy()

            # Determine action
            was_promoted = obj.is_promoted
            act_name = "UPDATE_STATE" if was_promoted else "UPDATE_BUFFER"

            # Update
            obj.sum_h += h_raw.astype(np.float64)
            obj.token_counts[y_t] = obj.token_counts.get(y_t, 0) + 1
            obj.count += 1
            obj.last_update_step = step
            if was_promoted:
                obj.n_update_sta += 1
            else:
                obj.n_update_buf += 1

            promoted_now = False
            if not was_promoted and obj.count >= promotion_support:
                obj.is_promoted   = True
                obj.promoted_step = step
                promoted_now = True
                act_name = "UPDATE_BUFFER_PROMOTE"

            ent_aft = _entropy(obj.token_counts, obj.count)
            pur_aft = _purity(obj.token_counts, obj.count)
            py_aft  = _p_y(obj.token_counts, y_t, obj.count)
            cos_chg = _proto_cosine(sum_h_before, obj.sum_h)

            evt_rows.append({
                "event_id":          event_id,
                "object_id":         obj.object_id,
                "teacher_id":        z_t,
                "stream_step":       step,
                "action_type":       act_name,
                "true_token":        y_t,
                "support_before":    sup_bef,
                "support_after":     obj.count,
                "entropy_before":    ent_bef,
                "entropy_after":     ent_aft,
                "purity_before":     pur_bef,
                "purity_after":      pur_aft,
                "p_y_true_before":   py_bef,
                "p_y_true_after":    py_aft,
                "proto_cosine_sim":  cos_chg,
                "immediate_reward":  float("nan"),
                "trajectory":        "oracle",
            })
            event_id += 1

        if step % 50_000 == 0 and step > 0:
            n_p = sum(1 for o in objects.values() if o.is_promoted)
            n_b = sum(1 for o in objects.values() if not o.is_promoted)
            print(f"    step={step:,}  pers={n_p}  buf={n_b}  events={event_id:,}")

    # End-of-stream promote remaining buffers
    for obj in objects.values():
        if not obj.is_promoted and obj.count > 0:
            obj.is_promoted   = True
            obj.promoted_step = N

    # Build object rows
    for z_t, obj in objects.items():
        ent = _entropy(obj.token_counts, obj.count)
        pur = _purity(obj.token_counts, obj.count)
        proto = _proto_norm(obj.sum_h)
        obj_rows.append({
            "object_id":          obj.object_id,
            "teacher_id":         z_t,
            "creation_step":      obj.created_step,
            "creation_action":    "CREATE_BUFFER",
            "promotion_step":     obj.promoted_step,
            "last_update_step":   obj.last_update_step,
            "final_support":      obj.count,
            "final_n_token_types": len(obj.token_counts),
            "final_entropy":      ent,
            "final_purity":       pur,
            "n_update_buf":       obj.n_update_buf,
            "n_update_sta":       obj.n_update_sta,
            "is_persistent":      obj.is_promoted,
            "state_row_index":    z_t,  # oracle: row == teacher_id
            "trajectory":         "oracle",
        })

    return obj_rows, evt_rows


def build_oracle_provenance(args, h_orig, h_norm, y_np, nll_gpt, N, D, out_dir):
    # Load teacher assignments
    tag      = f"{args.teacher}_B{args.budget}"
    npz_path = Path(args.oracle_dir) / "teacher_assignments" / f"{tag}_assignments.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Teacher assignments not found: {npz_path}")
    teacher_ids = np.load(npz_path)["teacher_ids"].astype(np.int64)
    print(f"  Loaded teacher assignments: {len(teacher_ids):,}")

    print(f"  Replaying oracle stream ...")
    obj_rows, evt_rows = _replay_oracle(
        h_orig, h_norm, y_np, teacher_ids, args.promotion_support)

    return obj_rows, evt_rows


# ══════════════════════════════════════════════════════════════════════════════
# Bandit trajectory
# ══════════════════════════════════════════════════════════════════════════════

def _build_bandit_provenance_mvp4b1(args, h_orig, h_norm, y_np, nll_gpt, gpt_ent,
                                     N, D, out_dir, device):
    """MVP 4b.1 format: BanditMLP (mlp_weights.pt), single object_budget LiveMemory,
    no ACT_PROMOTE_AND_UPDATE (auto-promotion inside UPDATE_BUFFER)."""
    from mvp4b1_common import (
        LiveMemory, extract_features, N_FEAT,
        ACT_UPDATE_STATE, ACT_UPDATE_BUFFER, ACT_CREATE_BUFFER, ACT_DEFER, ACT_NAMES,
    )
    from train_mvp4b1_scorer import BanditMLP

    bandit_dir    = Path(args.bandit_dir)
    model_dir     = bandit_dir / "scorer_models"
    object_budget = args.budget
    promo_support = args.promotion_support
    k_states      = 8
    k_buffers     = 8
    max_exemplars = 16
    bootstrap_min = 50

    if args.scorer == "mlp":
        sd = torch.load(model_dir / "mlp_weights.pt", map_location=device,
                        weights_only=False)
        model = BanditMLP(N_FEAT).to(device).eval()
        model.load_state_dict(sd)
        scorer_name = "mlp"
    elif args.scorer == "gbm":
        with open(model_dir / "gbm_model.pkl", "rb") as f:
            model = pickle.load(f)
        scorer_name = "gbm"
    else:
        raise ValueError(f"Unknown scorer: {args.scorer}")

    with open(model_dir / "feature_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    mem = LiveMemory(object_budget, D, device)

    set_seed(args.seed)
    random.seed(args.seed)

    obj_rows = []
    evt_rows = []
    event_id = 0

    def _score(feat_matrix):
        X = scaler.transform(feat_matrix.astype(np.float32))
        if scorer_name == "mlp":
            with torch.no_grad():
                return model(torch.from_numpy(X).float().to(device)).cpu().numpy()
        return model.predict(X).astype(np.float32)

    print(f"  Replaying MVP 4b.1 bandit rollout (scorer={scorer_name}) ...")

    for step in range(N):
        h_raw = h_orig[step]
        h_n   = h_norm[step]
        y_t   = int(y_np[step])

        n_pers     = mem.n_persistent
        n_buf      = mem.n_buffers

        if mem.n_total > 0:
            sc, bc, si = mem.get_candidates(h_n, k_states, k_buffers)
        else:
            sc, bc, si = [], [], (0., 0., 0., 0.)

        action_specs = [(ACT_DEFER, None, 0.0, 0)]
        if not mem.is_full:
            action_specs.append((ACT_CREATE_BUFFER, None, 0.0, 0))
        for rank, (sim, obj) in enumerate(sc):
            action_specs.append((ACT_UPDATE_STATE, obj, sim, rank))
        for rank, (sim, obj) in enumerate(bc):
            action_specs.append((ACT_UPDATE_BUFFER, obj, sim, rank))

        if mem.n_total < bootstrap_min and not mem.is_full:
            chosen_act, chosen_obj, chosen_sim, chosen_rank = (ACT_CREATE_BUFFER, None, 0.0, 0)
            chosen_score = 0.0
        else:
            feat_matrix = np.zeros((len(action_specs), N_FEAT), dtype=np.float32)
            for ai, (atype, cobj, csim, crank) in enumerate(action_specs):
                feat_matrix[ai] = extract_features(
                    step, N, y_t, float(nll_gpt[step]), float(gpt_ent[step]),
                    n_pers, n_buf, object_budget,
                    si, cobj, csim, crank, atype,
                )
            scores = _score(feat_matrix)
            best   = int(np.argmax(scores))
            chosen_act, chosen_obj, chosen_sim, chosen_rank = action_specs[best]
            chosen_score = float(scores[best])

        if chosen_obj is not None:
            sup_bef = chosen_obj.count
            ent_bef = _entropy(chosen_obj.token_counts, chosen_obj.count)
            pur_bef = _purity(chosen_obj.token_counts, chosen_obj.count)
            py_bef  = _p_y(chosen_obj.token_counts, y_t, chosen_obj.count)
            sh_bef  = chosen_obj.sum_h.copy()
        else:
            sup_bef = ent_bef = pur_bef = py_bef = 0
            sh_bef  = None

        if chosen_act == ACT_DEFER:
            best_sim = max(
                max((sim for sim, _ in sc), default=0.0),
                max((sim for sim, _ in bc), default=0.0),
            )
            evt_rows.append({
                "event_id": event_id, "object_id": -1, "teacher_id": -1,
                "stream_step": step, "action_type": "DEFER", "true_token": y_t,
                "support_before": 0, "support_after": 0,
                "entropy_before": 0.0, "entropy_after": 0.0,
                "purity_before": 0.0, "purity_after": 0.0,
                "p_y_true_before": 0.0, "p_y_true_after": 0.0,
                "proto_cosine_sim": best_sim, "immediate_reward": float("nan"),
                "trajectory": "bandit",
            })
            event_id += 1

        elif chosen_act == ACT_CREATE_BUFFER:
            new_obj = mem.create_buffer(step, h_raw, y_t, step)
            if new_obj is not None:
                evt_rows.append({
                    "event_id": event_id, "object_id": new_obj.obj_id,
                    "teacher_id": -1, "stream_step": step,
                    "action_type": "CREATE_BUFFER", "true_token": y_t,
                    "support_before": 0, "support_after": 1,
                    "entropy_before": 0.0, "entropy_after": 0.0,
                    "purity_before": 0.0, "purity_after": 1.0,
                    "p_y_true_before": 0.0, "p_y_true_after": 1.0,
                    "proto_cosine_sim": float("nan"), "immediate_reward": float("nan"),
                    "trajectory": "bandit",
                })
                event_id += 1

        else:
            obj = chosen_obj
            if obj is not None:
                was_promoted = obj.is_promoted
                mem.update(obj.obj_id, h_raw, y_t, step, max_exemplars, promo_support)
                sup_aft = obj.count
                ent_aft = _entropy(obj.token_counts, obj.count)
                pur_aft = _purity(obj.token_counts, obj.count)
                py_aft  = _p_y(obj.token_counts, y_t, obj.count)
                cos_chg = _proto_cosine(sh_bef, obj.sum_h) if sh_bef is not None else float("nan")
                # Auto-promote happened on this step → emit combined action name
                act_name = "UPDATE_BUFFER_PROMOTE" if (
                    not was_promoted and obj.is_promoted
                ) else ACT_NAMES[chosen_act]
                evt_rows.append({
                    "event_id": event_id, "object_id": obj.obj_id,
                    "teacher_id": -1, "stream_step": step,
                    "action_type": act_name, "true_token": y_t,
                    "support_before": sup_bef, "support_after": sup_aft,
                    "entropy_before": ent_bef, "entropy_after": ent_aft,
                    "purity_before": pur_bef, "purity_after": pur_aft,
                    "p_y_true_before": py_bef, "p_y_true_after": py_aft,
                    "proto_cosine_sim": cos_chg, "immediate_reward": float("nan"),
                    "trajectory": "bandit",
                })
                event_id += 1

        if step % 50_000 == 0 and step > 0:
            print(f"    step={step:,}  pers={mem.n_persistent}  "
                  f"buf={mem.n_buffers}  events={event_id:,}")

    for obj in sorted(mem.objects.values(), key=lambda o: o.obj_id):
        ent = _entropy(obj.token_counts, obj.count)
        pur = _purity(obj.token_counts, obj.count) if obj.count > 0 else 0.0
        obj_rows.append({
            "object_id": obj.obj_id, "teacher_id": obj.teacher_id,
            "creation_step": obj.created_step, "creation_action": "CREATE_BUFFER",
            "promotion_step": obj.promoted_step,
            "last_update_step": obj.last_update_step,
            "final_support": obj.count,
            "final_n_token_types": len(obj.token_counts),
            "final_entropy": ent, "final_purity": pur,
            "n_update_buf": -1, "n_update_sta": -1,
            "is_persistent": obj.is_promoted,
            "state_row_index": obj.obj_id, "trajectory": "bandit",
        })

    return obj_rows, evt_rows


def build_bandit_provenance(args, h_orig, h_norm, y_np, nll_gpt, gpt_ent,
                             N, D, out_dir, device):
    bandit_dir = Path(args.bandit_dir)
    model_dir  = bandit_dir / "scorer_models"

    # Route to v2 for MVP 4b.1 format (saves mlp_weights.pt, not mlp.pt)
    if (model_dir / "mlp_weights.pt").exists():
        return _build_bandit_provenance_mvp4b1(
            args, h_orig, h_norm, y_np, nll_gpt, gpt_ent, N, D, out_dir, device)

    # Import the MVP 4b LiveMemory / LiveObj
    from build_bandit_write_reward_dataset import (
        LiveObj, LiveMemory, _make_new_obj, _update_inplace, _proto, _entropy_d,
        extract_features, FEATURE_NAMES, N_FEAT,
        ACT_UPDATE_STATE, ACT_UPDATE_BUFFER, ACT_PROMOTE_AND_UPDATE,
        ACT_CREATE_BUFFER, ACT_DEFER, ACT_NAMES,
    )
    from train_bandit_write_scorer import ActionValueMLP

    bandit_dir = Path(args.bandit_dir)
    model_dir  = bandit_dir / "scorer_models"
    schema     = json.load(open(model_dir / "training_schema.json"))

    # Load scorer
    if args.scorer == "mlp":
        ckpt  = torch.load(model_dir / "mlp.pt", map_location=device,
                           weights_only=False)
        model = ActionValueMLP(ckpt["in_dim"], ckpt["hidden"])
        model.load_state_dict(ckpt["state_dict"])
        model.to(device).eval()
        scorer_name = "mlp"
    else:
        with open(model_dir / "gbm.pkl", "rb") as f:
            model = pickle.load(f)
        scorer_name = "gbm"

    with open(model_dir / "feature_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Read rollout config
    cfg_path = bandit_dir / "config.json"
    cfg      = json.load(open(cfg_path)) if cfg_path.exists() else {}
    state_budget  = cfg.get("state_budget",  args.budget)
    buffer_budget = cfg.get("buffer_budget", args.budget)
    bootstrap_min = cfg.get("bootstrap_min", 50)
    k_states      = cfg.get("k_states",      8)
    k_buffers     = cfg.get("k_buffers",     8)
    max_exemplars = cfg.get("max_exemplars",  16)

    p_gpt = np.exp(-nll_gpt).astype(np.float32)
    mem   = LiveMemory(state_budget, buffer_budget, D, device)

    set_seed(args.seed)
    random.seed(args.seed)

    obj_rows  = []
    evt_rows  = []
    event_id  = 0

    def _score(feat_matrix):
        X = scaler.transform(feat_matrix.astype(np.float32))
        if scorer_name == "mlp":
            with torch.no_grad():
                return model(torch.from_numpy(X).float().to(device)).cpu().numpy()
        else:
            return model.predict(X).astype(np.float32)

    print(f"  Replaying bandit rollout (scorer={scorer_name}) ...")

    for step in range(N):
        h_raw = h_orig[step]
        h_n   = h_norm[step]
        y_t   = int(y_np[step])

        n_pers     = mem.n_persistent
        n_buf      = mem.n_buffers
        total_objs = n_pers + n_buf

        if total_objs > 0:
            sc, bc, si = mem.get_candidates(h_n, k_states, k_buffers)
        else:
            sc, bc, si = [], [], (0., 0., 0., 0.)

        # Build action specs
        action_specs = [(ACT_DEFER, None, 0.0, 0)]
        if n_buf < buffer_budget:
            action_specs.append((ACT_CREATE_BUFFER, None, 0.0, 0))
        for rank, (sim, obj) in enumerate(sc):
            action_specs.append((ACT_UPDATE_STATE, obj, sim, rank))
        for rank, (sim, obj) in enumerate(bc):
            action_specs.append((ACT_UPDATE_BUFFER, obj, sim, rank))
            action_specs.append((ACT_PROMOTE_AND_UPDATE, obj, sim, rank))
        if not action_specs:
            action_specs = [(ACT_DEFER, None, 0.0, 0)]

        if total_objs < bootstrap_min and n_buf < buffer_budget:
            chosen_act, chosen_obj, chosen_sim, chosen_rank = (
                ACT_CREATE_BUFFER, None, 0.0, 0)
            chosen_score = 0.0
        else:
            feat_matrix = np.zeros((len(action_specs), N_FEAT), dtype=np.float32)
            for ai, (atype, cobj, csim, crank) in enumerate(action_specs):
                feat_matrix[ai] = extract_features(
                    step, N, y_t, float(nll_gpt[step]), float(gpt_ent[step]),
                    n_pers, n_buf, state_budget, buffer_budget,
                    si, cobj, csim, crank, atype,
                )
            scores = _score(feat_matrix)
            best   = int(np.argmax(scores))
            chosen_act, chosen_obj, chosen_sim, chosen_rank = action_specs[best]
            chosen_score = float(scores[best])

        # Log DEFER decision (no object modified, but decision must be recorded)
        if chosen_act == ACT_DEFER:
            best_cand_sim = max((sim for sim, _ in sc), default=0.0) if sc else 0.0
            best_cand_sim = max(best_cand_sim,
                                max((sim for sim, _ in bc), default=0.0) if bc else 0.0)
            evt_rows.append({
                "event_id":         event_id,
                "object_id":        -1,
                "teacher_id":       -1,
                "stream_step":      step,
                "action_type":      "DEFER",
                "true_token":       y_t,
                "support_before":   0,
                "support_after":    0,
                "entropy_before":   0.0,
                "entropy_after":    0.0,
                "purity_before":    0.0,
                "purity_after":     0.0,
                "p_y_true_before":  0.0,
                "p_y_true_after":   0.0,
                "proto_cosine_sim": best_cand_sim,
                "immediate_reward": float("nan"),
                "trajectory":       "bandit",
            })
            event_id += 1

        if chosen_act != ACT_DEFER:
            if chosen_obj is not None:
                sup_bef = chosen_obj.count
                ent_bef = _entropy_d(chosen_obj.token_counts, chosen_obj.count)
                pur_bef = (_purity(chosen_obj.token_counts, chosen_obj.count)
                           if chosen_obj.count > 0 else 0.0)
                py_bef  = chosen_obj.token_counts.get(y_t, 0) / max(chosen_obj.count, 1)
                sh_bef  = chosen_obj.sum_h.copy()
            else:
                sup_bef = ent_bef = pur_bef = py_bef = 0
                sh_bef  = None

        # Execute action
        if chosen_act == ACT_DEFER:
            pass
        elif chosen_act == ACT_CREATE_BUFFER:
            if n_buf < buffer_budget:
                mem.create_buffer(step, h_raw, y_t, step)
        elif chosen_act == ACT_UPDATE_STATE:
            if chosen_obj is not None:
                _update_inplace(chosen_obj, h_raw, y_t, step, max_exemplars)
                mem._sync_proto(chosen_obj)
        elif chosen_act == ACT_UPDATE_BUFFER:
            if chosen_obj is not None:
                _update_inplace(chosen_obj, h_raw, y_t, step, max_exemplars)
                if not chosen_obj.is_promoted and chosen_obj.count >= args.promotion_support:
                    chosen_obj.is_promoted   = True
                    chosen_obj.promoted_step = step
                    row = mem.obj_to_row[chosen_obj.obj_id]
                    if row in mem.buffer_rows:
                        mem.buffer_rows.remove(row)
                        mem.persistent_rows.append(row)
                mem._sync_proto(chosen_obj)
        elif chosen_act == ACT_PROMOTE_AND_UPDATE:
            if chosen_obj is not None:
                _update_inplace(chosen_obj, h_raw, y_t, step, max_exemplars)
                if not chosen_obj.is_promoted:
                    chosen_obj.is_promoted   = True
                    chosen_obj.promoted_step = step
                    row = mem.obj_to_row[chosen_obj.obj_id]
                    if row in mem.buffer_rows:
                        mem.buffer_rows.remove(row)
                        mem.persistent_rows.append(row)
                mem._sync_proto(chosen_obj)

        # Record event
        if chosen_act != ACT_DEFER:
            if chosen_act == ACT_CREATE_BUFFER:
                # newly created object: find it by step (pseudo_tid = step)
                new_obj = mem.objects.get(step) if hasattr(mem, 'objects') else None
                # Actually LiveMemory keyed by teacher_id. The create used step as pseudo_tid.
                new_obj = mem.objects.get(mem.teacher_to_obj.get(step, -1), None) if new_obj is None else new_obj
                if new_obj is None and mem.n_total > 0:
                    # fallback: get last created
                    new_obj = mem.objects.get(mem._next_id - 1, None)
                evt_rows.append({
                    "event_id":         event_id,
                    "object_id":        new_obj.obj_id if new_obj else -1,
                    "teacher_id":       -1,
                    "stream_step":      step,
                    "action_type":      ACT_NAMES[chosen_act],
                    "true_token":       y_t,
                    "support_before":   0,
                    "support_after":    1 if new_obj else 0,
                    "entropy_before":   0.0,
                    "entropy_after":    0.0,
                    "purity_before":    0.0,
                    "purity_after":     1.0,
                    "p_y_true_before":  0.0,
                    "p_y_true_after":   1.0,
                    "proto_cosine_sim": float("nan"),
                    "immediate_reward": float("nan"),
                    "trajectory":       "bandit",
                })
            else:
                obj = chosen_obj
                sup_aft = obj.count
                ent_aft = _entropy_d(obj.token_counts, obj.count)
                pur_aft = (_purity(obj.token_counts, obj.count)
                           if obj.count > 0 else 0.0)
                py_aft  = obj.token_counts.get(y_t, 0) / max(obj.count, 1)
                cos_chg = _proto_cosine(sh_bef, obj.sum_h) if sh_bef is not None else float("nan")
                evt_rows.append({
                    "event_id":         event_id,
                    "object_id":        obj.obj_id,
                    "teacher_id":       -1,
                    "stream_step":      step,
                    "action_type":      ACT_NAMES[chosen_act],
                    "true_token":       y_t,
                    "support_before":   sup_bef,
                    "support_after":    sup_aft,
                    "entropy_before":   ent_bef,
                    "entropy_after":    ent_aft,
                    "purity_before":    pur_bef,
                    "purity_after":     pur_aft,
                    "p_y_true_before":  py_bef,
                    "p_y_true_after":   py_aft,
                    "proto_cosine_sim": cos_chg,
                    "immediate_reward": float("nan"),
                    "trajectory":       "bandit",
                })
            event_id += 1

        if step % 50_000 == 0 and step > 0:
            print(f"    step={step:,}  pers={mem.n_persistent}  "
                  f"buf={mem.n_buffers}  events={event_id:,}")

    # Build object rows (bandit): row index == obj_id (sorted by obj_id in _save_state)
    for obj in sorted(mem.objects.values(), key=lambda o: o.obj_id):
        ent = _entropy_d(obj.token_counts, obj.count)
        pur = _purity(obj.token_counts, obj.count) if obj.count > 0 else 0.0
        obj_rows.append({
            "object_id":          obj.obj_id,
            "teacher_id":         obj.teacher_id,
            "creation_step":      obj.created_step,
            "creation_action":    "CREATE_BUFFER",
            "promotion_step":     obj.promoted_step,
            "last_update_step":   obj.last_update_step,
            "final_support":      obj.count,
            "final_n_token_types": len(obj.token_counts),
            "final_entropy":      ent,
            "final_purity":       pur,
            "n_update_buf":       -1,  # not tracked separately for bandit
            "n_update_sta":       -1,
            "is_persistent":      obj.is_promoted,
            "state_row_index":    obj.obj_id,  # bandit: row == obj_id
            "trajectory":         "bandit",
        })

    return obj_rows, evt_rows


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory",        required=True, choices=["oracle", "bandit"])
    ap.add_argument("--source",            required=True,
                    help="Raw data dir with train.pt / val.pt")
    ap.add_argument("--datastore_dir",     default=None,
                    help="Dir containing states/datastore.pt (precomputed embeddings). "
                         "Defaults to --source if not set.")
    ap.add_argument("--oracle_dir",        default=None)
    ap.add_argument("--bandit_dir",        default=None)
    ap.add_argument("--output",            required=True)
    ap.add_argument("--teacher",           default="minibatch_kmeans")
    ap.add_argument("--budget",            type=int, default=10000)
    ap.add_argument("--scorer",            default="mlp")
    ap.add_argument("--promotion_support", type=int, default=8)
    ap.add_argument("--seed",              type=int, default=42)
    ap.add_argument("--device",            default="cuda")
    ap.add_argument("--force",             action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device  = get_device({"device": args.device})
    src     = Path(args.source)
    out_dir = Path(args.output)
    prov_dir = out_dir / "provenance"
    prov_dir.mkdir(parents=True, exist_ok=True)

    # datastore_dir: where states/datastore.pt lives (precomputed embeddings)
    ds_dir = Path(args.datastore_dir) if args.datastore_dir else src

    traj     = args.trajectory
    obj_path = prov_dir / f"{traj}_objects.parquet"
    evt_path = prov_dir / f"{traj}_events.parquet"

    if obj_path.exists() and evt_path.exists() and not args.force:
        print(f"[cached] {obj_path.name}, {evt_path.name}")
        return

    print(f"\nLoading datastore from {ds_dir} ...")
    h_orig, h_norm, y_np, nll_gpt, gpt_ent, N, D = _load_datastore(ds_dir)
    print(f"  N={N:,}  D={D}")

    if traj == "oracle":
        if args.oracle_dir is None:
            raise ValueError("--oracle_dir required for oracle trajectory")
        obj_rows, evt_rows = build_oracle_provenance(
            args, h_orig, h_norm, y_np, nll_gpt, N, D, out_dir)
    else:
        if args.bandit_dir is None:
            raise ValueError("--bandit_dir required for bandit trajectory")
        obj_rows, evt_rows = build_bandit_provenance(
            args, h_orig, h_norm, y_np, nll_gpt, gpt_ent, N, D, out_dir, device)

    obj_df = pd.DataFrame(obj_rows)
    evt_df = pd.DataFrame(evt_rows)

    obj_df.to_parquet(obj_path, index=False)
    evt_df.to_parquet(evt_path, index=False)

    print(f"\nProvenance complete for '{traj}':")
    print(f"  Objects:      {len(obj_df):,}  -> {obj_path}")
    print(f"  Write events: {len(evt_df):,}  -> {evt_path}")
    if len(obj_df) > 0:
        print(f"  Persistent:   {obj_df['is_persistent'].sum():,}")
        act_cts = evt_df["action_type"].value_counts()
        for act, cnt in act_cts.items():
            print(f"    {act:<28}  {cnt:>8,}")


if __name__ == "__main__":
    main()

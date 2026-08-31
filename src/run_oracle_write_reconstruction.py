"""
run_oracle_write_reconstruction.py  --  MVP 4a-0: Oracle Reconstruction

Streams all datastore examples in sequential order using oracle teacher
assignments to drive the BUFFER → UPDATE → PROMOTE lifecycle.

Exports two reconstruction modes:
  exact_teacher_prototype    batch token counts + saved offline prototype
  sequential_running_mean    batch token counts + sequential float64 running sum

MATCH recall diagnostic:
  For each example where the oracle target already exists in memory, records
  the query hidden state and target object ID. After streaming, computes
  cosine recall@K against the FINAL prototype matrices (over-estimate for
  early queries; noted in report).

Accumulation rule:
  Uses sum_h (float64) of raw un-normalised h vectors, identical to
  build_predictive_states.compute_state_stats. normalize(sum_h) equals
  the offline prototype when assignments match exactly.

Usage:
  python src/run_oracle_write_reconstruction.py \\
    --source            outputs_scale_sweep/scale_200k_seed42 \\
    --offline_states_dir outputs_mvp2b_state_write/states \\
    --output            outputs_mvp4a0_oracle_reconstruction \\
    --teacher           minibatch_kmeans \\
    --budget            10000 \\
    --promotion_support 8 \\
    --seed              42 \\
    --device            cuda
"""

import sys
import json
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn.functional as F

from utils import get_device, set_seed
from build_predictive_states import (
    accumulate_token_counts,
    compute_state_stats,
    TOP_K,
)

VOCAB = 50257


# ── in-memory object types ────────────────────────────────────────────────────

@dataclass
class OracleObj:
    """Unified in-memory representation for both buffer and persistent objects."""
    object_id:        int
    teacher_id:       int
    sum_h:            np.ndarray        # float64, unnormalised accumulator
    token_counts:     dict              # token_id -> count (int)
    count:            int
    created_step:     int
    last_update_step: int
    is_promoted:      bool = False
    promoted_step:    int  = -1


def _update_obj(obj: OracleObj, h_raw: np.ndarray, y_t: int, step: int):
    obj.sum_h += h_raw.astype(np.float64)
    obj.token_counts[int(y_t)] = obj.token_counts.get(int(y_t), 0) + 1
    obj.count += 1
    obj.last_update_step = step


def _sequential_prototype(obj: OracleObj) -> np.ndarray:
    """Return float32 L2-normalised prototype from accumulated sum_h."""
    n = np.linalg.norm(obj.sum_h)
    if n < 1e-8:
        return np.zeros(obj.sum_h.shape, dtype=np.float32)
    return (obj.sum_h / n).astype(np.float32)


# ── oracle streaming loop ─────────────────────────────────────────────────────

def oracle_stream(
    h_orig:            np.ndarray,      # [N, D] raw fp32
    y_np:              np.ndarray,      # [N] int64
    teacher_ids:       np.ndarray,      # [N] int64
    promotion_support: int = 8,
) -> tuple:
    """
    Process examples in stream order using oracle assignments.
    Returns (objects, action_stats, match_queries).

    match_queries: list of (step, h_norm_f32, z_t, n_pers, n_buf,
                            target_is_persistent, target_count_at_step)
    """
    N, D = h_orig.shape
    h_norms = (h_orig / (np.linalg.norm(h_orig, axis=1, keepdims=True) + 1e-8)).astype(np.float32)

    objects: dict[int, OracleObj] = {}   # teacher_id -> OracleObj
    next_obj_id = 0

    n_create  = 0
    n_upd_buf = 0
    n_promote = 0
    n_upd_sta = 0

    match_queries = []   # recorded for MATCH recall

    for step in range(N):
        h_raw = h_orig[step]        # unnormalised (float32)
        h_n   = h_norms[step]       # normalised
        y_t   = int(y_np[step])
        z_t   = int(teacher_ids[step])

        if z_t not in objects:
            # ── CREATE_BUFFER ─────────────────────────────────────────────────
            obj = OracleObj(
                object_id        = next_obj_id,
                teacher_id       = z_t,
                sum_h            = h_raw.astype(np.float64),
                token_counts     = {y_t: 1},
                count            = 1,
                created_step     = step,
                last_update_step = step,
                is_promoted      = False,
            )
            objects[z_t] = obj
            next_obj_id += 1
            n_create += 1

        else:
            obj = objects[z_t]

            # Record MATCH recall query (target already exists)
            n_pers = sum(1 for o in objects.values() if o.is_promoted)
            n_buf  = sum(1 for o in objects.values() if not o.is_promoted)
            match_queries.append((
                step, h_n, z_t, n_pers, n_buf,
                obj.is_promoted, obj.count,
            ))

            if not obj.is_promoted:
                # ── UPDATE_BUFFER ─────────────────────────────────────────────
                _update_obj(obj, h_raw, y_t, step)
                n_upd_buf += 1
                if obj.count >= promotion_support:
                    obj.is_promoted  = True
                    obj.promoted_step = step
                    n_promote += 1
            else:
                # ── UPDATE_STATE ──────────────────────────────────────────────
                _update_obj(obj, h_raw, y_t, step)
                n_upd_sta += 1

        if (step + 1) % 50_000 == 0:
            n_p = sum(1 for o in objects.values() if o.is_promoted)
            n_b = sum(1 for o in objects.values() if not o.is_promoted)
            print(f"    step={step+1:,}  persistent={n_p:,}  buffers={n_b:,}  "
                  f"match_queries={len(match_queries):,}")

    # Promote remaining non-empty buffers
    n_end_promote = 0
    for obj in objects.values():
        if not obj.is_promoted and obj.count > 0:
            obj.is_promoted   = True
            obj.promoted_step = N
            n_end_promote += 1

    action_stats = {
        "N":                 int(N),
        "promotion_support": int(promotion_support),
        "n_create":          int(n_create),
        "n_update_buffer":   int(n_upd_buf),
        "n_promote":         int(n_promote),
        "n_update_state":    int(n_upd_sta),
        "n_end_promote":     int(n_end_promote),
        "final_n_objects":   int(len(objects)),
        "n_match_queries":   int(len(match_queries)),
    }
    print(f"\n  Stream done:")
    for k, v in action_stats.items():
        print(f"    {k}: {v}")

    return objects, action_stats, match_queries


# ── MATCH recall ──────────────────────────────────────────────────────────────

def compute_match_recall(
    match_queries: list,
    objects:       dict,
    device:        torch.device,
    k_values:      list = [1, 4, 8, 16],
    batch_size:    int  = 2048,
) -> dict:
    """
    Evaluate MATCH recall against FINAL memory state.
    NOTE: uses final prototypes (over-estimates recall for early-step queries).
    """
    if not match_queries:
        return {"N_queries": 0, "note": "no queries recorded"}

    # Build final prototype matrices
    persistent = [(tid, o) for tid, o in objects.items() if o.is_promoted]
    buffers    = [(tid, o) for tid, o in objects.items() if not o.is_promoted]
    combined   = persistent + buffers

    def _build(items):
        if not items:
            return None, None
        tids  = np.array([tid for tid, _ in items], dtype=np.int32)
        proto = np.stack([_sequential_prototype(o) for _, o in items])  # [M, D]
        return tids, torch.from_numpy(proto)

    p_tids, p_proto = _build(persistent)
    b_tids, b_proto = _build(buffers)
    a_tids, a_proto = _build(combined)

    steps          = np.array([q[0] for q in match_queries], np.int32)
    h_queries      = np.stack([q[1] for q in match_queries]).astype(np.float32)
    z_targets      = np.array([q[2] for q in match_queries], np.int32)
    target_counts  = np.array([q[6] for q in match_queries], np.int32)
    target_is_pers = np.array([q[5] for q in match_queries], bool)

    N_q   = len(match_queries)
    max_k = max(k_values)

    def _recall(proto_t, tids_np, name=""):
        if proto_t is None:
            return {}
        proto_gpu = proto_t.float().to(device)
        hits = {k: np.zeros(N_q, dtype=np.int8) for k in k_values}
        for s in range(0, N_q, batch_size):
            e     = min(s + batch_size, N_q)
            q_gpu = torch.from_numpy(h_queries[s:e]).to(device)
            sims  = q_gpu @ proto_gpu.T                          # [B, M]
            top_k = min(max_k, sims.shape[1])
            topki = sims.topk(top_k, dim=-1).indices.cpu().numpy()  # [B, top_k]
            for bi in range(e - s):
                z = int(z_targets[s + bi])
                top_tids = tids_np[topki[bi]]
                for k in k_values:
                    if k <= top_k:
                        hits[k][s + bi] = int(z in top_tids[:k])
        return {f"recall@{k}": float(hits[k].mean()) for k in k_values}

    results = {
        "N_queries":   N_q,
        "note":        "recall computed against FINAL memory state (over-estimates early queries)",
        "persistent":  _recall(p_proto, p_tids, "persistent"),
        "buffers":     _recall(b_proto, b_tids, "buffers"),
        "combined":    _recall(a_proto, a_tids, "combined"),
    }

    # Breakdowns using combined recall@8
    if a_proto is not None:
        a_gpu = a_proto.float().to(device)
        hit8  = np.zeros(N_q, dtype=np.int8)
        for s in range(0, N_q, batch_size):
            e     = min(s + batch_size, N_q)
            q_gpu = torch.from_numpy(h_queries[s:e]).to(device)
            sims  = q_gpu @ a_gpu.T
            topki = sims.topk(min(8, sims.shape[1]), dim=-1).indices.cpu().numpy()
            for bi in range(e - s):
                z = int(z_targets[s + bi])
                hit8[s + bi] = int(z in a_tids[topki[bi]])

        N_stream = max(1, int(steps.max()) + 1)
        quartile = np.clip(steps * 4 // N_stream, 0, 3)
        c = results["combined"]

        c["by_quartile"] = {
            f"Q{q+1}": float(hit8[quartile == q].mean())
            for q in range(4) if (quartile == q).any()
        }
        buckets = np.digitize(target_counts, [2, 5, 20, 100])
        bucket_labels = ["1", "2-4", "5-19", "20-99", "100+"]
        c["by_support_bucket"] = {
            bucket_labels[b]: float(hit8[buckets == b].mean())
            for b in range(5) if (buckets == b).any()
        }
        c["persistent_target_recall8"] = (
            float(hit8[target_is_pers].mean()) if target_is_pers.any() else None)
        c["buffer_target_recall8"] = (
            float(hit8[~target_is_pers].mean()) if (~target_is_pers).any() else None)

    return results


# ── state artifact export ─────────────────────────────────────────────────────

def build_state_artifact(
    objects:       dict,
    teacher_state: dict,
    h_orig:        np.ndarray,
    y_np:          np.ndarray,
    nll_gpt:       np.ndarray,
    gpt_entropy:   np.ndarray,
    teacher_ids:   np.ndarray,
    mode:          str,
    teacher_name:  str,
    budget:        int,
) -> dict:
    """
    Build a canonical state artifact dict compatible with run_method_budget.

    Both modes share the same batch-recomputed token counts (from oracle labels).
    They differ only in prototype_h source.

    mode == 'exact_teacher_prototype':
        prototype_h taken from saved offline teacher (control experiment)
    mode == 'sequential_running_mean':
        prototype_h = normalize(sum_h) from sequential stream (float64 accum)
    """
    B = budget
    ids, cnts, total = accumulate_token_counts(teacher_ids, y_np, B)
    stats = compute_state_stats(h_orig, teacher_ids, B, ids, cnts, total,
                                 nll_gpt, gpt_entropy)

    if mode == "exact_teacher_prototype":
        stats["prototype_h"] = teacher_state["prototype_h"].clone().float()
        method_name = f"oracle_exact_{teacher_name}"
    elif mode == "sequential_running_mean":
        D     = h_orig.shape[1]
        proto = torch.zeros(B, D, dtype=torch.float32)
        for tid, obj in objects.items():
            if 0 <= tid < B:
                proto[tid] = torch.from_numpy(_sequential_prototype(obj))
        stats["prototype_h"] = proto
        method_name = f"oracle_seq_{teacher_name}"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    stats.update({
        "method": method_name,
        "budget": budget,
        "config": {"oracle_mode": mode, "teacher": teacher_name, "budget": budget},
        "extra":  {"oracle": True, "mode": mode},
    })
    return stats, method_name


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",              required=True)
    ap.add_argument("--offline_states_dir",  required=True)
    ap.add_argument("--output",              required=True)
    ap.add_argument("--teacher",             default="minibatch_kmeans")
    ap.add_argument("--budget",              type=int, default=10000)
    ap.add_argument("--promotion_support",   type=int, default=8)
    ap.add_argument("--seed",                type=int, default=42)
    ap.add_argument("--device",              default="cuda")
    ap.add_argument("--force",               action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device({"device": args.device})
    src    = Path(args.source)
    out    = Path(args.output)
    B      = args.budget
    tag    = f"{args.teacher}_B{B}"

    out.mkdir(parents=True, exist_ok=True)
    recon_dir = out / "reconstructions"
    match_dir = out / "match_diagnostics"
    recon_dir.mkdir(parents=True, exist_ok=True)
    match_dir.mkdir(parents=True, exist_ok=True)

    # ── load assignments ──────────────────────────────────────────────────────
    npz_path = out / "teacher_assignments" / f"{tag}_assignments.npz"
    if not npz_path.exists():
        print(f"ERROR: assignments not found: {npz_path}")
        print("  Run build_oracle_teacher_assignments.py first.")
        sys.exit(1)
    data        = np.load(npz_path)
    teacher_ids = data["teacher_ids"].astype(np.int64)
    print(f"Loaded assignments: {len(teacher_ids):,} rows")

    # ── load teacher artifact ─────────────────────────────────────────────────
    teacher_path = Path(args.offline_states_dir) / f"{tag}.pt"
    teacher_state = torch.load(teacher_path, weights_only=False)

    # ── load datastore ────────────────────────────────────────────────────────
    print("Loading datastore ...")
    ds  = torch.load(src / "states" / "datastore.pt", weights_only=False)
    N   = len(ds["y"])
    y_np = ds["y"].numpy().astype(np.int64)
    h_f  = ds["h"].float()
    h_orig = h_f.numpy()
    nll_gpt     = ds.get("nll_gpt",     torch.zeros(N)).float().numpy()
    gpt_entropy = ds.get("gpt_entropy", torch.zeros(N)).float().numpy()
    print(f"  N={N:,}  D={h_orig.shape[1]}")

    # ── oracle streaming ──────────────────────────────────────────────────────
    action_done = out / "oracle_action_stats.json"
    if action_done.exists() and not args.force:
        print(f"[cached] oracle stream already run. Loading action stats ...")
        with open(action_done) as f:
            action_stats = json.load(f)
        # Need to re-run to get objects + match_queries for artifact export
        # Only skip if both reconstruction artifacts already exist
        both_exist = all(
            (recon_dir / m / "states" /
             f"oracle_{'exact' if 'exact' in m else 'seq'}_{args.teacher}_B{B}.pt").exists()
            for m in ["exact_teacher_prototype", "sequential_running_mean"]
        )
        match_exists = (match_dir / "target_recall.json").exists()
        if both_exist and match_exists:
            print("[cached] All outputs exist, skipping stream.")
            return
        print("  Re-running stream to regenerate missing outputs ...")

    print(f"\nOracle stream (promotion_support={args.promotion_support}) ...")
    objects, action_stats, match_queries = oracle_stream(
        h_orig, y_np, teacher_ids,
        promotion_support=args.promotion_support,
    )
    with open(action_done, "w") as f:
        json.dump(action_stats, f, indent=2)

    # ── MATCH recall ──────────────────────────────────────────────────────────
    recall_path = match_dir / "target_recall.json"
    if not recall_path.exists() or args.force:
        print("\nComputing MATCH recall ...")
        recall = compute_match_recall(match_queries, objects, device)
        with open(recall_path, "w") as f:
            json.dump(recall, f, indent=2)
        comb = recall.get("combined", {})
        for k in [1, 4, 8, 16]:
            v = comb.get(f"recall@{k}")
            if v is not None:
                print(f"  combined recall@{k}: {v:.4f}")

    # ── export state artifacts ────────────────────────────────────────────────
    for mode in ["exact_teacher_prototype", "sequential_running_mean"]:
        mode_dir = recon_dir / mode / "states"
        mode_dir.mkdir(parents=True, exist_ok=True)

        sd, method_name = build_state_artifact(
            objects, teacher_state, h_orig, y_np, nll_gpt, gpt_entropy,
            teacher_ids, mode, args.teacher, B,
        )
        state_path = mode_dir / f"{method_name}_B{B}.pt"
        if state_path.exists() and not args.force:
            print(f"[cached] {state_path.name}")
        else:
            torch.save(sd, state_path)
            print(f"Saved: {state_path}  ({state_path.stat().st_size/1e6:.1f} MB)")

        # Reconstruction stats
        r_stats = {
            "mode":              mode,
            "teacher":           tag,
            "method_name":       method_name,
            "n_nonempty_states": int((sd["assigned_count"] > 0).sum().item()),
            "total_assigned":    int(sd["assigned_count"].sum().item()),
            "mean_state_count":  float(sd["assigned_count"].float().mean().item()),
        }
        r_stats.update(action_stats)
        stats_path = recon_dir / mode / "reconstruction_stats.json"
        with open(stats_path, "w") as f:
            json.dump(r_stats, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()

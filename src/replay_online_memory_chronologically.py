"""
replay_online_memory_chronologically.py  --  MVP 4c-1 Stage 4 (True Replay)

TRUE online chronological memory replay on a single training stream.

For each step t in [0, T):
  1. READ using M_t  — the memory that exists BEFORE any write at step t
     - compute top-k retrieval from current prototypes with h_t
     - compute NLL and leave-one-out gains for selected objects
  2. WRITE using the historical action logged for step t
     - CREATE / UPDATE / PROMOTE exactly as the original trajectory did
     - memory becomes M_{t+1}

Hard invariant at every read:
  creation_step < stream_step   for ALL retrieved objects
  (READ-before-WRITE: step t cannot retrieve what step t just wrote)

State-history validation:
  100 randomly sampled objects × 5 stream-step checkpoints each
  Independent reconstruction from events; comparison against replay snapshot.
  Required: support exact, token counts exact, proto cosine >= 0.999999

Requires (student-model hidden states, 768-dim, NOT teacher/datastore 1024-dim):
  source/states/train.pt             (preferred)
  source/states/controller_train.pt  (fallback — may be a subset)
  Fields: h [N, D], y [N], p_gpt_true [N], nll_gpt [N]

Outputs in {output}/temporal_replay/:
  {traj}_replay_reads.parquet    (stream_step, object_id, gain_j, sim, delay, ...)
  {traj}_replay_writes.parquet   (WRITE events annotated with object_age)
  {traj}_replay_summary.json

Outputs in {output}/validation/:
  state_history_validation.json
  {traj}_state_history_failure_samples.csv   (>= 50 per-field mismatch rows)
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import set_seed

EPS         = 1e-10
FIXED_K     = 4
FIXED_TAU   = 0.05
FIXED_ALPHA = 0.75
FIXED_BETA  = 0.0
VOCAB       = 50257

WRITE_TYPES = {"CREATE", "CREATE_BUFFER", "UPDATE_BUFFER", "UPDATE_BUFFER_PROMOTE",
               "UPDATE_STATE", "PROMOTE", "PROMOTE_AND_UPDATE"}


# ── NLL computation ───────────────────────────────────────────────────────────

def _build_global_freq(y: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(y.long(), minlength=VOCAB).float()
    return (counts + 1.0) / (counts.sum() + VOCAB)


def _nll_and_gains(
    y_t: int, p_gpt: float, nll_gpt: float,
    top_oids: list, top_sims: torch.Tensor,
    token_counts: dict, support: dict,
    tau: float, alpha: float, beta: float, p_global_y: float,
) -> tuple[float, dict]:
    """
    Returns (nll_full, {oid: gain_j}).
    gain_j = NLL_without_j − NLL_full  (positive = object helped)
    """
    k = len(top_oids)
    if k == 0:
        return nll_gpt, {}

    weights  = torch.softmax(top_sims / (tau + EPS), dim=0)

    p_state  = torch.tensor(
        [(token_counts.get(oid, {}).get(y_t, 0) + beta * p_global_y)
         / (support.get(oid, 0) + beta + EPS)
         for oid in top_oids],
        dtype=torch.float32,
    )

    p_local  = float((weights * p_state).sum().clamp(EPS))
    p_full   = max(alpha * p_gpt + (1.0 - alpha) * p_local, EPS)
    nll_full = -np.log(p_full)

    gains = {}
    for i, oid in enumerate(top_oids):
        if k == 1:
            nll_j = nll_gpt
        else:
            rem      = [j for j in range(k) if j != i]
            w_rem    = torch.softmax(top_sims[rem] / (tau + EPS), dim=0)
            pl_rem   = float((w_rem * p_state[rem]).sum().clamp(EPS))
            nll_j    = -np.log(max(alpha * p_gpt + (1.0 - alpha) * pl_rem, EPS))
        gains[oid] = nll_j - nll_full

    return nll_full, gains


# ── Incremental memory ────────────────────────────────────────────────────────

class IncrementalMemory:
    """Incrementally reconstructed memory.  All h vectors are on CPU."""

    def __init__(self):
        self._sum_h:   dict[int, tuple] = {}   # oid → (sum_h Tensor, count)
        self.tok_cnt:  dict[int, dict]  = {}   # oid → {token: count}
        self.support:  dict[int, int]   = {}   # oid → total writes
        self.birth:    dict[int, int]   = {}   # oid → creation stream_step
        self.obj_type: dict[int, str]   = {}   # oid → "buffer"|"state"

    def ids(self) -> list:
        return list(self._sum_h.keys())

    def proto(self, oid: int) -> torch.Tensor:
        s, n = self._sum_h[oid]
        return s / n

    def snapshot(self, oid: int) -> dict | None:
        if oid not in self._sum_h:
            return None
        s, n = self._sum_h[oid]
        return {
            "support":      self.support.get(oid, 0),
            "token_counts": dict(self.tok_cnt.get(oid, {})),
            "proto_h":      (s / n).clone(),
            "n_writes":     n,
        }

    def create(self, oid: int, h: torch.Tensor, y: int,
               step: int, otype: str = "buffer"):
        self._sum_h[oid]  = (h.clone(), 1)
        self.tok_cnt[oid] = {y: 1}
        self.support[oid] = 1
        self.birth[oid]   = step
        self.obj_type[oid] = otype

    def update(self, oid: int, h: torch.Tensor, y: int):
        if oid not in self._sum_h:
            return
        ps, pn           = self._sum_h[oid]
        self._sum_h[oid] = (ps + h, pn + 1)
        tc = self.tok_cnt[oid]
        tc[y] = tc.get(y, 0) + 1
        self.support[oid] = self.support.get(oid, 0) + 1

    def promote(self, oid: int):
        if oid in self.obj_type:
            self.obj_type[oid] = "state"


def _apply_event(mem: IncrementalMemory, act: str, oid: int,
                 h_t: torch.Tensor, y_t: int, step: int):
    act = act.upper()
    if act == "DEFER" or oid < 0:
        return
    if act in ("CREATE_BUFFER",):
        mem.create(oid, h_t, y_t, step, "buffer")
    elif act == "CREATE":
        mem.create(oid, h_t, y_t, step, "state")
    elif act in ("UPDATE_BUFFER", "UPDATE_STATE"):
        if oid not in mem._sum_h:
            mem.create(oid, h_t, y_t, step, "buffer" if "BUFFER" in act else "state")
        else:
            mem.update(oid, h_t, y_t)
    elif act in ("PROMOTE", "PROMOTE_AND_UPDATE", "UPDATE_BUFFER_PROMOTE"):
        # UPDATE_BUFFER_PROMOTE: oracle emits this when an object hits promotion
        # support on the same step as an update — must update h AND promote.
        if oid in mem.obj_type:
            mem.promote(oid)
        else:
            mem.create(oid, h_t, y_t, step, "state")
        if act in ("PROMOTE_AND_UPDATE", "UPDATE_BUFFER_PROMOTE"):
            mem.update(oid, h_t, y_t)


# ── State-history validation ─────────────────────────────────────────────────

def _recon_at_step(oid: int, events_df: pd.DataFrame,
                   h_train: torch.Tensor, t: int) -> dict | None:
    """Independently reconstruct oid's memory state BEFORE the write at step t.

    Uses strict < t (not <=) because snapshots are taken BEFORE applying the write
    event at stream_step == cp_t.  The write at exactly step t is NOT yet applied.

    Pure PROMOTE events change obj_type only — they do not add an h vector — so
    they are skipped when accumulating sum_h / support.
    """
    evts = events_df[
        (events_df["object_id"] == oid) &
        (events_df["stream_step"] < t) &     # strict < : exclude write at t itself
        (~events_df["action_type"].str.upper().isin({"DEFER"}))
    ].sort_values("stream_step")

    if len(evts) == 0:
        return None

    sum_h = None
    n     = 0
    tc: dict = {}
    sup   = 0

    for _, ev in evts.iterrows():
        s   = int(ev["stream_step"])
        act = str(ev.get("action_type", "CREATE")).upper()
        y   = int(ev["true_token"])

        if s >= len(h_train):
            continue

        if act == "PROMOTE":
            # Pure promotion — changes obj_type but does NOT add an h vector.
            continue

        h_i   = h_train[s].float()
        sum_h = h_i.clone() if sum_h is None else sum_h + h_i
        n    += 1
        tc[y] = tc.get(y, 0) + 1
        sup  += 1

    if sum_h is None or n == 0:
        return None
    return {"support": sup, "token_counts": tc, "proto_h": sum_h / n, "n_writes": n}


def run_state_history_validation(
    snapshots: dict,          # {(oid, checkpoint_t): snap_dict}
    events_df: pd.DataFrame,
    h_train: torch.Tensor,
    required_cosine: float = 0.999999,
    traj: str = "unknown",
    val_dir: Path | None = None,
) -> dict:
    """
    Compare replay snapshots against independent reconstruction at same checkpoint.
    Produces detailed failure diagnostics (>= 50 samples) saved to CSV.
    """
    if not snapshots:
        return {"status": "NO_SNAPSHOTS", "pass": False}

    n_total       = 0
    n_sup_ok      = 0
    n_tok_ok      = 0
    n_proto_ok    = 0
    n_absence_ok  = 0
    n_absence_fail = 0
    details       = []
    failure_rows: list[dict] = []   # detailed per-field failures for diagnostics

    # Pre-index events by object_id for fast lookup
    obj_events_cache: dict[int, pd.DataFrame] = {}

    def _get_obj_events(oid: int) -> pd.DataFrame:
        if oid not in obj_events_cache:
            obj_events_cache[oid] = events_df[
                (events_df["object_id"] == oid) &
                (~events_df["action_type"].str.upper().isin({"DEFER"}))
            ].sort_values("stream_step")
        return obj_events_cache[oid]

    for (oid, t), snap in snapshots.items():
        recon = _recon_at_step(oid, events_df, h_train, t)

        # Pre-creation absence check: snap=None means we expected object absent
        if snap is None:
            if recon is None:
                n_absence_ok += 1    # correctly absent in both
            else:
                n_absence_fail += 1  # object in reconstruction but not in replay
            continue

        if recon is None:
            continue
        n_total += 1

        sup_ok = int(snap["support"]) == int(recon["support"])
        tok_ok = snap["token_counts"] == recon["token_counts"]

        p1 = F.normalize(snap["proto_h"].unsqueeze(0), dim=-1)[0]
        p2 = F.normalize(recon["proto_h"].unsqueeze(0), dim=-1)[0]
        cos = float((p1 * p2).sum().clamp(-1, 1))
        cos_ok = cos >= required_cosine

        n_sup_ok   += int(sup_ok)
        n_tok_ok   += int(tok_ok)
        n_proto_ok += int(cos_ok)

        if len(details) < 50:
            details.append({
                "object_id":     int(oid),
                "checkpoint":    int(t),
                "support_match": sup_ok,
                "token_match":   tok_ok,
                "proto_cosine":  round(cos, 8),
            })

        # Detailed diagnostics for failures (up to 200 rows total)
        if not (sup_ok and tok_ok and cos_ok) and len(failure_rows) < 200:
            obj_evts = _get_obj_events(oid)
            before   = obj_evts[obj_evts["stream_step"] < t]
            at_after = obj_evts[obj_evts["stream_step"] >= t]

            last_wr_step   = int(before["stream_step"].iloc[-1])  if len(before)   > 0 else None
            last_wr_action = str(before["action_type"].iloc[-1])  if len(before)   > 0 else None
            next_wr_step   = int(at_after["stream_step"].iloc[0]) if len(at_after) > 0 else None
            next_wr_action = str(at_after["action_type"].iloc[0]) if len(at_after) > 0 else None

            base = {
                "trajectory":        traj,
                "object_id":         int(oid),
                "stream_step":       int(t),
                "last_write_step":   last_wr_step,
                "last_write_action": last_wr_action,
                "next_write_step":   next_wr_step,
                "next_write_action": next_wr_action,
            }

            if not sup_ok:
                abs_err = abs(int(snap["support"]) - int(recon["support"]))
                failure_rows.append({
                    **base,
                    "field":          "support",
                    "replayed_value": int(snap["support"]),
                    "expected_value": int(recon["support"]),
                    "absolute_error": abs_err,
                    "relative_error": round(abs_err / max(int(recon["support"]), 1), 6),
                })
            if not tok_ok:
                snap_total = sum(snap["token_counts"].values())
                recon_total = sum(recon["token_counts"].values())
                failure_rows.append({
                    **base,
                    "field":          "token_counts_total",
                    "replayed_value": snap_total,
                    "expected_value": recon_total,
                    "absolute_error": abs(snap_total - recon_total),
                    "relative_error": round(abs(snap_total - recon_total) / max(recon_total, 1), 6),
                })
            if not cos_ok:
                failure_rows.append({
                    **base,
                    "field":          "proto_cosine",
                    "replayed_value": round(cos, 8),
                    "expected_value": 1.0,
                    "absolute_error": round(1.0 - cos, 8),
                    "relative_error": None,
                })

    if n_total == 0:
        return {"status": "NO_VALID_COMPARISONS", "pass": False}

    sup_rate   = n_sup_ok   / n_total
    tok_rate   = n_tok_ok   / n_total
    proto_rate = n_proto_ok / n_total

    print(f"\n  State history validation ({n_total} checkpoint comparisons):")
    print(f"    Support exact match:     {sup_rate:.1%}")
    print(f"    Token-count exact match: {tok_rate:.1%}")
    print(f"    Prototype cosine ≥ {required_cosine}: {proto_rate:.1%}")

    if failure_rows:
        print(f"    Failure samples collected: {len(failure_rows)}")
        if val_dir is not None:
            val_dir.mkdir(parents=True, exist_ok=True)
            fail_path = val_dir / f"{traj}_state_history_failure_samples.csv"
            pd.DataFrame(failure_rows).to_csv(fail_path, index=False)
            print(f"    Saved failure diagnostics → {fail_path}")

    n_absence_total = n_absence_ok + n_absence_fail
    absence_ok_rate = round(n_absence_ok / max(n_absence_total, 1), 6)
    n_obj_checked   = len(set(oid for (oid, _t) in snapshots))

    if n_absence_total:
        print(f"    Pre-creation absence checks: {n_absence_ok}/{n_absence_total} correct "
              f"({absence_ok_rate:.1%})")

    overall_pass = (sup_rate == 1.0 and tok_rate == 1.0 and proto_rate == 1.0
                    and n_absence_fail == 0)

    return {
        "n_objects_checked":        n_obj_checked,
        "n_comparisons":            n_total,
        "n_absence_checks":         n_absence_total,
        "precreation_absence_rate": absence_ok_rate,
        "support_exact_rate":       round(sup_rate, 6),
        "token_exact_rate":         round(tok_rate, 6),
        "proto_cosine_rate":        round(proto_rate, 6),
        "required_cosine":          required_cosine,
        "pass":                     overall_pass,
        "n_failure_samples":        len(failure_rows),
        "details":                  details,
    }


# ── Main replay loop ──────────────────────────────────────────────────────────

def run_replay(
    traj: str,
    events_df: pd.DataFrame,
    h_train: torch.Tensor,      # [N, D] student-model h vectors
    y_train: torch.Tensor,      # [N] true tokens
    p_gpt_train: torch.Tensor,  # [N] GPT base prob of true token
    nll_gpt_train: torch.Tensor,# [N] GPT base NLL
    max_steps: int | None,
    seed: int,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:

    rng   = np.random.default_rng(seed)
    N     = len(h_train)
    T     = min(max_steps, N) if max_steps else N

    print(f"\n  Training h vectors: {N:,} steps × {h_train.shape[1]}-dim")
    print(f"  Replay steps: {T:,}")

    # Build global token distribution for beta-smoothing
    P_global = _build_global_freq(y_train[:T])

    # Group events by stream_step for O(1) lookup
    events_by_step: dict[int, list] = defaultdict(list)
    for _, ev in events_df.iterrows():
        s = int(ev["stream_step"])
        if s < T:
            events_by_step[s].append(ev.to_dict())

    # Pre-select objects and checkpoints for state-history validation
    write_events = events_df[
        (events_df["action_type"].str.upper().isin(WRITE_TYPES)) &
        (events_df["object_id"] >= 0) &
        (events_df["stream_step"] < T)
    ]
    oid_counts = write_events.groupby("object_id").size()

    # Rich objects (≥ 5 writes): sample up to 100, checkpoints at write-event steps
    rich = oid_counts[oid_counts >= 5].index.tolist()
    n_rich = min(100, len(rich))
    rich_sample = set(rng.choice(rich, n_rich, replace=False).tolist() if rich else [])

    # Sparse objects (1-4 writes): take ALL; use time-based checkpoints after creation
    sparse = oid_counts[(oid_counts >= 1) & (oid_counts < 5)].index.tolist()
    sparse_sample = set(sparse)

    val_oids = rich_sample | sparse_sample

    # For each selected object, assign validation checkpoints
    val_checkpoints: dict[int, list[int]] = {}
    for oid in val_oids:
        steps = sorted(
            write_events[write_events["object_id"] == oid]["stream_step"].tolist()
        )
        if not steps:
            continue
        create_step = steps[0]

        if oid in rich_sample:
            # Rich: 5 checkpoints spread across write-event stream steps
            cp_idx = np.linspace(0, len(steps) - 1, 5).astype(int)
            checkpoints = [steps[i] for i in cp_idx]
        else:
            # Sparse/CREATE-only: pre-creation absence check (create_step itself)
            # + time-based post-creation checkpoints so object state is stable
            post = []
            for offset in [1, 2, 50, 500]:
                c = create_step + offset
                if c < T:
                    post.append(c)
            late = max(create_step + 1000, T - 1)
            if create_step < late < T and late not in post:
                post.append(late)
            checkpoints = [create_step] + sorted(set(post))[:4]

        val_checkpoints[oid] = checkpoints

    # Snapshots: {(oid, checkpoint_t): state_dict}
    snapshots: dict = {}

    # ── Replay loop ───────────────────────────────────────────────────────────
    mem        = IncrementalMemory()
    read_rows  = []
    n_read_steps   = 0
    n_read_objects = 0
    violations     = 0

    for t in tqdm(range(T), desc=f"  [{traj}] stream steps"):
        h_t      = h_train[t].float()
        y_t      = int(y_train[t])
        p_gpt_t  = float(p_gpt_train[t])
        nll_gpt_t = float(nll_gpt_train[t])
        p_global_y = float(P_global[y_t])

        active = mem.ids()

        # ── READ using M_t ────────────────────────────────────────────────────
        if active:
            # Build prototype matrix  [B, D]
            protos = torch.stack([mem.proto(oid) for oid in active])
            p_norm = F.normalize(protos, dim=-1)
            h_norm = F.normalize(h_t.unsqueeze(0), dim=-1)[0]
            sims   = (p_norm @ h_norm).clamp(-1, 1)

            k_eff  = min(FIXED_K, len(active))
            top_v, top_i = sims.topk(k_eff)
            top_oids     = [active[i] for i in top_i.tolist()]
            top_sims     = top_v

            # Invariant: all retrieved objects were born before t
            for oid in top_oids:
                born = mem.birth.get(oid, t)
                if born >= t:
                    violations += 1

            nll_full, gains = _nll_and_gains(
                y_t, p_gpt_t, nll_gpt_t,
                top_oids, top_sims,
                mem.tok_cnt, mem.support,
                FIXED_TAU, FIXED_ALPHA, FIXED_BETA, p_global_y,
            )

            n_read_steps += 1
            for oid, gain_j in gains.items():
                born  = mem.birth.get(oid, 0)
                delay = t - born
                read_rows.append({
                    "stream_step":      t,
                    "object_id":        oid,
                    "gain_j":           gain_j,
                    "sim":              float(top_sims[top_oids.index(oid)]),
                    "nll_full":         nll_full,
                    "nll_gpt":          nll_gpt_t,
                    "true_token":       y_t,
                    "object_age":       t - born,
                    "creation_step":    born,
                    "support_at_query": mem.support.get(oid, 0),
                    "delay":            delay,
                    "trajectory":       traj,
                })
                n_read_objects += 1

        # ── Snapshot for validation (BEFORE write at t) ───────────────────────
        for oid, checkpoints in val_checkpoints.items():
            for cp_t in checkpoints:
                if cp_t == t and (oid, cp_t) not in snapshots:
                    if oid in mem._sum_h:
                        snapshots[(oid, cp_t)] = mem.snapshot(oid)
                    elif oid in sparse_sample:
                        # Pre-creation absence check: record None so validation
                        # can confirm the object is correctly absent before CREATE
                        snapshots[(oid, cp_t)] = None

        # ── WRITE using historical action at step t ───────────────────────────
        for ev in events_by_step[t]:
            _apply_event(mem, ev["action_type"], int(ev["object_id"]),
                         h_t, y_t, t)

    # ── Collect write events within replay window ─────────────────────────────
    write_ev_in_range = events_df[
        (events_df["stream_step"] < T) &
        (events_df["action_type"].str.upper().isin(WRITE_TYPES)) &
        (events_df["object_id"] >= 0)
    ].copy()
    write_ev_in_range["object_age"] = write_ev_in_range.apply(
        lambda r: int(r["stream_step"]) - mem.birth.get(int(r["object_id"]), int(r["stream_step"])),
        axis=1,
    )

    reads_df  = pd.DataFrame(read_rows)
    writes_df = write_ev_in_range

    summary = {
        "trajectory":       traj,
        "T_replay":         T,
        "n_read_steps":     n_read_steps,
        "n_read_objects":   n_read_objects,
        "n_write_events":   len(writes_df),
        "n_active_objects": len(mem.ids()),
        "invariant_violations": violations,
        "invariant_ok":         violations == 0,
    }

    if len(reads_df) > 0:
        summary["delay_min"]  = int(reads_df["delay"].min())
        summary["delay_max"]  = int(reads_df["delay"].max())
        summary["delay_mean"] = float(reads_df["delay"].mean())
        neg = int((reads_df["delay"] < 0).sum())
        summary["neg_delay_count"] = neg
        summary["neg_delay_ok"]    = neg == 0

    print(f"\n  Replay complete:")
    print(f"    Read steps:   {n_read_steps:,}")
    print(f"    Read records: {n_read_objects:,}")
    print(f"    Write events: {len(writes_df):,}")
    print(f"    Active objs:  {len(mem.ids()):,}")
    print(f"    Invariant violations (READ before birth): {violations}")

    return reads_df, writes_df, snapshots, summary, mem, val_checkpoints


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_train_data(source: Path) -> dict:
    """
    Load student-model training hidden states from the source directory.

    Priority order:
      1. states/train.pt          -- explicitly built training h vectors
      2. states/datastore.pt      -- the main student datastore (same data used
                                     by build_write_object_provenance.py line 92)
      3. states/controller_train.pt -- controller training subset (DIFFERENT
                                       sequence, last resort, likely to fail
                                       stream-alignment checks)

    The 'h' field must be D=768 (student GPT-2 base), NOT 1024 (teacher).
    The file MUST be the same source as what build_write_object_provenance.py
    used when building the events — otherwise stream_step indices won't align.
    """
    candidates = [
        source / "states" / "train.pt",
        source / "states" / "datastore.pt",        # ← primary: same file as provenance builder
        source / "states" / "controller_train.pt",  # ← last resort: likely different sequence
    ]
    for p in candidates:
        if p.exists():
            print(f"  Loading training data from: {p.name}")
            data = torch.load(p, weights_only=False, map_location="cpu")
            h = data["h"]
            print(f"    h shape: {tuple(h.shape)}  (must be [N, 768])")
            if h.shape[-1] != 768:
                print(f"    WARNING: h is {h.shape[-1]}-dim, not 768 — "
                      f"this may be the teacher model (wrong embedding space).")
            if p.name == "controller_train.pt":
                print(f"    WARNING: controller_train.pt is a different sequence "
                      f"from the provenance events. Stream-step indices will NOT "
                      f"align. Use datastore.pt or train.pt instead.")
            return data

    raise FileNotFoundError(
        f"Training hidden-state file not found in {source / 'states'}/\n"
        f"Tried: {[p.name for p in candidates]}\n"
        f"The replay requires the SAME h-vector file that build_write_object_provenance.py\n"
        f"used (source/states/datastore.pt).  stream_step indices must align."
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory",    required=True, choices=["oracle", "bandit"])
    ap.add_argument("--source",        required=True,
                    help="Source dir with states/train.pt (student model h vectors)")
    ap.add_argument("--oracle_dir",    required=True)
    ap.add_argument("--bandit_dir",    default=None)
    ap.add_argument("--output",        required=True)
    ap.add_argument("--device",        default="cuda",
                    help="torch device for similarity computation (cuda or cpu)")
    ap.add_argument("--max_steps",     type=int, default=None,
                    help="Limit replay to first N stream steps (None = all)")
    ap.add_argument("--seed",          type=int, default=42)
    ap.add_argument("--force",         action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    traj    = args.trajectory
    source  = Path(args.source)
    out_dir = Path(args.output)
    rep_dir = out_dir / "temporal_replay"
    val_dir = out_dir / "validation"
    rep_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    reads_path   = rep_dir / f"{traj}_replay_reads.parquet"
    writes_path  = rep_dir / f"{traj}_replay_writes.parquet"
    summary_path = rep_dir / f"{traj}_replay_summary.json"
    val_path     = val_dir / "state_history_validation.json"

    if reads_path.exists() and writes_path.exists() and not args.force:
        # Also check that the validation JSON has a valid entry for this trajectory.
        # If n_comparisons is absent (None / key missing), the validation is stale
        # from a pre-fix run and Stage 03 must re-run to regenerate it.
        val_stale = True
        if val_path.exists():
            try:
                entry = json.loads(val_path.read_text()).get(traj, {})
                if entry.get("n_comparisons") is not None and "pass" in entry:
                    val_stale = False
            except Exception:
                pass
        if not val_stale:
            print(f"[cached] {reads_path.name}, {writes_path.name}")
            return
        print(f"  Replay parquets exist but validation for '{traj}' is stale "
              f"(n_comparisons absent) — re-running to regenerate.")

    # Load provenance events
    prov_dir = out_dir / "provenance"
    evt_path = prov_dir / f"{traj}_events.parquet"
    if not evt_path.exists():
        print(f"ERROR: {evt_path} not found. Run build_write_object_provenance.py first.")
        sys.exit(1)
    events_df = pd.read_parquet(evt_path)
    print(f"Loaded {len(events_df):,} events for {traj} "
          f"({(events_df['action_type'] == 'DEFER').sum():,} DEFER, "
          f"{events_df['action_type'].str.upper().isin(WRITE_TYPES).sum():,} WRITE)")

    # Load training h vectors (student model, 768-dim)
    train_data  = _load_train_data(source)
    h_train     = train_data["h"].float()
    y_train     = train_data["y"].long()
    p_gpt_train = train_data["p_gpt_true"].float()
    nll_gpt_t   = train_data["nll_gpt"].float()

    print(f"Training stream: {len(h_train):,} steps, h-dim={h_train.shape[1]}")

    reads_df, writes_df, snapshots, summary, mem, val_checkpoints = run_replay(
        traj, events_df, h_train, y_train, p_gpt_train, nll_gpt_t,
        args.max_steps, args.seed, out_dir,
    )

    if len(reads_df) == 0:
        print("WARNING: Zero read records produced. "
              "Check that events.parquet and train.pt are aligned.")
    else:
        neg = int((reads_df["delay"] < 0).sum())
        if neg > 0:
            print(f"AUDIT INVALID: {neg} READ records have delay < 0 (pre-creation access).")
        else:
            print(f"  All delays >= 0. Invariant satisfied.")

    # State history validation (with detailed failure diagnostics saved to CSV)
    val_result = run_state_history_validation(
        snapshots, events_df, h_train, traj=traj, val_dir=val_dir
    )
    # Merge results from both trajectories (accumulate)
    existing_val = {}
    if val_path.exists():
        with open(val_path) as f:
            existing_val = json.load(f)
    existing_val[traj] = val_result
    with open(val_path, "w") as f:
        json.dump(existing_val, f, indent=2, default=str)
    print(f"  State history validation: pass={val_result.get('pass')}")

    # Bandit reconstruction diagnostics (explains any n_comparisons=None root cause)
    if traj == "bandit":
        write_ev    = events_df[events_df["action_type"].str.upper().isin(WRITE_TYPES)]
        prov_oids   = set(write_ev["object_id"].unique().tolist())
        replay_oids = set(mem.ids())
        n_cands     = sum(len(v) for v in val_checkpoints.values())
        n_captured  = sum(1 for v in snapshots.values() if v is not None)
        n_absence   = sum(1 for v in snapshots.values() if v is None)
        n_rejected  = n_cands - len(snapshots)
        create_ev   = events_df[events_df["action_type"].str.upper().isin(
                          {"CREATE", "CREATE_BUFFER"})]
        diag = {
            "n_bandit_objects":              int(len(prov_oids)),
            "n_bandit_create_events":        int(len(create_ev)),
            "n_bandit_update_events":        int((events_df["action_type"].str.upper().isin(
                                                  {"UPDATE_BUFFER", "UPDATE_STATE"})).sum()),
            "n_bandit_promote_events":       int((events_df["action_type"].str.upper() ==
                                                  "UPDATE_BUFFER_PROMOTE").sum()),
            "n_defer_events":                int((events_df["action_type"].str.upper() ==
                                                  "DEFER").sum()),
            "n_objects_found_in_replay":     int(len(replay_oids)),
            "n_objects_found_in_provenance": int(len(prov_oids)),
            "n_matching_object_ids":         int(len(prov_oids & replay_oids)),
            "n_val_oids_selected":           len(val_checkpoints),
            "n_candidate_snapshots":         n_cands,
            "n_captured_snapshots":          n_captured,
            "n_absence_captures":            n_absence,
            "n_rejected_snapshots":          n_rejected,
            "first_20_provenance_object_ids": sorted(list(prov_oids))[:20],
            "first_20_replay_object_ids":     sorted(list(replay_oids))[:20],
            "earliest_creation_step":        int(create_ev["stream_step"].min())
                                             if len(create_ev) > 0 else None,
            "latest_creation_step":          int(create_ev["stream_step"].max())
                                             if len(create_ev) > 0 else None,
            "replay_steps_available":        int(summary["T_replay"]),
            "c06_fixed":                     val_result.get("n_comparisons", 0) > 0,
            "c06_n_comparisons":             val_result.get("n_comparisons"),
            "c06_pass":                      val_result.get("pass"),
        }
        diag_path = val_dir / "bandit_reconstruction_diagnostics.json"
        with open(diag_path, "w") as f:
            json.dump(diag, f, indent=2, default=str)
        print(f"  Bandit diagnostics → {diag_path}")

    # Save outputs
    reads_df.to_parquet(reads_path, index=False)
    writes_df.to_parquet(writes_path, index=False)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nSaved:")
    print(f"  {reads_path}   ({len(reads_df):,} rows)")
    print(f"  {writes_path}  ({len(writes_df):,} rows)")


if __name__ == "__main__":
    main()

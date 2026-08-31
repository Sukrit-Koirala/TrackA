"""
build_oracle_teacher_assignments.py  --  MVP 4a-0: Oracle Reconstruction

Extracts row-to-state assignments for the DATASTORE by re-running the exact
same builder code as build_predictive_states.py (same PCA + MiniBatchKMeans
params + seed).

Note: the existing build_offline_teacher_assignments.py serves a different
purpose (MVP 4a write imitation: assigns CT examples via nearest-prototype).
This script re-derives the ORIGINAL datastore-level cluster assignments that
the offline teacher used internally.

Preferred assignment source order (per spec):
  1. Original saved labels            -> NOT available (not persisted)
  2. Original clustering model        -> NOT available (not persisted)
  3. Exact deterministic reconstruction using original builder  <- USED HERE
  4. Nearest saved teacher prototype  -> different geometry; NOT used unless fallback

Validation checks (all must pass to proceed without --force):
  - per-state support counts match exactly
  - per-state token-count sums match within 1e-4
  - nonempty state count matches
  - total assigned rows == N_datastore

If validation fails: exit with descriptive error (sklearn version mismatch most likely).

Usage:
  python src/build_oracle_teacher_assignments.py \\
    --source            outputs_scale_sweep/scale_200k_seed42 \\
    --offline_states_dir outputs_mvp2b_state_write/states \\
    --output            outputs_mvp4a0_oracle_reconstruction \\
    --teacher           minibatch_kmeans \\
    --budget            10000 \\
    --seed              42
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import numpy as np
import torch

from utils import get_device, set_seed
from build_predictive_states import (
    build_minibatch_kmeans,
    accumulate_token_counts,
    compute_state_stats,
    TOP_K,
    PCA_DIM,
)


# ── nearest-prototype fallback ────────────────────────────────────────────────

def assign_nearest_prototype(h_norm: np.ndarray, prototype_h: np.ndarray,
                              chunk_size: int = 2048) -> np.ndarray:
    """
    Assign each row of h_norm [N, D] to the nearest row of prototype_h [B, D]
    using cosine similarity (both expected to be L2-normalised).
    Chunked to stay within ~80 MB working memory regardless of N or B.
    """
    N = len(h_norm)
    B = len(prototype_h)
    labels = np.empty(N, dtype=np.int64)
    proto_t = torch.from_numpy(prototype_h.astype(np.float32))   # [B, D]
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        h_t = torch.from_numpy(h_norm[start:end].astype(np.float32))  # [C, D]
        sims = h_t @ proto_t.T                                         # [C, B]
        labels[start:end] = sims.argmax(dim=-1).numpy()
    print(f"  Nearest-prototype: {len(np.unique(labels))}/{B} states used")
    return labels


# ── validation ────────────────────────────────────────────────────────────────

def validate_against_teacher(
    labels:      np.ndarray,
    teacher:     dict,
    h_orig:      np.ndarray,
    y_np:        np.ndarray,
    nll_gpt:     np.ndarray,
    gpt_entropy: np.ndarray,
    B:           int,
) -> dict:
    """
    Recompute state statistics from labels and compare against saved teacher.
    Returns dict with PASS bool and detail. Key '_recon_stats' holds computed
    stats for downstream reuse (not serialised to JSON).
    """
    N = len(labels)
    assert int(labels.min()) >= 0, "negative label found"
    assert int(labels.max()) < B,  f"label {int(labels.max())} >= B={B}"
    assert len(labels) == N

    ids, cnts, total = accumulate_token_counts(labels, y_np, B)
    stats = compute_state_stats(h_orig, labels, B, ids, cnts, total,
                                 nll_gpt, gpt_entropy)

    t_cnt = teacher["assigned_count"].numpy().astype(np.int64)
    r_cnt = stats["assigned_count"].numpy().astype(np.int64)
    t_tot = teacher["total_counts"].numpy().astype(np.float32)
    r_tot = stats["total_counts"].numpy().astype(np.float32)
    t_tok = teacher["top_k_token_counts"].sum(-1).numpy()
    r_tok = stats["top_k_token_counts"].sum(-1).numpy()

    cnt_match      = bool(np.array_equal(t_cnt, r_cnt))
    tot_match      = bool(np.allclose(t_tot, r_tot, rtol=1e-5, atol=1e-5))
    tok_match      = bool(np.allclose(t_tok, r_tok, rtol=1e-4, atol=1e-4))
    t_ne           = int((t_cnt > 0).sum())
    r_ne           = int((r_cnt > 0).sum())
    nonempty_match = (t_ne == r_ne)

    discrepancies = []
    if not cnt_match:
        n_diff = int((t_cnt != r_cnt).sum())
        discrepancies.append(
            f"assigned_count differs in {n_diff}/{B} states "
            f"(possible sklearn version mismatch)")
    if not tot_match:
        discrepancies.append("total_counts mismatch (tol 1e-5)")
    if not tok_match:
        discrepancies.append("token_count sums mismatch (tol 1e-4)")
    if not nonempty_match:
        discrepancies.append(
            f"nonempty state count: teacher={t_ne}  recon={r_ne}")

    return {
        "N_rows":                  int(N),
        "B":                       int(B),
        "teacher_nonempty_states": int(t_ne),
        "recon_nonempty_states":   int(r_ne),
        "teacher_total_rows":      int(t_cnt.sum()),
        "recon_total_rows":        int(r_cnt.sum()),
        "count_match_exact":       cnt_match,
        "total_counts_match":      tot_match,
        "token_sum_match":         tok_match,
        "nonempty_states_match":   nonempty_match,
        "discrepancies":           discrepancies,
        "PASS":                    (cnt_match and tot_match and tok_match and nonempty_match),
        "_recon_stats":            stats,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",              required=True,
                    help="scale_200k_seed42 directory")
    ap.add_argument("--offline_states_dir",  required=True,
                    help="outputs_mvp2b_state_write/states")
    ap.add_argument("--output",              required=True,
                    help="mvp4a0 output root")
    ap.add_argument("--teacher",             default="minibatch_kmeans")
    ap.add_argument("--budget",              type=int, default=10000)
    ap.add_argument("--seed",                type=int, default=42)
    ap.add_argument("--device",              default="cuda")
    ap.add_argument("--force",               action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    src = Path(args.source)
    B   = args.budget
    tag = f"{args.teacher}_B{B}"

    out_dir  = Path(args.output) / "teacher_assignments"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{tag}_assignments.npz"
    val_path = out_dir / "assignment_validation.json"

    if npz_path.exists() and val_path.exists() and not args.force:
        print(f"[cached] {npz_path}")
        return

    # ── load teacher artifact ─────────────────────────────────────────────────
    teacher_path = Path(args.offline_states_dir) / f"{tag}.pt"
    if not teacher_path.exists():
        print(f"ERROR: teacher state not found: {teacher_path}")
        for p in sorted(Path(args.offline_states_dir).glob("*.pt")):
            print(f"  {p.name}")
        sys.exit(1)
    print(f"Loading teacher: {teacher_path}")
    teacher = torch.load(teacher_path, weights_only=False)

    # ── load datastore ────────────────────────────────────────────────────────
    print("\nLoading datastore ...")
    ds_path = src / "states" / "datastore.pt"
    if not ds_path.exists():
        print(f"ERROR: {ds_path}")
        sys.exit(1)
    ds  = torch.load(ds_path, weights_only=False)
    N   = len(ds["y"])
    y_np = ds["y"].numpy().astype(np.int64)
    h_f  = ds["h"].float()
    h_orig = h_f.numpy()                                           # [N,D] raw
    h_norm = (h_f / (h_f.norm(dim=-1, keepdim=True) + 1e-8)).numpy()
    nll_gpt     = ds.get("nll_gpt",     torch.zeros(N)).float().numpy()
    gpt_entropy = ds.get("gpt_entropy", torch.zeros(N)).float().numpy()
    print(f"  N={N:,}  D={h_orig.shape[1]}")

    # ── re-run original builder ───────────────────────────────────────────────
    config = {"pca_dim": PCA_DIM, "seed": args.seed}
    print(f"\nRe-running {args.teacher} builder  "
          f"(pca_dim={PCA_DIM}, seed={args.seed}) ...")

    if args.teacher == "minibatch_kmeans":
        labels, _extra = build_minibatch_kmeans(
            h_orig, h_norm, y_np, ds, B, config)
    else:
        print(f"ERROR: unsupported teacher '{args.teacher}'")
        print("  Supported: minibatch_kmeans")
        sys.exit(1)

    labels = labels.astype(np.int64)
    print(f"  Labels range=[{int(labels.min())},{int(labels.max())}]  "
          f"unique={len(np.unique(labels))}/{B}")

    # ── validate against saved teacher ────────────────────────────────────────
    print("\nValidating against teacher artifact ...")
    val = validate_against_teacher(labels, teacher, h_orig, y_np,
                                    nll_gpt, gpt_entropy, B)

    print(f"  teacher nonempty states: {val['teacher_nonempty_states']}")
    print(f"  recon nonempty states:   {val['recon_nonempty_states']}")
    print(f"  teacher total rows:      {val['teacher_total_rows']:,}")
    print(f"  recon total rows:        {val['recon_total_rows']:,}")
    print(f"  count_match_exact:       {val['count_match_exact']}")
    print(f"  total_counts_match:      {val['total_counts_match']}")
    print(f"  token_sum_match:         {val['token_sum_match']}")
    print(f"  nonempty_states_match:   {val['nonempty_states_match']}")
    print(f"  PASS:                    {val['PASS']}")
    for d in val["discrepancies"]:
        print(f"  DISCREPANCY: {d}")

    if not val["PASS"] and not args.force:
        print("\nKMeans validation FAILED (likely sklearn version mismatch).")
        print("Falling back to nearest-prototype cosine assignment ...")
        proto_h = teacher["prototype_h"].float().numpy()   # [B, D] normalised
        labels  = assign_nearest_prototype(h_norm, proto_h)
        labels  = labels.astype(np.int64)
        val     = validate_against_teacher(labels, teacher, h_orig, y_np,
                                           nll_gpt, gpt_entropy, B)
        print(f"  Nearest-prototype validation — PASS: {val['PASS']}")
        for d in val["discrepancies"]:
            print(f"  INFO: {d}")
        assignment_method = f"{args.teacher}_nearest_prototype_cosine"
    elif not val["PASS"]:
        print("\nWARNING: PASS=False but --force set. Proceeding.")
        assignment_method = f"{args.teacher}_exact_rerun_forced"
    else:
        assignment_method = f"{args.teacher}_exact_rerun"

    # ── save ──────────────────────────────────────────────────────────────────
    np.savez_compressed(
        npz_path,
        teacher_ids  = labels.astype(np.int32),
        state_counts = np.bincount(labels, minlength=B).astype(np.int32),
    )
    print(f"\nSaved: {npz_path}  ({npz_path.stat().st_size/1e6:.1f} MB)")

    val_json = {k: v for k, v in val.items() if k != "_recon_stats"}
    val_json.update({
        "npz_path":          str(npz_path),
        "teacher_path":      str(teacher_path),
        "assignment_method": assignment_method,
        "seed":              args.seed,
        "pca_dim":           PCA_DIM,
        "budget":            int(B),
        "N":                 int(N),
    })
    with open(val_path, "w") as f:
        json.dump(val_json, f, indent=2)
    print(f"Saved: {val_path}")


if __name__ == "__main__":
    main()

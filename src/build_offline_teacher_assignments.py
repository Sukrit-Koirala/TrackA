"""
build_offline_teacher_assignments.py  --  MVP 4a Stage 1

Compute teacher assignments for all datastore examples by
assigning each h_i to the nearest offline prototype (cosine sim).

Saves:
  outputs_mvp4a.../teachers/<teacher_name>_assignments.pt
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse, time
import numpy as np
import torch

from utils import get_device, set_seed

EPS = 1e-8
CHUNK = 1024


def _norm(h: torch.Tensor) -> torch.Tensor:
    return h / (h.norm(dim=-1, keepdim=True) + EPS)


def compute_assignments(ct_h_norm: torch.Tensor, proto_norm: torch.Tensor, device, chunk: int = CHUNK) -> torch.Tensor:
    """Chunked argmax cosine similarity. Returns LongTensor[N]."""
    N = ct_h_norm.shape[0]
    B = proto_norm.shape[0]
    proto_gpu = proto_norm.to(device)
    teacher_ids = torch.zeros(N, dtype=torch.long)
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        blk = ct_h_norm[start:end].to(device)     # [C, d]
        sims = blk @ proto_gpu.T                   # [C, B]
        teacher_ids[start:end] = sims.argmax(dim=1).cpu()
    return teacher_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",     required=True,
                        help="scale_200k_seed42 directory")
    parser.add_argument("--states_dir", required=True,
                        help="MVP 2b states directory")
    parser.add_argument("--teachers",   nargs="+",
                        default=["utility_weighted_B10000", "minibatch_kmeans_B10000"])
    parser.add_argument("--output",     required=True,
                        help="mvp4a output root")
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--force",      action="store_true")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device({"device": args.device})
    src    = Path(args.source)
    sdir   = Path(args.states_dir)
    out    = Path(args.output) / "teachers"
    out.mkdir(parents=True, exist_ok=True)

    print(f"\nbuild_offline_teacher_assignments")
    print(f"  Source:   {src}")
    print(f"  States:   {sdir}")
    print(f"  Teachers: {args.teachers}")
    print(f"  Output:   {out}")

    # Load datastore hidden states
    ct_data = torch.load(src / "states" / "controller_train.pt", weights_only=False)
    ct_h = _norm(ct_data["h"].float())       # [N, d]
    ct_y = ct_data["y"]                       # [N]
    N, d = ct_h.shape
    print(f"  N={N:,}  d={d}")

    for teacher_name in args.teachers:
        out_path = out / f"{teacher_name}_assignments.pt"
        if out_path.exists() and not args.force:
            print(f"\n  [cached] {teacher_name}")
            continue

        state_file = sdir / f"{teacher_name}.pt"
        if not state_file.exists():
            print(f"\n  [SKIP] {teacher_name} — state file not found: {state_file}")
            continue

        print(f"\n  [{teacher_name}]")
        t0 = time.time()

        states = torch.load(state_file, weights_only=False)
        proto  = _norm(states["prototype_h"].float())
        B      = len(proto)
        print(f"    Loaded {B} prototypes")

        teacher_ids = compute_assignments(ct_h, proto, device)

        # Compute assignment distribution stats
        counts   = torch.bincount(teacher_ids, minlength=B).float()
        assigned = int((counts > 0).sum())
        print(f"    Assigned {assigned}/{B} prototypes used  "
              f"mean_per_state={N/max(assigned,1):.1f}  "
              f"elapsed={time.time()-t0:.1f}s")

        save = {
            "teacher_name":  teacher_name,
            "budget":        B,
            "teacher_ids":   teacher_ids,          # LongTensor[N]
            "prototype_h":   proto.cpu(),           # [B, d]
            "state_counts":  counts.cpu(),          # [B] assignment counts
            "token_counts":  states.get("top_k_token_ids", None),
        }
        torch.save(save, out_path)
        print(f"    Saved {out_path}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()

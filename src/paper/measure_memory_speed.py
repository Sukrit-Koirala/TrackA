"""
measure_memory_speed.py  --  Track A Paper  (Experiment 3)

Measure storage footprint and retrieval latency for every state file
and raw-memory baseline, plus the full raw datastore as reference.

Outputs:
  efficiency/memory_speed_results.csv
  reports/efficiency_summary.md
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse, json, time
import numpy as np
import torch
import pandas as pd

from utils import get_device, set_seed

WARMUP_QUERIES = 50
TIMED_QUERIES  = 200


# ── memory measurement ────────────────────────────────────────────────────────

def measure_state_memory(sf: dict) -> dict:
    """Compute storage breakdown in MB for an MVP 2b state file."""
    def mb(t):
        if isinstance(t, torch.Tensor):
            return t.nbytes / 1e6
        return 0.0

    key_mb   = mb(sf.get("prototype_h", torch.tensor([])))
    val_mb   = (mb(sf.get("top_k_token_ids",    torch.tensor([]))) +
                mb(sf.get("top_k_token_counts", torch.tensor([]))))
    cnt_mb   = (mb(sf.get("total_counts",    torch.tensor([]))) +
                mb(sf.get("assigned_count",  torch.tensor([]))))
    meta_mb  = (mb(sf.get("state_entropy",   torch.tensor([]))) +
                mb(sf.get("state_purity",    torch.tensor([]))) +
                mb(sf.get("mean_nll",        torch.tensor([]))) +
                mb(sf.get("mean_gpt_entropy",torch.tensor([]))))
    total_mb = key_mb + val_mb + cnt_mb + meta_mb

    return {
        "key_memory_MB":      round(key_mb,   3),
        "value_memory_MB":    round(val_mb,   3),
        "metadata_memory_MB": round(cnt_mb + meta_mb, 3),
        "total_memory_MB":    round(total_mb, 3),
    }


def measure_ds_memory(ds_data: dict) -> dict:
    """Full raw datastore reference."""
    h_mb  = ds_data["h"].nbytes / 1e6
    y_mb  = ds_data["y"].nbytes / 1e6
    total = h_mb + y_mb
    return {
        "key_memory_MB":      round(h_mb,  3),
        "value_memory_MB":    round(y_mb,  3),
        "metadata_memory_MB": 0.0,
        "total_memory_MB":    round(total, 3),
    }


# ── retrieval latency ─────────────────────────────────────────────────────────

def time_state_retrieval(
    proto_h: torch.Tensor,     # [B, D]
    val_h:   torch.Tensor,     # [N_val, D]  normalised
    device:  torch.device,
    k: int = 32,
    n_timed: int = TIMED_QUERIES,
    n_warmup: int = WARMUP_QUERIES,
) -> dict:
    """
    Time top-k cosine retrieval against a state prototype matrix.
    Returns ms per query and queries per second.
    """
    B   = proto_h.shape[0]
    N   = val_h.shape[0]
    D   = proto_h.shape[1]

    proto_gpu = proto_h.float().to(device)
    val_gpu   = val_h.float().to(device)

    # Use single-query batches to measure per-query latency
    n_total = min(n_warmup + n_timed, N)
    queries  = val_gpu[:n_total]

    CHUNK = 512   # query chunk size for batched retrieval
    k_use = min(k, B)

    def run_retrieval(qs):
        """Run chunked top-k retrieval for a batch of queries."""
        results = []
        for s in range(0, len(qs), CHUNK):
            q_chunk = qs[s:s+CHUNK]              # [C, D]
            sims    = q_chunk @ proto_gpu.T       # [C, B]
            topk    = sims.topk(k_use, dim=-1)   # [C, k]
            results.append(topk.indices)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return results

    # Warmup
    run_retrieval(queries[:n_warmup])

    # Timed
    n_timed_actual = min(n_timed, N - n_warmup)
    if n_timed_actual <= 0:
        n_timed_actual = min(n_timed, N)
        timed_qs = queries[:n_timed_actual]
    else:
        timed_qs = queries[n_warmup:n_warmup + n_timed_actual]

    t0 = time.perf_counter()
    run_retrieval(timed_qs)
    elapsed = time.perf_counter() - t0

    ms_per_query = elapsed * 1000 / n_timed_actual
    qps          = n_timed_actual / elapsed

    return {
        "retrieval_ms_per_query":   round(ms_per_query, 4),
        "retrieval_queries_per_sec": round(qps, 1),
        "n_timed_queries":           n_timed_actual,
        "k_retrieved":               k_use,
        "device":                    str(device),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",       required=True,
                        help="scale_200k_seed42 directory")
    parser.add_argument("--output",       required=True,
                        help="paper output directory")
    parser.add_argument("--states_dir",   default=None,
                        help="directory with *_B*.pt state files "
                             "(default: <output>/states)")
    parser.add_argument("--raw_dir",      default=None,
                        help="directory with raw baseline .pt files "
                             "(default: <output>/raw_baselines)")
    parser.add_argument("--methods",      nargs="+", default=None,
                        help="state methods to include (default: auto-discover)")
    parser.add_argument("--budgets",      nargs="+", type=int, default=None)
    parser.add_argument("--k",            type=int, default=32)
    parser.add_argument("--device",       default="cuda")
    parser.add_argument("--force",        action="store_true")
    args = parser.parse_args()

    set_seed(42)
    device   = get_device({"device": args.device})
    src      = Path(args.source)
    out      = Path(args.output)
    eff_dir  = out / "efficiency"
    rep_dir  = out / "reports"
    eff_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    csv_path = eff_dir / "memory_speed_results.csv"
    if csv_path.exists() and not args.force:
        print(f"[cached] {csv_path}")
        return

    states_dir = Path(args.states_dir) if args.states_dir else out / "states"
    raw_dir    = Path(args.raw_dir)    if args.raw_dir    else out / "raw_baselines"

    print(f"\nmeasure_memory_speed")
    print(f"  States:  {states_dir}")
    print(f"  Raw:     {raw_dir}")
    print(f"  Device:  {device}")

    # Load val queries
    val_path = src / "states" / "val.pt"
    val_data = torch.load(val_path, weights_only=False)
    val_h    = val_data["h"].float()
    val_h    = val_h / (val_h.norm(dim=-1, keepdim=True) + 1e-8)
    print(f"  Val queries: {len(val_h):,}")

    rows = []

    # ── full raw datastore reference ──────────────────────────────────────────
    ds_path = src / "states" / "datastore.pt"
    if ds_path.exists():
        print(f"\n  [full_datastore]")
        ds_data = torch.load(ds_path, weights_only=False)
        N_ds    = ds_data["h"].shape[0]
        mem     = measure_ds_memory(ds_data)
        ds_h    = ds_data["h"].float()
        ds_h    = ds_h / (ds_h.norm(dim=-1, keepdim=True) + 1e-8)
        spd     = time_state_retrieval(ds_h, val_h, device, k=args.k)
        rows.append({
            "method": "full_datastore", "budget": N_ds,
            "memory_type": "raw_datastore",
            "num_objects": N_ds,
            **mem, **spd,
            "q_nll": float("nan"), "fixed_nll": float("nan"),
        })
        del ds_data, ds_h

    # ── state codebooks ───────────────────────────────────────────────────────
    if states_dir.exists():
        state_files = sorted(states_dir.glob("*.pt"))
        for sfp in state_files:
            tag = sfp.stem
            print(f"\n  [state] {tag}")
            try:
                sf   = torch.load(sfp, weights_only=False)
                B    = sf["prototype_h"].shape[0]
                mem  = measure_state_memory(sf)
                spd  = time_state_retrieval(sf["prototype_h"], val_h, device, k=args.k)

                method = sf.get("method", tag.split("_B")[0])
                budget = sf.get("budget", B)

                # Try to load Q-metrics if available
                q_nll    = float("nan")
                fixed_nll = float("nan")
                q_dir    = out / "q_read" / "states" / tag
                if not q_dir.exists():
                    q_dir = out / tag   # fallback to MVP 2c layout
                qm_path  = q_dir / "q_metrics.json"
                bfm_path = q_dir / "best_fixed_metrics.json"
                if qm_path.exists():
                    with open(qm_path) as f:
                        q_nll = json.load(f).get("q_state_nll", float("nan"))
                if bfm_path.exists():
                    with open(bfm_path) as f:
                        fixed_nll = json.load(f).get("val_nll", float("nan"))

                rows.append({
                    "method": method, "budget": budget,
                    "memory_type": "state_codebook",
                    "num_objects": B,
                    **mem, **spd,
                    "q_nll": q_nll, "fixed_nll": fixed_nll,
                })
            except Exception as e:
                print(f"    [WARN] {tag}: {e}")

    # ── raw baselines ─────────────────────────────────────────────────────────
    if raw_dir.exists():
        raw_files = sorted(raw_dir.glob("*.pt"))
        for rfp in raw_files:
            tag = rfp.stem
            print(f"\n  [raw] {tag}")
            try:
                sf   = torch.load(rfp, weights_only=False)
                B    = sf["prototype_h"].shape[0]
                mem  = measure_state_memory(sf)
                spd  = time_state_retrieval(sf["prototype_h"], val_h, device, k=args.k)

                method = sf.get("method", tag.split("_B")[0])
                budget = sf.get("budget", B)

                q_nll    = float("nan")
                fixed_nll = float("nan")
                q_dir    = out / "q_read" / "raw_baselines" / tag
                qm_path  = q_dir / "q_metrics.json"
                bfm_path = q_dir / "best_fixed_metrics.json"
                if qm_path.exists():
                    with open(qm_path) as f:
                        q_nll = json.load(f).get("q_state_nll", float("nan"))
                if bfm_path.exists():
                    with open(bfm_path) as f:
                        fixed_nll = json.load(f).get("val_nll", float("nan"))

                rows.append({
                    "method": method, "budget": budget,
                    "memory_type": "raw_examples",
                    "num_objects": B,
                    **mem, **spd,
                    "q_nll": q_nll, "fixed_nll": fixed_nll,
                })
            except Exception as e:
                print(f"    [WARN] {tag}: {e}")

    if not rows:
        print("  No files found.")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(["memory_type", "budget"])
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}  ({len(df)} rows)")

    # ── summary markdown ──────────────────────────────────────────────────────
    lines = [
        "# Experiment 3: Memory and Retrieval Speed\n",
        "| method | budget | type | objects | total_MB | ms/query | q_nll | fixed_nll |",
        "|--------|--------|------|---------|----------|----------|-------|-----------|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['method']:<28} | {int(r['budget']):>7} | {r['memory_type']:<15} |"
            f" {int(r['num_objects']):>7} | {r['total_memory_MB']:>8.2f} |"
            f" {r['retrieval_ms_per_query']:>8.3f} |"
            f" {r['q_nll']!s:>7} | {r['fixed_nll']!s:>9} |"
        )
    summary_path = rep_dir / "efficiency_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {summary_path}")


if __name__ == "__main__":
    main()

"""
analyze_dime_efficiency.py  --  Track A Paper  (Efficiency Analysis)

Loads existing experiment outputs and computes:
  * entry-count compression vs raw kNN
  * byte-level memory estimates (dense vs sparse DIME, conservative vs practical)
  * retrieval latency (key search + value aggregation)
  * NLL tradeoff across sparse top-M values

Does NOT re-run hidden-state extraction.

Usage:
  python src/paper/analyze_dime_efficiency.py \\
    --source      scale_200k_seed42 \\
    --output      outputs_track_a_offline_paper \\
    --dataset     TinyStories \\
    --model       gpt2 \\
    --seeds       42 123 999 \\
    --device      cuda
"""

import sys
import math
import json
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

from utils import get_device, set_seed
from paper.measure_memory_speed import time_state_retrieval

SCRIPT_DIR = Path(__file__).parent

SPARSE_M_VALUES = [4, 8, 16, 32, 64, 128]
VOCAB_SIZE_DEFAULT = 50257   # GPT-2 tokenizer

# dtype byte sizes
BYTES = {"float32": 4, "float16": 2, "int64": 8, "int32": 4}


# ── memory accounting ─────────────────────────────────────────────────────────

def _memory_raw(N: int, D: int, mode: str = "conservative") -> dict:
    """
    Raw kNN entry:  hidden vector (D) + one token id.
    Conservative: float32 key, int64 token id.
    Practical:    float16 key, int32 token id.
    """
    kb = BYTES["float32"] if mode == "conservative" else BYTES["float16"]
    vb = BYTES["int64"]   if mode == "conservative" else BYTES["int32"]
    key_bytes = N * D * kb
    val_bytes = N * vb
    return {
        "key_mb":   key_bytes / 1e6,
        "value_mb": val_bytes / 1e6,
        "meta_mb":  0.0,
        "total_mb": (key_bytes + val_bytes) / 1e6,
        "key_dtype":   f"float{32 if mode=='conservative' else 16}",
        "value_dtype": f"int{64 if mode=='conservative' else 32}",
        "mode": mode,
    }


def _memory_dime_dense(B: int, D: int, V: int, mode: str = "conservative") -> dict:
    """
    DIME dense entry: hidden vector (D) + full-vocab distribution (V floats).
    """
    kb = BYTES["float32"] if mode == "conservative" else BYTES["float16"]
    pb = BYTES["float32"] if mode == "conservative" else BYTES["float16"]
    key_bytes = B * D * kb
    val_bytes = B * V * pb
    return {
        "key_mb":   key_bytes / 1e6,
        "value_mb": val_bytes / 1e6,
        "meta_mb":  0.0,
        "total_mb": (key_bytes + val_bytes) / 1e6,
        "key_dtype":   f"float{32 if mode=='conservative' else 16}",
        "value_dtype": f"float{32 if mode=='conservative' else 16}",
        "mode": mode,
    }


def _memory_dime_sparse(B: int, D: int, M: int, mode: str = "conservative") -> dict:
    """
    DIME sparse top-M entry: hidden vector (D) + M token ids + M probabilities.
    """
    kb = BYTES["float32"] if mode == "conservative" else BYTES["float16"]
    ib = BYTES["int64"]   if mode == "conservative" else BYTES["int32"]
    pb = BYTES["float32"] if mode == "conservative" else BYTES["float16"]
    key_bytes = B * D * kb
    val_bytes = B * M * (ib + pb)
    return {
        "key_mb":   key_bytes / 1e6,
        "value_mb": val_bytes / 1e6,
        "meta_mb":  0.0,
        "total_mb": (key_bytes + val_bytes) / 1e6,
        "key_dtype":   f"float{32 if mode=='conservative' else 16}",
        "value_dtype": f"int{64 if mode=='conservative' else 32}+float{32 if mode=='conservative' else 16}",
        "mode": mode,
    }


def _actual_memory_from_file(sf: dict) -> dict:
    """Actual bytes on disk from the .pt tensors (float32 state file as saved)."""
    def mb(k): return sf[k].nbytes / 1e6 if k in sf else 0.0
    key_mb  = mb("prototype_h")
    val_mb  = mb("top_k_token_ids") + mb("top_k_token_counts")
    meta_mb = mb("total_counts") + mb("state_entropy") + mb("state_purity")
    return {
        "key_mb":   round(key_mb,  3),
        "value_mb": round(val_mb,  3),
        "meta_mb":  round(meta_mb, 3),
        "total_mb": round(key_mb + val_mb + meta_mb, 3),
        "key_dtype": str(sf["prototype_h"].dtype),
        "value_dtype": f"{sf['top_k_token_ids'].dtype}/{sf['top_k_token_counts'].dtype}",
        "mode": "actual",
    }


# ── Q NLL loading ─────────────────────────────────────────────────────────────

def _load_q_metrics(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _find_q_nll(seed_out: Path, method: str, budget: int) -> float:
    """
    Try several canonical paths for Q-read results.
    Returns q_state_nll or nan.
    """
    candidates = [
        seed_out / "q_read" / "states" / f"{method}_B{budget}" / "q_metrics.json",
        seed_out / "q_read" / "states" / f"{method}_{budget}"  / "q_metrics.json",
        seed_out / "ablations" / "q_read" / "original" / f"{method}_B{budget}" / "q_metrics.json",
        seed_out / f"{method}_B{budget}" / "q_metrics.json",
    ]
    for p in candidates:
        m = _load_q_metrics(p)
        if m and m.get("q_state_nll") is not None:
            return float(m["q_state_nll"])
    return float("nan")


def _find_top_m_q_nll(seed_out: Path, method: str, budget: int, M: int) -> float:
    """Q NLL for top-M ablation variant (from run_state_object_ablations.py output)."""
    tag = f"top{M}"
    candidates = [
        seed_out / "ablations" / "q_read" / f"{method}_abl_{tag}_B{budget}" / "q_metrics.json",
        seed_out / "ablations" / "q_read" / f"{method}_abl_top{M}" / f"q_metrics.json",
    ]
    for p in candidates:
        m = _load_q_metrics(p)
        if m and m.get("q_state_nll") is not None:
            return float(m["q_state_nll"])
    return float("nan")


def _find_full_raw_q_nll(seed_out: Path) -> tuple[float, int]:
    """Returns (q_nll, n_entries) for the full raw datastore baseline."""
    candidates = [
        seed_out / "q_read" / "full_raw" / "evaluate_q_read_summary.json",
        seed_out / "full_raw" / "evaluate_q_read_summary.json",
    ]
    for p in candidates:
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            results = d.get("results", [{}])
            r = results[0] if results else {}
            nll = r.get("q_state_nll") or float("nan")
            n   = r.get("budget") or d.get("full_raw_n") or 0
            return float(nll), int(n)
    return float("nan"), 0


def _find_gpt_nll(seed_out: Path) -> float:
    for p in [seed_out / "extraction_summary.json",
              seed_out / "reports" / "extraction_summary.json"]:
        if p.exists():
            with open(p) as f:
                v = json.load(f).get("gpt_nll")
            if v is not None:
                return float(v)
    return float("nan")


# ── latency measurement ───────────────────────────────────────────────────────

def _time_value_aggregation(
    proto_h: torch.Tensor,     # [B, D]
    top_k_ids: torch.Tensor,   # [B, K]
    top_k_cnts: torch.Tensor,  # [B, K]
    val_h: torch.Tensor,       # [Q, D]
    device: torch.device,
    top_k: int = 8,
    n_warmup: int = 50,
    n_timed: int = 200,
    sparse_M: int | None = None,
) -> dict:
    """
    Measure value aggregation time after key search.
    For each query, finds top-k states and combines their distributions.
    If sparse_M is set, truncates distributions to top-M tokens.
    """
    B   = proto_h.shape[0]
    K   = min(top_k, B)
    Q   = val_h.shape[0]

    proto_gpu = proto_h.float().to(device)
    ids_gpu   = top_k_ids.to(device)
    cnts_gpu  = top_k_cnts.float().to(device)
    if sparse_M is not None:
        cnts_gpu = cnts_gpu.clone()
        cnts_gpu[:, sparse_M:] = 0.0
    val_gpu   = val_h.float().to(device)

    n_total = min(n_warmup + n_timed, Q)

    def _run(qs):
        sims  = qs @ proto_gpu.T                      # [C, B]
        tidx  = sims.topk(K, dim=-1).indices          # [C, K]
        cnts  = cnts_gpu[tidx]                        # [C, K, vocab_K]
        wts   = sims.gather(1, tidx).unsqueeze(-1)    # [C, K, 1]
        wts   = F.softmax(wts.squeeze(-1), dim=-1).unsqueeze(-1)
        _out  = (cnts * wts).sum(1)                   # [C, vocab_K]
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Warmup
    _run(val_gpu[:n_warmup])

    n_t = min(n_timed, Q - n_warmup)
    if n_t <= 0:
        n_t = min(n_timed, Q)
        timed_qs = val_gpu[:n_t]
    else:
        timed_qs = val_gpu[n_warmup: n_warmup + n_t]

    t0 = time.perf_counter()
    _run(timed_qs)
    elapsed = time.perf_counter() - t0

    return {
        "value_ms_per_query": round(elapsed * 1000 / n_t, 4),
        "n_timed": n_t,
    }


# ── per-seed analysis ─────────────────────────────────────────────────────────

def analyze_seed(
    seed_out: Path,
    data_src: Path,
    dataset: str,
    model: str,
    seed: int,
    device: torch.device,
    vocab_size: int = VOCAB_SIZE_DEFAULT,
) -> dict | None:
    """Full efficiency analysis for one seed directory."""

    print(f"\n  Analyzing seed {seed} in {seed_out}")

    # ── load val hidden states ────────────────────────────────────────────────
    val_pt = None
    for p in [data_src / "states" / "val.pt", seed_out / "states" / "val.pt"]:
        if p.exists():
            val_pt = p
            break
    if val_pt is None:
        print(f"  [WARN] val.pt not found — skipping latency")
        val_h = None
    else:
        d = torch.load(val_pt, weights_only=False)
        val_h = d["h"].float()
        val_h = val_h / (val_h.norm(dim=-1, keepdim=True) + 1e-8)
        print(f"  Val queries: {len(val_h):,}")

    # ── full raw reference ────────────────────────────────────────────────────
    fr_q_nll, fr_n = _find_full_raw_q_nll(seed_out)
    gpt_nll         = _find_gpt_nll(seed_out)

    # If we don't have fr_n from summary, try to infer from datastore
    if fr_n == 0:
        for ds_p in [seed_out / "states" / "datastore.pt",
                     data_src / "states" / "datastore.pt"]:
            if ds_p.exists():
                ds = torch.load(ds_p, weights_only=False, map_location="cpu")
                fr_n = len(ds["y"])
                del ds
                break

    print(f"  GPT NLL:       {gpt_nll:.4f}" if math.isfinite(gpt_nll) else "  GPT NLL:       —")
    print(f"  Full raw Q NLL:{fr_q_nll:.4f}  N={fr_n:,}" if math.isfinite(fr_q_nll) else "  Full raw: —")

    # ── scan available state files ────────────────────────────────────────────
    states_dir = seed_out / "states"
    if not states_dir.exists():
        print(f"  [WARN] states dir not found: {states_dir}")
        return None

    state_files = sorted(states_dir.glob("*_B*.pt"))
    if not state_files:
        print(f"  [WARN] no state files in {states_dir}")
        return None

    best_row   = None
    best_q_nll = float("inf")
    all_state_rows = []

    for sfp in state_files:
        stem  = sfp.stem
        parts = stem.rsplit("_B", 1)
        if len(parts) != 2:
            continue
        method, bud_str = parts
        try:
            budget = int(bud_str)
        except ValueError:
            continue

        q_nll = _find_q_nll(seed_out, method, budget)
        sf    = torch.load(sfp, weights_only=False, map_location="cpu")
        B     = sf["prototype_h"].shape[0]
        D     = sf["prototype_h"].shape[1]
        K_stored = sf["top_k_token_ids"].shape[1]

        row = {
            "method": method, "budget": budget, "B": B, "D": D,
            "K_stored": K_stored, "q_nll": q_nll,
            "delta_vs_gpt":      q_nll - gpt_nll  if math.isfinite(q_nll) and math.isfinite(gpt_nll)  else float("nan"),
            "delta_vs_full_raw": q_nll - fr_q_nll if math.isfinite(q_nll) and math.isfinite(fr_q_nll) else float("nan"),
            "_sf": sf,
        }
        all_state_rows.append(row)
        if math.isfinite(q_nll) and q_nll < best_q_nll:
            best_q_nll = q_nll
            best_row   = row

    if best_row is None:
        print("  [WARN] no Q NLL results found for any state file")
        # Fall back to largest budget
        best_row = max(all_state_rows, key=lambda r: r["budget"])

    method = best_row["method"]
    budget = best_row["budget"]
    B      = best_row["B"]
    D      = best_row["D"]
    sf     = best_row["_sf"]

    print(f"  Best DIME:     {method} B={budget}  Q={best_q_nll:.4f}")

    # ── top-M Q NLLs ─────────────────────────────────────────────────────────
    top_m_nlls = {}
    for M in SPARSE_M_VALUES:
        top_m_nlls[M] = _find_top_m_q_nll(seed_out, method, budget, M)

    # ── memory accounting ─────────────────────────────────────────────────────
    mem_rows = []
    for mode in ("conservative", "practical"):
        if fr_n > 0:
            mr = _memory_raw(fr_n, D, mode)
            mem_rows.append({"format": "full_raw", "mode": mode,
                             "entries": fr_n, "q_nll": fr_q_nll, **mr})
            er = _memory_raw(B, D, mode)
            mem_rows.append({"format": "equal_budget_raw", "mode": mode,
                             "entries": B, "q_nll": float("nan"), **er})
        md = _memory_dime_dense(B, D, vocab_size, mode)
        mem_rows.append({"format": "DIME_dense", "mode": mode,
                         "entries": B, "q_nll": best_q_nll, **md})
        for M in SPARSE_M_VALUES:
            ms = _memory_dime_sparse(B, D, M, mode)
            mem_rows.append({"format": f"DIME_top{M}", "mode": mode,
                             "entries": B, "q_nll": top_m_nlls[M], **ms})

    # Actual on-disk (float32 state file)
    ma = _actual_memory_from_file(sf)
    mem_rows.append({"format": "DIME_actual_file", "mode": "actual",
                     "entries": B, "q_nll": best_q_nll, **ma})

    # Compute ratio vs full raw conservative
    fr_total = next((r["total_mb"] for r in mem_rows
                     if r["format"] == "full_raw" and r["mode"] == "conservative"), None)
    for r in mem_rows:
        r["ratio_vs_full_raw"] = (r["total_mb"] / fr_total
                                  if fr_total and fr_total > 0 else float("nan"))
        r["compression_vs_full_raw"] = (fr_total / r["total_mb"]
                                        if fr_total and r["total_mb"] > 0 else float("nan"))

    # ── latency ───────────────────────────────────────────────────────────────
    latency_rows = []
    if val_h is not None:
        print(f"  Measuring latency ...")

        # DIME key search
        proto_h = sf["prototype_h"]
        t_search = time_state_retrieval(proto_h, val_h, device)
        latency_rows.append({
            "format": "DIME_dense", "entries": B,
            "search_ms":  t_search["retrieval_ms_per_query"],
            "value_ms":   0.0,
            "total_ms":   t_search["retrieval_ms_per_query"],
            "qps":        t_search["retrieval_queries_per_sec"],
        })

        # DIME with value aggregation
        ids  = sf["top_k_token_ids"]
        cnts = sf["top_k_token_counts"]
        tv   = _time_value_aggregation(proto_h, ids, cnts, val_h, device)
        total_ms = t_search["retrieval_ms_per_query"] + tv["value_ms_per_query"]
        latency_rows[-1].update({
            "value_ms": tv["value_ms_per_query"],
            "total_ms": total_ms,
        })

        # Sparse DIME value aggregation (search is same, value differs)
        for M in [32, 64]:
            tv_s = _time_value_aggregation(proto_h, ids, cnts, val_h, device, sparse_M=M)
            latency_rows.append({
                "format": f"DIME_top{M}", "entries": B,
                "search_ms": t_search["retrieval_ms_per_query"],
                "value_ms":  tv_s["value_ms_per_query"],
                "total_ms":  t_search["retrieval_ms_per_query"] + tv_s["value_ms_per_query"],
                "qps":       1000.0 / (t_search["retrieval_ms_per_query"] + tv_s["value_ms_per_query"] + 1e-9),
            })

        # Full raw key search (if datastore available)
        ds_h_loaded = False
        for ds_p in [seed_out / "states" / "datastore.pt",
                     data_src / "states" / "datastore.pt"]:
            if ds_p.exists():
                print(f"  Loading datastore for raw latency ...")
                ds = torch.load(ds_p, weights_only=False, map_location="cpu")
                ds_h = ds["h"].float()
                ds_h = ds_h / (ds_h.norm(dim=-1, keepdim=True) + 1e-8)
                t_raw = time_state_retrieval(ds_h, val_h, device)
                latency_rows.append({
                    "format": "full_raw", "entries": len(ds_h),
                    "search_ms": t_raw["retrieval_ms_per_query"],
                    "value_ms":  0.001,   # token label lookup ≈ negligible
                    "total_ms":  t_raw["retrieval_ms_per_query"],
                    "qps":       t_raw["retrieval_queries_per_sec"],
                })
                del ds, ds_h
                ds_h_loaded = True
                break
        if not ds_h_loaded:
            print(f"  [WARN] datastore.pt not found — raw latency skipped")

    # ── entry-count rows ──────────────────────────────────────────────────────
    entry_rows = []
    if fr_n > 0:
        entry_rows.append({
            "format": "full_raw", "entries": fr_n,
            "entry_ratio": 1.0, "entry_compression": 1.0, "q_nll": fr_q_nll,
        })
        entry_rows.append({
            "format": "equal_budget_raw", "entries": B,
            "entry_ratio": B / fr_n, "entry_compression": fr_n / B, "q_nll": float("nan"),
        })
    entry_rows.append({
        "format": f"DIME_{method}_B{B}", "entries": B,
        "entry_ratio": B / fr_n if fr_n > 0 else float("nan"),
        "entry_compression": fr_n / B if fr_n > 0 and B > 0 else float("nan"),
        "q_nll": best_q_nll,
    })

    for r in entry_rows:
        r.update({"dataset": dataset, "model": model, "seed": seed})
    for r in mem_rows:
        r.update({"dataset": dataset, "model": model, "seed": seed,
                  "method": method, "budget": budget})
    for r in latency_rows:
        r.update({"dataset": dataset, "model": model, "seed": seed,
                  "method": method, "budget": budget})

    return {
        "seed": seed, "dataset": dataset, "model": model,
        "method": method, "budget": budget, "B": B, "D": D,
        "best_q_nll": best_q_nll, "fr_q_nll": fr_q_nll,
        "gpt_nll": gpt_nll, "fr_n": fr_n,
        "top_m_nlls": top_m_nlls,
        "entry_rows": entry_rows,
        "mem_rows":   mem_rows,
        "latency_rows": latency_rows,
    }


# ── reporting ─────────────────────────────────────────────────────────────────

def _f(v, d=4):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    return f"{float(v):.{d}f}"


def _fmtd(v, d=4):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    return f"{float(v):+.{d}f}"


def _compute_verdict(results: list[dict]) -> str:
    if not results:
        return "NO_DATA"

    best_dime_q = min((r["best_q_nll"] for r in results if math.isfinite(r["best_q_nll"])),
                      default=float("nan"))
    fr_q        = next((r["fr_q_nll"]  for r in results if math.isfinite(r["fr_q_nll"])),
                       float("nan"))
    B           = results[0]["B"]
    fr_n        = results[0]["fr_n"]
    top_m_nlls  = results[0]["top_m_nlls"]

    if not math.isfinite(best_dime_q) or not math.isfinite(fr_q):
        return "NO_DATA"

    dime_matches_raw  = (best_dime_q - fr_q) <= 0.02
    entry_compress    = fr_n / B if B > 0 and fr_n > 0 else 0.0

    sparse32_nll  = top_m_nlls.get(32, float("nan"))
    sparse64_nll  = top_m_nlls.get(64, float("nan"))
    best_sparse   = min(v for v in [sparse32_nll, sparse64_nll] if math.isfinite(v)) \
                    if any(math.isfinite(v) for v in [sparse32_nll, sparse64_nll]) \
                    else float("nan")

    sparse_ok   = math.isfinite(best_sparse) and (best_sparse - fr_q) <= 0.02
    entry_ok    = entry_compress >= 4.0  # <= 25% entries

    # Check if sparse DIME uses less total memory than full raw (conservative)
    mem_sparse = _memory_dime_sparse(B, results[0]["D"], 32, "conservative")["total_mb"]
    mem_fr     = _memory_raw(fr_n, results[0]["D"], "conservative")["total_mb"]
    mem_ok     = mem_sparse < mem_fr if (fr_n > 0 and mem_fr > 0) else False

    # Latency check: DIME faster than raw
    lat_rows = results[0]["latency_rows"]
    dime_lat = next((r["total_ms"] for r in lat_rows if "DIME" in r["format"]), None)
    raw_lat  = next((r["total_ms"] for r in lat_rows if r["format"] == "full_raw"), None)
    lat_ok   = (dime_lat is not None and raw_lat is not None and dime_lat < raw_lat)

    if dime_matches_raw and entry_ok and sparse_ok and mem_ok and lat_ok:
        return "STRONG_EFFICIENCY_SUPPORT"
    elif dime_matches_raw and entry_ok:
        return "ENTRY_ONLY_SUPPORT"
    elif entry_compress >= 2.0:
        return "MIXED"
    return "FAIL"


def _build_report(results: list[dict], dataset: str, model: str) -> str:
    if not results:
        return "# No results\n"

    r0 = results[0]  # representative seed for tables
    verdict = _compute_verdict(results)
    fr_n    = r0["fr_n"]
    B       = r0["B"]
    D       = r0["D"]
    method  = r0["method"]
    budget  = r0["budget"]

    # Cross-seed averages
    def _avg(key):
        vals = [r[key] for r in results if math.isfinite(r.get(key, float("nan")))]
        return sum(vals) / len(vals) if vals else float("nan")

    best_dime_q = _avg("best_q_nll")
    fr_q        = _avg("fr_q_nll")
    gpt_nll     = _avg("gpt_nll")
    top_m_avg   = {}
    for M in SPARSE_M_VALUES:
        vals = [r["top_m_nlls"].get(M, float("nan")) for r in results]
        vals = [v for v in vals if math.isfinite(v)]
        top_m_avg[M] = sum(vals) / len(vals) if vals else float("nan")

    lines = [f"# DIME Efficiency Analysis\n",
             f"Dataset: {dataset}  |  Model: {model}  |  Seeds: {[r['seed'] for r in results]}\n",
             f"Verdict: **{verdict}**\n"]

    # ── Table 1: Entry-count compression ─────────────────────────────────────
    lines += [
        "## Table 1: Entry-Count Compression\n",
        f"| Memory Format | Entries | Entry Ratio | Compression | Q NLL |",
        f"|---------------|--------:|------------:|------------:|------:|",
    ]
    if fr_n > 0:
        lines.append(f"| full_raw            | {fr_n:>7,} | 100.0%     | 1.0×        | {_f(fr_q)} |")
        lines.append(f"| equal_budget_raw    | {B:>7,} | {B/fr_n*100:5.1f}%     | {fr_n/B:5.1f}×       | —      |")
    lines.append(f"| DIME {method} B{B} | {B:>7,} | {B/fr_n*100 if fr_n>0 else float('nan'):5.1f}%     | {fr_n/B if fr_n>0 else float('nan'):5.1f}×       | {_f(best_dime_q)} |")
    lines.append("")

    # ── Table 2: Byte-level memory ────────────────────────────────────────────
    lines += [
        "## Table 2: Byte-Level Memory Estimate\n",
        "| Format | Mode | Key MB | Value MB | Total MB | Ratio vs Full Raw | Q NLL |",
        "|--------|------|-------:|---------:|---------:|------------------:|------:|",
    ]
    fr_total_cons = _memory_raw(fr_n, D, "conservative")["total_mb"] if fr_n > 0 else float("nan")
    formats = (
        [("full_raw",      fr_n, fr_q,       lambda m: _memory_raw(fr_n, D, m))] if fr_n > 0 else []
    ) + [
        ("DIME_dense",     B,    best_dime_q, lambda m: _memory_dime_dense(B, D, VOCAB_SIZE_DEFAULT, m)),
    ] + [
        (f"DIME_top{M}",   B,    top_m_avg.get(M, float("nan")),
         lambda m, M=M: _memory_dime_sparse(B, D, M, m))
        for M in SPARSE_M_VALUES
    ]
    for (fmt, entries, q, mem_fn) in formats:
        for mode in ("conservative", "practical"):
            mem = mem_fn(mode)
            ratio = mem["total_mb"] / fr_total_cons if fr_total_cons else float("nan")
            lines.append(
                f"| {fmt:<18} | {mode:<12} | {mem['key_mb']:>7.1f} | {mem['value_mb']:>8.1f} |"
                f" {mem['total_mb']:>8.1f} | {ratio:>17.3f} | {_f(q)} |"
            )
    lines.append("")

    # ── Table 3: Sparse top-M tradeoff ────────────────────────────────────────
    lines += [
        "## Table 3: Sparse Top-M Accuracy / Memory Tradeoff\n",
        "| Top-M | Q NLL | Δ vs Dense DIME | Total MB (practical) | Ratio vs Full Raw | Verdict |",
        "|------:|------:|----------------:|---------------------:|------------------:|---------|",
    ]
    dense_q = best_dime_q
    for M in SPARSE_M_VALUES:
        q   = top_m_avg.get(M, float("nan"))
        mem = _memory_dime_sparse(B, D, M, "practical")
        ratio = mem["total_mb"] / fr_total_cons if fr_total_cons else float("nan")
        delta = (q - dense_q) if math.isfinite(q) and math.isfinite(dense_q) else float("nan")
        verd  = ("≈ dense" if math.isfinite(delta) and abs(delta) <= 0.01
                 else "close"  if math.isfinite(delta) and delta <= 0.02
                 else "worse"  if math.isfinite(delta) else "—")
        lines.append(
            f"| {M:>5} | {_f(q)} | {_fmtd(delta, 4):>15} |"
            f" {mem['total_mb']:>20.1f} | {ratio:>17.3f} | {verd} |"
        )
    lines.append("")

    # ── Table 4: Latency ──────────────────────────────────────────────────────
    lat_rows = r0.get("latency_rows", [])
    if lat_rows:
        lines += [
            "## Table 4: Retrieval Latency\n",
            "| Format | Entries | Search ms/q | Value ms/q | Total ms/q | QPS |",
            "|--------|--------:|------------:|-----------:|-----------:|----:|",
        ]
        for lr in lat_rows:
            lines.append(
                f"| {lr['format']:<18} | {lr['entries']:>7,} |"
                f" {lr['search_ms']:>11.4f} | {lr.get('value_ms', 0.0):>10.4f} |"
                f" {lr['total_ms']:>10.4f} | {lr.get('qps', 0.0):>9.1f} |"
            )
        lines.append("")

    # ── Verdict block ─────────────────────────────────────────────────────────
    lines += [f"## Verdict: {verdict}\n"]
    if verdict == "STRONG_EFFICIENCY_SUPPORT":
        lines.append(
            "DIME reduces the number of retrievable memory entries and, when stored with "
            "sparse top-M distributions, also reduces estimated memory footprint and retrieval "
            "latency while preserving prediction quality.")
    elif verdict == "ENTRY_ONLY_SUPPORT":
        lines.append(
            "DIME reduces the number of retrievable memory entries required to match full raw "
            "Q-read. We report this as entry-count compression rather than byte-level compression, "
            "since byte-level savings depend on sparse distribution storage and implementation details.")
    elif verdict == "MIXED":
        lines.append(
            "DIME provides object-count compression and better prediction per entry, but practical "
            "memory and runtime savings depend on implementation choices such as sparse distribution "
            "storage and nearest-neighbor indexing.")
    else:
        lines.append(
            "DIME improves prediction per entry but should not be described as cheaper than raw "
            "kNN under the tested implementation.")

    return "\n".join(lines)


def _print_summary(results: list[dict], dataset: str, model: str):
    if not results:
        return
    r0 = results[0]
    verdict = _compute_verdict(results)

    def _avg(key):
        vals = [r[key] for r in results if math.isfinite(r.get(key, float("nan")))]
        return sum(vals) / len(vals) if vals else float("nan")

    best_dime_q = _avg("best_q_nll")
    fr_q        = _avg("fr_q_nll")
    gpt_nll     = _avg("gpt_nll")
    B, fr_n, D  = r0["B"], r0["fr_n"], r0["D"]

    best_sparse_M = None
    best_sparse_q = float("nan")
    for M in SPARSE_M_VALUES:
        vals = [r["top_m_nlls"].get(M, float("nan")) for r in results]
        vals = [v for v in vals if math.isfinite(v)]
        q = sum(vals) / len(vals) if vals else float("nan")
        if math.isfinite(q) and (not math.isfinite(best_sparse_q) or q < best_sparse_q):
            best_sparse_q = q
            best_sparse_M = M

    mem_dense_cons   = _memory_dime_dense(B, D, VOCAB_SIZE_DEFAULT, "conservative")["total_mb"]
    mem_sparse_prac  = _memory_dime_sparse(B, D, 32, "practical")["total_mb"] if B > 0 else float("nan")
    mem_fr_cons      = _memory_raw(fr_n, D, "conservative")["total_mb"] if fr_n > 0 else float("nan")
    mem_ratio        = mem_sparse_prac / mem_fr_cons if math.isfinite(mem_fr_cons) and mem_fr_cons > 0 else float("nan")

    lat_rows = r0.get("latency_rows", [])
    raw_lat  = next((r["total_ms"] for r in lat_rows if r["format"] == "full_raw"), float("nan"))
    dime_lat = next((r["total_ms"] for r in lat_rows if "DIME_top32" in r["format"]), float("nan"))
    if not math.isfinite(dime_lat):
        dime_lat = next((r["total_ms"] for r in lat_rows if "DIME" in r["format"]), float("nan"))
    lat_ratio = dime_lat / raw_lat if math.isfinite(raw_lat) and math.isfinite(dime_lat) and raw_lat > 0 else float("nan")

    print(f"\n{'='*60}")
    print(f"  DIME Efficiency Analysis")
    print(f"  {'─'*40}")
    print(f"  Dataset:                     {dataset}")
    print(f"  Model:                       {model}")
    print(f"  Full raw entries:            {fr_n:,}")
    print(f"  Best DIME entries:           {B:,}  ({r0['method']})")
    print(f"  Entry compression:           {fr_n/B:.1f}×" if B > 0 and fr_n > 0 else "  Entry compression:           —")
    print(f"  Full raw Q NLL:              {_f(fr_q)}")
    print(f"  Best DIME Q NLL:             {_f(best_dime_q)}")
    print(f"  Best sparse top-M:           top{best_sparse_M}" if best_sparse_M else "  Best sparse top-M:           —")
    print(f"  Sparse top-M Q NLL:          {_f(best_sparse_q)}")
    print(f"  Dense DIME memory (cons.):   {mem_dense_cons:.1f} MB")
    print(f"  Sparse top32 DIME (prac.):   {mem_sparse_prac:.1f} MB")
    print(f"  Full raw memory (cons.):     {_f(mem_fr_cons, 1)} MB")
    print(f"  Memory ratio vs full raw:    {_f(mem_ratio, 3)}")
    print(f"  Full raw retrieval ms/q:     {_f(raw_lat)}")
    print(f"  DIME retrieval ms/q:         {_f(dime_lat)}")
    print(f"  Latency ratio:               {_f(lat_ratio, 3)}")
    print(f"  Verdict:                     {verdict}")
    print(f"{'='*60}")

    if verdict == "STRONG_EFFICIENCY_SUPPORT":
        safe_claim = (
            f"DIME compresses to {fr_n/B:.0f}× fewer retrievable entries "
            f"while matching full raw Q NLL, and with sparse top-{best_sparse_M} "
            f"distributions also reduces estimated memory to {mem_ratio:.0%} of full raw.")
        unsafe_claim = (
            f"Do not claim byte-level savings without specifying sparse storage "
            f"and float16 implementation details.")
    elif verdict == "ENTRY_ONLY_SUPPORT":
        safe_claim = (
            f"DIME achieves {fr_n/B:.0f}× entry-count compression vs full raw "
            f"with matching prediction quality.")
        unsafe_claim = (
            f"Do not claim byte-level or latency savings — "
            f"these depend on implementation-specific storage choices.")
    else:
        safe_claim   = "Report entry-count compression only; do not claim general efficiency superiority."
        unsafe_claim = "Do not claim DIME is cheaper than raw kNN without qualification."

    print(f"\n  Paper-safe efficiency claim: {safe_claim}")
    print(f"  Unsafe claim to avoid:       {unsafe_claim}\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",   default=None,
                        help="Dir with states/val.pt + datastore.pt. "
                             "Auto-discovered if omitted.")
    parser.add_argument("--output",   required=True,
                        help="Paper output root (seeds live as subdirs if multi-seed)")
    parser.add_argument("--dataset",  default="TinyStories")
    parser.add_argument("--model",    default="gpt2")
    parser.add_argument("--seeds",    nargs="+", type=int, default=[42])
    parser.add_argument("--vocab_size", type=int, default=VOCAB_SIZE_DEFAULT)
    parser.add_argument("--device",   default="cuda")
    parser.add_argument("--force",    action="store_true")
    args = parser.parse_args()

    set_seed(42)
    device  = get_device({"device": args.device})
    out_root = Path(args.output)
    rep_dir  = out_root / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)

    sentinel = rep_dir / "dime_efficiency_verdict.json"
    if sentinel.exists() and not args.force:
        print(f"[cached] {sentinel}  (use --force to rerun)")
        with open(sentinel) as f:
            print(json.dumps(json.load(f), indent=2))
        return

    print(f"\nanalyze_dime_efficiency")
    print(f"  Output: {out_root}")
    print(f"  Dataset: {args.dataset}  Model: {args.model}")
    print(f"  Seeds: {args.seeds}")

    results = []
    for seed in args.seeds:
        # Determine seed output dir
        seed_out = out_root / f"seed{seed}"
        if not seed_out.exists():
            seed_out = out_root   # single-seed layout

        # Find data source (val.pt / datastore.pt)
        if args.source:
            data_src = Path(args.source)
        else:
            # Auto-discover: try seed_out first, then CWD siblings
            data_src = None
            for cand in [seed_out, seed_out / "..", *Path.cwd().iterdir()]:
                cand = Path(cand).resolve()
                if (cand / "states" / "val.pt").exists():
                    data_src = cand
                    print(f"  [auto] data source: {data_src}")
                    break
            if data_src is None:
                print(f"  [WARN] val.pt not found for seed {seed}, using seed_out")
                data_src = seed_out

        r = analyze_seed(seed_out, data_src, args.dataset, args.model,
                         seed, device, args.vocab_size)
        if r:
            results.append(r)

    if not results:
        print("No results — exiting.")
        return

    # ── aggregate and save ────────────────────────────────────────────────────
    all_mem_rows     = []
    all_entry_rows   = []
    all_latency_rows = []
    for r in results:
        all_mem_rows     += r["mem_rows"]
        all_entry_rows   += r["entry_rows"]
        all_latency_rows += r["latency_rows"]

    pd.DataFrame(all_entry_rows).to_csv(   rep_dir / "dime_efficiency_results.csv",     index=False)
    pd.DataFrame(all_mem_rows).to_csv(     rep_dir / "dime_efficiency_memory_table.csv", index=False)
    pd.DataFrame(all_latency_rows).to_csv( rep_dir / "dime_efficiency_latency_table.csv", index=False)

    # Summary markdown
    md = _build_report(results, args.dataset, args.model)
    with open(rep_dir / "dime_efficiency_summary.md", "w", encoding="utf-8") as f:
        f.write(md)

    # Verdict JSON
    verdict = _compute_verdict(results)
    r0 = results[0]
    fr_n, B, D = r0["fr_n"], r0["B"], r0["D"]
    verdict_json = {
        "dataset": args.dataset, "model": args.model,
        "seeds": args.seeds, "verdict": verdict,
        "best_dime_method": r0["method"], "best_dime_budget": r0["budget"],
        "fr_n": fr_n, "dime_entries": B,
        "entry_compression": round(fr_n / B, 2) if B > 0 and fr_n > 0 else None,
        "fr_q_nll":   r0["fr_q_nll"]   if math.isfinite(r0["fr_q_nll"])   else None,
        "best_dime_q_nll": r0["best_q_nll"] if math.isfinite(r0["best_q_nll"]) else None,
        "top_m_nlls": {str(M): (v if math.isfinite(v) else None)
                       for M, v in r0["top_m_nlls"].items()},
        "dense_dime_memory_mb_conservative": round(_memory_dime_dense(B, D, args.vocab_size, "conservative")["total_mb"], 1),
        "sparse_top32_dime_memory_mb_practical": round(_memory_dime_sparse(B, D, 32, "practical")["total_mb"], 1),
        "full_raw_memory_mb_conservative": round(_memory_raw(fr_n, D, "conservative")["total_mb"], 1) if fr_n > 0 else None,
    }
    with open(sentinel, "w") as f:
        json.dump(verdict_json, f, indent=2)

    for p in [rep_dir / "dime_efficiency_results.csv",
              rep_dir / "dime_efficiency_memory_table.csv",
              rep_dir / "dime_efficiency_latency_table.csv",
              rep_dir / "dime_efficiency_summary.md",
              sentinel]:
        print(f"  Saved: {p}")

    _print_summary(results, args.dataset, args.model)


if __name__ == "__main__":
    main()

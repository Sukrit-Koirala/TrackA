"""
analyze_oracle_write_reconstruction.py  --  MVP 4a-0: Oracle Reconstruction

Compares oracle-reconstructed states against the offline teacher:
  - prototype cosine similarity and L2 distance
  - token distribution KL / JS divergence
  - support count and total-count equality
  - MATCH recall analysis

Reads from completed outputs; writes:
  analysis/state_alignment_{mode}.csv
  analysis/aggregate_state_metrics.json
  report.md

Usage:
  python src/analyze_oracle_write_reconstruction.py \\
    --offline_states_dir outputs_mvp2b_state_write/states \\
    --output             outputs_mvp4a0_oracle_reconstruction \\
    --teacher            minibatch_kmeans \\
    --budget             10000 \\
    --promotion_support  8 \\
    --seed               42
"""

import sys
import json
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import numpy as np
import torch
import pandas as pd

from utils import set_seed

VOCAB  = 50257
MODES  = ["exact_teacher_prototype", "sequential_running_mean"]


# ── distribution utilities ────────────────────────────────────────────────────

def _dist_from_topk(ids: np.ndarray, cnts: np.ndarray) -> np.ndarray:
    dist = np.zeros(VOCAB, dtype=np.float64)
    total = float(cnts.sum())
    if total <= 0:
        return dist
    for tid, cnt in zip(ids, cnts):
        if cnt > 0:
            dist[int(tid)] = float(cnt) / total
    return dist


def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    mask = p > 0
    if not mask.any():
        return 0.0
    return float(np.sum(p[mask] * np.log((p[mask] + eps) / (q[mask] + eps))))


def _js(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


# ── per-state comparison ──────────────────────────────────────────────────────

def compare_states(teacher_sf: dict, recon_sf: dict, B: int) -> pd.DataFrame:
    """Build per-state comparison DataFrame."""
    t_proto = teacher_sf["prototype_h"].float().numpy()       # [B, D]
    r_proto = recon_sf["prototype_h"].float().numpy()
    t_ids   = teacher_sf["top_k_token_ids"].numpy()
    t_cnts  = teacher_sf["top_k_token_counts"].numpy()
    r_ids   = recon_sf["top_k_token_ids"].numpy()
    r_cnts  = recon_sf["top_k_token_counts"].numpy()
    t_cnt   = teacher_sf["assigned_count"].numpy()
    r_cnt   = recon_sf["assigned_count"].numpy()
    t_ent   = teacher_sf["state_entropy"].numpy()
    r_ent   = recon_sf["state_entropy"].numpy()
    t_pur   = teacher_sf["state_purity"].numpy()
    r_pur   = recon_sf["state_purity"].numpy()

    rows = []
    for ci in range(B):
        if t_cnt[ci] == 0 and r_cnt[ci] == 0:
            continue
        tp, rp = t_proto[ci], r_proto[ci]
        nt, nr = float(np.linalg.norm(tp)), float(np.linalg.norm(rp))
        cos_sim = float(np.dot(tp, rp) / (nt * nr + 1e-8))
        l2_dist = float(np.linalg.norm(tp - rp))

        t_dist = _dist_from_topk(t_ids[ci], t_cnts[ci])
        r_dist = _dist_from_topk(r_ids[ci], r_cnts[ci])

        rows.append({
            "state_id":         ci,
            "support_count":    int(t_cnt[ci]),
            "recon_count":      int(r_cnt[ci]),
            "count_match":      bool(t_cnt[ci] == r_cnt[ci]),
            "proto_cos_sim":    cos_sim,
            "proto_l2_dist":    l2_dist,
            "kl_teacher_recon": _kl(t_dist, r_dist),
            "js_divergence":    _js(t_dist, r_dist),
            "entropy_diff":     float(r_ent[ci] - t_ent[ci]),
            "purity_diff":      float(r_pur[ci] - t_pur[ci]),
        })
    return pd.DataFrame(rows)


def _agg(df: pd.DataFrame, col: str) -> dict:
    vals = df[col].dropna().values.astype(float)
    if len(vals) == 0:
        return {}
    w = df["support_count"].values[:len(vals)].astype(float)
    w_sum = w.sum()
    wm = float((vals * w).sum() / w_sum) if w_sum > 0 else float("nan")
    return {
        "mean":         float(vals.mean()),
        "median":       float(np.median(vals)),
        "p10":          float(np.percentile(vals, 10)),
        "p90":          float(np.percentile(vals, 90)),
        "p99":          float(np.percentile(vals, 99)),
        "min":          float(vals.min()),
        "max":          float(vals.max()),
        "weighted_mean": wm,
    }


# ── report ────────────────────────────────────────────────────────────────────

def _load_json(p: Path) -> dict:
    return json.load(open(p)) if p.exists() else {}


def _f(d: dict | None, k: str, fmt: str = ".4f") -> str:
    v = (d or {}).get(k)
    return f"{v:{fmt}}" if v is not None else "—"


def write_report(
    out:            Path,
    teacher_name:   str,
    budget:         int,
    promotion_support: int,
    teacher_sf:     dict,
    all_agg:        dict,
    all_dfs:        dict,
    recall:         dict,
    teacher_m:      dict,
    exact_m:        dict,
    seq_m:          dict,
    action_stats:   dict,
):
    n_states = int((teacher_sf["assigned_count"] > 0).sum().item())

    exact_q  = (exact_m  or {}).get("q_state_nll")
    seq_q    = (seq_m    or {}).get("q_state_nll")
    ref_q    = (teacher_m or {}).get("q_state_nll") or 2.3015

    exact_gap = (exact_q - ref_q) if exact_q else None
    seq_gap   = (seq_q   - ref_q) if seq_q   else None

    def _verd(gap):
        if gap is None:           return "UNKNOWN (evaluation not run)"
        if abs(gap) <= 0.005:     return f"PASS (|gap|={abs(gap):.4f} ≤ 0.005)"
        if gap <= 0.02:           return f"PASS-sequential (gap={gap:+.4f} ≤ 0.02)"
        if gap <= 0.05:           return f"WARNING (gap={gap:+.4f} ≤ 0.05, proto drift)"
        return                           f"FAIL (gap={gap:+.4f} > 0.05)"

    exact_verd = _verd(exact_gap)
    seq_verd   = _verd(seq_gap)

    if "FAIL" in exact_verd:
        go = "NO-GO: implementation cannot reproduce the teacher under oracle allocation"
    elif "FAIL" in seq_verd or "WARNING" in seq_verd:
        go = "GO WITH WARNING: machinery works but prototype updates need investigation"
    elif exact_q is None and seq_q is None:
        go = "UNKNOWN: evaluation not yet run"
    else:
        go = "GO: reconstruction machinery is valid"

    comb = (recall or {}).get("combined", {})

    lines = [
        "# MVP 4a-0: Oracle Sequential Reconstruction — Report\n",
        "## 1. Experiment Goal\n",
        "**This is an oracle diagnostic, not imitation learning or natural discovery.**\n",
        "The oracle uses saved offline teacher IDs to drive a",
        "BUFFER → UPDATE → PROMOTE lifecycle over the datastore stream.",
        "The goal is to determine whether the sequential online WRITE machinery",
        "can recover the quality of the successful offline predictive codebook",
        "when allocation decisions are perfect.\n",
        f"Teacher: `{teacher_name}_B{budget}`  |  Promotion support: {promotion_support}\n",
        "## 2. Existing Code Reused\n",
        "| Module | Functions / Classes reused |",
        "|--------|---------------------------|",
        "| `build_predictive_states.py` | `build_minibatch_kmeans`, `fit_pca`, `accumulate_token_counts`, `compute_state_stats`, `TOP_K`, `PCA_DIM` |",
        "| `train_q_state_read.py`      | `run_method_budget` (end-to-end Q-read training + evaluation) |",
        "| `utils.py`                   | `get_device`, `set_seed` |",
        "",
        "State artifact schema: exact `build_predictive_states.py` format (same keys, dtypes, TOP_K=256).",
        "Evaluation: unmodified `run_method_budget` with same splits, action grid, smoothing, Q-MLP.\n",
        "## 3. Teacher Assignment Recovery\n",
        f"- **Method**: Re-run `MiniBatchKMeans(n_clusters={budget}, n_init=3, max_iter=200,",
        f"  batch_size=4096, random_state=42)` on PCA-64 of L2-normalised datastore h vectors.",
        "- **Source**: Original labels not saved by builder; re-derived deterministically.",
        "- **Validation**: See `teacher_assignments/assignment_validation.json`.\n",
        "## 4. Oracle Stream Behavior\n",
    ]

    if action_stats:
        lines += [
            "| Action | Count |",
            "|--------|------:|",
            f"| CREATE_BUFFER       | {action_stats.get('n_create', '—'):,} |",
            f"| UPDATE_BUFFER       | {action_stats.get('n_update_buffer', '—'):,} |",
            f"| PROMOTE_BUFFER      | {action_stats.get('n_promote', '—'):,} |",
            f"| UPDATE_STATE        | {action_stats.get('n_update_state', '—'):,} |",
            f"| End-stream promotes | {action_stats.get('n_end_promote', '—'):,} |",
            f"| **Final objects**   | **{action_stats.get('final_n_objects', '—'):,}** |",
            "",
        ]

    lines += [
        "## 5. Reconstruction Fidelity\n",
        "| Metric | exact_teacher_prototype | sequential_running_mean |",
        "|--------|------------------------:|------------------------:|",
    ]

    def _agg_row(col, label, fmt=".6f"):
        e = all_agg.get("exact_teacher_prototype", {}).get(col, {})
        s = all_agg.get("sequential_running_mean",  {}).get(col, {})
        ev = f"{e['mean']:{fmt}}" if e else "—"
        sv = f"{s['mean']:{fmt}}" if s else "—"
        return f"| {label:<40} | {ev:>23} | {sv:>23} |"

    lines += [
        _agg_row("proto_cos_sim",     "Proto cosine sim (mean)"),
        _agg_row("proto_l2_dist",     "Proto L2 dist (mean)"),
        _agg_row("kl_teacher_recon",  "KL(teacher||recon) (mean)"),
        _agg_row("js_divergence",     "JS divergence (mean)"),
        _agg_row("entropy_diff",      "Entropy diff mean (recon-teacher)"),
        "",
        "Support-count match: "
        + ", ".join(f"{m}: {all_agg.get(m,{}).get('count_match_fraction',0):.4f}"
                    for m in MODES) + "\n",
        "## 6. MATCH Recall\n",
        "Recall computed against **final** memory state (approximation: over-estimates",
        "early-stream queries).\n",
        "| Pool | recall@1 | recall@4 | recall@8 | recall@16 |",
        "|------|--------:|--------:|--------:|---------:|",
    ]
    for pool in ["persistent", "buffers", "combined"]:
        p = (recall or {}).get(pool, {})
        def _r(k):
            v = p.get(f"recall@{k}")
            return f"{v:.4f}" if v is not None else "—"
        lines.append(f"| {pool:<11} | {_r(1)} | {_r(4)} | {_r(8)} | {_r(16)} |")

    if comb.get("by_quartile"):
        lines += ["", "recall@8 by stream quartile (combined):"]
        for q, v in comb["by_quartile"].items():
            lines.append(f"  {q}: {v:.4f}")
    if comb.get("by_support_bucket"):
        lines += ["", "recall@8 by target support count at query time (combined):"]
        for b, v in comb["by_support_bucket"].items():
            lines.append(f"  support={b}: {v:.4f}")

    lines += [
        "",
        "**Would top-K cosine MATCH be sufficient for a future learned WRITE policy?**",
        f"recall@8={comb.get('recall@8', 'N/A')}  recall@16={comb.get('recall@16', 'N/A')}",
        "",
        "## 7. READ Performance\n",
        "| Memory | States | Fixed NLL | Q-read NLL | Oracle NLL |",
        "| ------ | -----: | --------: | ---------: | ---------: |",
        f"| Original offline teacher          | {n_states:>6} |"
        f" {_f(teacher_m,'best_fixed_nll')} | {_f(teacher_m,'q_state_nll')} |"
        f" {_f(teacher_m,'oracle_nll')} |",
        f"| Exact-prototype oracle recon      | {n_states:>6} |"
        f" {_f(exact_m,'best_fixed_nll')} | {_f(exact_m,'q_state_nll')} |"
        f" {_f(exact_m,'oracle_nll')} |",
        f"| Sequential-prototype recon        | {n_states:>6} |"
        f" {_f(seq_m,'best_fixed_nll')} | {_f(seq_m,'q_state_nll')} |"
        f" {_f(seq_m,'oracle_nll')} |",
        f"| MVP 3b best reference             |    519 |         — |     2.4204 |          — |",
        f"| Full raw datastore Q-read         | 200000 |         — |     2.2977 |          — |",
        "",
        "## 8. Diagnosis\n",
        f"**Did exact reconstruction reproduce the teacher?**  {exact_verd}",
        "",
        f"**Did sequential aggregation preserve performance?**  {seq_verd}",
        "",
        f"**Is MATCH candidate recall sufficient?**  "
        f"recall@8={comb.get('recall@8', 'N/A')}  recall@16={comb.get('recall@16', 'N/A')}",
        "",
        "**Performance gap source (if any):**",
        "- If exact fails: assignment mismatch, token-count mismatch, or evaluation pipeline issue.",
        "- If sequential fails but exact passes: sequential prototype accumulation loses retrieval geometry.",
        "- If both pass: oracle reconstruction is valid; WRITE machinery ready for next phase.",
        "",
        "## 9. Go / No-Go Decision\n",
        f"**{go}**\n",
        "## 10. Claims Allowed\n",
        "This diagnostic supports ONLY the following claims:\n",
        "- Whether BUFFER → UPDATE → PROMOTE can reproduce offline state token distributions.",
        "- Whether sequential sum_h aggregation recovers prototype geometry under perfect assignment.",
        "- Whether top-K cosine MATCH exposes the correct target object frequently enough.",
        "",
        "This diagnostic does **NOT** support claims about:",
        "- Natural discovery of predictive states.",
        "- Performance of a learned WRITE policy.",
        "- Generalisation beyond the tested teacher and dataset.",
    ]

    report_path = out / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved: {report_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",              default=None)
    ap.add_argument("--offline_states_dir",  required=True)
    ap.add_argument("--output",              required=True)
    ap.add_argument("--teacher",             default="minibatch_kmeans")
    ap.add_argument("--budget",              type=int, default=10000)
    ap.add_argument("--promotion_support",   type=int, default=8,
                    help="Used only for the report text")
    ap.add_argument("--seed",                type=int, default=42)
    ap.add_argument("--force",               action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    out = Path(args.output)
    B   = args.budget
    tag = f"{args.teacher}_B{B}"

    analysis_dir = out / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # ── load teacher ──────────────────────────────────────────────────────────
    teacher_path = Path(args.offline_states_dir) / f"{tag}.pt"
    if not teacher_path.exists():
        print(f"ERROR: {teacher_path}")
        sys.exit(1)
    teacher_sf = torch.load(teacher_path, weights_only=False)

    # ── per-mode state comparison ─────────────────────────────────────────────
    all_agg: dict = {}
    all_dfs: dict = {}

    for mode in MODES:
        method = (f"oracle_exact_{args.teacher}" if "exact" in mode
                  else f"oracle_seq_{args.teacher}")
        sp = out / "reconstructions" / mode / "states" / f"{method}_B{B}.pt"
        if not sp.exists():
            print(f"[SKIP] {sp}")
            continue

        recon_sf = torch.load(sp, weights_only=False)
        print(f"\nComparing teacher vs {mode} ...")
        df = compare_states(teacher_sf, recon_sf, B)
        csv_path = analysis_dir / f"state_alignment_{mode}.csv"
        df.to_csv(csv_path, index=False)
        all_dfs[mode] = df

        agg = {}
        for col in ["proto_cos_sim", "proto_l2_dist", "kl_teacher_recon",
                     "js_divergence", "entropy_diff", "purity_diff"]:
            agg[col] = _agg(df, col)
        agg["count_match_fraction"] = float(df["count_match"].mean())
        all_agg[mode] = agg

        print(f"  proto_cos_sim:   {agg['proto_cos_sim']['mean']:.6f}")
        print(f"  proto_l2_dist:   {agg['proto_l2_dist']['mean']:.6f}")
        print(f"  kl_t||r:         {agg['kl_teacher_recon']['mean']:.6f}")
        print(f"  js_divergence:   {agg['js_divergence']['mean']:.6f}")
        print(f"  count_match:     {agg['count_match_fraction']:.4f}")

    with open(analysis_dir / "aggregate_state_metrics.json", "w") as f:
        json.dump(all_agg, f, indent=2)

    # ── recall ────────────────────────────────────────────────────────────────
    recall = _load_json(out / "match_diagnostics" / "target_recall.json")
    if recall:
        comb = recall.get("combined", {})
        print("\nMATCH recall (combined):")
        for k in [1, 4, 8, 16]:
            v = comb.get(f"recall@{k}")
            if v is not None:
                print(f"  recall@{k}: {v:.4f}")

    # ── evaluation metrics ────────────────────────────────────────────────────
    eval_dir  = out / "evaluation"
    teacher_m = _load_json(eval_dir / "teacher_reference_metrics.json")
    exact_m   = _load_json(eval_dir / "exact_teacher_prototype_metrics.json")
    seq_m     = _load_json(eval_dir / "sequential_running_mean_metrics.json")
    action_stats = _load_json(out / "oracle_action_stats.json")

    # ── report ────────────────────────────────────────────────────────────────
    write_report(
        out, args.teacher, B, args.promotion_support,
        teacher_sf, all_agg, all_dfs,
        recall, teacher_m, exact_m, seq_m, action_stats,
    )


if __name__ == "__main__":
    main()

"""
analyze_aggregate_write_states.py  --  MVP 3b analysis

Aggregates health + q_metrics files, compares against baselines and MVP 3a,
generates AGGREGATE_WRITE_REPORT.md.

Usage:
  python branch_a_gate_mvp/src/analyze_aggregate_write_states.py \\
    --output branch_a_gate_mvp/outputs_mvp3b_aggregate_write \\
    --states_dir branch_a_gate_mvp/outputs_mvp3b_aggregate_write/states
"""

import sys, re, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import numpy as np
import torch
import pandas as pd

# ── Canonical baselines (never change) ───────────────────────────────────────
GPT_ONLY_NLL        = 2.5533
FULL_DS_FIXED       = 2.3873
FULL_DS_QMLP        = 2.2977
MVP2C_BEST_OVERALL  = 2.2852   # minibatch_kmeans B=50k

MVP3A_BEST          = 2.4391   # adaptive budget_filling B=10k
MVP3A_BEST_TAG      = "budget_filling_thr0.99_max64_ema128_B10000"

MVP2C_BY_BUDGET = {
    10000: {"utility_weighted": 2.2963, "minibatch_kmeans": float("nan")},
    25000: {"minibatch_kmeans": 2.2937},
    50000: {"minibatch_kmeans": 2.2852},
}

EPS = 1e-10


def _best_mvp2c_at_budget(budget: int) -> float:
    d = MVP2C_BY_BUDGET.get(int(budget), {})
    vals = [v for v in d.values() if not np.isnan(v)]
    return min(vals) if vals else float("nan")


def parse_tag(tag: str) -> dict:
    info = {"tag": tag, "method": "unknown", "budget": 0}
    m = re.search(r"_B(\d+)$", tag)
    if m:
        info["budget"] = int(m.group(1))
    for meth in ["aggregate_basic", "aggregate_budget_filling",
                  "aggregate_conflict_aware", "aggregate_promote_pure"]:
        if tag.startswith(meth):
            info["method"] = meth
            break
    for pattern, key in [
        (r"sim([\d.]+)",  "sim_threshold"),
        (r"_p([\d.]+)",   "min_compat_prob"),
        (r"_rel(\d+)",    "min_reliable_count"),
        (r"_prom(\d+)",   "promote_count"),
        (r"_pur([\d.]+)", "promote_purity"),
        (r"_ema(\d+)",    "ema_cap"),
    ]:
        mp = re.search(pattern, tag)
        if mp:
            try:
                info[key] = float(mp.group(1))
            except ValueError:
                pass
    return info


def load_all_q_metrics(q_read_dir: Path) -> list:
    rows = []
    for p in sorted(q_read_dir.glob("*/q_metrics.json")):
        try:
            with open(p) as f:
                m = json.load(f)
            info = parse_tag(m.get("tag", p.parent.name))
            m.update(info)
            rows.append(m)
        except Exception as e:
            print(f"  [warn] failed to load {p}: {e}")
    return rows


def load_health_metrics(reports_dir: Path) -> dict:
    """Load per-tag health json files -> dict tag -> health dict."""
    out = {}
    for p in sorted(reports_dir.glob("health_*.json")):
        try:
            with open(p) as f:
                h = json.load(f)
            out[h["tag"]] = h
        except Exception:
            pass
    return out


def load_state_stats(states_dir: Path, tag: str) -> dict:
    p = states_dir / f"{tag}.pt"
    if not p.exists():
        return {}
    try:
        s  = torch.load(p, weights_only=False)
        B  = int(s.get("actual_num_states", len(s["prototype_h"])))
        tc = s["total_counts"].numpy()[:B]
        return {
            "actual_num_states":   B,
            "mean_state_count":    float(tc.mean()),
            "median_state_count":  float(np.median(tc)),
            "frac_count_le_1":     float((tc <= 1).mean()),
            "frac_count_le_5":     float((tc <= 5).mean()),
            "mean_state_entropy":  float(s["state_entropy"].numpy()[:B].mean()),
            "num_buffers_promoted": int(s.get("num_buffers_promoted", 0)),
            "num_buffers_dropped":  int(s.get("num_buffers_dropped",  0)),
            "num_buffers_final":    int(s.get("num_buffers_final",    0)),
            "num_direct_updates":   int(s.get("num_direct_updates",   0)),
            "num_buffer_updates":   int(s.get("num_buffer_updates",   0)),
            "num_buffer_creates":   int(s.get("num_buffer_creates",   0)),
        }
    except Exception as e:
        print(f"  [warn] load_state_stats failed for {tag}: {e}")
        return {}


def build_df(q_metrics: list, states_dir, health_map: dict) -> pd.DataFrame:
    rows = []
    for m in q_metrics:
        tag    = m.get("tag", "")
        budget = int(m.get("budget", 0))
        q_nll  = float(m.get("q_state_nll", float("nan")))
        fixed  = float(m.get("best_fixed_nll", float("nan")))
        oracle = float(m.get("oracle_nll",    float("nan")))
        B      = int(m.get("num_states", 0))
        method = m.get("method", "unknown")

        mvp2c_same = _best_mvp2c_at_budget(budget)
        st = load_state_stats(states_dir, tag) if states_dir else {}
        h  = health_map.get(tag, {})

        # Merge all sources
        actual_B     = st.get("actual_num_states", B)
        med_count    = h.get("median_state_count",  st.get("median_state_count",  float("nan")))
        mean_count   = h.get("mean_state_count",    st.get("mean_state_count",    float("nan")))
        frac_le1     = h.get("frac_count_le_1",     st.get("frac_count_le_1",     float("nan")))
        frac_le5     = h.get("frac_count_le_5",     st.get("frac_count_le_5",     float("nan")))
        hit_at_4     = h.get("hit_at_4",     float("nan"))
        hit_at_1     = h.get("hit_at_1",     float("nan"))
        hit_at_8     = h.get("hit_at_8",     float("nan"))
        local_nll_k4 = h.get("fixed_local_nll_k4", float("nan"))
        health_pass  = h.get("health_pass", None)

        n_prom = h.get("num_buffers_promoted", st.get("num_buffers_promoted", 0))
        n_drop = h.get("num_buffers_dropped",  st.get("num_buffers_dropped",  0))
        n_fin  = h.get("num_buffers_final",    st.get("num_buffers_final",    0))
        n_dir  = h.get("num_direct_updates",   st.get("num_direct_updates",   0))
        n_bup  = h.get("num_buffer_updates",   st.get("num_buffer_updates",   0))
        n_bcr  = h.get("num_buffer_creates",   st.get("num_buffer_creates",   0))
        dir_frac = n_dir / max(n_dir + n_bup + n_bcr, 1)

        top_acts = m.get("top_actions", [])

        rows.append({
            "tag":                    tag,
            "method":                 method,
            "budget":                 budget,
            "actual_num_states":      actual_B,
            "budget_usage_fraction":  actual_B / max(budget, 1),
            "median_state_count":     med_count,
            "mean_state_count":       mean_count,
            "frac_count_le_1":        frac_le1,
            "frac_count_le_5":        frac_le5,
            "hit_at_1":               hit_at_1,
            "hit_at_4":               hit_at_4,
            "hit_at_8":               hit_at_8,
            "fixed_local_nll_k4":     local_nll_k4,
            "health_pass":            health_pass,
            "best_fixed_nll":         fixed,
            "q_state_nll":            q_nll,
            "oracle_nll":             oracle,
            "delta_vs_gpt":           q_nll - GPT_ONLY_NLL,
            "delta_vs_full_ds_qmlp":  q_nll - FULL_DS_QMLP,
            "delta_vs_mvp3a_best":    q_nll - MVP3A_BEST,
            "delta_vs_mvp2c_same":    q_nll - mvp2c_same,
            "delta_vs_mvp2c_best":    q_nll - MVP2C_BEST_OVERALL,
            "retrieval_usage":        float(m.get("retrieval_usage", float("nan"))),
            "avg_k_states":           float(m.get("avg_k_states",    float("nan"))),
            "oracle_gap_vs_q":        oracle - q_nll,
            "num_buffers_promoted":   n_prom,
            "num_buffers_dropped":    n_drop,
            "num_buffers_final":      n_fin,
            "num_direct_updates":     n_dir,
            "num_buffer_updates":     n_bup,
            "num_buffer_creates":     n_bcr,
            "direct_fraction":        dir_frac,
            "top_action_1":           top_acts[0]["name"] if top_acts else "",
            "top_action_1_frac":      top_acts[0]["frac"] if top_acts else float("nan"),
            "sim_threshold":          float(m.get("sim_threshold", float("nan"))),
            "min_compat_prob":        float(m.get("min_compat_prob", float("nan"))),
            "min_reliable_count":     float(m.get("min_reliable_count", float("nan"))),
            "promote_count":          float(m.get("promote_count", float("nan"))),
        })

    return pd.DataFrame(rows)


def _verdict(df: pd.DataFrame) -> str:
    if df.empty:
        return "INSUFFICIENT DATA"
    best_q = float(df["q_state_nll"].min())
    best_hit4 = float(df["hit_at_4"].max()) if "hit_at_4" in df.columns else float("nan")
    if best_q < FULL_DS_QMLP:
        return "VERY STRONG SUCCESS"
    if best_q < _best_mvp2c_at_budget(df["budget"].min()):
        return "STRONG SUCCESS"
    if best_q < MVP3A_BEST - 0.01:
        return "SUCCESS"
    if best_q < MVP3A_BEST:
        return "PARTIAL SUCCESS"
    return "FAILURE"


def _diagnosis(df: pd.DataFrame) -> str:
    if df.empty:
        return "C. Insufficient data."
    best_q = float(df["q_state_nll"].min())
    if best_q < FULL_DS_QMLP:
        return "A. Aggregate gate fixed the online WRITE issue."
    if best_q < MVP3A_BEST - 0.01:
        return "B. Aggregate gate partially helped but still does not match offline states."
    return "C. Aggregate gate failed; offline global clustering remains necessary."


def generate_report(df: pd.DataFrame, health_map: dict, out: Path) -> str:
    v = _verdict(df)
    diag = _diagnosis(df)

    lines = [
        "# MVP 3b Aggregate Gate / Buffer-Commit WRITE — Report",
        "",
        "## 1. Goal",
        "",
        "MVP 3a failed because online WRITE made permanent decisions too early.",
        "Naive CREATE and UPDATE forced every example into an immediate irrevocable commitment,",
        "producing either too few states (over-merging) or too many singleton states (over-splitting).",
        "",
        "MVP 3b tests whether an AGGREGATE gate — a temporary buffer layer — improves online state",
        "formation through delayed commitment.",
        "The gate chain is: MATCH → COMPATIBILITY → AGGREGATE → COMMIT",
        "",
        "## 2. Method",
        "",
        "**Persistent states**: long-lived prototypes with reliable token count distributions.",
        "**Aggregate buffers**: temporary proto-states that accumulate evidence before committing.",
        "**Promotion**: buffer → persistent state, when buffer.count >= promote_count",
        "  and buffer.entropy <= max_promote_entropy.",
        "**Drop**: buffer is discarded if it is old and has too little support.",
        "**Compatibility rule**: persistent state is compatible if",
        "  nearest_sim >= sim_threshold AND",
        "  (state_count < min_reliable_count OR p_state(y) >= min_compat_prob OR entropy <= max_entropy_for_blind_update)",
        "",
        "## 3. Canonical Baselines",
        "",
        "| Baseline | NLL |",
        "|----------|-----|",
        f"| GPT-only | {GPT_ONLY_NLL:.4f} |",
        f"| full_ds fixed kNN | {FULL_DS_FIXED:.4f} |",
        f"| full_ds Q-MLP | {FULL_DS_QMLP:.4f} |",
        f"| MVP 2c best (minibatch B=50k) | {MVP2C_BEST_OVERALL:.4f} |",
        f"| MVP 2c utility_weighted B=10k | 2.2963 |",
        f"| MVP 2c minibatch B=10k | 2.3015 |",
        f"| MVP 2c minibatch B=25k | 2.2937 |",
        f"| MVP 3a best adaptive (budget_filling B=10k) | {MVP3A_BEST:.4f} |",
        "",
    ]

    # ── Section 4: state health ───────────────────────────────────────────────
    lines += ["## 4. State Health Before Q-Read", ""]

    if not df.empty and "hit_at_4" in df.columns:
        lines += [
            "| tag | B | usage | med | sing% | hit@4 | local_nll_k4 | health |",
            "|-----|---|-------|-----|-------|-------|--------------|--------|",
        ]
        for _, r in df.sort_values("hit_at_4", ascending=False).iterrows():
            hp = r.get("health_pass", None)
            hstr = "PASS" if hp is True else ("FAIL" if hp is False else "n/a")
            lines.append(
                f"| {str(r['tag'])[:50]} | {int(r['actual_num_states']) if not np.isnan(r['actual_num_states']) else '?'} "
                f"| {r['budget_usage_fraction']:.0%} "
                f"| {r['median_state_count']:.1f} "
                f"| {r['frac_count_le_1']:.1%} "
                f"| {r['hit_at_4']:.2%} "
                f"| {r['fixed_local_nll_k4']:.4f} "
                f"| {hstr} |"
            )
        lines.append("")

        n_pass = int(df["health_pass"].sum()) if "health_pass" in df.columns else "?"
        avg_hit4_pass = df[df["health_pass"] == True]["hit_at_4"].mean() if "health_pass" in df.columns else float("nan")
        avg_med_pass  = df[df["health_pass"] == True]["median_state_count"].mean() if "health_pass" in df.columns else float("nan")

        lines.append(f"Health pass: {n_pass}/{len(df)} configs")
        if not np.isnan(avg_hit4_pass):
            lines.append(f"Average hit@4 (passing configs): {avg_hit4_pass:.2%}")
        if not np.isnan(avg_med_pass):
            lines.append(f"Average median state count (passing): {avg_med_pass:.1f}")
        lines.append("")

        # Compare to MVP 3a
        lines += [
            "**Comparison to MVP 3a adaptive (pre-fix baseline):**",
            f"- MVP 3a adaptive mean hit@4: 65.70%",
            f"- MVP 3a adaptive median state count: ~1.5",
            f"- MVP 3b best hit@4: {df['hit_at_4'].max():.2%}",
            f"- MVP 3b best median count: {df['median_state_count'].max():.1f}",
            "",
        ]
        if df["hit_at_4"].max() > 0.70:
            lines.append("**State quality improved over MVP 3a.** Aggregate buffer prevents singleton proliferation.")
        else:
            lines.append("**State quality did not clearly improve over MVP 3a.** Singleton fraction remains high.")
        lines.append("")

    # ── Section 5: main Q-read results ────────────────────────────────────────
    lines += ["## 5. Main Q-Read Results", ""]

    if df.empty:
        lines.append("No Q-eval results.")
    else:
        df_s = df.dropna(subset=["q_state_nll"]).sort_values("q_state_nll")

        lines += [
            "| method | B | actual_B | health | fixed | Q NLL | oracle | d_qmlp | d_mvp3a | d_mvp2c |",
            "|--------|---|----------|--------|-------|-------|--------|--------|---------|---------|",
        ]
        for _, r in df_s.iterrows():
            hp = r.get("health_pass", None)
            hstr = "PASS" if hp is True else ("FAIL" if hp is False else "n/a")
            lines.append(
                f"| {r['method']} | {int(r['budget'])} "
                f"| {int(r['actual_num_states']) if not np.isnan(r['actual_num_states']) else '?'} "
                f"| {hstr} "
                f"| {r['best_fixed_nll']:.4f} "
                f"| {r['q_state_nll']:.4f} "
                f"| {r['oracle_nll']:.4f} "
                f"| {r['delta_vs_full_ds_qmlp']:+.4f} "
                f"| {r['delta_vs_mvp3a_best']:+.4f} "
                f"| {r['delta_vs_mvp2c_same']:+.4f} |"
            )
        lines.append("")

        best_r = df_s.iloc[0]
        lines += [
            f"**Best result: {best_r['q_state_nll']:.4f}**  ({best_r['tag']})",
            "",
            f"- vs GPT-only ({GPT_ONLY_NLL:.4f}): {best_r['delta_vs_gpt']:+.4f}",
            f"- vs full_ds Q-MLP ({FULL_DS_QMLP:.4f}): {best_r['delta_vs_full_ds_qmlp']:+.4f}",
            f"- vs MVP 3a best ({MVP3A_BEST:.4f}): {best_r['delta_vs_mvp3a_best']:+.4f}",
            f"- vs MVP 2c same budget: {best_r['delta_vs_mvp2c_same']:+.4f}",
            f"- vs MVP 2c overall best ({MVP2C_BEST_OVERALL:.4f}): {best_r['delta_vs_mvp2c_best']:+.4f}",
            "",
        ]

        n_beat_mvp3a  = int((df_s["q_state_nll"] < MVP3A_BEST).sum())
        n_beat_qmlp   = int((df_s["q_state_nll"] < FULL_DS_QMLP).sum())
        n_beat_mvp2c  = int((df_s["q_state_nll"] < MVP2C_BEST_OVERALL).sum())
        lines += [
            f"Variants beating MVP 3a best ({MVP3A_BEST:.4f}): {n_beat_mvp3a}/{len(df_s)}",
            f"Variants beating full_ds_qmlp ({FULL_DS_QMLP:.4f}): {n_beat_qmlp}/{len(df_s)}",
            f"Variants beating MVP 2c best ({MVP2C_BEST_OVERALL:.4f}): {n_beat_mvp2c}/{len(df_s)}",
            "",
        ]

    # ── Section 6: verdict ────────────────────────────────────────────────────
    lines += [
        "## 6. Did AGGREGATE Help?",
        "",
        f"**Verdict: {v}**",
        "",
    ]

    if v == "VERY STRONG SUCCESS":
        lines.append(
            "aggregate_write beats full_ds_qmlp = 2.2977. "
            "Delayed commitment via aggregate buffers enables online states that match or exceed offline."
        )
    elif v == "STRONG SUCCESS":
        lines.append(
            "aggregate_write approaches offline MVP 2c at the same budget. "
            "Aggregate buffers substantially improve state quality over naked online WRITE."
        )
    elif v == "SUCCESS":
        lines.append(
            "aggregate_write clearly beats MVP 3a adaptive best = 2.4391. "
            "Delayed commitment improved state quality but gap to offline states remains."
        )
    elif v == "PARTIAL SUCCESS":
        lines.append(
            "aggregate_write marginally improves over MVP 3a but does not clearly beat it. "
            "Hyperparameter tuning or promotion rule revision needed."
        )
    else:
        lines.append(
            "aggregate_write failed to improve over MVP 3a adaptive best. "
            "States still have insufficient support. Offline global clustering remains necessary."
        )
    lines.append("")

    # ── Section 7: buffer behavior ─────────────────────────────────────────────
    lines += ["## 7. Buffer Behavior", ""]

    if not df.empty:
        lines += [
            "| tag | n_promoted | n_dropped | n_final_buf | direct% | buf_update% |",
            "|-----|------------|-----------|-------------|---------|-------------|",
        ]
        for _, r in df.sort_values("q_state_nll").head(8).iterrows():
            n_all = max(r["num_direct_updates"] + r["num_buffer_updates"] + r["num_buffer_creates"], 1)
            buf_upd_frac = r["num_buffer_updates"] / n_all
            lines.append(
                f"| {str(r['tag'])[:50]} "
                f"| {int(r['num_buffers_promoted'])} "
                f"| {int(r['num_buffers_dropped'])} "
                f"| {int(r['num_buffers_final'])} "
                f"| {r['direct_fraction']:.1%} "
                f"| {buf_upd_frac:.1%} |"
            )
        lines.append("")

    # ── Section 8: fragmentation comparison ────────────────────────────────────
    lines += ["## 8. Did Delayed Commitment Fix Fragmentation?", ""]

    if not df.empty and "median_state_count" in df.columns:
        mvp3a_median = 1.5
        mvp3a_hit4   = 0.6570
        mvp3a_sing   = 0.532   # max observed across MVP 3a variants

        best_med  = float(df["median_state_count"].max())
        best_hit4 = float(df["hit_at_4"].max()) if "hit_at_4" in df.columns else float("nan")
        best_sing = float(df["frac_count_le_1"].min()) if "frac_count_le_1" in df.columns else float("nan")

        lines += [
            "| Metric | MVP 3a adaptive | MVP 3b aggregate (best) | Improved? |",
            "|--------|-----------------|-------------------------|-----------|",
            f"| median state count | {mvp3a_median:.1f} | {best_med:.1f} | "
            f"{'YES' if best_med > mvp3a_median * 1.5 else 'NO'} |",
            f"| hit@4 | {mvp3a_hit4:.2%} | {best_hit4:.2%} | "
            f"{'YES' if best_hit4 > mvp3a_hit4 + 0.02 else 'NO'} |",
            f"| singleton fraction (best) | {mvp3a_sing:.2%} | {best_sing:.2%} | "
            f"{'YES' if best_sing < mvp3a_sing - 0.05 else 'NO'} |",
            "",
        ]

    # ── Section 9: per-method best ─────────────────────────────────────────────
    lines += ["## 9. Per-Method Best Results", ""]

    if not df.empty:
        df_q = df.dropna(subset=["q_state_nll"])
        if not df_q.empty:
            lines += [
                "| method | best budget | actual_B | hit@4 | Q NLL | d_vs_qmlp | d_vs_mvp3a |",
                "|--------|-------------|----------|-------|-------|-----------|------------|",
            ]
            for meth, grp in df_q.groupby("method"):
                r = grp.loc[grp["q_state_nll"].idxmin()]
                lines.append(
                    f"| {meth} | {int(r['budget'])} "
                    f"| {int(r['actual_num_states']) if not np.isnan(r['actual_num_states']) else '?'} "
                    f"| {r['hit_at_4']:.2%} "
                    f"| {r['q_state_nll']:.4f} "
                    f"| {r['delta_vs_full_ds_qmlp']:+.4f} "
                    f"| {r['delta_vs_mvp3a_best']:+.4f} |"
                )
            lines.append("")

    # ── Section 10: final diagnosis ────────────────────────────────────────────
    lines += [
        "## 10. Final Diagnosis",
        "",
        f"**{diag}**",
        "",
    ]

    if v in ("VERY STRONG SUCCESS", "STRONG SUCCESS", "SUCCESS"):
        lines += [
            "## 11. Next Recommendation",
            "",
            "Aggregate gate succeeded. Next step: move toward a learned WRITE policy.",
            "Options:",
            "- Learned compatibility threshold (contextual bandit on WRITE decisions)",
            "- Use aggregate-buffer actions (PROMOTE / KEEP / DROP / MERGE) as a supervised signal",
            "- Distill offline state assignments as teacher labels for online WRITE imitator",
            "",
        ]
    elif v == "PARTIAL SUCCESS":
        lines += [
            "## 11. Next Recommendation",
            "",
            "Tune predictive compatibility and promotion rules.",
            "Consider:",
            "- Lower promote_count (currently >= 4; try 2) to fill budget faster",
            "- Widen buffer compatibility (buffer_sim_threshold 0.985)",
            "- Add post-hoc prune: drop promoted states with count < 4 after stream ends",
            "",
        ]
    else:
        lines += [
            "## 11. Next Recommendation",
            "",
            "Online WRITE is insufficient. Use offline states as teacher labels for learned WRITE.",
            "Options:",
            "- Train a WRITE imitator to reproduce offline state assignments online",
            "- Use offline cluster assignments as supervision for an online assignment network",
            "- Hybrid: run offline clustering on a small seed set, then grow online",
            "",
        ]

    report = "\n".join(lines) + "\n"
    rpath  = out / "AGGREGATE_WRITE_REPORT.md"
    with open(rpath, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report: {rpath}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",     required=True)
    parser.add_argument("--states_dir", default=None)
    parser.add_argument("--force",      action="store_true")
    args = parser.parse_args()

    out   = Path(args.output)
    qrd   = out / "q_read"
    rdir  = out / "reports"
    sdir  = Path(args.states_dir) if args.states_dir else None

    csv_path = rdir / "aggregate_write_results.csv"
    rdir.mkdir(parents=True, exist_ok=True)

    if csv_path.exists() and csv_path.stat().st_size > 10 and not args.force:
        print(f"Loading cached results from {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print(f"\nLoading q_metrics.json from {qrd} ...")
        q_metrics = load_all_q_metrics(qrd)
        print(f"  Found {len(q_metrics)} result files")
        health_map = load_health_metrics(rdir)
        print(f"  Found {len(health_map)} health files")
        df = build_df(q_metrics, sdir, health_map)
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

    # Also save summary JSON
    summary_path = rdir / "aggregate_write_summary.json"
    if not df.empty:
        df_q = df.dropna(subset=["q_state_nll"])
        summary = {
            "n_configs_health_pass": int(df.get("health_pass", pd.Series([])).sum()) if "health_pass" in df.columns else 0,
            "n_configs_q_eval": len(df_q),
            "best_q_nll":  float(df_q["q_state_nll"].min()) if not df_q.empty else float("nan"),
            "best_tag":    str(df_q.loc[df_q["q_state_nll"].idxmin(), "tag"]) if not df_q.empty else "",
            "best_hit4":   float(df["hit_at_4"].max()) if "hit_at_4" in df.columns else float("nan"),
            "best_med_count": float(df["median_state_count"].max()) if "median_state_count" in df.columns else float("nan"),
            "baselines": {
                "gpt_only": GPT_ONLY_NLL, "full_ds_qmlp": FULL_DS_QMLP,
                "mvp3a_best": MVP3A_BEST, "mvp2c_best": MVP2C_BEST_OVERALL,
            },
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary: {summary_path}")

    if df.empty:
        print("No results to analyze."); return

    print(f"\n{len(df)} variants loaded")
    cols = [c for c in ["method","budget","q_state_nll","hit_at_4",
                         "median_state_count","frac_count_le_1","delta_vs_full_ds_qmlp"]
            if c in df.columns]
    print(df[cols].to_string(index=False))

    health_map = load_health_metrics(rdir)
    generate_report(df, health_map, out)

    if not df.dropna(subset=["q_state_nll"]).empty:
        df_q = df.dropna(subset=["q_state_nll"])
        best = df_q.loc[df_q["q_state_nll"].idxmin()]
        print(f"\nBest: {best['tag']}")
        print(f"  Q NLL = {best['q_state_nll']:.4f}")
        print(f"  delta_vs_mvp3a   = {best['delta_vs_mvp3a_best']:+.4f}")
        print(f"  delta_vs_qmlp    = {best['delta_vs_full_ds_qmlp']:+.4f}")
        print(f"  delta_vs_mvp2c   = {best['delta_vs_mvp2c_best']:+.4f}")


if __name__ == "__main__":
    main()

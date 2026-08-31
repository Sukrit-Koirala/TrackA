"""
analyze_bandit_write_rollout.py  --  MVP 4b Stage 3 analysis

Analyzes the bandit-write rollout: memory health over time, action usage,
post-hoc offline alignment with oracle, and comparison with baselines.
Produces report.md.

Usage:
  python src/analyze_bandit_write_rollout.py \\
    --output             outputs_mvp4b_bandit_write_fast \\
    --oracle_dir         outputs_mvp4a0_oracle_reconstruction \\
    --scorer             mlp \\
    --budget             10000
"""

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ACT_NAMES = ["UPDATE_STATE", "UPDATE_BUFFER", "PROMOTE_AND_UPDATE",
             "CREATE_BUFFER", "DEFER"]

REFERENCE_QNLL = {
    "GPT-only":          2.5533,
    "MVP3b_best":        2.4204,
    "oracle_teacher":    None,   # filled from evaluation
    "full_raw_200k":     2.2977,
}


def _load_json(p: Path) -> dict:
    if not p.exists():
        return {}
    try:
        return json.load(open(p))
    except json.JSONDecodeError:
        print(f"  WARNING: corrupt JSON at {p}, skipping")
        return {}


def _fmt(v, decimals=4) -> str:
    if v is None: return "—"
    if isinstance(v, float): return f"{v:.{decimals}f}"
    return str(v)


def _pct(v) -> str:
    if v is None: return "—"
    return f"{float(v)*100:.1f}%"


def section(title: str) -> str:
    return f"\n## {title}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output",     required=True)
    ap.add_argument("--oracle_dir", default=None,
                    help="outputs_mvp4a0_oracle_reconstruction for reference metrics")
    ap.add_argument("--scorer",     default="mlp", choices=["mlp", "gbm"])
    ap.add_argument("--budget",     type=int, default=10000)
    ap.add_argument("--force",      action="store_true")
    args = ap.parse_args()

    out_dir    = Path(args.output)
    roll_dir   = out_dir / "rollout"
    eval_dir   = out_dir / "evaluation"
    audit_dir  = out_dir / "reward_audit"
    model_dir  = out_dir / "scorer_models"
    analysis_dir = out_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    sentinel = analysis_dir / "analysis_complete.json"
    if sentinel.exists() and not args.force:
        print(f"[cached] {sentinel}")
        return

    method_name = f"bandit_write_{args.scorer}"
    B           = args.budget

    # ── load rollout stats + log ──────────────────────────────────────────────
    roll_stats = _load_json(roll_dir / "rollout_stats.json")
    log_path   = roll_dir / "rollout_log.parquet"
    log_df     = pd.read_parquet(log_path) if log_path.exists() else pd.DataFrame()

    # ── load evaluation metrics ────────────────────────────────────────────────
    bandit_m = _load_json(eval_dir / f"{method_name}_metrics.json")
    if not bandit_m:
        # try the per-budget directory structure
        bandit_m = _load_json(eval_dir / f"{method_name}_B{B}" / "q_metrics.json")

    # ── load oracle reference ─────────────────────────────────────────────────
    oracle_q = None
    seq_q    = None
    if args.oracle_dir:
        od = Path(args.oracle_dir)
        oracle_m = _load_json(od / "evaluation" / "teacher_reference_metrics.json")
        seq_m    = _load_json(od / "evaluation" / "sequential_running_mean_metrics.json")
        oracle_q = oracle_m.get("q_state_nll") if oracle_m else None
        seq_q    = seq_m.get("q_state_nll")    if seq_m    else None

    # ── load scorer eval ──────────────────────────────────────────────────────
    scorer_eval = _load_json(out_dir / "scorer_eval" / "scorer_eval.json")
    audit       = _load_json(audit_dir / "audit_report.json")
    train_schema = _load_json(model_dir / "training_schema.json")

    # ── memory growth analysis ─────────────────────────────────────────────────
    print("Analyzing memory health ...")
    mem_analysis = {}
    if not log_df.empty:
        final_row = log_df.iloc[-1]
        peak_pers = int(log_df["n_persistent"].max())
        peak_buf  = int(log_df["n_buffers"].max())
        final_pers = int(roll_stats.get("n_persistent", 0))
        final_buf  = int(roll_stats.get("n_buffers", 0))
        budget_util = (final_pers + final_buf) / max(B, 1)

        mem_analysis = {
            "peak_persistent":  peak_pers,
            "peak_buffers":     peak_buf,
            "final_persistent": final_pers,
            "final_buffers":    final_buf,
            "budget_utilization": budget_util,
        }

        # Memory growth by quartile
        if "step" in log_df.columns:
            step_max = int(log_df["step"].max()) + 1
            q_growth = {}
            for q in range(4):
                lo = step_max * q // 4
                hi = step_max * (q + 1) // 4
                qdf = log_df[(log_df["step"] >= lo) & (log_df["step"] < hi)]
                if not qdf.empty:
                    q_growth[f"Q{q+1}"] = {
                        "mean_persistent": float(qdf["n_persistent"].mean()),
                        "mean_buffers":    float(qdf["n_buffers"].mean()),
                    }
            mem_analysis["growth_by_quartile"] = q_growth

    # ── action usage ───────────────────────────────────────────────────────────
    action_counts = roll_stats.get("action_counts", {})
    action_fracs  = roll_stats.get("action_fracs",  {})
    n_stream      = roll_stats.get("n_stream", 1)

    print("Action usage:")
    for aname in ACT_NAMES:
        cnt  = action_counts.get(aname, 0)
        frac = action_fracs.get(aname, cnt / max(n_stream, 1))
        print(f"  {aname:<22}  {cnt:>8,}  ({frac*100:5.1f}%)")

    # ── Q-read comparison table ────────────────────────────────────────────────
    q_bandit = bandit_m.get("q_state_nll") if bandit_m else None
    REFERENCE_QNLL["oracle_teacher"] = oracle_q

    comparisons = [
        {"method": method_name,           "q_nll": q_bandit,
         "n_states": roll_stats.get("n_total_objects")},
        {"method": "oracle_seq_running",  "q_nll": seq_q,
         "n_states": B},
        {"method": "oracle_teacher",      "q_nll": oracle_q,
         "n_states": B},
        {"method": "MVP3b_best",          "q_nll": 2.4204,
         "n_states": 519},
        {"method": "GPT_only",            "q_nll": 2.5533,
         "n_states": 0},
        {"method": "full_raw_200k",       "q_nll": 2.2977,
         "n_states": 200000},
    ]
    cmp_df = pd.DataFrame(comparisons)
    cmp_df.to_csv(analysis_dir / "q_nll_comparison.csv", index=False)

    # ── post-hoc offline alignment: check if bandit beats oracle seq ──────────
    gaps = {}
    if q_bandit is not None:
        if oracle_q is not None:
            gaps["vs_oracle_teacher"] = q_bandit - oracle_q
        if seq_q is not None:
            gaps["vs_oracle_seq"]     = q_bandit - seq_q
        gaps["vs_mvp3b"]         = q_bandit - 2.4204
        gaps["vs_gpt"]           = q_bandit - 2.5533
        gaps["vs_full_raw"]      = q_bandit - 2.2977

    print("\nQ-NLL gaps (positive = worse than reference):")
    for k, v in gaps.items():
        direction = "better" if v < 0 else "worse"
        print(f"  {k:<28}  {v:+.4f}  ({direction})")

    # ── build verdict ─────────────────────────────────────────────────────────
    verdict_checks = {}
    if q_bandit is not None:
        verdict_checks["beats_gpt_only"]   = q_bandit < 2.5533
        verdict_checks["beats_mvp3b"]      = q_bandit < 2.4204
        verdict_checks["within_0.05_oracle"] = (
            abs(gaps.get("vs_oracle_teacher", 999)) <= 0.05
            if oracle_q else None
        )
    verdict = "PASS" if all(v for v in verdict_checks.values() if v is not None) else "PARTIAL"

    # ── generate report.md ────────────────────────────────────────────────────
    report_lines = [
        "# MVP 4b: Bandit-WRITE Rollout Analysis",
        f"\nScorer: `{args.scorer}`  Budget: {B}",
        "",
        section("Q-read Evaluation Results"),
        "| Method | Q-NLL | n_states |",
        "|--------|-------|----------|",
    ]
    for row in comparisons:
        q = _fmt(row["q_nll"])
        n = row.get("n_states") or "—"
        report_lines.append(f"| {row['method']} | {q} | {n} |")

    report_lines += [
        "",
        section("Q-NLL Gaps"),
        "| Comparison | Gap (+ = worse) |",
        "|------------|-----------------|",
    ]
    for k, v in gaps.items():
        report_lines.append(f"| {k} | {v:+.4f} |")

    report_lines += [
        "",
        section("Memory Health"),
        f"- Final persistent states: {mem_analysis.get('final_persistent', '—')}",
        f"- Final buffer states:     {mem_analysis.get('final_buffers', '—')}",
        f"- Peak persistent:         {mem_analysis.get('peak_persistent', '—')}",
        f"- Budget utilization:      {_pct(mem_analysis.get('budget_utilization'))}",
        "",
        section("Action Usage"),
        "| Action | Count | Fraction |",
        "|--------|-------|----------|",
    ]
    for aname in ACT_NAMES:
        cnt  = action_counts.get(aname, 0)
        frac = action_fracs.get(aname, 0)
        report_lines.append(f"| {aname} | {cnt:,} | {frac*100:.1f}% |")

    report_lines += [
        "",
        section("Scorer Performance"),
    ]
    if scorer_eval:
        for mname, mres in scorer_eval.get("models", {}).items():
            rho  = mres.get("spearman_rho", "—")
            mae  = mres.get("mae", "—")
            reg  = mres.get("regret", {})
            mrgret = reg.get("mean_regret", "—")
            match_frac = reg.get("action_match_frac", "—")
            report_lines.append(
                f"- **{mname.upper()}**: Spearman={_fmt(rho)}  "
                f"MAE={_fmt(mae)}  "
                f"regret={_fmt(mrgret)}  "
                f"action_match={_pct(match_frac)}"
            )

    report_lines += [
        "",
        section("Reward Dataset Audit"),
        f"- N records: {audit.get('n_records', '—'):,}" if audit else "- Not available",
        f"- Audit verdict: **{audit.get('verdict', '—')}**" if audit else "",
    ]
    if audit:
        for gate, passed in audit.get("gates", {}).items():
            status = "PASS" if passed else "FAIL"
            report_lines.append(f"  - {status}: {gate}")

    report_lines += [
        "",
        section("Overall Verdict"),
    ]
    for check, passed in verdict_checks.items():
        status = "PASS" if passed else ("WARN" if passed is None else "FAIL")
        report_lines.append(f"- {status}: {check}")
    report_lines.append(f"\n**==> MVP 4b verdict: {verdict}**")

    report_md = "\n".join(report_lines)
    report_path = out_dir / "report.md"
    with open(report_path, "w") as f:
        f.write(report_md)
    print(f"\nSaved: {report_path}")

    # ── save analysis JSON ────────────────────────────────────────────────────
    analysis = {
        "method_name":     method_name,
        "budget":          B,
        "q_nll_bandit":    q_bandit,
        "q_nll_gaps":      gaps,
        "memory_health":   mem_analysis,
        "action_counts":   action_counts,
        "action_fracs":    action_fracs,
        "verdict_checks":  verdict_checks,
        "verdict":         verdict,
    }
    with open(sentinel, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"Saved: {sentinel}")

    print(f"\n{'='*60}")
    print(f"MVP 4b Analysis Complete  -->  {verdict}")
    print(f"{'='*60}")
    print(f"  Bandit Q-NLL:      {_fmt(q_bandit)}")
    print(f"  Oracle seq Q-NLL:  {_fmt(seq_q)}")
    print(f"  Oracle ref Q-NLL:  {_fmt(oracle_q)}")


if __name__ == "__main__":
    main()

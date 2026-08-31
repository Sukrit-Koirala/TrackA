"""
analyze_mvp4b1.py  --  MVP 4b.1 Stage 7

Analyzes results and answers 7 research questions:
  Q1: Does the on-policy scorer fix distribution shift?
  Q2: Which rollout variant (A vs B bootstrap) wins?
  Q3: Does seed 999 (held-out) match seed 123 (dev)?
  Q4: Is MVP 4b.1 better than MVP 4b on Q-NLL?
  Q5: Does memory health improve vs MVP 4b?
  Q6: Do distinct reward signals matter? (local vs full vs penalized)
  Q7: Does MVP 4b.1 beat MVP3b?

Produces report.md in the output directory.

Usage:
  python src/analyze_mvp4b1.py \\
    --output outputs_mvp4b1_onpolicy_bandit_write_clean_fast \\
    --oracle_dir outputs_mvp4a0_oracle_reconstruction \\
    --scorer mlp --object_budget 10000
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mvp4b1_common import (
    ARTIFACT_VERSION, ACT_NAMES,
    atomic_write_json, validate_json_cache,
)

BASELINES = {
    "GPT-only":          2.5533,
    "MVP3b_best":        2.4204,
    "MVP4b_best":        2.4321,   # from previous run
    "Oracle-teacher":    2.3026,
    "Oracle-seq":        2.3061,
    "Full-raw-200k":     2.2977,
}


def _load_json(p: Path) -> dict:
    if not p.exists():
        return {}
    try:
        return json.load(open(p))
    except json.JSONDecodeError:
        print(f"  WARNING: corrupt JSON at {p}")
        return {}


def _fmt(v, d=4) -> str:
    if v is None: return "—"
    if isinstance(v, float): return f"{v:.{d}f}"
    return str(v)


def _pct(v) -> str:
    if v is None: return "—"
    return f"{float(v)*100:.1f}%"


def _delta(a, b, positive_is="better") -> str:
    if a is None or b is None:
        return "—"
    d = a - b
    direction = "better" if (d < 0) == (positive_is == "better") else "worse"
    return f"{d:+.4f} ({direction})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output",       required=True)
    ap.add_argument("--oracle_dir",   default=None)
    ap.add_argument("--scorer",       default="mlp")
    ap.add_argument("--object_budget", type=int, default=10000)
    ap.add_argument("--force",        action="store_true")
    args = ap.parse_args()

    out_dir  = Path(args.output)
    eval_dir = out_dir / "evaluation"
    sentinel = out_dir / "analysis" / "analysis_complete.json"
    sentinel.parent.mkdir(parents=True, exist_ok=True)

    cached = validate_json_cache(sentinel, ["artifact_version", "verdict"])
    if cached and not args.force:
        print(f"[cached] {sentinel}")
        return

    B = args.object_budget

    # Load all data
    eval_summary    = _load_json(eval_dir / "eval_summary.json")
    audit_report    = _load_json(out_dir / "reward_audit" / "audit_report.json")
    scorer_eval     = _load_json(out_dir / "scorer_eval" / "scorer_eval.json")
    train_schema    = _load_json(out_dir / "scorer_models" / "training_schema.json")

    # Rollout stats for each variant
    roll_dir  = out_dir / "rollout"
    roll_tags = [
        f"mvp4b1_{args.scorer}_v{v}_s{s}"
        for s in [123, 999] for v in ["A", "B"]
    ]
    roll_stats = {tag: _load_json(roll_dir / f"rollout_stats_{tag}.json")
                  for tag in roll_tags}

    # Oracle Q-NLL
    oracle_q, seq_q = None, None
    if args.oracle_dir:
        od  = Path(args.oracle_dir)
        om  = _load_json(od / "evaluation" / "teacher_reference_metrics.json")
        sm  = _load_json(od / "evaluation" / "sequential_running_mean_metrics.json")
        oracle_q = om.get("q_state_nll")
        seq_q    = sm.get("q_state_nll")

    results = eval_summary.get("results", {})

    def _q(tag):
        return results.get(tag, {}).get("q_state_nll")

    # Best across all variants
    valid_qs = [(tag, _q(tag)) for tag in roll_tags if _q(tag) is not None]
    best_tag, best_q = (None, None) if not valid_qs else \
        min(valid_qs, key=lambda x: x[1])

    # ── 7 Research Questions ──────────────────────────────────────────────────

    rq = {}

    # Q1: Does on-policy scorer fix distribution shift?
    # Compare seed 123 variants (dev rollout, trained on oracle+lm+onpolicy)
    q_a123 = _q(f"mvp4b1_{args.scorer}_vA_s123")
    q_b123 = _q(f"mvp4b1_{args.scorer}_vB_s123")
    rq["Q1_distribution_shift"] = {
        "question": "Does on-policy scorer fix the distribution shift that hurt MVP 4b?",
        "mvp4b_q_nll": BASELINES["MVP4b_best"],
        "mvp4b1_best_dev": min(v for v in [q_a123, q_b123] if v) if any([q_a123, q_b123]) else None,
        "verdict": (
            "YES - on-policy training improves distribution shift"
            if (best_q and best_q < BASELINES["MVP4b_best"])
            else "NO - distribution shift not resolved by on-policy training"
        ),
    }

    # Q2: A vs B bootstrap
    q_va_s123 = _q(f"mvp4b1_{args.scorer}_vA_s123")
    q_vb_s123 = _q(f"mvp4b1_{args.scorer}_vB_s123")
    rq["Q2_bootstrap_variant"] = {
        "question": "Which bootstrap variant (A=minimal, B=50) gives better Q-NLL?",
        "variant_A_s123": q_va_s123,
        "variant_B_s123": q_vb_s123,
        "winner": ("A" if (q_va_s123 or 999) < (q_vb_s123 or 999)
                   else "B" if q_vb_s123 else "unknown"),
    }

    # Q3: Generalization to held-out seed
    q_va_s999 = _q(f"mvp4b1_{args.scorer}_vA_s999")
    q_vb_s999 = _q(f"mvp4b1_{args.scorer}_vB_s999")
    dev_best  = min(v for v in [q_va_s123, q_vb_s123] if v) if any([q_va_s123, q_vb_s123]) else None
    hld_best  = min(v for v in [q_va_s999, q_vb_s999] if v) if any([q_va_s999, q_vb_s999]) else None
    gap = (hld_best - dev_best) if (dev_best and hld_best) else None
    rq["Q3_generalization"] = {
        "question": "Does MVP 4b.1 generalize to a held-out stream order (seed 999)?",
        "dev_best_q":       dev_best,
        "held_out_best_q":  hld_best,
        "gap_held_minus_dev": gap,
        "verdict": ("GENERALIZE" if gap and abs(gap) < 0.005
                    else "MODEST_GAP" if gap and abs(gap) < 0.02
                    else "POOR_GENERALIZATION"),
    }

    # Q4: MVP 4b.1 vs MVP 4b
    rq["Q4_vs_mvp4b"] = {
        "question": "Is MVP 4b.1 better than MVP 4b?",
        "mvp4b_q_nll":   BASELINES["MVP4b_best"],
        "mvp4b1_best":   best_q,
        "delta":         (best_q - BASELINES["MVP4b_best"]) if best_q else None,
        "verdict": ("YES" if (best_q and best_q < BASELINES["MVP4b_best"])
                    else "NO"),
    }

    # Q5: Memory health
    health = {}
    for tag in roll_tags:
        rs = roll_stats.get(tag, {})
        health[tag] = {
            "n_persistent": rs.get("n_persistent"),
            "n_buffers":    rs.get("n_buffers"),
            "n_total":      rs.get("n_total_objects"),
            "defer_pct": (rs.get("action_fracs", {}).get("DEFER", 0) * 100),
        }
    rq["Q5_memory_health"] = {
        "question": "Does memory fill healthily (no DEFER dominance)?",
        "per_variant": health,
        "verdict": (
            "HEALTHY"
            if all(h.get("defer_pct", 100) < 80 for h in health.values() if h.get("n_total"))
            else "DEGENERATE"
        ),
    }

    # Q6: Distinct rewards
    audit_verdict = audit_report.get("verdict", "UNKNOWN")
    gate_distinct = audit_report.get("gates", {}).get("distinct_rewards", False)
    rq["Q6_distinct_rewards"] = {
        "question": "Do distinct reward signals (local vs full vs penalized) matter?",
        "audit_verdict":        audit_verdict,
        "distinct_rewards_gate": gate_distinct,
        "verdict": ("YES - signals are distinct" if gate_distinct
                    else "FAIL - signals collapsed"),
    }

    # Q7: MVP 4b.1 vs MVP3b
    rq["Q7_vs_mvp3b"] = {
        "question": "Does MVP 4b.1 beat MVP3b?",
        "mvp3b_q_nll":   BASELINES["MVP3b_best"],
        "mvp4b1_best":   best_q,
        "delta":         (best_q - BASELINES["MVP3b_best"]) if best_q else None,
        "verdict": ("YES" if (best_q and best_q < BASELINES["MVP3b_best"])
                    else "NO"),
    }

    # ── Report.md ─────────────────────────────────────────────────────────────

    lines = [
        "# MVP 4b.1 Analysis: Clean On-Policy Bandit-WRITE",
        f"\nScorer: `{args.scorer}`  Budget: {B}",
        "",
        "## Q-NLL Results",
        "",
        "| Method | Q-NLL | vs GPT-only | vs MVP3b | vs Oracle |",
        "|--------|-------|-------------|----------|-----------|",
    ]
    rows = [
        ("GPT-only",       BASELINES["GPT-only"],      None, None, None),
        ("MVP3b",          BASELINES["MVP3b_best"],     None, None, None),
        ("MVP4b",          BASELINES["MVP4b_best"],     None, None, None),
        ("Oracle-teacher", oracle_q or BASELINES["Oracle-teacher"], None, None, None),
        ("Oracle-seq",     seq_q    or BASELINES["Oracle-seq"],     None, None, None),
    ]
    for tag in roll_tags:
        q = _q(tag)
        if q:
            rows.append((tag, q, None, None, None))
    for name, q, *_ in rows:
        vs_gpt   = _delta(q, BASELINES["GPT-only"],    "better") if q else "—"
        vs_mvp3b = _delta(q, BASELINES["MVP3b_best"],  "better") if q else "—"
        vs_oracle = _delta(q, oracle_q or BASELINES["Oracle-teacher"], "better") if q else "—"
        lines.append(f"| {name} | {_fmt(q)} | {vs_gpt} | {vs_mvp3b} | {vs_oracle} |")

    lines += ["", "## Research Questions", ""]
    for key, rqd in rq.items():
        lines.append(f"### {key}")
        lines.append(f"**{rqd['question']}**")
        verdict = rqd.get("verdict", "—")
        lines.append(f"\n**Verdict:** {verdict}\n")
        for k, v in rqd.items():
            if k in ("question", "verdict"):
                continue
            if isinstance(v, dict):
                lines.append(f"- {k}:")
                for kk, vv in v.items():
                    lines.append(f"  - {kk}: {_fmt(vv) if isinstance(vv, float) else vv}")
            else:
                lines.append(f"- {k}: {_fmt(v) if isinstance(v, float) else v}")
        lines.append("")

    lines += [
        "## Memory Health Per Variant",
        "",
        "| Variant | Persistent | Buffers | Total | DEFER% |",
        "|---------|-----------|---------|-------|--------|",
    ]
    for tag in roll_tags:
        h = health.get(tag, {})
        lines.append(
            f"| {tag} | {h.get('n_persistent','—')} | "
            f"{h.get('n_buffers','—')} | {h.get('n_total','—')} | "
            f"{_fmt(h.get('defer_pct'), 1)}% |"
        )

    lines += [
        "",
        "## Scorer Performance",
        "",
    ]
    sc_models = scorer_eval.get("models", {})
    for mname, mr in sc_models.items():
        rho  = mr.get("spearman_rho", "—")
        mae  = mr.get("mae", "—")
        reg  = mr.get("regret", {})
        lines.append(
            f"- **{mname.upper()}**: Spearman={_fmt(rho)}  "
            f"MAE={_fmt(mae)}  "
            f"regret={_fmt(reg.get('mean_regret'))}  "
            f"action_match={_pct(reg.get('action_match_frac'))}"
        )
        ps = mr.get("per_source", {})
        for src, sr in ps.items():
            lines.append(
                f"  - {src}: rho={_fmt(sr.get('spearman_rho'))}  "
                f"MAE={_fmt(sr.get('mae'))}"
            )

    # Overall verdict
    beats_mvp4b   = (best_q and best_q < BASELINES["MVP4b_best"])
    beats_mvp3b   = (best_q and best_q < BASELINES["MVP3b_best"])
    beats_gpt     = (best_q and best_q < BASELINES["GPT-only"])
    verdict       = ("STRONG" if beats_mvp3b
                     else "PARTIAL" if beats_mvp4b
                     else "FAIL")

    lines += [
        "",
        "## Overall Verdict",
        "",
        f"| Check | Result |",
        f"|-------|--------|",
        f"| beats GPT-only     | {'YES' if beats_gpt   else 'NO'} |",
        f"| beats MVP3b        | {'YES' if beats_mvp3b else 'NO'} |",
        f"| beats MVP4b        | {'YES' if beats_mvp4b else 'NO'} |",
        f"| memory healthy     | {'YES' if rq['Q5_memory_health']['verdict']=='HEALTHY' else 'NO'} |",
        f"| distinct rewards   | {'YES' if gate_distinct else 'NO'} |",
        f"",
        f"**==> MVP 4b.1 verdict: {verdict}**",
        f"",
        f"Best: `{best_tag}` Q-NLL={_fmt(best_q)}",
    ]

    report_md = "\n".join(lines)
    report_path = out_dir / "report.md"
    with open(report_path, "w") as f:
        f.write(report_md)
    print(f"\nSaved: {report_path}")

    analysis = {
        "artifact_version": ARTIFACT_VERSION,
        "scorer":           args.scorer,
        "object_budget":    B,
        "best_tag":         best_tag,
        "best_q_nll":       best_q,
        "baselines":        BASELINES,
        "beats_mvp4b":      beats_mvp4b,
        "beats_mvp3b":      beats_mvp3b,
        "beats_gpt":        beats_gpt,
        "verdict":          verdict,
        "research_questions": rq,
    }
    atomic_write_json(sentinel, analysis)
    print(f"Saved: {sentinel}")

    print("\n" + "=" * 60)
    print(f"MVP 4b.1 Analysis  -->  {verdict}")
    print("=" * 60)
    print(f"  Best Q-NLL: {_fmt(best_q)}  ({best_tag})")
    print(f"  vs MVP4b:  {_delta(best_q, BASELINES['MVP4b_best'], 'better')}")
    print(f"  vs MVP3b:  {_delta(best_q, BASELINES['MVP3b_best'], 'better')}")
    print(f"  vs oracle: {_delta(best_q, oracle_q or BASELINES['Oracle-teacher'], 'better')}")


if __name__ == "__main__":
    main()

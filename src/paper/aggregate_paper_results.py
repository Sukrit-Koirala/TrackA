"""
aggregate_paper_results.py  --  Track A Paper

Aggregates all experiment outputs into unified CSVs and a final Markdown
report for the paper.

Reads from:
  <output>/q_read/states/          — Q-metrics for state codebooks
  <output>/q_read/raw_baselines/   — Q-metrics for raw baselines
  <output>/efficiency/             — memory + speed results
  <output>/ablations/              — state-object ablation results
  <output>/reports/mechanism_diagnostics.csv

Writes:
  reports/main_results.csv
  reports/equal_budget_results.csv
  reports/efficiency_results.csv
  reports/ablation_results.csv
  reports/all_results_long.csv
  reports/TRACK_A_OFFLINE_PAPER_REPORT.md
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse, json, math
import numpy as np
import pandas as pd

# ── canonical baselines ───────────────────────────────────────────────────────
GPT_NLL       = 2.5533
FULL_DS_FIXED = 2.3873
FULL_DS_QMLP  = 2.2977


# ── loaders ───────────────────────────────────────────────────────────────────

def load_q_metrics_dir(q_dir: Path, memory_type: str) -> list[dict]:
    rows = []
    if not q_dir.exists():
        return rows
    for sub in sorted(q_dir.iterdir()):
        qm_path = sub / "q_metrics.json"
        if not qm_path.exists():
            continue
        bfm_path = sub / "best_fixed_metrics.json"
        oracle_path = sub / "oracle_metrics.json"
        try:
            with open(qm_path) as f:
                qm = json.load(f)
            fixed_nll  = float("nan")
            oracle_nll = float("nan")
            if bfm_path.exists():
                with open(bfm_path) as f:
                    fixed_nll = json.load(f).get("val_nll", float("nan"))
            if oracle_path.exists():
                with open(oracle_path) as f:
                    oracle_nll = json.load(f).get("val_nll", float("nan"))
            rows.append({
                "method":      qm.get("method", sub.name),
                "budget":      qm.get("budget", 0),
                "memory_type": memory_type,
                "num_objects": qm.get("n_states", 0),
                "fixed_nll":   fixed_nll,
                "q_nll":       qm.get("q_state_nll", float("nan")),
                "oracle_nll":  oracle_nll,
                "avg_k_states":       qm.get("avg_k_states",  float("nan")),
                "retrieval_usage":    qm.get("retrieval_usage", float("nan")),
                "delta_q_vs_fixed":   qm.get("delta_q_vs_fixed", float("nan")),
                "oracle_gap_vs_q":    qm.get("oracle_gap_vs_q", float("nan")),
                "tag":         sub.name,
            })
        except Exception as e:
            print(f"  [warn] {qm_path}: {e}")
    return rows


def load_full_raw_result(out: Path) -> dict | None:
    """Load full-raw datastore Q-read summary if available."""
    summary_path = out / "q_read" / "full_raw" / "evaluate_q_read_summary.json"
    if not summary_path.exists():
        return None
    try:
        with open(summary_path) as f:
            data = json.load(f)
        results = data.get("results", [])
        if not results:
            return None
        r = results[0]
        return {
            "full_raw_fixed_nll":  r.get("best_fixed_nll"),
            "full_raw_q_nll":      r.get("q_state_nll"),
            "full_raw_oracle_nll": r.get("oracle_nll"),
            "full_raw_n":          r.get("budget"),
        }
    except Exception as e:
        print(f"  [warn] full_raw summary: {e}")
        return None


def load_all_q_results(out: Path) -> pd.DataFrame:
    rows = []
    rows += load_q_metrics_dir(out / "q_read" / "states",        "state_codebook")
    rows += load_q_metrics_dir(out / "q_read" / "raw_baselines",  "raw_examples")
    # Also load from flat layout (legacy MVP 2c paths)
    for sub in sorted(out.iterdir()):
        if sub.is_dir() and (sub / "q_metrics.json").exists():
            if sub.name in ("q_read", "reports", "states", "raw_baselines",
                            "ablations", "diagnostics", "efficiency", "figures",
                            "logs", "configs", "sentinels"):
                continue
            qm_path = sub / "q_metrics.json"
            try:
                with open(qm_path) as f:
                    qm = json.load(f)
                rows.append({
                    "method":      qm.get("method", sub.name),
                    "budget":      qm.get("budget", 0),
                    "memory_type": "state_codebook",
                    "num_objects": qm.get("n_states", 0),
                    "fixed_nll":   qm.get("best_fixed_nll", float("nan")),
                    "q_nll":       qm.get("q_state_nll", float("nan")),
                    "oracle_nll":  qm.get("oracle_nll", float("nan")),
                    "avg_k_states":       qm.get("avg_k_states",  float("nan")),
                    "retrieval_usage":    qm.get("retrieval_usage", float("nan")),
                    "delta_q_vs_fixed":   qm.get("delta_q_vs_fixed", float("nan")),
                    "oracle_gap_vs_q":    qm.get("oracle_gap_vs_q", float("nan")),
                    "tag": sub.name,
                })
            except Exception:
                pass

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates(subset=["tag"])
    return df.sort_values(["memory_type", "method", "budget"])


# ── report generation ─────────────────────────────────────────────────────────

def compute_verdicts(df_main: pd.DataFrame, *,
                     gpt_nll: float = GPT_NLL,
                     full_ds_qmlp: float = FULL_DS_QMLP,
                     full_ds_fixed: float = FULL_DS_FIXED) -> dict:
    """Compute GREEN / YELLOW / RED verdicts for paper readiness."""
    if df_main.empty:
        return {"overall": "RED", "reason": "no results"}

    state_df = df_main[df_main["memory_type"] == "state_codebook"]
    raw_df   = df_main[df_main["memory_type"] == "raw_examples"]

    verdicts = {}

    # 1. Does state codebook beat GPT-only?
    if not state_df.empty:
        beats_gpt = (state_df["q_nll"] < gpt_nll).all()
        verdicts["beats_gpt_all"] = bool(beats_gpt)
    else:
        verdicts["beats_gpt_all"] = False

    # 2. Does state codebook beat equal-budget raw memory?
    if not state_df.empty and not raw_df.empty:
        beats_raw_count = 0
        total_budget_comparisons = 0
        best_raw_q = float("inf")
        for budget in state_df["budget"].unique():
            s_best = state_df[state_df["budget"] == budget]["q_nll"].min()
            r_best = raw_df[raw_df["budget"] == budget]["q_nll"].min()
            if not (math.isnan(s_best) or math.isnan(r_best)):
                total_budget_comparisons += 1
                if s_best < r_best:
                    beats_raw_count += 1
                best_raw_q = min(best_raw_q, r_best)
        verdicts["beats_raw_frac"]  = (beats_raw_count / max(total_budget_comparisons, 1))
        verdicts["best_raw_q_nll"]  = float(best_raw_q) if math.isfinite(best_raw_q) else float("nan")
    else:
        verdicts["beats_raw_frac"] = float("nan")
        verdicts["best_raw_q_nll"] = float("nan")

    # 3. Does state codebook approach full raw Q-MLP?
    if not state_df.empty:
        best_q = state_df["q_nll"].min()
        delta  = float(best_q - full_ds_qmlp)
        verdicts["best_q_nll"]           = float(best_q)
        verdicts["delta_vs_full_ds_qmlp"] = delta
        verdicts["approaches_full_ds_q"]  = delta <= 0.02
        verdicts["beats_full_ds_fixed"]   = float(best_q) < full_ds_fixed
    else:
        verdicts["best_q_nll"] = float("nan")
        verdicts["delta_vs_full_ds_qmlp"] = float("nan")
        verdicts["approaches_full_ds_q"] = False
        verdicts["beats_full_ds_fixed"]  = False

    # Overall verdict
    if (verdicts["beats_gpt_all"]
            and not math.isnan(verdicts["beats_raw_frac"])
            and verdicts["beats_raw_frac"] >= 0.8
            and verdicts["approaches_full_ds_q"]):
        verdicts["overall"] = "GREEN"
        verdicts["reason"]  = "beats GPT everywhere, beats raw at all budgets, approaches full Q-MLP"
    elif (verdicts["beats_gpt_all"]
          and not math.isnan(verdicts["beats_raw_frac"])
          and verdicts["beats_raw_frac"] >= 0.5):
        verdicts["overall"] = "YELLOW"
        verdicts["reason"]  = "beats GPT, beats raw in most cases, but does not approach full Q-MLP"
    else:
        verdicts["overall"] = "RED"
        verdicts["reason"]  = "does not consistently beat GPT or raw baselines"

    return verdicts


def format_nll(v) -> str:
    if isinstance(v, float) and math.isnan(v):
        return "—"
    try:
        return f"{float(v):.4f}"
    except Exception:
        return str(v)


def generate_report(
    df_main: pd.DataFrame,
    df_eff:  pd.DataFrame,
    df_abl:  pd.DataFrame,
    df_diag: pd.DataFrame,
    verdicts: dict,
    dataset:  str = "TinyStories",
    model:    str = "GPT-2 small",
    gpt_nll:      float = GPT_NLL,
    full_ds_qmlp: float = FULL_DS_QMLP,
    full_ds_fixed: float = FULL_DS_FIXED,
) -> str:
    lines = []
    p = lines.append

    p("# Track A Offline Paper: WRITE_IMITATION_REPORT")
    p("")
    p(f"Dataset: {dataset}  |  Backbone: {model}  |  Datastore: 200k")
    p("")

    # ── 1. Executive Verdict ─────────────────────────────────────────────────
    p("## 1. Executive Verdict")
    v = verdicts
    color = v["overall"]
    p(f"**{color}: {v.get('reason', '')}**")
    p("")
    bullets = [
        f"State codebooks beat GPT everywhere: {'YES' if v.get('beats_gpt_all') else 'NO'}",
        f"State codebooks beat equal-budget raw: "
        f"{v.get('beats_raw_frac', float('nan')):.0%} of budget comparisons",
        f"Best Q NLL: {format_nll(v.get('best_q_nll'))}  "
        f"(full raw Q-MLP: {FULL_DS_QMLP:.4f}  delta: "
        f"{format_nll(v.get('delta_vs_full_ds_qmlp', float('nan')))})",
    ]
    for b in bullets:
        p(f"- {b}")
    p("")
    if color == "GREEN":
        p("**Paper readiness: ready for paper draft.**")
    elif color == "GREEN_MAIN_RESULT_BUT_ABLATIONS_MISSING":
        p("**Paper readiness: main result qualifies — run ablations to complete.**")
    elif color == "YELLOW":
        p("**Paper readiness: workshop/arXiv prototype.  Needs more validation.**")
    else:
        p("**Paper readiness: internal result only.  Do not submit.**")
    p("")

    # ── 2. Main Claim ────────────────────────────────────────────────────────
    p("## 2. Main Claim Tested")
    p("> Predictive state codebooks replace raw examples with "
      "prototype + next-token distribution objects.  "
      "With learned Q-read control, they match or outperform raw-example "
      "retrieval using fewer memory objects.")
    p("")

    # ── 3. Main Results ──────────────────────────────────────────────────────
    p("## 3. Main Results")
    p("")
    p("### Baselines")
    p(f"| System | NLL |")
    p(f"|--------|-----|")
    p(f"| GPT-only | {gpt_nll:.4f} |")
    p(f"| Full raw datastore (fixed read) | {full_ds_fixed:.4f} |")
    p(f"| Full raw datastore (Q-MLP) | {full_ds_qmlp:.4f} |")
    p("")

    if not df_main.empty:
        state_df = df_main[df_main["memory_type"] == "state_codebook"]
        raw_df   = df_main[df_main["memory_type"] == "raw_examples"]

        p("### State Codebook Results")
        p("| method | budget | fixed_nll | q_nll | oracle_nll | delta_vs_gpt | delta_vs_full_q |")
        p("|--------|--------|-----------|-------|------------|--------------|-----------------|")
        for _, r in state_df.sort_values(["method", "budget"]).iterrows():
            dg  = float(r["q_nll"]) - gpt_nll      if not math.isnan(r["q_nll"]) else float("nan")
            dfq = float(r["q_nll"]) - full_ds_qmlp if not math.isnan(r["q_nll"]) else float("nan")
            p(f"| {r['method']:<28} | {int(r['budget']):>7} | "
              f"{format_nll(r['fixed_nll'])} | {format_nll(r['q_nll'])} | "
              f"{format_nll(r['oracle_nll'])} | {format_nll(dg)} | {format_nll(dfq)} |")
        p("")

        if not raw_df.empty:
            p("### Equal-Budget Raw Baselines")
            p("| method | budget | fixed_nll | q_nll |")
            p("|--------|--------|-----------|-------|")
            for _, r in raw_df.sort_values(["method", "budget"]).iterrows():
                p(f"| {r['method']:<28} | {int(r['budget']):>7} | "
                  f"{format_nll(r['fixed_nll'])} | {format_nll(r['q_nll'])} |")
            p("")

    # ── 4. Seed Stability ────────────────────────────────────────────────────
    p("## 4. Seed Stability")
    p("*(Run `--run_seed_sweep` with multiple seeds to populate this section.)*")
    p("")

    # ── 5. Equal-Budget Comparison ───────────────────────────────────────────
    p("## 5. Equal-Budget Comparison: State Codebook vs Raw Examples")
    p("")
    if not df_main.empty:
        state_df = df_main[df_main["memory_type"] == "state_codebook"]
        raw_df   = df_main[df_main["memory_type"] == "raw_examples"]
        budgets  = sorted(set(list(state_df["budget"].unique()) +
                               list(raw_df["budget"].unique())))
        if budgets:
            p("| budget | best_state_q_nll | best_raw_q_nll | state_wins |")
            p("|--------|-----------------|----------------|------------|")
            for b in budgets:
                sq = state_df[state_df["budget"] == b]["q_nll"].min()
                rq = raw_df[raw_df["budget"] == b]["q_nll"].min()
                wins = "YES" if (not math.isnan(sq) and not math.isnan(rq)
                                 and sq < rq) else ("?" if math.isnan(sq) or math.isnan(rq) else "NO")
                p(f"| {b:>7} | {format_nll(sq)} | {format_nll(rq)} | {wins} |")
            p("")
    else:
        p("*(No results yet.)*")
        p("")

    # ── 6. Efficiency ────────────────────────────────────────────────────────
    p("## 6. Efficiency")
    p("")
    if not df_eff.empty:
        cols = ["method", "budget", "memory_type", "num_objects",
                "total_memory_MB", "retrieval_ms_per_query", "q_nll"]
        present = [c for c in cols if c in df_eff.columns]
        p("| " + " | ".join(present) + " |")
        p("|" + "|".join("-"*(len(c)+2) for c in present) + "|")
        for _, r in df_eff.iterrows():
            def fmt(v, c):
                if c == "total_memory_MB":
                    return f"{float(v):.1f}" if not math.isnan(float(v)) else "nan"
                if c == "retrieval_ms_per_query":
                    return f"{float(v):.3f}" if not math.isnan(float(v)) else "nan"
                if c in ("q_nll", "fixed_nll"):
                    return format_nll(v)
                return str(v)
            p("| " + " | ".join(fmt(r.get(c, ""), c) for c in present) + " |")
        p("")
    else:
        p("*(Run `--run_efficiency` to populate.)*")
        p("")

    # ── 7. State-Object Ablations ────────────────────────────────────────────
    p("## 7. State-Object Ablations")
    p("*Does the token distribution matter?*")
    p("")
    if not df_abl.empty:
        p("| method | budget | variant | fixed_nll | q_nll |")
        p("|--------|--------|---------|-----------|-------|")
        for _, r in df_abl.sort_values(["base_method", "budget", "variant"]).iterrows():
            p(f"| {r['base_method']:<20} | {int(r['budget']):>7} | {r['variant']:<20} |"
              f" {format_nll(r['fixed_nll'])} | {format_nll(r['q_nll'])} |")
        p("")
    else:
        p("*(Run `--run_state_ablations` to populate.)*")
        p("")

    # ── 8. Q-Read Ablation ───────────────────────────────────────────────────
    p("## 8. Q-Read Ablation (2×2)")
    p("")
    if not df_main.empty:
        state_df = df_main[df_main["memory_type"] == "state_codebook"]
        raw_df   = df_main[df_main["memory_type"] == "raw_examples"]
        p("| | fixed read | Q-read |")
        p("|---|---|---|")
        if not state_df.empty:
            p(f"| State codebook | {format_nll(state_df['fixed_nll'].min())} | "
              f"{format_nll(state_df['q_nll'].min())} |")
        if not raw_df.empty:
            p(f"| Raw examples | {format_nll(raw_df['fixed_nll'].min())} | "
              f"{format_nll(raw_df['q_nll'].min())} |")
        p("")
    else:
        p("*(No results yet.)*")
        p("")

    # ── 9. Mechanism Diagnostics ─────────────────────────────────────────────
    p("## 9. Mechanism Diagnostics")
    p("")
    if not df_diag.empty:
        cols = ["method", "budget", "n_states", "median_state_count",
                "mean_state_purity", "hit_at_4", "mean_p_state_true", "q_nll"]
        present = [c for c in cols if c in df_diag.columns]
        p("| " + " | ".join(present) + " |")
        p("|" + "|".join("-"*(len(c)+2) for c in present) + "|")
        for _, r in df_diag.sort_values(["method", "budget"]).iterrows():
            p("| " + " | ".join(format_nll(r.get(c, "")) for c in present) + " |")
        p("")
    else:
        p("*(Run `--run_diagnostics` to populate.)*")
        p("")

    # ── 10. Extra Dataset / Model ─────────────────────────────────────────────
    p("## 10. Extra Dataset / Model Results")
    p("*(Run `--run_extra_dataset` or `--run_second_model` to populate.)*")
    p("")

    # ── 11. Failure Cases ────────────────────────────────────────────────────
    p("## 11. Failure Cases")
    p("")
    if not df_main.empty:
        state_df = df_main[df_main["memory_type"] == "state_codebook"]
        failures = state_df[state_df["q_nll"] >= GPT_NLL]
        if failures.empty:
            p("No failure cases: state codebooks beat GPT-only on all evaluated configs.")
        else:
            p("State codebook did NOT beat GPT-only on:")
            for _, r in failures.iterrows():
                p(f"  - {r['method']} B={r['budget']}: q_nll={format_nll(r['q_nll'])}")
    else:
        p("*(No results yet.)*")
    p("")

    # ── 12. Paper Readiness ───────────────────────────────────────────────────
    p("## 12. Paper Readiness Verdict")
    p("")
    color = verdicts["overall"]
    if color == "GREEN":
        p("**Result qualifies as a serious paper candidate.**")
        p(f"{dataset} result is strong on the fixed benchmark.  "
          "Cross-dataset generalization remains pending.")
    elif color == "GREEN_MAIN_RESULT_BUT_ABLATIONS_MISSING":
        p("**Main result qualifies — ablations have not yet been run.**")
        p("Run `--run_state_ablations` and re-aggregate to obtain a full GREEN verdict.")
    elif color == "YELLOW":
        p("**Result qualifies as a workshop or arXiv paper.**")
        p("TinyStories result is promising but generalisation evidence is weak.  "
          "Run additional datasets / models before main conference submission.")
    else:
        p("**Result is an internal prototype only.**")
        p("State codebooks do not consistently outperform baselines.  "
          "Further analysis required before any submission.")
    p("")

    # ── 13. Next Recommendations ──────────────────────────────────────────────
    p("## 13. Next Recommendations")
    p("")
    recs = []
    if not verdicts.get("beats_gpt_all"):
        recs.append("Investigate why state codebooks fail to beat GPT-only on some configs.")
    if verdicts.get("beats_raw_frac", 0) < 0.8:
        recs.append("Increase budget range or try more aggressive state formation to "
                    "improve over raw baselines.")
    if not verdicts.get("approaches_full_ds_q"):
        recs.append("Try larger state budgets (50k, 100k) to approach full raw Q-MLP.")
    recs.append("Run seed sweep (seeds 42, 123, 999) to confirm stability.")
    recs.append("Run WikiText-2 replication to confirm cross-dataset generalization.")
    recs.append("Run second model (GPT-2 medium or Pythia-160M) to confirm "
                "result is not GPT-2-small-specific.")
    for r in recs:
        p(f"- {r}")
    p("")
    p("---")
    p("*Generated by aggregate_paper_results.py*")

    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",  required=True,
                        help="paper output directory")
    parser.add_argument("--dataset",       default="TinyStories")
    parser.add_argument("--model",         default="GPT-2 small")
    parser.add_argument("--gpt_nll",       type=float, default=None,
                        help="GPT-only NLL for this dataset (default: TinyStories constant)")
    parser.add_argument("--full_ds_qmlp",  type=float, default=None,
                        help="Full-DS Q-MLP NLL reference (default: TinyStories constant)")
    parser.add_argument("--full_ds_fixed", type=float, default=None,
                        help="Full-DS fixed NLL reference (default: TinyStories constant)")
    parser.add_argument("--force",         action="store_true")
    args = parser.parse_args()

    out     = Path(args.output)
    rep_dir = out / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)

    report_path = rep_dir / "TRACK_A_OFFLINE_PAPER_REPORT.md"
    if report_path.exists() and not args.force:
        print(f"[cached] {report_path}")
        return

    print(f"\naggregate_paper_results")
    print(f"  Output: {out}")

    # ── load all results ──────────────────────────────────────────────────────
    print("  Loading Q-read results ...")
    df_main = load_all_q_results(out)
    print(f"  {len(df_main)} Q-read rows")

    full_raw = load_full_raw_result(out)
    if full_raw:
        print(f"  Full-raw Q NLL: {full_raw.get('full_raw_q_nll')}  "
              f"fixed: {full_raw.get('full_raw_fixed_nll')}")

    df_eff = pd.DataFrame()
    eff_path = out / "efficiency" / "memory_speed_results.csv"
    if eff_path.exists():
        df_eff = pd.read_csv(eff_path)
        print(f"  {len(df_eff)} efficiency rows")

    df_abl = pd.DataFrame()
    abl_path = out / "ablations" / "state_object_ablation_results.csv"
    if abl_path.exists():
        df_abl = pd.read_csv(abl_path)
        print(f"  {len(df_abl)} ablation rows")

    df_diag = pd.DataFrame()
    diag_path = rep_dir / "mechanism_diagnostics.csv"
    if diag_path.exists():
        df_diag = pd.read_csv(diag_path)
        print(f"  {len(df_diag)} diagnostic rows")

    # ── compute verdicts ──────────────────────────────────────────────────────
    verdicts = compute_verdicts(
        df_main,
        gpt_nll=args.gpt_nll           or GPT_NLL,
        full_ds_qmlp=args.full_ds_qmlp or FULL_DS_QMLP,
        full_ds_fixed=args.full_ds_fixed or FULL_DS_FIXED,
    )
    if verdicts["overall"] == "GREEN" and df_abl.empty:
        verdicts["overall"] = "GREEN_MAIN_RESULT_BUT_ABLATIONS_MISSING"
        verdicts["reason"]  = verdicts.get("reason", "") + " (ablations not yet run)"

    # Attach full_raw results to verdicts so the caller can read them
    if full_raw:
        verdicts["full_raw_fixed_nll"]  = full_raw.get("full_raw_fixed_nll")
        verdicts["full_raw_q_nll"]      = full_raw.get("full_raw_q_nll")
        verdicts["full_raw_oracle_nll"] = full_raw.get("full_raw_oracle_nll")
        verdicts["full_raw_n"]          = full_raw.get("full_raw_n")

    print(f"  Overall verdict: {verdicts['overall']}  ({verdicts.get('reason','')})")

    # ── save aggregated CSVs ──────────────────────────────────────────────────
    if not df_main.empty:
        state_df = df_main[df_main["memory_type"] == "state_codebook"]
        raw_df   = df_main[df_main["memory_type"] == "raw_examples"]
        if not state_df.empty:
            state_df.to_csv(rep_dir / "main_results.csv", index=False)
        if not raw_df.empty:
            raw_df.to_csv(rep_dir / "equal_budget_results.csv", index=False)
        df_main.to_csv(rep_dir / "all_results_long.csv", index=False)
        print(f"  Saved CSVs to {rep_dir}")

    if not df_eff.empty:
        df_eff.to_csv(rep_dir / "efficiency_results.csv", index=False)
    if not df_abl.empty:
        df_abl.to_csv(rep_dir / "ablation_results.csv", index=False)

    # ── verdicts JSON ─────────────────────────────────────────────────────────
    with open(rep_dir / "verdicts.json", "w") as f:
        json.dump(verdicts, f, indent=2)

    # ── generate report ───────────────────────────────────────────────────────
    report = generate_report(
        df_main, df_eff, df_abl, df_diag, verdicts,
        dataset=args.dataset, model=args.model,
        gpt_nll=args.gpt_nll           or GPT_NLL,
        full_ds_qmlp=args.full_ds_qmlp or FULL_DS_QMLP,
        full_ds_fixed=args.full_ds_fixed or FULL_DS_FIXED,
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  Report: {report_path}")
    print(f"  Verdict: {verdicts['overall']}")


if __name__ == "__main__":
    main()

"""
run_q_read_ablation.py  --  Track A Paper  (Experiment 5)

2×2 comparison: (memory_type) × (read_policy)
  raw_examples   + fixed read
  raw_examples   + Q-read
  state_codebook + fixed read
  state_codebook + Q-read

This defends against the criticism that results are only from Q-control,
not from the states themselves.

Reads existing Q-metrics from:
  <output>/q_read/states/           -- state codebook Q-read results
  <output>/q_read/raw_baselines/    -- raw baseline Q-read results

Outputs:
  reports/q_read_ablation_results.csv
  reports/q_read_ablation_summary.md
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse, json, math
import pandas as pd

GPT_NLL       = 2.5533
FULL_DS_FIXED = 2.3873
FULL_DS_QMLP  = 2.2977


def load_q_dir(q_dir: Path, memory_type: str) -> list[dict]:
    rows = []
    if not q_dir.exists():
        return rows
    for sub in sorted(q_dir.iterdir()):
        qm_path  = sub / "q_metrics.json"
        bfm_path = sub / "best_fixed_metrics.json"
        if not qm_path.exists():
            continue
        try:
            with open(qm_path) as f:
                qm = json.load(f)
            fixed_nll = float("nan")
            if bfm_path.exists():
                with open(bfm_path) as f:
                    fixed_nll = json.load(f).get("val_nll", float("nan"))
            rows.append({
                "method":      qm.get("method", sub.name),
                "budget":      int(qm.get("budget", 0)),
                "memory_type": memory_type,
                "fixed_nll":   fixed_nll,
                "q_nll":       qm.get("q_state_nll", float("nan")),
                "oracle_nll":  qm.get("oracle_nll",  float("nan")),
                "q_gain":      fixed_nll - qm.get("q_state_nll", float("nan")),
                "delta_vs_gpt":  qm.get("q_state_nll", float("nan")) - GPT_NLL,
                "delta_vs_full_raw_q":
                    qm.get("q_state_nll", float("nan")) - FULL_DS_QMLP,
            })
        except Exception as e:
            print(f"  [warn] {qm_path}: {e}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",  required=True,
                        help="paper output directory")
    parser.add_argument("--budgets", nargs="+", type=int, default=None)
    parser.add_argument("--force",   action="store_true")
    args = parser.parse_args()

    out     = Path(args.output)
    rep_dir = out / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)

    csv_path = rep_dir / "q_read_ablation_results.csv"
    if csv_path.exists() and not args.force:
        print(f"[cached] {csv_path}")
        return

    print(f"\nrun_q_read_ablation")
    print(f"  Output: {out}")

    rows = []
    rows += load_q_dir(out / "q_read" / "states",        "state_codebook")
    rows += load_q_dir(out / "q_read" / "raw_baselines",  "raw_examples")

    # Also check legacy flat layout
    for sub in sorted(out.iterdir()):
        if not sub.is_dir():
            continue
        if sub.name in ("q_read", "reports", "states", "raw_baselines",
                        "ablations", "diagnostics", "efficiency", "figures",
                        "logs", "configs", "sentinels"):
            continue
        qm_path  = sub / "q_metrics.json"
        bfm_path = sub / "best_fixed_metrics.json"
        if qm_path.exists():
            try:
                with open(qm_path) as f:
                    qm = json.load(f)
                fixed_nll = float("nan")
                if bfm_path.exists():
                    with open(bfm_path) as f:
                        fixed_nll = json.load(f).get("val_nll", float("nan"))
                rows.append({
                    "method":      qm.get("method", sub.name),
                    "budget":      int(qm.get("budget", 0)),
                    "memory_type": "state_codebook",
                    "fixed_nll":   fixed_nll,
                    "q_nll":       qm.get("q_state_nll", float("nan")),
                    "oracle_nll":  qm.get("oracle_nll",  float("nan")),
                    "q_gain":      fixed_nll - qm.get("q_state_nll", float("nan")),
                    "delta_vs_gpt":  qm.get("q_state_nll", float("nan")) - GPT_NLL,
                    "delta_vs_full_raw_q":
                        qm.get("q_state_nll", float("nan")) - FULL_DS_QMLP,
                })
            except Exception:
                pass

    if not rows:
        print("  No Q-read results found.  Run evaluate_q_read.py first.")
        return

    df = pd.DataFrame(rows)
    if args.budgets:
        df = df[df["budget"].isin(args.budgets)]

    df = df.drop_duplicates(subset=["method", "budget", "memory_type"])
    df = df.sort_values(["memory_type", "method", "budget"])
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}  ({len(df)} rows)")

    # ── 2×2 summary ───────────────────────────────────────────────────────────
    budgets = sorted(df["budget"].unique())
    lines   = [
        "# Experiment 5: Q-Read Ablation (2×2)\n",
        "Separates the effect of state formation from learned read control.\n",
    ]
    for budget in budgets:
        sub = df[df["budget"] == budget]
        state_row = sub[sub["memory_type"] == "state_codebook"]
        raw_row   = sub[sub["memory_type"] == "raw_examples"]

        def best_fixed(r): return r["fixed_nll"].min() if len(r) else float("nan")
        def best_q(r):     return r["q_nll"].min()     if len(r) else float("nan")

        sf = best_fixed(state_row)
        sq = best_q(state_row)
        rf = best_fixed(raw_row)
        rq = best_q(raw_row)

        def fmt(v): return f"{v:.4f}" if not math.isnan(v) else "—"

        lines += [
            f"### Budget B={budget}",
            "",
            f"|                    | Fixed read | Q-read |",
            f"|--------------------|------------|--------|",
            f"| **State codebook** | {fmt(sf)} | {fmt(sq)} |",
            f"| **Raw examples**   | {fmt(rf)} | {fmt(rq)} |",
            "",
        ]

        # Verdict
        if (not math.isnan(sq) and not math.isnan(rq) and sq < rq
                and not math.isnan(sf) and sf < rf):
            verdict = "States help under BOTH fixed and Q-read. Result is due to states."
        elif not math.isnan(sq) and not math.isnan(rq) and sq < rq:
            verdict = "Q-read with states wins; fixed read advantage unclear."
        elif not math.isnan(sf) and sf < rf:
            verdict = "States help under fixed read but not additionally with Q."
        else:
            verdict = "No clear advantage from states at this budget."
        lines.append(f"**Verdict B={budget}:** {verdict}\n")

    summary_path = rep_dir / "q_read_ablation_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {summary_path}")

    # Print 2×2 to stdout
    print(f"\n{'='*60}")
    print(f"  2×2 Q-Read Ablation  (GPT={GPT_NLL}  full_ds_Q={FULL_DS_QMLP})")
    print(f"{'='*60}")
    for budget in budgets:
        sub = df[df["budget"] == budget]
        sc  = sub[sub["memory_type"] == "state_codebook"]
        rw  = sub[sub["memory_type"] == "raw_examples"]
        print(f"\n  Budget B={budget}:")
        print(f"    State codebook:  fixed={best_fixed(sc):.4f}  Q={best_q(sc):.4f}")
        print(f"    Raw examples:    fixed={best_fixed(rw):.4f}  Q={best_q(rw):.4f}")


if __name__ == "__main__":
    main()

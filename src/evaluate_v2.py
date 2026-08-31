"""
evaluate_v2.py

Unified evaluation table for all controller methods.

Loads saved metrics from all prior steps and prints a single comparison table:
  - GPT-only
  - Best fixed kNN/action (CT-selected)
  - Old classifier MLP controller
  - Best entropy threshold heuristic (original)
  - Best extended threshold heuristic (run_heuristics.py)
  - Q-MLP full action set
  - Q-MLP small set A / B / C
  - Q-DecisionTree / Q-RandomForest / Q-GradientBoosting
  - [DIAG] Oracle per-example (val)
  - [DIAG] Val-oracle best fixed action

Columns: method, val_nll, val_ppl, avg_k, ret%, delta_vs_gpt, delta_vs_best_fixed

Saves: outputs/reports/final_metrics_v2.{csv,json}
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json
import math
import numpy as np
import pandas as pd

from utils import load_config, ensure_dirs, ppl


def load_json(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def fmt_delta(d: float) -> str:
    if math.isnan(d):
        return "    —   "
    sign = "+" if d > 0 else ""
    return f"{sign}{d:.4f}"


def main():
    parser = argparse.ArgumentParser(description="Unified evaluation table v2")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    reports_dir = Path(cfg["reports_dir"])

    # ── load all saved results ─────────────────────────────────────────────────
    baselines   = load_json(reports_dir / "fixed_baselines.json")
    ctrl_res    = load_json(reports_dir / "controller_results.json")
    q_res       = load_json(reports_dir / "q_controller_metrics.json")
    sk_res      = load_json(reports_dir / "sklearn_controller_metrics.json")
    best_heur   = load_json(reports_dir / "best_heuristics.json")

    # ── anchor values ──────────────────────────────────────────────────────────
    gpt_val_nll      = float("nan")
    best_fixed_val_nll = float("nan")

    if baselines:
        gpt_val_nll        = baselines["gpt_only_val_nll"]
        best_fixed_val_nll = baselines["best_train_selected_val_nll"]

    def delta_gpt(nll: float) -> float:
        return nll - gpt_val_nll if not math.isnan(nll) else float("nan")

    def delta_fixed(nll: float) -> float:
        return nll - best_fixed_val_nll if not math.isnan(nll) else float("nan")

    NaN = float("nan")
    rows: list[dict] = []

    def add(method: str, val_nll: float, avg_k: float, ret: float, note: str = ""):
        rows.append({
            "method":              method,
            "val_nll":             val_nll,
            "val_ppl":             ppl(val_nll) if not math.isnan(val_nll) else NaN,
            "avg_k":               avg_k,
            "retrieval_usage":     ret,
            "delta_vs_gpt":        delta_gpt(val_nll),
            "delta_vs_best_fixed": delta_fixed(val_nll),
            "note":                note,
        })

    # ── baseline rows ──────────────────────────────────────────────────────────
    if baselines:
        add("GPT only",
            baselines["gpt_only_val_nll"], 0.0, 0.0)
        add("Best fixed kNN (CT-selected)",
            baselines["best_train_selected_val_nll"],
            baselines["best_train_selected_val_k"],
            baselines["best_train_selected_val_ret"],
            note=baselines["best_train_selected_name"])
    else:
        add("GPT only", NaN, NaN, NaN, "fixed_baselines.json not found")

    # ── old entropy heuristic ─────────────────────────────────────────────────
    if ctrl_res and "entropy_heuristic_val" in ctrl_res:
        eh = ctrl_res["entropy_heuristic_val"]
        ec = ctrl_res.get("entropy_heuristic_config", {})
        add("Entropy heuristic (original)",
            eh["mean_nll"], eh["mean_k"], eh["retrieval_usage"],
            note=f"thresh={ec.get('threshold',NaN):.2f} act={ec.get('retrieval_action','?')}")
    else:
        add("Entropy heuristic (original)", NaN, NaN, NaN, "controller_results.json not found")

    # ── old classifier MLP ────────────────────────────────────────────────────
    if ctrl_res and "learned_controller_val" in ctrl_res:
        lc = ctrl_res["learned_controller_val"]
        add("Old classifier MLP",
            lc["nll"], lc["mean_k"], lc["retrieval_usage"])
    else:
        add("Old classifier MLP", NaN, NaN, NaN, "controller_results.json not found")

    # ── best extended heuristic ───────────────────────────────────────────────
    if best_heur:
        add("Best threshold heuristic (extended)",
            best_heur["val_nll"], best_heur["val_k"], best_heur["val_ret_usage"],
            note=f"feat={best_heur['feature']} {best_heur['direction']} "
                 f"act={best_heur['retrieval_action']}")
    else:
        add("Best threshold heuristic (extended)", NaN, NaN, NaN,
            "run run_heuristics.py first")

    # ── Q-MLP variants ────────────────────────────────────────────────────────
    for set_name in ("full", "A", "B", "C"):
        key = f"Q-MLP-{set_name}"
        if q_res and key in q_res:
            m = q_res[key]
            add(key, m["mean_nll"], m["mean_k"], m["retrieval_usage"])
        else:
            add(key, NaN, NaN, NaN, "run train_q_controller.py first")

    # ── sklearn Q-controllers ─────────────────────────────────────────────────
    for sk_name in ("Q-DecisionTree", "Q-RandomForest", "Q-GradientBoosting"):
        if sk_res and sk_name in sk_res:
            m = sk_res[sk_name]
            add(sk_name, m["mean_nll"], m["mean_k"], m["retrieval_usage"])
        else:
            add(sk_name, NaN, NaN, NaN, "run train_sklearn_controllers.py first")

    # ── diagnostics ───────────────────────────────────────────────────────────
    if ctrl_res and "oracle_val" in ctrl_res:
        ov = ctrl_res["oracle_val"]
        add("[DIAG] Oracle per-example (val)",
            ov["nll"], NaN, NaN, "val-oracle — not a fair comparison")
    if baselines:
        add("[DIAG] Best fixed (val-oracle)",
            baselines["oracle_val_nll"], NaN, NaN,
            f"action={baselines['oracle_val_name']}")

    # ── print table ───────────────────────────────────────────────────────────
    print()
    print("=" * 95)
    print("FINAL EVALUATION TABLE v2")
    print("=" * 95)
    hdr = (f"{'Method':<42} {'val_nll':>7}  {'val_ppl':>9}  {'avg_k':>6}  "
           f"{'ret%':>5}  {'Δgpt':>8}  {'Δfix':>8}")
    print(hdr)
    print("-" * 95)

    for r in rows:
        nll  = f"{r['val_nll']:>7.4f}"   if not math.isnan(r["val_nll"])          else "    nan"
        pp   = f"{r['val_ppl']:>9.2f}"   if not math.isnan(r["val_ppl"])           else "      nan"
        k    = f"{r['avg_k']:>6.1f}"     if (not math.isnan(r["avg_k"]) and
                                              r["avg_k"] == r["avg_k"])             else "   —  "
        ret  = f"{r['retrieval_usage']*100:>4.0f}%" if not math.isnan(
                                              r["retrieval_usage"])                 else "   —  "
        dgpt = fmt_delta(r["delta_vs_gpt"])
        dfix = fmt_delta(r["delta_vs_best_fixed"])
        m    = r["method"]
        tag  = "  " if not m.startswith("[") else ""
        print(f"{tag}{m:<42} {nll}  {pp}  {k}  {ret}  {dgpt}  {dfix}")

    print("=" * 95)

    # ── success condition ──────────────────────────────────────────────────────
    print()
    learned_rows = [r for r in rows
                    if r["method"] not in ("GPT only",
                                           "Best fixed kNN (CT-selected)",
                                           "Entropy heuristic (original)")
                    and not r["method"].startswith("[DIAG]")
                    and not math.isnan(r["val_nll"])]

    if learned_rows and not math.isnan(best_fixed_val_nll):
        best_learned = min(learned_rows, key=lambda r: r["val_nll"])
        if best_learned["val_nll"] < best_fixed_val_nll:
            print(f"✓ SUCCESS: {best_learned['method']} "
                  f"NLL {best_learned['val_nll']:.4f} < "
                  f"best fixed NLL {best_fixed_val_nll:.4f}  "
                  f"(Δ = {best_learned['val_nll']-best_fixed_val_nll:+.4f})")
        else:
            best_k = baselines.get("best_train_selected_val_k", NaN) if baselines else NaN
            eff    = [r for r in learned_rows
                      if r["val_nll"] <= best_fixed_val_nll * 1.01
                      and not math.isnan(r["avg_k"])
                      and r["avg_k"] < best_k]
            if eff:
                e = min(eff, key=lambda r: r["avg_k"])
                print(f"✓ SUCCESS (efficiency): {e['method']} "
                      f"matches best fixed NLL at lower avg k "
                      f"({e['avg_k']:.1f} vs {best_k:.1f})")
            else:
                print(f"✗ No learned controller beat best fixed kNN "
                      f"(best learned: {best_learned['method']} "
                      f"Δ = {best_learned['val_nll']-best_fixed_val_nll:+.4f})")

    # ── save ──────────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(reports_dir / "final_metrics_v2.csv", index=False)
    with open(reports_dir / "final_metrics_v2.json", "w") as f:
        json.dump(rows, f, indent=2)

    print(f"\nSaved: {reports_dir / 'final_metrics_v2.csv'}")
    print(f"Saved: {reports_dir / 'final_metrics_v2.json'}")
    print("\nDone.")


if __name__ == "__main__":
    main()

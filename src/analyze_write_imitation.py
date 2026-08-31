"""
analyze_write_imitation.py  --  MVP 4a Stage 6

Loads q_metrics.json and health JSON files, builds CSV,
and generates WRITE_IMITATION_REPORT.md.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse, json, math
import numpy as np
import pandas as pd

# ── canonical baselines ───────────────────────────────────────────────────────
GPT_ONLY_NLL   = 2.5533
FULL_DS_FIXED  = 2.3873
FULL_DS_QMLP   = 2.2977
MVP2C_BEST     = 2.2852   # minibatch_kmeans B=50k
MVP3A_BEST     = 2.4391
MVP3B_BEST     = 2.4204

MVP2C_BY_TEACHER = {
    "utility_weighted_B10000":    2.2963,
    "minibatch_kmeans_B10000":    2.3015,
    "utility_weighted_B25000":    2.2936,
    "minibatch_kmeans_B25000":    2.2937,
}

ACTION_NAMES = ["UPDATE_STATE", "UPDATE_BUFFER", "CREATE_BUFFER", "PROMOTE_BUFFER"]


# ── loaders ───────────────────────────────────────────────────────────────────

def load_q_metrics(qrd: Path) -> list[dict]:
    rows = []
    if not qrd.exists():
        return rows
    for d in sorted(qrd.iterdir()):
        mf = d / "q_metrics.json"
        if mf.exists():
            with open(mf) as f:
                rows.append(json.load(f))
    return rows


def load_health(rdir: Path) -> dict:
    out = {}
    for p in rdir.glob("health_*.json"):
        tag = p.stem[len("health_"):]
        with open(p) as f:
            out[tag] = json.load(f)
    return out


def load_train_metrics(mdir: Path) -> dict:
    out = {}
    for p in sorted(mdir.glob("*_metrics.json")):
        with open(p) as f:
            d = json.load(f)
        tname = d.get("teacher_name", p.stem.replace("_metrics", ""))
        out[tname] = d
    return out


# ── build dataframe ───────────────────────────────────────────────────────────

def parse_tag(tag: str) -> tuple:
    """Extract (teacher_name, budget) from state file stem."""
    # Pattern: <teacher_name>_imitator_B<budget>
    if "_imitator_B" in tag:
        parts = tag.rsplit("_imitator_B", 1)
        return parts[0], int(parts[1])
    return tag, -1


def build_df(q_metrics, sdir, health_map):
    rows = []
    for m in q_metrics:
        tag           = m["tag"]
        teacher, budget = parse_tag(tag)
        h = health_map.get(tag, {})

        # Try to load state file for extra metadata
        sf_path = sdir / f"{tag}.pt" if sdir else None
        sf_meta = {}
        if sf_path and sf_path.exists():
            try:
                import torch
                sf = torch.load(sf_path, weights_only=False)
                sf_meta = {
                    "actual_num_states": int(sf.get("actual_num_states", m["num_states"])),
                    "num_buffers_promoted": int(sf.get("num_buffers_promoted", 0)),
                    "num_direct_updates": int(sf.get("num_direct_updates", 0)),
                    "num_buffer_updates": int(sf.get("num_buffer_updates", 0)),
                    "num_buffer_creates": int(sf.get("num_buffer_creates", 0)),
                    "teacher_alignment_stats": sf.get("teacher_alignment_stats", {}),
                }
            except Exception:
                pass

        B    = sf_meta.get("actual_num_states", m.get("num_states", 0))
        q_nll = m.get("q_state_nll", float("nan"))

        mvp2c_same = MVP2C_BY_TEACHER.get(teacher, float("nan"))

        row = {
            "tag":               tag,
            "teacher_name":      teacher,
            "budget":            budget,
            "actual_num_states": B,
            "budget_usage_fraction": B / max(budget, 1),
            "median_state_count": h.get("median_state_count", float("nan")),
            "mean_state_count":  h.get("mean_state_count", float("nan")),
            "frac_count_le_1":   h.get("frac_count_le_1", float("nan")),
            "frac_count_le_5":   h.get("frac_count_le_5", float("nan")),
            "hit_at_1":          h.get("hit_at_1", float("nan")),
            "hit_at_4":          h.get("hit_at_4", float("nan")),
            "hit_at_8":          h.get("hit_at_8", float("nan")),
            "fixed_local_nll_k4": h.get("fixed_local_nll_k4", float("nan")),
            "health_pass":       h.get("health_pass", None),
            "health_reason":     h.get("health_reason", ""),
            "best_fixed_state_nll": m.get("best_fixed_nll", float("nan")),
            "q_state_nll":       q_nll,
            "oracle_nll":        m.get("oracle_nll", float("nan")),
            "gpt_nll":           m.get("gpt_nll", GPT_ONLY_NLL),
            "delta_vs_mvp3b_best":       q_nll - MVP3B_BEST,
            "delta_vs_mvp3a_best":       q_nll - MVP3A_BEST,
            "delta_vs_full_ds_qmlp":     q_nll - FULL_DS_QMLP,
            "delta_vs_mvp2c_same_teacher": q_nll - mvp2c_same,
            "delta_vs_mvp2c_best":       q_nll - MVP2C_BEST,
            "retrieval_usage":   m.get("retrieval_usage", float("nan")),
            "avg_k_states":      m.get("avg_k_states", float("nan")),
            "num_buffers_promoted": sf_meta.get("num_buffers_promoted", 0),
            "teacher_alignment_purity": sf_meta.get("teacher_alignment_stats", {}).get(
                                            "mean_teacher_purity", float("nan")),
        }
        if m.get("top_actions"):
            row["top_action"] = m["top_actions"][0].get("name", "") if m["top_actions"] else ""
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values("q_state_nll").reset_index(drop=True)
    return df


# ── report generation ─────────────────────────────────────────────────────────

def _verdict(df):
    df_q = df.dropna(subset=["q_state_nll"])
    if df_q.empty:
        return "NO RESULTS"
    best = float(df_q["q_state_nll"].min())
    if best < FULL_DS_QMLP:
        return "VERY STRONG SUCCESS"
    if best < 2.40:
        return "STRONG SUCCESS"
    if best < MVP3B_BEST - 0.005:
        return "SUCCESS"
    if best < MVP3B_BEST + 0.01:
        return "PARTIAL SUCCESS"
    return "FAILURE"


def generate_report(df, health_map, train_metrics, out: Path):
    lines = []
    a = lines.append

    a("# MVP 4a Write Imitation Report\n")
    a("## 1. Goal\n")
    a("Offline states (MVP 2c) work well. Online threshold/aggregate WRITE (MVP 3a/3b) does not.")
    a("MVP 4a tests whether an online WRITE policy can imitate the offline state-assignment")
    a("behavior using only local online features (no teacher ID at inference time).\n")

    # ── 2. Teacher setup ──────────────────────────────────────────────────────
    a("## 2. Teacher Setup\n")
    if not df.empty:
        teachers = df["teacher_name"].unique().tolist()
        for t in teachers:
            mv = MVP2C_BY_TEACHER.get(t, float("nan"))
            a(f"- `{t}`  offline Q NLL = {mv:.4f}")
    a("")
    a("Teacher assignments computed via nearest-prototype cosine similarity")
    a("(argmax over offline prototype set). No `assignment_for_datastore` available in")
    a("MVP 2b state files.\n")

    # ── 3. Imitation model quality ────────────────────────────────────────────
    a("## 3. Imitation Model Quality\n")
    if train_metrics:
        for tname, tm in sorted(train_metrics.items()):
            a(f"### {tname}\n")
            a(f"- Samples: {tm.get('n_samples', '?'):,}  "
              f"Train={tm.get('n_train', '?'):,}  Dev={tm.get('n_dev', '?'):,}")
            a(f"- Train accuracy: **{tm.get('train_acc', 0):.4f}**")
            a(f"- Dev accuracy: **{tm.get('dev_acc', 0):.4f}**")
            a("")
            a("Action distribution in training data:")
            dist = tm.get("action_distribution", {})
            total_d = sum(dist.values()) or 1
            for act, cnt in dist.items():
                a(f"  - {act}: {cnt:,} ({cnt/total_d:.1%})")
            a("")
            pc = tm.get("per_class_dev", {})
            if pc:
                a("Per-class dev metrics:")
                a("| Action | Prec | Rec | F1 |")
                a("|--------|------|-----|----|")
                for act, m in pc.items():
                    a(f"| {act} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |")
            a("")
    else:
        a("_No training metrics found._\n")

    # ── 4. State health ───────────────────────────────────────────────────────
    a("## 4. State Health\n")
    a("Comparison baselines:")
    a("- MVP 3a adaptive: hit@4 ≈ 65.70%, median count ≈ 1.5, singletons 51–74%")
    a("- MVP 3b aggregate: hit@4 51–64%, 0% singletons, actual states 366–519")
    a("- Offline MVP 2c: hit@4 ≈ 77.26%, median count ≈ 18.9\n")

    if not df.empty:
        cols = ["tag", "actual_num_states", "budget_usage_fraction",
                "median_state_count", "frac_count_le_1",
                "hit_at_4", "fixed_local_nll_k4", "health_pass"]
        dc = [c for c in cols if c in df.columns]
        a("| tag | B | usage | med | sing% | hit@4 | local_nll | health |")
        a("|-----|---|-------|-----|-------|-------|-----------|--------|")
        for _, r in df[dc].iterrows():
            tag_short = str(r.get("tag", ""))[:50]
            B_     = int(r.get("actual_num_states", 0))
            usage  = f"{r.get('budget_usage_fraction', 0):.0%}"
            med    = f"{r.get('median_state_count', 0):.1f}"
            sing   = f"{r.get('frac_count_le_1', 0):.1%}"
            h4     = f"{r.get('hit_at_4', float('nan')):.2%}" if not math.isnan(r.get('hit_at_4', float('nan'))) else "n/a"
            lnll   = f"{r.get('fixed_local_nll_k4', float('nan')):.4f}" if not math.isnan(r.get('fixed_local_nll_k4', float('nan'))) else "n/a"
            hp_    = "PASS" if r.get("health_pass") else ("FAIL" if r.get("health_pass") is False else "n/a")
            a(f"| {tag_short} | {B_} | {usage} | {med} | {sing} | {h4} | {lnll} | {hp_} |")
        a("")
        n_pass = int(df["health_pass"].sum()) if "health_pass" in df.columns else 0
        a(f"Health pass: {n_pass}/{len(df)} configs\n")

    # ── 5. Q-read results ─────────────────────────────────────────────────────
    a("## 5. Q-Read Results\n")
    a("| teacher | B | actual_B | fixed | Q NLL | oracle | d_3b | d_qmlp | d_2c_same |")
    a("|---------|---|----------|-------|-------|--------|------|--------|-----------|")

    df_q = df.dropna(subset=["q_state_nll"]) if not df.empty else df
    for _, r in df_q.iterrows():
        a(f"| {r['teacher_name']} | {r['budget']} | {r['actual_num_states']} "
          f"| {r.get('best_fixed_state_nll', float('nan')):.4f} "
          f"| {r['q_state_nll']:.4f} "
          f"| {r.get('oracle_nll', float('nan')):.4f} "
          f"| {r.get('delta_vs_mvp3b_best', float('nan')):+.4f} "
          f"| {r.get('delta_vs_full_ds_qmlp', float('nan')):+.4f} "
          f"| {r.get('delta_vs_mvp2c_same_teacher', float('nan')):+.4f} |")

    if not df_q.empty:
        best_row = df_q.iloc[0]
        a(f"\n**Best result: {best_row['q_state_nll']:.4f}**  ({best_row['tag']})\n")
        a(f"- vs GPT-only (2.5533):      {best_row['q_state_nll'] - GPT_ONLY_NLL:+.4f}")
        a(f"- vs full_ds Q-MLP (2.2977): {best_row['delta_vs_full_ds_qmlp']:+.4f}")
        a(f"- vs MVP 3b best (2.4204):   {best_row['delta_vs_mvp3b_best']:+.4f}")
        a(f"- vs MVP 2c best (2.2852):   {best_row['delta_vs_mvp2c_best']:+.4f}")
        n_beat_3b   = int((df_q["q_state_nll"] < MVP3B_BEST).sum())
        n_beat_qmlp = int((df_q["q_state_nll"] < FULL_DS_QMLP).sum())
        a(f"\nVariants beating MVP 3b best (2.4204): {n_beat_3b}/{len(df_q)}")
        a(f"Variants beating full_ds_qmlp (2.2977): {n_beat_qmlp}/{len(df_q)}")
    a("")

    # ── 6. Verdict ────────────────────────────────────────────────────────────
    a("## 6. Did Imitation Solve Online WRITE?\n")
    verdict = _verdict(df_q) if not df_q.empty else "NO RESULTS"
    a(f"**Verdict: {verdict}**\n")

    a("Criteria:")
    a("- SUCCESS: beats MVP 3b best (2.4204) and improves state health metrics")
    a("- STRONG SUCCESS: Q NLL < 2.40")
    a("- VERY STRONG SUCCESS: approaches offline MVP 2c B=10k (~2.30)")
    a("- FAILURE: cannot beat Aggregate WRITE or state health remains bad\n")

    # ── 7. Failure analysis ───────────────────────────────────────────────────
    a("## 7. Failure Analysis\n")
    if not df_q.empty:
        best = float(df_q["q_state_nll"].min())
        if best >= MVP3B_BEST - 0.001:
            a("Imitation did not beat MVP 3b Aggregate WRITE.\n")
            a("Possible causes:")
            a("- Distribution shift: teacher-forced rollout vs learned-policy rollout")
            a("- Classifier learned labels but rollout diverged from training distribution")
            a("- Feature set insufficient to distinguish good from bad WRITE decisions")
            a("- Teacher label quality: assignment by nearest prototype may not match")
            a("  offline k-means assignments used during MVP 2c training\n")
            a("Recommended next steps:")
            a("- DAgger-style correction: roll out learned policy, relabel, retrain")
            a("- Increase promote_count to require more buffer evidence")
            a("- Add pairwise compatibility features between example and state")
        else:
            a("Imitation improved over MVP 3b.\n")
            if best >= 2.40:
                a("Gap to 2.40 threshold remains. Recommend:")
                a("- DAgger correction loop to reduce distribution shift")
                a("- Reward-trained Q-WRITE using imitation as initialization")
            else:
                a("Strong results. Recommend proceeding to reward-trained Q-WRITE.")
    a("")

    # ── 8. Next recommendation ────────────────────────────────────────────────
    a("## 8. Next Recommendation\n")
    if not df_q.empty and float(df_q["q_state_nll"].min()) < MVP3B_BEST - 0.01:
        a("Imitation works. Proceed to reward-trained Q-WRITE / contextual bandit.")
        a("Use imitation model as initialization for reward-based fine-tuning.")
    elif not df_q.empty and float(df_q["q_state_nll"].min()) < MVP3B_BEST + 0.02:
        a("Partial success. Add DAgger-style data aggregation:")
        a("1. Roll out current learned policy on datastore")
        a("2. At each visited state/buffer, compute teacher-correct action using offline teacher")
        a("3. Add new samples to training dataset")
        a("4. Retrain and rebuild states")
    else:
        a("Imitation failed or produced marginal improvement. Options:")
        a("- Use offline states as persistent teacher memory directly (MVP 2c)")
        a("- Train pairwise compatibility model instead of action classifier")
        a("- Add DAgger loop before abandoning imitation approach")
    a("")

    report = "\n".join(lines)
    rpath  = out / "WRITE_IMITATION_REPORT.md"
    rpath.write_text(report, encoding="utf-8")
    print(f"Report: {rpath}")
    return rpath


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",      required=True)
    parser.add_argument("--states_dir",  default=None)
    parser.add_argument("--models_dir",  default=None)
    parser.add_argument("--force",       action="store_true")
    args = parser.parse_args()

    out  = Path(args.output)
    qrd  = out / "q_read"
    rdir = out / "reports"
    rdir.mkdir(parents=True, exist_ok=True)

    csv_path = rdir / "write_imitation_results.csv"

    # Load train metrics
    mdir = Path(args.models_dir) if args.models_dir else out / "models"
    train_metrics = load_train_metrics(mdir) if mdir.exists() else {}

    if csv_path.exists() and csv_path.stat().st_size > 10 and not args.force:
        print(f"Loading cached results from {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print(f"\nLoading q_metrics.json from {qrd} ...")
        q_metrics = load_q_metrics(qrd)
        print(f"  Found {len(q_metrics)} result files")
        health_map = load_health(rdir)
        print(f"  Found {len(health_map)} health files")
        sdir = Path(args.states_dir) if args.states_dir else out / "states"
        df = build_df(q_metrics, sdir if sdir.exists() else None, health_map)
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

    # Summary JSON
    if not df.empty:
        df_q = df.dropna(subset=["q_state_nll"])
        summary = {
            "n_configs_q_eval": len(df_q),
            "best_q_nll":   float(df_q["q_state_nll"].min()) if not df_q.empty else float("nan"),
            "best_tag":     str(df_q.iloc[0]["tag"]) if not df_q.empty else "",
            "mvp3b_best":   MVP3B_BEST,
            "mvp3a_best":   MVP3A_BEST,
            "full_ds_qmlp": FULL_DS_QMLP,
            "mvp2c_best":   MVP2C_BEST,
        }
        with open(rdir / "write_imitation_summary.json", "w") as f:
            import json; json.dump(summary, f, indent=2)

    if df.empty:
        print("No results to analyze."); return

    print(f"\n{len(df)} variants loaded")
    show_cols = [c for c in ["teacher_name", "budget", "actual_num_states",
                              "hit_at_4", "q_state_nll", "delta_vs_mvp3b_best"]
                 if c in df.columns]
    print(df[show_cols].to_string(index=False))

    health_map = load_health(rdir)
    generate_report(df, health_map, train_metrics, out)

    df_q = df.dropna(subset=["q_state_nll"])
    if not df_q.empty:
        best = df_q.iloc[0]
        print(f"\nBest: {best['tag']}")
        print(f"  Q NLL = {best['q_state_nll']:.4f}")
        print(f"  delta_vs_mvp3b  = {best.get('delta_vs_mvp3b_best', float('nan')):+.4f}")
        print(f"  delta_vs_qmlp   = {best.get('delta_vs_full_ds_qmlp', float('nan')):+.4f}")
        print(f"  delta_vs_mvp2c  = {best.get('delta_vs_mvp2c_best', float('nan')):+.4f}")


if __name__ == "__main__":
    main()

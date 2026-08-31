"""
audit_bandit_write_rewards.py  --  MVP 4b Stage 1 Audit

Validates the counterfactual reward dataset and issues a go/no-go verdict
for proceeding to scorer training.

Go/no-go criteria:
  1. Reward values are finite and non-trivial (std > 1e-4)
  2. At least 20% of decision points have meaningful best-action margin (>0.001)
  3. Best actions not dominated by one trivial action in >85% of cases
  4. Reward contains information beyond nearest cosine similarity
     (partial Spearman rho with cand_sim removed > 0.03)
  5. At least one feature family correlates measurably with reward_full
     (|Spearman rho| > 0.05 for at least one feature)
  6. Teacher agreement is diagnostic only -- reported but not a gate

Usage:
  python src/audit_bandit_write_rewards.py \\
    --output outputs_mvp4b_bandit_write_fast \\
    --reward_col reward_penalized
"""

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


FEATURE_NAMES = [
    "stream_progress", "gpt_nll", "gpt_entropy",
    "n_persistent", "n_buffers",
    "budget_frac_persistent", "budget_frac_buffers",
    "state_top1_sim", "state_top2_sim", "state_sim_gap",
    "buffer_top1_sim", "buffer_top2_sim",
    "act_update_state", "act_update_buffer", "act_promote_update",
    "act_create_buffer", "act_defer",
    "cand_sim", "cand_rank_norm", "cand_log_support",
    "cand_is_persistent", "cand_entropy", "cand_p_y_true",
    "cand_age_frac", "cand_recency_frac",
]

FEATURE_FAMILIES = {
    "global_context":    ["stream_progress", "gpt_nll", "gpt_entropy"],
    "memory_occupancy":  ["n_persistent", "n_buffers",
                          "budget_frac_persistent", "budget_frac_buffers"],
    "similarity_global": ["state_top1_sim", "state_top2_sim", "state_sim_gap",
                          "buffer_top1_sim", "buffer_top2_sim"],
    "action_type":       ["act_update_state", "act_update_buffer",
                          "act_promote_update", "act_create_buffer", "act_defer"],
    "candidate":         ["cand_sim", "cand_rank_norm", "cand_log_support",
                          "cand_is_persistent", "cand_entropy", "cand_p_y_true",
                          "cand_age_frac", "cand_recency_frac"],
}

ACT_NAMES = ["UPDATE_STATE", "UPDATE_BUFFER", "PROMOTE_AND_UPDATE",
             "CREATE_BUFFER", "DEFER"]


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return 0.0, 1.0
    r, p = stats.spearmanr(x[mask], y[mask])
    return float(r), float(p)


def section(title: str) -> str:
    return f"\n{'='*70}\n{title}\n{'='*70}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output",     required=True)
    ap.add_argument("--reward_col", default="reward_penalized",
                    choices=["reward_full", "reward_penalized",
                             "reward_local_only", "reward_local_history"])
    ap.add_argument("--margin_threshold", type=float, default=0.001,
                    help="min reward gap to count as meaningful margin")
    ap.add_argument("--force",  action="store_true")
    args = ap.parse_args()

    out_dir   = Path(args.output)
    data_dir  = out_dir / "reward_dataset"
    audit_dir = out_dir / "reward_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    sentinel = audit_dir / "audit_report.json"
    if sentinel.exists() and not args.force:
        print(f"[cached] {sentinel}")
        return

    # ── load data ─────────────────────────────────────────────────────────────
    actions_path = data_dir / "actions.parquet"
    if not actions_path.exists():
        raise FileNotFoundError(f"reward dataset not found: {actions_path}")

    print(f"Loading {actions_path} ...")
    df = pd.read_parquet(actions_path)
    print(f"  {len(df):,} action records")

    schema_path = data_dir / "feature_schema.json"
    schema = json.load(open(schema_path)) if schema_path.exists() else {}

    rcol = args.reward_col
    if rcol not in df.columns:
        print(f"WARNING: {rcol} not in columns, falling back to reward_full")
        rcol = "reward_full"

    reward = df[rcol].values.astype(np.float64)
    steps  = df["step"].values
    acts   = df["action_type"].values

    # ── criterion 1: finite and non-trivial ───────────────────────────────────
    print(section("CRITERION 1: Reward finiteness and spread"))
    n_inf = int(~np.isfinite(reward).sum())
    n_nan = int(np.isnan(reward).sum())
    reward_std  = float(np.nanstd(reward))
    reward_mean = float(np.nanmean(reward))
    reward_p5   = float(np.nanpercentile(reward, 5))
    reward_p95  = float(np.nanpercentile(reward, 95))

    print(f"  N records:   {len(df):,}")
    print(f"  N nan:       {n_nan}")
    print(f"  N inf:       {n_inf}")
    print(f"  mean:        {reward_mean:.5f}")
    print(f"  std:         {reward_std:.5f}")
    print(f"  p5 / p95:    {reward_p5:.5f} / {reward_p95:.5f}")

    c1_pass = (n_nan == 0 and n_inf == 0 and reward_std > 1e-4)
    print(f"  --> {'PASS' if c1_pass else 'FAIL'}")

    # ── per-action-type reward distributions ──────────────────────────────────
    print(section("Reward distribution by action type"))
    act_stats = {}
    for at, aname in enumerate(ACT_NAMES):
        mask = acts == at
        if not mask.any():
            continue
        r = reward[mask]
        act_stats[aname] = {
            "n":    int(mask.sum()),
            "mean": float(np.nanmean(r)),
            "std":  float(np.nanstd(r)),
            "p25":  float(np.nanpercentile(r, 25)),
            "p50":  float(np.nanpercentile(r, 50)),
            "p75":  float(np.nanpercentile(r, 75)),
        }
        s = act_stats[aname]
        print(f"  {aname:<22}  n={s['n']:>8,}  mean={s['mean']:+.5f}  "
              f"std={s['std']:.5f}  med={s['p50']:+.5f}")

    # ── criterion 2: meaningful best-action margin ────────────────────────────
    print(section("CRITERION 2: Best-action margin at decision points"))
    dp_groups  = df.groupby("step")[rcol]
    n_dp       = dp_groups.ngroups
    margins    = []
    for step_val, grp in dp_groups:
        sorted_r = np.sort(grp.values)[::-1]
        if len(sorted_r) >= 2:
            margins.append(sorted_r[0] - sorted_r[1])
        else:
            margins.append(0.0)
    margins = np.array(margins)

    frac_meaningful = float((margins > args.margin_threshold).mean())
    print(f"  Decision points:     {n_dp:,}")
    print(f"  Margin > {args.margin_threshold:.4f}:  "
          f"{frac_meaningful*100:.1f}%")
    print(f"  Mean margin:         {margins.mean():.5f}")
    print(f"  Median margin:       {np.median(margins):.5f}")

    c2_pass = frac_meaningful >= 0.20
    print(f"  --> {'PASS' if c2_pass else 'FAIL'} (need >=20%)")

    # ── criterion 3: action dominance ────────────────────────────────────────
    print(section("CRITERION 3: Action dominance check"))
    best_actions = []
    for step_val, grp in df.groupby("step"):
        best_idx = grp[rcol].idxmax()
        best_actions.append(int(df.loc[best_idx, "action_type"]))

    best_actions = np.array(best_actions)
    dom_counts   = {}
    for at, aname in enumerate(ACT_NAMES):
        cnt = int((best_actions == at).sum())
        frac = cnt / max(len(best_actions), 1)
        dom_counts[aname] = {"count": cnt, "frac": float(frac)}
        print(f"  {aname:<22}  best={cnt:>6,}  ({frac*100:5.1f}%)")

    max_dom_frac = max(v["frac"] for v in dom_counts.values())
    c3_pass = max_dom_frac <= 0.85
    print(f"  Max dominance:       {max_dom_frac*100:.1f}%")
    print(f"  --> {'PASS' if c3_pass else 'FAIL'} (need <=85%)")

    # ── criterion 4: reward info beyond cand_sim ──────────────────────────────
    print(section("CRITERION 4: Reward information beyond cand_sim"))
    feat_avail = [f for f in FEATURE_NAMES if f in df.columns]
    cand_sim_col = df["cand_sim"].values if "cand_sim" in df.columns else np.zeros(len(df))

    rho_raw,  p_raw  = _spearman(cand_sim_col, reward)
    print(f"  Spearman(cand_sim, reward):  rho={rho_raw:.4f}  p={p_raw:.4e}")

    # Compute residual reward after removing linear component of cand_sim
    mask_finite = np.isfinite(cand_sim_col) & np.isfinite(reward)
    if mask_finite.sum() > 10:
        from scipy.stats import rankdata
        r_sim = rankdata(cand_sim_col[mask_finite])
        r_rew = rankdata(reward[mask_finite])
        # linear residual in rank space
        m = np.cov(r_sim, r_rew)[0, 1] / (np.var(r_sim) + 1e-12)
        resid = r_rew - m * r_sim
        # correlation of other features with residual
        max_resid_rho = 0.0
        for fname in feat_avail:
            if fname == "cand_sim":
                continue
            fx = df[fname].values[mask_finite].astype(np.float64)
            rho_r, _ = _spearman(fx, resid)
            max_resid_rho = max(max_resid_rho, abs(rho_r))
    else:
        max_resid_rho = 0.0

    print(f"  Max |rho| with residual (other features): {max_resid_rho:.4f}")
    c4_pass = max_resid_rho > 0.03
    print(f"  --> {'PASS' if c4_pass else 'FAIL'} (need >0.03)")

    # ── criterion 5: feature-reward correlations ──────────────────────────────
    print(section("CRITERION 5: Feature-reward correlations"))
    feat_corr   = {}
    family_best = {}
    for fname in feat_avail:
        fx = df[fname].values.astype(np.float64)
        rho, p_val = _spearman(fx, reward)
        feat_corr[fname] = {"rho": rho, "p": p_val}

    for family, feats in FEATURE_FAMILIES.items():
        best_rho = 0.0
        for f in feats:
            if f in feat_corr:
                best_rho = max(best_rho, abs(feat_corr[f]["rho"]))
        family_best[family] = best_rho

    print(f"  {'Feature':<28}  {'Spearman rho':>13}  {'p-value':>12}")
    print(f"  {'-'*28}  {'-'*13}  {'-'*12}")
    for fname in feat_avail:
        c = feat_corr[fname]
        sig = "*" if abs(c["rho"]) > 0.05 else ""
        print(f"  {fname:<28}  {c['rho']:>+13.4f}  {c['p']:>12.4e} {sig}")

    print(f"\n  Family max |rho|:")
    for fam, rho in family_best.items():
        sig = "  <-- PASS" if rho > 0.05 else ""
        print(f"    {fam:<22}  {rho:.4f}{sig}")

    c5_pass = any(v > 0.05 for v in family_best.values())
    print(f"\n  --> {'PASS' if c5_pass else 'FAIL'} (need >=1 family with |rho|>0.05)")

    # ── teacher agreement (post-hoc diagnostic only) ──────────────────────────
    print(section("Teacher agreement (diagnostic only, not a gate)"))
    if "cand_is_promoted" in df.columns:
        # Oracle chose promote: how often is PROMOTE_AND_UPDATE the best action?
        # Approximate teacher action: at each step, teacher promotes when a buffer
        # reaches promotion_support threshold. We can only approximate this from
        # the dataset's action flags.
        promoted_mask = df["cand_is_promoted"].values.astype(bool)
        promo_rewards = {}
        for at, aname in enumerate(ACT_NAMES):
            r_prom  = reward[(acts == at) &  promoted_mask]
            r_nprom = reward[(acts == at) & ~promoted_mask]
            promo_rewards[aname] = {
                "mean_promoted":     float(np.nanmean(r_prom))  if len(r_prom)  > 0 else None,
                "mean_unpromoted":   float(np.nanmean(r_nprom)) if len(r_nprom) > 0 else None,
            }
            if len(r_prom) > 0 or len(r_nprom) > 0:
                print(f"  {aname:<22}  prom={promo_rewards[aname]['mean_promoted']!s:>10}  "
                      f"non-prom={promo_rewards[aname]['mean_unpromoted']!s:>10}")
    else:
        promo_rewards = {}
        print("  cand_is_promoted not available")

    # ── reward-signal stability check across quartiles ─────────────────────────
    print(section("Reward signal stability across stream quartiles"))
    step_max = int(steps.max()) + 1
    quartile_stats = {}
    for q in range(4):
        lo = step_max * q // 4
        hi = step_max * (q + 1) // 4
        mask = (steps >= lo) & (steps < hi)
        r_q  = reward[mask]
        quartile_stats[f"Q{q+1}"] = {
            "mean": float(np.nanmean(r_q)) if len(r_q) > 0 else None,
            "std":  float(np.nanstd(r_q))  if len(r_q) > 0 else None,
            "n":    int(mask.sum()),
        }
        s = quartile_stats[f"Q{q+1}"]
        print(f"  Q{q+1} (steps {lo:,}-{hi:,}):  "
              f"n={s['n']:>8,}  mean={s['mean']:+.5f}  std={s['std']:.5f}")

    # ── go/no-go verdict ──────────────────────────────────────────────────────
    print(section("GO / NO-GO VERDICT"))
    gates = {
        "C1_finite_and_nontrivial":    c1_pass,
        "C2_meaningful_margin_20pct":  c2_pass,
        "C3_no_trivial_dominance":     c3_pass,
        "C4_info_beyond_sim":          c4_pass,
        "C5_feature_correlation":      c5_pass,
    }
    all_pass = all(gates.values())
    for name, passed in gates.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")

    verdict = "GO" if all_pass else "NO-GO"
    print(f"\n  ==> Stage 1 verdict: {verdict}")

    # ── save ──────────────────────────────────────────────────────────────────
    report = {
        "reward_col":         rcol,
        "n_records":          len(df),
        "n_decision_points":  n_dp,
        "reward_stats": {
            "mean": reward_mean, "std": reward_std,
            "p5": reward_p5, "p95": reward_p95,
            "n_nan": n_nan, "n_inf": n_inf,
        },
        "action_reward_stats":   act_stats,
        "margin_stats": {
            "threshold":           args.margin_threshold,
            "frac_meaningful":     frac_meaningful,
            "mean_margin":         float(margins.mean()),
            "median_margin":       float(np.median(margins)),
        },
        "action_dominance":      dom_counts,
        "feature_correlations":  {f: feat_corr[f] for f in feat_avail if f in feat_corr},
        "family_best_rho":       family_best,
        "sim_partial_analysis": {
            "rho_raw":          rho_raw,
            "max_resid_rho":    max_resid_rho,
        },
        "teacher_agreement_diagnostic": promo_rewards,
        "quartile_stability":    quartile_stats,
        "gates":                 gates,
        "verdict":               verdict,
    }
    with open(audit_dir / "audit_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Save feature correlation CSV for easy inspection
    corr_rows = []
    for fname in feat_avail:
        fam = next((k for k, v in FEATURE_FAMILIES.items() if fname in v), "other")
        c   = feat_corr.get(fname, {"rho": 0.0, "p": 1.0})
        corr_rows.append({"feature": fname, "family": fam,
                           "spearman_rho": c["rho"], "p_value": c["p"]})
    corr_df = pd.DataFrame(corr_rows).sort_values("spearman_rho", key=abs, ascending=False)
    corr_df.to_csv(audit_dir / "feature_reward_correlations.csv", index=False)

    # Save per-decision-point best-action margin CSV
    step_margins = pd.DataFrame({"step": list(df.groupby("step").groups.keys()),
                                  "margin": margins})
    step_margins.to_parquet(audit_dir / "decision_margins.parquet", index=False)

    print(f"\nSaved: {audit_dir}/audit_report.json")
    print(f"Saved: {audit_dir}/feature_reward_correlations.csv")
    print(f"Saved: {audit_dir}/decision_margins.parquet")

    if verdict == "NO-GO":
        print("\nWARNING: Stage 1 audit failed. Investigate the reward dataset "
              "before proceeding to scorer training.")
        import sys
        sys.exit(1)

    print("\nAll gates passed. Proceed to Stage 2: scorer training.")


if __name__ == "__main__":
    main()

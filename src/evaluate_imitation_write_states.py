"""
evaluate_imitation_write_states.py  --  MVP 4a Stage 5

Health check + Q-state-read evaluation for imitation-built states.

Reuses MVP 3b evaluation code unchanged.
Health thresholds slightly relaxed vs MVP 3b (target > MVP 3a baseline).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse, json, time
import torch

from utils import get_device, set_seed
from evaluate_aggregate_write_states import (
    compute_state_health,
    eval_state_file,
    load_full_ds_baselines,
)

# Adjusted thresholds for MVP 4a: target above MVP 3a adaptive (65.70%)
HEALTH_MIN_HIT4  = 0.65    # vs 0.70 in MVP 3b
HEALTH_MAX_SING  = 0.40
HEALTH_MIN_MED   = 2
GPT_ONLY_NLL     = 2.5533
HEALTH_MAX_NLL_DELTA = 0.10  # more lenient for imitation states


def passes_health_check_4a(h: dict) -> tuple:
    if h["hit_at_4"] < HEALTH_MIN_HIT4:
        return False, f"hit@4={h['hit_at_4']:.2%} < {HEALTH_MIN_HIT4:.0%}"
    if h["frac_count_le_1"] > HEALTH_MAX_SING:
        return False, f"singleton_frac={h['frac_count_le_1']:.2%} > {HEALTH_MAX_SING:.0%}"
    if h["median_state_count"] < HEALTH_MIN_MED:
        return False, f"median_count={h['median_state_count']:.1f} < {HEALTH_MIN_MED}"
    nll_delta = h["fixed_local_nll_k4"] - GPT_ONLY_NLL
    if nll_delta > HEALTH_MAX_NLL_DELTA:
        return False, f"local_nll_k4={h['fixed_local_nll_k4']:.4f} > GPT+{HEALTH_MAX_NLL_DELTA}"
    return True, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",           required=True)
    parser.add_argument("--states_dir",       required=True)
    parser.add_argument("--output",           required=True)
    parser.add_argument("--max_q_samples",    type=int, default=800_000)
    parser.add_argument("--n_epochs",         type=int, default=30)
    parser.add_argument("--fast_action_grid", action="store_true")
    parser.add_argument("--eval_all",         action="store_true",
                        help="Q-eval even health-failing configs")
    parser.add_argument("--skip_health",      action="store_true")
    parser.add_argument("--force_neighbors",  action="store_true")
    parser.add_argument("--force_rewards",    action="store_true")
    parser.add_argument("--force_train",      action="store_true")
    parser.add_argument("--force_eval",       action="store_true")
    parser.add_argument("--force_health",     action="store_true")
    parser.add_argument("--force",            action="store_true")
    parser.add_argument("--tags",             nargs="*", default=None)
    parser.add_argument("--device",           default="cuda")
    parser.add_argument("--seed",             type=int, default=42)
    args = parser.parse_args()

    if args.force:
        args.force_neighbors = args.force_rewards = args.force_train = \
            args.force_eval = args.force_health = True

    set_seed(args.seed)
    device = get_device({"device": args.device})
    src    = Path(args.source)
    sdir   = Path(args.states_dir)
    out    = Path(args.output)
    rdir   = out / "reports"
    rdir.mkdir(parents=True, exist_ok=True)

    state_files = sorted(sdir.glob("*.pt"))
    if args.tags:
        state_files = [f for f in state_files if f.stem in args.tags]
    if not state_files:
        print(f"No state files found in {sdir}"); return

    print(f"\nevaluate_imitation_write_states")
    print(f"  Source:   {src}")
    print(f"  States:   {sdir}  ({len(state_files)} files)")
    print(f"  Output:   {out}")
    print(f"  Samples:  {args.max_q_samples:,}  fast_grid={args.fast_action_grid}")
    print(f"  eval_all: {args.eval_all}")

    # ── Stage 1: health ───────────────────────────────────────────────────────
    health_results = {}
    if not args.skip_health:
        print(f"\n{'='*55}\nStage 1: health checks\n{'='*55}")
        for i, sf in enumerate(state_files):
            print(f"[{i+1}/{len(state_files)}] {sf.stem}")
            hpath = rdir / f"health_{sf.stem}.json"

            if hpath.exists() and not args.force_health:
                with open(hpath) as f:
                    h = json.load(f)
                print(f"  [cached] hit@4={h['hit_at_4']:.2%}  "
                      f"med={h['median_state_count']:.1f}  "
                      f"sing={h['frac_count_le_1']:.1%}")
            else:
                t0 = time.time()
                try:
                    h = compute_state_health(sf, src, device)
                except Exception as e:
                    print(f"  [ERROR in health] {e}")
                    import traceback; traceback.print_exc()
                    continue
                with open(hpath, "w") as f:
                    json.dump(h, f, indent=2)
                elapsed = time.time() - t0
                print(f"  hit@4={h['hit_at_4']:.2%}  med={h['median_state_count']:.1f}  "
                      f"sing={h['frac_count_le_1']:.1%}  "
                      f"local_nll_k4={h['fixed_local_nll_k4']:.4f}  t={elapsed:.0f}s")

            ok, reason = passes_health_check_4a(h)
            h["health_pass"]   = ok
            h["health_reason"] = reason
            health_results[sf.stem] = h
            status = "PASS" if ok else f"FAIL ({reason})"
            print(f"  Health: {status}")

        print(f"\nHealth summary: "
              f"{sum(v['health_pass'] for v in health_results.values())}/{len(health_results)} "
              f"configs passed")

    # ── Stage 2: Q-state-read ─────────────────────────────────────────────────
    any_force = (args.force_health or args.force_neighbors or args.force_rewards
                 or args.force_train or args.force_eval)

    print(f"\n{'='*55}\nStage 2: Q-state-read evaluation\n{'='*55}")
    all_metrics = []

    for i, sf in enumerate(state_files):
        h    = health_results.get(sf.stem, {})
        ok   = h.get("health_pass", True)  # default True if health was skipped

        if not ok and not args.eval_all:
            print(f"\n[{i+1}/{len(state_files)}] {sf.stem}  [skipped: health FAIL]")
            continue

        print(f"\n[{i+1}/{len(state_files)}] {sf.stem}")
        m = eval_state_file(
            sf, src, out, device,
            max_q_samples=args.max_q_samples,
            n_epochs=args.n_epochs,
            fast_action_grid=args.fast_action_grid,
            force_neighbors=args.force_neighbors,
            force_rewards=args.force_rewards,
            force_train=args.force_train,
            force_eval=args.force_eval,
            seed=args.seed,
        )
        if m:
            all_metrics.append(m)

    print(f"\n{'='*55}")
    print(f"Evaluated {len(all_metrics)} state files")
    if all_metrics:
        best = min(all_metrics, key=lambda m: m.get("q_state_nll", float("inf")))
        print(f"Best Q NLL: {best['q_state_nll']:.4f}  ({best['tag']})")

    print("\nDone.")


if __name__ == "__main__":
    main()

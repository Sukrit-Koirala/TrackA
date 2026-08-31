"""
build_imitation_write_states.py  --  MVP 4a Stage 4

Roll out the trained WRITE imitation policy online to build
persistent predictive states.

Safety constraints applied:
  - PROMOTE_BUFFER only if buffer count >= min_promote_count
  - UPDATE_STATE only if >= 1 state exists
  - UPDATE_BUFFER only if >= 1 buffer exists
  - Fallback chain: invalid → next valid → CREATE_BUFFER

Saves state files in MVP 2b-compatible format.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse, json, time
import numpy as np
import torch

from utils import get_device, set_seed
# Import shared memory class and feature builder
from build_write_imitation_dataset import (
    OnlineMemory,
    build_features,
    ACTION_NAMES,
    ACT_UPDATE_STATE, ACT_UPDATE_BUFFER, ACT_CREATE_BUFFER, ACT_PROMOTE_BUFFER,
    N_ACTIONS, FEAT_DIM, EPS,
)
from train_write_imitation import WriteImitationMLP

MIN_PROMOTE_COUNT = 4   # safety: never promote a buffer with fewer examples


def load_policy(model_path: Path, device) -> WriteImitationMLP:
    ckpt   = torch.load(model_path, weights_only=False)
    in_dim = ckpt["in_dim"]
    model  = WriteImitationMLP(in_dim).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.x_mu  = ckpt["x_mu"]
    model.x_std = ckpt["x_std"]
    model.eval()
    return model


def apply_policy(model, feat_np: np.ndarray, device) -> int:
    x = torch.tensor(feat_np, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        return int(model.predict(x).item())


def run_imitation_rollout(
    ct_data, state_budget: int, buf_budget: int, device,
    model, hp: dict,
) -> tuple:
    N  = ct_data["h"].shape[0]
    d  = ct_data["h"].shape[1]
    mem = OnlineMemory(state_budget, buf_budget, d, hp.get("ema_cap", 128), device)

    h_all   = ct_data["h"].float()
    y_all   = ct_data["y"].numpy()
    nll_all = ct_data["nll_gpt"].numpy()
    ent_all = ct_data["gpt_entropy"].numpy()
    t1_all  = ct_data["gpt_top1_prob"].numpy()
    t2_all  = ct_data["gpt_top2_prob"].numpy()

    act_cnt  = {k: 0 for k in range(N_ACTIONS)}
    _norm_fn = lambda h: h / (h.norm() + EPS)

    for i in range(N):
        h_cpu = h_all[i]
        h_gpu = _norm_fn(h_cpu).to(device)
        y_i   = int(y_all[i])
        nll_i = float(nll_all[i])
        ent_i = float(ent_all[i])
        t1_i  = float(t1_all[i])
        t2_i  = float(t2_all[i])

        s_id, s_sim = mem.find_nearest_state(h_gpu)
        b_id, b_sim = mem.find_nearest_buffer(h_gpu)

        feat = build_features(
            mem, h_gpu, y_i, nll_i, ent_i, t1_i, t2_i,
            i, N, s_id, s_sim, b_id, b_sim,
        )

        action = apply_policy(model, feat, device)

        # ── Safety constraints + fallback chain ───────────────────────────────
        executed = False

        if action == ACT_PROMOTE_BUFFER:
            b_ok = (b_id >= 0
                    and float(mem.b_total[b_id]) >= MIN_PROMOTE_COUNT
                    and mem.n_states < state_budget)
            if b_ok:
                mem.promote_buffer(b_id, h_cpu, y_i, nll_i, ent_i, -1, i)
                executed = True
            else:
                action = ACT_UPDATE_BUFFER  # fall through

        if not executed and action == ACT_UPDATE_STATE:
            if s_id >= 0:
                mem.update_state(s_id, h_cpu, y_i, nll_i, ent_i, -1, i)
                executed = True
            else:
                action = ACT_UPDATE_BUFFER  # fall through

        if not executed and action == ACT_UPDATE_BUFFER:
            if b_id >= 0:
                mem.update_buffer(b_id, h_cpu, y_i, nll_i, ent_i, -1, i)
                executed = True
            else:
                action = ACT_CREATE_BUFFER  # fall through

        if not executed:  # ACT_CREATE_BUFFER (also catches fallbacks)
            mem.create_buffer(h_cpu, y_i, nll_i, ent_i, -1, i)
            action = ACT_CREATE_BUFFER

        act_cnt[action] = act_cnt.get(action, 0) + 1

        if (i + 1) % 10000 == 0:
            pct  = (i + 1) / N * 100
            dist = " ".join(f"{ACTION_NAMES[k]}={act_cnt[k]}" for k in range(N_ACTIONS))
            print(f"      {i+1}/{N} ({pct:.0f}%)  states={mem.n_states}  bufs={mem.n_buffers}  {dist}")

    return mem, act_cnt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",        required=True)
    parser.add_argument("--models_dir",    required=True)
    parser.add_argument("--teachers",      nargs="+",
                        default=["utility_weighted_B10000", "minibatch_kmeans_B10000"])
    parser.add_argument("--budgets",       nargs="+", type=int, default=[10000])
    parser.add_argument("--output",        required=True)
    parser.add_argument("--buf_frac",      type=float, default=0.05)
    parser.add_argument("--ema_cap",       type=int,   default=128)
    parser.add_argument("--device",        default="cuda")
    parser.add_argument("--force",         action="store_true")
    parser.add_argument("--seed",          type=int,   default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device  = get_device({"device": args.device})
    src     = Path(args.source)
    mdir    = Path(args.models_dir)
    out     = Path(args.output) / "states"
    out.mkdir(parents=True, exist_ok=True)

    print(f"\nbuild_imitation_write_states")
    print(f"  Source:   {src}")
    print(f"  Models:   {mdir}")
    print(f"  Budgets:  {args.budgets}")
    print(f"  Output:   {out}")

    ct_data = torch.load(src / "states" / "controller_train.pt", weights_only=False)

    for teacher_name in args.teachers:
        model_path = mdir / f"{teacher_name}_write_imitation_mlp.pt"
        if not model_path.exists():
            print(f"\n  [SKIP] {teacher_name} — model not found: {model_path}")
            continue

        print(f"\n  Loading policy: {model_path}")
        model = load_policy(model_path, device)

        for budget in args.budgets:
            stem     = f"{teacher_name}_imitator_B{budget}"
            out_path = out / f"{stem}.pt"

            if out_path.exists() and not args.force:
                print(f"  [cached] {stem}")
                continue

            buf_budget = max(200, int(budget * args.buf_frac))
            hp = {"ema_cap": args.ema_cap, "state_budget": budget, "buf_budget": buf_budget}

            print(f"\n  [{stem}]")
            print(f"    state_budget={budget}  buf_budget={buf_budget}")
            t0 = time.time()

            mem, act_cnt = run_imitation_rollout(ct_data, budget, buf_budget, device, model, hp)

            if mem.n_states == 0:
                print(f"    [WARN] No states created. Skipping.")
                continue

            print(f"    Final: states={mem.n_states}  buffers={mem.n_buffers}  "
                  f"elapsed={time.time()-t0:.0f}s")
            dist = {ACTION_NAMES[k]: int(act_cnt[k]) for k in range(N_ACTIONS)}
            print(f"    Actions: {dist}")

            # Add promotion stats
            n_promotes = int(act_cnt[ACT_PROMOTE_BUFFER])
            state_file = mem.export_state_file(
                method="write_imitation",
                teacher_name=teacher_name,
                budget=budget,
                action_counts=act_cnt,
                hyperparams=hp,
            )
            state_file["num_buffers_promoted"] = n_promotes
            state_file["num_buffers_dropped"]  = 0   # evictions counted differently
            state_file["num_direct_updates"]   = int(act_cnt[ACT_UPDATE_STATE])
            state_file["num_buffer_updates"]   = int(act_cnt[ACT_UPDATE_BUFFER])
            state_file["num_buffer_creates"]   = int(act_cnt[ACT_CREATE_BUFFER])

            torch.save(state_file, out_path)
            print(f"    Saved {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

"""
analyze_state_behavior.py  —  MVP 2c state analysis

Per-state statistics + Q-read usage for each (method, budget).

Loads state file, val neighbors, val predictions, q_model (for action list).
Distributes Q-improvement to selected states weighted by softmax(sims/tau).

Outputs:
  <out>/state_stats/<method>_B<budget>_state_stats.csv
  <out>/state_stats/<method>_B<budget>_state_stats.json
"""

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import numpy as np
import torch
import pandas as pd

EPS = 1e-10


def softmax_np(x: np.ndarray, tau: float = 1.0) -> np.ndarray:
    x = x.astype(np.float64) / tau
    x -= x.max()
    e = np.exp(x)
    return e / (e.sum() + EPS)


def try_tokenizer():
    try:
        from transformers import GPT2Tokenizer
        tok = GPT2Tokenizer.from_pretrained("gpt2")
        def decode(tid: int) -> str:
            try:
                return tok.decode([int(tid)])
            except Exception:
                return f"#{int(tid)}"
        return decode
    except Exception:
        return lambda tid: f"#{int(tid)}"


def parse_tag(stem: str):
    import re
    m = re.match(r"^(.+)_B(\d+)$", stem)
    return (m.group(1), int(m.group(2))) if m else (stem, 0)


def analyze_method_budget(
    method: str, budget: int,
    src: Path, states_dir: Path, q_read_dir: Path, out_dir: Path,
    max_top_tokens: int = 20,
) -> pd.DataFrame | None:
    tag         = f"{method}_B{budget}"
    state_path  = states_dir / f"{tag}.pt"
    mb_dir      = q_read_dir / tag
    nbr_path    = q_read_dir / "state_neighbors" / f"{tag}_val_top32.pt"
    pred_path   = mb_dir / "val_predictions.pt"
    model_path  = mb_dir / "q_model.pt"

    if not state_path.exists():
        print(f"  [skip] state file missing: {state_path}")
        return None

    states   = torch.load(state_path, weights_only=False)
    B        = len(states["prototype_h"])
    tok_ids  = states["top_k_token_ids"].long()       # [B, TOP_K]
    tok_cnts = states["top_k_token_counts"].float()   # [B, TOP_K]
    total    = states["total_counts"].float()          # [B]
    assigned = states["assigned_count"].long()         # [B]
    entropy  = states["state_entropy"].float()         # [B]
    purity   = states["state_purity"].float()          # [B]
    mean_nll = states.get("mean_nll", torch.zeros(B)).float()
    mean_gpt_ent = states.get("mean_gpt_entropy", torch.zeros(B)).float()

    decode = try_tokenizer()

    # Per-state token stats (from state file directly)
    n_unique   = (tok_cnts > 0).sum(-1).numpy().astype(int)   # [B]
    top1_tid   = tok_ids[:, 0].numpy()
    top1_cnt   = tok_cnts[:, 0].numpy()
    tot_np     = total.numpy()
    top1_prob  = np.where(tot_np > 0, top1_cnt / (tot_np + EPS), 0.0)

    # Q-usage accumulators
    times_sel  = np.zeros(B, dtype=np.int64)
    times_top1 = np.zeros(B, dtype=np.int64)
    times_top4 = np.zeros(B, dtype=np.int64)
    sum_sim    = np.zeros(B)
    sum_util_g = np.zeros(B)   # weighted utility vs GPT
    sum_pos_g  = np.zeros(B)
    sum_neg_g  = np.zeros(B)
    sum_util_f = np.zeros(B)   # weighted utility vs fixed

    has_q = nbr_path.exists() and pred_path.exists() and model_path.exists()
    if has_q:
        nbrs   = torch.load(nbr_path,  weights_only=False)
        preds  = torch.load(pred_path, weights_only=False)
        ckpt   = torch.load(model_path, weights_only=False)
        val_ids  = nbrs["ids"].numpy()          # [N_val, 32]
        val_sims = nbrs["sims"].numpy()         # [N_val, 32]
        chosen   = preds["chosen"].numpy()      # [N_val]
        q_nlls   = preds["q_nlls"].numpy()      # [N_val]
        fixed_nl = preds["fixed_nlls"].numpy()  # [N_val]
        actions  = ckpt["actions"]

        val_data = torch.load(src / "states" / "val.pt", weights_only=False)
        nll_gpt  = val_data["nll_gpt"].numpy()   # [N_val]

        N_val = len(chosen)
        for i in range(N_val):
            ai  = int(chosen[i])
            act = actions[ai]
            k   = act["k_states"]
            if k == 0:
                continue
            tau   = act["tau"]
            k_eff = min(k, val_ids.shape[1])
            sids  = val_ids[i, :k_eff]
            sims  = val_sims[i, :k_eff]
            wts   = softmax_np(sims, tau)

            util_g = float(nll_gpt[i]) - float(q_nlls[i])
            util_f = float(fixed_nl[i]) - float(q_nlls[i])

            for j in range(k_eff):
                sid = int(sids[j])
                w   = float(wts[j])
                times_sel[sid] += 1
                if j == 0: times_top1[sid] += 1
                if j < 4:  times_top4[sid] += 1
                sum_sim[sid]   += float(sims[j])
                sum_util_g[sid] += w * util_g
                if util_g > 0: sum_pos_g[sid] += w * util_g
                else:          sum_neg_g[sid] += w * util_g
                sum_util_f[sid] += w * util_f
    else:
        print(f"  [warn] Q outputs missing for {tag}, computing state-only stats")

    # Build rows
    rows = []
    for sid in range(B):
        k_tok = int(min(max_top_tokens, int((tok_cnts[sid] > 0).sum().item())))
        tids  = tok_ids[sid, :k_tok].tolist()
        tcnts = tok_cnts[sid, :k_tok].tolist()
        tot_s = float(total[sid])
        top_tokens = [
            {"id": int(t), "str": decode(t),
             "count": int(c), "prob": c / (tot_s + EPS)}
            for t, c in zip(tids, tcnts)
        ]

        sel  = int(times_sel[sid])
        msim = float(sum_sim[sid]  / sel) if sel > 0 else float("nan")
        mug  = float(sum_util_g[sid] / sel) if sel > 0 else float("nan")
        mpg  = float(sum_pos_g[sid]  / sel) if sel > 0 else float("nan")
        mng  = float(sum_neg_g[sid]  / sel) if sel > 0 else float("nan")
        muf  = float(sum_util_f[sid] / sel) if sel > 0 else float("nan")

        rows.append({
            "state_id":              sid,
            "method":                method,
            "budget":                budget,
            "assigned_count":        int(assigned[sid]),
            "total_count":           float(total[sid]),
            "state_entropy":         float(entropy[sid]),
            "state_purity":          float(purity[sid]),
            "num_unique_tokens":     int(n_unique[sid]),
            "top1_token_id":         int(top1_tid[sid]),
            "top1_token_str":        decode(int(top1_tid[sid])),
            "top1_token_prob":       float(top1_prob[sid]),
            "mean_nll":              float(mean_nll[sid]),
            "mean_gpt_entropy":      float(mean_gpt_ent[sid]),
            "times_selected":        sel,
            "times_top1":            int(times_top1[sid]),
            "times_top4":            int(times_top4[sid]),
            "mean_sim_when_sel":     msim,
            "net_util_vs_gpt":       float(sum_util_g[sid]),
            "pos_util_vs_gpt":       float(sum_pos_g[sid]),
            "neg_util_vs_gpt":       float(sum_neg_g[sid]),
            "net_util_vs_fixed":     float(sum_util_f[sid]),
            "mean_util_per_sel":     mug,
            "mean_pos_per_sel":      mpg,
            "mean_neg_per_sel":      mng,
            "mean_util_fixed_per_sel": muf,
            "top_tokens_json":       json.dumps(top_tokens[:10]),
        })

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_p  = out_dir / f"{tag}_state_stats.csv"
    json_p = out_dir / f"{tag}_state_stats.json"
    df.to_csv(csv_p, index=False)
    with open(json_p, "w") as f:
        json.dump(rows, f, indent=2)

    n_active = int((times_sel > 0).sum())
    top_util = float(df["net_util_vs_gpt"].max())
    print(f"  [{tag}]  B={B}  active={n_active}/{B} ({n_active/B:.1%})  "
          f"top_util={top_util:.4f}  total_util={float(df['net_util_vs_gpt'].sum()):.2f}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",     required=True)
    parser.add_argument("--states_dir", required=True)
    parser.add_argument("--q_read_dir", required=True)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--methods",  nargs="+",
                        default=["minibatch_kmeans", "utility_weighted", "query_kmeans"])
    parser.add_argument("--budgets",  nargs="+", type=int,
                        default=[10000, 25000, 50000])
    parser.add_argument("--force",    action="store_true")
    args = parser.parse_args()

    src        = Path(args.source)
    states_dir = Path(args.states_dir)
    q_read_dir = Path(args.q_read_dir)
    out        = Path(args.output) / "state_stats"

    print(f"\nanalyze_state_behavior")
    print(f"  States: {states_dir}  Q: {q_read_dir}  Out: {out}")

    for method in args.methods:
        for budget in args.budgets:
            tag = f"{method}_B{budget}"
            csv_p = out / f"{tag}_state_stats.csv"
            if csv_p.exists() and not args.force:
                print(f"  [cached] {tag}")
                continue
            analyze_method_budget(method, budget, src, states_dir, q_read_dir, out)


if __name__ == "__main__":
    main()

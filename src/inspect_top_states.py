"""
inspect_top_states.py  —  MVP 2c state inspection

Human-readable inspection files for top states by various criteria.

Loads state_stats CSV (from analyze_state_behavior.py), state file,
val neighbors, val predictions, val data.

Outputs per (method, budget):
  <out>/inspection/<method>_B<budget>/
    top_states_by_usage.txt
    top_states_by_positive_utility.txt
    top_states_by_net_utility.txt
    top_states_by_hurt.txt
    top_states_by_purity.txt
    top_states_by_entropy.txt
"""

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import numpy as np
import torch
import pandas as pd

EPS = 1e-10


def softmax_np(x: np.ndarray, tau: float) -> np.ndarray:
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
                return f"#{tid}"
        return decode
    except Exception:
        return lambda tid: f"#{int(tid)}"


def format_state(row: dict, states: dict, decode) -> str:
    sid      = int(row["state_id"])
    tok_ids  = states["top_k_token_ids"][sid]
    tok_cnts = states["top_k_token_counts"][sid]
    total    = float(states["total_counts"][sid])

    k_tok = int((tok_cnts > 0).sum().item())
    top_n = min(20, k_tok)

    lines = [
        f"state_id:         {sid}",
        f"method:           {row.get('method', '?')}",
        f"budget:           {row.get('budget', '?')}",
        f"assigned_count:   {int(row.get('assigned_count', 0)):,}",
        f"total_count:      {float(row.get('total_count', 0)):,.0f}",
        f"state_entropy:    {float(row.get('state_entropy', 0)):.4f}",
        f"state_purity:     {float(row.get('state_purity', 0)):.4f}",
        f"num_unique_tokens:{int(row.get('num_unique_tokens', 0))}",
        f"top1_token:       '{decode(int(row.get('top1_token_id', 0)))}'"
        f"  prob={float(row.get('top1_token_prob', 0)):.4f}",
        f"mean_nll:         {float(row.get('mean_nll', 0)):.4f}",
        "",
        f"Q-read usage:",
        f"  times_selected: {int(row.get('times_selected', 0))}",
        f"  times_top1:     {int(row.get('times_top1', 0))}",
        f"  times_top4:     {int(row.get('times_top4', 0))}",
        f"  mean_sim_when_sel: {float(row.get('mean_sim_when_sel', float('nan'))):.4f}",
        f"  net_util_vs_gpt:   {float(row.get('net_util_vs_gpt', 0)):+.4f}",
        f"  pos_util_vs_gpt:   {float(row.get('pos_util_vs_gpt', 0)):+.4f}",
        f"  neg_util_vs_gpt:   {float(row.get('neg_util_vs_gpt', 0)):+.4f}",
        f"  mean_util_per_sel: {float(row.get('mean_util_per_sel', float('nan'))):+.4f}",
        "",
        f"Top {top_n} next-token distribution:",
    ]

    for j in range(top_n):
        tid  = int(tok_ids[j])
        cnt  = float(tok_cnts[j])
        prob = cnt / (total + EPS)
        lines.append(f"  {j+1:2d}. '{decode(tid)}'  count={int(cnt):,}  prob={prob:.4f}")

    lines += ["", "=" * 60, ""]
    return "\n".join(lines)


def format_query_example(i: int, val_data: dict, val_ids: np.ndarray,
                          val_sims: np.ndarray, chosen: np.ndarray,
                          q_nlls: np.ndarray, fixed_nlls: np.ndarray,
                          nll_gpt: np.ndarray, actions: list, decode) -> str:
    ai    = int(chosen[i])
    act   = actions[ai]
    k     = act["k_states"]
    sid0  = int(val_ids[i, 0])

    return "\n".join([
        f"  val_idx={i}",
        f"  y_true_token='{decode(int(val_data['y'][i]))}'  (id={int(val_data['y'][i])})",
        f"  gpt_nll={float(nll_gpt[i]):.4f}",
        f"  q_nll={float(q_nlls[i]):.4f}",
        f"  fixed_nll={float(fixed_nlls[i]):.4f}",
        f"  improvement_vs_gpt={float(nll_gpt[i]) - float(q_nlls[i]):+.4f}",
        f"  improvement_vs_fixed={float(fixed_nlls[i]) - float(q_nlls[i]):+.4f}",
        f"  chosen_action={act['name']}  (k={k}  tau={act['tau']}  "
        f"alpha={act['alpha']}  beta={act['beta']})",
        f"  nearest_state_sim={float(val_sims[i, 0]):.4f}",
        "  ---",
    ])


def write_top_states_file(
    path: Path, title: str, top_rows: list[dict],
    states: dict, decode,
    val_data: dict | None = None,
    val_ids: np.ndarray | None = None,
    val_sims: np.ndarray | None = None,
    chosen: np.ndarray | None = None,
    q_nlls: np.ndarray | None = None,
    fixed_nlls: np.ndarray | None = None,
    nll_gpt: np.ndarray | None = None,
    actions: list | None = None,
    n_query_examples: int = 8,
):
    has_q = (val_ids is not None and chosen is not None)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n")
        f.write(f"# {path.name}  n_states={len(top_rows)}\n\n")

        for row in top_rows:
            f.write(format_state(row, states, decode))

            if has_q:
                sid    = int(row["state_id"])
                # find val queries that selected this state (top1)
                is_top1 = (val_ids[:, 0] == sid) & (chosen != 0)
                q_idxs  = np.where(is_top1)[0]
                if len(q_idxs) > 0:
                    f.write(f"\nValidation queries routed to state {sid} (top1):\n")
                    sample = q_idxs[:n_query_examples]
                    for qi in sample:
                        f.write(format_query_example(
                            qi, val_data, val_ids, val_sims, chosen,
                            q_nlls, fixed_nlls, nll_gpt, actions, decode
                        ) + "\n")
                f.write("\n")


def inspect_method_budget(
    method: str, budget: int,
    src: Path, states_dir: Path, q_read_dir: Path,
    stats_dir: Path, out_dir: Path,
    n_top: int = 20,
    n_examples: int = 8,
):
    tag       = f"{method}_B{budget}"
    csv_path  = stats_dir / f"{tag}_state_stats.csv"
    state_path = states_dir / f"{tag}.pt"

    if not csv_path.exists():
        print(f"  [skip] stats missing: {csv_path}  (run analyze_state_behavior first)")
        return
    if not state_path.exists():
        print(f"  [skip] state file missing: {state_path}")
        return

    df = pd.read_csv(csv_path)
    states = torch.load(state_path, weights_only=False)
    decode = try_tokenizer()

    # Load Q outputs
    mb_dir    = q_read_dir / tag
    nbr_path  = q_read_dir / "state_neighbors" / f"{tag}_val_top32.pt"
    pred_path = mb_dir / "val_predictions.pt"
    model_path = mb_dir / "q_model.pt"

    val_data = val_ids = val_sims = chosen = q_nlls = fixed_nlls = nll_gpt = actions = None
    has_q = nbr_path.exists() and pred_path.exists() and model_path.exists()
    if has_q:
        nbrs     = torch.load(nbr_path,  weights_only=False)
        preds    = torch.load(pred_path, weights_only=False)
        ckpt     = torch.load(model_path, weights_only=False)
        val_data = torch.load(src / "states" / "val.pt", weights_only=False)
        val_ids  = nbrs["ids"].numpy()
        val_sims = nbrs["sims"].numpy()
        chosen   = preds["chosen"].numpy()
        q_nlls   = preds["q_nlls"].numpy()
        fixed_nlls = preds["fixed_nlls"].numpy()
        nll_gpt  = val_data["nll_gpt"].numpy()
        actions  = ckpt["actions"]

    insp_dir = out_dir / tag
    insp_dir.mkdir(parents=True, exist_ok=True)

    criteria = [
        ("top_states_by_usage.txt",
         "Top States by Times Selected",
         df.sort_values("times_selected", ascending=False).head(n_top)),
        ("top_states_by_positive_utility.txt",
         "Top States by Positive Q-Utility vs GPT",
         df.sort_values("pos_util_vs_gpt", ascending=False).head(n_top)),
        ("top_states_by_net_utility.txt",
         "Top States by Net Q-Utility vs GPT",
         df.sort_values("net_util_vs_gpt", ascending=False).head(n_top)),
        ("top_states_by_hurt.txt",
         "Most Harmful States (by Negative Utility)",
         df.sort_values("neg_util_vs_gpt").head(n_top)),
        ("top_states_by_purity.txt",
         "Top States by Purity",
         df[df["times_selected"] > 0].sort_values("state_purity", ascending=False).head(n_top)),
        ("top_states_by_entropy.txt",
         "Top States by Entropy (selected states only)",
         df[df["times_selected"] > 0].sort_values("state_entropy", ascending=False).head(n_top)),
    ]

    for fname, title, top_df in criteria:
        rows = top_df.to_dict("records")
        write_top_states_file(
            insp_dir / fname,
            f"{title} | {tag}",
            rows, states, decode,
            val_data, val_ids, val_sims, chosen,
            q_nlls, fixed_nlls, nll_gpt, actions,
            n_query_examples=n_examples,
        )

    print(f"  [{tag}]  inspection files -> {insp_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",     required=True)
    parser.add_argument("--states_dir", required=True)
    parser.add_argument("--q_read_dir", required=True)
    parser.add_argument("--stats_dir",  required=True,
                        help="outputs_mvp2c_state_analysis/state_stats/")
    parser.add_argument("--output",     required=True)
    parser.add_argument("--methods",  nargs="+",
                        default=["minibatch_kmeans", "utility_weighted", "query_kmeans"])
    parser.add_argument("--budgets",  nargs="+", type=int,
                        default=[10000, 25000, 50000])
    parser.add_argument("--n_top",      type=int, default=20)
    parser.add_argument("--n_examples", type=int, default=8)
    args = parser.parse_args()

    src        = Path(args.source)
    states_dir = Path(args.states_dir)
    q_read_dir = Path(args.q_read_dir)
    stats_dir  = Path(args.stats_dir)
    out        = Path(args.output) / "inspection"

    print(f"\ninspect_top_states")
    for method in args.methods:
        for budget in args.budgets:
            inspect_method_budget(
                method, budget, src, states_dir, q_read_dir,
                stats_dir, out, args.n_top, args.n_examples,
            )


if __name__ == "__main__":
    main()

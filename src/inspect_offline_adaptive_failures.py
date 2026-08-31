"""
inspect_offline_adaptive_failures.py  --  MVP 3a qualitative inspection

Creates readable failure inspection files comparing offline vs adaptive
state reads on specific validation examples.

Outputs per comparison pair:
  inspection/<pair>/offline_helps_adaptive_fails.txt
  inspection/<pair>/adaptive_overmerged_cases.txt
  inspection/<pair>/adaptive_fragmented_cases.txt
  inspection/<pair>/similarity_high_prediction_wrong.txt
  inspection/<pair>/true_token_missing_from_adaptive.txt
  inspection/<pair>/offline_state_better_distribution.txt
"""

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-10

# Pairs to inspect: (label, offline_tag, adaptive_tag, budget)
PAIR_SPECS = [
    ("minibatch_B10k_vs_budgetfilling_B10k",
     "minibatch_kmeans", "budget_filling", 10000),
    ("utility_B10k_vs_tokenconflict_B10k",
     "utility_weighted", "token_conflict", 10000),
    ("minibatch_B25k_vs_budgetfilling_B25k",
     "minibatch_kmeans", "budget_filling", 25000),
    ("utility_B25k_vs_tokenconflict_B25k",
     "utility_weighted", "token_conflict", 25000),
]


def try_tokenizer():
    try:
        from transformers import GPT2Tokenizer
        tok = GPT2Tokenizer.from_pretrained("gpt2")
        def decode(tid):
            try:
                return tok.decode([int(tid)])
            except Exception:
                return f"#{int(tid)}"
        return decode
    except Exception:
        return lambda tid: f"#{int(tid)}"


def find_state_file(states_dir: Path, prefix: str, budget: int):
    exact = states_dir / f"{prefix}_B{budget}.pt"
    if exact.exists():
        return exact, f"{prefix}_B{budget}"
    for p in sorted(states_dir.glob(f"{prefix}*_B{budget}.pt")):
        return p, p.stem
    return None, None


def find_q_dir(q_base: Path, tag: str):
    p = q_base / tag
    if p.exists() and (p / "val_predictions.pt").exists():
        return p
    return None


def find_nbr_file(nbr_dir: Path, tag: str):
    p = nbr_dir / f"{tag}_val_top32.pt"
    return p if p.exists() else None


def load_one(states_dir, q_base, prefix, budget, nbr_in_qbase=False):
    state_path, tag = find_state_file(states_dir, prefix, budget)
    if state_path is None:
        return None, None
    # offline: nbr is inside q_base (e.g. mvp2c_q_state_read/state_neighbors)
    # adaptive: nbr is at q_base.parent level (e.g. mvp3a_fast/state_neighbors)
    if nbr_in_qbase:
        nbr_dir = q_base / "state_neighbors"
    else:
        nbr_dir = q_base.parent / "state_neighbors"
    nbr_path = find_nbr_file(nbr_dir, tag)
    q_dir    = find_q_dir(q_base, tag)
    if nbr_path is None or q_dir is None:
        return None, None

    states  = torch.load(state_path, weights_only=False)
    nbrs    = torch.load(nbr_path,   weights_only=False)
    preds   = torch.load(q_dir / "val_predictions.pt", weights_only=False)

    actions = []
    q_model_path = q_dir / "q_model.pt"
    if q_model_path.exists():
        ckpt = torch.load(q_model_path, weights_only=False)
        actions = ckpt.get("actions", [])

    return {
        "tag":    tag,
        "states": states,
        "ids":    nbrs["ids"],    # [N_val, 32]
        "sims":   nbrs["sims"],   # [N_val, 32]
        "q_nlls":     preds["q_nlls"],
        "fixed_nlls": preds["fixed_nlls"],
        "oracle_nlls": preds["oracle_nlls"],
        "chosen":     preds["chosen"],
        "actions":    actions,
    }, tag


def p_state_for(states, sid, y):
    """P_state(y) for state sid, token y."""
    tok_ids  = states["top_k_token_ids"][sid]   # [TOP_K]
    tok_cnts = states["top_k_token_counts"][sid] # [TOP_K]
    total    = float(states["total_counts"][sid])
    if total == 0:
        return 0.0
    match = (tok_ids.long() == int(y))
    raw   = float(tok_cnts[match].sum())
    return raw / total


def top_tokens_str(states, sid, decode, n=8):
    tok_ids  = states["top_k_token_ids"][sid]   # [TOP_K]
    tok_cnts = states["top_k_token_counts"][sid] # [TOP_K]
    total    = float(states["total_counts"][sid])
    items = []
    for j in range(min(n, len(tok_ids))):
        cnt = float(tok_cnts[j])
        if cnt == 0:
            break
        tid  = int(tok_ids[j])
        prob = cnt / (total + EPS)
        items.append(f"'{decode(tid)}'({prob:.3f})")
    return " ".join(items)


def format_example(
    i: int,
    val_data: dict,
    off: dict,
    ada: dict,
    decode,
    off_label: str = "offline",
    ada_label: str = "adaptive",
) -> str:
    nll_gpt = float(val_data["nll_gpt"][i])
    y       = int(val_data["y"][i])
    p_gpt   = float(val_data["p_gpt_true"][i])
    gpt_entropy = float(val_data.get("gpt_entropy", torch.zeros_like(val_data["y"]))[i])

    off_sid  = int(off["ids"][i, 0])
    ada_sid  = int(ada["ids"][i, 0])
    off_sim  = float(off["sims"][i, 0])
    ada_sim  = float(ada["sims"][i, 0])

    off_q = float(off["q_nlls"][i])
    ada_q = float(ada["q_nlls"][i])

    off_pst = p_state_for(off["states"], off_sid, y)
    ada_pst = p_state_for(ada["states"], ada_sid, y)

    off_ent = float(off["states"]["state_entropy"][off_sid])
    off_pur = float(off["states"]["state_purity"][off_sid])
    off_cnt = float(off["states"]["total_counts"][off_sid])
    ada_ent = float(ada["states"]["state_entropy"][ada_sid])
    ada_pur = float(ada["states"]["state_purity"][ada_sid])
    ada_cnt = float(ada["states"]["total_counts"][ada_sid])

    off_act = off["actions"][int(off["chosen"][i])]["name"] if off["actions"] else "?"
    ada_act = ada["actions"][int(ada["chosen"][i])]["name"] if ada["actions"] else "?"

    return "\n".join([
        f"  val_idx={i}",
        f"  true_token='{decode(y)}'  (id={y})",
        f"  gpt_nll={nll_gpt:.4f}  p_gpt_true={p_gpt:.4f}  gpt_entropy={gpt_entropy:.3f}",
        f"",
        f"  [{off_label}] tag={off['tag']}",
        f"    nearest_state={off_sid}  sim={off_sim:.4f}",
        f"    state_entropy={off_ent:.3f}  purity={off_pur:.3f}  count={off_cnt:.0f}",
        f"    p_state_true={off_pst:.5f}",
        f"    top_tokens: {top_tokens_str(off['states'], off_sid, decode)}",
        f"    chosen_action={off_act}  q_nll={off_q:.4f}  d_vs_gpt={nll_gpt-off_q:+.4f}",
        f"",
        f"  [{ada_label}] tag={ada['tag']}",
        f"    nearest_state={ada_sid}  sim={ada_sim:.4f}",
        f"    state_entropy={ada_ent:.3f}  purity={ada_pur:.3f}  count={ada_cnt:.0f}",
        f"    p_state_true={ada_pst:.5f}",
        f"    top_tokens: {top_tokens_str(ada['states'], ada_sid, decode)}",
        f"    chosen_action={ada_act}  q_nll={ada_q:.4f}  d_vs_gpt={nll_gpt-ada_q:+.4f}",
        f"  ---",
    ])


def write_inspection_file(path: Path, title: str, idxs: list,
                           val_data: dict, off: dict, ada: dict,
                           decode, max_examples: int = 50):
    N = min(len(idxs), max_examples)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n")
        f.write(f"# n_examples={N} (of {len(idxs)})\n\n")
        for i in idxs[:N]:
            f.write(format_example(i, val_data, off, ada, decode))
            f.write("\n\n")


def inspect_pair(pair_label: str, off: dict, ada: dict,
                 val_data: dict, out_dir: Path,
                 decode, max_examples: int = 50):
    insp_dir = out_dir / pair_label
    insp_dir.mkdir(parents=True, exist_ok=True)

    nll_gpt  = val_data["nll_gpt"]
    off_q    = off["q_nlls"]
    ada_q    = ada["q_nlls"]

    N_val = len(nll_gpt)
    y     = val_data["y"]

    # 1. Offline helps a lot, adaptive fails
    off_delta = nll_gpt - off_q    # positive = offline helped
    ada_delta = nll_gpt - ada_q    # positive = adaptive helped
    mask1 = (off_delta > 0.5) & (ada_delta < 0.0)
    idxs1 = mask1.nonzero(as_tuple=True)[0].tolist()
    # Sort by how much offline helps minus adaptive hurts
    idxs1 = sorted(idxs1, key=lambda i: float(off_delta[i]) - float(ada_delta[i]), reverse=True)
    write_inspection_file(
        insp_dir / "offline_helps_adaptive_fails.txt",
        f"Offline helps (d>0.5), Adaptive fails (d<0) | {pair_label}",
        idxs1, val_data, off, ada, decode, max_examples)

    # 2. Adaptive over-merged: large state count, low sim discrimination
    ada_sims = ada["sims"][:, 0]
    ada_ids0 = ada["ids"][:, 0]
    ada_counts = ada["states"]["total_counts"][ada_ids0]
    off_sims   = off["sims"][:, 0]
    off_ids0   = off["ids"][:, 0]
    off_counts = off["states"]["total_counts"][off_ids0]

    # Over-merged: adaptive nearest state has very high count, still hurts
    mask2 = (ada_counts > ada_counts.quantile(0.9)) & (ada_delta < 0)
    idxs2 = mask2.nonzero(as_tuple=True)[0].tolist()
    idxs2 = sorted(idxs2, key=lambda i: float(ada_counts[i]), reverse=True)
    write_inspection_file(
        insp_dir / "adaptive_overmerged_cases.txt",
        f"Adaptive over-merged (top-10% count, still hurts) | {pair_label}",
        idxs2, val_data, off, ada, decode, max_examples)

    # 3. Adaptive fragmented: tiny states with count=1
    mask3 = (ada_counts <= 1) & (ada_delta < 0)
    idxs3 = mask3.nonzero(as_tuple=True)[0].tolist()
    idxs3 = sorted(idxs3, key=lambda i: float(ada_delta[i]))  # most hurt first
    write_inspection_file(
        insp_dir / "adaptive_fragmented_cases.txt",
        f"Adaptive fragmented (singleton states, hurts) | {pair_label}",
        idxs3, val_data, off, ada, decode, max_examples)

    # 4. High adaptive sim but prediction wrong (p_state_true near 0)
    tok_ids_ada  = ada["states"]["top_k_token_ids"].long()
    tok_cnts_ada = ada["states"]["top_k_token_counts"]
    total_ada    = ada["states"]["total_counts"]
    TOP_K        = tok_ids_ada.shape[1]

    y_exp = y.view(-1, 1).expand(N_val, TOP_K)
    ids1  = ada["ids"][:, 0]
    match = (tok_ids_ada[ids1] == y_exp).any(-1)  # True if true token in top-K
    raw   = (tok_cnts_ada[ids1] * (tok_ids_ada[ids1] == y_exp).float()).sum(-1)
    p_ada_true = raw / (total_ada[ids1] + EPS)

    mask4 = (ada_sims > 0.98) & (p_ada_true < 0.01) & (ada_delta < 0)
    idxs4 = mask4.nonzero(as_tuple=True)[0].tolist()
    idxs4 = sorted(idxs4, key=lambda i: float(ada_sims[i]) - float(p_ada_true[i]), reverse=True)
    write_inspection_file(
        insp_dir / "similarity_high_prediction_wrong.txt",
        f"High sim (>0.98) but p_state_true<0.01 | {pair_label}",
        idxs4, val_data, off, ada, decode, max_examples)

    # 5. True token missing from adaptive nearest state
    mask5 = (~match.bool()) & (ada_delta < -0.1)
    idxs5 = mask5.nonzero(as_tuple=True)[0].tolist()
    idxs5 = sorted(idxs5, key=lambda i: float(ada_delta[i]))
    write_inspection_file(
        insp_dir / "true_token_missing_from_adaptive.txt",
        f"True token absent from adaptive nearest state (and hurts) | {pair_label}",
        idxs5, val_data, off, ada, decode, max_examples)

    # 6. Offline state has much better distribution than adaptive
    tok_ids_off  = off["states"]["top_k_token_ids"].long()
    tok_cnts_off = off["states"]["top_k_token_counts"]
    total_off    = off["states"]["total_counts"]
    ids1_off     = off["ids"][:, 0]
    TOP_K_off    = tok_ids_off.shape[1]
    y_exp_off    = y.view(-1, 1).expand(N_val, TOP_K_off)
    raw_off      = (tok_cnts_off[ids1_off] * (tok_ids_off[ids1_off] == y_exp_off).float()).sum(-1)
    p_off_true   = raw_off / (total_off[ids1_off] + EPS)

    mask6 = (p_off_true > 0.05) & (p_ada_true < 0.005) & (off_delta > ada_delta + 0.5)
    idxs6 = mask6.nonzero(as_tuple=True)[0].tolist()
    idxs6 = sorted(idxs6, key=lambda i: float(p_off_true[i]) - float(p_ada_true[i]), reverse=True)
    write_inspection_file(
        insp_dir / "offline_state_better_distribution.txt",
        f"Offline p_state_true>0.05, adaptive<0.005 | {pair_label}",
        idxs6, val_data, off, ada, decode, max_examples)

    # Print summary
    print(f"  [{pair_label}]")
    print(f"    offline_helps_adaptive_fails: {len(idxs1)}")
    print(f"    adaptive_fragmented:          {len(idxs3)}")
    print(f"    sim_high_pred_wrong:          {len(idxs4)}")
    print(f"    true_token_missing:           {len(idxs5)}")
    print(f"    offline_dist_better:          {len(idxs6)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",               required=True)
    parser.add_argument("--offline_states_dir",   required=True)
    parser.add_argument("--offline_q_dir",        required=True)
    parser.add_argument("--adaptive_states_dir",  required=True)
    parser.add_argument("--adaptive_q_dir",       required=True)
    parser.add_argument("--output",               required=True)
    parser.add_argument("--max_examples",         type=int, default=50)
    parser.add_argument("--budgets",   nargs="+", type=int, default=[10000, 25000])
    args = parser.parse_args()

    src     = Path(args.source)
    out_dir = Path(args.output) / "inspection"
    out_dir.mkdir(parents=True, exist_ok=True)

    val_data = torch.load(src / "states" / "val.pt", weights_only=False)
    decode   = try_tokenizer()
    print(f"inspect_offline_adaptive_failures  N_val={len(val_data['y']):,}")

    off_sdir  = Path(args.offline_states_dir)
    off_qdir  = Path(args.offline_q_dir)
    ada_sdir  = Path(args.adaptive_states_dir)
    ada_qdir  = Path(args.adaptive_q_dir)

    for pair_label, off_meth, ada_meth, budget in PAIR_SPECS:
        if budget not in args.budgets:
            continue
        print(f"\n--- {pair_label} ---")

        off, off_tag = load_one(off_sdir, off_qdir, off_meth, budget, nbr_in_qbase=True)
        ada, ada_tag = load_one(ada_sdir, ada_qdir, ada_meth, budget, nbr_in_qbase=False)

        if off is None:
            print(f"  [skip] offline {off_meth} B={budget} not found")
            continue
        if ada is None:
            print(f"  [skip] adaptive {ada_meth} B={budget} not found")
            continue

        print(f"  offline={off_tag}  adaptive={ada_tag}")
        inspect_pair(pair_label, off, ada, val_data, out_dir, decode, args.max_examples)

    print(f"\nInspection files: {out_dir}")


if __name__ == "__main__":
    main()

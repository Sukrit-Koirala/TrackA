"""
audit_splits.py

Split integrity audit for the Branch A Gate-Chain MVP.

Checks that the three splits (datastore / controller_train / val) are truly
disjoint — at the story level (if story_split was used), at the context-snippet
level, and at the token-window hash level.

Also audits whether any retrieved neighbor comes from the same story as its
query (requires story_split metadata in neighbor files).

Outputs (in cfg["audit_dir"]):
  split_overlap_report.json
  split_overlap_report.txt
  neighbor_source_report.json
  neighbor_source_report.txt
  top_similarity_examples.txt
  retrieval_help_high_sim.txt
  retrieval_hurt_high_sim.txt
  AUDIT_REPORT.md        (written by compare_split_results.py after full run)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json
import numpy as np
import torch

from utils import load_config, ensure_dirs, set_seed, build_action_grid
from gate_chain import evaluate_action_for_queries


# ── helpers ────────────────────────────────────────────────────────────────────

def story_ids_from_meta(meta: list[dict]) -> set[int]:
    return {m["story_id"] for m in meta if "story_id" in m}


def context_snippets(meta: list[dict]) -> set[str]:
    return {m.get("context_snippet", "")[:100] for m in meta}


def window_hashes(meta: list[dict]) -> set[str]:
    return {m.get("context_window_hash", "") for m in meta}


def overlap_report(name_a: str, set_a: set, name_b: str, set_b: set) -> dict:
    inter = set_a & set_b
    return {
        "pair":       f"{name_a} ∩ {name_b}",
        "size_a":     len(set_a),
        "size_b":     len(set_b),
        "overlap":    len(inter),
        "overlap_pct_a": 100 * len(inter) / max(len(set_a), 1),
        "overlap_pct_b": 100 * len(inter) / max(len(set_b), 1),
    }


# ── main audit logic ───────────────────────────────────────────────────────────

def audit_states(cfg: dict) -> dict:
    states_dir = Path(cfg["states_dir"])
    splits = {}
    for name in ("datastore", "controller_train", "val"):
        path = states_dir / f"{name}.pt"
        if path.exists():
            splits[name] = torch.load(path, weights_only=False)
        else:
            print(f"  Warning: {path} not found — skipping")

    if len(splits) < 2:
        return {"error": "fewer than 2 splits found"}

    results: dict = {"story_overlap": [], "snippet_overlap": [], "window_hash_overlap": []}

    names = list(splits.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            na, nb = names[i], names[j]
            ma, mb = splits[na]["metadata"], splits[nb]["metadata"]

            # Story-level
            sa, sb = story_ids_from_meta(ma), story_ids_from_meta(mb)
            if sa or sb:
                results["story_overlap"].append(overlap_report(na, sa, nb, sb))

            # Context snippet (first 100 chars)
            ca, cb = context_snippets(ma), context_snippets(mb)
            results["snippet_overlap"].append(overlap_report(na, ca, nb, cb))

            # Token-window hash
            ha, hb = window_hashes(ma), window_hashes(mb)
            if any(h for h in ha) and any(h for h in hb):
                results["window_hash_overlap"].append(overlap_report(na, ha, nb, hb))

    return results


def audit_neighbors(cfg: dict) -> dict:
    nbrs_dir   = Path(cfg["neighbors_dir"])
    k_max      = cfg["max_k"]
    results    = {}

    for split_name in ("controller_train", "val"):
        path = nbrs_dir / f"{split_name}_top{k_max}.pt"
        if not path.exists():
            continue
        nbrs = torch.load(path, weights_only=False)

        if "query_story_ids" not in nbrs or "neighbor_story_ids" not in nbrs:
            results[split_name] = {
                "note": "story_id metadata not present — run with story_split: true"
            }
            continue

        q_ids  = nbrs["query_story_ids"]        # [N]
        nb_ids = nbrs["neighbor_story_ids"]      # [N, k]
        N, K   = nb_ids.shape

        # For each query, does any of its top-k neighbors share the same story_id?
        q_exp   = q_ids.unsqueeze(1).expand_as(nb_ids)   # [N, k]
        same    = (nb_ids == q_exp)                       # [N, k]
        any_same = same.any(dim=1)                        # [N]
        top1_same = same[:, 0]                            # [N] — nearest neighbor same story

        results[split_name] = {
            "total_queries":              N,
            "queries_with_any_same_story_neighbor":  int(any_same.sum()),
            "queries_with_top1_same_story":          int(top1_same.sum()),
            "fraction_any_same_story":    float(any_same.float().mean()),
            "mean_same_story_neighbors_per_query": float(same.float().sum(1).mean()),
        }
        print(f"    {split_name}: {any_same.sum()}/{N} queries have a same-story neighbor")

    return results


def top_similarity_examples(cfg: dict, n: int = 50) -> list[dict]:
    """
    Return the top-n val examples with highest nearest-neighbor similarity.
    """
    nbrs_dir   = Path(cfg["neighbors_dir"])
    states_dir = Path(cfg["states_dir"])
    k_max      = cfg["max_k"]

    val_path  = states_dir  / "val.pt"
    nbrs_path = nbrs_dir    / f"val_top{k_max}.pt"
    ds_path   = states_dir  / "datastore.pt"

    if not all(p.exists() for p in (val_path, nbrs_path, ds_path)):
        return []

    val_data = torch.load(val_path,  weights_only=False)
    nbrs     = torch.load(nbrs_path, weights_only=False)
    ds_data  = torch.load(ds_path,   weights_only=False)

    sims   = nbrs["neighbor_sims"][:, 0].float()   # nearest neighbor sim [N]
    top_idx = sims.argsort(descending=True)[:n].tolist()

    val_meta  = val_data["metadata"]
    ds_meta   = ds_data["metadata"]
    ds_y_ids  = ds_data["y"]
    nbr_idx   = nbrs["neighbor_indices"]
    nbr_y     = nbrs["neighbor_y"]
    q_story   = nbrs.get("query_story_ids")
    nbr_story = nbrs.get("neighbor_story_ids")

    examples = []
    for i in top_idx:
        ni     = int(nbr_idx[i, 0])
        sim    = float(sims[i])
        qm     = val_meta[i]
        nm     = ds_meta[ni] if ni < len(ds_meta) else {}
        examples.append({
            "query_idx":           i,
            "query_story_id":      q_story[i].item() if q_story is not None else None,
            "neighbor_story_id":   nbr_story[i, 0].item() if nbr_story is not None else None,
            "nearest_sim":         sim,
            "query_ctx":           qm.get("context_snippet", "")[-120:],
            "neighbor_ctx":        nm.get("context_snippet", "")[-120:],
            "query_y_str":         qm.get("y_str", "?"),
            "neighbor_y_str":      nm.get("y_str", "?"),
            "next_tokens_match":   qm.get("y_str") == nm.get("y_str"),
        })
    return examples


# ── format helpers ─────────────────────────────────────────────────────────────

def fmt_overlap_block(records: list[dict]) -> str:
    if not records:
        return "  (no story_id metadata — old random split)\n"
    lines = []
    for r in records:
        inter_str = f"{r['overlap']} ({r['overlap_pct_a']:.2f}% of A, {r['overlap_pct_b']:.2f}% of B)"
        lines.append(f"  {r['pair']}: overlap = {inter_str}")
    return "\n".join(lines)


def fmt_neighbor_block(nbr_res: dict) -> str:
    lines = []
    for split, r in nbr_res.items():
        if "note" in r:
            lines.append(f"  {split}: {r['note']}")
        else:
            lines.append(
                f"  {split}: "
                f"{r['queries_with_any_same_story_neighbor']}/{r['total_queries']} queries "
                f"have a same-story neighbor  "
                f"({r['fraction_any_same_story']*100:.2f}%)")
    return "\n".join(lines)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Split integrity audit")
    parser.add_argument("--config", default="configs/clean_storysplit.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    set_seed(cfg.get("seed", 42))

    audit_dir = Path(cfg.get("audit_dir", "outputs_clean_storysplit/audit"))
    audit_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1] Auditing split overlaps …")
    state_res = audit_states(cfg)

    print("\n[2] Auditing neighbor sources …")
    nbr_res   = audit_neighbors(cfg)

    # ── text report ────────────────────────────────────────────────────────────
    lines = [
        "# Split Overlap Audit Report",
        "",
        "## Story-level overlap (should be 0 for story_split: true)",
        fmt_overlap_block(state_res.get("story_overlap", [])),
        "",
        "## Context-snippet overlap (first 100 chars)",
        fmt_overlap_block(state_res.get("snippet_overlap", [])),
        "",
        "## Token-window hash overlap",
        fmt_overlap_block(state_res.get("window_hash_overlap", [])),
        "",
        "## Neighbor source audit",
        fmt_neighbor_block(nbr_res),
    ]
    txt = "\n".join(lines)
    print("\n" + txt)

    (audit_dir / "split_overlap_report.txt").write_text(txt, encoding="utf-8")
    with open(audit_dir / "split_overlap_report.json", "w") as f:
        json.dump({"state_overlaps": state_res, "neighbor_sources": nbr_res}, f, indent=2)
    print(f"\nSaved: {audit_dir / 'split_overlap_report.txt'}")

    # ── neighbor source report ─────────────────────────────────────────────────
    nbr_txt = "# Neighbor Source Report\n\n" + fmt_neighbor_block(nbr_res)
    (audit_dir / "neighbor_source_report.txt").write_text(nbr_txt, encoding="utf-8")
    with open(audit_dir / "neighbor_source_report.json", "w") as f:
        json.dump(nbr_res, f, indent=2)
    print(f"Saved: {audit_dir / 'neighbor_source_report.txt'}")

    # ── top-similarity examples ────────────────────────────────────────────────
    print("\n[3] Collecting top-similarity val examples …")
    top_examples = top_similarity_examples(cfg, n=50)
    if top_examples:
        ex_lines = [f"# Top {len(top_examples)} val examples by nearest-neighbor similarity\n"]
        for e in top_examples:
            ex_lines.append(
                f"=== Query {e['query_idx']} ===\n"
                f"  nearest_sim:    {e['nearest_sim']:.6f}\n"
                f"  query_story:    {e['query_story_id']}\n"
                f"  neighbor_story: {e['neighbor_story_id']}\n"
                f"  same_story:     {e['query_story_id'] == e['neighbor_story_id'] if e['query_story_id'] is not None else 'unknown'}\n"
                f"  next_tok_match: {e['next_tokens_match']}\n"
                f"  query_ctx:    …{e['query_ctx']}\n"
                f"  neighbor_ctx: …{e['neighbor_ctx']}\n"
                f"  query_y:      '{e['query_y_str']}'  neighbor_y: '{e['neighbor_y_str']}'\n"
            )
        (audit_dir / "top_similarity_examples.txt").write_text(
            "\n".join(ex_lines), encoding="utf-8")
        print(f"Saved: {audit_dir / 'top_similarity_examples.txt'}")

    # ── retrieval help/hurt with high similarity ───────────────────────────────
    print("\n[4] Retrieval help/hurt at high similarity …")
    try:
        states_dir = Path(cfg["states_dir"])
        nbrs_dir   = Path(cfg["neighbors_dir"])
        k_max      = cfg["max_k"]
        val_data   = torch.load(states_dir / "val.pt",             weights_only=False)
        val_nbrs   = torch.load(nbrs_dir   / f"val_top{k_max}.pt", weights_only=False)

        actions     = build_action_grid(cfg)
        # Use the first retrieval action as reference for "retrieval NLL"
        ret_action  = next(a for a in actions if a["k"] > 0)
        m_ret = evaluate_action_for_queries(
            val_data, val_nbrs, ret_action,
            max_k=k_max, lambda_cost=0.0, eps=cfg.get("eps", 1e-12),
        )
        nll_gpt = val_data["nll_gpt"].float().numpy()
        nll_ret = m_ret["per_example_nll"].numpy()
        helps   = nll_gpt - nll_ret        # positive = retrieval helped
        sim1    = val_nbrs["neighbor_sims"][:, 0].float().numpy()

        val_meta = val_data["metadata"]
        ds_data  = torch.load(states_dir / "datastore.pt", weights_only=False)
        ds_meta  = ds_data["metadata"]
        nbr_idx  = val_nbrs["neighbor_indices"]

        def fmt_pair(i: int, label: str) -> str:
            ni = int(nbr_idx[i, 0])
            qm = val_meta[i]
            nm = ds_meta[ni] if ni < len(ds_meta) else {}
            return (
                f"=== {label} | idx={i}  sim={sim1[i]:.4f}  Δ={helps[i]:+.4f} ===\n"
                f"  query_ctx:    …{qm.get('context_snippet','')[-100:]}\n"
                f"  neighbor_ctx: …{nm.get('context_snippet','')[-100:]}\n"
                f"  query_y: '{qm.get('y_str','?')}'  neighbor_y: '{nm.get('y_str','?')}'\n"
            )

        # Top 25 highest-sim where retrieval helped / hurt
        order_by_sim = np.argsort(sim1)[::-1]

        help_lines = ["# High-similarity examples where retrieval helped\n"]
        hurt_lines = ["# High-similarity examples where retrieval hurt\n"]
        for i in order_by_sim[:500]:
            if helps[i] > 0 and len(help_lines) <= 26:
                help_lines.append(fmt_pair(i, "HELP"))
            if helps[i] < 0 and len(hurt_lines) <= 26:
                hurt_lines.append(fmt_pair(i, "HURT"))

        (audit_dir / "retrieval_help_high_sim.txt").write_text(
            "\n".join(help_lines), encoding="utf-8")
        (audit_dir / "retrieval_hurt_high_sim.txt").write_text(
            "\n".join(hurt_lines), encoding="utf-8")
        print(f"Saved: {audit_dir / 'retrieval_help_high_sim.txt'}")
        print(f"Saved: {audit_dir / 'retrieval_hurt_high_sim.txt'}")
    except Exception as e:
        print(f"  Skipped retrieval help/hurt audit: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()

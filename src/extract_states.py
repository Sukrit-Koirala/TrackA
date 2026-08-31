"""
extract_states.py

Load TinyStories, run frozen GPT-2, extract final-layer hidden states and
next-token statistics for every token position.

Two splitting modes:
  story_split: false (default)
    Positions are randomly assigned across splits.  Chunks from the same
    story can appear in datastore, controller_train, and val simultaneously.
    This is the original behaviour.

  story_split: true
    Each TinyStories story is assigned to exactly one split before any
    token extraction.  No story ever contributes positions to more than
    one split.  This is the clean setting for auditing leakage.

Saved files (cfg["states_dir"]/):
  datastore.pt  controller_train.pt  val.pt

Each file is a dict:
  h                     FloatTensor [N, 768]  (fp16 on disk)
  y                     LongTensor  [N]       true next-token id
  p_gpt_true            FloatTensor [N]
  nll_gpt               FloatTensor [N]
  gpt_entropy           FloatTensor [N]
  gpt_top1_id           LongTensor  [N]
  gpt_top1_prob         FloatTensor [N]
  gpt_top2_prob         FloatTensor [N]
  metadata              list[dict]
    chunk_idx           int
    pos_in_chunk        int
    context_snippet     str   (last ≤60 tokens decoded)
    y_str               str
    context_window_hash str   (hash of last ≤32 input token IDs — for duplicate audit)
    story_id            int   (only when story_split: true)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import torch
import numpy as np
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset

from utils import load_config, ensure_dirs, get_device, set_seed


# ── model loading ──────────────────────────────────────────────────────────────

def load_frozen_gpt2(model_name: str, device: torch.device):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    model.requires_grad_(False)
    model = model.to(device)
    return tokenizer, model


# ── token stream collection (original mode) ────────────────────────────────────

def collect_chunks(cfg: dict, tokenizer) -> list[list[int]]:
    """
    Stream dataset, concatenate tokens, cut into non-overlapping
    chunks of length seq_len.  Returns plain list[list[int]] chunks.
    Used when story_split: false.
    Supports any HuggingFace dataset via dataset/dataset_config/text_field/split_train cfg keys.
    """
    seq_len      = cfg["seq_len"]
    total_pos    = (cfg["datastore_positions"]
                    + cfg["controller_train_positions"]
                    + cfg["val_positions"])
    chunks_needed = int(total_pos / (seq_len - 1) * 1.2) + 50

    dataset_id  = cfg["dataset"]
    dataset_cfg = cfg.get("dataset_config") or cfg.get("hf_config")
    train_split = cfg.get("split_train", "train")
    text_field  = cfg.get("text_field", "text")

    dataset = load_dataset(dataset_id, dataset_cfg, split=train_split,
                           streaming=True, trust_remote_code=True)

    token_buffer: list[int] = []
    chunks: list[list[int]] = []

    for story in tqdm(dataset, desc="Tokenising"):
        text = story.get(text_field, "") if isinstance(story, dict) else story[text_field]
        if not text or not text.strip():
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids.append(tokenizer.eos_token_id)
        token_buffer.extend(ids)

        while len(token_buffer) >= seq_len:
            chunks.append(token_buffer[:seq_len])
            token_buffer = token_buffer[seq_len:]

        if len(chunks) >= chunks_needed:
            break

    print(f"  Collected {len(chunks)} chunks  ({len(chunks)*(seq_len-1):,} positions)")
    return chunks


# ── story-level chunk collection ───────────────────────────────────────────────

def collect_chunks_story_split(
    cfg: dict,
    tokenizer,
    rng: np.random.Generator,
) -> tuple[list[tuple[int, list[int]]], list[tuple[int, list[int]]], list[tuple[int, list[int]]]]:
    """
    Assign each TinyStories story to exactly one split (ds / ct / val),
    then tokenise and cut into seq_len chunks within each story.

    Returns three lists of (story_id, chunk_tokens) tuples.
    No story_id appears in more than one list.
    """
    seq_len  = cfg["seq_len"]
    n_ds     = cfg["datastore_positions"]
    n_ct     = cfg["controller_train_positions"]
    n_val    = cfg["val_positions"]
    ds_frac  = cfg.get("datastore_story_frac", 0.70)
    ct_frac  = cfg.get("controller_train_story_frac", 0.15)

    # With buffer: need 3× the positions we'll actually use
    # so random subsampling within each split never runs short.
    pos_per_chunk = seq_len - 1   # 127
    needed_ds     = int(n_ds  * 3 / pos_per_chunk) + 50
    needed_ct     = int(n_ct  * 3 / pos_per_chunk) + 50
    needed_val    = int(n_val * 3 / pos_per_chunk) + 50

    ds_chunks: list[tuple[int, list[int]]]  = []
    ct_chunks: list[tuple[int, list[int]]]  = []
    val_chunks: list[tuple[int, list[int]]] = []

    dataset  = load_dataset(cfg["dataset"], split="train", streaming=True,
                            trust_remote_code=True)
    story_id = 0

    pbar = tqdm(dataset, desc="Collecting story chunks")
    for story in pbar:
        ids = tokenizer.encode(story["text"], add_special_tokens=False)
        ids.append(tokenizer.eos_token_id)

        # Only take stories that yield at least one full seq_len chunk
        if len(ids) < seq_len:
            story_id += 1
            continue

        story_chunks = []
        for start in range(0, len(ids) - seq_len + 1, seq_len):
            chunk = ids[start : start + seq_len]
            if len(chunk) == seq_len:
                story_chunks.append(chunk)

        if not story_chunks:
            story_id += 1
            continue

        # Deterministic assignment based on rng draw
        r = rng.random()
        if r < ds_frac:
            ds_chunks.extend((story_id, c) for c in story_chunks)
        elif r < ds_frac + ct_frac:
            ct_chunks.extend((story_id, c) for c in story_chunks)
        else:
            val_chunks.extend((story_id, c) for c in story_chunks)

        story_id += 1
        pbar.set_postfix(ds=len(ds_chunks), ct=len(ct_chunks), val=len(val_chunks))

        if (len(ds_chunks) >= needed_ds
                and len(ct_chunks) >= needed_ct
                and len(val_chunks) >= needed_val):
            break

    print(f"  Stories processed: {story_id}")
    print(f"  Chunks: ds={len(ds_chunks)}  ct={len(ct_chunks)}  val={len(val_chunks)}")
    return ds_chunks, ct_chunks, val_chunks


# ── batched extraction ─────────────────────────────────────────────────────────

def extract_batched(
    cfg: dict,
    model: GPT2LMHeadModel,
    tokenizer,
    chunks: list[list[int]],
    device: torch.device,
    story_ids_per_chunk: list[int] | None = None,
) -> dict:
    """
    Run GPT-2 on every chunk in batches.  Collects hidden states and
    next-token statistics for each of the (seq_len-1) positions per chunk.

    story_ids_per_chunk: if provided (story-level split mode), the story_id
    for each chunk is stored in per-position metadata.
    """
    seq_len    = cfg["seq_len"]
    batch_size = cfg.get("extract_batch_size", 16)

    all_h, all_y, all_p_gpt, all_nll_gpt = [], [], [], []
    all_entropy, all_top1_id, all_top1_prob, all_top2_prob = [], [], [], []
    all_meta: list[dict] = []

    n_chunks         = len(chunks)
    chunk_ids_global = 0

    for batch_start in tqdm(range(0, n_chunks, batch_size), desc="Extracting"):
        batch_chunks = chunks[batch_start : batch_start + batch_size]
        B = len(batch_chunks)

        input_ids = torch.tensor(batch_chunks, dtype=torch.long, device=device)

        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True)

        hidden = out.hidden_states[-1]    # [B, L, D]
        logits = out.logits               # [B, L, V]

        h_pred    = hidden[:, :-1, :]    # [B, L-1, D]
        lgts_pred = logits[:, :-1, :]    # [B, L-1, V]
        y_target  = input_ids[:, 1:]     # [B, L-1]

        probs  = torch.softmax(lgts_pred.float(), dim=-1)
        lprobs = torch.log_softmax(lgts_pred.float(), dim=-1)

        p_true   = probs.gather(-1, y_target.unsqueeze(-1)).squeeze(-1)
        nll      = -torch.log(p_true + 1e-12)
        entropy  = -(probs * lprobs).sum(-1)
        top2     = probs.topk(2, dim=-1)
        top1_id  = top2.indices[:, :, 0]
        top1_p   = top2.values[:, :, 0]
        top2_p   = top2.values[:, :, 1]

        L1 = seq_len - 1
        all_h.append(h_pred.reshape(B * L1, -1).half().cpu())
        all_y.append(y_target.reshape(B * L1).cpu())
        all_p_gpt.append(p_true.reshape(B * L1).cpu())
        all_nll_gpt.append(nll.reshape(B * L1).cpu())
        all_entropy.append(entropy.reshape(B * L1).cpu())
        all_top1_id.append(top1_id.reshape(B * L1).cpu())
        all_top1_prob.append(top1_p.reshape(B * L1).cpu())
        all_top2_prob.append(top2_p.reshape(B * L1).cpu())

        for bi in range(B):
            chunk = batch_chunks[bi]
            cid   = chunk_ids_global + bi
            sid   = story_ids_per_chunk[cid] if story_ids_per_chunk is not None else None
            for pos in range(L1):
                ctx_start = max(0, pos - 59)
                ctx_str   = tokenizer.decode(chunk[ctx_start : pos + 1],
                                             skip_special_tokens=True)
                y_str     = tokenizer.decode([chunk[pos + 1]], skip_special_tokens=True)
                # Hash of last ≤32 input token IDs for duplicate-context auditing
                tail      = chunk[max(0, pos - 31) : pos + 1]
                win_hash  = "|".join(map(str, tail))
                meta: dict = {
                    "chunk_idx":          cid,
                    "pos_in_chunk":       pos,
                    "context_snippet":    ctx_str,
                    "y_str":              y_str,
                    "context_window_hash": win_hash,
                }
                if sid is not None:
                    meta["story_id"] = sid
                all_meta.append(meta)

        chunk_ids_global += B

    return {
        "h":             torch.cat(all_h),
        "y":             torch.cat(all_y),
        "p_gpt_true":    torch.cat(all_p_gpt).float(),
        "nll_gpt":       torch.cat(all_nll_gpt).float(),
        "gpt_entropy":   torch.cat(all_entropy).float(),
        "gpt_top1_id":   torch.cat(all_top1_id),
        "gpt_top1_prob": torch.cat(all_top1_prob).float(),
        "gpt_top2_prob": torch.cat(all_top2_prob).float(),
        "metadata":      all_meta,
    }


# ── subsample within a pre-split states dict ──────────────────────────────────

def subsample_states(states: dict, n: int, rng: np.random.Generator) -> dict:
    """Randomly select n positions from states dict."""
    total = len(states["y"])
    if total <= n:
        return states
    idx = rng.choice(total, size=n, replace=False)
    idx_sorted = np.sort(idx)
    out: dict = {}
    for k, v in states.items():
        if k == "metadata":
            out[k] = [v[i] for i in idx_sorted]
        else:
            out[k] = v[torch.from_numpy(idx_sorted)]
    return out


# ── split and save (original mode) ─────────────────────────────────────────────

def split_and_save(cfg: dict, states: dict, rng: np.random.Generator):
    total_pos = len(states["y"])
    n_ds  = cfg["datastore_positions"]
    n_ct  = cfg["controller_train_positions"]
    n_val = cfg["val_positions"]
    need  = n_ds + n_ct + n_val

    if total_pos < need:
        raise RuntimeError(
            f"Collected {total_pos} positions but need {need}.")

    idx     = rng.permutation(total_pos)[:need]
    ds_idx  = idx[:n_ds]
    ct_idx  = idx[n_ds : n_ds + n_ct]
    val_idx = idx[n_ds + n_ct :]

    def subset(d: dict, mask_idx) -> dict:
        out: dict = {}
        for k, v in d.items():
            if k == "metadata":
                out[k] = [v[i] for i in mask_idx]
            else:
                out[k] = v[mask_idx]
        return out

    states_dir = Path(cfg["states_dir"]) if "states_dir" in cfg \
                 else Path(cfg["output_dir"]) / "states"
    states_dir.mkdir(parents=True, exist_ok=True)
    for name, idx_arr, fname in [
        ("datastore",        ds_idx,  "datastore.pt"),
        ("controller_train", ct_idx,  "controller_train.pt"),
        ("val",              val_idx, "val.pt"),
    ]:
        sub  = subset(states, idx_arr)
        path = states_dir / fname
        torch.save(sub, path)
        print(f"  Saved {name} ({len(idx_arr):,} positions) → {path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    set_seed(cfg.get("seed", 42))
    rng    = np.random.default_rng(cfg.get("split_seed", cfg.get("seed", 42)))
    device = get_device(cfg)
    print(f"Device: {device}")

    print("\n[1/3] Loading model …")
    tokenizer, model = load_frozen_gpt2(cfg["model_name"], device)
    print(f"  GPT-2 loaded  (hidden_dim={model.config.n_embd}, "
          f"vocab={model.config.vocab_size})")

    story_split = cfg.get("story_split", False)

    if story_split:
        print(f"\n[2/3] Story-level split  "
              f"(ds={cfg.get('datastore_story_frac',0.70):.0%}  "
              f"ct={cfg.get('controller_train_story_frac',0.15):.0%}  "
              f"val={1-cfg.get('datastore_story_frac',0.70)-cfg.get('controller_train_story_frac',0.15):.0%})")
        ds_chunks, ct_chunks, val_chunks = collect_chunks_story_split(
            cfg, tokenizer, rng)

        states_dir = Path(cfg["states_dir"]) if "states_dir" in cfg \
                     else Path(cfg["output_dir"]) / "states"
        states_dir.mkdir(parents=True, exist_ok=True)
        for split_name, split_chunks, n_target in [
            ("datastore",        ds_chunks,  cfg["datastore_positions"]),
            ("controller_train", ct_chunks,  cfg["controller_train_positions"]),
            ("val",              val_chunks, cfg["val_positions"]),
        ]:
            print(f"\n[3/3 – {split_name}] Extracting {len(split_chunks)} chunks …")
            plain_chunks   = [c for _, c in split_chunks]
            story_ids_list = [s for s, _ in split_chunks]
            states = extract_batched(cfg, model, tokenizer,
                                     plain_chunks, device,
                                     story_ids_per_chunk=story_ids_list)
            states = subsample_states(states, n_target, rng)
            path   = states_dir / f"{split_name}.pt"
            torch.save(states, path)
            print(f"  Saved {split_name} ({len(states['y']):,} positions) → {path}")
    else:
        print("\n[2/3] Collecting token chunks (original mode, no story boundary) …")
        chunks = collect_chunks(cfg, tokenizer)

        print("\n[3/3] Extracting hidden states …")
        states = extract_batched(cfg, model, tokenizer, chunks, device)
        print(f"  Total positions extracted: {len(states['y']):,}")

        print("\nSplitting and saving …")
        split_and_save(cfg, states, rng)

    print("\nDone.")


if __name__ == "__main__":
    main()

"""utils.py — shared helpers: config loading, device selection, seeding."""

from pathlib import Path
import random
import yaml
import numpy as np
import torch


def load_config(cfg_path: str) -> dict:
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def ensure_dirs(cfg: dict):
    for key in ["states_dir", "neighbors_dir", "reports_dir", "inspection_dir",
                "inspection_v2_dir", "models_dir", "audit_dir"]:
        path = cfg.get(key)
        if path:
            Path(path).mkdir(parents=True, exist_ok=True)


def get_device(cfg: dict) -> torch.device:
    req = cfg.get("device", "cuda")
    if req == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_action_grid(cfg: dict) -> list[dict]:
    """Build the fixed candidate action set: GPT-only + all (k, tau, alpha) combos."""
    actions = [{"name": "gpt_only", "k": 0, "tau": 0.1, "alpha": 1.0}]
    for k in cfg["k_values"]:
        for tau in cfg["tau_values"]:
            for alpha in cfg["alpha_values"]:
                name = f"k{k}_t{tau}_a{alpha}"
                actions.append({"name": name, "k": k, "tau": tau, "alpha": alpha})
    return actions


def ppl(nll: float) -> float:
    return float(np.exp(nll))

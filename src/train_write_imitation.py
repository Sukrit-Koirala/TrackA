"""
train_write_imitation.py  --  MVP 4a Stage 3

Train MLP classifier: obs_features → action class.

Architecture: input_dim → 128 → 128 → N_ACTIONS
Loss:         cross-entropy with class weights (sqrt inverse frequency)

Saves:
  outputs_mvp4a.../models/<teacher_name>_write_imitation_mlp.pt
  outputs_mvp4a.../models/<teacher_name>_metrics.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from utils import get_device, set_seed

# Must match build_write_imitation_dataset.py
ACTION_NAMES = ["UPDATE_STATE", "UPDATE_BUFFER", "CREATE_BUFFER", "PROMOTE_BUFFER"]
N_ACTIONS    = 4


class WriteImitationMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, n_actions: int = N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, n_actions),
        )
        self.x_mu  = None
        self.x_std = None

    def forward(self, x):
        return self.net(x)

    def predict(self, x_raw: torch.Tensor) -> torch.Tensor:
        """x_raw: unnormalised features → action class."""
        if self.x_mu is not None:
            x = (x_raw - self.x_mu.to(x_raw.device)) / (self.x_std.to(x_raw.device) + 1e-8)
        else:
            x = x_raw
        logits = self.forward(x)
        return logits.argmax(dim=-1)


def train_model(
    X_tr, y_tr, X_dev, y_dev,
    x_mu, x_std,
    device,
    n_epochs: int = 30,
    batch_size: int = 1024,
    lr: float = 1e-3,
    seed: int = 42,
):
    set_seed(seed)
    in_dim = X_tr.shape[1]
    model  = WriteImitationMLP(in_dim).to(device)
    model.x_mu  = x_mu.cpu()
    model.x_std = x_std.cpu()

    # Class weights: full inverse frequency (not sqrt) so rare classes get
    # equal total gradient to the dominant UPDATE_BUFFER class.
    class_counts = torch.bincount(y_tr, minlength=N_ACTIONS).float().clamp(min=1)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * N_ACTIONS
    class_weights = class_weights.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    tr_ds  = TensorDataset(X_tr.to(device), y_tr.to(device))
    tr_dl  = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    dev_ds = TensorDataset(X_dev.to(device), y_dev.to(device))
    dev_dl = DataLoader(dev_ds, batch_size=4096, shuffle=False)

    history = []
    best_dev_loss = float("inf")
    best_state = None
    patience = 0
    PATIENCE_LIMIT = 8

    for ep in range(1, n_epochs + 1):
        model.train()
        tr_loss = tr_n = 0
        for xb, yb in tr_dl:
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            tr_loss += float(loss) * len(xb)
            tr_n    += len(xb)
        sched.step()

        model.eval()
        dev_loss = dev_n = dev_correct = 0
        with torch.no_grad():
            for xb, yb in dev_dl:
                logits   = model(xb)
                dev_loss += float(criterion(logits, yb)) * len(xb)
                dev_n    += len(xb)
                dev_correct += int((logits.argmax(1) == yb).sum())

        tr_l  = tr_loss / max(tr_n, 1)
        dev_l = dev_loss / max(dev_n, 1)
        dev_a = dev_correct / max(dev_n, 1)
        history.append({"epoch": ep, "tr_loss": tr_l, "dev_loss": dev_l, "dev_acc": dev_a})

        if ep % 5 == 0:
            print(f"      ep {ep:3d}  tr={tr_l:.5f}  dev={dev_l:.5f}  acc={dev_a:.3f}")

        if dev_l < best_dev_loss:
            best_dev_loss = dev_l
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience      = 0
        else:
            patience += 1
            if patience >= PATIENCE_LIMIT:
                print(f"      Early stop at epoch {ep}")
                break

    model.load_state_dict(best_state)
    return model, history


def per_class_metrics(y_true, y_pred):
    result = {}
    for c, name in enumerate(ACTION_NAMES):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-8)
        result[name] = {"tp": tp, "fp": fp, "fn": fn,
                        "precision": prec, "recall": rec, "f1": f1}
    return result


def confusion_matrix(y_true, y_pred, n=N_ACTIONS):
    cm = np.zeros((n, n), int)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[t][p] += 1
    return cm.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_dir", required=True)
    parser.add_argument("--teachers",     nargs="+",
                        default=["utility_weighted_B10000", "minibatch_kmeans_B10000"])
    parser.add_argument("--output",       required=True)
    parser.add_argument("--n_epochs",     type=int,   default=30)
    parser.add_argument("--batch_size",   type=int,   default=1024)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--train_frac",   type=float, default=0.8)
    parser.add_argument("--device",       default="cuda")
    parser.add_argument("--force",        action="store_true")
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device  = get_device({"device": args.device})
    ddir    = Path(args.datasets_dir)
    out     = Path(args.output) / "models"
    out.mkdir(parents=True, exist_ok=True)

    print(f"\ntrain_write_imitation")
    print(f"  Datasets: {ddir}")
    print(f"  Output:   {out}")

    for teacher_name in args.teachers:
        model_path  = out / f"{teacher_name}_write_imitation_mlp.pt"
        metric_path = out / f"{teacher_name}_metrics.json"

        if model_path.exists() and not args.force:
            print(f"\n  [cached] {teacher_name}")
            continue

        ds_path = ddir / f"{teacher_name}_write_imitation_dataset.pt"
        if not ds_path.exists():
            print(f"\n  [SKIP] {teacher_name} — dataset not found: {ds_path}")
            continue

        print(f"\n  [{teacher_name}]")
        t0 = time.time()

        ds      = torch.load(ds_path, weights_only=False)
        X_all   = ds["X"].float()           # [N, feat_dim]
        y_all   = ds["y_action"].long()     # [N]
        N       = len(y_all)

        print(f"    Samples: {N:,}")
        dist = {ACTION_NAMES[c]: int((y_all == c).sum()) for c in range(N_ACTIONS)}
        print(f"    Action dist: {dist}")

        # Normalise
        x_mu  = X_all.mean(0)
        x_std = X_all.std(0).clamp(min=1e-6)
        X_norm = (X_all - x_mu) / x_std

        # Train/dev split (chronological)
        n_tr = int(N * args.train_frac)
        X_tr, y_tr   = X_norm[:n_tr],  y_all[:n_tr]
        X_dev, y_dev = X_norm[n_tr:],  y_all[n_tr:]
        print(f"    Train={len(y_tr):,}  Dev={len(y_dev):,}")

        model, history = train_model(
            X_tr, y_tr, X_dev, y_dev,
            x_mu, x_std, device,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
        )

        # Evaluate
        model.eval()
        with torch.no_grad():
            X_tr_d  = X_tr.to(device)
            X_dev_d = X_dev.to(device)
            y_tr_p  = model(X_tr_d).argmax(1).cpu()
            y_dev_p = model(X_dev_d).argmax(1).cpu()

        tr_acc  = float((y_tr_p  == y_tr).float().mean())
        dev_acc = float((y_dev_p == y_dev).float().mean())
        print(f"    train_acc={tr_acc:.4f}  dev_acc={dev_acc:.4f}  "
              f"elapsed={time.time()-t0:.0f}s")

        pc_dev = per_class_metrics(y_dev.numpy(), y_dev_p.numpy())
        cm_dev = confusion_matrix(y_dev.numpy(), y_dev_p.numpy())

        for aname, m in pc_dev.items():
            print(f"      {aname:<20s}  prec={m['precision']:.3f}  rec={m['recall']:.3f}  "
                  f"f1={m['f1']:.3f}  (tp={m['tp']} fn={m['fn']})")

        # Save model
        torch.save({
            "state_dict":  model.state_dict(),
            "x_mu":        x_mu,
            "x_std":       x_std,
            "in_dim":      X_all.shape[1],
            "n_actions":   N_ACTIONS,
            "teacher_name": teacher_name,
            "action_names": ACTION_NAMES,
        }, model_path)

        metrics = {
            "teacher_name":  teacher_name,
            "n_samples":     N,
            "n_train":       len(y_tr),
            "n_dev":         len(y_dev),
            "train_acc":     tr_acc,
            "dev_acc":       dev_acc,
            "action_distribution": dist,
            "per_class_dev": pc_dev,
            "confusion_matrix_dev": cm_dev,
            "history":       history,
            "elapsed_s":     time.time() - t0,
        }
        with open(metric_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"    Saved {model_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

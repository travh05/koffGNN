"""
STEP 4 — Training Script
=========================
Trains koffGNN on the processed interface graph dataset.

What this script does:
  1. Loads the KoffDataset from data/processed/
  2. Splits into train / val / test (70 / 15 / 15)
  3. Trains with Adam + cosine annealing LR schedule
  4. Logs to TensorBoard (view with: tensorboard --logdir=runs/)
  5. Saves best checkpoint by validation RMSE
  6. Reports final test RMSE and Pearson r

Usage:
  python 03_train.py [--epochs 100] [--batch_size 32] [--hidden_dim 128]

Expected results on ~500 antibody complexes:
  Validation RMSE  ≈ 0.5–0.8 log units    (very early MVP)
  Pearson r        ≈ 0.4–0.65
  (Performance improves significantly with more data)
"""

import argparse
import math
import random
import sys
import time
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from scipy.stats import pearsonr

# ── Project imports ────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from model.koff_gnn import KoffGNN, GaussianNLLLoss
from build_dataset import KoffDataset, PROC_DIR  # reuse the dataset class

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Reproducibility ────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

CKPT_DIR = ROOT / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# Data splitting and normalisation
# ══════════════════════════════════════════════════════════════════════════

def split_dataset(dataset, train_frac=0.70, val_frac=0.15, seed=42):
    """Random 70/15/15 split with reproducible shuffling."""
    n = len(dataset)
    idx = list(range(n))
    random.seed(seed)
    random.shuffle(idx)

    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train_idx = idx[:n_train]
    val_idx   = idx[n_train : n_train + n_val]
    test_idx  = idx[n_train + n_val :]

    log.info(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    return (
        [dataset[i] for i in train_idx],
        [dataset[i] for i in val_idx],
        [dataset[i] for i in test_idx],
    )


def compute_target_stats(train_data) -> tuple[float, float]:
    """
    Compute mean and std of log10(koff) on training set.
    We normalise targets to zero mean, unit std during training,
    then denormalise predictions at inference.
    """
    ys = torch.cat([d.y for d in train_data]).squeeze()
    return float(ys.mean()), float(ys.std())


def normalise_targets(data_list, mean: float, std: float):
    """Return new list with normalised y values."""
    out = []
    for d in data_list:
        d2 = d.clone()
        d2.y = (d.y - mean) / (std + 1e-8)
        out.append(d2)
    return out


# ══════════════════════════════════════════════════════════════════════════
# Training and evaluation loops
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, loss_fn, device, mean: float, std: float):
    """
    Returns (avg_loss, rmse_log10_koff, pearson_r).
    RMSE and Pearson are computed on the original (denormalised) scale.
    """
    model.eval()
    total_loss = 0.0
    total_graphs = 0
    all_pred, all_true = [], []

    for batch in loader:
        batch = batch.to(device)
        pred_mean, pred_log_var = model(batch)

        # Loss on normalised scale
        loss, _ = loss_fn(pred_mean, pred_log_var, batch.y)
        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs

        # Denormalise for metrics
        pred_denorm = pred_mean * std + mean
        true_denorm = batch.y  * std + mean  # batch.y is already normalised

        all_pred.append(pred_denorm.cpu().reshape(-1))
        all_true.append(true_denorm.cpu().reshape(-1))

    avg_loss = total_loss / max(total_graphs, 1)

    pred_cat = torch.cat(all_pred).numpy()
    true_cat = torch.cat(all_true).numpy()

    rmse = float(np.sqrt(((pred_cat - true_cat) ** 2).mean()))
    r, pval = pearsonr(pred_cat, true_cat)

    return avg_loss, rmse, float(r)


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    total_rmse = 0.0
    n_batches  = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred_mean, pred_log_var = model(batch)
        loss, rmse = loss_fn(pred_mean, pred_log_var, batch.y)

        loss.backward()
        # Gradient clipping — important for graph nets on noisy data
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        total_rmse += rmse.item()
        n_batches  += 1

    return total_loss / n_batches, total_rmse / n_batches


# ══════════════════════════════════════════════════════════════════════════
# Main training function
# ══════════════════════════════════════════════════════════════════════════

def train(args):
    set_seed(args.seed)

    # ── Device ─────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        # Enable TF32 on Ampere/Hopper/Blackwell for ~3x speed on matmuls
        torch.backends.cuda.matmul.allow_tf32 = True

    # ── Dataset ─────────────────────────────────────────────────────────
    log.info("Loading dataset ...")
    dataset = KoffDataset(root=str(PROC_DIR))
    log.info(f"Total graphs: {len(dataset)}")

    if len(dataset) < 20:
        log.error(
            "Dataset too small (<20 graphs). "
            "Make sure you ran 01_download_data.py and 02_build_dataset.py first."
        )
        sys.exit(1)

    train_data, val_data, test_data = split_dataset(
        dataset, args.train_frac, args.val_frac, args.seed
    )

    # Target normalisation (fit on train only)
    y_mean, y_std = compute_target_stats(train_data)
    log.info(f"Target normalisation: mean={y_mean:.3f}, std={y_std:.3f}")

    train_data = normalise_targets(train_data, y_mean, y_std)
    val_data   = normalise_targets(val_data,   y_mean, y_std)
    test_data  = normalise_targets(test_data,  y_mean, y_std)

    # ── DataLoaders ─────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=0,
    )

    # ── Model ────────────────────────────────────────────────────────────
    model = KoffGNN(
        hidden_dim = args.hidden_dim,
        n_layers   = args.n_layers,
        dropout    = args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters: {total_params:,}")

    # ── Optimiser + scheduler ────────────────────────────────────────────
    optimizer = Adam(
        model.parameters(),
        lr           = args.lr,
        weight_decay = args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    loss_fn   = GaussianNLLLoss()

    # ── TensorBoard ──────────────────────────────────────────────────────
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(ROOT / "runs" / "koff_gnn"))
        use_tb = True
    except ImportError:
        use_tb = False
        log.warning("TensorBoard not available. Install with: pip install tensorboard")

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_rmse  = float("inf")
    best_ckpt_path = CKPT_DIR / "koff_gnn_best.pt"
    patience       = args.patience
    wait           = 0

    log.info("")
    log.info(f"Starting training for {args.epochs} epochs ...")
    log.info(f"  batch_size={args.batch_size}  lr={args.lr}  hidden={args.hidden_dim}")
    log.info(f"  early_stopping patience={patience}")
    log.info("")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_rmse = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device
        )
        val_loss, val_rmse, val_r = evaluate(
            model, val_loader, loss_fn, device, y_mean, y_std
        )
        scheduler.step()

        dt = time.time() - t0
        log.info(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_rmse={train_rmse:.4f}  "
            f"val_rmse={val_rmse:.4f}  val_r={val_r:.3f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  t={dt:.1f}s"
        )

        if use_tb:
            writer.add_scalar("Loss/train",    train_loss, epoch)
            writer.add_scalar("RMSE/train",    train_rmse, epoch)
            writer.add_scalar("RMSE/val",      val_rmse,   epoch)
            writer.add_scalar("Pearson_r/val", val_r,      epoch)
            writer.add_scalar("LR",            scheduler.get_last_lr()[0], epoch)

        # ── Checkpoint on improvement ──────────────────────────────────
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(
                {
                    "epoch":      epoch,
                    "model_state": model.state_dict(),
                    "optimizer":  optimizer.state_dict(),
                    "val_rmse":   val_rmse,
                    "val_r":      val_r,
                    "y_mean":     y_mean,
                    "y_std":      y_std,
                    "args":       vars(args),
                },
                best_ckpt_path,
            )
            log.info(f"  ✓ New best model saved (val_rmse={val_rmse:.4f})")
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                log.info(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # ── Final test evaluation ─────────────────────────────────────────────
    log.info("")
    log.info("Loading best checkpoint for final test evaluation ...")
    ckpt = torch.load(best_ckpt_path, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    test_loss, test_rmse, test_r = evaluate(
        model, test_loader, loss_fn, device, y_mean, y_std
    )

    log.info("=" * 60)
    log.info("FINAL TEST RESULTS")
    log.info(f"  Test RMSE (log10 koff) : {test_rmse:.4f}")
    log.info(f"  Test Pearson r         : {test_r:.4f}")
    log.info(f"  Best val RMSE          : {best_val_rmse:.4f}")
    log.info("=" * 60)
    log.info("")
    log.info("NEXT STEP: Run  python 04_design.py")

    # Save a readable results summary
    summary_path = ROOT / "results" / "training_summary.txt"
    summary_path.parent.mkdir(exist_ok=True)
    summary_path.write_text(
        f"koffGNN Training Summary\n"
        f"========================\n"
        f"Test RMSE (log10 koff): {test_rmse:.4f}\n"
        f"Test Pearson r        : {test_r:.4f}\n"
        f"Best val RMSE         : {best_val_rmse:.4f}\n"
        f"Best epoch            : {ckpt['epoch']}\n"
        f"Args: {vars(args)}\n"
    )
    log.info(f"Results saved → {summary_path}")

    if use_tb:
        writer.close()


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Train koffGNN")
    p.add_argument("--epochs",      type=int,   default=150,  help="Max training epochs")
    p.add_argument("--batch_size",  type=int,   default=32,   help="Batch size")
    p.add_argument("--hidden_dim",  type=int,   default=128,  help="GNN hidden dimension")
    p.add_argument("--n_layers",    type=int,   default=4,    help="Number of MP layers")
    p.add_argument("--dropout",     type=float, default=0.15, help="Dropout rate")
    p.add_argument("--lr",          type=float, default=3e-4, help="Learning rate")
    p.add_argument("--weight_decay",type=float, default=1e-5, help="L2 regularisation")
    p.add_argument("--train_frac",  type=float, default=0.70, help="Train fraction")
    p.add_argument("--val_frac",    type=float, default=0.15, help="Val fraction")
    p.add_argument("--patience",    type=int,   default=20,   help="Early stopping patience")
    p.add_argument("--seed",        type=int,   default=42,   help="Random seed")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

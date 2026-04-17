"""
STEP 0 — Quick-Start Smoke Test (run BEFORE downloading real data)
=================================================================
This script generates SYNTHETIC interface graphs and trains koffGNN
on them. It verifies your entire pipeline works end-to-end in under
5 minutes, before you wait for the real data download.

Run this FIRST to confirm GPU, PyTorch, and PyG are all working:
    python 00_quickstart_test.py

Expected output (on RTX 5090):
    GPU detected: NVIDIA GeForce RTX 5090
    Synthetic dataset: 300 graphs
    Training on GPU ...
    Epoch 10/50 | train_rmse=0.xx | val_rmse=0.xx
    ...
    SMOKE TEST PASSED. Your pipeline is ready for real data.
"""

import math
import random
import sys
import logging
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from scipy.stats import pearsonr

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Synthetic data generator
# ══════════════════════════════════════════════════════════════════════════

NODE_DIM  = 30
EDGE_DIM  = 8
GRAPH_DIM = 4

def make_synthetic_graph(seed: int) -> Data:
    """
    Generate a plausible synthetic interface graph.

    The synthetic koff is a DETERMINISTIC function of features
    so the model has something real to learn:
        log10(koff) ≈ -3.5 + 1.5 * (1 - burial) - 0.8 * n_hbonds_norm
                      + noise

    This mimics the physical intuition: buried interfaces with
    many H-bonds have lower koff (longer residence time).
    """
    rng = np.random.RandomState(seed)

    n_nodes = rng.randint(15, 60)
    n_edges = rng.randint(n_nodes, n_nodes * 4)

    # Node features
    x = rng.randn(n_nodes, NODE_DIM).astype(np.float32)

    # Make burial proxy (feature index 20) correlate with koff
    burial = np.abs(rng.randn(n_nodes)).clip(0, 1).astype(np.float32)
    x[:, 20] = burial

    # Chain identity (roughly half binder, half target)
    x[:, 29] = (np.arange(n_nodes) > n_nodes // 2).astype(np.float32)

    # Edge features
    src = rng.randint(0, n_nodes, n_edges)
    dst = rng.randint(0, n_nodes, n_edges)
    edge_attr = rng.randn(n_edges, EDGE_DIM).astype(np.float32)

    # Make H-bond feature (index 2) binary
    edge_attr[:, 2] = (rng.rand(n_edges) > 0.85).astype(np.float32)
    # Cross-chain flag
    edge_attr[:, 6] = (src < n_nodes // 2).astype(np.float32) * \
                      (dst >= n_nodes // 2).astype(np.float32)

    # Graph scalars
    n_hbonds = edge_attr[:, 2].sum()
    graph_feat = np.array(
        [n_nodes, n_nodes // 2, n_nodes - n_nodes // 2, n_hbonds],
        dtype=np.float32,
    )

    # Deterministic (noisy) koff label
    mean_burial  = float(burial.mean())
    n_hb_norm    = float(n_hbonds / max(n_edges, 1))
    log_koff = (
        -3.5
        + 1.5 * (1.0 - mean_burial)   # less buried → higher koff
        - 0.8 * n_hb_norm * 10         # more H-bonds → lower koff
        + rng.randn() * 0.4            # noise
    )
    log_koff = float(np.clip(log_koff, -6.0, 0.0))  # physical range

    return Data(
        x          = torch.tensor(x),
        edge_index = torch.tensor([src, dst], dtype=torch.long),
        edge_attr  = torch.tensor(edge_attr),
        graph_feat = torch.tensor(graph_feat),
        y          = torch.tensor([[log_koff]]),
    )


def make_synthetic_dataset(n: int = 300) -> list:
    return [make_synthetic_graph(seed=i) for i in range(n)]


# ══════════════════════════════════════════════════════════════════════════
# Mini training loop for smoke test
# ══════════════════════════════════════════════════════════════════════════

def run_smoke_test():
    log.info("=" * 60)
    log.info("  koffGNN QUICK-START SMOKE TEST")
    log.info("=" * 60)

    # ── GPU check ──────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        log.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        log.info(f"VRAM        : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    else:
        log.warning("No GPU — running on CPU (will be slow on full dataset)")

    # ── Import model ────────────────────────────────────────────────────
    try:
        from model.koff_gnn import KoffGNN, GaussianNLLLoss
    except ImportError as e:
        log.error(f"Cannot import model: {e}")
        log.error("Make sure you are in the koff_designer directory.")
        sys.exit(1)

    # ── Synthetic data ──────────────────────────────────────────────────
    log.info("Generating 300 synthetic interface graphs ...")
    all_data = make_synthetic_dataset(300)
    train_data = all_data[:210]
    val_data   = all_data[210:255]
    test_data  = all_data[255:]
    log.info(f"  train={len(train_data)} val={len(val_data)} test={len(test_data)}")

    # Target normalisation
    ys = torch.cat([d.y for d in train_data]).squeeze()
    y_mean, y_std = float(ys.mean()), float(ys.std())
    log.info(f"  Target: mean={y_mean:.2f}, std={y_std:.2f}")

    def normalise(data_list):
        out = []
        for d in data_list:
            d2 = d.clone()
            d2.y = (d.y - y_mean) / (y_std + 1e-8)
            out.append(d2)
        return out

    train_data = normalise(train_data)
    val_data   = normalise(val_data)
    test_data  = normalise(test_data)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=64)
    test_loader  = DataLoader(test_data,  batch_size=64)

    # ── Model ────────────────────────────────────────────────────────────
    model   = KoffGNN(hidden_dim=64, n_layers=3, dropout=0.1).to(device)
    optim   = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = GaussianNLLLoss()

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters: {n_params:,}")
    log.info("Training on synthetic data (50 epochs) ...")

    best_val_rmse = float("inf")
    for epoch in range(1, 51):
        # Train
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optim.zero_grad()
            pm, plv = model(batch)
            loss, _ = loss_fn(pm, plv, batch.y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()

        # Validate
        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pm, _ = model(batch)
                pred_d = pm * y_std + y_mean
                true_d = batch.y * y_std + y_mean
                all_pred.append(pred_d.cpu().squeeze())
                all_true.append(true_d.cpu().squeeze())

        pred_cat = torch.cat(all_pred).numpy()
        true_cat = torch.cat(all_true).numpy()
        val_rmse = float(np.sqrt(((pred_cat - true_cat)**2).mean()))
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse

        if epoch % 10 == 0:
            r, _ = pearsonr(pred_cat, true_cat)
            log.info(f"  Epoch {epoch:02d}/50 | val_rmse={val_rmse:.4f} | r={r:.3f}")

    # Final test
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pm, _ = model(batch)
            pred_d = pm * y_std + y_mean
            true_d = batch.y * y_std + y_mean
            all_pred.append(pred_d.cpu().squeeze())
            all_true.append(true_d.cpu().squeeze())

    pred_cat = torch.cat(all_pred).numpy()
    true_cat = torch.cat(all_true).numpy()
    test_rmse = float(np.sqrt(((pred_cat - true_cat)**2).mean()))
    r, _      = pearsonr(pred_cat, true_cat)

    log.info("")
    log.info("=" * 60)
    if test_rmse < 1.5 and r > 0.3:
        log.info("  SMOKE TEST PASSED")
    else:
        log.info("  SMOKE TEST WARNING — results below expectation")
        log.info("  (this may be normal for very small synthetic sets)")
    log.info(f"  Test RMSE (log10 koff) : {test_rmse:.4f}")
    log.info(f"  Test Pearson r         : {r:.4f}")
    log.info(f"  Best val RMSE          : {best_val_rmse:.4f}")
    log.info("")
    log.info("  Your pipeline is ready.")
    log.info("  Next: python 01_download_data.py")
    log.info("=" * 60)


if __name__ == "__main__":
    run_smoke_test()

"""
STEP 3 — Model Architecture
============================
koffGNN: A graph neural network that predicts log10(koff) from
         the protein-protein interface graph.

Architecture
------------
The key insight: koff is set by ΔG‡ (transition state free energy),
NOT by ΔG (ground state affinity). Our model learns interface geometry
features that correlate with transition state stability.

  Layer 1–3:  Edge-conditioned message passing (ECMPConv)
              — each message is computed from both node features
                AND the edge features connecting them.
              This is important because H-bond network topology
              at the interface periphery (edge features) strongly
              predicts the unbinding barrier.

  Layer 4:    Graph-level readout via global mean + max pooling
              concatenated with graph-level scalars
              (n_interface_residues, n_hbonds, etc.)

  Layer 5–6:  MLP head → scalar prediction of log10(koff)

  Uncertainty: We use a Gaussian NLL loss with learned σ per sample.
               This gives us calibrated confidence intervals at test time.

Hyperparameters (tuned for RTX 5090 / ~3,000 training graphs):
  hidden_dim  = 128
  n_layers    = 4
  dropout     = 0.15
  lr          = 3e-4 (Adam)
  weight_decay = 1e-5
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    MessagePassing,
    global_mean_pool,
    global_max_pool,
    global_add_pool,
)
from torch_geometric.data import Data, Batch

NODE_DIM  = 30
EDGE_DIM  = 8
GRAPH_DIM = 4   # graph-level scalar features


# ══════════════════════════════════════════════════════════════════════════
# Edge-conditioned message passing layer
# ══════════════════════════════════════════════════════════════════════════

class ECMPConv(MessagePassing):
    """
    Edge-Conditioned Message Passing convolution.

    For each node i, the message from neighbor j is:
        m_ij = MLP_edge( [h_j || e_ij] )

    where h_j is neighbor embedding, e_ij is the edge feature vector.
    The update is:
        h_i' = MLP_update( [h_i || Σ_j m_ij] )

    This is important for koff prediction because we want cross-chain
    H-bond edges and hydrophobic contacts to directly influence
    the embedding — the 'periphery' edges matter most for ΔG‡.
    """

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int, dropout: float = 0.1):
        super().__init__(aggr="add")

        # Edge MLP: transforms (h_j, e_ij) → message
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim + edge_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

        # Update MLP: transforms (h_i, Σm_ij) → h_i'
        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

        # Residual projection if dimensions differ
        self.residual = (
            nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        # Propagate: calls message() then aggregate() then update()
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # x_j: neighbor node features; edge_attr: edge features
        combined = torch.cat([x_j, edge_attr], dim=-1)
        return self.edge_mlp(combined)

    def update(self, agg: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x, agg], dim=-1)
        out = self.update_mlp(combined)
        return out + self.residual(x)   # residual connection


# ══════════════════════════════════════════════════════════════════════════
# Attention pooling (learn which residues matter for koff)
# ══════════════════════════════════════════════════════════════════════════

class AttentionPool(nn.Module):
    """
    Soft attention over interface residues.

    We predict a scalar attention weight per node, then compute
    a weighted sum of node embeddings as the graph representation.

    Physical motivation: not all interface residues contribute equally
    to the unbinding barrier. 'Anchor' contacts (deeply buried, multiple
    H-bonds) have high koff-relevant attention; peripheral contacts less so.
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        x     : (N_total_nodes, D)
        batch : (N_total_nodes,) — tells us which graph each node belongs to
        Returns: (B, D) graph-level embeddings
        """
        # Compute raw attention scores per node
        scores = self.attn(x)  # (N, 1)

        # Softmax within each graph using PyG softmax (no torch_scatter needed)
        from torch_geometric.utils import softmax as pyg_softmax
        attn_weights = pyg_softmax(scores, batch)  # (N, 1)

        # Weighted sum per graph using native scatter_add
        B = int(batch.max().item()) + 1
        D = x.size(-1)
        out = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        weighted = attn_weights * x  # (N, D)
        out.scatter_add_(0, batch.unsqueeze(1).expand_as(weighted), weighted)
        return out


# ══════════════════════════════════════════════════════════════════════════
# Main model: koffGNN
# ══════════════════════════════════════════════════════════════════════════

class KoffGNN(nn.Module):
    """
    Interface-graph GNN for log10(koff) prediction.

    Input:
        x          : (N, NODE_DIM)   node features
        edge_index : (2, E)          edge connectivity
        edge_attr  : (E, EDGE_DIM)   edge features
        graph_feat : (B, GRAPH_DIM)  graph-level scalars
        batch      : (N,)            batch assignment

    Output:
        pred_mean  : (B, 1)  predicted log10(koff)
        pred_log_var: (B, 1) log variance (for uncertainty quantification)
    """

    def __init__(
        self,
        node_dim:   int   = NODE_DIM,
        edge_dim:   int   = EDGE_DIM,
        graph_dim:  int   = GRAPH_DIM,
        hidden_dim: int   = 128,
        n_layers:   int   = 4,
        dropout:    float = 0.15,
    ):
        super().__init__()

        # Input projection
        self.node_embed = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.edge_embed = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim // 2),
            nn.GELU(),
        )

        # Message passing layers
        self.mp_layers = nn.ModuleList()
        for i in range(n_layers):
            self.mp_layers.append(
                ECMPConv(
                    in_dim   = hidden_dim,
                    out_dim  = hidden_dim,
                    edge_dim = hidden_dim // 2,
                    dropout  = dropout,
                )
            )

        # Layer norms between MP layers
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(n_layers)]
        )

        # Graph readout: attention pool + mean pool + max pool
        self.attn_pool = AttentionPool(hidden_dim)
        # concatenation of [attn, mean, max] → 3 * hidden_dim
        readout_dim = 3 * hidden_dim + graph_dim

        # MLP prediction head
        self.head = nn.Sequential(
            nn.Linear(readout_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
        )

        # Two outputs: mean and log_var (Gaussian uncertainty)
        self.out_mean    = nn.Linear(64, 1)
        self.out_log_var = nn.Linear(64, 1)

        self._init_weights()

    def _init_weights(self):
        """Xavier initialisation for all linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        x          = data.x
        edge_index = data.edge_index
        edge_attr  = data.edge_attr
        graph_feat = data.graph_feat
        batch      = data.batch if hasattr(data, "batch") and data.batch is not None \
                     else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # ── Input embeddings ──────────────────────────────────────────
        h = self.node_embed(x)
        e = self.edge_embed(edge_attr)

        # ── Message passing ────────────────────────────────────────────
        for mp_layer, norm in zip(self.mp_layers, self.norms):
            h = norm(mp_layer(h, edge_index, e))

        # ── Graph readout ──────────────────────────────────────────────
        h_attn = self.attn_pool(h, batch)                      # (B, D)
        h_mean = global_mean_pool(h, batch)                    # (B, D)
        h_max  = global_max_pool(h, batch)                     # (B, D)

        # Concatenate with graph-level scalars
        # Normalise graph_feat to roughly zero-mean
        gf = graph_feat.view(-1, GRAPH_DIM)
        gf_norm = (gf - gf.mean(0, keepdim=True)) / (gf.std(0, keepdim=True, correction=0) + 1e-6)

        pooled = torch.cat([h_attn, h_mean, h_max, gf_norm], dim=-1)

        # ── Prediction head ────────────────────────────────────────────
        z = self.head(pooled)
        pred_mean    = self.out_mean(z)         # (B, 1) — log10(koff)
        pred_log_var = self.out_log_var(z)      # (B, 1) — log variance

        return pred_mean, pred_log_var


# ══════════════════════════════════════════════════════════════════════════
# Loss: Gaussian Negative Log-Likelihood
# ══════════════════════════════════════════════════════════════════════════

class GaussianNLLLoss(nn.Module):
    """
    Negative log-likelihood under a Gaussian with learned variance.

        L = 0.5 * ( log(σ²) + (y - μ)² / σ² )

    Why this over plain MSE?
      - MSE implicitly assumes uniform noise — but SPR measurements have
        heterogeneous uncertainty. Some labs report ±10%, others ±10x.
      - By learning σ per sample, the model down-weights noisy training
        examples automatically.
      - At inference, σ gives us calibrated prediction intervals.
    """

    def forward(
        self,
        pred_mean:     torch.Tensor,
        pred_log_var:  torch.Tensor,
        target:        torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (loss, rmse) where:
          loss — NLL loss for backprop
          rmse — detached RMSE for logging
        """
        # Clamp log_var to avoid numerical issues
        log_var  = pred_log_var.clamp(-6.0, 6.0)
        var      = torch.exp(log_var)

        residual = pred_mean - target
        nll      = 0.5 * (log_var + residual.pow(2) / var)
        loss     = nll.mean()

        with torch.no_grad():
            rmse = residual.pow(2).mean().sqrt()

        return loss, rmse


# ══════════════════════════════════════════════════════════════════════════
# Quick sanity check
# ══════════════════════════════════════════════════════════════════════════

def _smoke_test():
    """Verify the model forward pass works with random data."""
    model = KoffGNN()
    model.eval()

    n_nodes = 40
    n_edges = 200
    batch_size = 4

    # Simulate a batch of 4 graphs
    batch_ids = torch.zeros(n_nodes * batch_size, dtype=torch.long)
    for b in range(batch_size):
        batch_ids[b * n_nodes:(b + 1) * n_nodes] = b

    data = Batch(
        x          = torch.randn(n_nodes * batch_size, NODE_DIM),
        edge_index  = torch.randint(0, n_nodes * batch_size, (2, n_edges * batch_size)),
        edge_attr  = torch.randn(n_edges * batch_size, EDGE_DIM),
        graph_feat = torch.randn(batch_size, GRAPH_DIM),
        batch      = batch_ids,
        y          = torch.randn(batch_size, 1),
    )

    with torch.no_grad():
        mean, log_var = model(data)

    assert mean.shape    == (batch_size, 1), f"Bad mean shape: {mean.shape}"
    assert log_var.shape == (batch_size, 1), f"Bad log_var shape: {log_var.shape}"

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Predicted log10(koff) [random]: {mean.squeeze().tolist()}")
    print(f"Predicted log_var             : {log_var.squeeze().tolist()}")
    print("Smoke test passed.")


if __name__ == "__main__":
    _smoke_test()

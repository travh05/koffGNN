import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from torch_geometric.utils import softmax as pyg_softmax
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from torch_geometric.utils import softmax as pyg_softmax


NODE_DIM  = 1290
EDGE_DIM  = 8
GRAPH_DIM = 4

class ECMPConv(MessagePassing):
    def __init__(self, in_dim, out_dim, edge_dim, dropout=0.1):
        super().__init__(aggr="add")
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim + edge_dim, out_dim), nn.LayerNorm(out_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(out_dim, out_dim))
        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim), nn.LayerNorm(out_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(out_dim, out_dim))
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    def message(self, x_j, edge_attr):
        return self.edge_mlp(torch.cat([x_j, edge_attr], dim=-1))
    def update(self, agg, x):
        return self.update_mlp(torch.cat([x, agg], dim=-1)) + self.residual(x)

class AttentionPool(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(in_dim,64), nn.Tanh(), nn.Linear(64,1))
    def forward(self, x, batch):
        w = pyg_softmax(self.attn(x), batch)
        B = int(batch.max().item()) + 1
        out = torch.zeros(B, x.size(-1), device=x.device)
        out.scatter_add_(0, batch.unsqueeze(1).expand_as(x), w * x)
        return out

class KoffGNNEsm(nn.Module):
    def __init__(self, node_dim=NODE_DIM, edge_dim=EDGE_DIM,
                 graph_dim=GRAPH_DIM, hidden_dim=256, n_layers=4, dropout=0.15):
        super().__init__()
        self.node_embed = nn.Sequential(
            nn.Linear(node_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.edge_embed = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim//2), nn.GELU())
        self.mp_layers = nn.ModuleList([
            ECMPConv(hidden_dim, hidden_dim, hidden_dim//2, dropout)
            for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.attn_pool = AttentionPool(hidden_dim)
        readout_dim = 3 * hidden_dim + graph_dim
        self.head = nn.Sequential(
            nn.Linear(readout_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(256, 64), nn.GELU())
        self.out_mean    = nn.Linear(64, 1)
        self.out_log_var = nn.Linear(64, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, data):
        x, ei, ea = data.x, data.edge_index, data.edge_attr
        gf = data.graph_feat.view(-1, GRAPH_DIM)
        batch = data.batch if data.batch is not None else \
                torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        h = self.node_embed(x)
        e = self.edge_embed(ea)
        for mp, norm in zip(self.mp_layers, self.norms):
            h = norm(mp(h, ei, e))
        ha = self.attn_pool(h, batch)
        hm = global_mean_pool(h, batch)
        hx = global_max_pool(h, batch)
        gfn = (gf - gf.mean(0,keepdim=True)) / (gf.std(0,keepdim=True,correction=0)+1e-6)
        z = self.head(torch.cat([ha, hm, hx, gfn], dim=-1))
        return self.out_mean(z), self.out_log_var(z)


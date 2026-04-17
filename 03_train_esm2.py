"""Train koffGNN with ESM-2 node features (node_dim=1290)."""
import argparse, random, sys, time, logging
import numpy as np
import torch, torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset
from scipy.stats import pearsonr
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from model.koff_gnn import GaussianNLLLoss

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

NODE_DIM  = 1290
EDGE_DIM  = 8
GRAPH_DIM = 4
CKPT_DIR  = ROOT / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

# ── Inline model with correct node_dim ────────────────────────────────────
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from torch_geometric.utils import softmax as pyg_softmax

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

# ── Load ESM-2 dataset ─────────────────────────────────────────────────────
class EsmDataset(InMemoryDataset):
    def __init__(self):
        super().__init__(root="data/processed")
        self.data, self.slices = torch.load(
            "data/processed/koff_dataset_esm2.pt", weights_only=False)

def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

@torch.no_grad()
def evaluate(model, loader, loss_fn, device, mean, std):
    model.eval()
    total_loss, total_graphs = 0.0, 0
    all_pred, all_true = [], []
    for batch in loader:
        batch = batch.to(device)
        pm, plv = model(batch)
        loss, _ = loss_fn(pm, plv, batch.y)
        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs
        all_pred.append((pm * std + mean).cpu().reshape(-1))
        all_true.append((batch.y * std + mean).cpu().reshape(-1))
    pred = torch.cat(all_pred).numpy()
    true = torch.cat(all_true).numpy()
    rmse = float(np.sqrt(((pred-true)**2).mean()))
    r, _ = pearsonr(pred, true)
    return total_loss/max(total_graphs,1), rmse, float(r)

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device} — {torch.cuda.get_device_name(0)}")
    torch.backends.cuda.matmul.allow_tf32 = True

    dataset = EsmDataset()
    log.info(f"ESM-2 dataset: {len(dataset)} graphs, node_dim={dataset[0].x.shape[1]}")

    n = len(dataset)
    idx = list(range(n)); random.shuffle(idx)
    n_train = int(n*0.70); n_val = int(n*0.15)
    train_d = [dataset[i] for i in idx[:n_train]]
    val_d   = [dataset[i] for i in idx[n_train:n_train+n_val]]
    test_d  = [dataset[i] for i in idx[n_train+n_val:]]
    log.info(f"Split: train={len(train_d)} val={len(val_d)} test={len(test_d)}")

    ys = torch.cat([d.y for d in train_d]).squeeze()
    mean, std = float(ys.mean()), float(ys.std())
    log.info(f"Target: mean={mean:.3f} std={std:.3f}")

    def norm(dl):
        out = []
        for d in dl:
            d2 = d.clone(); d2.y = (d.y - mean)/(std+1e-8); out.append(d2)
        return out

    train_d, val_d, test_d = norm(train_d), norm(val_d), norm(test_d)
    tl = DataLoader(train_d, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    vl = DataLoader(val_d,   batch_size=args.batch_size*2, shuffle=False, num_workers=0)
    el = DataLoader(test_d,  batch_size=args.batch_size*2, shuffle=False, num_workers=0)

    model = KoffGNNEsm(hidden_dim=args.hidden_dim, n_layers=args.n_layers,
                       dropout=args.dropout).to(device)
    log.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt  = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    loss_fn = GaussianNLLLoss()

    best_val, wait = float("inf"), 0
    ckpt = CKPT_DIR / "koff_gnn_esm2_best.pt"

    for epoch in range(1, args.epochs+1):
        model.train()
        for batch in tl:
            batch = batch.to(device)
            opt.zero_grad()
            pm, plv = model(batch)
            loss, _ = loss_fn(pm, plv, batch.y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        _, val_rmse, val_r = evaluate(model, vl, loss_fn, device, mean, std)
        log.info(f"Epoch {epoch:03d}/{args.epochs} val_rmse={val_rmse:.4f} val_r={val_r:.3f} lr={sched.get_last_lr()[0]:.2e}")
        if val_rmse < best_val:
            best_val = val_rmse
            torch.save({"epoch":epoch,"model_state":model.state_dict(),
                        "y_mean":mean,"y_std":std,"val_rmse":val_rmse,"val_r":val_r,
                        "hidden_dim":args.hidden_dim,"n_layers":args.n_layers}, ckpt)
            log.info(f"  ✓ Saved (val_rmse={val_rmse:.4f})")
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                log.info(f"Early stopping at epoch {epoch}")
                break

    c = torch.load(ckpt, weights_only=False)
    model.load_state_dict(c["model_state"])
    _, test_rmse, test_r = evaluate(model, el, loss_fn, device, mean, std)
    log.info("=" * 60)
    log.info(f"FINAL  test_rmse={test_rmse:.4f}  test_r={test_r:.4f}  best_val={best_val:.4f}")
    log.info("=" * 60)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=200)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--hidden_dim", type=int,   default=256)
    p.add_argument("--n_layers",   type=int,   default=4)
    p.add_argument("--dropout",    type=float, default=0.15)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--wd",         type=float, default=1e-4)
    p.add_argument("--patience",   type=int,   default=30)
    p.add_argument("--seed",       type=int,   default=42)
    train(p.parse_args())

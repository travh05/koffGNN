"""
esm2_features.py — Add ESM-2 embeddings to interface graphs
============================================================
Replaces AA one-hot (dim 0:20) with ESM-2 per-residue embeddings (dim 1280).
New node_dim = 1280 + 10 (structural features) = 1290

Run once to rebuild the dataset with ESM-2 features:
    python esm2_features.py
"""
import torch
import numpy as np
import logging
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Load ESM-2 once ────────────────────────────────────────────────────────
def load_esm2():
    import esm
    log.info("Loading ESM-2 650M...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().cuda()
    batch_converter = alphabet.get_batch_converter()
    log.info("ESM-2 loaded.")
    return model, alphabet, batch_converter

@torch.no_grad()
def get_esm2_embeddings(sequences: list[str], model, batch_converter, layer=33) -> torch.Tensor:
    """
    Get per-residue ESM-2 embeddings for a list of sequences.
    Returns list of tensors, each shape (seq_len, 1280).
    """
    data = [(f"seq{i}", s) for i, s in enumerate(sequences)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.cuda()
    results = model(tokens, repr_layers=[layer])
    # Shape: (batch, seq_len+2, 1280) — strip BOS/EOS tokens
    embs = results["representations"][layer][:, 1:-1, :]
    return [embs[i, :len(seq), :].cpu() for i, seq in enumerate(sequences)]


def rebuild_dataset_with_esm2():
    """
    Load existing dataset, add ESM-2 embeddings to node features,
    save new dataset to data/processed/processed/koff_dataset_esm2.pt
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from build_dataset import KoffDataset, PROC_DIR
    from torch_geometric.data import Data
    from Bio.PDB import PDBParser
    from build_dataset import THREE_TO_ONE, PROC_DIR

    PDB_DIR = Path("data/raw/pdb")
    RAW_DIR = Path("data/raw")

    # Load existing dataset
    log.info("Loading existing dataset...")
    dataset = KoffDataset(root=str(PROC_DIR))
    log.info(f"  {len(dataset)} graphs")

    esm_model, alphabet, batch_converter = load_esm2()

    # Load summary to get chain info
    import pandas as pd
    summary = pd.read_csv(RAW_DIR / "sabdab_summary.tsv", sep="\t")
    summary["pdb"] = summary["pdb"].astype(str).str.lower()
    pdb_to_chains = {
        row["pdb"]: (str(row.get("hchain","H")), str(row.get("antigen_chain","A")))
        for _, row in summary.iterrows()
    }

    parser = PDBParser(QUIET=True)

    new_graphs = []
    skipped = 0

    for idx in range(len(dataset)):
        graph = dataset[idx]
        pdb_id = str(graph.pdb_id).lower()
        pdb_path = PDB_DIR / f"{pdb_id}.pdb"

        if not pdb_path.exists():
            skipped += 1
            continue

        # Get chain sequences from structure
        try:
            struct = parser.get_structure(pdb_id, str(pdb_path))
            chains = {c.get_id(): "".join(
                THREE_TO_ONE.get(r.get_resname().strip(), "X")
                for r in c.get_residues()
                if r.get_id()[0] == " "
            ) for c in struct[0].get_chains()}
        except Exception:
            skipped += 1
            continue

        hchain, agchain = pdb_to_chains.get(pdb_id, ("H", "A"))
        binder_seq = chains.get(hchain, "")
        target_seq = chains.get(agchain, "")

        if not binder_seq or not target_seq:
            # Try first two chains
            chain_list = list(chains.keys())
            if len(chain_list) >= 2:
                binder_seq = chains[chain_list[0]]
                target_seq = chains[chain_list[1]]
            else:
                skipped += 1
                continue

        if not binder_seq or not target_seq:
            skipped += 1
            continue

        # Get ESM-2 embeddings for both chains
        try:
            embs = get_esm2_embeddings([binder_seq, target_seq], esm_model, batch_converter)
            binder_emb = embs[0]  # (len_binder, 1280)
            target_emb = embs[1]  # (len_target, 1280)
        except Exception as e:
            log.debug(f"ESM-2 failed for {pdb_id}: {e}")
            skipped += 1
            continue

        # Match embeddings to interface nodes
        # graph.x[:, 29] = chain label (0=binder, 1=target)
        # graph.x[:, 0:20] = AA one-hot (to be replaced)
        n_nodes = graph.x.shape[0]
        chain_labels = graph.x[:, 29]

        # Structural features (keep everything except one-hot: dims 20-29)
        struct_feats = graph.x[:, 20:]  # (n_nodes, 10)

        # Build ESM-2 feature matrix
        esm_feats = torch.zeros(n_nodes, 1280)
        b_idx = 0  # pointer into binder_emb
        t_idx = 0  # pointer into target_emb

        for ni in range(n_nodes):
            if chain_labels[ni] < 0.5:  # binder node
                if b_idx < binder_emb.shape[0]:
                    esm_feats[ni] = binder_emb[b_idx]
                    b_idx += 1
            else:  # target node
                if t_idx < target_emb.shape[0]:
                    esm_feats[ni] = target_emb[t_idx]
                    t_idx += 1

        # New node features: [ESM-2 (1280) | structural (10)] = 1290 dims
        new_x = torch.cat([esm_feats, struct_feats], dim=-1)

        new_graph = Data(
            x          = new_x,
            edge_index = graph.edge_index,
            edge_attr  = graph.edge_attr,
            graph_feat = graph.graph_feat,
            y          = graph.y,
            pdb_id     = graph.pdb_id,
            koff       = graph.koff,
        )
        new_graphs.append(new_graph)

        if (idx + 1) % 100 == 0:
            log.info(f"  [{idx+1}/{len(dataset)}] processed={len(new_graphs)} skipped={skipped}")

    log.info(f"Done: {len(new_graphs)} graphs with ESM-2 features, {skipped} skipped")

    # Save
    out_path = Path(str(PROC_DIR)) / "koff_dataset_esm2.pt"
    import torch_geometric
    from torch_geometric.data import InMemoryDataset
    data, slices = InMemoryDataset.collate(new_graphs)
    torch.save((data, slices), out_path)
    log.info(f"Saved → {out_path}")
    log.info(f"New node_dim: {new_graphs[0].x.shape[1]}")
    return out_path, len(new_graphs)


if __name__ == "__main__":
    out_path, n = rebuild_dataset_with_esm2()
    log.info(f"Dataset ready: {n} graphs at {out_path}")
    log.info("Now run:")
    log.info("  python 03_train_esm2.py")

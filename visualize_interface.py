"""
visualize_interface.py
======================
Extracts interface graph from a PDB + koffGNN attention weights,
outputs a JSON file for the 3D browser viewer.

Run:
    python visualize_interface.py --pdb data/raw/pdb/5lqb.pdb --chain A
    
Then open viewer.html in browser (or it auto-opens).
"""

import json
import math
import sys
import argparse
import webbrowser
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def extract_viz_data(pdb_path: str, binder_chain: str, device=None):
    """
    Returns a dict with:
      - nodes: list of {id, x, y, z, chain, resname, resnum, attn_weight, burial}
      - edges: list of {src, dst, type, weight, cross_chain}
      - meta:  {pdb, n_nodes, n_edges, log_koff_pred, koff_pred}
    """
    from build_dataset import pdb_to_graph, THREE_TO_ONE
    from Bio.PDB import PDBParser

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pdb_path = Path(pdb_path)

    # ── Load koffGNN ────────────────────────────────────────────────────
    from esm2_model import KoffGNNEsm
    ckpt = torch.load(
        ROOT / "checkpoints" / "koff_gnn_esm2_best.pt",
        map_location=device, weights_only=False
    )
    model = KoffGNNEsm(
        hidden_dim=ckpt.get("hidden_dim", 256),
        n_layers=ckpt.get("n_layers", 4),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    y_mean, y_std = ckpt["y_mean"], ckpt["y_std"]

    # ── ESM-2 ────────────────────────────────────────────────────────────
    import esm as esm_lib
    esm_model, alphabet = esm_lib.pretrained.esm2_t33_650M_UR50D()
    esm_model = esm_model.eval().to(device)
    bc = alphabet.get_batch_converter()

    # ── Parse structure ──────────────────────────────────────────────────
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("x", str(pdb_path))

    chains = {}
    chain_residues = {}
    for c in struct[0].get_chains():
        res_list = [r for r in c.get_residues() if r.get_id()[0] == " "]
        if res_list:
            seq = "".join(THREE_TO_ONE.get(r.get_resname().strip(), "X") for r in res_list)
            chains[c.get_id()] = seq
            chain_residues[c.get_id()] = res_list

    binder_seq = chains.get(binder_chain, "")
    target_chain_id = next((c for c in chains if c != binder_chain), None)
    target_seq = chains.get(target_chain_id, "")

    if not binder_seq or not target_seq:
        raise ValueError("Could not find binder/target sequences")

    # ── Get ESM-2 embeddings ─────────────────────────────────────────────
    def get_emb(seq):
        _, _, tokens = bc([("s", seq)])
        with torch.no_grad():
            out = esm_model(tokens.to(device), repr_layers=[33])
        return out["representations"][33][0, 1:len(seq)+1].cpu()

    binder_emb = get_emb(binder_seq)
    target_emb = get_emb(target_seq)

    # ── Build graph ───────────────────────────────────────────────────────
    graph = pdb_to_graph(pdb_path, koff=1e-3)
    if graph is None:
        raise ValueError("Could not parse interface graph")

    # Rebuild with ESM-2 features
    chain_labels = graph.x[:, 29]
    struct_feats  = graph.x[:, 20:]
    n_nodes = graph.x.shape[0]
    esm_feat = torch.zeros(n_nodes, 1280)
    b_idx = t_idx = 0
    binder_mask = []
    for ni in range(n_nodes):
        if chain_labels[ni] < 0.5:
            if b_idx < binder_emb.shape[0]:
                esm_feat[ni] = binder_emb[b_idx]; b_idx += 1
            binder_mask.append(True)
        else:
            if t_idx < target_emb.shape[0]:
                esm_feat[ni] = target_emb[t_idx]; t_idx += 1
            binder_mask.append(False)

    graph.x = torch.cat([esm_feat, struct_feats], dim=-1)
    graph = graph.to(device)
    graph.batch = torch.zeros(n_nodes, dtype=torch.long, device=device)

    # ── Forward pass with attention hook ─────────────────────────────────
    attn_weights = {}

    def hook_attn(module, input, output):
        # Capture attention weights from AttentionPool
        x_in = input[0]
        batch_in = input[1]
        scores = module.attn(x_in)
        from torch_geometric.utils import softmax as pyg_softmax
        w = pyg_softmax(scores, batch_in)
        attn_weights["weights"] = w.detach().cpu().squeeze().numpy()

    handle = model.attn_pool.register_forward_hook(hook_attn)

    with torch.no_grad():
        pm, plv = model(graph)

    handle.remove()

    log_koff = (pm.item() * y_std) + y_mean
    koff_pred = 10 ** log_koff

    # ── Extract 3D coordinates ────────────────────────────────────────────
    # Map interface nodes back to residues by sequence order
    binder_res_list = chain_residues.get(binder_chain, [])
    target_res_list = chain_residues.get(target_chain_id, [])

    CONTACT_CUTOFF = 8.0

    def get_ca(res):
        try: return res["CA"].get_vector().get_array()
        except: return None

    # Find interface residues (same logic as build_dataset)
    iface_b, iface_t = set(), set()
    for ri in binder_res_list:
        ca_i = get_ca(ri)
        if ca_i is None: continue
        for rj in target_res_list:
            ca_j = get_ca(rj)
            if ca_j is None: continue
            if np.linalg.norm(ca_i - ca_j) <= CONTACT_CUTOFF:
                iface_b.add(ri.get_id())
                iface_t.add(rj.get_id())

    iface_binder = [r for r in binder_res_list if r.get_id() in iface_b]
    iface_target = [r for r in target_res_list if r.get_id() in iface_t]
    all_iface = [(r, 0) for r in iface_binder] + [(r, 1) for r in iface_target]

    # Center coordinates
    coords = []
    for res, _ in all_iface:
        ca = get_ca(res)
        coords.append(ca if ca is not None else np.zeros(3))
    coords = np.array(coords)
    center = coords.mean(axis=0)
    coords -= center

    # Attention weights (pad/trim to match)
    aw = attn_weights.get("weights", np.ones(n_nodes) / n_nodes)
    if len(aw) != len(all_iface):
        aw = np.ones(len(all_iface)) / len(all_iface)

    # Normalise attention to [0,1]
    aw = (aw - aw.min()) / (aw.max() - aw.min() + 1e-8)

    # ── Build node list ───────────────────────────────────────────────────
    nodes = []
    for i, ((res, chain_label), coord, w) in enumerate(zip(all_iface, coords, aw)):
        burial = float(graph.x[i, 1280].cpu()) if i < graph.x.shape[0] else 0.5
        nodes.append({
            "id":          i,
            "x":           float(coord[0]),
            "y":           float(coord[1]),
            "z":           float(coord[2]),
            "chain":       binder_chain if chain_label == 0 else target_chain_id,
            "chain_label": int(chain_label),
            "resname":     THREE_TO_ONE.get(res.get_resname().strip(), "X"),
            "resnum":      int(res.get_id()[1]),
            "attn_weight": float(w),
            "burial":      float(burial),
        })

    # ── Build edge list ───────────────────────────────────────────────────
    EDGE_CUTOFF = 12.0
    edges = []
    edge_index = graph.edge_index.cpu().numpy()
    edge_attr  = graph.edge_attr.cpu().numpy()

    for ei in range(edge_index.shape[1]):
        src, dst = int(edge_index[0, ei]), int(edge_index[1, ei])
        ea = edge_attr[ei] if ei < len(edge_attr) else np.zeros(8)

        # Determine edge type from features
        is_hbond     = float(ea[2]) > 0.5
        is_salt      = float(ea[3]) > 0.5
        is_hydro     = float(ea[4]) > 0.5
        is_pi        = float(ea[5]) > 0.5
        is_cross     = float(ea[6]) > 0.5
        dist_norm    = float(ea[0])

        if is_hbond:     etype = "hbond"
        elif is_salt:    etype = "salt"
        elif is_pi:      etype = "pi"
        elif is_hydro:   etype = "hydrophobic"
        else:            etype = "contact"

        edges.append({
            "src":        src,
            "dst":        dst,
            "type":       etype,
            "cross_chain": bool(is_cross),
            "dist_norm":  dist_norm,
        })

    # ── Also extract full backbone for ribbon ────────────────────────────
    backbone = []
    for chain_id, res_list in [(binder_chain, binder_res_list),
                                (target_chain_id, target_res_list)]:
        chain_coords = []
        for res in res_list:
            ca = get_ca(res)
            if ca is not None:
                c = ca - center
                chain_coords.append([float(c[0]), float(c[1]), float(c[2])])
        if chain_coords:
            backbone.append({
                "chain": chain_id,
                "is_binder": chain_id == binder_chain,
                "coords": chain_coords,
            })

    result = {
        "meta": {
            "pdb":         pdb_path.stem,
            "binder_chain": binder_chain,
            "target_chain": target_chain_id,
            "n_nodes":     len(nodes),
            "n_edges":     len(edges),
            "log_koff":    round(log_koff, 4),
            "koff_pred":   float(f"{koff_pred:.4e}"),
            "tau_s":       round(1.0 / koff_pred),
            "tau_h":       round(1.0 / koff_pred / 3600, 2),
            "y_mean":      y_mean,
            "y_std":       y_std,
        },
        "nodes":    nodes,
        "edges":    edges,
        "backbone": backbone,
    }

    return result


def main(args):
    print(f"Extracting interface from {args.pdb} chain {args.chain}...")
    data = extract_viz_data(args.pdb, args.chain)
    print(f"  {data['meta']['n_nodes']} interface nodes")
    print(f"  {data['meta']['n_edges']} edges")
    print(f"  Predicted koff = {data['meta']['koff_pred']} s⁻¹")
    print(f"  Residence time = {data['meta']['tau_h']} h")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2))
    print(f"  Saved → {out_path}")

    if args.open:
        viewer = Path(__file__).parent / "viewer.html"
        if viewer.exists():
            webbrowser.open(f"file://{viewer.resolve()}")

    return data


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pdb",   required=True)
    p.add_argument("--chain", default="A")
    p.add_argument("--out",   default="results/viz/interface.json")
    p.add_argument("--open",  action="store_true")
    main(p.parse_args())

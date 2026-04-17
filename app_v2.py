"""
app_v2.py — koffGNN Unified Research Interface
================================================
Single Flask app combining:
  1. Generative MCMC design
  2. Embedded 3D interface viewer
  3. Mutation impact scanner (heatmap)
  4. Calibration / uncertainty display

Run:
    python app_v2.py
    Open http://localhost:5000
"""

import json, math, os, queue, sys, threading, time, uuid
from pathlib import Path

import numpy as np
import requests
from flask import Flask, Response, jsonify, render_template, request, send_file

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

app = Flask(__name__, template_folder=str(ROOT / "templates_v2"))
app.config["PDB_DIR"]     = ROOT / "data" / "raw" / "pdb"
app.config["RESULTS_DIR"] = ROOT / "results" / "web"
app.config["PDB_DIR"].mkdir(parents=True, exist_ok=True)
app.config["RESULTS_DIR"].mkdir(parents=True, exist_ok=True)

_job_queues  = {}
_job_results = {}
_models      = {}

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# ══════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════

def get_models():
    if _models: return _models
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from esm2_model import KoffGNNEsm
    ckpt = torch.load(ROOT / "checkpoints" / "koff_gnn_esm2_best.pt",
                      map_location=device, weights_only=False)
    koffgnn = KoffGNNEsm(hidden_dim=ckpt.get("hidden_dim",256),
                          n_layers=ckpt.get("n_layers",4), dropout=0.0).to(device)
    koffgnn.load_state_dict(ckpt["model_state"])
    koffgnn.eval()
    import esm
    esm_m, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_m = esm_m.eval().to(device)
    _models.update({"koffgnn": koffgnn, "esm": esm_m, "alphabet": alphabet,
                    "y_mean": ckpt["y_mean"], "y_std": ckpt["y_std"], "device": device})
    return _models


def get_esm2_emb(seq, esm_m, bc, device):
    import torch
    _, _, tokens = bc([("s", seq)])
    with torch.no_grad():
        out = esm_m(tokens.to(device), repr_layers=[33])
    return out["representations"][33][0, 1:len(seq)+1].cpu()


def score_graph(graph, koffgnn, y_mean, y_std, device):
    import torch
    g = graph.clone().to(device)
    g.batch = torch.zeros(g.x.size(0), dtype=torch.long, device=device)
    with torch.no_grad():
        pm, plv = koffgnn(g)
    lk  = (pm.item() * y_std) + y_mean
    lv  = plv.item()
    std_lk = math.exp(lv / 2) * y_std
    return lk, std_lk


def build_esm2_graph(pdb_path, binder_chain, esm_m, bc, device):
    """Build interface graph with ESM-2 node features."""
    import torch
    from build_dataset import pdb_to_graph, THREE_TO_ONE
    from Bio.PDB import PDBParser

    graph = pdb_to_graph(Path(pdb_path), koff=1e-3)
    if graph is None: raise ValueError("Could not parse interface")

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("x", str(pdb_path))
    chains = {}
    for c in struct[0].get_chains():
        seq = "".join(THREE_TO_ONE.get(r.get_resname().strip(),"X")
                      for r in c.get_residues() if r.get_id()[0]==" ")
        if seq: chains[c.get_id()] = seq

    binder_seq = chains.get(binder_chain, "")
    tgt_chain  = next((c for c in chains if c != binder_chain), None)
    target_seq = chains.get(tgt_chain, "")

    binder_emb = get_esm2_emb(binder_seq, esm_m, bc, device)
    target_emb = get_esm2_emb(target_seq, esm_m, bc, device)

    chain_labels = graph.x[:, 29]
    struct_feats = graph.x[:, 20:]
    n_nodes = graph.x.shape[0]
    esm_feat = torch.zeros(n_nodes, 1280)
    b_idx = t_idx = 0
    binder_node_indices = []
    for ni in range(n_nodes):
        if chain_labels[ni] < 0.5:
            if b_idx < binder_emb.shape[0]: esm_feat[ni] = binder_emb[b_idx]; b_idx += 1
            binder_node_indices.append(ni)
        else:
            if t_idx < target_emb.shape[0]: esm_feat[ni] = target_emb[t_idx]; t_idx += 1

    graph.x = torch.cat([esm_feat, struct_feats], dim=-1)
    return graph, binder_seq, target_seq, tgt_chain, binder_node_indices


# ══════════════════════════════════════════════════════════════════════════
# Routes — Structure
# ══════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index_v2.html")


@app.route("/chains", methods=["POST"])
def get_chains():
    pdb_path = None
    if "pdb_file" in request.files and request.files["pdb_file"].filename:
        f = request.files["pdb_file"]
        pdb_path = app.config["PDB_DIR"] / f.filename
        f.save(str(pdb_path))
    elif request.form.get("pdb_id"):
        pid = request.form["pdb_id"].strip().lower()
        pdb_path = app.config["PDB_DIR"] / f"{pid}.pdb"
        if not pdb_path.exists():
            r = requests.get(f"https://files.rcsb.org/download/{pid.upper()}.pdb", timeout=30)
            if r.status_code != 200:
                return jsonify({"error": f"PDB {pid} not found"}), 404
            pdb_path.write_bytes(r.content)
    else:
        return jsonify({"error": "Provide PDB file or ID"}), 400

    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("x", str(pdb_path))
    chains = {}
    for c in struct[0].get_chains():
        res = [r for r in c.get_residues() if r.get_id()[0]==" "]
        if res: chains[c.get_id()] = len(res)
    return jsonify({"chains": chains, "pdb_path": str(pdb_path), "pdb_name": pdb_path.name})


# ══════════════════════════════════════════════════════════════════════════
# Route — Interface data for 3D viewer
# ══════════════════════════════════════════════════════════════════════════

@app.route("/interface", methods=["POST"])
def get_interface():
    """Extract interface graph + koffGNN attention weights for 3D viewer."""
    import torch
    data         = request.json
    pdb_path     = data["pdb_path"]
    binder_chain = data["binder_chain"]

    m = get_models()
    device, koffgnn = m["device"], m["koffgnn"]
    esm_m, bc = m["esm"], m["alphabet"].get_batch_converter()
    y_mean, y_std = m["y_mean"], m["y_std"]

    from build_dataset import pdb_to_graph, THREE_TO_ONE
    from Bio.PDB import PDBParser

    graph, binder_seq, target_seq, tgt_chain, binder_node_indices = \
        build_esm2_graph(pdb_path, binder_chain, esm_m, bc, device)

    # Attention hook
    attn_w = {}
    def hook(module, inp, out):
        from torch_geometric.utils import softmax as pyg_softmax
        scores = module.attn(inp[0])
        w = pyg_softmax(scores, inp[1])
        attn_w["w"] = w.detach().cpu().squeeze().numpy()
    h = m["koffgnn"].attn_pool.register_forward_hook(hook)

    lk, std_lk = score_graph(graph, koffgnn, y_mean, y_std, device)
    h.remove()

    aw = attn_w.get("w", np.ones(graph.x.shape[0]))
    aw = (aw - aw.min()) / (aw.max() - aw.min() + 1e-8)

    # Get 3D coordinates
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("x", str(pdb_path))
    chain_residues = {}
    for c in struct[0].get_chains():
        res = [r for r in c.get_residues() if r.get_id()[0]==" "]
        if res: chain_residues[c.get_id()] = res

    def get_ca(res):
        try: return res["CA"].get_vector().get_array()
        except: return None

    br = chain_residues.get(binder_chain, [])
    tr = chain_residues.get(tgt_chain, [])

    iface_b, iface_t = set(), set()
    for ri in br:
        ca_i = get_ca(ri)
        if ca_i is None: continue
        for rj in tr:
            ca_j = get_ca(rj)
            if ca_j is None: continue
            if np.linalg.norm(ca_i - ca_j) <= 8.0:
                iface_b.add(ri.get_id()); iface_t.add(rj.get_id())

    all_iface = [(r, 0) for r in br if r.get_id() in iface_b] + \
                [(r, 1) for r in tr if r.get_id() in iface_t]
    coords = np.array([get_ca(r) if get_ca(r) is not None else np.zeros(3) for r, _ in all_iface])
    center = coords.mean(0)
    coords -= center

    nodes = []
    for i, ((res, cl), coord) in enumerate(zip(all_iface, coords)):
        burial = float(graph.x[i, 1280].cpu()) if i < graph.x.shape[0] else 0.5
        nodes.append({
            "id": i, "x": float(coord[0]), "y": float(coord[1]), "z": float(coord[2]),
            "chain": binder_chain if cl == 0 else tgt_chain, "chain_label": int(cl),
            "resname": THREE_TO_ONE.get(res.get_resname().strip(), "X"),
            "resnum": int(res.get_id()[1]),
            "attn_weight": float(aw[i]) if i < len(aw) else 0.0,
            "burial": float(burial),
            "local_idx": i,
        })

    ei = graph.edge_index.cpu().numpy()
    ea = graph.edge_attr.cpu().numpy()
    edges = []
    for k in range(ei.shape[1]):
        src, dst = int(ei[0,k]), int(ei[1,k])
        e = ea[k] if k < len(ea) else np.zeros(8)
        t = "hbond" if e[2]>0.5 else "salt" if e[3]>0.5 else \
            "pi" if e[5]>0.5 else "hydrophobic" if e[4]>0.5 else "contact"
        edges.append({"src":src,"dst":dst,"type":t,
                      "cross_chain":bool(e[6]>0.5),"dist_norm":float(e[0])})

    backbone = []
    for cid, rlist in [(binder_chain, br), (tgt_chain, tr)]:
        cc = []
        for res in rlist:
            ca = get_ca(res)
            if ca is not None:
                c = ca - center
                cc.append([float(c[0]),float(c[1]),float(c[2])])
        if cc: backbone.append({"chain":cid,"is_binder":cid==binder_chain,"coords":cc})

    return jsonify({
        "meta": {
            "pdb": Path(pdb_path).stem, "binder_chain": binder_chain,
            "target_chain": tgt_chain, "n_nodes": len(nodes), "n_edges": len(edges),
            "log_koff": round(lk,4), "koff_pred": float(f"{10**lk:.4e}"),
            "tau_s": round(1/(10**lk)), "tau_h": round(1/(10**lk)/3600, 2),
            "std_log_koff": round(std_lk, 4),
            "binder_seq": binder_seq, "target_seq": target_seq,
        },
        "nodes": nodes, "edges": edges, "backbone": backbone,
    })


# ══════════════════════════════════════════════════════════════════════════
# Route — Mutation scanner
# ══════════════════════════════════════════════════════════════════════════

@app.route("/mutant_scan", methods=["POST"])
def mutant_scan():
    """
    For a given residue position in the binder, compute predicted koff
    for all 20 amino acid substitutions.
    Returns a 20-element list of {aa, log_koff, delta_log_koff}.
    """
    import torch
    data         = request.json
    pdb_path     = data["pdb_path"]
    binder_chain = data["binder_chain"]
    node_idx     = int(data["node_idx"])   # index in binder interface nodes

    m = get_models()
    device, koffgnn = m["device"], m["koffgnn"]
    esm_m, bc = m["esm"], m["alphabet"].get_batch_converter()
    y_mean, y_std = m["y_mean"], m["y_std"]

    graph, binder_seq, target_seq, tgt_chain, binder_node_indices = \
        build_esm2_graph(pdb_path, binder_chain, esm_m, bc, device)

    # Baseline koff
    base_lk, _ = score_graph(graph, koffgnn, y_mean, y_std, device)

    if node_idx >= len(binder_node_indices):
        return jsonify({"error": "node_idx out of range"}), 400

    graph_node_idx = binder_node_indices[node_idx]
    seq_pos        = node_idx  # approximate: local binder interface index

    results = []
    original_aa = binder_seq[seq_pos] if seq_pos < len(binder_seq) else "X"

    for aa in AMINO_ACIDS:
        # Mutate sequence at this position
        mutant_seq = binder_seq[:seq_pos] + aa + binder_seq[seq_pos+1:]

        # Get new ESM-2 embedding for mutant binder
        mut_emb = get_esm2_emb(mutant_seq, esm_m, bc, device)

        # Reconstruct graph with mutant embedding at this node
        mut_graph = graph.clone()
        if seq_pos < mut_emb.shape[0]:
            mut_graph.x[graph_node_idx, :1280] = mut_emb[seq_pos].to(device)

        lk, std = score_graph(mut_graph, koffgnn, y_mean, y_std, device)
        results.append({
            "aa":            aa,
            "log_koff":      round(lk, 4),
            "koff":          float(f"{10**lk:.4e}"),
            "delta_log_koff": round(lk - base_lk, 4),
            "is_original":   aa == original_aa,
            "tau_h":         round(1/(10**lk)/3600, 2),
        })

    results.sort(key=lambda x: x["log_koff"])

    return jsonify({
        "node_idx":    node_idx,
        "original_aa": original_aa,
        "seq_pos":     seq_pos,
        "base_log_koff": round(base_lk, 4),
        "mutations":   results,
    })


# ══════════════════════════════════════════════════════════════════════════
# Route — MCMC Design (streaming)
# ══════════════════════════════════════════════════════════════════════════

def run_mcmc(job_id, pdb_path, binder_chain, target_log_koff, n_chains, n_steps):
    q = _job_queues[job_id]
    def emit(ev, d): q.put({"event": ev, "data": d})

    try:
        import torch
        m = get_models()
        device, koffgnn = m["device"], m["koffgnn"]
        esm_m, bc = m["esm"], m["alphabet"].get_batch_converter()
        y_mean, y_std = m["y_mean"], m["y_std"]

        emit("status", {"msg": "Building interface graph..."})
        graph, binder_seq, target_seq, tgt_chain, binder_node_indices = \
            build_esm2_graph(pdb_path, binder_chain, esm_m, bc, device)
        emit("info", {"n_binder_nodes": len(binder_node_indices),
                      "binder_seq": binder_seq})

        IDX_TO_AA = {i: aa for i, aa in enumerate(AMINO_ACIDS)}
        AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

        base_graph = graph.clone().to(device)
        bi = binder_node_indices
        n_binder = len(bi)

        def score_chain(seq_list):
            seq_str = "".join(seq_list[:len(binder_seq)])
            emb = get_esm2_emb(seq_str, esm_m, bc, device)
            new_x = base_graph.x.clone()
            for li, ni in enumerate(bi):
                if li < emb.shape[0]: new_x[ni, :1280] = emb[li].to(device)
            g = base_graph.clone(); g.x = new_x
            g.batch = torch.zeros(new_x.size(0), dtype=torch.long, device=device)
            with torch.no_grad(): pm, plv = koffgnn(g)
            lk = (pm.item() * y_std) + y_mean
            return lk, (lk - target_log_koff)**2

        # Initialise chains
        chains = []
        for _ in range(n_chains):
            seq = list(binder_seq)
            for i in range(len(seq)):
                if np.random.rand() < 0.3:
                    seq[i] = IDX_TO_AA[np.random.randint(20)]
            chains.append(seq)

        chain_lk = []; chain_e = []
        for c in chains:
            lk, e = score_chain(c); chain_lk.append(lk); chain_e.append(e)

        T_start, T_end = 2.0, 0.05
        accept_counts = [0]*n_chains

        emit("status", {"msg": f"Running {n_chains} MCMC chains × {n_steps} steps..."})

        for step in range(n_steps):
            frac = step / max(n_steps-1, 1)
            T    = T_start * (T_end/T_start)**frac

            for c in range(n_chains):
                pos    = np.random.randint(min(n_binder, len(chains[c])))
                old_aa = chains[c][pos]
                new_aa = IDX_TO_AA[np.random.randint(20)]
                while new_aa == old_aa: new_aa = IDX_TO_AA[np.random.randint(20)]

                chains[c][pos] = new_aa
                new_lk, new_e  = score_chain(chains[c])
                delta_e = new_e - chain_e[c]
                accept  = delta_e < 0 or np.random.rand() < math.exp(max(-delta_e/T, -500))
                if accept:
                    chain_e[c] = new_e; chain_lk[c] = new_lk; accept_counts[c] += 1
                else:
                    chains[c][pos] = old_aa

            if (step+1) % 10 == 0:
                emit("progress", {
                    "step": step+1, "n_steps": n_steps,
                    "chains": [round(lk,4) for lk in chain_lk],
                    "target": target_log_koff, "T": round(T,4),
                    "pct": round(100*(step+1)/n_steps),
                })

        # Collect results
        all_results = []
        for c in range(n_chains):
            lk  = chain_lk[c]
            seq = "".join(chains[c])
            all_results.append({
                "chain": c+1, "sequence": seq,
                "log10_koff": round(lk,4), "koff_pred": float(f"{10**lk:.4e}"),
                "delta_log_koff": round(abs(lk - target_log_koff), 4),
                "residence_time_h": round(1/(10**lk)/3600, 2),
                "in_target_range": abs(lk - target_log_koff) < 0.5,
                "accept_rate": round(accept_counts[c]/n_steps, 3),
            })
        all_results.sort(key=lambda x: x["delta_log_koff"])

        import pandas as pd
        out = app.config["RESULTS_DIR"] / f"{job_id}.csv"
        pd.DataFrame(all_results).to_csv(out, index=False)

        _job_results[job_id] = {
            "results": all_results, "csv_path": str(out),
            "n_hits": sum(1 for r in all_results if r["in_target_range"]),
            "n_total": len(all_results),
            "best_seq": all_results[0]["sequence"] if all_results else "",
            "best_koff": all_results[0]["koff_pred"] if all_results else None,
            "best_delta": all_results[0]["delta_log_koff"] if all_results else None,
            "best_tau_h": all_results[0]["residence_time_h"] if all_results else None,
            "pdb_path": pdb_path, "binder_chain": binder_chain,
        }
        emit("done", _job_results[job_id])

    except Exception as ex:
        import traceback
        emit("error", {"msg": str(ex), "trace": traceback.format_exc()})
    finally:
        q.put(None)


@app.route("/design", methods=["POST"])
def start_design():
    data = request.json
    job_id = str(uuid.uuid4())[:8]
    _job_queues[job_id] = queue.Queue()
    threading.Thread(target=run_mcmc, args=(
        job_id, data["pdb_path"], data["binder_chain"],
        math.log10(float(data["target_koff"])),
        int(data.get("n_sequences", 5)),
        int(data.get("n_steps", 200)),
    ), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/stream/<job_id>")
def stream(job_id):
    def generate():
        q = _job_queues.get(job_id)
        if not q:
            yield f"data: {json.dumps({'event':'error','data':{'msg':'Job not found'}})}\n\n"
            return
        while True:
            try:
                item = q.get(timeout=60)
                if item is None: yield f"data: {json.dumps({'event':'end'})}\n\n"; break
                yield f"data: {json.dumps(item)}\n\n"
            except queue.Empty:
                yield f"data: {json.dumps({'event':'heartbeat'})}\n\n"
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})


@app.route("/download/<job_id>")
def download(job_id):
    r = _job_results.get(job_id)
    if not r: return "Not found", 404
    return send_file(r["csv_path"], as_attachment=True,
                     download_name=f"koffgnn_{job_id}.csv")


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  koffGNN Unified Research Interface v2")
    print("  Open http://localhost:5000")
    print("="*55 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)

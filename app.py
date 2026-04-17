"""
app.py — koffGNN Web Interface
================================
Local Flask server. Run with:
    python app.py
Then open http://localhost:5000 in your browser.

Features:
  - Upload PDB file or enter PDB ID (auto-downloads from RCSB)
  - Select binder chain, target koff, and design method
  - Real-time streaming progress via Server-Sent Events
  - Live koff convergence plot
  - Results table with sequences and predicted residence times
  - CSV download
"""

import json
import math
import os
import queue
import sys
import threading
import time
import uuid
from pathlib import Path

import requests
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_file,
)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = ROOT / "data" / "raw" / "pdb"
app.config["RESULTS_FOLDER"] = ROOT / "results" / "web"
app.config["UPLOAD_FOLDER"].mkdir(parents=True, exist_ok=True)
app.config["RESULTS_FOLDER"].mkdir(parents=True, exist_ok=True)

# Per-job progress queues and results store
_job_queues: dict[str, queue.Queue] = {}
_job_results: dict[str, dict]       = {}
_job_status:  dict[str, str]        = {}


# ══════════════════════════════════════════════════════════════════════════
# Lazy model loading (load once, reuse)
# ══════════════════════════════════════════════════════════════════════════

_models = {}

def get_models():
    if _models:
        return _models

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from esm2_model import KoffGNNEsm
    ckpt = torch.load(
        ROOT / "checkpoints" / "koff_gnn_esm2_best.pt",
        map_location=device, weights_only=False
    )
    koffgnn = KoffGNNEsm(
        hidden_dim=ckpt.get("hidden_dim", 256),
        n_layers=ckpt.get("n_layers", 4),
        dropout=0.0,
    ).to(device)
    koffgnn.load_state_dict(ckpt["model_state"])
    koffgnn.eval()

    import esm
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model = esm_model.eval().to(device)

    _models.update({
        "koffgnn":  koffgnn,
        "esm":      esm_model,
        "alphabet": alphabet,
        "y_mean":   ckpt["y_mean"],
        "y_std":    ckpt["y_std"],
        "device":   device,
    })
    return _models


# ══════════════════════════════════════════════════════════════════════════
# Background design thread
# ══════════════════════════════════════════════════════════════════════════

def run_design(job_id: str, pdb_path: str, binder_chain: str,
               target_koff: float, method: str,
               n_sequences: int, n_steps: int):
    """Runs in a background thread. Emits progress events to the job queue."""

    q = _job_queues[job_id]

    def emit(event_type: str, data: dict):
        q.put({"event": event_type, "data": data})

    try:
        import torch
        from koff_generative import (
            build_base_graph,
            get_esm2_embedding,
            mckoff,
            precompute_aa_embeddings,
        )

        m       = get_models()
        device  = m["device"]
        koffgnn = m["koffgnn"]
        esm_m   = m["esm"]
        alph    = m["alphabet"]
        bc      = alph.get_batch_converter()
        y_mean  = m["y_mean"]
        y_std   = m["y_std"]

        emit("status", {"msg": "Loading amino acid embedding matrix..."})
        AA_EMB = precompute_aa_embeddings(esm_m, bc, device)

        emit("status", {"msg": f"Parsing interface from {Path(pdb_path).name}..."})
        base = build_base_graph(
            Path(pdb_path), binder_chain, esm_m, bc, device
        )
        emit("info", {
            "binder_residues": base["n_binder_nodes"],
            "binder_seq_len":  len(base["binder_seq"]),
            "target_seq_len":  len(base["target_seq"]),
        })

        target_log_koff = math.log10(target_koff)
        all_results = []

        # ── MCMC ──────────────────────────────────────────────────────────
        if method in ("mcmc", "both"):
            emit("status", {"msg": f"Running MCMCKoff: {n_sequences} chains × {n_steps} steps"})

            esm_model_mcmc, alphabet_mcmc = esm_m, alph
            bc_mcmc = bc
            bi      = base["binder_node_indices"].tolist()
            base_graph = base["graph"].clone().to(device)
            n_binder   = base["n_binder_nodes"]

            import numpy as np
            from build_dataset import THREE_TO_ONE

            AA_TO_IDX_L = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
            IDX_TO_AA_L = {i: aa for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}

            binder_list = list(base["binder_seq"])
            chains = []
            for _ in range(n_sequences):
                seq = binder_list.copy()
                for i in range(len(seq)):
                    if np.random.rand() < 0.3:
                        seq[i] = IDX_TO_AA_L[np.random.randint(20)]
                chains.append(seq)

            def score_chain(seq_list):
                seq_str    = "".join(seq_list[:len(base["binder_seq"])])
                binder_emb = get_esm2_embedding(seq_str, esm_model_mcmc, bc_mcmc, device)
                new_x      = base_graph.x.clone()
                for li, ni in enumerate(bi):
                    if li < binder_emb.shape[0]:
                        new_x[ni, :1280] = binder_emb[li].to(device)
                g       = base_graph.clone()
                g.x     = new_x
                g.batch = torch.zeros(new_x.size(0), dtype=torch.long, device=device)
                with torch.no_grad():
                    pm, _ = koffgnn(g)
                lk = (pm.item() * y_std) + y_mean
                return lk, (lk - target_log_koff) ** 2

            koffgnn.eval()
            chain_lk  = []
            chain_e   = []
            for c in chains:
                lk, e = score_chain(c)
                chain_lk.append(lk)
                chain_e.append(e)

            T_start, T_end = 2.0, 0.05
            accept_counts  = [0] * n_sequences
            step_log       = []

            for step in range(n_steps):
                frac = step / max(n_steps - 1, 1)
                T    = T_start * (T_end / T_start) ** frac

                for c in range(n_sequences):
                    pos    = np.random.randint(min(n_binder, len(chains[c])))
                    old_aa = chains[c][pos]
                    new_aa = IDX_TO_AA_L[np.random.randint(20)]
                    while new_aa == old_aa:
                        new_aa = IDX_TO_AA_L[np.random.randint(20)]

                    chains[c][pos] = new_aa
                    new_lk, new_e  = score_chain(chains[c])
                    delta_e = new_e - chain_e[c]
                    accept  = delta_e < 0 or np.random.rand() < math.exp(
                        max(-delta_e / T, -500)
                    )
                    if accept:
                        chain_e[c]  = new_e
                        chain_lk[c] = new_lk
                        accept_counts[c] += 1
                    else:
                        chains[c][pos] = old_aa

                # Emit progress every 10 steps
                if (step + 1) % 10 == 0:
                    step_log.append({
                        "step":      step + 1,
                        "chain_lk":  [round(lk, 4) for lk in chain_lk],
                        "target":    target_log_koff,
                        "T":         round(T, 4),
                    })
                    emit("progress", {
                        "step":    step + 1,
                        "n_steps": n_steps,
                        "chains":  [round(lk, 4) for lk in chain_lk],
                        "target":  target_log_koff,
                        "T":       round(T, 4),
                        "pct":     round(100 * (step + 1) / n_steps),
                    })

            for c in range(n_sequences):
                lk  = chain_lk[c]
                seq = "".join(chains[c])
                all_results.append({
                    "method":          "MCMCKoff",
                    "chain":           c + 1,
                    "sequence":        seq,
                    "log10_koff":      round(lk, 4),
                    "koff_pred":       round(10 ** lk, 6),
                    "delta_log_koff":  round(abs(lk - target_log_koff), 4),
                    "residence_time_s": round(1.0 / (10 ** lk)),
                    "residence_time_h": round(1.0 / (10 ** lk) / 3600, 2),
                    "in_target_range": abs(lk - target_log_koff) < 0.5,
                    "accept_rate":     round(accept_counts[c] / n_steps, 3),
                })

        # Sort by delta_log_koff
        all_results.sort(key=lambda x: x["delta_log_koff"])

        # Save CSV
        import pandas as pd
        out_path = app.config["RESULTS_FOLDER"] / f"{job_id}.csv"
        pd.DataFrame(all_results).to_csv(out_path, index=False)

        _job_results[job_id] = {
            "results":        all_results,
            "n_hits":         sum(1 for r in all_results if r["in_target_range"]),
            "n_total":        len(all_results),
            "best_delta":     all_results[0]["delta_log_koff"] if all_results else None,
            "best_koff":      all_results[0]["koff_pred"] if all_results else None,
            "best_tau":       all_results[0]["residence_time_s"] if all_results else None,
            "target_koff":    target_koff,
            "target_log":     target_log_koff,
            "csv_path":       str(out_path),
        }
        _job_status[job_id] = "done"
        emit("done", _job_results[job_id])

    except Exception as ex:
        import traceback
        _job_status[job_id] = "error"
        emit("error", {"msg": str(ex), "trace": traceback.format_exc()})

    finally:
        q.put(None)  # sentinel


# ══════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chains", methods=["POST"])
def get_chains():
    """Return chain IDs and lengths for a PDB."""
    pdb_path = None

    if "pdb_file" in request.files and request.files["pdb_file"].filename:
        f        = request.files["pdb_file"]
        pdb_path = app.config["UPLOAD_FOLDER"] / f.filename
        f.save(str(pdb_path))
    elif request.form.get("pdb_id"):
        pdb_id   = request.form["pdb_id"].strip().lower()
        pdb_path = app.config["UPLOAD_FOLDER"] / f"{pdb_id}.pdb"
        if not pdb_path.exists():
            r = requests.get(
                f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb",
                timeout=30
            )
            if r.status_code != 200:
                return jsonify({"error": f"PDB {pdb_id} not found on RCSB"}), 404
            pdb_path.write_bytes(r.content)
    else:
        return jsonify({"error": "Provide a PDB file or PDB ID"}), 400

    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    struct  = parser.get_structure("x", str(pdb_path))
    chains  = {}
    for c in struct[0].get_chains():
        res = [r for r in c.get_residues() if r.get_id()[0] == " "]
        if res:
            chains[c.get_id()] = len(res)

    return jsonify({
        "chains":   chains,
        "pdb_path": str(pdb_path),
        "pdb_name": pdb_path.name,
    })


@app.route("/design", methods=["POST"])
def start_design():
    """Start a design job. Returns job_id."""
    data         = request.json
    job_id       = str(uuid.uuid4())[:8]
    pdb_path     = data["pdb_path"]
    binder_chain = data["binder_chain"]
    target_koff  = float(data["target_koff"])
    method       = data.get("method", "mcmc")
    n_sequences  = int(data.get("n_sequences", 5))
    n_steps      = int(data.get("n_steps", 200))

    _job_queues[job_id] = queue.Queue()
    _job_status[job_id] = "running"

    t = threading.Thread(
        target=run_design,
        args=(job_id, pdb_path, binder_chain, target_koff,
              method, n_sequences, n_steps),
        daemon=True,
    )
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/stream/<job_id>")
def stream(job_id):
    """Server-Sent Events stream for real-time progress."""
    def generate():
        q = _job_queues.get(job_id)
        if q is None:
            yield f"data: {json.dumps({'event':'error','data':{'msg':'Job not found'}})}\n\n"
            return
        while True:
            try:
                item = q.get(timeout=60)
                if item is None:
                    yield f"data: {json.dumps({'event':'end'})}\n\n"
                    break
                yield f"data: {json.dumps(item)}\n\n"
            except queue.Empty:
                yield f"data: {json.dumps({'event':'heartbeat'})}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@app.route("/download/<job_id>")
def download(job_id):
    result = _job_results.get(job_id)
    if not result:
        return "Job not found", 404
    return send_file(result["csv_path"], as_attachment=True,
                     download_name=f"koffgnn_{job_id}.csv")


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  koffGNN Web Interface")
    print("  Open http://localhost:5000 in your browser")
    print("="*55 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)

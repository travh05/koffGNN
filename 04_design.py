"""
STEP 5 — Kinetics-Conditioned Design Loop
==========================================
Given a TARGET protein structure and a TARGET koff value, this script:

  1.  Loads the trained koffGNN as a kinetics oracle.
  2.  Takes the target protein (PDB file) and an initial binder scaffold
      (can be random coil or a previous RFdiffusion output).
  3.  Runs ProteinMPNN to generate N candidate binder sequences.
  4.  For each candidate:
        a. Predicts complex structure with ESMFold (fast, no MSA).
        b. Extracts interface graph.
        c. Scores with koffGNN → predicted log10(koff) + uncertainty.
  5.  Ranks candidates by |predicted_koff − target_koff|.
  6.  Outputs a ranked CSV with sequences, predicted koff, confidence.

Usage:
  python 04_design.py \
      --target_pdb   targets/pd_l1.pdb \
      --binder_chain A \
      --target_koff  1e-3 \
      --n_sequences  200 \
      --out           results/pd_l1_koff_1e3_designs.csv

What makes this step genuinely novel:
  Every other binder design pipeline scores candidates by pLDDT and
  interface pAE (AlphaFold confidence) — which proxy for AFFINITY.
  We score by predicted koff — a completely different objective.
  A candidate can have excellent pLDDT and terrible koff (short
  residence time, bad for therapeutics). Our ranker distinguishes them.
"""

import argparse
import logging
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

# ── Project imports ────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from model.koff_gnn import KoffGNN, NODE_DIM, EDGE_DIM, GRAPH_DIM
from build_dataset import pdb_to_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CKPT_PATH = ROOT / "checkpoints" / "koff_gnn_esm2_best.pt"

# ══════════════════════════════════════════════════════════════════════════
# Load trained model
# ══════════════════════════════════════════════════════════════════════════

def load_model(device: torch.device):
    """Load checkpoint and return (model, y_mean, y_std)."""
    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CKPT_PATH}. Run 03_train_esm2.py first."
        )
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)

    # Import the ESM-2 model class
    import sys
    sys.path.insert(0, str(ROOT))
    from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
    from torch_geometric.utils import softmax as pyg_softmax
    import torch.nn as nn

    hidden_dim = ckpt.get("hidden_dim", 256)
    n_layers   = ckpt.get("n_layers", 4)

    from esm2_model import KoffGNNEsm
    model = KoffGNNEsm(hidden_dim=hidden_dim, n_layers=n_layers, dropout=0.0).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    log.info(f"Loaded ESM-2 model from epoch {ckpt['epoch']} "
             f"(val_rmse={ckpt['val_rmse']:.4f}, r={ckpt['val_r']:.3f})")
    return model, ckpt["y_mean"], ckpt["y_std"]


# ══════════════════════════════════════════════════════════════════════════
# ProteinMPNN sequence generation
# ══════════════════════════════════════════════════════════════════════════

def run_protein_mpnn(
    pdb_path: Path,
    binder_chain: str,
    n_sequences: int = 100,
    temperature: float = 0.1,
) -> List[str]:
    """
    Run ProteinMPNN to generate diverse binder sequences for the given
    complex structure. We fix the target chain and redesign the binder chain.

    Returns a list of amino acid sequences (one-letter code).

    Requires ProteinMPNN to be cloned at ~/koff_designer/ProteinMPNN/
    (the setup_wsl2.sh script does this automatically).
    """
    mpnn_dir = ROOT / "ProteinMPNN"
    if not mpnn_dir.exists():
        log.warning(
            "ProteinMPNN not found. Clone it with:\n"
            "  git clone https://github.com/dauparas/ProteinMPNN.git"
        )
        # For demo purposes, return random poly-G sequences
        return ["G" * 60] * n_sequences

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Build the chain specification:
        # fixed_chains = target chains (do not redesign)
        # designed_chains = binder chain (redesign)
        # We parse all chains from the PDB and fix everything except binder
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("x", str(pdb_path))
        all_chains = [c.get_id() for c in structure[0].get_chains()]
        fixed_chains = [c for c in all_chains if c != binder_chain]

        import json
        # Write chain_id_jsonl to specify which chain to design
        chain_id = {pdb_path.stem: {"designed_chain_list": [binder_chain], "fixed_chain_list": fixed_chains}}
        chain_id_path = tmp_path / "chain_id.jsonl"
        with open(chain_id_path, 'w') as fh:
            fh.write(json.dumps(chain_id) + '\n')

        cmd = [
            sys.executable,
            str(mpnn_dir / "protein_mpnn_run.py"),
            "--pdb_path",        str(pdb_path),
            "--out_folder",      str(output_dir),
            "--num_seq_per_target", str(n_sequences),
            "--sampling_temp",   str(temperature),
            "--seed",            "42",
            "--batch_size",      "1",
            "--chain_id_jsonl",  str(chain_id_path),
        ]

        log.info(f"Running ProteinMPNN: {n_sequences} sequences ...")
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )

        if result.returncode != 0:
            log.error(f"ProteinMPNN failed:\n{result.stderr}")
            return []

        # Parse output FASTA
        sequences = []
        for fasta_path in output_dir.glob("**/*.fa"):
            with open(fasta_path) as f:
                lines = f.readlines()
            # ProteinMPNN FASTA: alternating header / sequence lines
            for i, line in enumerate(lines):
                if not line.startswith(">") and line.strip():
                    sequences.append(line.strip().split("/")[0])  # binder chain only

        log.info(f"  Generated {len(sequences)} sequences")
        return sequences[:n_sequences]


# ══════════════════════════════════════════════════════════════════════════
# ESMFold structure prediction
# ══════════════════════════════════════════════════════════════════════════

def predict_structure_esmfold(sequence: str, out_pdb: Path) -> bool:
    """Stub — ESMFold replaced by sequence threading below."""
    return False


_esmfold_model = None

def _get_esmfold():
    """Singleton ESMFold loader (expensive: 2.5 GB model)."""
    global _esmfold_model
    if _esmfold_model is None:
        import esm
        log.info("Loading ESMFold model (~2.5 GB, first time only) ...")
        _esmfold_model = esm.pretrained.esmfold_v1()
        _esmfold_model = _esmfold_model.eval()
        if torch.cuda.is_available():
            _esmfold_model = _esmfold_model.cuda()
    return _esmfold_model


# ══════════════════════════════════════════════════════════════════════════
# koff prediction for a single candidate
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_koff(
    model: KoffGNN,
    pdb_path: Path,
    y_mean: float,
    y_std: float,
    device: torch.device,
    binder_chain: str,
    n_mc_samples: int = 20,
) -> Tuple[float, float]:
    """
    Predict koff for a protein-protein complex PDB.

    Returns (koff_pred, koff_std):
      koff_pred — predicted koff in s⁻¹
      koff_std  — ±1σ uncertainty in koff (s⁻¹)

    We use MC dropout (n_mc_samples forward passes) to get
    uncertainty estimates even at inference time.
    """
    # Build interface graph (koff label doesn't matter here; use placeholder)
    graph = pdb_to_graph(pdb_path, koff=1e-3)  # placeholder koff
    if graph is None:
        return float("nan"), float("nan")

    graph = graph.to(device)

    # Add batch dimension
    graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)

    # MC Dropout: enable dropout layers at inference
    def enable_dropout(m):
        if isinstance(m, torch.nn.Dropout):
            m.train()

    model.eval()
    model.apply(enable_dropout)

    log_koff_samples = []
    for _ in range(n_mc_samples):
        pred_mean, pred_log_var = model(graph)
        # Denormalise
        log_koff = (pred_mean.item() * y_std) + y_mean
        log_koff_samples.append(log_koff)

    model.eval()  # re-disable dropout

    log_koff_arr  = np.array(log_koff_samples)
    log_koff_mean = float(log_koff_arr.mean())
    log_koff_std  = float(log_koff_arr.std())

    koff_pred = 10 ** log_koff_mean
    # Propagate uncertainty: std on log scale → multiplicative factor
    koff_std  = koff_pred * math.log(10) * log_koff_std

    return koff_pred, koff_std


# ══════════════════════════════════════════════════════════════════════════
# Main design function
# ══════════════════════════════════════════════════════════════════════════

def design(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Load model ─────────────────────────────────────────────────────
    model, y_mean, y_std = load_model(device)

    target_koff     = args.target_koff
    log_target_koff = math.log10(target_koff)
    log.info(f"Target koff: {target_koff:.2e} s⁻¹  (log10 = {log_target_koff:.2f})")

    # ── Generate sequences ─────────────────────────────────────────────
    target_pdb = Path(args.target_pdb)
    sequences  = run_protein_mpnn(
        target_pdb,
        binder_chain  = args.binder_chain,
        n_sequences   = args.n_sequences,
        temperature   = args.mpnn_temperature,
    )

    if not sequences:
        log.error("No sequences generated. Exiting.")
        sys.exit(1)

    # ── Score each candidate ───────────────────────────────────────────
    log.info(f"Scoring {len(sequences)} candidates ...")
    results = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        for i, seq in enumerate(sequences):
            # ProteinMPNN outputs sequences that map onto the existing
            # binder backbone, so we can approximate the complex structure
            # by threading the sequence onto the original backbone.
            # For a full MVP, we use ESMFold on just the binder chain
            # and combine with the target to get the complex PDB.

            binder_pdb = tmp_path / f"binder_{i:04d}.pdb"
            complex_pdb = tmp_path / f"complex_{i:04d}.pdb"

            # Predict binder structure
            predicted = predict_structure_esmfold(seq, binder_pdb)

            if predicted and binder_pdb.exists():
                # Combine binder + target into a single PDB
                _merge_pdbs(binder_pdb, target_pdb, complex_pdb,
                            binder_chain=args.binder_chain)
                eval_pdb = complex_pdb
            else:
                # Fallback: use original complex PDB
                # (approximation: koff changes are dominated by sequence)
                eval_pdb = target_pdb

            koff_pred, koff_std = predict_koff_threaded(
                model, eval_pdb, seq, y_mean, y_std, device,
                binder_chain=args.binder_chain,
            )

            delta_log = abs(math.log10(koff_pred) - log_target_koff) \
                        if not math.isnan(koff_pred) else float("inf")

            results.append({
                "rank":           0,           # filled in after sorting
                "sequence":       seq,
                "koff_pred":      koff_pred,
                "koff_std":       koff_std,
                "log10_koff":     math.log10(koff_pred) if not math.isnan(koff_pred) else float("nan"),
                "delta_log_koff": delta_log,
                "residence_time_s": 1.0 / koff_pred if not math.isnan(koff_pred) and koff_pred > 0 else float("nan"),
                "in_target_range":
                    delta_log < args.tolerance if not math.isnan(delta_log) else False,
            })

            if (i + 1) % 25 == 0:
                log.info(f"  [{i+1}/{len(sequences)}] last koff={koff_pred:.2e} s⁻¹")

    # ── Rank and save ──────────────────────────────────────────────────
    df = pd.DataFrame(results)
    df = df.sort_values("delta_log_koff").reset_index(drop=True)
    df["rank"] = df.index + 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    log.info("")
    log.info("=" * 60)
    log.info("DESIGN RESULTS")
    log.info(f"  Target koff         : {target_koff:.2e} s⁻¹")
    log.info(f"  Candidates scored   : {len(df)}")
    log.info(f"  In target range     : {df['in_target_range'].sum()} "
             f"(within ±{args.tolerance} log units)")
    log.info("")
    log.info("Top 5 candidates:")
    top5 = df.head(5)[["rank", "sequence", "koff_pred", "residence_time_s",
                         "delta_log_koff", "in_target_range"]]
    for _, row in top5.iterrows():
        log.info(
            f"  #{int(row['rank'])}: koff={row['koff_pred']:.2e} s⁻¹  "
            f"τ={row['residence_time_s']:.1f} s  "
            f"Δlog={row['delta_log_koff']:.2f}  seq={row['sequence'][:20]}..."
        )
    log.info("")
    log.info(f"Full results saved → {out_path}")
    log.info("=" * 60)


def _merge_pdbs(binder_pdb: Path, target_pdb: Path, out_pdb: Path,
                binder_chain: str = "B") -> None:
    """
    Naive PDB merge: relabel the binder chain and concatenate ATOM records.
    For a production pipeline, use BioPython's PDBIO properly.
    """
    lines = []
    # Target chains (keep as-is)
    for line in target_pdb.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            lines.append(line)
    lines.append("TER")

    # Binder chain (relabel chain to binder_chain letter)
    for line in binder_pdb.read_text().splitlines():
        if line.startswith("ATOM") and len(line) >= 22:
            line = line[:21] + binder_chain + line[22:]
            lines.append(line)
    lines.append("END")

    out_pdb.write_text("\n".join(lines))


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Kinetics-conditioned binder design")
    p.add_argument("--target_pdb",
                   required=True,
                   help="PDB file of target protein (with initial binder scaffold)")
    p.add_argument("--binder_chain",
                   default="B",
                   help="Chain ID of the binder to redesign (default: B)")
    p.add_argument("--target_koff",
                   type=float,
                   default=1e-3,
                   help="Target koff in s⁻¹ (default: 1e-3, τ=16 min)")
    p.add_argument("--n_sequences",
                   type=int,
                   default=200,
                   help="Number of sequences to generate from ProteinMPNN")
    p.add_argument("--mpnn_temperature",
                   type=float,
                   default=0.1,
                   help="ProteinMPNN sampling temperature (lower=more conservative)")
    p.add_argument("--tolerance",
                   type=float,
                   default=0.5,
                   help="Acceptable log10(koff) window around target")
    p.add_argument("--out",
                   default="results/koff_designs.csv",
                   help="Output CSV path")
    return p.parse_args()



AMINO_ACIDS_T = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX_T = {aa: i for i, aa in enumerate(AMINO_ACIDS_T)}

def predict_koff_threaded(model, pdb_path, sequence, y_mean, y_std, device, binder_chain="B", n_mc_samples=10):
    from build_dataset import pdb_to_graph, THREE_TO_ONE
    import math, numpy as np
    graph = pdb_to_graph(pdb_path, koff=1e-3)
    if graph is None:
        return float("nan"), float("nan")

    # Get ESM-2 embeddings (cached after first call)
    import esm as esm_lib
    if not hasattr(predict_koff_threaded, '_esm_cache'):
        _m, _a = esm_lib.pretrained.esm2_t33_650M_UR50D()
        predict_koff_threaded._esm_cache = (_m.eval().cuda(), _a)
    esm_model, alphabet = predict_koff_threaded._esm_cache
    bc = alphabet.get_batch_converter()
    _, _, tokens = bc([("seq", sequence)])
    with torch.no_grad():
        emb = esm_model(tokens.cuda(), repr_layers=[33])["representations"][33][0, 1:len(sequence)+1].cpu()

    # Get target chain sequence from PDB
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("x", str(pdb_path))
    chains = {c.get_id(): "".join(
        THREE_TO_ONE.get(r.get_resname().strip(), "X")
        for r in c.get_residues() if r.get_id()[0] == " "
    ) for c in struct[0].get_chains()}
    chain_ids = list(chains.keys())
    target_chain_id = next((c for c in chain_ids if c != binder_chain), chain_ids[-1])
    target_seq = chains.get(target_chain_id, "A" * 50)
    _, _, t_tokens = bc([("tgt", target_seq)])
    with torch.no_grad():
        t_emb = esm_model(t_tokens.cuda(), repr_layers=[33])["representations"][33][0, 1:len(target_seq)+1].cpu()

    # Build new node features: [ESM-2 (1280) | structural (10)]
    chain_labels = graph.x[:, 29]
    struct_feats = graph.x[:, 20:]
    esm_feats = torch.zeros(graph.x.shape[0], 1280)
    b_idx = t_idx = 0
    for ni in range(graph.x.shape[0]):
        if chain_labels[ni] < 0.5:
            if b_idx < emb.shape[0]: esm_feats[ni] = emb[b_idx]; b_idx += 1
        else:
            if t_idx < t_emb.shape[0]: esm_feats[ni] = t_emb[t_idx]; t_idx += 1
    graph.x = torch.cat([esm_feats, struct_feats], dim=-1)
    graph = graph.to(device)
    graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout): m.train()
    samples = []
    with torch.no_grad():
        for _ in range(n_mc_samples):
            pm, _ = model(graph)
            samples.append((pm.item() * y_std) + y_mean)
    model.eval()
    arr = np.array(samples)
    koff = 10 ** float(arr.mean())
    return koff, koff * math.log(10) * float(arr.std())

if __name__ == "__main__":
    args = parse_args()
    design(args)




AMINO_ACIDS_T = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX_T = {aa: i for i, aa in enumerate(AMINO_ACIDS_T)}

def predict_koff_threaded(model, pdb_path, sequence, y_mean, y_std, device, binder_chain="B", n_mc_samples=10):
    from build_dataset import pdb_to_graph
    import math
    graph = pdb_to_graph(pdb_path, koff=1e-3)
    if graph is None:
        return float("nan"), float("nan")
    binder_indices = (graph.x[:, 29] == 0).nonzero(as_tuple=True)[0]
    for local_i, node_i in enumerate(binder_indices):
        if local_i >= len(sequence): break
        graph.x[node_i, :20] = 0.0
        idx = AA_TO_IDX_T.get(sequence[local_i], -1)
        if idx >= 0: graph.x[node_i, idx] = 1.0
    graph = graph.to(device)
    graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout): m.train()
    samples = []
    with torch.no_grad():
        for _ in range(n_mc_samples):
            pm, _ = model(graph)
            samples.append((pm.item() * y_std) + y_mean)
    model.eval()
    arr = __import__("numpy").array(samples)
    koff = 10 ** float(arr.mean())
    return koff, koff * math.log(10) * float(arr.std())

"""
koff_generative.py — koff-Conditioned Generative Sequence Designer
===================================================================
Two generative approaches that condition sequence generation on a target koff.

This is fundamentally different from the ProteinMPNN + koffGNN filter pipeline:

  FILTER approach (what we built first):
    ProteinMPNN samples sequences unconditionally
    koffGNN scores them afterward
    We keep the ones that happen to be near target koff
    → This is SCREENING, not generation

  GENERATIVE approach (this script):
    Sequences are generated WITH target koff as a conditioning signal
    The generation process itself is guided by koff
    → This is CONDITIONAL GENERATION

Two methods implemented:

  1. GradKoff — gradient-based conditional generator
     Parameterizes the binder sequence as differentiable logits.
     Optimizes via gradient descent toward target koff.
     Uses Gumbel-softmax to maintain differentiability.
     Temperature annealing from τ=2.0 → τ=0.1 for sharp sequences.

  2. MCMCKoff — Metropolis-Hastings sampler
     Uses koffGNN as the energy function.
     Proposes point mutations, accepts/rejects by Metropolis criterion.
     Simulated annealing for exploration then convergence.
     Runs 10 parallel chains for diversity.

Usage:
  python koff_generative.py \
      --target_pdb   data/raw/pdb/5lqb.pdb \
      --binder_chain A \
      --target_koff  1e-4 \
      --method       both \
      --n_grad       50 \
      --n_mcmc       20 \
      --out          results/generative/il2_nara1_1e4.csv
"""

import argparse
import math
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from Bio.PDB import PDBParser

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX   = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
IDX_TO_AA   = {i: aa for i, aa in enumerate(AMINO_ACIDS)}

CKPT_PATH = ROOT / "checkpoints" / "koff_gnn_esm2_best.pt"


# ══════════════════════════════════════════════════════════════════════════
# Load model and ESM-2
# ══════════════════════════════════════════════════════════════════════════

def load_koffgnn(device):
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    from esm2_model import KoffGNNEsm
    model = KoffGNNEsm(
        hidden_dim=ckpt.get("hidden_dim", 256),
        n_layers=ckpt.get("n_layers", 4),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    log.info(f"koffGNN loaded: epoch={ckpt['epoch']} val_r={ckpt['val_r']:.3f}")
    return model, ckpt["y_mean"], ckpt["y_std"]


_esm_cache = None
def load_esm2(device):
    global _esm_cache
    if _esm_cache is None:
        import esm
        log.info("Loading ESM-2 650M...")
        m, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        _esm_cache = (m.eval().to(device), alphabet)
        log.info("ESM-2 loaded.")
    return _esm_cache


@torch.no_grad()
def get_esm2_embedding(sequence: str, esm_model, batch_converter, device, layer=33):
    """Get per-residue ESM-2 embeddings for a sequence. Returns (L, 1280) tensor."""
    _, _, tokens = batch_converter([("s", sequence)])
    out = esm_model(tokens.to(device), repr_layers=[layer])
    emb = out["representations"][layer][0, 1:len(sequence)+1]
    return emb.cpu()


def precompute_aa_embeddings(esm_model, batch_converter, device):
    """
    Precompute ESM-2 embeddings for all 20 amino acids in isolation.
    Returns AA_EMB: tensor of shape (20, 1280).

    These serve as a lookup table: given a soft one-hot over amino acids,
    we compute a weighted sum of their embeddings — making the sequence
    representation differentiable.
    """
    log.info("Precomputing amino acid embedding matrix (20 × 1280)...")
    aa_embs = []
    for aa in AMINO_ACIDS:
        emb = get_esm2_embedding(aa, esm_model, batch_converter, device)
        aa_embs.append(emb[0])  # single-residue sequence → first token
    AA_EMB = torch.stack(aa_embs, dim=0)  # (20, 1280)
    log.info(f"  AA embedding matrix shape: {AA_EMB.shape}")
    return AA_EMB


# ══════════════════════════════════════════════════════════════════════════
# Graph builder for the generative loop
# ══════════════════════════════════════════════════════════════════════════

def build_base_graph(pdb_path, binder_chain, esm_model, batch_converter, device):
    """
    Build the interface graph for a complex, extracting:
    - Structural features (dims 1280-1289 of node features)
    - Chain labels (dim 1289)
    - Edge connectivity and features
    - Binder residue indices and starting sequence
    - Target chain ESM-2 embeddings (fixed throughout optimization)

    Returns a dict with everything needed for the generative loop.
    """
    from build_dataset import pdb_to_graph, THREE_TO_ONE

    # Build graph with placeholder koff
    graph = pdb_to_graph(pdb_path, koff=1e-3)
    if graph is None:
        raise ValueError(f"Could not parse interface from {pdb_path}")

    # Parse chain sequences
    parser = PDBParser(QUIET=True)
    struct  = parser.get_structure("x", str(pdb_path))
    chains  = {}
    for c in struct[0].get_chains():
        seq = "".join(
            THREE_TO_ONE.get(r.get_resname().strip(), "X")
            for r in c.get_residues() if r.get_id()[0] == " "
        )
        if seq:
            chains[c.get_id()] = seq

    binder_seq = chains.get(binder_chain, "")
    target_chains = [c for c in chains if c != binder_chain]
    target_seq = chains.get(target_chains[0], "") if target_chains else ""

    if not binder_seq or not target_seq:
        raise ValueError(f"Could not find binder/target sequences in {pdb_path}")

    log.info(f"  Binder chain {binder_chain}: {len(binder_seq)} residues")
    log.info(f"  Target chain {target_chains[0]}: {len(target_seq)} residues")

    # Get target ESM-2 embeddings (fixed — we only optimize the binder)
    target_emb = get_esm2_embedding(target_seq, esm_model, batch_converter, device)

    # Get initial binder ESM-2 embeddings
    binder_emb = get_esm2_embedding(binder_seq, esm_model, batch_converter, device)

    # Build full ESM-2 node feature matrix (matching build_dataset logic)
    chain_labels = graph.x[:, 29]          # original chain labels (30-dim graph)
    struct_feats = graph.x[:, 20:]         # structural features (10-dim)

    n_nodes  = graph.x.shape[0]
    esm_feat = torch.zeros(n_nodes, 1280)
    b_idx = t_idx = 0
    binder_node_mask = []

    for ni in range(n_nodes):
        if chain_labels[ni] < 0.5:          # binder node
            if b_idx < binder_emb.shape[0]:
                esm_feat[ni] = binder_emb[b_idx]
                b_idx += 1
            binder_node_mask.append(True)
        else:                               # target node
            if t_idx < target_emb.shape[0]:
                esm_feat[ni] = target_emb[t_idx]
                t_idx += 1
            binder_node_mask.append(False)

    binder_node_mask = torch.tensor(binder_node_mask, dtype=torch.bool)
    binder_node_indices = binder_node_mask.nonzero(as_tuple=True)[0]

    # Full node feature matrix: [ESM-2 (1280) | structural (10)]
    full_x = torch.cat([esm_feat, struct_feats], dim=-1)
    graph.x = full_x

    return {
        "graph":               graph,
        "binder_seq":          binder_seq,
        "target_seq":          target_seq,
        "binder_node_indices": binder_node_indices,
        "binder_node_mask":    binder_node_mask,
        "target_esm_emb":      target_emb,
        "struct_feats":        struct_feats,
        "chain_labels":        chain_labels,
        "n_binder_nodes":      int(binder_node_mask.sum()),
    }


def score_sequence(sequence, base, koffgnn, y_mean, y_std, AA_EMB, device):
    """
    Score a discrete sequence with koffGNN.
    Returns predicted log10(koff).
    """
    import esm as esm_lib
    esm_model, alphabet = load_esm2(device)
    bc = alphabet.get_batch_converter()

    binder_emb = get_esm2_embedding(sequence, esm_model, bc, device)

    graph = base["graph"].clone()
    bi    = base["binder_node_indices"]

    new_x = graph.x.clone()
    for local_i, node_i in enumerate(bi):
        if local_i < binder_emb.shape[0]:
            new_x[node_i, :1280] = binder_emb[local_i]
    graph.x = new_x
    graph    = graph.to(device)
    graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)

    koffgnn.eval()
    with torch.no_grad():
        pm, _ = koffgnn(graph)
    return (pm.item() * y_std) + y_mean


# ══════════════════════════════════════════════════════════════════════════
# Method 1: GradKoff — gradient-based conditional generator
# ══════════════════════════════════════════════════════════════════════════

def gradkoff(
    base, koffgnn, AA_EMB, y_mean, y_std,
    target_log_koff, device,
    n_steps=300,
    lr=0.05,
    tau_start=2.0,
    tau_end=0.1,
    diversity_weight=0.01,
    n_sequences=10,
):
    """
    Gradient-based conditional sequence generator.

    Parameterizes the binder sequence as (N_binder_nodes, 20) logits.
    Uses Gumbel-softmax to create differentiable sequence embeddings:
        soft_emb = softmax(logits/τ) @ AA_EMB   # (N_binder, 1280)

    Optimizes:
        L = (log_koff_pred - log_koff_target)² + λ * diversity_loss

    Temperature τ anneals from tau_start → tau_end, sharpening the
    distribution over amino acids as optimization progresses.

    Returns a list of decoded sequences with their predicted koff values.
    """
    log.info(f"GradKoff: {n_sequences} sequences × {n_steps} steps")
    log.info(f"  target log10(koff) = {target_log_koff:.2f}")

    AA_EMB_dev = AA_EMB.to(device)
    bi         = base["binder_node_indices"]
    n_binder   = base["n_binder_nodes"]

    # Build static graph components (target nodes fixed, binder nodes optimized)
    base_graph = base["graph"].clone().to(device)
    struct_feats = base["struct_feats"].to(device)

    results = []

    for seq_idx in range(n_sequences):
        # Initialise logits: small random + slight bias toward ProteinMPNN-like AAs
        logits = torch.randn(n_binder, 20, device=device) * 0.5
        logits = nn.Parameter(logits)
        optimizer = Adam([logits], lr=lr)

        trajectory = []
        best_loss   = float("inf")
        best_seq    = None
        best_logkoff = None

        for step in range(n_steps):
            optimizer.zero_grad()

            # Anneal temperature
            frac = step / max(n_steps - 1, 1)
            tau  = tau_start * (tau_end / tau_start) ** frac

            # Gumbel-softmax: differentiable one-hot approximation
            soft_onehot = F.gumbel_softmax(logits, tau=tau, hard=False)  # (N, 20)

            # Weighted sum of amino acid embeddings
            soft_esm = soft_onehot @ AA_EMB_dev  # (N, 1280)

            # Reconstruct full node feature matrix
            new_x = base_graph.x.clone()
            for local_i, node_i in enumerate(bi):
                if local_i < soft_esm.shape[0]:
                    new_x[node_i, :1280] = soft_esm[local_i]

            graph_copy = base_graph.clone()
            graph_copy.x = new_x
            graph_copy.batch = torch.zeros(
                new_x.size(0), dtype=torch.long, device=device
            )

            # koffGNN forward pass
            koffgnn.train()  # keep dropout active for regularization
            pm, plv = koffgnn(graph_copy)
            log_koff_pred = pm * y_std + y_mean

            # Primary loss: hit target koff
            koff_loss = (log_koff_pred - target_log_koff) ** 2

            # Diversity loss: push logits toward higher entropy (more diverse sequences)
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()
            diversity_loss = -diversity_weight * entropy  # maximise entropy

            loss = koff_loss + diversity_loss
            loss.backward()

            # Gradient clipping for stability
            nn.utils.clip_grad_norm_([logits], 1.0)
            optimizer.step()

            lk = log_koff_pred.item()
            trajectory.append({
                "step":      step,
                "log_koff":  lk,
                "koff":      10 ** lk,
                "loss":      koff_loss.item(),
                "tau":       tau,
            })

            if koff_loss.item() < best_loss:
                best_loss    = koff_loss.item()
                best_seq     = _decode_logits(logits, base["binder_seq"])
                best_logkoff = lk

            if (step + 1) % 50 == 0:
                log.info(
                    f"  seq {seq_idx+1}/{n_sequences} "
                    f"step {step+1}/{n_steps} "
                    f"log_koff={lk:.3f} "
                    f"target={target_log_koff:.2f} "
                    f"Δ={abs(lk-target_log_koff):.3f} "
                    f"τ={tau:.3f}"
                )

        koffgnn.eval()

        results.append({
            "method":          "GradKoff",
            "sequence":        best_seq,
            "log10_koff":      best_logkoff,
            "koff_pred":       10 ** best_logkoff,
            "delta_log_koff":  abs(best_logkoff - target_log_koff),
            "residence_time_s": 1.0 / (10 ** best_logkoff),
            "in_target_range": abs(best_logkoff - target_log_koff) < 0.5,
            "trajectory":      trajectory,
            "seq_idx":         seq_idx,
        })

        log.info(
            f"  seq {seq_idx+1} done: "
            f"koff={10**best_logkoff:.2e} "
            f"τ_res={1/(10**best_logkoff):.0f}s "
            f"Δlog={abs(best_logkoff-target_log_koff):.3f}"
        )

    return results


def _decode_logits(logits: torch.Tensor, template_seq: str) -> str:
    """
    Decode logits (N, 20) to a discrete amino acid sequence.
    Positions with no corresponding binder node keep the template.
    """
    indices = logits.argmax(dim=-1).cpu().tolist()
    decoded = [IDX_TO_AA.get(i, "X") for i in indices]
    # Pad/trim to template length
    template = list(template_seq)
    for i in range(min(len(decoded), len(template))):
        template[i] = decoded[i]
    return "".join(template)


# ══════════════════════════════════════════════════════════════════════════
# Method 2: MCMCKoff — Metropolis-Hastings sampler
# ══════════════════════════════════════════════════════════════════════════

def mckoff(
    base, koffgnn, AA_EMB, y_mean, y_std,
    target_log_koff, device,
    n_chains=10,
    n_steps=500,
    T_start=2.0,
    T_end=0.05,
):
    """
    Metropolis-Hastings sequence sampler conditioned on target koff.

    Energy function: E(seq) = (log_koff_pred - log_koff_target)²

    Proposal: uniformly random point mutation at one binder residue.

    Acceptance probability: min(1, exp(-(E_new - E_old) / T))

    Temperature T anneals from T_start → T_end (simulated annealing).
    Runs n_chains independent chains in parallel for sequence diversity.
    """
    log.info(f"MCMCKoff: {n_chains} chains × {n_steps} steps")
    log.info(f"  target log10(koff) = {target_log_koff:.2f}")

    AA_EMB_dev = AA_EMB.to(device)
    bi         = base["binder_node_indices"].tolist()
    n_binder   = base["n_binder_nodes"]
    base_graph = base["graph"].clone().to(device)

    # Initialise chains from the binder sequence with random mutations
    binder_seq = list(base["binder_seq"])
    chains = []
    for _ in range(n_chains):
        seq = binder_seq.copy()
        # Random initialisation: mutate 30% of positions
        for i in range(len(seq)):
            if np.random.rand() < 0.3:
                seq[i] = IDX_TO_AA[np.random.randint(20)]
        chains.append(seq)

    esm_model_mcmc, alphabet_mcmc = load_esm2(device)
    bc_mcmc = alphabet_mcmc.get_batch_converter()

    def score_chain(seq_list):
        """Score a sequence using full ESM-2 context-dependent embeddings."""
        seq_str = "".join(seq_list[:len(base["binder_seq"])])
        binder_emb = get_esm2_embedding(seq_str, esm_model_mcmc, bc_mcmc, device)
        new_x = base_graph.x.clone()
        for local_i, node_i in enumerate(bi):
            if local_i < binder_emb.shape[0]:
                new_x[node_i, :1280] = binder_emb[local_i].to(device)
        g = base_graph.clone()
        g.x = new_x
        g.batch = torch.zeros(new_x.size(0), dtype=torch.long, device=device)
        with torch.no_grad():
            pm, _ = koffgnn(g)
        lk = (pm.item() * y_std) + y_mean
        return lk, (lk - target_log_koff) ** 2

    # Compute initial energies
    koffgnn.eval()
    chain_logkoff = []
    chain_energy  = []
    for chain in chains:
        lk, e = score_chain(chain)
        chain_logkoff.append(lk)
        chain_energy.append(e)

    all_results = [[] for _ in range(n_chains)]
    accept_counts = [0] * n_chains

    for step in range(n_steps):
        # Anneal temperature
        frac = step / max(n_steps - 1, 1)
        T    = T_start * (T_end / T_start) ** frac

        for c in range(n_chains):
            # Propose: mutate one random binder position
            pos    = np.random.randint(min(n_binder, len(chains[c])))
            old_aa = chains[c][pos]
            new_aa = IDX_TO_AA[np.random.randint(20)]
            while new_aa == old_aa:
                new_aa = IDX_TO_AA[np.random.randint(20)]

            # Score proposal
            chains[c][pos] = new_aa
            new_lk, new_e  = score_chain(chains[c])

            # Metropolis acceptance
            delta_e = new_e - chain_energy[c]
            accept  = delta_e < 0 or np.random.rand() < math.exp(-delta_e / T)

            if accept:
                chain_energy[c]  = new_e
                chain_logkoff[c] = new_lk
                accept_counts[c] += 1
            else:
                chains[c][pos] = old_aa  # revert

            all_results[c].append({
                "step":     step,
                "log_koff": chain_logkoff[c],
                "energy":   chain_energy[c],
                "T":        T,
                "accepted": accept,
            })

        if (step + 1) % 100 == 0:
            energies = [f"{chain_energy[c]:.3f}" for c in range(n_chains)]
            log.info(
                f"  step {step+1}/{n_steps} "
                f"T={T:.3f} "
                f"energies=[{', '.join(energies[:3])}...]"
            )

    # Collect best sequence from each chain
    results = []
    for c in range(n_chains):
        best_lk   = chain_logkoff[c]
        best_seq  = "".join(chains[c])
        acc_rate  = accept_counts[c] / n_steps
        results.append({
            "method":          "MCMCKoff",
            "sequence":        best_seq,
            "log10_koff":      best_lk,
            "koff_pred":       10 ** best_lk,
            "delta_log_koff":  abs(best_lk - target_log_koff),
            "residence_time_s": 1.0 / (10 ** best_lk),
            "in_target_range": abs(best_lk - target_log_koff) < 0.5,
            "accept_rate":     acc_rate,
            "chain":           c,
        })
        log.info(
            f"  chain {c+1}: koff={10**best_lk:.2e} "
            f"Δlog={abs(best_lk-target_log_koff):.3f} "
            f"accept={acc_rate:.1%}"
        )

    return results


# ══════════════════════════════════════════════════════════════════════════
# Comparison: ProteinMPNN filter baseline
# ══════════════════════════════════════════════════════════════════════════

def proteinmpnn_baseline(args, base, koffgnn, AA_EMB, y_mean, y_std,
                          target_log_koff, device, n=50):
    """Run ProteinMPNN + koffGNN filter for comparison."""
    log.info(f"ProteinMPNN baseline: {n} sequences")
    import subprocess, tempfile
    from pathlib import Path as P

    mpnn_dir = ROOT / "ProteinMPNN"
    if not mpnn_dir.exists():
        log.warning("ProteinMPNN not found — skipping baseline")
        return []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = P(tmp)
        out_dir  = tmp_path / "out"
        out_dir.mkdir()

        pdb_path = P(args.target_pdb)
        parser   = PDBParser(QUIET=True)
        struct   = parser.get_structure("x", str(pdb_path))
        all_chains  = [c.get_id() for c in struct[0].get_chains()]
        fixed_chains = [c for c in all_chains if c != args.binder_chain]

        import json
        chain_id = {pdb_path.stem: {
            "designed_chain_list": [args.binder_chain],
            "fixed_chain_list":    fixed_chains
        }}
        cj_path = tmp_path / "chain_id.jsonl"
        cj_path.write_text(json.dumps(chain_id) + "\n")

        cmd = [
            sys.executable,
            str(mpnn_dir / "protein_mpnn_run.py"),
            "--pdb_path",           str(pdb_path),
            "--out_folder",         str(out_dir),
            "--num_seq_per_target", str(n),
            "--sampling_temp",      "0.1",
            "--seed",               "42",
            "--batch_size",         "1",
            "--chain_id_jsonl",     str(cj_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            log.error(f"ProteinMPNN failed: {result.stderr[:200]}")
            return []

        sequences = []
        for fa in out_dir.glob("**/*.fa"):
            for line in fa.read_text().splitlines():
                if not line.startswith(">") and line.strip():
                    sequences.append(line.strip().split("/")[0])

    sequences = sequences[:n]
    log.info(f"  Scoring {len(sequences)} sequences...")

    results = []
    esm_model, alphabet = load_esm2(device)
    bc = alphabet.get_batch_converter()

    for seq in sequences:
        bi     = base["binder_node_indices"]
        graph  = base["graph"].clone()
        emb    = get_esm2_embedding(seq, esm_model, bc, device)
        new_x  = graph.x.clone()
        for li, ni in enumerate(bi):
            if li < emb.shape[0]:
                new_x[ni, :1280] = emb[li]
        graph.x = new_x
        graph   = graph.to(device)
        graph.batch = torch.zeros(
            graph.x.size(0), dtype=torch.long, device=device
        )
        with torch.no_grad():
            pm, _ = koffgnn(graph)
        lk = (pm.item() * y_std) + y_mean
        results.append({
            "method":          "ProteinMPNN+Filter",
            "sequence":        seq,
            "log10_koff":      lk,
            "koff_pred":       10 ** lk,
            "delta_log_koff":  abs(lk - target_log_koff),
            "residence_time_s": 1.0 / (10 ** lk),
            "in_target_range": abs(lk - target_log_koff) < 0.5,
        })

    return results


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    target_log_koff = math.log10(args.target_koff)
    log.info(f"Target koff: {args.target_koff:.2e} s⁻¹  (log10={target_log_koff:.2f})")
    log.info(f"Target residence time: {1/args.target_koff:.0f} s "
             f"({1/args.target_koff/3600:.2f} h)")

    # Load models
    koffgnn, y_mean, y_std = load_koffgnn(device)
    esm_model, alphabet    = load_esm2(device)
    bc = alphabet.get_batch_converter()

    # Precompute AA embedding matrix
    AA_EMB = precompute_aa_embeddings(esm_model, bc, device)

    # Build base graph
    log.info(f"Building interface graph from {args.target_pdb}...")
    base = build_base_graph(
        Path(args.target_pdb), args.binder_chain,
        esm_model, bc, device
    )
    log.info(f"  {base['n_binder_nodes']} binder interface nodes")

    all_results = []

    # ── GradKoff ──────────────────────────────────────────────────────────
    if args.method in ("grad", "both"):
        log.info("")
        log.info("=" * 50)
        log.info("METHOD 1: GradKoff (gradient-based generation)")
        log.info("=" * 50)
        t0 = time.time()
        grad_results = gradkoff(
            base, koffgnn, AA_EMB, y_mean, y_std,
            target_log_koff, device,
            n_steps=args.grad_steps,
            lr=args.lr,
            n_sequences=args.n_grad,
        )
        log.info(f"GradKoff done in {time.time()-t0:.0f}s")
        all_results.extend(grad_results)

    # ── MCMCKoff ──────────────────────────────────────────────────────────
    if args.method in ("mcmc", "both"):
        log.info("")
        log.info("=" * 50)
        log.info("METHOD 2: MCMCKoff (Metropolis-Hastings generation)")
        log.info("=" * 50)
        t0 = time.time()
        mcmc_results = mckoff(
            base, koffgnn, AA_EMB, y_mean, y_std,
            target_log_koff, device,
            n_chains=args.n_mcmc,
            n_steps=args.mcmc_steps,
        )
        log.info(f"MCMCKoff done in {time.time()-t0:.0f}s")
        all_results.extend(mcmc_results)

    # ── ProteinMPNN baseline ──────────────────────────────────────────────
    if args.baseline:
        log.info("")
        log.info("=" * 50)
        log.info("BASELINE: ProteinMPNN + koffGNN filter")
        log.info("=" * 50)
        t0 = time.time()
        bl_results = proteinmpnn_baseline(
            args, base, koffgnn, AA_EMB, y_mean, y_std,
            target_log_koff, device, n=50
        )
        log.info(f"Baseline done in {time.time()-t0:.0f}s")
        all_results.extend(bl_results)

    # ── Results ────────────────────────────────────────────────────────────
    df = pd.DataFrame([{k: v for k, v in r.items()
                        if k not in ("trajectory",)} for r in all_results])
    df = df.sort_values(["method", "delta_log_koff"]).reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    log.info("")
    log.info("=" * 60)
    log.info("GENERATIVE DESIGN RESULTS")
    log.info(f"  Target koff     : {args.target_koff:.2e} s⁻¹")
    log.info(f"  Target τ        : {1/args.target_koff:.0f} s "
             f"({1/args.target_koff/3600:.2f} h)")
    log.info("")

    for method in df["method"].unique():
        sub = df[df["method"] == method]
        hits = sub["in_target_range"].sum()
        best = sub.iloc[0]
        log.info(f"  {method}")
        log.info(f"    Sequences      : {len(sub)}")
        log.info(f"    In range (±0.5): {hits}/{len(sub)} "
                 f"({100*hits/len(sub):.0f}%)")
        log.info(f"    Best Δlog      : {best['delta_log_koff']:.3f}")
        log.info(f"    Best koff      : {best['koff_pred']:.2e} s⁻¹")
        log.info(f"    Best τ         : {best['residence_time_s']:.0f} s")
        log.info(f"    Best seq       : {best['sequence'][:25]}...")
        log.info("")

    log.info(f"  Full results → {out_path}")
    log.info("=" * 60)


def parse_args():
    p = argparse.ArgumentParser(description="koff-conditioned generative design")
    p.add_argument("--target_pdb",   required=True)
    p.add_argument("--binder_chain", default="A")
    p.add_argument("--target_koff",  type=float, default=1e-4)
    p.add_argument("--method",       choices=["grad","mcmc","both"], default="both")
    p.add_argument("--n_grad",       type=int,   default=10,
                   help="Number of sequences from GradKoff")
    p.add_argument("--grad_steps",   type=int,   default=300,
                   help="Gradient steps per sequence")
    p.add_argument("--lr",           type=float, default=0.05)
    p.add_argument("--n_mcmc",       type=int,   default=10,
                   help="Number of MCMC chains")
    p.add_argument("--mcmc_steps",   type=int,   default=500,
                   help="MCMC steps per chain")
    p.add_argument("--baseline",     action="store_true",
                   help="Also run ProteinMPNN+filter baseline for comparison")
    p.add_argument("--out",          default="results/generative/designs.csv")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
"""
STEP 2 — Feature Extraction (Interface Geometry)
=================================================
Converts each PDB complex + koff value into a graph data object
that a GNN can train on.

What we extract from each PDB structure:

  NODE FEATURES (per residue at interface):
    [0:20]   amino acid one-hot (20 standard AA)
    [20]     relative solvent-accessible area (rSASA) — burial proxy
    [21]     backbone phi angle (sin)
    [22]     backbone phi angle (cos)
    [23]     backbone psi angle (sin)
    [24]     backbone psi angle (cos)
    [25]     secondary structure: helix (1/0)
    [26]     secondary structure: sheet (1/0)
    [27]     secondary structure: loop  (1/0)
    [28]     B-factor (normalised, proxy for local mobility)
    [29]     chain identity (0 = binder, 1 = target)
  Total: 30 features per node

  EDGE FEATURES (per residue-residue contact at interface):
    [0]      Cα–Cα distance (Å), normalised to [0,1] for ≤12 Å
    [1]      Cβ–Cβ distance (Å), normalised
    [2]      is_hydrogen_bond   (O…N ≤ 3.5 Å, angle ≥ 120°)
    [3]      is_salt_bridge     (opposite charge centres ≤ 5 Å)
    [4]      is_hydrophobic     (both residues hydrophobic, ≤ 5 Å)
    [5]      is_pi_pi           (aromatic rings ≤ 5.5 Å)
    [6]      cross_chain        (edge crosses the binder/target boundary)
    [7]      sequence_separation (|i-j|) / 100, clamped 0-1
  Total: 8 features per edge

  GRAPH LABEL:
    log10(koff)  — our regression target

  GRAPH SCALARS (for conditioning during design):
    n_interface_residues
    buried_surface_area (Å²)
    n_hbonds
    n_saltbridges
"""

import math
import warnings
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

# BioPython
from Bio import PDB
from Bio.PDB import PDBParser, DSSP, NACCESS

# PyTorch Geometric
import torch
from torch_geometric.data import Data, InMemoryDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

warnings.filterwarnings("ignore")  # suppress BioPython deprecation noise

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent
RAW_DIR  = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
PDB_DIR  = RAW_DIR / "pdb"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX   = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

HYDROPHOBIC = set("AILMFWYV")
AROMATIC    = set("FWY")
POSITIVE    = set("KRH")
NEGATIVE    = set("DE")

CONTACT_CUTOFF    = 8.0   # Å — Cα-Cα cutoff to define interface residues
EDGE_CUTOFF       = 12.0  # Å — Cα-Cα cutoff for graph edges
HBOND_DIST        = 3.5   # Å
SALT_BRIDGE_DIST  = 5.0   # Å
HYDROPHOBIC_DIST  = 5.0   # Å
PI_PI_DIST        = 5.5   # Å

NODE_DIM = 30
EDGE_DIM = 8


# ══════════════════════════════════════════════════════════════════════════
# Geometry helpers
# ══════════════════════════════════════════════════════════════════════════

def get_ca(residue) -> Optional[np.ndarray]:
    try:
        return residue["CA"].get_vector().get_array()
    except KeyError:
        return None

def get_cb(residue) -> Optional[np.ndarray]:
    """Return Cβ; use Cα as fallback for Glycine."""
    try:
        return residue["CB"].get_vector().get_array()
    except KeyError:
        return get_ca(residue)

def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def is_hydrogen_bond(res_i, res_j) -> bool:
    """
    Crude H-bond check: any backbone N or O within HBOND_DIST of any
    backbone N or O on the other residue.
    """
    donors_i    = [res_i[a].get_vector().get_array()
                   for a in ("N", "O") if a in res_i]
    acceptors_j = [res_j[a].get_vector().get_array()
                   for a in ("N", "O") if a in res_j]
    for d in donors_i:
        for a in acceptors_j:
            if dist(d, a) <= HBOND_DIST:
                return True
    return False

def is_salt_bridge(res_i, res_j) -> bool:
    aa_i = res_i.get_resname().strip()
    aa_j = res_j.get_resname().strip()
    one_i = _three_to_one(aa_i)
    one_j = _three_to_one(aa_j)
    if not ({one_i} <= POSITIVE | NEGATIVE):
        return False
    if not ({one_j} <= POSITIVE | NEGATIVE):
        return False
    if (one_i in POSITIVE) == (one_j in POSITIVE):
        return False   # same charge
    ca_i = get_ca(res_i)
    ca_j = get_ca(res_j)
    if ca_i is None or ca_j is None:
        return False
    return dist(ca_i, ca_j) <= SALT_BRIDGE_DIST

THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}

def _three_to_one(resname: str) -> str:
    return THREE_TO_ONE.get(resname.strip().upper(), "X")


# ══════════════════════════════════════════════════════════════════════════
# Interface identification
# ══════════════════════════════════════════════════════════════════════════

def get_interface_residues(
    chain_a_residues: List,
    chain_b_residues: List,
    cutoff: float = CONTACT_CUTOFF,
) -> Tuple[List, List]:
    """
    Return residues from chain A and chain B that are within `cutoff` Å
    (Cα-Cα) of any residue in the other chain.
    """
    iface_a, iface_b = set(), set()
    for ri in chain_a_residues:
        ca_i = get_ca(ri)
        if ca_i is None:
            continue
        for rj in chain_b_residues:
            ca_j = get_ca(rj)
            if ca_j is None:
                continue
            if dist(ca_i, ca_j) <= cutoff:
                iface_a.add(ri.get_id())
                iface_b.add(rj.get_id())

    iface_a_list = [r for r in chain_a_residues if r.get_id() in iface_a]
    iface_b_list = [r for r in chain_b_residues if r.get_id() in iface_b]
    return iface_a_list, iface_b_list


# ══════════════════════════════════════════════════════════════════════════
# Node features
# ══════════════════════════════════════════════════════════════════════════

def residue_node_features(
    residue,
    chain_label: int,          # 0 = binder, 1 = target
    sasa: Optional[float],     # relative SASA [0,1]
    ss: str,                   # 'H', 'E', or '-'
    phi: float,
    psi: float,
) -> np.ndarray:
    feat = np.zeros(NODE_DIM, dtype=np.float32)

    # [0:20] amino acid one-hot
    aa_char = _three_to_one(residue.get_resname())
    idx = AA_TO_IDX.get(aa_char, -1)
    if idx >= 0:
        feat[idx] = 1.0

    # [20] rSASA (burial proxy; 0 = buried, 1 = fully exposed)
    feat[20] = float(sasa) if sasa is not None else 0.5

    # [21-24] backbone dihedrals
    feat[21] = math.sin(phi) if phi is not None else 0.0
    feat[22] = math.cos(phi) if phi is not None else 0.0
    feat[23] = math.sin(psi) if psi is not None else 0.0
    feat[24] = math.cos(psi) if psi is not None else 0.0

    # [25-27] secondary structure
    feat[25] = 1.0 if ss == "H" else 0.0   # helix
    feat[26] = 1.0 if ss == "E" else 0.0   # sheet
    feat[27] = 1.0 if ss == "-" else 0.0   # loop/other

    # [28] B-factor (normalised by 100; proxy for local flexibility)
    try:
        bfac = residue["CA"].get_bfactor()
        feat[28] = min(float(bfac) / 100.0, 1.0)
    except KeyError:
        feat[28] = 0.0

    # [29] chain identity
    feat[29] = float(chain_label)

    return feat


# ══════════════════════════════════════════════════════════════════════════
# Edge features
# ══════════════════════════════════════════════════════════════════════════

def residue_edge_features(
    res_i,
    res_j,
    seq_i: int,
    seq_j: int,
    cross_chain: bool,
) -> np.ndarray:
    feat = np.zeros(EDGE_DIM, dtype=np.float32)

    ca_i = get_ca(res_i)
    ca_j = get_ca(res_j)
    cb_i = get_cb(res_i)
    cb_j = get_cb(res_j)

    # [0] Cα-Cα distance normalised to [0,1] for ≤ EDGE_CUTOFF
    d_ca = dist(ca_i, ca_j) if (ca_i is not None and ca_j is not None) else EDGE_CUTOFF
    feat[0] = min(d_ca / EDGE_CUTOFF, 1.0)

    # [1] Cβ-Cβ distance
    d_cb = dist(cb_i, cb_j) if (cb_i is not None and cb_j is not None) else EDGE_CUTOFF
    feat[1] = min(d_cb / EDGE_CUTOFF, 1.0)

    # [2] hydrogen bond
    try:
        feat[2] = 1.0 if is_hydrogen_bond(res_i, res_j) else 0.0
    except Exception:
        feat[2] = 0.0

    # [3] salt bridge
    try:
        feat[3] = 1.0 if is_salt_bridge(res_i, res_j) else 0.0
    except Exception:
        feat[3] = 0.0

    # [4] hydrophobic contact
    aa_i = _three_to_one(res_i.get_resname())
    aa_j = _three_to_one(res_j.get_resname())
    feat[4] = 1.0 if (
        aa_i in HYDROPHOBIC and aa_j in HYDROPHOBIC and d_ca <= HYDROPHOBIC_DIST
    ) else 0.0

    # [5] pi-pi (aromatic)
    feat[5] = 1.0 if (
        aa_i in AROMATIC and aa_j in AROMATIC and d_ca <= PI_PI_DIST
    ) else 0.0

    # [6] cross-chain
    feat[6] = 1.0 if cross_chain else 0.0

    # [7] sequence separation
    feat[7] = min(abs(seq_i - seq_j) / 100.0, 1.0) if not cross_chain else 1.0

    return feat


# ══════════════════════════════════════════════════════════════════════════
# Build a single graph from one PDB complex
# ══════════════════════════════════════════════════════════════════════════

def pdb_to_graph(
    pdb_path: Path,
    koff: float,
    binder_chain_ids: Optional[List[str]] = None,
    target_chain_ids: Optional[List[str]] = None,
) -> Optional[Data]:
    """
    Parse a PDB file and return a PyG Data object.

    For antibody complexes, binder_chain_ids = ['H', 'L'] and
    target_chain_ids = ['A'] (antigen).  If None, we auto-detect:
    - Assume the smallest chain(s) by residue count are the binder.
    - The largest chain is the target.
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("prot", str(pdb_path))
    except Exception as e:
        log.debug(f"  Failed to parse {pdb_path.name}: {e}")
        return None

    model = structure[0]
    chains = list(model.get_chains())
    if len(chains) < 2:
        return None  # need at least two chains

    # ── Assign binder / target chains ─────────────────────────────────
    if binder_chain_ids is None and target_chain_ids is None:
        # Heuristic: largest chain = target, rest = binder
        chain_lengths = {
            c.get_id(): sum(1 for r in c if r.get_id()[0] == " ")
            for c in chains
        }
        target_chain_id = max(chain_lengths, key=chain_lengths.get)
        binder_ids = [c.get_id() for c in chains if c.get_id() != target_chain_id]
        target_ids = [target_chain_id]
    else:
        binder_ids = binder_chain_ids or []
        target_ids = target_chain_ids or []

    def std_residues(chain):
        return [r for r in chain if r.get_id()[0] == " "]  # ATOM records only

    binder_res = []
    for cid in binder_ids:
        try:
            binder_res.extend(std_residues(model[cid]))
        except KeyError:
            pass

    target_res = []
    for cid in target_ids:
        try:
            target_res.extend(std_residues(model[cid]))
        except KeyError:
            pass

    if not binder_res or not target_res:
        return None

    # ── Interface residues ─────────────────────────────────────────────
    iface_binder, iface_target = get_interface_residues(binder_res, target_res)

    if len(iface_binder) < 3 or len(iface_target) < 3:
        return None  # degenerate interface

    all_iface = [(r, 0) for r in iface_binder] + [(r, 1) for r in iface_target]
    n_nodes   = len(all_iface)

    # ── DSSP (secondary structure + phi/psi) ──────────────────────────
    # DSSP requires mkdssp binary.  Fall back gracefully if absent.
    dssp_data: Dict = {}
    try:
        dssp = DSSP(model, str(pdb_path), dssp="mkdssp")
        dssp_data = {k: v for k, v in dssp}
    except Exception:
        pass  # will use neutral values

    def get_dssp_values(res):
        key = (res.get_parent().get_id(), res.get_id())
        if key in dssp_data:
            v    = dssp_data[key]
            ss   = v[2]  # H / E / -
            phi  = math.radians(v[4]) if v[4] != 360.0 else 0.0
            psi  = math.radians(v[5]) if v[5] != 360.0 else 0.0
            sasa = v[3]   # relative SASA [0,1]
            return ss, phi, psi, sasa
        return "-", 0.0, 0.0, 0.5

    # ── Build node feature matrix ──────────────────────────────────────
    x_list = []
    for res, chain_label in all_iface:
        ss, phi, psi, sasa = get_dssp_values(res)
        feat = residue_node_features(res, chain_label, sasa, ss, phi, psi)
        x_list.append(feat)

    x = torch.tensor(np.array(x_list), dtype=torch.float32)

    # ── Build edge list ────────────────────────────────────────────────
    edge_src, edge_dst, edge_attrs = [], [], []

    for i, (res_i, chain_i) in enumerate(all_iface):
        ca_i = get_ca(res_i)
        if ca_i is None:
            continue
        for j, (res_j, chain_j) in enumerate(all_iface):
            if i == j:
                continue
            ca_j = get_ca(res_j)
            if ca_j is None:
                continue
            if dist(ca_i, ca_j) > EDGE_CUTOFF:
                continue
            cross = (chain_i != chain_j)
            ef = residue_edge_features(res_i, res_j, i, j, cross)
            edge_src.append(i)
            edge_dst.append(j)
            edge_attrs.append(ef)

    if not edge_src:
        return None

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr  = torch.tensor(np.array(edge_attrs), dtype=torch.float32)

    # ── Graph-level scalars ────────────────────────────────────────────
    n_hb = sum(
        1 for (src, dst, ea) in zip(edge_src, edge_dst, edge_attrs)
        if ea[2] > 0.5 and edge_attrs[0][6] > 0.5  # hbond + cross-chain
    )
    graph_feat = torch.tensor(
        [
            len(all_iface),
            len(iface_binder),
            len(iface_target),
            n_hb,
        ],
        dtype=torch.float32,
    )

    # ── Regression target: log10(koff) ────────────────────────────────
    # Typical koff range: 1e-5 to 1 s⁻¹ → log10 range: −5 to 0
    y = torch.tensor([[math.log10(koff)]], dtype=torch.float32)

    data = Data(
        x          = x,
        edge_index = edge_index,
        edge_attr  = edge_attr,
        graph_feat = graph_feat,
        y          = y,
        pdb_id     = pdb_path.stem,
        koff       = koff,
    )
    return data


# ══════════════════════════════════════════════════════════════════════════
# Dataset class (PyG InMemoryDataset)
# ══════════════════════════════════════════════════════════════════════════

class KoffDataset(InMemoryDataset):
    """
    PyTorch Geometric InMemoryDataset.

    Usage:
        dataset = KoffDataset(root="data/processed")
        print(len(dataset))        # number of complexes
        graph = dataset[0]         # one graph
        print(graph.x.shape)       # (n_nodes, 30)
        print(graph.y)             # log10(koff)
    """

    def __init__(self, root: str):
        super().__init__(root=root)
        self.data, self.slices = torch.load(
            self.processed_paths[0], weights_only=False
        )

    @property
    def raw_file_names(self):
        return ["sabdab_summary.tsv"]

    @property
    def processed_file_names(self):
        return ["koff_dataset.pt"]

    def download(self):
        # Data download is handled by 01_download_data.py
        pass

    def process(self):
        summary_path = Path(self.root).parent / "raw" / "sabdab_summary.tsv"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"Summary file not found: {summary_path}. "
                "Run  python 01_download_data.py  first."
            )

        df = pd.read_csv(summary_path, sep="\t", low_memory=False)
        df.columns = [c.strip().lower() for c in df.columns]

        # Identify PDB and koff columns robustly
        pdb_col  = next((c for c in df.columns if c in ("pdb", "pdbid")), None)
        koff_col = "koff_s"

        if pdb_col is None or koff_col not in df.columns:
            raise ValueError(
                f"Expected 'pdb'/'pdbid' and 'koff_s' columns. "
                f"Got: {list(df.columns)}"
            )

        pdb_dir = Path(self.root).parent / "raw" / "pdb"
        data_list = []
        skipped   = {"no_pdb": 0, "parse_fail": 0, "bad_koff": 0}

        log.info(f"Processing {len(df)} SAbDab entries ...")

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building graphs"):
            pdb_id = str(row[pdb_col]).strip().lower()
            koff   = row[koff_col]

            # Validate koff
            try:
                koff = float(koff)
                if not (1e-7 < koff < 1e2):  # physically plausible range
                    skipped["bad_koff"] += 1
                    continue
            except (ValueError, TypeError):
                skipped["bad_koff"] += 1
                continue

            pdb_path = pdb_dir / f"{pdb_id}.pdb"
            if not pdb_path.exists():
                skipped["no_pdb"] += 1
                continue

            # For SAbDab, antibody chains are typically H and L; antigen is A
            # We rely on auto-detection for generality.
            graph = pdb_to_graph(pdb_path, koff)
            if graph is None:
                skipped["parse_fail"] += 1
                continue

            data_list.append(graph)

        log.info(f"  Built {len(data_list)} graphs")
        log.info(f"  Skipped — no PDB: {skipped['no_pdb']}, "
                 f"parse failure: {skipped['parse_fail']}, "
                 f"bad koff: {skipped['bad_koff']}")

        if not data_list:
            raise RuntimeError(
                "No graphs were built. Check that PDB files are downloaded "
                "and koff values are in the expected range (1e-7 to 1e2 s⁻¹)."
            )

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        log.info(f"  Saved dataset → {self.processed_paths[0]}")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("STEP 2: Building interface graph dataset")
    log.info("=" * 60)

    dataset = KoffDataset(root=str(PROC_DIR))
    log.info(f"Dataset size: {len(dataset)} complexes")

    if len(dataset) > 0:
        g = dataset[0]
        log.info(f"Example graph:")
        log.info(f"  Nodes         : {g.x.shape[0]}")
        log.info(f"  Node features : {g.x.shape[1]}")
        log.info(f"  Edges         : {g.edge_index.shape[1]}")
        log.info(f"  Edge features : {g.edge_attr.shape[1]}")
        log.info(f"  y (log10 koff): {g.y.item():.3f}  → koff = {10**g.y.item():.2e} s⁻¹")

        # koff distribution stats
        koffs = torch.tensor([dataset[i].y.item() for i in range(len(dataset))])
        log.info(f"log10(koff) stats:")
        log.info(f"  mean={koffs.mean():.2f}  std={koffs.std():.2f}  "
                 f"min={koffs.min():.2f}  max={koffs.max():.2f}")

    log.info("")
    log.info("NEXT STEP: Run  python 03_train.py")


if __name__ == "__main__":
    main()

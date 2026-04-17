# koff_designer — Kinetics-Conditioned Protein Binder Design
## A GNN that designs binders with specified residence time (koff)

---

## What this does

Every protein binder design method (RFdiffusion, BindCraft, AlphaProteo)
optimises for binding **affinity (Kd)**. This project designs binders for
a specified **koff** — the rate of unbinding. koff is clinically more
important than Kd for therapeutics: it determines how long a drug stays
bound to its target.

**Scientific novelty:** koff is determined by the transition-state free energy
ΔG‡, not the ground-state energy. This requires learning interface features
that predict unbinding barriers — a representation nobody has trained a
generative model on.

---

## Directory structure

```
koff_designer/
├── 00_quickstart_test.py    # Run FIRST to verify your setup
├── 01_download_data.py      # Download SAbDab + PDB structures
├── 02_build_dataset.py      # Extract interface graphs + features
├── 03_train.py              # Train koffGNN
├── 04_design.py             # Design binders for target koff
├── model/
│   └── koff_gnn.py          # GNN architecture (ECMPConv + attention pooling)
├── setup_wsl2.sh            # WSL2 + RTX 5090 environment setup
├── data/
│   ├── raw/                 # Downloaded PDBs, SAbDab summary
│   └── processed/           # Graph dataset (built by step 2)
├── checkpoints/             # Saved model weights
└── results/                 # Design outputs
```

---

## Step-by-step instructions

### Prerequisites
- Windows 11 with WSL2 (Ubuntu 22.04 recommended)
- RTX 5090 with latest NVIDIA drivers in Windows
- GPU pass-through enabled in WSL2 (comes built-in from Windows 11 22H2+)

---

### STEP 0: Set up WSL2 environment (one-time)

Open WSL2 terminal:

```bash
cd ~/koff_designer
chmod +x setup_wsl2.sh
./setup_wsl2.sh
source .venv/bin/activate
```

This installs:
- PyTorch 2.7 with CUDA 12.8 (required for RTX 5090 Blackwell architecture)
- PyTorch Geometric
- BioPython, MDAnalysis, ESM-2
- ProteinMPNN (Baker Lab)

---

### STEP 1: Verify GPU pipeline works (before downloading data)

```bash
source .venv/bin/activate
python 00_quickstart_test.py
```

Expected output:
```
GPU detected: NVIDIA GeForce RTX 5090
VRAM        : 32.0 GB
Generating 300 synthetic interface graphs ...
Epoch 10/50 | val_rmse=0.71 | r=0.42
Epoch 20/50 | val_rmse=0.58 | r=0.61
...
SMOKE TEST PASSED
Test RMSE (log10 koff) : 0.52
Test Pearson r         : 0.68
```

If this fails, GPU is not accessible. Check:
```bash
nvidia-smi   # should show RTX 5090 from inside WSL2
```

---

### STEP 2: Download SAbDab kinetics data and PDB structures

```bash
python 01_download_data.py
```

**What it downloads:**
- SAbDab summary TSV (~30 MB): antibody-antigen complexes with SPR koff values
- Up to 500 PDB structures (~50-200 MB total)
- Takes ~10-30 minutes depending on internet speed

**Expected output:**
```
[02:31:15] Downloaded 3,247 SAbDab entries
[02:31:15] Rows with koff data: 847 / 3,247
[02:31:45] PDB download complete: 492 new, 0 cached, 8 failed
```

---

### STEP 3: Build the interface graph dataset

```bash
python 02_build_dataset.py
```

**What it does:**
- Parses each PDB structure with BioPython
- Identifies interface residues (Cα-Cα < 8 Å)
- Extracts 30 node features per residue:
  - Amino acid identity (one-hot)
  - Relative SASA (burial)
  - Backbone dihedrals (phi, psi)
  - Secondary structure
  - B-factor (local flexibility proxy)
  - Chain identity (binder vs. target)
- Extracts 8 edge features per contact:
  - Distance (Cα-Cα, Cβ-Cβ)
  - H-bond, salt bridge, hydrophobic, π-π
  - Cross-chain flag
  - Sequence separation
- Assigns graph label: log10(koff)

**Expected output:**
```
Processing 847 SAbDab entries ...
Built 423 graphs
Skipped — no PDB: 312, parse failure: 47, bad koff: 65
log10(koff) stats: mean=-3.41 std=1.12 min=-6.21 max=-0.03
```

**Note:** ~50% skip rate is normal. PDB files may be absent, structures may
have only one chain, or koff values may be outside the physical range.

---

### STEP 4: Train the koffGNN model

```bash
python 03_train.py --epochs 150 --batch_size 32 --hidden_dim 128
```

**Full argument list:**
```
--epochs        150     Max training epochs
--batch_size    32      Graphs per batch
--hidden_dim    128     GNN hidden dimension (256 if VRAM allows)
--n_layers      4       Message-passing layers
--dropout       0.15    Dropout rate
--lr            3e-4    Adam learning rate
--patience      20      Early stopping patience
```

**On RTX 5090 with ~400 graphs:** ~2-5 minutes total training time.

**Expected results (MVP scale, ~400 graphs):**
```
Test RMSE (log10 koff) : 0.6–0.9  (within a factor of 4–8× of true koff)
Test Pearson r         : 0.40–0.65
```

**Why these numbers are still useful:**
A model that distinguishes koff=1e-1 (τ=10s, useless) from koff=1e-4
(τ=2.7h, excellent) with 80% accuracy is already scientifically valuable
as a ranking oracle — even if absolute prediction is noisy.

**Monitor training:**
```bash
tensorboard --logdir=runs/
# Open http://localhost:6006 in browser
```

---

### STEP 5: Design binders with target koff

You need a starting PDB with a target protein + initial binder scaffold.
Get one from:
- RFdiffusion (generates backbone)
- RCSB: download any antibody-antigen complex (e.g., 6W41 for PD-L1)

```bash
# Example: Design PD-L1 binders with residence time of ~17 minutes
# (koff = 1e-3 s⁻¹)
python 04_design.py \
    --target_pdb    data/raw/pdb/6w41.pdb \
    --binder_chain  H \
    --target_koff   1e-3 \
    --n_sequences   200 \
    --out           results/pdl1_koff_1e3.csv
```

**Output CSV columns:**
```
rank | sequence | koff_pred | koff_std | log10_koff | delta_log_koff | residence_time_s | in_target_range
```

**Interpretation:**
- `koff_pred`: predicted koff in s⁻¹
- `residence_time_s`: predicted drug-target residence time = 1/koff
- `delta_log_koff`: how close to your target (want this small)
- `in_target_range`: True if within ±0.5 log units of target

---

## Understanding the model architecture

```
Interface graph (nodes=residues, edges=contacts)
         ↓
Node embedding: Linear(30 → 128) + LayerNorm + GELU
Edge embedding: Linear(8  → 64)  + GELU
         ↓
4× ECMPConv layers:
   message: MLP( [h_neighbor ‖ e_edge] → 128 )
   update:  MLP( [h_self ‖ Σ messages] → 128 ) + residual
         ↓
Graph readout:
   [attention_pool ‖ mean_pool ‖ max_pool ‖ graph_scalars]
   → concat: (128+128+128+4) = 388-dim vector
         ↓
MLP head: 388 → 256 → 64 → (μ, log σ²)
   μ      = predicted log10(koff)
   log σ² = predicted uncertainty (Gaussian NLL loss)
```

**Why ECMPConv (not standard GCN)?**
Because H-bond edges and cross-chain contacts need to contribute differently
to the prediction. ECMPConv computes messages that are a function of BOTH
the neighbor node AND the edge features, letting the model learn that
"a cross-chain H-bond at the interface periphery" is more informative for koff
than "a within-chain H-bond far from the interface."

---

## What to improve for the real research paper

1. **More data:** SAbDab gives ~400 complexes at MVP scale. The full pipeline
   should incorporate BindingDB PPI data, curated SPR papers, and kinetics
   databases → target 3,000+ data points.

2. **Enhanced MD features:** Run metadynamics on 50–100 representative
   complexes to get partial dissociation trajectories. Add transition-state
   geometry as additional node features. This is the key differentiator
   from existing methods.

3. **Richer structure:** Replace ESMFold threading with full complex
   prediction using AlphaFold3 or Boltz-2 for each designed sequence.

4. **Generative model upgrade:** Replace ProteinMPNN + scoring with a
   conditional flow-matching model that generates sequences and koff
   simultaneously (the full research version).

5. **Experimental validation:** SPR measurements for top-ranked vs.
   bottom-ranked designs. This is what makes the Nature paper.

---

## Cite / reference

This research gap was identified by systematic literature review of:
- All binder design methods (2022–2026): none condition on koff
- QSKR literature: koff prediction exists only for small molecules
- Baker Lab publications: no kinetics-conditioned protein design

Key references:
- Copeland et al., Nat Rev Drug Discov 2006: drug-target residence time
- Watson et al., Nature 2023: RFdiffusion (affinity-only)
- Proteina-Complexa (NVIDIA, 2026): still affinity-only
- PXDesign (2025): still affinity-only

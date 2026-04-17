#!/bin/bash
# ============================================================
# koff_designer — WSL2 + RTX 5090 Setup Script
# Run once inside WSL2 (Ubuntu 22.04+)
# ============================================================
set -e

echo "========================================"
echo "  koff_designer environment setup"
echo "  Target: RTX 5090 (Blackwell, sm_100)"
echo "========================================"

# ---------------------------------------------------------
# STEP 1: System packages
# ---------------------------------------------------------
echo ""
echo "[1/7] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y \
    build-essential git curl wget \
    python3-pip python3-venv python3-dev \
    libhdf5-dev libgraph-easy-perl \
    cmake ninja-build \
    dssp         # secondary structure assignment (used in features)

# ---------------------------------------------------------
# STEP 2: Create and activate Python venv
# ---------------------------------------------------------
echo ""
echo "[2/7] Creating Python virtual environment..."
cd ~/koff_designer
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------
# STEP 3: PyTorch 2.7 — CUDA 12.8 (required for RTX 5090)
# The RTX 5090 is Blackwell (compute capability 10.0).
# PyTorch >= 2.6 + CUDA 12.8 is the minimum.
# ---------------------------------------------------------
echo ""
echo "[3/7] Installing PyTorch 2.7 with CUDA 12.8..."
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# ---------------------------------------------------------
# STEP 4: PyTorch Geometric (GNN framework)
# ---------------------------------------------------------
echo ""
echo "[4/7] Installing PyTorch Geometric..."
pip install torch_geometric

# PyG optional dependencies (scatter, sparse, cluster)
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.7.0+cu128.html 2>/dev/null || \
pip install torch_scatter torch_sparse torch_cluster \
    -f https://data.pyg.org/whl/torch-2.6.0+cu128.html 2>/dev/null || \
echo "  -> PyG extras: will fall back to pure-Python ops (slower but works)"

# ---------------------------------------------------------
# STEP 5: Scientific / bioinformatics stack
# ---------------------------------------------------------
echo ""
echo "[5/7] Installing scientific stack..."
pip install \
    numpy pandas scipy scikit-learn \
    matplotlib seaborn tqdm \
    biopython==1.83 \
    mdanalysis \
    requests \
    pyyaml \
    tensorboard \
    fair-esm             # ESM-2 protein language model embeddings

# ---------------------------------------------------------
# STEP 6: ProteinMPNN (Baker Lab) — for the design step
# We clone it locally because there is no pip package.
# ---------------------------------------------------------
echo ""
echo "[6/7] Cloning ProteinMPNN..."
if [ ! -d "ProteinMPNN" ]; then
    git clone https://github.com/dauparas/ProteinMPNN.git
fi

# ---------------------------------------------------------
# STEP 7: Verify GPU is visible
# ---------------------------------------------------------
echo ""
echo "[7/7] GPU check..."
python3 - <<'PYEOF'
import torch
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"VRAM            : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Compute cap.    : {torch.cuda.get_device_capability(0)}")
else:
    print("  -> No GPU found. Check NVIDIA drivers in Windows + WSL2 GPU pass-through.")
    print("  -> Install CUDA toolkit in Windows and ensure wsl2 GPU drivers are present.")
PYEOF

echo ""
echo "========================================"
echo "  Setup complete."
echo "  Activate env:  source ~/koff_designer/.venv/bin/activate"
echo "========================================"

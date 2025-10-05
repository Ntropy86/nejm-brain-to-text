
#!/usr/bin/env bash

# setup.sh
# Cross-platform setup script for nejm-brain-to-text
# This script creates a conda env and installs Python deps.
# It detects macOS (Apple Silicon / Intel) and Linux and installs
# an appropriate PyTorch build: CUDA wheel for Linux with CUDA,
# and macOS (MPS) compatible wheel for Apple machines.

# Ensure conda is available
if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found in PATH. Please install Miniconda/Anaconda and retry."
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

# Create conda environment with Python 3.10 (idempotent)
ENV_NAME=b2txt25
PY_VER=3.10
if conda env list | grep -q "^${ENV_NAME}[[:space:]]"; then
    echo "Conda env '${ENV_NAME}' already exists. Skipping creation."
else
    conda create -n ${ENV_NAME} python=${PY_VER} -y
fi

# Activate the new environment
conda activate ${ENV_NAME}

# Upgrade pip
python -m pip install --upgrade pip

# Detect platform and choose PyTorch install
OS_NAME=$(uname -s)
ARCH=$(uname -m)

echo "Detected OS=${OS_NAME} ARCH=${ARCH}"

if [[ "${OS_NAME}" == "Darwin" ]]; then
    # macOS: prefer PyTorch with MPS (Apple Metal Performance Shaders)
    # Use the stable cpu/macos build from PyPI which enables MPS on Apple Silicon
    echo "Installing PyTorch (macOS/MPS-compatible)..."
    # Recent PyTorch stable releases provide macOS wheels on PyPI.
    python -m pip install --upgrade "torch" torchvision torchaudio
    EXTRA_PIP_PACKAGES=(
        redis==5.2.1
        jupyter==1.1.1
        numpy==2.1.2
        pandas==2.3.0
        matplotlib==3.10.1
        scipy==1.15.2
        scikit-learn==1.6.1
        tqdm==4.67.1
        g2p_en==2.1.0
        h5py==3.13.0
        omegaconf==2.3.0
        editdistance==0.8.1
        -e .
        huggingface-hub==0.33.1
        transformers==4.53.0
        tokenizers==0.21.2
        accelerate==1.8.1
    )
    # bitsandbytes is not supported on macOS in many setups; leave it out by default
    echo "Note: 'bitsandbytes' is skipped on macOS by default as it's typically Linux/CUDA-only."

else
    # Assume Linux by default
    echo "Installing PyTorch (Linux/CUDA) using index-url for CUDA 12.6..."
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    EXTRA_PIP_PACKAGES=(
        redis==5.2.1
        jupyter==1.1.1
        numpy==2.1.2
        pandas==2.3.0
        matplotlib==3.10.1
        scipy==1.15.2
        scikit-learn==1.6.1
        tqdm==4.67.1
        g2p_en==2.1.0
        h5py==3.13.0
        omegaconf==2.3.0
        editdistance==0.8.1
        -e .
        huggingface-hub==0.33.1
        transformers==4.53.0
        tokenizers==0.21.2
        accelerate==1.8.1
        bitsandbytes==0.46.0
    )
fi

echo
echo "Installing additional Python packages..."
python -m pip install "${EXTRA_PIP_PACKAGES[@]}"

echo
echo "Setup complete!"
echo "Verify it worked by activating the conda environment with: conda activate ${ENV_NAME}"
echo "On macOS, to use MPS (Metal) backend in PyTorch, set device='mps' in your code or use torch.device('mps') when available."

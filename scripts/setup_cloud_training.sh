#!/bin/bash
# Cloud Training Setup Script for TinyRecursiveInference
# This script automates environment setup for various cloud GPU platforms

set -e  # Exit on error

echo "========================================="
echo "TinyRecursiveInference Cloud Setup"
echo "========================================="
echo ""

# Detect GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✓ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    echo "  Memory: ${GPU_MEM} MB"
    echo ""
else
    echo "⚠ WARNING: No GPU detected! Training will be very slow."
    echo ""
fi

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip wheel setuptools -q
echo "✓ pip upgraded"
echo ""

# Detect if we're on Colab
if [ -d "/content" ] && [ -f "/usr/local/lib/python*/dist-packages/google/colab/_ipython.py" ]; then
    echo "✓ Google Colab environment detected"
    PLATFORM="colab"
    # Colab already has PyTorch installed
    echo "  Using pre-installed PyTorch"
else
    echo "Installing PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
    echo "✓ PyTorch installed"
    PLATFORM="other"
fi
echo ""

# Install requirements
echo "Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt -q
    echo "✓ Dependencies installed"
else
    echo "⚠ requirements.txt not found!"
    exit 1
fi
echo ""

# Optional: adam-atan2 optimizer
read -p "Install adam-atan2 optimizer? (slower, optional) [y/N]: " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing adam-atan2 (this may take 5-10 minutes)..."
    pip install --no-cache-dir --no-build-isolation adam-atan2
    echo "✓ adam-atan2 installed"
else
    echo "⊘ Skipping adam-atan2 (will use standard PyTorch Adam)"
fi
echo ""

# Check if dataset exists
echo "Checking for datasets..."
if [ -d "data" ] && [ "$(ls -A data)" ]; then
    echo "✓ Found existing datasets:"
    ls -1 data/
else
    echo "⚠ No datasets found in data/"
    echo ""
    read -p "Prepare ARC-AGI dataset now? [Y/n]: " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        echo "Preparing ARC-AGI dataset..."

        # Choose augmentation level based on GPU memory
        if [ ! -z "$GPU_MEM" ] && [ "$GPU_MEM" -gt 30000 ]; then
            NUM_AUG=1000
            echo "  High memory GPU detected, using 1000 augmentations"
        else
            NUM_AUG=100
            echo "  Limited memory GPU, using 100 augmentations"
        fi

        python -m dataset.build_arc_dataset \
          --input-file-prefix kaggle/combined/arc-agi \
          --output-dir data/arc1-cloud-aug-${NUM_AUG} \
          --subsets training evaluation \
          --test-set-name evaluation \
          --num-aug ${NUM_AUG}

        echo "✓ Dataset prepared: data/arc1-cloud-aug-${NUM_AUG}"
    fi
fi
echo ""

# Setup W&B (optional)
read -p "Configure Weights & Biases? [y/N]: " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Please enter your W&B API key (from https://wandb.ai/authorize):"
    read -s WANDB_KEY
    wandb login $WANDB_KEY
    echo "✓ W&B configured"
else
    export WANDB_MODE=disabled
    echo "⊘ W&B disabled (metrics will print to console only)"
fi
echo ""

# Recommend config based on GPU
echo "========================================="
echo "Setup Complete! 🎉"
echo "========================================="
echo ""
echo "Recommended configuration based on your GPU:"
echo ""

if [ ! -z "$GPU_MEM" ]; then
    if [ "$GPU_MEM" -gt 35000 ]; then
        CONFIG="cfg_cloud_high_memory"
        BATCH_SIZE=256
        ARCH="trm"
        echo "  Config: config/${CONFIG}.yaml"
        echo "  Architecture: Standard TRM (7M params)"
        echo "  Batch size: ${BATCH_SIZE}"
    elif [ "$GPU_MEM" -gt 14000 ]; then
        CONFIG="cfg_cloud_single_gpu"
        BATCH_SIZE=64
        ARCH="trm"
        echo "  Config: config/${CONFIG}.yaml"
        echo "  Architecture: Standard TRM (7M params)"
        echo "  Batch size: ${BATCH_SIZE}"
    else
        CONFIG="cfg_cloud_single_gpu"
        BATCH_SIZE=32
        ARCH="trm_tiny"
        echo "  Config: config/${CONFIG}.yaml"
        echo "  Architecture: Tiny TRM (2M params)"
        echo "  Batch size: ${BATCH_SIZE}"
        echo "  ⚠ Limited memory - using reduced model"
    fi
else
    CONFIG="cfg_cloud_single_gpu"
    BATCH_SIZE=64
    ARCH="trm"
    echo "  Config: config/${CONFIG}.yaml (default)"
    echo "  Architecture: Standard TRM (7M params)"
    echo "  Batch size: ${BATCH_SIZE}"
fi

echo ""
echo "To start training, run:"
echo ""
echo "  python pretrain.py \\"
echo "    --config-name ${CONFIG} \\"
echo "    arch=${ARCH} \\"
echo "    global_batch_size=${BATCH_SIZE} \\"
echo "    +run_name=\"my_experiment\""
echo ""
echo "For more help, see: docs/CLOUD_TRAINING.md"
echo ""

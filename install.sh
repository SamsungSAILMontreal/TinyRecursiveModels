#!/bin/bash
set -e

echo "=== TinyRecursiveModels Setup ==="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Source the shell config to make uv available
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    fi
fi

echo "uv version: $(uv --version)"

# Sync dependencies
echo "Installing dependencies..."
uv sync

echo "Downlading datasets..."
uv run python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To run scripts, use: uv run python <script.py>"
echo "Example: uv run python pretrain.py arch=trm ..."
echo ""
echo "Optional: Login to Weights & Biases for experiment tracking:"
echo "  uv run wandb login YOUR-LOGIN"

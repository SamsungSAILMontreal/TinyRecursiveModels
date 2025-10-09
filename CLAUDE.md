# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is the implementation of **Tiny Recursion Model (TRM)**, a recursive reasoning approach that achieves 45% accuracy on ARC-AGI-1 and 8% on ARC-AGI-2 using only a 7M parameter neural network. The core innovation is using recursive reasoning with tiny models instead of massive foundational models.

Paper: https://arxiv.org/abs/2510.04871

## Key Architecture Concepts

### Recursive Reasoning Architecture

TRM works by recursively improving predictions through two levels of iteration:

- **High-level cycles (H_cycles)**: Outer loop that updates the answer embedding `y`
- **Low-level cycles (L_cycles)**: Inner loop that updates latent reasoning state `z` given question `x`, current answer `y`, and current latent `z`

The model maintains a carry state (`TinyRecursiveReasoningModel_ACTV1Carry`) containing:
- `inner_carry`: Latent states `z_H` and `z_L`
- `steps`: Current iteration count for adaptive computation time (ACT)
- `halted`: Boolean flags indicating when sequences have finished reasoning
- `current_data`: The batch data being processed

### Main Components

1. **Model Entry Point**: [pretrain.py](pretrain.py) - Main training script using PyTorch DDP
2. **Core Architecture**: [models/recursive_reasoning/trm.py](models/recursive_reasoning/trm.py) - TRM implementation
3. **Dataset Handling**: [puzzle_dataset.py](puzzle_dataset.py) - Loads and batches puzzle examples
4. **Model Variants**:
   - `trm`: Standard TRM with attention layers
   - `trm_singlez`: TRM with single latent state
   - `trm_hier6`: TRM with hierarchical structure
   - `hrm`: Hierarchical Reasoning Model baseline
   - `transformers_baseline`: Standard transformer baseline

### Configuration System

Uses Hydra for hierarchical configs:
- Base config: [config/cfg_pretrain.yaml](config/cfg_pretrain.yaml)
- Architecture configs: `config/arch/*.yaml`
- Override via command line: `arch=trm arch.L_layers=2 arch.H_cycles=3`

Key config parameters:
- `global_batch_size`: Total batch size across all GPUs
- `H_cycles`/`L_cycles`: Number of recursive iterations
- `L_layers`: Number of transformer layers in reasoning module
- `halt_max_steps`: Maximum ACT steps
- `puzzle_emb_ndim`: Dimension of learned puzzle embeddings

## Common Development Commands

### Environment Setup

```bash
pip install --upgrade pip wheel setuptools
pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
pip install -r requirements.txt
pip install --no-cache-dir --no-build-isolation adam-atan2
wandb login YOUR-LOGIN  # Optional: for experiment tracking
```

### Dataset Preparation

```bash
# ARC-AGI-1
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation

# ARC-AGI-2
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2

# Sudoku-Extreme
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000

# Maze-Hard
python dataset/build_maze_dataset.py
```

### Training

Single GPU:
```bash
python pretrain.py arch=trm data_paths="[data/arc1concept-aug-1000]" +run_name="my_run"
```

Multi-GPU (DDP with torchrun):
```bash
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  arch=trm \
  data_paths="[data/arc1concept-aug-1000]" \
  arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 \
  +run_name="my_run" ema=True
```

Multi-node training: See [AGENTS.md](AGENTS.md) for detailed instructions on distributed training across multiple machines.

### Testing and Evaluation

Evaluation runs automatically during training at `eval_interval` epochs. To run evaluation only:

```bash
python pretrain.py \
  arch=trm \
  data_paths="[data/arc1concept-aug-1000]" \
  load_checkpoint="checkpoints/path/to/checkpoint" \
  epochs=0  # Skip training
```

Evaluators are defined in `config/cfg_pretrain.yaml` under the `evaluators` key.

### Checkpointing

- Checkpoints saved to: `checkpoints/{project_name}/{run_name}/`
- Load checkpoint: `load_checkpoint=path/to/checkpoint`
- Auto-checkpoint: `checkpoint_every_eval=True`

## Code Structure and Data Flow

### Training Loop Flow

1. **Initialization** ([pretrain.py:536-596](pretrain.py))
   - Parse Hydra config and sync across ranks
   - Create dataloaders with DDP-aware sampling
   - Initialize model with `create_model()` which:
     - Loads model class dynamically via `load_model_class()`
     - Wraps with loss head (e.g., `ACTLossHead`)
     - Applies `torch.compile()` unless `DISABLE_COMPILE` is set
   - Setup optimizers: `CastedSparseEmbeddingSignSGD_Distributed` for puzzle embeddings, `AdamATan2` for weights
   - Initialize EMA helper if `ema=True`

2. **Training Iteration** ([pretrain.py:602-613](pretrain.py))
   - For each batch, call `train_batch()` which:
     - Initializes carry state on first step
     - Forward pass through model with carry
     - Backward pass with gradient scaling by `1/global_batch_size`
     - All-reduce gradients across GPUs
     - Apply optimizer with cosine LR schedule
   - Update EMA if enabled

3. **Evaluation** ([pretrain.py:345-486](pretrain.py))
   - Switch to EMA model if `ema=True`
   - Process all test batches
   - Run custom evaluators (e.g., ARC accuracy)
   - Save predictions if `eval_save_outputs` is set

### Dataset Format

Datasets are stored as NumPy arrays in `{output_dir}/{split}/` with:
- `{set_name}__inputs.npy`: Input sequences (shape: [N, seq_len])
- `{set_name}__labels.npy`: Target sequences (shape: [N, seq_len])
- `{set_name}__puzzle_identifiers.npy`: Puzzle IDs for embedding lookup (shape: [N])
- `{set_name}__puzzle_indices.npy`: Indices marking puzzle boundaries
- `{set_name}__group_indices.npy`: Indices marking augmentation groups
- `dataset.json`: Metadata with vocab_size, seq_len, etc.

During training, `PuzzleDataset` samples batches by:
1. Shuffling augmentation groups
2. Sampling random examples from each group
3. Packing into `global_batch_size` batches
4. Splitting across DDP ranks

### Adaptive Computation Time (ACT)

The model uses ACT to dynamically adjust computation:
- Q-network predicts halt/continue signals
- Training explores different halt times via `halt_exploration_prob`
- Evaluation uses max steps for consistent batching
- Q-learning loss trains the halt decision

## Important Notes

### Distributed Training

- Uses PyTorch DDP (DistributedDataParallel)
- NCCL backend for GPU communication
- GLOO backend for CPU operations in evaluators
- Gradients are all-reduced after backward pass
- Only rank 0 logs to wandb and saves checkpoints

### Memory Management

- Forward pass uses `bfloat16` by default
- Model is compiled with `torch.compile()` for speed
- During evaluation, predictions are moved to CPU to save GPU memory
- Carry states are detached to prevent gradient accumulation

### Puzzle Embeddings

- Learned per-puzzle embeddings stored in `CastedSparseEmbedding`
- Zero-initialized for new puzzles
- Trained with separate optimizer and LR (`puzzle_emb_lr`)
- Resized automatically when loading checkpoints with different puzzle counts

### Architecture Customization

To create a new architecture variant:
1. Add config in `config/arch/your_arch.yaml`
2. Implement model in `models/recursive_reasoning/your_arch.py`
3. Follow the interface: `__init__(config_dict)`, `initial_carry(batch)`, `forward(carry, batch)`
4. Train with: `arch=your_arch`

### Gradient Checkpointing

Only the last H-cycle has gradients enabled ([trm.py:208-216](models/recursive_reasoning/trm.py)). Earlier cycles run with `torch.no_grad()` to save memory while still performing recursive reasoning.

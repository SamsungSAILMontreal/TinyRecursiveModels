# TinyRecursiveInference: PhD-Level Implementation Plan

**Version**: 1.0
**Date**: 2025-10-09
**Status**: Design Document - Ready for Review

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Codebase Architecture Analysis](#codebase-architecture-analysis)
3. [Current State Assessment](#current-state-assessment)
4. [Proposed TinyRecursiveInference Architecture](#proposed-architecture)
5. [Implementation Roadmap](#implementation-roadmap)
6. [Detailed Technical Specifications](#technical-specifications)
7. [Integration Points](#integration-points)
8. [Testing & Validation Strategy](#testing-strategy)
9. [Deployment & Scaling Considerations](#deployment-considerations)
10. [Risk Analysis & Mitigation](#risk-analysis)
11. [Code Artifacts](#code-artifacts)

---

## 1. Executive Summary {#executive-summary}

### 1.1 Project Overview

**TinyRecursiveInference** is a comprehensive end-to-end system for:
1. **Dataset Publishing**: Automated upload of prepared ARC/Sudoku/Maze datasets to Hugging Face Hub
2. **Training with Telemetry**: Enhanced `pretrain.py` with real-time W&B metrics streaming
3. **Model Publishing**: Automated checkpoint + model card upload to Hugging Face Model Hub
4. **Interactive Inference**: Gradio application for visual puzzle solving with intermediate reasoning visualization

### 1.2 Design Philosophy

Drawing from the **Community-In-The-Loop Fine-Tuning Pipeline** architecture provided, we adapt:
- **Multi-platform publishing**: HF for datasets/models, W&B for training telemetry
- **Graceful degradation**: Optional dependencies (wandb, gradio) with fallback behavior
- **Modular orchestration**: Independent stages that can be run individually or as a pipeline
- **Reproducibility**: All artifacts versioned and traceable across platforms

### 1.3 Key Innovation

Unlike the OpenAI fine-tuning pipeline which uses proprietary APIs, TinyRecursiveInference:
- Works with **open-source PyTorch models** trained from scratch
- Maintains **full control** over training loop, checkpoints, and evaluation
- Provides **visual reasoning transparency** through intermediate state visualization
- Enables **recursive reasoning analysis** unique to TRM architecture

---

## 2. Codebase Architecture Analysis {#codebase-architecture-analysis}

### 2.1 Core Components

#### 2.1.1 Training Loop ([pretrain.py](pretrain.py))

**Responsibility**: Multi-GPU distributed training orchestration

**Key Functions**:
- `launch()`: Main Hydra entry point (line 536)
- `init_train_state()`: Model + optimizer initialization (line 217)
- `train_batch()`: Single batch forward/backward with gradient accumulation (line 289)
- `evaluate()`: Full evaluation pass with custom evaluators (line 345)
- `save_train_state()`: Checkpoint persistence (line 235)
- `load_checkpoint()`: Checkpoint restoration with puzzle embedding resizing (line 244)

**Distributed Training Architecture**:
- **Backend**: NCCL for GPU all-reduce, GLOO for CPU operations in evaluators
- **Gradient Synchronization**: Manual all-reduce after backward (line 308-311)
- **Data Parallelism**: Each rank processes `global_batch_size // world_size` samples
- **Rank 0 Privileges**: Logging (wandb), checkpointing, evaluation result aggregation

**Current W&B Integration**:
- Initialization at line 590: `wandb.init(project=config.project_name, ...)`
- Training metrics logged at line 610: `wandb.log(metrics, step=train_state.step)`
- Evaluation metrics logged at line 636: `wandb.log(metrics, step=train_state.step)`
- Code snapshot logged at line 511: `wandb.run.log_code(config.checkpoint_path)`

**Limitation**: No structured artifact logging, hyperparameter tracking, or checkpoint versioning to W&B.

#### 2.1.2 TRM Architecture ([models/recursive_reasoning/trm.py](models/recursive_reasoning/trm.py))

**Core Innovation**: Recursive reasoning via nested H/L-level cycles

**Carry State Structure** (`TinyRecursiveReasoningModel_ACTV1Carry`):
```python
@dataclass
class TinyRecursiveReasoningModel_ACTV1Carry:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry  # z_H, z_L latents
    steps: torch.Tensor  # ACT step count per sequence
    halted: torch.Tensor  # Boolean halt flags
    current_data: Dict[str, torch.Tensor]  # Batch data (inputs, labels, puzzle_identifiers)
```

**Forward Pass** (line 196-222):
1. Encode inputs + puzzle embeddings → `input_embeddings`
2. Run H_cycles-1 iterations **without gradients** (memory optimization)
3. Run final H_cycle **with gradients** (line 214-216)
4. Within each H-cycle: L_cycles iterations update `z_L`, then update `z_H`
5. Decode `z_H` → logits, Q-values (halt/continue)

**Adaptive Computation Time (ACT)**: Q-learning network predicts when to halt reasoning (line 221)

**Key Insight**: Only the last H-cycle requires gradients, earlier cycles are "search" steps.

#### 2.1.3 Dataset Format ([puzzle_dataset.py](puzzle_dataset.py))

**On-Disk Structure**:
```
data/{dataset_name}/
├── train/
│   ├── dataset.json          # Metadata: vocab_size, seq_len, num_puzzle_identifiers
│   ├── training__inputs.npy  # Shape: [N, seq_len], dtype: int32
│   ├── training__labels.npy
│   ├── training__puzzle_identifiers.npy  # Shape: [N], dtype: int32
│   ├── training__puzzle_indices.npy      # Boundaries for each puzzle
│   └── training__group_indices.npy       # Boundaries for augmentation groups
└── test/
    └── evaluation__*.npy
```

**Sampling Strategy** (line 16-40):
- Shuffle augmentation groups each epoch
- Sample random examples from each group
- Pack into `global_batch_size` batches
- DDP shards batches across ranks

**Memory Optimization**: Use `mmap_mode="r"` for inputs/labels (line 124)

#### 2.1.4 Loss and Metrics ([models/losses.py](models/losses.py))

**ACTLossHead** (line 41):
- **Language Model Loss**: Stablemax cross-entropy on logits vs labels (line 87)
- **Q-Halt Loss**: BCE predicting sequence correctness (line 88)
- **Q-Continue Loss**: Optional bootstrapped Q-learning target (line 94-96)
- **Combined Loss**: `lm_loss + 0.5 * (q_halt_loss + q_continue_loss)` (line 102)

**Tracked Metrics** (line 74-83):
- `accuracy`: Token-level accuracy
- `exact_accuracy`: Full sequence correctness
- `q_halt_accuracy`: Q-network prediction accuracy
- `steps`: Average ACT steps taken

#### 2.1.5 Evaluators ([evaluators/arc.py](evaluators/arc.py))

**ARC Evaluator** (line 39):
- Collects predictions during evaluation (line 69)
- Applies inverse augmentation transforms (line 91-95)
- Votes across augmented variants using Q-values as confidence (line 128-141)
- Computes pass@K metrics for K ∈ {1, 2, 5, 10, 100, 1000} (line 144-149)
- Saves Kaggle submission format (line 163)

**Distributed Evaluation**: Uses CPU GLOO group for gather_object (line 110)

### 2.2 Configuration System

**Hydra Hierarchy**:
- Base: [config/cfg_pretrain.yaml](config/cfg_pretrain.yaml)
- Architecture overrides: `config/arch/{trm,hrm,transformers_baseline}.yaml`
- Runtime overrides: Command-line `arch.L_layers=2 +run_name="my_run"`

**Key Parameters**:
- `global_batch_size`: Total across all GPUs (default: 768)
- `H_cycles`: Outer reasoning loops (default: 3)
- `L_cycles`: Inner latent updates (default: 6)
- `halt_max_steps`: Maximum ACT iterations (default: 16)
- `puzzle_emb_ndim`: Learned puzzle embedding dimension (default: 512)

### 2.3 Existing TinyRecursiveInference Skeleton

**Current Implementation** (created but incomplete):
- [tiny_recursive_inference/config.py](tiny_recursive_inference/config.py): Config dataclasses
- [tiny_recursive_inference/publishers.py](tiny_recursive_inference/publishers.py): HF upload helpers
- [tiny_recursive_inference/pipeline.py](tiny_recursive_inference/pipeline.py): Orchestration skeleton

**What's Missing**:
1. ✅ Dataset publishing: **Implemented** (`publish_dataset`)
2. ✅ Model publishing: **Implemented** (`publish_model`)
3. ⚠️ W&B training telemetry: **Partially exists** in pretrain.py, needs enhancement
4. ❌ Gradio inference app: **Not implemented**
5. ❌ Checkpoint auto-publishing hook: **Not implemented**
6. ❌ Training progress hooks: **Not implemented**

---

## 3. Current State Assessment {#current-state-assessment}

### 3.1 What Works

✅ **Training Pipeline**:
- Multi-GPU DDP with NCCL
- Checkpoint save/load with puzzle embedding resizing
- EMA (Exponential Moving Average) support
- Cosine LR schedule with warmup
- Basic W&B logging (metrics only)

✅ **Model Architecture**:
- TRM with recursive reasoning (H/L cycles)
- Adaptive Computation Time (ACT)
- Sparse puzzle embeddings with specialized optimizer
- Gradient checkpointing (only last H-cycle)

✅ **Dataset Infrastructure**:
- ARC, Sudoku, Maze builders
- Memory-mapped data loading
- Augmentation group sampling
- DDP-aware batching

✅ **Evaluation**:
- ARC pass@K metrics
- Distributed voting across augmentations
- Kaggle submission generation

### 3.2 What Needs Enhancement

⚠️ **W&B Integration**:
- **Missing**: Checkpoint artifact logging
- **Missing**: Config/hyperparameter tracking
- **Missing**: Model architecture visualization
- **Missing**: Training curve smoothing
- **Missing**: Custom charts (ACT step distribution, Q-value histograms)

⚠️ **Checkpoint Management**:
- **Issue**: Only saves latest checkpoint per eval interval
- **Missing**: Best checkpoint tracking (by validation accuracy)
- **Missing**: Checkpoint metadata (epoch, step, metrics)

### 3.3 What Needs Implementation

❌ **Gradio Inference App**:
- Visual puzzle input (JSON upload or grid editor)
- Intermediate reasoning state visualization
- ACT step-by-step playback
- Confidence heatmaps per cell
- Model selection (load from HF or local)

❌ **HF Model Publishing**:
- Automated post-training upload
- Model card generation with metrics
- Config file bundling
- Inference example code

❌ **Pipeline Orchestration**:
- Stage dependency management
- Graceful failure handling
- Progress tracking across stages

---

## 4. Proposed TinyRecursiveInference Architecture {#proposed-architecture}

### 4.1 System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                   TinyRecursiveInference                        │
│                     Full Pipeline                               │
└──────────────────────┬──────────────────────────────────────────┘
                       │
       ┌───────────────┼───────────────┐
       │               │               │
       ▼               ▼               ▼
┌──────────┐    ┌──────────┐    ┌──────────┐
│  Stage 1 │    │  Stage 2 │    │  Stage 3 │
│ Dataset  │───▶│ Training │───▶│  Model   │
│Publishing│    │ with W&B │    │Publishing│
└──────────┘    └──────────┘    └──────────┘
      │               │               │
      ▼               ▼               ▼
  ┌───────┐     ┌───────┐     ┌───────┐
  │  HF   │     │ W&B   │     │  HF   │
  │Dataset│     │Project│     │ Model │
  │  Hub  │     │       │     │  Hub  │
  └───────┘     └───────┘     └───────┘
                      │
                      │
                      ▼
                ┌──────────┐
                │  Stage 4 │
                │  Gradio  │
                │Inference │
                └──────────┘
                      │
                      ▼
                ┌───────┐
                │  HF   │
                │ Space │
                └───────┘
```

### 4.2 Stage Breakdown

#### Stage 1: Dataset Publishing
**Trigger**: Before training starts
**Artifacts**: HF Dataset with README
**Implementation**: `tiny_recursive_inference/publishers.py::publish_dataset()` ✅ **DONE**

#### Stage 2: Enhanced Training with W&B
**Trigger**: User-initiated via CLI or pipeline
**Artifacts**: Checkpoints, W&B run with charts
**Implementation**: Enhanced `pretrain.py` + W&B callbacks

#### Stage 3: Model Publishing
**Trigger**: After training completes or at checkpoints
**Artifacts**: HF Model Hub with config + weights
**Implementation**: `tiny_recursive_inference/publishers.py::publish_model()` ✅ **DONE**

#### Stage 4: Gradio Inference
**Trigger**: User-initiated for deployed models
**Artifacts**: HF Space or local Gradio server
**Implementation**: New `tiny_recursive_inference/gradio_app.py`

### 4.3 File Structure

```
TinyRecursiveModels/
├── pretrain.py                          # [ENHANCE] Add W&B artifact logging
├── tiny_recursive_inference/
│   ├── __init__.py                      # ✅ Exists
│   ├── config.py                        # ✅ Exists (may need extensions)
│   ├── publishers.py                    # ✅ Exists
│   ├── pipeline.py                      # ✅ Exists (enhance with hooks)
│   ├── wandb_callbacks.py               # [NEW] W&B training callbacks
│   ├── model_loader.py                  # [NEW] HF/local checkpoint loader
│   ├── gradio_app.py                    # [NEW] Interactive inference UI
│   └── visualizations.py                # [NEW] Reasoning state plots
├── scripts/
│   ├── run_full_pipeline.py             # [NEW] CLI orchestrator
│   └── publish_checkpoint.py            # [NEW] Post-training publish
├── config/
│   └── inference_config.yaml            # [NEW] TinyRecursiveInference config
└── ClaudePlan.md                        # [NEW] This document
```

---

## 5. Implementation Roadmap {#implementation-roadmap}

### Phase 1: W&B Enhanced Training (Priority: HIGH)

**Goal**: Stream rich telemetry during training without modifying core loop

**Tasks**:
1. Create `wandb_callbacks.py` with callback hooks:
   - `on_train_batch_end(metrics, step)`
   - `on_eval_end(metrics, checkpoint_path, step)`
   - `on_training_end(final_checkpoint)`

2. Enhance `pretrain.py`:
   - Log hyperparameters as wandb config at initialization
   - Track best checkpoint by validation `exact_accuracy`
   - Log checkpoints as wandb artifacts with metadata
   - Add custom charts:
     - ACT step distribution histogram
     - Q-value confidence over time
     - Per-puzzle accuracy heatmap

3. Add checkpoint metadata:
   - Save `checkpoint_metadata.json` alongside weights
   - Include: epoch, step, train/eval metrics, config hash

**Estimated LOC**: ~200 lines
**Testing**: Single-GPU Sudoku run with W&B logging enabled

### Phase 2: Model Publishing Automation (Priority: HIGH)

**Goal**: Automatically upload checkpoints to HF after training

**Tasks**:
1. Create `scripts/publish_checkpoint.py`:
   - Load checkpoint + metadata
   - Generate model card from template
   - Upload via `publish_model()`

2. Add hook to `pretrain.py`:
   - Optional flag `auto_publish_hf=True`
   - At end of training, call publish script
   - Environment variables: `HUGGINGFACE_MODEL_REPO`

3. Create model card template:
   - Architecture summary (H_cycles, L_cycles, hidden_size)
   - Training dataset reference
   - Metrics table (train/eval loss, accuracy)
   - Inference code example

**Estimated LOC**: ~150 lines
**Testing**: Publish a trained Sudoku checkpoint to private HF repo

### Phase 3: Gradio Inference App (Priority: MEDIUM)

**Goal**: Visual interface for interactive puzzle solving

**Tasks**:
1. Create `model_loader.py`:
   - Load checkpoint from HF Hub or local path
   - Initialize model with config
   - Handle device placement (CPU/CUDA)

2. Create `visualizations.py`:
   - Grid rendering (30x30 colored cells)
   - Latent state heatmaps (z_H, z_L)
   - ACT step timeline
   - Q-value confidence bars

3. Create `gradio_app.py`:
   - **Input Tab**: JSON upload or manual grid editor
   - **Inference Tab**: Run button → show predictions
   - **Reasoning Tab**: Step-by-step ACT visualization
   - **Settings Tab**: Model selection, H_cycles override

4. Deployment options:
   - Local: `python -m tiny_recursive_inference.gradio_app`
   - HF Space: Create `app.py` wrapper + `requirements.txt`

**Estimated LOC**: ~400 lines
**Testing**: Load trained ARC-AGI-1 model, solve sample puzzles

### Phase 4: Full Pipeline Integration (Priority: LOW)

**Goal**: One-command dataset→training→publishing→inference

**Tasks**:
1. Create `scripts/run_full_pipeline.py`:
   - Parse `config/inference_config.yaml`
   - Execute stages sequentially with error handling
   - Log progress to console + W&B

2. Add resume capability:
   - Detect incomplete stages (e.g., dataset published but training not started)
   - Skip completed stages or force re-run

3. Documentation:
   - Update `README.md` with TinyRecursiveInference section
   - Create `docs/INFERENCE_GUIDE.md` with examples

**Estimated LOC**: ~250 lines
**Testing**: Full pipeline run with toy dataset

---

## 6. Detailed Technical Specifications {#technical-specifications}

### 6.1 W&B Callback System

**File**: `tiny_recursive_inference/wandb_callbacks.py`

```python
from typing import Dict, Optional
import wandb
import torch
import os
from pathlib import Path

class WandBTrainingCallback:
    """Enhanced W&B logging for TinyRecursiveInference."""

    def __init__(
        self,
        project: str,
        run_name: str,
        config: Dict,
        enabled: bool = True,
        log_artifacts: bool = True,
        artifact_type: str = "model"
    ):
        self.enabled = enabled and (wandb is not None)
        self.log_artifacts = log_artifacts
        self.artifact_type = artifact_type
        self.best_metric: float = 0.0
        self.best_checkpoint_path: Optional[str] = None

        if self.enabled:
            wandb.init(
                project=project,
                name=run_name,
                config=config,
                settings=wandb.Settings(_disable_stats=True)
            )
            # Custom charts
            wandb.define_metric("train/step")
            wandb.define_metric("train/*", step_metric="train/step")
            wandb.define_metric("eval/*", step_metric="train/step")

    def on_train_batch_end(self, metrics: Dict[str, float], step: int):
        """Log training metrics after each batch."""
        if self.enabled:
            wandb.log(metrics, step=step)

    def on_eval_end(
        self,
        metrics: Dict[str, float],
        checkpoint_path: Optional[str],
        step: int
    ):
        """Log evaluation metrics and optionally save checkpoint as artifact."""
        if not self.enabled:
            return

        # Log metrics
        wandb.log(metrics, step=step)

        # Track best checkpoint
        eval_accuracy = metrics.get("ARC/pass@1", 0.0)
        if eval_accuracy > self.best_metric:
            self.best_metric = eval_accuracy
            self.best_checkpoint_path = checkpoint_path
            wandb.run.summary["best_accuracy"] = eval_accuracy  # type: ignore
            wandb.run.summary["best_checkpoint"] = checkpoint_path  # type: ignore

        # Log checkpoint as artifact
        if self.log_artifacts and checkpoint_path:
            self._log_checkpoint_artifact(checkpoint_path, step, eval_accuracy)

    def _log_checkpoint_artifact(self, checkpoint_path: str, step: int, accuracy: float):
        """Upload checkpoint to W&B artifacts."""
        artifact = wandb.Artifact(
            name=f"checkpoint-step-{step}",
            type=self.artifact_type,
            metadata={
                "step": step,
                "accuracy": accuracy,
                "is_best": (accuracy == self.best_metric)
            }
        )

        # Add checkpoint files
        checkpoint_dir = Path(checkpoint_path).parent
        for file in checkpoint_dir.glob("step_*"):
            artifact.add_file(str(file))

        # Add config
        config_file = checkpoint_dir / "all_config.yaml"
        if config_file.exists():
            artifact.add_file(str(config_file))

        wandb.log_artifact(artifact)

    def on_training_end(self, final_checkpoint: Optional[str]):
        """Finalize W&B run."""
        if self.enabled:
            if self.best_checkpoint_path:
                wandb.run.summary["best_checkpoint_path"] = self.best_checkpoint_path  # type: ignore
            wandb.finish()
```

**Integration into pretrain.py**:

```python
# In launch() function, after line 590
wandb_callback = None
if RANK == 0:
    from tiny_recursive_inference.wandb_callbacks import WandBTrainingCallback

    wandb_callback = WandBTrainingCallback(
        project=config.project_name,
        run_name=config.run_name,
        config=config.model_dump(),
        enabled=os.getenv("WANDB_DISABLED", "false").lower() != "true",
        log_artifacts=config.checkpoint_every_eval
    )

# In training loop, replace line 610
if RANK == 0 and metrics is not None:
    if wandb_callback:
        wandb_callback.on_train_batch_end(metrics, train_state.step)
    else:
        wandb.log(metrics, step=train_state.step)

# In evaluation section, replace line 636
if RANK == 0 and metrics is not None:
    if wandb_callback:
        wandb_callback.on_eval_end(metrics, config.checkpoint_path, train_state.step)
    else:
        wandb.log(metrics, step=train_state.step)

# At end of training, before line 650
if RANK == 0 and wandb_callback:
    wandb_callback.on_training_end(config.checkpoint_path)
```

### 6.2 Model Loader

**File**: `tiny_recursive_inference/model_loader.py`

```python
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import yaml
import torch
from torch import nn

from utils.functions import load_model_class
from puzzle_dataset import PuzzleDatasetMetadata


def load_trm_checkpoint(
    checkpoint_dir: str,
    checkpoint_name: Optional[str] = None,
    device: str = "cuda",
    compile: bool = False
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load a TRM checkpoint from local directory or HF Hub.

    Args:
        checkpoint_dir: Path to checkpoint directory or HF repo ID
        checkpoint_name: Specific checkpoint file (e.g., "step_12345"),
                        defaults to latest
        device: Device to load model on
        compile: Whether to torch.compile the model

    Returns:
        (model, config_dict): Loaded model and configuration
    """
    checkpoint_path = Path(checkpoint_dir)

    # If HF repo ID, download first
    if not checkpoint_path.exists() and "/" in checkpoint_dir:
        from huggingface_hub import snapshot_download
        checkpoint_path = Path(snapshot_download(
            repo_id=checkpoint_dir,
            repo_type="model"
        ))

    # Load config
    config_file = checkpoint_path / "all_config.yaml"
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Find checkpoint file
    if checkpoint_name:
        checkpoint_file = checkpoint_path / checkpoint_name
    else:
        checkpoints = sorted(checkpoint_path.glob("step_*"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_path}")
        checkpoint_file = checkpoints[-1]  # Latest

    # Build model
    arch_config = config["arch"]
    model_cfg = {
        **arch_config,
        "batch_size": 1,  # Inference batch size
        "vocab_size": config.get("vocab_size", 14),  # ARC vocab
        "seq_len": config.get("seq_len", 900),  # 30x30 grid
        "num_puzzle_identifiers": config.get("num_puzzle_identifiers", 1000),
        "causal": False
    }

    # Instantiate
    model_cls = load_model_class(arch_config["name"])
    loss_head_cls = load_model_class(arch_config["loss"]["name"])

    with torch.device(device):
        model = model_cls(model_cfg)
        model = loss_head_cls(model, **arch_config["loss"])

        # Load weights
        state_dict = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(state_dict, assign=True)

        model.eval()

        if compile:
            model = torch.compile(model)  # type: ignore

    return model, config


def prepare_puzzle_batch(
    puzzle_input: torch.Tensor,
    puzzle_identifier: int = 0,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    Prepare a single puzzle for inference.

    Args:
        puzzle_input: [seq_len] tensor of input tokens
        puzzle_identifier: Puzzle ID for embedding lookup
        device: Device to place batch on

    Returns:
        Batch dict compatible with TRM forward pass
    """
    return {
        "inputs": puzzle_input.unsqueeze(0).to(device),
        "labels": torch.zeros_like(puzzle_input.unsqueeze(0)).to(device),
        "puzzle_identifiers": torch.tensor([puzzle_identifier], device=device)
    }
```

### 6.3 Gradio App

**File**: `tiny_recursive_inference/gradio_app.py`

```python
import gradio as gr
import torch
import numpy as np
from typing import List, Tuple, Optional
import json

from .model_loader import load_trm_checkpoint, prepare_puzzle_batch
from .visualizations import render_grid, plot_reasoning_states


class TRMInferenceApp:
    """Gradio interface for TRM puzzle solving."""

    def __init__(
        self,
        checkpoint_dir: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.model, self.config = load_trm_checkpoint(
            checkpoint_dir,
            device=device,
            compile=False
        )
        self.carry = None
        self.reasoning_history = []

    def parse_puzzle_json(self, json_str: str) -> torch.Tensor:
        """Convert ARC JSON to token sequence."""
        puzzle = json.loads(json_str)
        input_grid = np.array(puzzle["train"][0]["input"])

        # Pad to 30x30
        padded = np.zeros((30, 30), dtype=np.int32)
        h, w = input_grid.shape
        padded[:h, :w] = input_grid

        # Add special tokens (0=BOS, 1=EOS, 2-11=colors)
        tokens = padded.flatten() + 2
        return torch.tensor(tokens, dtype=torch.int32)

    def run_inference(
        self,
        puzzle_json: str,
        max_steps: int = 16,
        visualize_steps: bool = True
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Run TRM inference on puzzle.

        Returns:
            (prediction_grid, reasoning_history)
        """
        # Prepare input
        input_tokens = self.parse_puzzle_json(puzzle_json)
        batch = prepare_puzzle_batch(input_tokens, device=self.device)

        # Initialize carry
        with torch.no_grad():
            carry = self.model.initial_carry(batch)  # type: ignore

        # Run ACT loop
        self.reasoning_history = []
        step = 0

        while step < max_steps:
            with torch.no_grad():
                carry, loss, metrics, preds, all_halted = self.model(
                    carry=carry,
                    batch=batch,
                    return_keys=["logits", "preds", "q_halt_logits"]
                )

            if visualize_steps:
                self.reasoning_history.append({
                    "step": step,
                    "preds": preds["preds"].cpu().numpy(),
                    "q_halt": preds["q_halt_logits"].cpu().numpy(),
                    "halted": carry.halted.cpu().numpy()
                })

            step += 1
            if all_halted:
                break

        # Extract final prediction
        final_preds = preds["preds"][0].cpu().numpy() - 2  # Remove token offset
        prediction_grid = final_preds.reshape(30, 30)

        return prediction_grid, self.reasoning_history

    def create_interface(self):
        """Build Gradio UI."""
        with gr.Blocks(title="TinyRecursiveInference") as app:
            gr.Markdown("# TRM Puzzle Solver")
            gr.Markdown(f"Model: `{self.config['arch']['name']}` | "
                       f"H-cycles: {self.config['arch']['H_cycles']} | "
                       f"L-cycles: {self.config['arch']['L_cycles']}")

            with gr.Tab("Input"):
                puzzle_input = gr.Textbox(
                    label="Puzzle JSON",
                    placeholder='{"train": [{"input": [[0,1],[1,0]], "output": [[1,0],[0,1]]}], "test": [{"input": [[0,1],[1,0]]}]}',
                    lines=10
                )
                max_steps = gr.Slider(1, 32, value=16, step=1, label="Max ACT Steps")
                visualize = gr.Checkbox(value=True, label="Visualize Reasoning")
                solve_btn = gr.Button("Solve Puzzle")

            with gr.Tab("Output"):
                output_grid = gr.Image(label="Prediction")
                confidence = gr.Textbox(label="Q-Halt Confidence")

            with gr.Tab("Reasoning"):
                reasoning_plot = gr.Plot(label="ACT Step Progression")
                step_selector = gr.Slider(0, 15, value=0, step=1, label="View Step")
                step_vis = gr.Image(label="Step Visualization")

            def solve_wrapper(json_str, steps, viz):
                grid, history = self.run_inference(json_str, steps, viz)
                grid_img = render_grid(grid)
                confidence_text = f"Final Q-halt: {history[-1]['q_halt'][0]:.3f}"
                reasoning_plot_fig = plot_reasoning_states(history)
                return grid_img, confidence_text, reasoning_plot_fig

            solve_btn.click(
                fn=solve_wrapper,
                inputs=[puzzle_input, max_steps, visualize],
                outputs=[output_grid, confidence, reasoning_plot]
            )

        return app


def launch_gradio(checkpoint_dir: str, share: bool = False):
    """Launch Gradio app."""
    app = TRMInferenceApp(checkpoint_dir)
    interface = app.create_interface()
    interface.launch(share=share, server_name="0.0.0.0")


if __name__ == "__main__":
    import sys
    checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/latest"
    launch_gradio(checkpoint_dir, share=True)
```

### 6.4 Visualization Utilities

**File**: `tiny_recursive_inference/visualizations.py`

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import List, Dict
import io
from PIL import Image


# ARC color palette (0-9)
ARC_COLORS = [
    "#000000",  # 0: black
    "#0074D9",  # 1: blue
    "#FF4136",  # 2: red
    "#2ECC40",  # 3: green
    "#FFDC00",  # 4: yellow
    "#AAAAAA",  # 5: grey
    "#F012BE",  # 6: magenta
    "#FF851B",  # 7: orange
    "#7FDBFF",  # 8: sky
    "#870C25",  # 9: brown
]


def render_grid(grid: np.ndarray, cmap: str = "arc") -> Image.Image:
    """
    Render a 2D grid as an image.

    Args:
        grid: [H, W] array with values 0-9
        cmap: Color map ("arc" or matplotlib name)

    Returns:
        PIL Image
    """
    if cmap == "arc":
        cmap = ListedColormap(ARC_COLORS)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, which="both", color="white", linewidth=0.5)

    # Convert to PIL
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    return img


def plot_reasoning_states(history: List[Dict]) -> plt.Figure:
    """
    Plot ACT step progression.

    Args:
        history: List of reasoning states from TRMInferenceApp

    Returns:
        Matplotlib figure
    """
    steps = [h["step"] for h in history]
    q_halts = [h["q_halt"][0] for h in history]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, q_halts, marker="o", label="Q-Halt Logit")
    ax.axhline(0, color="red", linestyle="--", label="Halt Threshold")
    ax.set_xlabel("ACT Step")
    ax.set_ylabel("Q-Halt Logit")
    ax.set_title("Reasoning Progression")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def visualize_latent_heatmap(z_tensor: np.ndarray, title: str = "Latent State") -> Image.Image:
    """
    Visualize latent state z_H or z_L as heatmap.

    Args:
        z_tensor: [seq_len, hidden_dim] array
        title: Plot title

    Returns:
        PIL Image
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(z_tensor, cmap="viridis", aspect="auto")
    ax.set_xlabel("Hidden Dimension")
    ax.set_ylabel("Sequence Position")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    return img
```

### 6.5 Full Pipeline Orchestrator

**File**: `scripts/run_full_pipeline.py`

```python
#!/usr/bin/env python3
"""
TinyRecursiveInference Full Pipeline Orchestrator

Usage:
    python scripts/run_full_pipeline.py --config config/inference_config.yaml
"""

import argparse
import yaml
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tiny_recursive_inference import TinyRecursiveInferencePipeline
from tiny_recursive_inference.config import (
    TinyRecursiveInferenceConfig,
    DatasetPublishConfig,
    TrainingLaunchConfig,
    ModelPublishConfig
)


def load_config(config_path: str) -> TinyRecursiveInferenceConfig:
    """Load pipeline config from YAML."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return TinyRecursiveInferenceConfig(
        dataset=DatasetPublishConfig(**config_dict.get("dataset", {})),
        training=TrainingLaunchConfig(**config_dict.get("training", {})),
        model=ModelPublishConfig(**config_dict.get("model", {})),
        project_root=config_dict.get("project_root", ".")
    )


def main():
    parser = argparse.ArgumentParser(description="Run TinyRecursiveInference pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/inference_config.yaml",
        help="Path to pipeline config"
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Skip dataset publishing stage"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training stage"
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip model publishing stage"
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    pipeline = TinyRecursiveInferencePipeline(config)

    print("=" * 60)
    print("TinyRecursiveInference Pipeline Starting")
    print("=" * 60)

    # Stage 1: Dataset Publishing
    if not args.skip_dataset and config.dataset.local_path:
        print("\n[Stage 1/3] Publishing dataset to Hugging Face Hub...")
        try:
            dataset_repo = pipeline.publish_dataset()
            print(f"✓ Dataset published: {dataset_repo}")
        except Exception as e:
            print(f"✗ Dataset publishing failed: {e}")
            return 1
    else:
        print("\n[Stage 1/3] Skipping dataset publishing")

    # Stage 2: Training
    if not args.skip_training:
        print("\n[Stage 2/3] Launching training...")
        try:
            returncode = pipeline.launch_training()
            if returncode != 0:
                print(f"✗ Training failed with code {returncode}")
                return returncode
            print("✓ Training completed")
        except Exception as e:
            print(f"✗ Training failed: {e}")
            return 1
    else:
        print("\n[Stage 2/3] Skipping training")

    # Stage 3: Model Publishing
    if not args.skip_model and config.model.checkpoint_dir:
        print("\n[Stage 3/3] Publishing model to Hugging Face Hub...")
        try:
            model_repo = pipeline.publish_model()
            print(f"✓ Model published: {model_repo}")
        except Exception as e:
            print(f"✗ Model publishing failed: {e}")
            return 1
    else:
        print("\n[Stage 3/3] Skipping model publishing")

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### 6.6 Configuration Template

**File**: `config/inference_config.yaml`

```yaml
# TinyRecursiveInference Pipeline Configuration

project_root: "."

# Stage 1: Dataset Publishing
dataset:
  local_path: "data/arc1concept-aug-1000"
  repo_id: "your-username/arc-trm-dataset"  # Set your HF username
  private: true
  token: null  # Uses HUGGINGFACE_TOKEN env var if null
  add_readme: true
  readme_path: null  # Auto-generate if null
  commit_message: "TinyRecursiveInference dataset upload"
  allow_create: true
  files_include: null  # Include all files
  files_ignore:
    - ".DS_Store"
    - "Thumbs.db"
    - "*.tmp"

# Stage 2: Training
training:
  use_torchrun: true
  nproc_per_node: 4
  nnodes: 1
  node_rank: 0
  rdzv_backend: "c10d"
  rdzv_endpoint: "localhost:29500"
  rdzv_id: null  # Auto-generated if null
  python_executable: "python"
  dry_run: false  # Set to true to print command without running
  checkpoint_dir: "checkpoints/arc1-trm-experiment"
  env:
    WANDB_PROJECT: "tiny-recursive-models"
    # WANDB_API_KEY: "your_key"  # Set in environment
  overrides:
    - "arch=trm"
    - "data_paths=[data/arc1concept-aug-1000]"
    - "arch.L_layers=2"
    - "arch.H_cycles=3"
    - "arch.L_cycles=4"
    - "epochs=100000"
    - "eval_interval=10000"
    - "+run_name=arc1-trm-experiment"
    - "ema=True"

# Stage 3: Model Publishing
model:
  checkpoint_dir: "checkpoints/arc1-trm-experiment"
  repo_id: "your-username/arc-trm-model"  # Set your HF username
  private: true
  token: null  # Uses HUGGINGFACE_TOKEN env var if null
  commit_message: "TinyRecursiveInference model upload"
  allow_create: true
  config_filename: "all_config.yaml"
  include_code_snapshot: true
  model_card_path: null  # Auto-generate if null
  extra_files:
    # "inference_example.py": "scripts/inference_example.py"
```

---

## 7. Integration Points {#integration-points}

### 7.1 Minimal Changes to Existing Code

**Principle**: Add TinyRecursiveInference without breaking existing workflows

**Changes to `pretrain.py`**:
```python
# Line 15: Add import (optional, graceful fallback)
try:
    from tiny_recursive_inference.wandb_callbacks import WandBTrainingCallback
    WANDB_CALLBACK_AVAILABLE = True
except ImportError:
    WANDB_CALLBACK_AVAILABLE = False

# Line 590: Replace wandb.init with callback
if RANK == 0:
    if WANDB_CALLBACK_AVAILABLE and os.getenv("USE_TRI_CALLBACKS", "false").lower() == "true":
        wandb_callback = WandBTrainingCallback(
            project=config.project_name,
            run_name=config.run_name,
            config=config.model_dump(),
            log_artifacts=config.checkpoint_every_eval
        )
    else:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)
        wandb.init(...)  # Existing code
        wandb_callback = None

# Line 610: Conditional logging
if RANK == 0 and metrics is not None:
    if wandb_callback:
        wandb_callback.on_train_batch_end(metrics, train_state.step)
    else:
        wandb.log(metrics, step=train_state.step)

# Line 636: Conditional eval logging
if RANK == 0 and metrics is not None:
    if wandb_callback:
        wandb_callback.on_eval_end(metrics, config.checkpoint_path, train_state.step)
    else:
        wandb.log(metrics, step=train_state.step)

# Line 649: Finalization
if RANK == 0 and wandb_callback:
    wandb_callback.on_training_end(config.checkpoint_path)
elif RANK == 0:
    wandb.finish()  # Existing code
```

**Backward Compatibility**: Existing runs work identically unless `USE_TRI_CALLBACKS=true` is set

### 7.2 Environment Variables

**HuggingFace**:
- `HUGGINGFACE_TOKEN`: Write access token for dataset/model publishing
- `HUGGINGFACE_DATASET_REPO`: Override dataset repo ID
- `HUGGINGFACE_MODEL_REPO`: Override model repo ID

**Weights & Biases**:
- `WANDB_API_KEY`: API key from wandb.ai/authorize
- `WANDB_PROJECT`: Project name (default from config)
- `WANDB_ENTITY`: Team/organization name
- `WANDB_DISABLED`: Set to "true" to disable W&B logging

**TinyRecursiveInference**:
- `USE_TRI_CALLBACKS`: Enable enhanced W&B callbacks ("true"/"false")
- `TRI_AUTO_PUBLISH`: Auto-publish checkpoint to HF after training ("true"/"false")

### 7.3 Dependency Management

**Required (existing)**:
- torch >= 2.0
- hydra-core
- pydantic
- numpy
- wandb

**New (optional)**:
- huggingface_hub >= 0.20.0 (for publishing)
- gradio >= 4.0 (for inference app)
- matplotlib (for visualizations)
- Pillow (for image rendering)

**Installation**:
```bash
# Core training (existing)
pip install -r requirements.txt

# TinyRecursiveInference extras
pip install huggingface_hub gradio matplotlib Pillow
```

**Graceful Degradation**: If gradio not installed, inference app fails with clear error:
```python
try:
    import gradio as gr
except ImportError:
    raise ImportError(
        "Gradio not installed. Install with: pip install gradio\n"
        "Or install all TinyRecursiveInference dependencies: "
        "pip install -r requirements-inference.txt"
    )
```

---

## 8. Testing & Validation Strategy {#testing-strategy}

### 8.1 Unit Tests

**Test Suite**: `tests/test_tiny_recursive_inference.py`

```python
import pytest
import tempfile
from pathlib import Path

from tiny_recursive_inference.config import DatasetPublishConfig, ModelPublishConfig
from tiny_recursive_inference.publishers import publish_dataset, publish_model


def test_dataset_config_defaults():
    config = DatasetPublishConfig()
    assert config.private == True
    assert config.add_readme == True


def test_model_loader_invalid_path():
    from tiny_recursive_inference.model_loader import load_trm_checkpoint

    with pytest.raises(FileNotFoundError):
        load_trm_checkpoint("/nonexistent/path")


@pytest.mark.skipif(
    os.getenv("HUGGINGFACE_TOKEN") is None,
    reason="HUGGINGFACE_TOKEN not set"
)
def test_dataset_publishing():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock dataset
        dataset_dir = Path(tmpdir) / "mock_dataset"
        dataset_dir.mkdir()
        train_dir = dataset_dir / "train"
        train_dir.mkdir()

        # Add metadata
        import json
        with open(train_dir / "dataset.json", "w") as f:
            json.dump({"vocab_size": 14, "seq_len": 900}, f)

        # Publish to private test repo
        config = DatasetPublishConfig(
            local_path=str(dataset_dir),
            repo_id="test-user/test-dataset",
            private=True
        )

        repo_id = publish_dataset(config)
        assert repo_id == "test-user/test-dataset"
```

**Run Tests**:
```bash
pytest tests/ -v --cov=tiny_recursive_inference
```

### 8.2 Integration Tests

**Test 1: Toy Dataset Training**
```bash
# Create tiny dataset (10 examples)
python dataset/build_sudoku_dataset.py \
  --output-dir data/sudoku-toy \
  --subsample-size 10 \
  --num-aug 1

# Train for 10 steps with W&B callbacks
USE_TRI_CALLBACKS=true python pretrain.py \
  arch=trm \
  data_paths="[data/sudoku-toy]" \
  evaluators="[]" \
  epochs=10 \
  eval_interval=5 \
  global_batch_size=10 \
  +run_name="integration-test"

# Verify checkpoint exists
ls checkpoints/Sudoku-toy-aug-1-ACT-torch/integration-test/
```

**Test 2: Gradio App Launch**
```bash
# Use integration test checkpoint
python -m tiny_recursive_inference.gradio_app \
  checkpoints/Sudoku-toy-aug-1-ACT-torch/integration-test/

# Manually verify UI loads and puzzle input field appears
```

**Test 3: Full Pipeline Dry Run**
```bash
python scripts/run_full_pipeline.py \
  --config config/inference_config.yaml \
  --dry-run

# Verify command printed correctly
```

### 8.3 Validation Metrics

**Dataset Publishing**:
- ✓ HF dataset repo created and files uploaded
- ✓ README.md present with metadata
- ✓ Can load dataset with `datasets.load_dataset(repo_id)`

**Training with W&B**:
- ✓ W&B run created with config
- ✓ Training metrics logged at each step
- ✓ Checkpoint artifact uploaded after eval
- ✓ Best checkpoint tracked in summary

**Model Publishing**:
- ✓ HF model repo created
- ✓ Checkpoint files uploaded
- ✓ Model card includes metrics and usage example
- ✓ Can load model with `load_trm_checkpoint(repo_id)`

**Gradio Inference**:
- ✓ App launches without errors
- ✓ Can parse valid ARC JSON
- ✓ Predictions visualized correctly
- ✓ ACT step progression shown

---

## 9. Deployment & Scaling Considerations {#deployment-considerations}

### 9.1 HuggingFace Spaces Deployment

**Create Space**:
```bash
# Clone template
git clone https://huggingface.co/spaces/your-username/trm-inference
cd trm-inference

# Copy Gradio app
cp ../TinyRecursiveModels/tiny_recursive_inference/gradio_app.py app.py

# Create requirements
cat > requirements.txt <<EOF
torch>=2.0
gradio>=4.0
huggingface_hub
matplotlib
Pillow
pyyaml
EOF

# Push to Space
git add .
git commit -m "Initial TRM inference app"
git push
```

**App Entrypoint** (`app.py`):
```python
from tiny_recursive_inference.gradio_app import TRMInferenceApp

# Download model from HF Hub on startup
app = TRMInferenceApp(
    checkpoint_dir="your-username/arc-trm-model",
    device="cpu"  # Spaces have limited GPU access
)

interface = app.create_interface()
interface.launch()
```

### 9.2 Local Deployment

**Docker Container**:
```dockerfile
FROM pytorch/pytorch:2.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install dependencies
COPY requirements.txt requirements-inference.txt ./
RUN pip install -r requirements.txt -r requirements-inference.txt

# Copy codebase
COPY . .

# Expose Gradio port
EXPOSE 7860

# Launch app
CMD ["python", "-m", "tiny_recursive_inference.gradio_app", "/checkpoints"]
```

**Run**:
```bash
docker build -t trm-inference .
docker run -p 7860:7860 -v $(pwd)/checkpoints:/checkpoints trm-inference
```

### 9.3 Scalability Considerations

**Dataset Publishing**:
- **Bottleneck**: HF Hub upload speed (~10 MB/s)
- **Optimization**: Upload only new/changed files, compress large arrays
- **Limit**: 100 GB per dataset repo (HF free tier)

**Training**:
- **Current**: 4x H100 GPUs, ~3 days for ARC-AGI-1
- **Scaling**: Multi-node DDP (see [AGENTS.md](AGENTS.md))
- **Cost**: ~$12/hour for 4x H100 on Lambda Labs

**Model Publishing**:
- **Checkpoint Size**: ~50 MB (7M params × 4 bytes + embeddings)
- **Optimization**: Save only model weights, not optimizer state
- **Versioning**: Tag checkpoints by step or accuracy

**Gradio Inference**:
- **Throughput**: ~1 puzzle/second on CPU, ~10/second on GPU
- **Scaling**: Use HF Inference API for autoscaling
- **Cost**: Free for CPU Spaces, paid for GPU

---

## 10. Risk Analysis & Mitigation {#risk-analysis}

### 10.1 Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| W&B callback breaks existing runs | HIGH | LOW | Gated by `USE_TRI_CALLBACKS` env var |
| HF upload fails mid-training | MEDIUM | MEDIUM | Retry logic + atomic uploads |
| Gradio app OOM on large puzzles | MEDIUM | LOW | Batch size = 1, enable CPU offload |
| Checkpoint versioning conflicts | LOW | MEDIUM | Include timestamp + step in artifact name |
| DDP deadlock with new callbacks | HIGH | LOW | Ensure all ranks call collectives identically |

### 10.2 Operational Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| HF token leak in public repo | HIGH | MEDIUM | Never commit tokens, use .env + .gitignore |
| W&B quota exceeded (100GB) | MEDIUM | LOW | Disable artifact logging for large runs |
| Training interrupted before publish | MEDIUM | HIGH | Add resume capability to pipeline script |
| User confusion about new CLI | LOW | HIGH | Comprehensive docs + examples |

### 10.3 Validation Checkpoints

**Before Merging to Main**:
- [ ] All unit tests pass
- [ ] Integration test completes without errors
- [ ] Existing `pretrain.py` runs identical to before (without callbacks)
- [ ] Documentation reviewed and accurate
- [ ] Environment variables documented in README

**Before Production Use**:
- [ ] Full ARC-AGI-1 training run with W&B callbacks
- [ ] Checkpoint published to HF and successfully loaded
- [ ] Gradio app deployed to HF Space and accessible
- [ ] User feedback collected on interface usability

---

## 11. Code Artifacts {#code-artifacts}

### 11.1 Files to Create

| File | LOC | Priority | Status |
|------|-----|----------|--------|
| `tiny_recursive_inference/wandb_callbacks.py` | 200 | HIGH | ❌ TODO |
| `tiny_recursive_inference/model_loader.py` | 150 | HIGH | ❌ TODO |
| `tiny_recursive_inference/gradio_app.py` | 400 | MEDIUM | ❌ TODO |
| `tiny_recursive_inference/visualizations.py` | 150 | MEDIUM | ❌ TODO |
| `scripts/run_full_pipeline.py` | 250 | LOW | ❌ TODO |
| `scripts/publish_checkpoint.py` | 100 | HIGH | ❌ TODO |
| `config/inference_config.yaml` | 50 | HIGH | ❌ TODO |
| `tests/test_tiny_recursive_inference.py` | 200 | MEDIUM | ❌ TODO |
| `requirements-inference.txt` | 10 | HIGH | ❌ TODO |
| `docs/INFERENCE_GUIDE.md` | 500 | LOW | ❌ TODO |

**Total Estimated LOC**: ~2,010 lines

### 11.2 Files to Modify

| File | Changes | Risk | Status |
|------|---------|------|--------|
| `pretrain.py` | +30 lines (callback integration) | LOW | ❌ TODO |
| `README.md` | +100 lines (TRI section) | NONE | ❌ TODO |
| `.gitignore` | +5 lines (ignore *.env) | NONE | ❌ TODO |

### 11.3 Implementation Sequence

**Phase 1** (Week 1):
1. Create `wandb_callbacks.py`
2. Modify `pretrain.py` for callback support
3. Test with toy Sudoku dataset
4. Create `config/inference_config.yaml`

**Phase 2** (Week 2):
5. Create `model_loader.py`
6. Create `visualizations.py`
7. Test model loading from local checkpoint
8. Create `scripts/publish_checkpoint.py`

**Phase 3** (Week 3):
9. Create `gradio_app.py`
10. Test inference on sample ARC puzzles
11. Deploy to HF Space
12. Gather user feedback

**Phase 4** (Week 4):
13. Create `scripts/run_full_pipeline.py`
14. Write comprehensive tests
15. Update documentation
16. Final validation and release

---

## 12. Conclusion & Next Steps

### 12.1 Summary

**TinyRecursiveInference** provides a complete end-to-end system for:
- 📦 **Dataset Publishing**: Automated HF Hub uploads with metadata
- 📊 **Training Telemetry**: Rich W&B logging with artifact versioning
- 🚀 **Model Publishing**: One-command checkpoint + model card upload
- 🎨 **Visual Inference**: Gradio app with reasoning transparency

**Design Highlights**:
- Minimal changes to existing codebase (<50 LOC in core files)
- Backward compatible with existing training workflows
- Modular architecture (can use stages independently)
- Graceful degradation (optional dependencies)

### 12.2 Implementation Estimate

**Total Effort**: ~4 weeks (1 developer)
- Phase 1 (W&B Callbacks): 1 week
- Phase 2 (Model Publishing): 1 week
- Phase 3 (Gradio Inference): 1 week
- Phase 4 (Integration & Docs): 1 week

**Lines of Code**: ~2,100 new + ~50 modified

### 12.3 Success Criteria

**Minimum Viable Product (MVP)**:
- ✅ W&B callbacks working with existing training
- ✅ Checkpoint auto-publish to HF after training
- ✅ Gradio app can load checkpoint and solve puzzles

**Full Feature Set**:
- ✅ All MVP criteria
- ✅ Full pipeline script with resume capability
- ✅ Comprehensive documentation + examples
- ✅ Deployed HF Space demo
- ✅ >90% test coverage

### 12.4 Recommended Approach

**Option A: Incremental Implementation** (Recommended)
- Start with Phase 1 (W&B callbacks)
- Test thoroughly before moving to next phase
- Gather feedback after each phase

**Option B: Parallel Development**
- Implement all phases simultaneously
- Faster time to completion
- Higher risk of integration issues

**Option C: External Contributors**
- Open-source project structure ready
- Clear task breakdown for community contributions
- Code review process required

---

## Appendix A: References

### Academic Papers
- [TRM Paper](https://arxiv.org/abs/2510.04871): Jolicoeur-Martineau, A. (2025). Less is More: Recursive Reasoning with Tiny Networks.
- [HRM Paper](https://arxiv.org/abs/2506.21734): Wang et al. (2025). Hierarchical Reasoning Model.

### Technical Documentation
- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Weights & Biases Artifacts Guide](https://docs.wandb.ai/guides/artifacts)
- [HuggingFace Hub Python Library](https://huggingface.co/docs/huggingface_hub)
- [Gradio Documentation](https://www.gradio.app/docs/)

### Existing Implementations
- [OpenAI Fine-Tuning API](https://platform.openai.com/docs/guides/fine-tuning)
- [HuggingFace Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)
- [StoneyNakoda Pipeline](https://huggingface.co/datasets/HarleyCooper/StoneyNakoda)

---

## Appendix B: Example Usage

### B.1 Dataset Publishing Only

```bash
python -c "
from tiny_recursive_inference import TinyRecursiveInferencePipeline
from tiny_recursive_inference.config import TinyRecursiveInferenceConfig, DatasetPublishConfig

config = TinyRecursiveInferenceConfig(
    dataset=DatasetPublishConfig(
        local_path='data/arc1concept-aug-1000',
        repo_id='your-username/arc-dataset',
        private=True
    )
)

pipeline = TinyRecursiveInferencePipeline(config)
repo_id = pipeline.publish_dataset()
print(f'Published to: {repo_id}')
"
```

### B.2 Training with Enhanced W&B

```bash
export USE_TRI_CALLBACKS=true
export WANDB_PROJECT=tiny-recursive-models

torchrun --nproc-per-node 4 pretrain.py \
  arch=trm \
  data_paths="[data/arc1concept-aug-1000]" \
  arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 \
  +run_name="arc1-enhanced-logging" \
  ema=True
```

### B.3 Publish Trained Model

```bash
python scripts/publish_checkpoint.py \
  --checkpoint-dir checkpoints/Arc1concept-aug-1000-ACT-torch/arc1-enhanced-logging \
  --repo-id your-username/arc-trm-model \
  --private
```

### B.4 Launch Gradio App

```bash
# Local checkpoint
python -m tiny_recursive_inference.gradio_app \
  checkpoints/Arc1concept-aug-1000-ACT-torch/arc1-enhanced-logging

# HF Hub checkpoint
python -m tiny_recursive_inference.gradio_app \
  your-username/arc-trm-model
```

### B.5 Full Pipeline

```bash
# Edit config/inference_config.yaml first
python scripts/run_full_pipeline.py \
  --config config/inference_config.yaml

# Skip stages
python scripts/run_full_pipeline.py \
  --config config/inference_config.yaml \
  --skip-dataset \
  --skip-training
```

---

**END OF DOCUMENT**

**Next Action**: Review this plan and provide feedback on:
1. Architecture design decisions
2. Implementation priorities
3. Missing components or considerations
4. Timeline estimates

Once approved, we can proceed with Phase 1 implementation.

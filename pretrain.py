from typing import Optional, Any, Sequence, List, Dict
from dataclasses import dataclass
from pathlib import Path
import os
import sys
import signal
import shutil
import yaml

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
try:
    import psutil
except ImportError:
    psutil = None
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from utils.env_setup import setup_training_env
setup_training_env()

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig
from adam_atan2_pytorch import AdamAtan2
try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False
    print("WARNING: bitsandbytes not available - using standard optimizers (4x more memory)")

# Global flag for graceful shutdown
shutdown_requested = False

def shutdown_signal_handler(sig, frame):
    """Handle Ctrl+C (SIGINT) gracefully.
    
    Sets a flag instead of immediately exiting to allow safe checkpoint saving.
    This prevents model corruption during backpropagation.
    """
    global shutdown_requested
    shutdown_requested = True
    print("\n" + "="*70)
    print("  STOP: SHUTDOWN REQUESTED (Ctrl+C detected)")
    print("  WAIT: Will save checkpoint at next safe point...")
    print("  WARNING: Press Ctrl+C again to force quit (may lose progress!)")
    print("="*70 + "\n")

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata, CapsuleDataLoaderWrapper
from utils.functions import load_model_class, get_model_source_path
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from models.ema import EMAHelper
from utils.gradient_monitor import GradientFlowMonitor
from utils.checkpointing import AsyncCheckpointManager
from utils.data_processing import process_vision_batch, detect_dataset_features
from utils.annealing import compute_expansion_penalty, compute_q_temperature, compute_warmup_progress
from dataset.training_progress import TrainingProgressTracker, get_training_manifest


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig


class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    data_paths: List[str]
    data_paths_test: List[str] = []
    # Evaluators
    evaluators: List[EvaluatorConfig] = []

    # Hyperparams
    global_batch_size: int
    epochs: int
    max_steps: Optional[int] = None  # Debugging/Verification limit

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    min_eval_interval: Optional[int] = 0 # when to start eval
    eval_save_outputs: List[str] = []

    ema: bool = False # use Exponential-Moving-Average
    ema_rate: float = 0.999 # EMA-rate
    freeze_weights: bool = False # If True, freeze weights and only learn the embeddings
    
    # DQN Stability Features
    dqn_warmup_ratio: float = 0.1  # Ratio of training for DQN warmup (0.1 = first 10%)
    dqn_warmup_steps: int = 5000  # Deprecated: use dqn_warmup_ratio
    freeze_representation_during_warmup: bool = True  # Freeze h_blocks/l_blocks during warmup
    
    # Optimizer Stability
    optimizer_type: str = "adamatan2"  # "adamatan2" or "adamw"
    enable_optimizer_fallback: bool = True  # Auto-switch to AdamW on NaN
    nan_threshold_for_fallback: int = 3  # Switch after N NaN occurrences
    
    # Expansion Penalty Annealing
    expansion_penalty_schedule: str = "cosine"  # "cosine", "linear", or "fixed"
    expansion_penalty_start: float = 0.1  # Initial penalty
    expansion_penalty_end: float = 0.001  # Final penalty
    expansion_anneal_ratio: float = 0.5  # Ratio of total training (0.5 = first 50%)
    expansion_anneal_steps: int = 50000  # Deprecated: use expansion_anneal_ratio
    
    # Replay Buffer Sampling
    replay_recent_fraction: float = 0.25  # Fraction from recent transitions
    replay_max_age: int = 100000  # Discard transitions older than this
    
    # Q-Head Temperature Annealing (Exploration -> Exploitation)
    enable_q_temperature_annealing: bool = True
    q_temperature_start: float = 1.0  # High temp = exploration
    q_temperature_end: float = 0.1  # Low temp = exploitation
    q_temperature_anneal_ratio: float = 1.0  # Ratio of total training (1.0 = 100%)
    q_temperature_schedule: str = "exponential"  # "linear", "exponential", "cosine"

def per_layer_gradient_normalization(model: nn.Module, config) -> dict:
    """
    Apply per-layer gradient normalization instead of global clipping.
    
    Benefits:
    - Prevents strong gradients in one module from being clipped
    - Allows different learning dynamics per component
    - More stable training for hierarchical models like TRM
    
    Args:
        model: TRM model
        config: Training config
    
    Returns:
        Dictionary of gradient norms per component
    """
    grad_norms = {}
    
    # Define module groups and their keywords
    module_groups = {
        'encoder': ['embed', 'input_proj', 'puzzle_emb'],
        'H_level': ['H_level', 'H_init', 'h_blocks'],
        'L_level': ['L_level', 'L_init', 'l_blocks'],
        'q_head': ['q_head'],
        'memory': ['memory'],
        'lm_head': ['lm_head', 'output']
    }
    
    # Get max norms per group (from config with defaults)
    max_norms = {
        'encoder': getattr(config.arch, 'encoder_max_grad_norm', 1.0),
        'H_level': getattr(config.arch, 'h_level_max_grad_norm', 1.0),
        'L_level': getattr(config.arch, 'l_level_max_grad_norm', 1.0),
        'q_head': getattr(config.arch, 'q_head_max_grad_norm', 2.0),  # Allow larger for RL
        'memory': getattr(config.arch, 'memory_max_grad_norm', 1.0),
        'lm_head': getattr(config.arch, 'lm_head_max_grad_norm', 1.0),
    }
    
    # Clip per group
    for group_name, keywords in module_groups.items():
        # Collect parameters for this group
        group_params = []
        for name, param in model.named_parameters():
            if param.grad is not None and any(kw in name for kw in keywords):
                group_params.append(param)
        
        if group_params:
            # Clip this group
            grad_norm = torch.nn.utils.clip_grad_norm_(
                group_params,
                max_norm=max_norms[group_name]
            )
            grad_norms[f'grad_{group_name}_norm'] = grad_norm.item()
    
    return grad_norms


@dataclass
class TrainState:
    model: nn.Module  # May be torch.compile wrapped
    original_model: nn.Module  # Unwrapped model for .parameters() and EMA
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int
    last_checkpoint_step: int = 0  # Track last saved checkpoint
    representation_frozen: bool = False  # Track if representation layers are frozen
    nan_count: int = 0  # Track NaN occurrences for optimizer fallback
    scaler: Optional[Any] = None  # GradScaler for mixed precision
    device: torch.device = torch.device("cpu")  # Device for tensors





def load_datasets(config: PretrainConfig, rank: int, world_size: int, split: str = 'train'):
    """Unified dataset loading with auto-detection."""
    
    # Check if semantic_mode is explicitly set in config
    semantic_mode = getattr(config, 'semantic_mode', None)
    
    # Auto-detect dataset features only if semantic_mode not explicitly set
    if semantic_mode is None:
        dataset_path = config.data_paths[0] if config.data_paths else ''
        features = detect_dataset_features(dataset_path)
        semantic_mode = features['is_capsule']
    else:
        # Still detect features for logging purposes
        dataset_path = config.data_paths[0] if config.data_paths else ''
        features = detect_dataset_features(dataset_path) if os.path.exists(dataset_path) else {'is_capsule': semantic_mode, 'has_checksums': False, 'has_children': False, 'enable_dqn': False}
    
    if rank == 0 and features['is_capsule']:
        print(f"\n[INFO] Auto-detected HESC capsule dataset")
        print(f"   Checksums: {features['has_checksums']}")
        print(f"   Children: {features['has_children']}")
        print(f"   DQN recommended: {features['enable_dqn']}")
    
    # Unified dataset loading using PuzzleDataset
    # This handles both standard token datasets and new streaming capsule datasets

    # Token-based mode (original logic)
    dataset_paths = config.data_paths if split == 'train' else config.data_paths_test
    
    kwargs = {}
    if hasattr(config, 'group_batch_size') and config.group_batch_size is not None:
        kwargs['group_batch_size'] = config.group_batch_size
    
    if hasattr(config, 'groups_per_batch'):
        kwargs['groups_per_batch'] = config.groups_per_batch
    
    # Calculate epochs_per_iter based on eval_interval (same logic as in launch function)
    epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    
    try:
        # Use PuzzleDataset for standard/token-based mode
        puzzle_config = PuzzleDatasetConfig(
            seed=42,
            dataset_paths=dataset_paths,
            global_batch_size=config.global_batch_size,
            test_set_mode=(split != 'train'),
            epochs_per_iter=epochs_per_iter,
            rank=rank,
            num_replicas=world_size
        )
        dataset = PuzzleDataset(puzzle_config, split=split)
        
        # Check for empty dataset to trigger auto-build
        if dataset.metadata.total_puzzles == 0:
            raise FileNotFoundError(f"Dataset '{split}' is empty (0 samples).")
    except FileNotFoundError as e:
        # Auto-fallback: Build dataset if missing
        if rank == 0:  # Only build on rank 0 in distributed training
            print(f"\n{'='*60}")
            print(f"Dataset not found: {e}")
            print(f"Auto-building dataset with default configuration...")
            print(f"{'='*60}\n")
            
            import subprocess
            # Use NEW multimodal builder with text rendering support
            # Build to the FIRST dataset path that training expects
            output_dir = dataset_paths[0] if dataset_paths else "datasets/vision_unified"
            build_cmd = [
                "python", "dataset/build_multimodal_dataset.py",
                "--config.source-paths", "kaggle/combined/arc-agi",
                "--config.output-dir", output_dir,
                "--config.render-text-to-image",
                "--config.use-capsules",
                "--config.seed", "42"
            ]
            
            print(f"Running: {' '.join(build_cmd)}\n")
            result = subprocess.run(build_cmd, check=True)
            
            if result.returncode == 0:
                print(f"\n{'='*60}")
                print(f"Dataset built successfully! Retrying data load...")
                print(f"{'='*60}\n")
            else:
                raise RuntimeError(f"Dataset build failed with code {result.returncode}")
        
        # Wait for rank 0 to finish building in distributed mode
        if world_size > 1:
            dist.barrier()
        
        # Retry loading dataset with ALL original kwargs
        dataset = PuzzleDataset(PuzzleDatasetConfig(
            seed=config.seed,
            dataset_paths=dataset_paths,
            global_batch_size=config.global_batch_size,
            test_set_mode=(split != 'train'),
            epochs_per_iter=epochs_per_iter,
            rank=rank,
            num_replicas=world_size,
            **kwargs
        ), split=split)
    
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=2,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True
    )
    return dataloader, dataset.metadata


def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int, total_steps: int = 100000):
    """Create model with loss head.
    
    Args:
        total_steps: Total training steps for annealing schedules
    """
    # Build model config: merge YAML arch config with dataset metadata
    # __pydantic_extra__ contains all extra YAML fields (causal, input_vocab_size, etc.)
    # Dataset metadata overrides seq_len/vocab_size/num_puzzle_identifiers
    
    # Auto-detect dataset type and enable appropriate training techniques
    semantic_mode = getattr(config, 'semantic_mode', False)
    enable_capsule_expansion = getattr(config.arch, 'enable_capsule_expansion', False)
    
    if rank == 0:
        print(f"\nDEBUG: Dataset Analysis:")
        print(f"   Semantic/Capsule mode: {semantic_mode}")
        print(f"   Capsule expansion: {enable_capsule_expansion}")
    
    # Filter out keys that will be explicitly set to avoid duplicate keyword argument error
    # IMPORTANT: Keep input_vocab_size from YAML (for vision with separate input/output vocabs)
    extra_config = {k: v for k, v in config.arch.__pydantic_extra__.items() 
                   if k not in ['batch_size', 'vocab_size', 'seq_len', 'num_puzzle_identifiers']}
    
    # Auto-enable DQN for capsule datasets with expansion
    if semantic_mode and enable_capsule_expansion:
        if 'enable_dqn' not in extra_config or not extra_config['enable_dqn']:
            if rank == 0:
                print(f"   DEBUG: Auto-enabling DQN for capsule expansion control")
            extra_config['enable_dqn'] = True
            # Set expansion-friendly DQN params
            if 'dqn_loss_weight' not in extra_config:
                extra_config['dqn_loss_weight'] = 0.5
            if 'enable_per' not in extra_config:
                extra_config['enable_per'] = True  # Prioritized experience replay
    
    model_cfg = dict(
        **extra_config,
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
    )
    
    model_cfg['semantic_mode'] = semantic_mode
    model_cfg['enable_capsule_expansion'] = enable_capsule_expansion
    model_cfg['total_steps'] = total_steps  # For annealing schedules
    
    if rank == 0:
        print(f"\nDEBUG: Model Config:")
        print(f"   arch: {config.arch.name}")
        print(f"   input_vocab_size: {model_cfg.get('input_vocab_size', 'NOT SET')}")
        print(f"   output_vocab_size: {model_cfg.get('output_vocab_size', 'NOT SET')}")
        print(f"   hidden_size: {model_cfg.get('hidden_size', 'NOT SET')}")
        print(f"   seq_len: {model_cfg.get('seq_len')}")
        print(f"   total_steps: {total_steps:,}")

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.device(device):
        model: nn.Module = model_cls(model_cfg)
        print(model)
        # Pass total_steps to loss head for annealing schedules
        loss_kwargs = dict(config.arch.loss.__pydantic_extra__)
        loss_kwargs['total_steps'] = total_steps
        model = loss_head_cls(model, **loss_kwargs)  # type: ignore
        
        # CRITICAL: torch.compile disabled for memory constraints
        # Even backend='eager' triggers AUTOTUNE and deferred cudagraphs that consume 14GB+
        # AUTOTUNE benchmarks multiple implementations, allocating temporary buffers
        if rank == 0:
            print("\n[WARNING] torch.compile DISABLED due to memory constraints")
            print("   Reason: Even eager backend triggers AUTOTUNE addmm benchmarking")
            print("   AUTOTUNE allocates temporary buffers for each kernel variant")
            print("   This causes OOM on 15GB GPU with 366M parameter model")
            print("   Trade-off: ~20-30% slower but fits in memory")
            print("   Set ENABLE_COMPILE=1 to force enable (will likely OOM)\n")

        # Load checkpoint (handled by AsyncCheckpointManager outside this function)
        # if rank == 0:
        #     load_checkpoint(model, config)

        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    # Helper function to create optimizer with fallback
    def create_optimizer_with_fallback(params, config, optimizer_type="main"):
        """Create optimizer with automatic fallback. Uses bitsandbytes 8-bit to save 75% memory."""
        lr = 1e-8  # Will be set by scheduler
        wd = config.weight_decay
        betas = (config.beta1, config.beta2)
        
        # Priority 1: Use bitsandbytes 8-bit optimizer (saves 75% memory)
        if BNB_AVAILABLE:
            try:
                optimizer = bnb.optim.AdamW8bit(
                    params, 
                    lr=lr, 
                    weight_decay=wd, 
                    betas=betas,
                    min_8bit_size=4096  # Only quantize large tensors
                )
                if rank == 0:
                    print(f"  Using bitsandbytes AdamW8bit optimizer for {optimizer_type} (saves 75% memory)")
                return optimizer
            except Exception as e:
                if rank == 0:
                    print(f"  [WARNING] bitsandbytes failed: {e}, falling back to standard optimizer")
        
        # Priority 2: Force AdamW if configured
        if config.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas)
            if rank == 0:
                print(f"  Using AdamW optimizer for {optimizer_type}")
        else:
            # Priority 3: Try AdamAtan2
            try:
                optimizer = AdamAtan2(params, lr=lr, weight_decay=wd, betas=betas)
                if rank == 0:
                    print(f"  Using AdamAtan2 optimizer for {optimizer_type}")
            except Exception as e:
                if config.enable_optimizer_fallback:
                    if rank == 0:
                        print(f"  [WARNING] AdamAtan2 failed: {e}")
                        print(f"  Falling back to AdamW optimizer for {optimizer_type}")
                    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas)
                else:
                    raise
        
        return optimizer
    
    # Optimizers and lr
    # Handle puzzle_emb_ndim=0 case: no puzzle embeddings, only main optimizer
    if config.arch.puzzle_emb_ndim == 0 or not hasattr(model.model, 'puzzle_emb'):
        # Text/Code/Vision models without puzzle-specific embeddings
        optimizers = [
            create_optimizer_with_fallback(model.parameters(), config, "main")
        ]
        optimizer_lrs = [
            config.lr
        ]
    elif config.freeze_weights:
        # Only train puzzle embeddings (freeze main model)
        if not hasattr(model.model, 'puzzle_emb'):
            raise ValueError("freeze_weights=True requires puzzle_emb_ndim > 0")
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            )
        ]
        optimizer_lrs = [
            config.puzzle_emb_lr
        ]
    else:
        # ARC models: train both puzzle embeddings and main model
        if not hasattr(model.model, 'puzzle_emb'):
            raise ValueError("Dual optimizer mode requires puzzle_emb_ndim > 0")
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            ),
            create_optimizer_with_fallback(model.parameters(), config, "main")
        ]
        optimizer_lrs = [
            config.puzzle_emb_lr,
            config.lr
        ]

    return model, optimizers, optimizer_lrs



def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    # Auto-adjust hyperparameters based on dataset characteristics
    semantic_mode = getattr(config, 'semantic_mode', False)
    dataset_size = train_metadata.total_groups
    
    if rank == 0:
        print(f"\nDEBUG: Auto-tuning hyperparameters:")
        print(f"   Dataset size: {dataset_size:,} samples")
    
    # Adjust learning rate based on dataset size and mode
    adjusted_lr = config.lr
    if semantic_mode:
        # Capsule mode: can use higher LR due to compressed inputs
        adjusted_lr = config.lr * 1.5
        if rank == 0:
            print(f"   LR boost for capsule mode: {config.lr:.2e} -> {adjusted_lr:.2e}")
    elif dataset_size < 1000:
        # Small dataset: reduce LR to prevent overfitting
        adjusted_lr = config.lr * 0.5
        if rank == 0:
            print(f"   LR reduction for small dataset: {config.lr:.2e} -> {adjusted_lr:.2e}")
    
    # Store adjusted hyperparameters
    config.lr = adjusted_lr
    
    # Calculate total training steps correctly
    # Steps per epoch = (total samples) / batch_size
    # Total steps = steps_per_epoch * epochs
    samples_per_epoch = train_metadata.total_groups * train_metadata.mean_puzzle_examples
    steps_per_epoch = max(1, int(samples_per_epoch / config.global_batch_size))
    total_steps = steps_per_epoch * config.epochs
    
    if rank == 0:
        print(f"   Samples per epoch: {samples_per_epoch:,}")
        print(f"   Steps per epoch: {steps_per_epoch:,}")
        print(f"   Total training steps: {total_steps:,} ({config.epochs} epochs)")

    # Model (pass total_steps for annealing)
    # Clear cache before model creation to avoid OOM
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    # Check if dataset is empty (which might happen if build failed or was interrupted)
    if train_metadata.total_puzzles == 0:
        print("\n[WARN] Dataset appears to be empty (0 samples).")
        # Raise FileNotFoundError to trigger auto-build logic if enabled
        raise FileNotFoundError("Dataset is empty (0 samples). Triggering auto-build.")

    model, optimizers, optimizer_lrs = create_model(config, train_metadata, rank=rank, world_size=world_size, total_steps=total_steps)
    
    # Store original model reference for parameter access and EMA
    original_model = model
    
    # torch.compile for 25% speedup (T4 compatible)
    # NOTE: torch.compile DISABLED - even eager backend triggers AUTOTUNE and OOM
    # AUTOTUNE benchmarks multiple kernel implementations
    # Re-enabled with safer configuration
    if getattr(config, 'compile_model', True):
        if os.name == 'nt':
             if rank == 0:
                 print(f"   [INFO] torch.compile disabled on Windows (experimental support)")
        else:
            if rank == 0:
                print(f"   [INFO] Compiling model with torch.compile (mode='default')...")
            # Use 'default' mode which is safer than 'max-autotune' for memory
            # 'reduce-overhead' is also good for small batches but can be memory hungry
            try:
                model = torch.compile(model, mode='default')
            except Exception as e:
                if rank == 0:
                    print(f"   [WARNING] torch.compile failed: {e}. Proceeding without compilation.")

    return TrainState(
        step=0,
        total_steps=total_steps,

        model=model,
        original_model=original_model,  # Store for EMA and parameter counting
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None,
        scaler=GradScaler(enabled=torch.cuda.is_available()) if getattr(config, 'use_mixed_precision', True) else None,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )


def get_checkpoint_name(dataset_path: str) -> str:
    """Get checkpoint filename from dataset path.
    
    Examples:
        data/text-wikitext2 -> text.pt
        data/arc-aug-1000 -> arc.pt
        data/text-tinystories -> text-tiny.pt
    """
    # Extract dataset name from path
    dataset_name = os.path.basename(dataset_path)
    
    # Map common patterns to simple names
    if 'text-wikitext' in dataset_name:
        return 'text.pt'
    elif 'text-tinystories' in dataset_name or 'tinystories' in dataset_name:
        return 'text-tiny.pt'
    elif 'arc' in dataset_name:
        return 'arc.pt'
    elif 'code' in dataset_name:
        return 'code.pt'
    elif 'vision' in dataset_name or 'cifar' in dataset_name or 'image' in dataset_name:
        return 'vision.pt'
    elif 'alpaca' in dataset_name:
        return 'alpaca.pt'
    elif 'sharegpt' in dataset_name:
        return 'sharegpt.pt'
    elif 'sudoku' in dataset_name:
        return 'sudoku.pt'
    elif 'maze' in dataset_name:
        return 'maze.pt'
    else:
        # Fallback: use sanitized dataset name
        return f"{dataset_name.replace('/', '_').replace('-', '_')}.pt"


def load_checkpoint(model: nn.Module, config: PretrainConfig, optimizers=None, train_state=None):
    """Load checkpoint with integrity verification.
    
    Supports cross-dataset loading (e.g., load text.pt into vision training).
    """
    # Initialize Async Checkpoint Manager
    checkpoint_manager = AsyncCheckpointManager(
        checkpoint_dir=config.checkpoint_path,
        max_keep=3,
        use_safetensors=True
    )

    # Load checkpoint if exists
    # Load checkpoint if exists
    start_step = 0
    
    # Auto-resume logic: If no specific checkpoint provided, look for latest in checkpoint_dir
    if not config.load_checkpoint and config.checkpoint_path:
        checkpoint_dir = Path(config.checkpoint_path)
        if checkpoint_dir.exists():
            checkpoints = sorted(
                [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
                key=lambda x: int(x.name.split('-')[-1])
            )
            if checkpoints:
                config.load_checkpoint = str(checkpoints[-1])
                print(f"[AUTO-RESUME] Found latest checkpoint: {config.load_checkpoint}")

    if config.load_checkpoint:
        try:
            extra_data = checkpoint_manager.load(config.load_checkpoint, model, optimizers)
            start_step = extra_data.get('step', 0)
            if train_state:
                train_state.step = start_step
            print(f"Resumed from step {start_step}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}. Starting from scratch.")
            
    return start_step


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )



def create_evaluators(config: PretrainConfig, eval_metadata: PuzzleDatasetMetadata) -> List[Any]:
    data_paths =config.data_paths_test if len(config.data_paths_test)>0 else config.data_paths
    # Initialize evaluators
    evaluators = []
    for cfg in config.evaluators:
        for data_path in data_paths:
            cls = load_model_class(cfg.name, "evaluators.")(
                data_path=data_path, eval_metadata=eval_metadata, **cfg.__pydantic_extra__
            )  # type: ignore
            evaluators.append(cls)

    return evaluators

def prepare_batch(batch: Any, device: torch.device, config: PretrainConfig) -> Dict[str, Any]:
    """
    Prepare batch for training: process vision samples, move to device, handle capsules.
    """
    # Handle raw_samples format (vision-unified pipeline) OR pre-processed vision batch
    if isinstance(batch, dict) and ('raw_samples' in batch or 'images' in batch):
        if 'raw_samples' in batch and 'images' not in batch:
            batch = process_vision_batch(batch)
        
        # Move new tensors to device
        if 'images' in batch:
            batch['images'] = batch['images'].to(device, non_blocking=True)
        if 'target_images' in batch:
            batch['target_images'] = batch['target_images'].to(device, non_blocking=True)
        
        # Map to model inputs
        if 'images' in batch:
            batch = {
                'inputs': batch['images'],
                'targets': batch.get('target_images'),
                'puzzle_identifiers': batch['puzzle_identifiers'].to(device),
                'labels': torch.zeros(batch['images'].shape[0], dtype=torch.long, device=device)
            }
    # Handle HESC capsules (tuple) vs dict batches
    elif isinstance(batch, tuple):
        # HESC mode: batch from TensorDataset
        if len(batch) == 3:
            # Capsules with children: (sketches, checksums, children)
            sketches, checksums, children = batch
            batch = {
                'inputs': sketches.to(device),
                'capsule_sketches': sketches.to(device),
                'capsule_checksums': checksums.to(device),
                'capsule_children': children.to(device),
                'labels': torch.zeros(sketches.shape[0], dtype=torch.long).to(device),
                'puzzle_identifiers': torch.zeros(sketches.shape[0], dtype=torch.long).to(device) # Dummy
            }
        elif len(batch) == 2:
             # Capsules without children: (sketches, checksums)
             sketches, checksums = batch
             batch = {
                'inputs': sketches.to(device),
                'capsule_sketches': sketches.to(device),
                'capsule_checksums': checksums.to(device),
                'labels': torch.zeros(sketches.shape[0], dtype=torch.long).to(device),
                'puzzle_identifiers': torch.zeros(sketches.shape[0], dtype=torch.long).to(device),
                'num_expansions': torch.zeros(sketches.shape[0], dtype=torch.long).to(device)
             }
        else:
            # Legacy semantic: (embeddings,)
            embeddings = batch[0].to(device)
            batch = {
                'inputs': embeddings,
                'labels': torch.zeros(embeddings.shape[0], dtype=torch.long).to(device),
                'puzzle_identifiers': torch.zeros(embeddings.shape[0], dtype=torch.long).to(device)
            }
    elif isinstance(batch, dict):
        # Standard dict batch
        for k, v in batch.items():
            # Convert numpy to tensor
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
                batch[k] = v
                
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)
        
        # Map PuzzleDataset keys to Capsule keys if 3D inputs (sketches)
        if 'inputs' in batch and batch['inputs'].dim() == 3:
             if 'capsule_sketches' not in batch:
                 batch['capsule_sketches'] = batch['inputs']
             if 'labels' in batch and 'capsule_checksums' not in batch:
                 batch['capsule_checksums'] = batch['labels']
             if 'children' in batch and 'capsule_children' not in batch:
                 batch['capsule_children'] = batch['children']
                
    # Add CapsuleState for expansion tracking (if enabled)
    if hasattr(config.arch, 'enable_capsule_expansion') and config.arch.enable_capsule_expansion:
        if 'inputs' in batch and batch['inputs'].dim() == 3:
            from models.capsule_state import CapsuleState
            
            children = batch.get('capsule_children', None)
            checksums = batch.get('capsule_checksums', None)
            
            capsule_state = CapsuleState(
                sketches=batch['inputs'].clone(),
                children=children,
                checksums=checksums
            )
            batch['capsule_state'] = capsule_state
            
    return batch


def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int, gradient_monitor=None):
    train_state.step += 1
    if train_state.step > train_state.total_steps:  # At most train_total_steps
        return
    
    # Debug/Verification limit
    if config.max_steps is not None and train_state.step > config.max_steps:
        return
    
    # DQN Warmup: Freeze representation layers during early training
    if config.freeze_representation_during_warmup:
        # Calculate warmup steps from ratio
        if hasattr(config, 'dqn_warmup_ratio'):
            dqn_warmup_steps = int(train_state.total_steps * config.dqn_warmup_ratio)
        else:
            # Fallback to deprecated steps
            dqn_warmup_steps = config.dqn_warmup_steps
        
        if train_state.step < dqn_warmup_steps and not train_state.representation_frozen:
            # Freeze representation layers (h_blocks, l_blocks)
            if rank == 0:
                print(f"\n{'='*70}")
                print(f"  [LOCKED] DQN WARMUP: Freezing representation layers")
                print(f"  Steps 0-{dqn_warmup_steps}: Supervised learning only")
                print(f"  Step {dqn_warmup_steps}+: Full DQN training")
                print(f"  Warmup ratio: {config.dqn_warmup_ratio*100:.1f}% of training")
                print(f"{'='*70}\n")
            
            for name, param in train_state.original_model.named_parameters():
                if 'h_blocks' in name or 'l_blocks' in name or 'h_level' in name or 'l_level' in name:
                    param.requires_grad = False
            
            train_state.representation_frozen = True
        
        elif train_state.step == dqn_warmup_steps and train_state.representation_frozen:
            # Unfreeze after warmup
            if rank == 0:
                print(f"\n{'='*70}")
                print(f"  [UNLOCKED] DQN WARMUP COMPLETE: Unfreezing representation layers")
                print(f"  Full model training enabled")
                print(f"{'='*70}\n")
            
            for param in train_state.original_model.parameters():
                param.requires_grad = True
            
            train_state.representation_frozen = False

    # Handle raw_samples format (vision-unified pipeline) OR pre-processed vision batch
    # Prepare batch (process vision, move to device, handle capsules)
    batch = prepare_batch(batch, train_state.device, config)

    # Always initialize fresh carry for new batch (independent puzzles)
    # This prevents memory leaks (graph persistence) and state contamination
    train_state.carry = train_state.original_model.initial_carry(batch)  # type: ignore

    # Task-Based Training (ALWAYS ENABLED - Single Unified Pipeline)
    # Model thinks for variable steps until task completion (ACT/PonderNet approach)
    max_thinking_steps = getattr(config, 'max_thinking_steps', 10)
    task_completion_threshold = getattr(config, 'task_completion_threshold', 0.95)
    use_amp = getattr(config, 'use_mixed_precision', True)
    
    # Think until completion or max steps
    total_loss = 0.0
    all_metrics = {}
    thinking_steps_used = 0
    task_completed = False
    
    for think_step in range(max_thinking_steps):
        if train_state.step == 1 and think_step == 0 and rank == 0:
            print(f"\n[STATS] GPU Memory at FIRST forward pass (step {train_state.step}):")
            if torch.cuda.is_available():
                print(f"  Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
                print(f"  Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
                print(f"  Max Allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
                print(f"  Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated())/1e9:.2f} GB\n")
            elif psutil is not None:
                vm = psutil.virtual_memory()
                print(f"  RAM Used: {vm.used/1e9:.2f} GB")
                print(f"  RAM Total: {vm.total/1e9:.2f} GB\n")
        
        # Use PyTorch native CPU offloading for activations (saves 3-4GB GPU memory)
        # This moves intermediate activations to pinned CPU memory during forward pass
        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            if use_amp:
                # Use mixed precision training
                device_type = "cuda" if torch.cuda.is_available() else "cpu"
                with autocast(device_type):
                    train_state.carry, step_loss, step_metrics, _, _ = train_state.model(
                        carry=train_state.carry, 
                        batch=batch, 
                        return_keys=[],
                        global_step=train_state.step
                    )
            else:
                train_state.carry, step_loss, step_metrics, _, _ = train_state.model(
                    carry=train_state.carry, 
                    batch=batch, 
                    return_keys=[],
                    global_step=train_state.step
                )
        
        for k, v in step_metrics.items():
            if k not in all_metrics:
                all_metrics[k] = v
            else:
                all_metrics[k] += v
        
        total_loss += step_loss
        thinking_steps_used += 1
        
        # Check task completion: Use halt probability from ACT
        # High halt = model is confident it finished reasoning
        if 'halt_prob' in step_metrics:
            halt_prob = step_metrics['halt_prob']
            if halt_prob > task_completion_threshold:
                task_completed = True
                break
        
        # Alternative: Check Q-value confidence
        if 'q_max' in step_metrics and 'q_std' in step_metrics:
            # High Q-value + low std = confident decision
            if step_metrics['q_max'] > 0.8 and step_metrics['q_std'] < 0.1:
                task_completed = True
                break
    
    # Average loss over thinking steps
    loss = total_loss / thinking_steps_used if thinking_steps_used > 0 else torch.tensor(0.0)
    
    # Average metrics
    for k in all_metrics:
        all_metrics[k] = all_metrics[k] / thinking_steps_used
    
    # Add task-based metrics
    all_metrics['thinking_steps'] = thinking_steps_used
    all_metrics['task_completed'] = float(task_completed)
    metrics = all_metrics
    
    # NaN Detection and Optimizer Fallback
    if torch.isnan(loss) or torch.isinf(loss):
        train_state.nan_count += 1
        
        if rank == 0:
            print(f"\n{'='*70}")
            print(f"  [WARNING] NaN/Inf DETECTED in loss at step {train_state.step}")
            print(f"  NaN count: {train_state.nan_count}/{config.nan_threshold_for_fallback}")
            print(f"  Loss value: {loss.item()}")
        
        if config.enable_optimizer_fallback and train_state.nan_count >= config.nan_threshold_for_fallback:
            if rank == 0:
                print(f"  [SWITCH] Switching to AdamW optimizer (fallback)")
                print(f"{'='*70}\n")
            
            # Recreate optimizers with AdamW
            old_config_type = config.optimizer_type
            config.optimizer_type = "adamw"
            
            # Preserve optimizer state if possible
            for i, opt in enumerate(train_state.optimizers):
                if isinstance(opt, torch.optim.AdamW):
                    continue  # Already AdamW
                
                # Create new AdamW optimizer
                new_opt = torch.optim.AdamW(
                    opt.param_groups[0]['params'],
                    lr=train_state.optimizer_lrs[i],
                    weight_decay=config.weight_decay,
                    betas=(config.beta1, config.beta2)
                )
                train_state.optimizers[i] = new_opt
            
            train_state.nan_count = 0  # Reset counter
        
        # Skip this batch
        if rank == 0:
            print(f"  [SKIP] Skipping batch due to NaN/Inf")
            print(f"{'='*70}\n")
        return

    # Gradient Accumulation (Winner Strategy - Large Effective Batch)
    accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
    accumulation_idx = train_state.step % accumulation_steps
    
    # Scale loss by accumulation steps (accumulate gradients over multiple batches)
    scaled_loss = loss / accumulation_steps
    
    # Backward pass with mixed precision scaling
    if use_amp and train_state.scaler is not None:
        # Scale loss and backward
        train_state.scaler.scale((1 / global_batch_size) * scaled_loss).backward()
    else:
        # Standard backward
        ((1 / global_batch_size) * scaled_loss).backward()

    # Allreduce
    if world_size > 1:
        for param in train_state.original_model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)
    
    # Unscale gradients before clipping (for mixed precision)
    if use_amp and train_state.scaler is not None:
        train_state.scaler.unscale_(train_state.optimizers[0])
    
    # Track gradients before clipping (for monitoring)
    grad_stats = None
    if gradient_monitor is not None and rank == 0:
        grad_stats = gradient_monitor.update()
    
    # Only step optimizer after accumulating N gradients (Winner Strategy)
    should_step = (accumulation_idx == accumulation_steps - 1)
    
    grad_norm = 0.0
    lr_this_step = None
    
    if should_step:
        # Gradient normalization: per-layer or global
        if getattr(config, 'enable_per_layer_grad_norm', False):
            # Per-layer gradient normalization (more fine-grained control)
            grad_norms_dict = per_layer_gradient_normalization(train_state.original_model, config)
            grad_norm = sum(grad_norms_dict.values()) / len(grad_norms_dict) if grad_norms_dict else 0.0
        else:
            # Global gradient clipping (default)
            grad_norm = torch.nn.utils.clip_grad_norm_(train_state.original_model.parameters(), max_norm=1.0)
                
        # Apply optimizer with mixed precision support
        for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
            lr_this_step = compute_lr(base_lr, config, train_state)

            for param_group in optim.param_groups:
                param_group['lr'] = lr_this_step
            
            # Use scaler.step() for mixed precision, or regular step()
            if use_amp and train_state.scaler is not None:
                train_state.scaler.step(optim)
            else:
                optim.step()
            
            optim.zero_grad()
        
        # Update scaler for next iteration (mixed precision)
        if use_amp and train_state.scaler is not None:
            train_state.scaler.update()

    # Reduce metrics
    if len(metrics):
        assert not any(getattr(v, 'requires_grad', False) for v in metrics.values())

        metric_keys = list(sorted(metrics.keys()))  # Sort keys to guarantee all processes use the same order.
        # Reduce and reconstruct
        device = "cuda" if torch.cuda.is_available() else "cpu"
        metric_values = torch.stack([torch.as_tensor(metrics[k], device=device) if not isinstance(metrics[k], torch.Tensor) else metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}
            
            # Postprocess
            count = max(reduced_metrics["count"], 1)  # Avoid NaNs
            reduced_metrics = {f"train/{k}": v / (global_batch_size if k.endswith("loss") else count) for k, v in reduced_metrics.items()}

            reduced_metrics["train/lr"] = lr_this_step
            
            # Log gradient norms
            if getattr(config, 'enable_per_layer_grad_norm', False) and isinstance(grad_norm, dict):
                # Log per-component norms
                for key, value in grad_norms_dict.items():
                    reduced_metrics[f"train/{key}"] = value
                # Average for compatibility
                reduced_metrics["train/grad_norm"] = grad_norm
            else:
                # Global norm
                reduced_metrics["train/grad_norm"] = grad_norm if isinstance(grad_norm, float) else grad_norm.item()
            
            # Add gradient flow statistics if available
            if grad_stats is not None:
                for key, value in grad_stats.items():
                    reduced_metrics[f"train/grad/{key}"] = value
            
            # Periodic codebook maintenance (prevent collapse)
            if hasattr(train_state.model, 'model') and hasattr(train_state.model.model, 'inner'):
                inner = train_state.model.model.inner
                if hasattr(inner, 'lm_head') and hasattr(inner.lm_head, 'codebook'):
                    codebook = inner.lm_head.codebook
                    
                    # Reset dead codes every 5000 steps
                    if train_state.step % 5000 == 0 and train_state.step > 0:
                        codebook.reset_dead_codes(min_usage=0.01)
                    
                    # Log usage stats every 1000 steps
                    if train_state.step % 1000 == 0:
                        stats = codebook.get_usage_stats()
                        for key, value in stats.items():
                            reduced_metrics[f"train/codebook/{key}"] = value
            
            return reduced_metrics

def evaluate(
    config: PretrainConfig,
    train_state: TrainState,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: List[Any],
    rank: int,
    world_size: int,
    cpu_group: Optional[dist.ProcessGroup],
):
    reduced_metrics = None

    # Skip evaluation if no eval data
    if eval_metadata is None:
        if rank == 0:
            print("Skipping evaluation: no eval data available")
        return {}

    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)

        # Run evaluation
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}

        save_preds = {}

        metric_keys = []
        metric_values = None

        carry = None
        processed_batches = 0
        
        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1
            if rank == 0:
                print(f"Processing batch {processed_batches}: {set_name}")
            
            # Handle capsule mode (same as train_batch)
            if isinstance(batch, tuple):
                if len(batch) == 3:
                    sketches, checksums, children = batch
                    batch = {
                        'inputs': sketches.to(device),
                        'capsule_sketches': sketches.to(device),
                        'capsule_checksums': checksums.to(device),
                        'capsule_children': children.to(device),
                        'labels': torch.zeros(sketches.shape[0], dtype=torch.long).to(device),
                        'puzzle_identifiers': torch.zeros(sketches.shape[0], dtype=torch.long).to(device)
                    }
                elif len(batch) == 2:
                    sketches, checksums = batch
                    batch = {
                        'inputs': sketches.to(device),
                        'capsule_sketches': sketches.to(device),
                        'capsule_checksums': checksums.to(device),
                        'labels': torch.zeros(sketches.shape[0], dtype=torch.long).to(device),
                        'puzzle_identifiers': torch.zeros(sketches.shape[0], dtype=torch.long).to(device)
                    }
                else:
                    embeddings = batch[0].to(device)
                    batch = {
                        'inputs': embeddings,
                        'labels': torch.zeros(embeddings.shape[0], dtype=torch.long).to(device),
                        'puzzle_identifiers': torch.zeros(embeddings.shape[0], dtype=torch.long).to(device)
                    }
            else:
                # Standard dict mode
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Add CapsuleState for expansion tracking (if enabled)
                if hasattr(config.arch, 'enable_capsule_expansion') and config.arch.enable_capsule_expansion:
                    if 'inputs' in batch and batch['inputs'].dim() == 3:  # [B, k, D]
                        from models.capsule_state import CapsuleState
                        
                        # Create CapsuleState with available components
                        children = batch.get('capsule_children', None)  # [B, k, m, D] if available
                        checksums = batch.get('capsule_checksums', None)  # [B, k, R] if available
                        
                        capsule_state = CapsuleState(
                            sketches=batch['inputs'].clone(),  # [B, k, D]
                            children=children,  # Full expansion support if available
                            checksums=checksums,  # Reconstructability tracking
                        )
                        batch['capsule_state'] = capsule_state
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            with torch.device(device):
                carry = train_state.original_model.initial_carry(batch)  # type: ignore

            # Forward
            inference_steps = 0
            while True:
                carry, loss, metrics, preds, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=return_keys
                )
                inference_steps += 1

                if all_finish:
                    break

            if rank == 0:
                print(f"  Completed inference in {inference_steps} steps")

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        save_preds.setdefault(k, [])
                        save_preds[k].append(v.cpu())  # Move to CPU for saving GPU memory

            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)

            del carry, loss, preds, batch, all_finish

            # Aggregate metrics
            set_id = set_ids[set_name]

            if metric_values is None:
                metric_keys = list(
                    sorted(metrics.keys())
                )  # Sort keys to guarantee all processes use the same order.
                metric_values = torch.zeros(
                    (len(set_ids), len(metrics.values())), dtype=torch.float32, device=device
                )

            metric_values[set_id] += torch.stack([torch.as_tensor(metrics[k], device=device) if not isinstance(metrics[k], torch.Tensor) else metrics[k] for k in metric_keys])

            del metrics

        # concatenate save preds
        save_preds = {k: torch.cat(v, dim=0) for k, v in save_preds.items()}

        # Save preds
        if config.checkpoint_path is not None and len(save_preds):
            # Each rank save predictions independently
            os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
            torch.save(
                save_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}")
            )

        del save_preds

        # Reduce to rank 0
        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)

            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {
                        metric_name: reduced_metrics[set_id, metric_id]
                        for metric_id, metric_name in enumerate(metric_keys)
                    }
                    for set_id, set_name in enumerate(set_ids)
                }

                # Postprocess
                for set_name, m in reduced_metrics.items():
                    count = m.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in m.items()}

        # Run evaluators
        if rank == 0:
            print(f"\nRunning {len(evaluators)} evaluator(s)...")
            
        for i, evaluator in enumerate(evaluators):
            if rank == 0:
                print(f"Running evaluator {i+1}/{len(evaluators)}: {evaluator.__class__.__name__}")
                
            # Path for saving
            evaluator_save_path = None
            if config.checkpoint_path is not None:
                evaluator_save_path = os.path.join(
                    config.checkpoint_path,
                    f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}",
                )
                os.makedirs(evaluator_save_path, exist_ok=True)

            # Run and log
            metrics = evaluator.result(evaluator_save_path, rank=rank, world_size=world_size, group=cpu_group)
            if rank == 0 and metrics is not None:
                if reduced_metrics is None:
                    reduced_metrics = {}

                reduced_metrics.update(metrics)
                print(f"  Completed {evaluator.__class__.__name__}")
                
        if rank == 0:
            print("All evaluators completed!")

    return reduced_metrics

def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None or wandb.run is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)

            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    # Log code
    wandb.run.log_code(config.checkpoint_path)


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore

        # Naming
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_paths[0]).capitalize()}-ACT-torch"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1
    CPU_PROCESS_GROUP = None

    CPU_PROCESS_GROUP = None

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        
        # CPU GLOO process group
        CPU_PROCESS_GROUP = dist.new_group(backend="gloo")
        assert (
            dist.get_rank(CPU_PROCESS_GROUP) == RANK and dist.get_world_size(CPU_PROCESS_GROUP) == WORLD_SIZE
        )

    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)
    
    # Validate critical config parameters
    if RANK == 0:
        assert 0 < config.epochs <= 1_000_000, f"Invalid epochs: {config.epochs}"
        assert 1 <= config.global_batch_size <= 8192, f"Invalid batch_size: {config.global_batch_size}"
        assert 0 < config.lr <= 1.0, f"Invalid learning rate: {config.lr}"
        assert config.lr_min_ratio >= 0, f"Invalid lr_min_ratio: {config.lr_min_ratio}"
        
        if config.eval_interval is not None:
            assert config.epochs % config.eval_interval == 0, \
                f"eval_interval ({config.eval_interval}) must divide epochs ({config.epochs})"

    # Register signal handler for graceful shutdown (only on rank 0 to avoid duplicate messages)
    if RANK == 0:
        signal.signal(signal.SIGINT, shutdown_signal_handler)
        print("\n" + "="*70)
        print("  INFO: Graceful shutdown enabled")
        print("  TIP: Press Ctrl+C to stop training and save checkpoint")
        print("="*70 + "\n")

    # Auto-resume: Check if checkpoint exists and auto-load if not specified
    if config.load_checkpoint is None and config.checkpoint_path is not None:
        # Look for dataset-specific checkpoint (e.g., text.pt, arc.pt)
        checkpoint_name = get_checkpoint_name(config.data_paths[0] if config.data_paths else 'model')
        checkpoint_file = os.path.join(config.checkpoint_path, checkpoint_name)
        
        # Fallback to latest.pt for backward compatibility
        if not os.path.exists(checkpoint_file):
            checkpoint_file = os.path.join(config.checkpoint_path, "latest.pt")
        
        if os.path.exists(checkpoint_file):
            if RANK == 0:
                print(f"\n{'='*70}")
                print(f"  AUTO-RESUME: Found existing checkpoint")
                print(f"  Loading from: {checkpoint_file}")
                print(f"{'='*70}\n")
            config.load_checkpoint = checkpoint_file
    
    # Initialize training progress tracker for consolidated dataset chunks
    progress_tracker = None
    if config.checkpoint_path is not None and RANK == 0:
        dataset_name = get_checkpoint_name(config.data_paths[0] if config.data_paths else 'model').replace('.pt', '')
        progress_tracker = TrainingProgressTracker(
            checkpoint_dir=Path(config.checkpoint_path),
            dataset_name=dataset_name
        )
        
        # Load existing progress if resuming
        if progress_tracker.has_progress():
            progress_tracker.load()
            
            # Show dataset manifest
            if config.data_paths:
                # Extract checkpoint directory from data path
                data_path = Path(config.data_paths[0])
                checkpoint_dir = data_path.parent / 'stream_checkpoints'
                if checkpoint_dir.exists():
                    manifest = get_training_manifest(checkpoint_dir)
                    print(f"\nDataset Manifest:")
                    print(f"   Available chunks: {len(manifest['available_chunks'])}")
                    print(f"   Chunks IDs: {manifest['available_chunks'][:10]}..." if len(manifest['available_chunks']) > 10 else f"   Chunk IDs: {manifest['available_chunks']}")
    
    # Seed RNGs to ensure consistency
    torch.random.manual_seed(config.seed + RANK)

    # Dataset
    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    train_loader, train_metadata = load_datasets(config, rank=RANK, world_size=WORLD_SIZE, split="train")
    try:
        eval_loader,  eval_metadata  = load_datasets(config, rank=RANK, world_size=WORLD_SIZE, split="test")
    except Exception as e:
        if RANK == 0:
            print(f"WARNING: No eval data found: {e}")
            print("   Training will continue without evaluation")
        eval_loader = eval_metadata = None

    try:
        evaluators = create_evaluators(config, eval_metadata)
    except Exception as e:
        if RANK == 0:
            print(f"WARNING: Failed to create evaluators: {e}")
            print("   Training will continue without custom evaluators")
        evaluators = []

    # Train state
    train_state = init_train_state(config, train_metadata, rank=RANK, world_size=WORLD_SIZE)

    # Progress bar and logger
    progress_bar = None
    ema_helper = None
    gradient_monitor = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)
        wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump(), settings=wandb.Settings(_disable_stats=True))  # type: ignore
        
        # Count parameters from original model (before torch.compile)
        num_params = sum(x.numel() for x in train_state.original_model.parameters())
        wandb.log({"num_params": num_params}, step=0)
        print(f"   Total parameters: {num_params:,}")
        save_code_and_config(config)
        
        # Initialize gradient flow monitor
        gradient_monitor = GradientFlowMonitor(train_state.model, track_every_n_steps=100)
        print('Initialized gradient flow monitor (tracking every 100 steps)')
        
    if config.ema:
        print('Setup EMA')
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.original_model)  # Use unwrapped model for EMA

    # Initialize Async Checkpoint Manager
    checkpoint_manager = AsyncCheckpointManager(
        checkpoint_dir=config.checkpoint_path,
        max_keep=3,
        use_safetensors=True
    )

    # Training Loop
    # Calculate starting iteration from restored step (for auto-resume)
    steps_per_iter = train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size
    start_iter = int(train_state.step / steps_per_iter) if train_state.step > 0 else 0
    
    if RANK == 0 and start_iter > 0:
        print(f"\n{'='*70}")
        print(f"  [RESUME] Resuming from iteration {start_iter}/{total_iters}")
        print(f"  [INFO] Epoch: {start_iter * train_epochs_per_iter}/{config.epochs}")
        print(f"  [INFO] Step: {train_state.step}/{train_state.total_steps}")
        print(f"{'='*70}\n")
    
    # Graceful shutdown handler
    try:
        for _iter_id in range(start_iter, total_iters):
            # Check for shutdown request at start of each iteration
            if shutdown_requested:
                if RANK == 0:
                    print("\n" + "="*70)
                    print("  💾 GRACEFUL SHUTDOWN IN PROGRESS")
                    print("="*70)
                    print(f"\n  Saving checkpoint at step {train_state.step}...")
                    print(f"\n  Saving checkpoint at step {train_state.step}...")
                    if RANK == 0:
                        checkpoint_manager.save_async(
                            step=train_state.step,
                            model=train_state.original_model,
                            optimizer=train_state.optimizers,
                            scaler=None,
                            scheduler=None,
                            extra_data={
                                'carry': train_state.carry,
                                'config_epochs': config.epochs,
                                'total_steps': train_state.total_steps,
                                'dataset': config.data_paths[0] if config.data_paths else 'unknown',
                            }
                        )
                        checkpoint_manager.wait_for_completion()
                    print("\n[STOP] Training stopped successfully!")
                    print(f"   Final step: {train_state.step}/{train_state.total_steps}")
                    print(f"   Progress: {(train_state.step/train_state.total_steps)*100:.1f}%")
                    print(f"   Checkpoint saved to: {config.checkpoint_path}")
                    print("\n[INFO] Resume training with the same command.\n")
                
                # Clean exit
                if dist.is_initialized():
                    dist.destroy_process_group()
                if RANK == 0:
                    checkpoint_manager.shutdown()
                    wandb.finish()
                return  # Exit gracefully
            
            print (f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {_iter_id * train_epochs_per_iter}")

            ############ Train Iter
            if RANK == 0:
                print("TRAIN")
            train_state.original_model.train()  # Use unwrapped model for .train()
            for set_name, batch, global_batch_size in train_loader:
                metrics = train_batch(config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE, gradient_monitor=gradient_monitor)
                
                # Periodic memory cleanup to prevent leaks
                if RANK == 0 and train_state.step % 1000 == 0 and gradient_monitor is not None:
                    gradient_monitor.clear_old_data(keep_last_n=100)
                    gradient_monitor.clear_old_data(keep_last_n=100)

                if RANK == 0 and metrics is not None:
                    wandb.log(metrics, step=train_state.step)
                    progress_bar.update(train_state.step - progress_bar.n)  # type: ignore
                if config.ema:
                    ema_helper.update(train_state.model)

            if _iter_id >= config.min_eval_interval:
                ############ Evaluation
                if RANK == 0:
                    print("EVALUATE")
                if config.ema:
                    print("SWITCH TO EMA")
                    train_state_eval = copy.deepcopy(train_state)
                    train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
                else:
                    train_state_eval = train_state
                train_state_eval.original_model.eval()  # Use unwrapped model for .eval()
                metrics = evaluate(config, 
                    train_state_eval, 
                    eval_loader, 
                    eval_metadata, 
                    evaluators,
                    rank=RANK, 
                    world_size=WORLD_SIZE,
                    cpu_group=CPU_PROCESS_GROUP)

                if RANK == 0 and metrics is not None:
                    wandb.log(metrics, step=train_state.step)
                    
                ############ Checkpointing
                if RANK == 0:
                    print("SAVE CHECKPOINT")
                if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
                    checkpoint_manager.save_async(
                        step=train_state.step,
                        model=train_state.original_model,
                        optimizer=train_state.optimizers,
                        scaler=None,
                        scheduler=None,
                        extra_data={
                            'carry': train_state.carry,
                            'config_epochs': config.epochs,
                            'total_steps': train_state.total_steps,
                            'dataset': config.data_paths[0] if config.data_paths else 'unknown',
                        }
                    )

                if config.ema:
                    del train_state_eval
    
    except KeyboardInterrupt:
        if RANK == 0:
            print("\n" + "="*70)
            print("  [STOP] TRAINING INTERRUPTED (Ctrl+C)")
            print("="*70)
            
            # Calculate lost progress
            lost_steps = train_state.step - train_state.last_checkpoint_step
            progress_pct = (train_state.last_checkpoint_step / train_state.total_steps) * 100 if train_state.total_steps > 0 else 0
            
            print("\nWARNING: NOT saving current state (may be unsafe during training)")
            print("\nCheckpoint Status:")
            print(f"   Last safe checkpoint: Step {train_state.last_checkpoint_step:,}")
            print(f"   Current training step: Step {train_state.step:,}")
            print(f"   Lost progress: {lost_steps:,} steps")
            print(f"   Saved progress: {progress_pct:.1f}% of total training")
            
            if config.checkpoint_path and config.data_paths:
                checkpoint_name = get_checkpoint_name(config.data_paths[0])
                print(f"\nLast checkpoint: {config.checkpoint_path}/{checkpoint_name}")
            
            print("\nINFO: Resume training with the same command to continue from last checkpoint.")
            print("\n" + "="*70)
            print("  Training stopped by user")
            print("="*70 + "\n")
        
        # Clean up
        if dist.is_initialized():
            dist.destroy_process_group()
        if RANK == 0:
            wandb.finish()
        
        sys.exit(0)

    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()

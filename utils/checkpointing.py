import os
import torch
import threading
import queue
import time
import json
import shutil
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path

try:
    from safetensors.torch import save_file as safe_save_file
    from safetensors.torch import load_file as safe_load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

logger = logging.getLogger(__name__)

class AsyncCheckpointManager:
    """
    Manages model checkpoints asynchronously to prevent training stalls.
    Uses safetensors for model weights (if available) and torch.save for optimizer states.
    """
    
    def __init__(self, checkpoint_dir: str, max_keep: int = 3, use_safetensors: bool = True):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep
        self.use_safetensors = use_safetensors and SAFETENSORS_AVAILABLE
        
        if use_safetensors and not SAFETENSORS_AVAILABLE:
            logger.warning("Safetensors requested but not installed. Falling back to torch.save.")
        
        # Queue for save requests
        self.save_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
        self.last_saved_step = -1
        
    def save_async(self, 
                   step: int, 
                   model: torch.nn.Module, 
                   optimizer: Union[torch.optim.Optimizer, List[torch.optim.Optimizer], None] = None,
                   scaler: Optional[Any] = None,
                   scheduler: Optional[Any] = None,
                   extra_data: Dict[str, Any] = None):
        """
        Submit a save request. This method returns immediately after copying tensors to CPU.
        """
        start_time = time.time()
        
        # 1. Prepare state dicts (CPU copy is fast but blocking - necessary for consistency)
        # We must move to CPU here because the model on GPU will change immediately after this call
        
        # Handle DDP/FSDP unwrapping if necessary (simplified for single/DDP)
        model_to_save = model.module if hasattr(model, 'module') else model
        
        # Copy model state to CPU
        model_state = {k: v.cpu().clone() for k, v in model_to_save.state_dict().items()}
        
        # Copy optimizer state
        optimizer_state = None
        if optimizer is not None:
            if isinstance(optimizer, list):
                optimizer_state = [opt.state_dict() for opt in optimizer]
            else:
                optimizer_state = optimizer.state_dict()
            
        payload = {
            'step': step,
            'model_state': model_state,
            'optimizer_state': optimizer_state,
            'scaler_state': scaler.state_dict() if scaler else None,
            'scheduler_state': scheduler.state_dict() if scheduler else None,
            'extra_data': extra_data or {}
        }
        
        # Add to queue
        self.save_queue.put(payload)
        
        logger.info(f"Checkpoint {step} queued for background saving (prep time: {time.time()-start_time:.4f}s)")

    def _worker_loop(self):
        while not self.stop_event.is_set():
            try:
                payload = self.save_queue.get(timeout=1.0)
            except queue.Empty:
                continue
                
            try:
                self._do_save(payload)
            except Exception as e:
                logger.error(f"Failed to save checkpoint asynchronously: {e}", exc_info=True)
            finally:
                self.save_queue.task_done()

    def _do_save(self, payload):
        step = payload['step']
        save_dir = self.checkpoint_dir / f"checkpoint-{step}"
        save_dir.mkdir(exist_ok=True)
        
        # Paths
        model_path = save_dir / ("model.safetensors" if self.use_safetensors else "model.pt")
        optimizer_path = save_dir / "optimizer.pt"
        meta_path = save_dir / "metadata.json"
        
        # 1. Save Model Weights
        if self.use_safetensors:
            safe_save_file(payload['model_state'], model_path)
        else:
            torch.save(payload['model_state'], model_path)
            
        # 2. Save Optimizer & Scheduler (always torch.save as they contain python objects)
        train_state = {
            'optimizer': payload['optimizer_state'],
            'scaler': payload['scaler_state'],
            'scheduler': payload['scheduler_state'],
            'extra': payload['extra_data']
        }
        torch.save(train_state, optimizer_path)
        
        # 3. Save Metadata
        with open(meta_path, 'w') as f:
            json.dump({'step': step, 'timestamp': time.time()}, f)
            
        # 4. Cleanup old checkpoints
        self._rotate_checkpoints()
        
        logger.info(f"Checkpoint {step} saved successfully to {save_dir}")
        self.last_saved_step = step

    def _rotate_checkpoints(self):
        # Find all checkpoint folders
        checkpoints = sorted(
            [d for d in self.checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda x: int(x.name.split('-')[-1])
        )
        
        # Remove oldest
        while len(checkpoints) > self.max_keep:
            to_remove = checkpoints.pop(0)
            shutil.rmtree(to_remove)
            logger.info(f"Removed old checkpoint: {to_remove}")

    def wait_for_completion(self):
        """Block until all pending saves are finished."""
        self.save_queue.join()

    def shutdown(self):
        """Finish pending saves and stop worker."""
        self.wait_for_completion()
        self.stop_event.set()
        self.worker_thread.join()

    def load(self, path: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None):
        """
        Load checkpoint from path. Handles both safetensors and pt.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint path not found: {path}")
            
        # Detect format
        if (path / "model.safetensors").exists():
            model_path = path / "model.safetensors"
            is_safetensors = True
        elif (path / "model.pt").exists():
            model_path = path / "model.pt"
            is_safetensors = False
        else:
            # Legacy single-file checkpoint support
            if path.is_file():
                return self._load_legacy(path, model, optimizer)
            raise FileNotFoundError(f"No model file found in {path}")
            
        logger.info(f"Loading model from {model_path}...")
        
        # Load Model
        if is_safetensors:
            state_dict = safe_load_file(model_path)
        else:
            state_dict = torch.load(model_path, map_location='cpu')
            
        # Handle DDP unwrapping
        model_to_load = model.module if hasattr(model, 'module') else model
        model_to_load.load_state_dict(state_dict)
        
        # Load Optimizer/Train State
        optimizer_path = path / "optimizer.pt"
        extra_data = {}
        if optimizer_path.exists():
            logger.info(f"Loading optimizer state from {optimizer_path}...")
            train_state = torch.load(optimizer_path, map_location='cpu')
            
            if optimizer and 'optimizer' in train_state:
                optimizer.load_state_dict(train_state['optimizer'])
            
            extra_data = train_state.get('extra', {})
            
        return extra_data

    def _load_legacy(self, path, model, optimizer):
        logger.info(f"Loading legacy checkpoint from {path}...")
        checkpoint = torch.load(path, map_location='cpu')
        
        model_to_load = model.module if hasattr(model, 'module') else model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        return checkpoint.get('extra_data', {})

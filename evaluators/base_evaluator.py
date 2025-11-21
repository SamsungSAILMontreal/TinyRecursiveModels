"""
Base evaluator interface for extensible evaluation.

All evaluators should inherit from BaseEvaluator and implement:
- required_outputs: set of keys needed from model
- begin_eval(): Called before evaluation starts
- update_batch(batch, preds): Process each batch
- result(save_path, rank, world_size, group): Compute final metrics
"""

from typing import Dict, Set, Optional, Any
from abc import ABC, abstractmethod
import torch
import torch.distributed as dist


class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluators.
    
    Provides common functionality and defines the interface.
    """
    
    @property
    @abstractmethod
    def required_outputs(self) -> Set[str]:
        """
        Set of output keys required from the model.
        
        Example: {"inputs", "puzzle_identifiers", "logits", "preds"}
        """
        pass
    
    @abstractmethod
    def begin_eval(self):
        """
        Called before evaluation loop starts.
        
        Use this to reset counters, clear caches, etc.
        """
        pass
    
    @abstractmethod
    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        """
        Process a single batch during evaluation.
        
        Args:
            batch: Input batch from dataloader
            preds: Model predictions (output, carry)
        """
        pass
    
    @abstractmethod
    def result(
        self, 
        save_path: Optional[str], 
        rank: int, 
        world_size: int, 
        group: Optional[dist.ProcessGroup] = None
    ) -> Optional[Dict[str, float]]:
        """
        Compute and return final evaluation metrics.
        
        Args:
            save_path: Directory to save outputs (can be None)
            rank: Current process rank (distributed training)
            world_size: Total number of processes
            group: Process group for distributed communication
            
        Returns:
            Dictionary of metrics (only on rank 0, None on other ranks)
            Example: {"accuracy": 0.85, "pass@1": 0.60}
        """
        pass
    
    def gather_to_rank0(self, data: Any, rank: int, world_size: int, group: Optional[dist.ProcessGroup] = None) -> Optional[list]:
        """
        Helper: Gather data from all ranks to rank 0.
        
        Args:
            data: Local data to gather
            rank: Current rank
            world_size: Total processes
            group: Process group
            
        Returns:
            List of data from all ranks (only on rank 0, None elsewhere)
        """
        gathered = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(data, gathered, dst=0, group=group)
        return gathered
    
    def log(self, message: str, rank: int):
        """Helper: Print message only on rank 0."""
        if rank == 0:
            print(message)


class AccuracyEvaluator(BaseEvaluator):
    """
    Simple accuracy evaluator for classification tasks.
    
    Usage in config:
        evaluators:
          - name: evaluators.base_evaluator@AccuracyEvaluator
            top_k: [1, 5]
    """
    
    required_outputs = {"labels", "logits"}
    
    def __init__(self, eval_metadata, top_k=(1, 5), **kwargs):
        self.top_k = top_k
        self.correct = {k: 0 for k in top_k}
        self.total = 0
    
    def begin_eval(self):
        self.correct = {k: 0 for k in self.top_k}
        self.total = 0
    
    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        logits = preds["logits"]
        labels = batch["labels"]
        
        # Move to CPU
        logits = logits.cpu()
        labels = labels.cpu()
        
        # Compute top-k accuracy
        for k in self.top_k:
            _, topk_preds = torch.topk(logits, k, dim=-1)
            correct = (topk_preds == labels.unsqueeze(-1)).any(dim=-1).sum().item()
            self.correct[k] += correct
        
        self.total += labels.numel()
    
    def result(self, save_path, rank, world_size, group=None):
        # Gather results
        local_data = (self.correct, self.total)
        gathered = self.gather_to_rank0(local_data, rank, world_size, group)
        
        if rank != 0:
            return None
        
        # Aggregate from all ranks
        total_correct = {k: 0 for k in self.top_k}
        total_samples = 0
        
        for correct_dict, count in gathered:
            for k in self.top_k:
                total_correct[k] += correct_dict[k]
            total_samples += count
        
        # Compute accuracies
        metrics = {}
        for k in self.top_k:
            acc = total_correct[k] / total_samples if total_samples > 0 else 0.0
            metrics[f"accuracy/top{k}"] = acc
        
        return metrics


class PerplexityEvaluator(BaseEvaluator):
    """
    Perplexity evaluator for language modeling tasks.
    
    Usage in config:
        evaluators:
          - name: evaluators.base_evaluator@PerplexityEvaluator
    """
    
    required_outputs = {"labels", "logits"}
    
    def __init__(self, eval_metadata, **kwargs):
        self.total_loss = 0.0
        self.total_tokens = 0
    
    def begin_eval(self):
        self.total_loss = 0.0
        self.total_tokens = 0
    
    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        import torch.nn.functional as F
        
        logits = preds["logits"]
        labels = batch["labels"]
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction='sum'
        )
        
        # Count valid tokens
        valid_tokens = (labels != -100).sum().item()
        
        self.total_loss += loss.item()
        self.total_tokens += valid_tokens
    
    def result(self, save_path, rank, world_size, group=None):
        # Gather results
        local_data = (self.total_loss, self.total_tokens)
        gathered = self.gather_to_rank0(local_data, rank, world_size, group)
        
        if rank != 0:
            return None
        
        # Aggregate
        total_loss = sum(loss for loss, _ in gathered)
        total_tokens = sum(tokens for _, tokens in gathered)
        
        # Compute perplexity
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            "perplexity": perplexity,
            "loss": avg_loss
        }

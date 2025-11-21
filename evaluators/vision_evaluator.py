"""
Vision task evaluators (classification, detection, captioning).

Evaluates model's visual understanding capabilities.
"""

from typing import Dict, Set, Optional
import torch
import torch.distributed as dist
import torch.nn.functional as F

from evaluators.base_evaluator import BaseEvaluator


class ImageClassificationEvaluator(BaseEvaluator):
    """
    Evaluator for image classification (CIFAR, ImageNet style).
    
    Usage in config:
        evaluators:
          - name: evaluators.vision_evaluator@ImageClassificationEvaluator
            top_k: [1, 5]
    """
    
    required_outputs = {"labels", "logits"}
    
    def __init__(self, eval_metadata, top_k=(1, 5), **kwargs):
        self.top_k = top_k
        self.correct = {k: 0 for k in top_k}
        self.total = 0
        self.per_class_correct = {}
        self.per_class_total = {}
    
    def begin_eval(self):
        self.correct = {k: 0 for k in self.top_k}
        self.total = 0
        self.per_class_correct = {}
        self.per_class_total = {}
    
    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        logits = preds["logits"].cpu()
        labels = batch["labels"].cpu()
        
        # Top-k accuracy
        for k in self.top_k:
            _, topk_preds = torch.topk(logits, k, dim=-1)
            correct_mask = (topk_preds == labels.unsqueeze(-1)).any(dim=-1)
            self.correct[k] += correct_mask.sum().item()
        
        # Per-class accuracy (for top-1)
        _, top1_preds = torch.topk(logits, 1, dim=-1)
        top1_preds = top1_preds.squeeze(-1)
        
        for label_val in labels.unique():
            label_val = label_val.item()
            mask = labels == label_val
            correct = (top1_preds[mask] == label_val).sum().item()
            
            self.per_class_correct[label_val] = self.per_class_correct.get(label_val, 0) + correct
            self.per_class_total[label_val] = self.per_class_total.get(label_val, 0) + mask.sum().item()
        
        self.total += labels.size(0)
    
    def result(self, save_path, rank, world_size, group=None):
        # Gather results
        local_data = (self.correct, self.total, self.per_class_correct, self.per_class_total)
        gathered = self.gather_to_rank0(local_data, rank, world_size, group)
        
        if rank != 0:
            return None
        
        # Aggregate
        total_correct = {k: 0 for k in self.top_k}
        total_samples = 0
        total_per_class_correct = {}
        total_per_class_total = {}
        
        for correct_dict, count, pc_correct, pc_total in gathered:
            for k in self.top_k:
                total_correct[k] += correct_dict[k]
            total_samples += count
            
            for class_id, correct_count in pc_correct.items():
                total_per_class_correct[class_id] = total_per_class_correct.get(class_id, 0) + correct_count
            for class_id, total_count in pc_total.items():
                total_per_class_total[class_id] = total_per_class_total.get(class_id, 0) + total_count
        
        # Compute metrics
        metrics = {}
        for k in self.top_k:
            acc = total_correct[k] / total_samples if total_samples > 0 else 0.0
            metrics[f"vision/top{k}_accuracy"] = acc
        
        # Mean per-class accuracy
        per_class_accs = []
        for class_id in total_per_class_total:
            if total_per_class_total[class_id] > 0:
                acc = total_per_class_correct[class_id] / total_per_class_total[class_id]
                per_class_accs.append(acc)
        
        mean_per_class_acc = sum(per_class_accs) / len(per_class_accs) if per_class_accs else 0.0
        metrics["vision/mean_per_class_accuracy"] = mean_per_class_acc
        
        self.log(f"\n{'='*60}", rank)
        self.log(f"Image Classification Results:", rank)
        for k in self.top_k:
            self.log(f"  Top-{k} Accuracy: {metrics[f'vision/top{k}_accuracy']:.2%}", rank)
        self.log(f"  Mean Per-Class Acc: {mean_per_class_acc:.2%}", rank)
        self.log(f"{'='*60}\n", rank)
        
        return metrics


class ImageCaptioningEvaluator(BaseEvaluator):
    """
    Evaluator for image captioning (COCO style).
    
    Computes BLEU, ROUGE, CIDEr metrics.
    
    Usage in config:
        evaluators:
          - name: evaluators.vision_evaluator@ImageCaptioningEvaluator
            metrics: [bleu, rouge, cider]
    """
    
    required_outputs = {"puzzle_identifiers", "preds"}
    
    def __init__(self, eval_metadata, metrics=("bleu",), **kwargs):
        self.metrics = metrics
        self.predictions = []
        self.references = []
    
    def begin_eval(self):
        self.predictions = []
        self.references = []
    
    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        """
        Store predictions and references.
        
        In practice, would decode tokens to text.
        """
        # Placeholder: decode predictions and store
        pass
    
    def compute_bleu(self, predictions: list, references: list) -> float:
        """Compute BLEU score (simplified)."""
        # In production, use nltk.translate.bleu_score
        return 0.0  # Placeholder
    
    def result(self, save_path, rank, world_size, group=None):
        # Gather results
        local_data = (self.predictions, self.references)
        gathered = self.gather_to_rank0(local_data, rank, world_size, group)
        
        if rank != 0:
            return None
        
        # Flatten
        all_preds = []
        all_refs = []
        for preds, refs in gathered:
            all_preds.extend(preds)
            all_refs.extend(refs)
        
        # Compute metrics
        metrics = {}
        if "bleu" in self.metrics:
            metrics["caption/bleu"] = self.compute_bleu(all_preds, all_refs)
        
        return metrics

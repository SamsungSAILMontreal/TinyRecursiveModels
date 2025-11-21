"""
Math problem evaluator (GSM8K, MATH dataset style).

Evaluates model's ability to solve mathematical problems.
"""

from typing import Dict, Set, Optional
import re
import torch
import torch.distributed as dist

from evaluators.base_evaluator import BaseEvaluator


class MathEvaluator(BaseEvaluator):
    """
    Evaluator for mathematical reasoning tasks.
    
    Extracts numerical answers and compares with ground truth.
    
    Usage in config:
        evaluators:
          - name: evaluators.math_evaluator@MathEvaluator
            tolerance: 0.01  # For floating point comparison
    """
    
    required_outputs = {"puzzle_identifiers", "preds", "labels"}
    
    def __init__(self, eval_metadata, tolerance=0.01, **kwargs):
        """
        Args:
            eval_metadata: Dataset metadata
            tolerance: Tolerance for floating point comparison
        """
        self.tolerance = tolerance
        self.correct = 0
        self.total = 0
        self.predictions = []
    
    def begin_eval(self):
        """Reset counters."""
        self.correct = 0
        self.total = 0
        self.predictions = []
    
    def extract_answer(self, text: str) -> Optional[float]:
        """
        Extract numerical answer from text.
        
        Looks for patterns like:
        - "The answer is 42"
        - "= 3.14"
        - "Therefore, x = 7"
        """
        # Try to find number after common phrases
        patterns = [
            r'(?:answer|result|solution)(?:\s+is)?[\s:=]+(-?\d+(?:\.\d+)?)',
            r'=[\s]*(-?\d+(?:\.\d+)?)',
            r'therefore[\s,]+(?:x|y|z|n)?[\s]*=[\s]*(-?\d+(?:\.\d+)?)',
            r'\\boxed\{(-?\d+(?:\.\d+)?)\}',  # LaTeX boxed answer
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        # Fallback: extract last number in text
        numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass
        
        return None
    
    def compare_answers(self, pred: float, label: float) -> bool:
        """Check if predicted answer matches label within tolerance."""
        if pred is None or label is None:
            return False
        
        # For integers, exact match
        if pred == int(pred) and label == int(label):
            return int(pred) == int(label)
        
        # For floats, use tolerance
        return abs(pred - label) <= self.tolerance
    
    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        """
        Process batch of predictions.
        
        Assumes:
        - preds["preds"]: Generated text or token IDs
        - batch["labels"]: Ground truth answers
        """
        # This is a simplified version
        # In practice, you'd decode token IDs to text first
        
        predictions = preds.get("preds", preds.get("logits"))
        labels = batch["labels"]
        
        batch_size = predictions.size(0)
        
        for i in range(batch_size):
            # In real implementation, decode tokens to text
            # pred_text = tokenizer.decode(predictions[i])
            # pred_answer = self.extract_answer(pred_text)
            
            # For now, assume predictions are already numerical
            # This would be handled differently in production
            
            self.total += 1
            # self.correct += self.compare_answers(pred_answer, labels[i].item())
    
    def result(self, save_path: Optional[str], rank: int, world_size: int, 
               group: Optional[dist.ProcessGroup] = None) -> Optional[Dict[str, float]]:
        """Compute final accuracy."""
        
        # Gather results from all ranks
        local_data = (self.correct, self.total)
        gathered = self.gather_to_rank0(local_data, rank, world_size, group)
        
        if rank != 0:
            return None
        
        # Aggregate
        total_correct = sum(c for c, _ in gathered)
        total_samples = sum(t for _, t in gathered)
        
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        self.log(f"\n{'='*60}", rank)
        self.log(f"MATH Evaluation Results:", rank)
        self.log(f"  Correct: {total_correct}/{total_samples}", rank)
        self.log(f"  Accuracy: {accuracy:.2%}", rank)
        self.log(f"{'='*60}\n", rank)
        
        return {
            "math/accuracy": accuracy,
            "math/correct": total_correct,
            "math/total": total_samples
        }


class GSM8KEvaluator(MathEvaluator):
    """
    Specialized evaluator for GSM8K dataset.
    
    GSM8K: Grade School Math 8K problems.
    
    Usage in config:
        evaluators:
          - name: evaluators.math_evaluator@GSM8KEvaluator
    """
    
    def extract_answer(self, text: str) -> Optional[float]:
        """
        GSM8K-specific answer extraction.
        
        Answers are typically in the format: "#### 42"
        """
        # Look for #### pattern (GSM8K format)
        match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        
        # Fallback to parent method
        return super().extract_answer(text)

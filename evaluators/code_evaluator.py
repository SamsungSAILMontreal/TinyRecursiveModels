"""
Code generation evaluator (HumanEval, MBPP style).

Evaluates model's ability to generate correct code.
"""

from typing import Dict, Set, Optional, List
import re
import subprocess
import tempfile
import os
import torch
import torch.distributed as dist

from evaluators.base_evaluator import BaseEvaluator


class CodeEvaluator(BaseEvaluator):
    """
    Evaluator for code generation tasks.
    
    Executes generated code against test cases.
    
    Usage in config:
        evaluators:
          - name: evaluators.code_evaluator@CodeEvaluator
            timeout: 5  # Execution timeout in seconds
            pass_k: [1, 10, 100]  # pass@k metrics
    """
    
    required_outputs = {"puzzle_identifiers", "preds"}
    
    def __init__(self, eval_metadata, timeout=5, pass_k=(1, 10, 100), **kwargs):
        """
        Args:
            eval_metadata: Dataset metadata
            timeout: Max execution time per test
            pass_k: List of k values for pass@k metric
        """
        self.timeout = timeout
        self.pass_k = pass_k
        self.results = []  # List of (problem_id, passed, code)
    
    def begin_eval(self):
        """Reset results."""
        self.results = []
    
    def extract_code(self, text: str, language="python") -> str:
        """
        Extract code from model output.
        
        Looks for code blocks like:
        ```python
        def solution():
            ...
        ```
        """
        # Try to find code block
        pattern = rf'```{language}\n(.*?)```'
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        # Fallback: return entire text if it looks like code
        if 'def ' in text or 'import ' in text:
            return text.strip()
        
        return ""
    
    def execute_code(self, code: str, test_cases: List[str]) -> bool:
        """
        Execute code against test cases.
        
        Args:
            code: Generated code
            test_cases: List of test assertions
            
        Returns:
            True if all tests pass, False otherwise
        """
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Write code and tests
            f.write(code + '\n\n')
            for test in test_cases:
                f.write(test + '\n')
            temp_path = f.name
        
        try:
            # Execute with timeout
            result = subprocess.run(
                ['python', temp_path],
                timeout=self.timeout,
                capture_output=True,
                text=True
            )
            
            # Check if execution succeeded
            passed = result.returncode == 0
            
        except subprocess.TimeoutExpired:
            passed = False
        except Exception as e:
            passed = False
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        return passed
    
    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        """
        Process batch of code generations.
        
        Note: This is simplified. Real implementation would:
        1. Decode token IDs to text
        2. Extract code blocks
        3. Load test cases for each problem
        4. Execute and record results
        """
        # Placeholder: In production, decode and execute
        # For now, just track that we processed the batch
        pass
    
    def compute_pass_at_k(self, n: int, c: int, k: int) -> float:
        """
        Compute pass@k metric.
        
        Args:
            n: Total number of samples
            c: Number of correct samples
            k: k value
            
        Returns:
            pass@k probability
        """
        if n < k:
            return 0.0
        
        if c >= k:
            return 1.0
        
        # Unbiased estimator: 1 - comb(n-c, k) / comb(n, k)
        from math import comb
        return 1.0 - comb(n - c, k) / comb(n, k)
    
    def result(self, save_path: Optional[str], rank: int, world_size: int,
               group: Optional[dist.ProcessGroup] = None) -> Optional[Dict[str, float]]:
        """Compute pass@k metrics."""
        
        # Gather results from all ranks
        gathered = self.gather_to_rank0(self.results, rank, world_size, group)
        
        if rank != 0:
            return None
        
        # Flatten results
        all_results = []
        for result_list in gathered:
            all_results.extend(result_list)
        
        # Group by problem ID
        problems = {}
        for problem_id, passed, code in all_results:
            problems.setdefault(problem_id, []).append(passed)
        
        # Compute pass@k for each k
        metrics = {}
        for k in self.pass_k:
            pass_rates = []
            for problem_id, results in problems.items():
                n = len(results)
                c = sum(results)
                pass_rates.append(self.compute_pass_at_k(n, c, k))
            
            avg_pass_rate = sum(pass_rates) / len(pass_rates) if pass_rates else 0.0
            metrics[f"code/pass@{k}"] = avg_pass_rate
        
        self.log(f"\n{'='*60}", rank)
        self.log(f"Code Generation Results:", rank)
        for k in self.pass_k:
            self.log(f"  pass@{k}: {metrics[f'code/pass@{k}']:.2%}", rank)
        self.log(f"{'='*60}\n", rank)
        
        return metrics


class HumanEvalEvaluator(CodeEvaluator):
    """
    Specialized evaluator for HumanEval benchmark.
    
    HumanEval: 164 hand-written programming problems.
    
    Usage in config:
        evaluators:
          - name: evaluators.code_evaluator@HumanEvalEvaluator
            pass_k: [1, 10, 100]
    """
    
    def __init__(self, eval_metadata, **kwargs):
        super().__init__(eval_metadata, **kwargs)
        # Could load HumanEval-specific test cases here

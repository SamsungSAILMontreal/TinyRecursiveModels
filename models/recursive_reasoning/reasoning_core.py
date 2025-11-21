import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any

class ReasoningGenerator(nn.Module):
    """
    Manages Chain-of-Thought (CoT) reasoning generation.
    
    Enforces reasoning budgets and validates thought chains.
    """
    
    def __init__(
        self,
        max_thought_tokens: int = 128,
        thought_start_token_id: int = None,
        thought_end_token_id: int = None,
    ):
        super().__init__()
        self.max_thought_tokens = max_thought_tokens
        self.thought_start_token_id = thought_start_token_id
        self.thought_end_token_id = thought_end_token_id
        
    def check_budget(self, current_thought_length: int) -> bool:
        """
        Check if reasoning budget is exceeded.
        
        Args:
            current_thought_length: Number of tokens in current thought chain.
            
        Returns:
            True if budget is exceeded (should force end of thought).
        """
        return current_thought_length >= self.max_thought_tokens

    def verify_thought_chain(self, tokens: List[int]) -> bool:
        """
        Verify structural integrity of a thought chain.
        
        Checks for proper nesting of <THOUGHT> and <END_THOUGHT>.
        
        Args:
            tokens: List of token IDs to verify.
            
        Returns:
            True if the chain is structurally valid (balanced tags).
        """
        if self.thought_start_token_id is None or self.thought_end_token_id is None:
            # If tokens aren't defined, we can't verify structure
            return True
            
        depth = 0
        for t in tokens:
            if t == self.thought_start_token_id:
                depth += 1
            elif t == self.thought_end_token_id:
                depth -= 1
                if depth < 0:
                    return False  # Closed without opening
                    
        return depth == 0  # Must be balanced

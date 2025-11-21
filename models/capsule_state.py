"""
Capsule state tracking for expansion management.
"""

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class CapsuleState:
    """
    Tracks which capsules have been expanded during reasoning.
    
    Attributes:
        sketches: [B, k, D] current sequence (may contain expanded children)
        children: [B, k, m, D] precomputed children embeddings
        checksums: [B, k, R] reconstructability signatures
        expanded_mask: [B, k] bool mask of expanded capsules
        expansion_positions: [B, k] indices where expansions occurred
        num_expansions: [B] count of expansions per sample
    """
    sketches: torch.Tensor  # Current input to TRM
    children: Optional[torch.Tensor] = None
    checksums: Optional[torch.Tensor] = None
    expanded_mask: Optional[torch.Tensor] = None
    expansion_positions: Optional[torch.Tensor] = None
    num_expansions: Optional[torch.Tensor] = None
    
    def __post_init__(self):
        if self.expanded_mask is None:
            B, k = self.sketches.shape[0], self.sketches.shape[1]
            device = self.sketches.device
            self.expanded_mask = torch.zeros(B, k, dtype=torch.bool, device=device)
            self.num_expansions = torch.zeros(B, dtype=torch.long, device=device)
    
    def expand_capsule(self, batch_idx: int, capsule_idx: int) -> bool:
        """
        Replace capsule sketch with its children embeddings.
        
        Args:
            batch_idx: Batch index
            capsule_idx: Capsule to expand
            
        Returns:
            True if expansion succeeded, False otherwise
        """
        if self.children is None:
            return False
        
        # Mark as expanded
        self.expanded_mask[batch_idx, capsule_idx] = True
        self.num_expansions[batch_idx] += 1
        
        # Get children embeddings [m, D]
        children_emb = self.children[batch_idx, capsule_idx]  # [m, D]
        
        # Replace sketch with children (splice into sequence)
        # Strategy: replace capsule_idx with first child, insert rest after
        current_seq = self.sketches[batch_idx]  # [k, D]
        
        # Insert children
        before = current_seq[:capsule_idx]  # [capsule_idx, D]
        after = current_seq[capsule_idx+1:]  # [k-capsule_idx-1, D]
        
        # New sequence: before + children + after
        new_seq = torch.cat([before, children_emb, after], dim=0)
        
        # Truncate or pad to maintain sequence length
        if new_seq.size(0) > current_seq.size(0):
            new_seq = new_seq[:current_seq.size(0)]
        elif new_seq.size(0) < current_seq.size(0):
            pad_len = current_seq.size(0) - new_seq.size(0)
            pad = torch.zeros(pad_len, new_seq.size(1), device=new_seq.device)
            new_seq = torch.cat([new_seq, pad], dim=0)
        
        self.sketches[batch_idx] = new_seq
        return True
    
    def get_expansion_cost(self, children_per_capsule: int = 4):
        """Compute total expansion cost for batch."""
        return self.num_expansions.float() * children_per_capsule * 0.01
    
    def get_reconstructability_bonus(self, threshold: float = 0.5):
        """Compute bonus based on checksum signals."""
        if self.checksums is None:
            return torch.zeros(self.sketches.size(0), device=self.sketches.device)
        
        checksum_norms = self.checksums.norm(dim=-1)  # [B, k]
        reconstructable = (checksum_norms > threshold).float()
        bonus = 0.1 * reconstructable.sum(dim=-1)
        return bonus

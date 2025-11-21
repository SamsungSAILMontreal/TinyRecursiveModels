"""
COCONUT-style Latent Planning Module
Implements continuous thought reasoning without language decoding
Based on Meta AI's COCONUT paper (Chain of Continuous Thought)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from models.bitnet import BitLinear


class LatentPlanningModule(nn.Module):
    """
    Explores multiple reasoning paths in latent space before committing to output.
    
    Key features:
    - Breadth-first exploration of reasoning paths
    - No language decoding (pure latent computation)
    - Path scoring using value estimation
    - Differentiable path selection
    
    Args:
        hidden_size: Dimension of hidden states
        num_paths: Number of parallel reasoning paths to explore (default: 4)
        planning_depth: How many planning steps to take (default: 2)
        use_path_dropout: Randomly drop paths during training for robustness
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_paths: int = 4,
        planning_depth: int = 2,
        use_path_dropout: bool = True,
        dropout_prob: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_paths = num_paths
        self.planning_depth = planning_depth
        self.use_path_dropout = use_path_dropout
        self.dropout_prob = dropout_prob
        
        # Path exploration: project hidden state to multiple paths
        self.path_projections = nn.ModuleList([
            nn.Sequential(
                BitLinear(hidden_size, hidden_size),
                nn.GELU(),
                BitLinear(hidden_size, hidden_size)
            ) for _ in range(num_paths)
        ])
        
        # Planning layers: refine each path
        self.planning_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ) for _ in range(planning_depth)
        ])
        
        # Path scoring: estimate value of each reasoning path
        self.path_scorer = nn.Sequential(
            BitLinear(hidden_size, hidden_size // 2),
            nn.GELU(),
            BitLinear(hidden_size // 2, 1)
        )
        
        # Path aggregation: combine selected paths
        self.aggregation = nn.Sequential(
            BitLinear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size)
        )
        
        print(f" Latent Planning Module:")
        print(f"   Paths: {num_paths}")
        print(f"   Planning depth: {planning_depth}")
        print(f"   Path dropout: {use_path_dropout}")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        return_path_scores: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Explore multiple reasoning paths and select the best.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            return_path_scores: If True, also return path scores for analysis
        
        Returns:
            refined_hidden_states: [batch, seq_len, hidden_size]
            path_scores (optional): [batch, num_paths] - scores for each path
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Step 1: Generate multiple reasoning paths
        # Each path is a different "way of thinking" about the problem
        paths = []
        for path_proj in self.path_projections:
            path = path_proj(hidden_states)  # [batch, seq_len, hidden]
            paths.append(path)
        
        # Step 2: Refine each path through planning layers
        # This simulates "thinking through" each approach
        refined_paths = []
        for path in paths:
            refined_path = path
            for planning_layer in self.planning_layers:
                refined_path = planning_layer(refined_path)
            
            # Optional: Path dropout during training
            if self.training and self.use_path_dropout:
                if torch.rand(1).item() < self.dropout_prob:
                    # Zero out this path (forces model to not rely on any single path)
                    refined_path = refined_path * 0
            
            refined_paths.append(refined_path)
        
        # Step 3: Score each path
        # Higher score = more promising reasoning direction
        path_scores = []
        for refined_path in refined_paths:
            # Pool across sequence for path-level scoring
            pooled = refined_path.mean(dim=1)  # [batch, hidden]
            score = self.path_scorer(pooled)  # [batch, 1]
            path_scores.append(score)
        
        path_scores = torch.cat(path_scores, dim=1)  # [batch, num_paths]
        path_weights = F.softmax(path_scores, dim=1)  # [batch, num_paths]
        
        # Step 4: Weighted combination of paths
        # Differentiable selection (not hard choice, so gradients flow to all paths)
        stacked_paths = torch.stack(refined_paths, dim=1)  # [batch, num_paths, seq_len, hidden]
        
        # Expand weights for broadcasting: [batch, num_paths, 1, 1]
        expanded_weights = path_weights.unsqueeze(-1).unsqueeze(-1)
        
        # Weighted sum: [batch, seq_len, hidden]
        combined = (stacked_paths * expanded_weights).sum(dim=1)
        
        # Step 5: Final aggregation with residual connection
        # Allows model to bypass planning if not helpful
        output = self.aggregation(combined) + hidden_states
        
        if return_path_scores:
            return output, path_scores
        return output
    
    def get_path_distribution(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Get the probability distribution over paths without modifying hidden states.
        Useful for analysis and visualization.
        
        Returns:
            path_probs: [batch, num_paths]
        """
        with torch.no_grad():
            _, path_scores = self.forward(hidden_states, return_path_scores=True)
            return F.softmax(path_scores, dim=1)


class AdaptiveLatentPlanning(nn.Module):
    """
    Extended version that can adaptively decide when to use latent planning.
    
    Uses a gating mechanism to determine if planning is needed for current input.
    For simple problems, bypasses planning (saves compute).
    For complex problems, activates full multi-path exploration.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_paths: int = 4,
        planning_depth: int = 2,
        use_adaptive_gate: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_adaptive_gate = use_adaptive_gate
        
        # Core planning module
        self.planning_module = LatentPlanningModule(
            hidden_size=hidden_size,
            num_paths=num_paths,
            planning_depth=planning_depth
        )
        
        # Adaptive gate: decides if planning is needed
        if use_adaptive_gate:
            self.complexity_gate = nn.Sequential(
                BitLinear(hidden_size, hidden_size // 4),
                nn.GELU(),
                BitLinear(hidden_size // 4, 1),
                nn.Sigmoid()
            )
            print(f"   Adaptive gating: enabled")
        else:
            self.complexity_gate = None
            print(f"   Adaptive gating: disabled (always plan)")
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Adaptively apply latent planning based on input complexity.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
        
        Returns:
            output: [batch, seq_len, hidden_size]
        """
        if not self.use_adaptive_gate:
            # Always plan
            return self.planning_module(hidden_states)
        
        # Estimate complexity
        pooled = hidden_states.mean(dim=1)  # [batch, hidden]
        gate_value = self.complexity_gate(pooled)  # [batch, 1]
        
        # Apply planning with gating
        planned = self.planning_module(hidden_states)
        
        # Interpolate: gate_value=1 => full planning, gate_value=0 => bypass
        gate_expanded = gate_value.unsqueeze(1)  # [batch, 1, 1]
        output = gate_expanded * planned + (1 - gate_expanded) * hidden_states
        
        return output

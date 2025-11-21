"""
Capsule-aware DQN with EXPAND action space.

Actions: {HALT, CONTINUE, EXPAND_0, EXPAND_1, ..., EXPAND_k}
Reward: task_gain - expansion_cost + reconstructability_bonus
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.bitnet import BitLinear


class CapsuleDQN(nn.Module):
    """
    DQN Q-head with capsule expansion and concept control.
    
    Action space:
    - HALT (0): Stop reasoning, emit <STOP>
    - CONTINUE (1): Continue recursive reasoning
    - EXPAND_i (2+i): Expand capsule i → children, emit <EXPAND>
    - EMIT_CONCEPT (2+k): Emit concept without expansion
    
    Integrates with HybridOutputHead control symbols.
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_capsules: int = 12,
        use_checksum: bool = True,
        checksum_dim: int = 32,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_capsules = num_capsules
        self.num_actions = 2 + num_capsules  # HALT, CONTINUE, EXPAND_0...EXPAND_{k-1}
        self.use_checksum = use_checksum
        
        # State encoder: TRM hidden + checksum signals
        input_dim = hidden_size
        if use_checksum:
            input_dim += checksum_dim * num_capsules
        
        # Q-network
        self.q_net = nn.Sequential(
            BitLinear(input_dim, 512),
            nn.ReLU(),
            BitLinear(512, 256),
            nn.ReLU(),
            BitLinear(256, self.num_actions)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.q_net:
            if isinstance(m, BitLinear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, trm_hidden, checksums=None):
        """
        Compute Q-values for all actions.
        
        Args:
            trm_hidden: [B, seq_len, hidden_size] TRM output
            checksums: [B, k, checksum_dim] reconstructability signals
        
        Returns:
            q_values: [B, num_actions] Q(s, a)
        """
        # Pool TRM hidden states
        state = trm_hidden.mean(dim=1)  # [B, hidden_size]
        
        # Concatenate checksum signals
        if self.use_checksum and checksums is not None:
            # Flatten checksums [B, k*checksum_dim]
            checksums_flat = checksums.view(checksums.size(0), -1)
            state = torch.cat([state, checksums_flat], dim=-1)
        
        # Q-values
        q_values = self.q_net(state)  # [B, num_actions]
        
        return q_values
    
    def select_action(self, q_values, epsilon=0.1, mask_expanded=None):
        """
        Epsilon-greedy action selection with expansion masking.
        
        Args:
            q_values: [B, num_actions]
            epsilon: Exploration probability
            mask_expanded: [B, k] bool mask of already-expanded capsules
        
        Returns:
            actions: [B] selected action indices
        """
        batch_size = q_values.size(0)
        device = q_values.device
        
        # Mask already-expanded capsules
        if mask_expanded is not None:
            # Set Q(EXPAND_i) = -inf for expanded capsules
            for i in range(self.num_capsules):
                expand_action_idx = 2 + i
                q_values[:, expand_action_idx] = torch.where(
                    mask_expanded[:, i],
                    torch.full_like(q_values[:, expand_action_idx], -1e9),
                    q_values[:, expand_action_idx]
                )
        
        # Epsilon-greedy
        random_mask = torch.rand(batch_size, device=device) < epsilon
        
        # Random actions
        random_actions = torch.randint(0, self.num_actions, (batch_size,), device=device)
        
        # Greedy actions
        greedy_actions = q_values.argmax(dim=-1)
        
        # Combine
        actions = torch.where(random_mask, random_actions, greedy_actions)
        
        return actions
    
    def decode_action(self, action_idx):
        """
        Decode action index to (action_type, target).
        
        Returns:
            ('halt', None) - emit <STOP>
            ('continue', None) - keep reasoning
            ('expand', capsule_idx) - expand capsule, emit <EXPAND>
            ('emit', None) - emit concept directly
        """
        if action_idx == 0:
            return ('halt', None)
        elif action_idx == 1:
            return ('continue', None)
        elif action_idx < 2 + self.num_capsules:
            return ('expand', action_idx - 2)
        else:
            return ('emit', None)
    
    def compute_expansion_cost(self, num_expansions, children_per_capsule=4):
        """Compute penalty for expanding capsules."""
        return 0.01 * num_expansions * children_per_capsule
    
    def compute_reconstructability_bonus(self, checksums, threshold=0.5):
        """
        Reward based on checksum signals.
        
        High checksum norm → high reconstructability → bonus for NOT expanding
        """
        checksum_norms = checksums.norm(dim=-1)  # [B, k]
        reconstructable_mask = checksum_norms > threshold
        bonus = 0.1 * reconstructable_mask.sum(dim=-1).float()
        return bonus


def compute_capsule_reward(
    prev_accuracy,
    curr_accuracy,
    num_expansions,
    checksums,
    is_terminal,
    seq_is_correct,
    children_per_capsule=4,
    num_concept_emissions=0,
):
    """
    HESC reward function with concept emission.
    
    Reward = Δ(task_score) - α·expansion_cost + β·concept_bonus + γ·reconstructability
    
    Encourages:
    - Task improvement (main signal)
    - Concept emission over expansion (efficiency)
    - High reconstructability (quality)
    """
    # Task improvement
    task_gain = 10.0 * (curr_accuracy - prev_accuracy)
    
    # Expansion penalty (expensive operation)
    expansion_cost = 0.01 * num_expansions * children_per_capsule
    
    # Concept emission bonus (cheap, efficient)
    concept_bonus = 0.005 * num_concept_emissions
    
    # Reconstructability bonus (checksum-based)
    checksum_norms = checksums.norm(dim=-1).mean(dim=-1)  # [B]
    reconstructable_bonus = 0.1 * torch.clamp(checksum_norms, 0, 1)
    
    # Terminal reward
    terminal_reward = torch.where(
        is_terminal,
        torch.where(seq_is_correct, 
                   torch.full_like(task_gain, 1.0),
                   torch.full_like(task_gain, -0.5)),
        torch.zeros_like(task_gain)
    )
    
    total_reward = task_gain - expansion_cost + concept_bonus + reconstructable_bonus + terminal_reward
    
    return total_reward

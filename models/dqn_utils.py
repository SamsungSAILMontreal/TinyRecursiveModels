"""
Utilities for DQN training: epsilon scheduling, running statistics, reward normalization.
"""

import torch
from utils.annealing import compute_q_temperature  # Import instead of duplicate
import math
from torch import Tensor
from typing import Optional


class RunningStats:
    """
    Track running mean and standard deviation for normalization.
    Uses Welford's online algorithm for numerical stability.
    """
    
    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize running statistics tracker.
        
        Args:
            epsilon: Small constant for numerical stability
        """
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.epsilon = epsilon
    
    def update(self, x: Tensor):
        """
        Update running statistics with new batch.
        
        Args:
            x: New values to incorporate [batch_size] or scalar
        """
        if isinstance(x, Tensor):
            batch_mean = x.mean().item()
            batch_var = x.var().item()
            batch_count = x.numel()
        else:
            batch_mean = float(x)
            batch_var = 0.0
            batch_count = 1
        
        # Welford's online algorithm
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / (self.count + batch_count)
        
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / (self.count + batch_count)
        self.var = M2 / (self.count + batch_count) if (self.count + batch_count) > 0 else 1.0
        self.count += batch_count
    
    def normalize(self, x: Tensor) -> Tensor:
        """
        Normalize tensor using running statistics.
        
        Args:
            x: Tensor to normalize
        
        Returns:
            Normalized tensor
        """
        return (x - self.mean) / (torch.sqrt(torch.tensor(self.var)) + self.epsilon)
    
    def denormalize(self, x: Tensor) -> Tensor:
        """
        Reverse normalization.
        
        Args:
            x: Normalized tensor
        
        Returns:
            Original scale tensor
        """
        return x * (torch.sqrt(torch.tensor(self.var)) + self.epsilon) + self.mean
    
    def reset(self):
        """Reset statistics."""
        self.mean = 0.0
        self.var = 1.0
        self.count = 0


def compute_epsilon(
    step: int,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_steps: int = 100000,
) -> float:
    """
    Compute epsilon for epsilon-greedy exploration with linear decay.
    
    Args:
        step: Current training step
        epsilon_start: Initial exploration rate
        epsilon_end: Final exploration rate
        epsilon_decay_steps: Number of steps to decay over
    
    Returns:
        Current epsilon value
    """
    if step >= epsilon_decay_steps:
        return epsilon_end
    
    progress = step / epsilon_decay_steps
    return epsilon_start + (epsilon_end - epsilon_start) * progress


# compute_q_temperature is now imported from utils.annealing


def select_action_epsilon_greedy(
    q_halt_logits: Tensor,
    epsilon: float,
    training: bool = True,
) -> Tensor:
    """
    DEPRECATED: Legacy 2-action epsilon-greedy.
    Use select_action_epsilon_greedy_3way for 3-action Q-head.
    
    Select action using epsilon-greedy strategy.
    
    Args:
        q_halt_logits: Q-values for halt action [batch_size]
        epsilon: Exploration rate (0 = greedy, 1 = random)
        training: Whether in training mode
    
    Returns:
        Boolean tensor [batch_size] indicating halt decision
    """
    if not training:
        # Evaluation: always greedy
        return q_halt_logits > 0
    
    batch_size = q_halt_logits.shape[0]
    device = q_halt_logits.device
    
    # Random mask: True = explore, False = exploit
    random_mask = torch.rand(batch_size, device=device) < epsilon
    
    # Random action: 50/50 halt/continue
    random_action = torch.rand(batch_size, device=device) < 0.5
    
    # Greedy action: halt if Q(halt) > 0
    greedy_action = q_halt_logits > 0
    
    # Combine: use random action where mask is True, greedy otherwise
    action = torch.where(random_mask, random_action, greedy_action)
    
    return action


def select_action_epsilon_greedy_3way(
    q_logits: Tensor,
    epsilon: float,
    training: bool = True,
) -> Tensor:
    """
    Select action using epsilon-greedy strategy for 3-action Q-head.
    
    Args:
        q_logits: Q-values [batch_size, 3] for [CONTINUE, HALT, EXPAND]
        epsilon: Exploration rate (0 = greedy, 1 = random)
        training: Whether in training mode
    
    Returns:
        Integer tensor [batch_size] with action indices {0, 1, 2}
    """
    if not training:
        # Evaluation: always greedy
        return torch.argmax(q_logits, dim=-1)
    
    batch_size = q_logits.shape[0]
    device = q_logits.device
    
    # Random mask: True = explore, False = exploit
    random_mask = torch.rand(batch_size, device=device) < epsilon
    
    # Random action: uniform over {0, 1, 2}
    random_action = torch.randint(0, 3, (batch_size,), device=device)
    
    # Greedy action: argmax of Q-values
    greedy_action = torch.argmax(q_logits, dim=-1)
    
    # Combine: use random action where mask is True, greedy otherwise
    action = torch.where(random_mask, random_action, greedy_action)
    
    return action


# Note: Reward shaping is now handled in losses.py ACTLossHead._store_transitions()
# This consolidates all reward logic in one place for semantic compression


def soft_update_target_network(
    online_params,
    target_params,
    tau: float = 0.005,
):
    """
    Soft update target network parameters (Polyak averaging).
    
    θ_target = τ * θ_online + (1 - τ) * θ_target
    
    Args:
        online_params: Parameters of online network
        target_params: Parameters of target network
        tau: Interpolation parameter (0 = no update, 1 = hard update)
    """
    with torch.no_grad():
        for target_param, online_param in zip(target_params, online_params):
            target_param.data.copy_(
                tau * online_param.data + (1.0 - tau) * target_param.data
            )


def hard_update_target_network(online_params, target_params):
    """
    Hard update target network (copy online network exactly).
    
    Args:
        online_params: Parameters of online network
        target_params: Parameters of target network
    """
    with torch.no_grad():
        for target_param, online_param in zip(target_params, online_params):
            target_param.data.copy_(online_param.data)

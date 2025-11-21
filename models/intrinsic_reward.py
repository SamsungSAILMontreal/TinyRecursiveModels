"""
Intrinsic motivation mechanisms for improved exploration and learning.
Implements curiosity-driven rewards, count-based bonuses, and prediction error rewards.
"""

import torch
from torch import nn, Tensor
from typing import Dict, Optional
from models.bitnet import BitLinear
import numpy as np


class CountBasedCuriosity:
    """
    Count-based exploration bonus using hash-based state counting.
    Rewards visiting novel states with: r_intrinsic = β / √(N(s))
    """
    
    def __init__(self, beta: float = 0.1, hash_dim: int = 32):
        """
        Args:
            beta: Bonus scaling factor
            hash_dim: Dimensionality of hash representation
        """
        self.beta = beta
        self.hash_dim = hash_dim
        self.state_counts = {}  # Hash -> count
        self.total_visits = 0
        
    def compute_hash(self, state: Tensor) -> str:
        """
        Compute locality-sensitive hash of state.
        
        Args:
            state: [batch, hidden_size]
        Returns:
            Hash string for each state in batch
        """
        # Simple random projection hash
        if not hasattr(self, 'projection'):
            self.projection = torch.randn(state.shape[-1], self.hash_dim, device=state.device) * 0.1
        
        # Project and binarize
        projected = torch.matmul(state, self.projection)
        binary_hash = (projected > 0).long()
        
        # Convert to string
        hashes = []
        for i in range(state.shape[0]):
            hash_str = ''.join(binary_hash[i].cpu().numpy().astype(str))
            hashes.append(hash_str)
        
        return hashes
    
    def compute_bonus(self, state: Tensor) -> Tensor:
        """
        Compute exploration bonus for states.
        
        Args:
            state: [batch, hidden_size]
        Returns:
            bonus: [batch] - intrinsic reward
        """
        hashes = self.compute_hash(state)
        bonuses = []
        
        for hash_str in hashes:
            count = self.state_counts.get(hash_str, 0)
            bonus = self.beta / (np.sqrt(count + 1) + 1e-8)
            bonuses.append(bonus)
            
            # Update counts
            self.state_counts[hash_str] = count + 1
            self.total_visits += 1
        
        return torch.tensor(bonuses, device=state.device, dtype=torch.float32)
    
    def reset(self):
        """Reset all counts."""
        self.state_counts.clear()
        self.total_visits = 0


class RandomNetworkDistillation(nn.Module):
    """
    Random Network Distillation (RND) for intrinsic curiosity.
    Rewards prediction error: r_intrinsic = ||f_target(s) - f_predictor(s)||²
    
    Reference: Burda et al. 2019 (Exploration by Random Network Distillation)
    """
    
    def __init__(self, hidden_size: int, rnd_hidden: int = 128):
        super().__init__()
        
        # Random target network (frozen)
        self.target_net = nn.Sequential(
            BitLinear(hidden_size, rnd_hidden),
            nn.ReLU(),
            BitLinear(rnd_hidden, rnd_hidden),
            nn.ReLU(),
            BitLinear(rnd_hidden, rnd_hidden)
        )
        
        # Predictor network (learned)
        self.predictor_net = nn.Sequential(
            BitLinear(hidden_size, rnd_hidden),
            nn.ReLU(),
            BitLinear(rnd_hidden, rnd_hidden),
            nn.ReLU(),
            BitLinear(rnd_hidden, rnd_hidden)
        )
        
        # Freeze target network
        for param in self.target_net.parameters():
            param.requires_grad = False
        
        # Running stats for normalization
        self.register_buffer('target_mean', torch.zeros(rnd_hidden))
        self.register_buffer('target_std', torch.ones(rnd_hidden))
        self.update_count = 0
    
    def forward(self, state: Tensor) -> Tensor:
        """
        Compute intrinsic reward (prediction error).
        
        Args:
            state: [batch, hidden_size]
        Returns:
            intrinsic_reward: [batch]
        """
        with torch.no_grad():
            target = self.target_net(state)
        
        prediction = self.predictor_net(state)
        
        # Compute prediction error (MSE)
        error = (target - prediction).pow(2).mean(dim=-1)
        
        return error
    
    def update_predictor(self, state: Tensor, optimizer: torch.optim.Optimizer) -> float:
        """
        Update predictor network to match target.
        
        Args:
            state: [batch, hidden_size]
            optimizer: Optimizer for predictor network
        Returns:
            loss: Prediction loss
        """
        with torch.no_grad():
            target = self.target_net(state)
            
            # Update running statistics
            self.update_count += 1
            alpha = min(1.0, 1.0 / self.update_count)
            self.target_mean = (1 - alpha) * self.target_mean + alpha * target.mean(0)
            self.target_std = (1 - alpha) * self.target_std + alpha * target.std(0)
        
        prediction = self.predictor_net(state)
        
        # Normalized MSE loss
        loss = ((target - prediction) / (self.target_std + 1e-8)).pow(2).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()


class ForwardDynamicsModel(nn.Module):
    """
    Forward dynamics model for prediction-based curiosity.
    Predicts next state: s_{t+1} = f(s_t, a_t)
    Intrinsic reward: r_intrinsic = ||s_{t+1} - f(s_t, a_t)||²
    
    Reference: Pathak et al. 2017 (Curiosity-driven Exploration)
    """
    
    def __init__(self, hidden_size: int, num_actions: int = 2, dynamics_hidden: int = 256):
        super().__init__()
        
        # Forward dynamics model
        self.dynamics_net = nn.Sequential(
            BitLinear(hidden_size + num_actions, dynamics_hidden),
            nn.ReLU(),
            BitLinear(dynamics_hidden, dynamics_hidden),
            nn.ReLU(),
            BitLinear(dynamics_hidden, hidden_size)
        )
    
    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        """
        Predict next state.
        
        Args:
            state: [batch, hidden_size]
            action: [batch, num_actions] (one-hot)
        Returns:
            predicted_next_state: [batch, hidden_size]
        """
        # Concatenate state and action
        state_action = torch.cat([state, action], dim=-1)
        return self.dynamics_net(state_action)
    
    def compute_curiosity_reward(self, state: Tensor, action: Tensor, next_state: Tensor) -> Tensor:
        """
        Compute curiosity reward as prediction error.
        
        Args:
            state: [batch, hidden_size]
            action: [batch, num_actions]
            next_state: [batch, hidden_size]
        Returns:
            curiosity_reward: [batch]
        """
        predicted_next = self.forward(state, action)
        error = (next_state - predicted_next).pow(2).mean(dim=-1)
        return error
    
    def update(self, state: Tensor, action: Tensor, next_state: Tensor, 
               optimizer: torch.optim.Optimizer) -> float:
        """
        Update forward dynamics model.
        
        Args:
            state: [batch, hidden_size]
            action: [batch, num_actions]
            next_state: [batch, hidden_size]
            optimizer: Optimizer for dynamics model
        Returns:
            loss: Dynamics prediction loss
        """
        predicted_next = self.forward(state, action)
        loss = nn.functional.mse_loss(predicted_next, next_state.detach())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()


class IntrinsicRewardModule:
    """
    Combined intrinsic reward module with multiple curiosity mechanisms.
    Computes: r_intrinsic = α₁ * r_count + α₂ * r_rnd + α₃ * r_dynamics
    """
    
    def __init__(self, 
                 hidden_size: int,
                 enable_count: bool = True,
                 enable_rnd: bool = True,
                 enable_dynamics: bool = False,
                 count_beta: float = 0.05,
                 rnd_weight: float = 0.1,
                 dynamics_weight: float = 0.05):
        """
        Args:
            hidden_size: Dimension of state representations
            enable_count: Enable count-based curiosity
            enable_rnd: Enable Random Network Distillation
            enable_dynamics: Enable forward dynamics curiosity
            count_beta: Scaling factor for count-based bonus
            rnd_weight: Weight for RND reward
            dynamics_weight: Weight for dynamics reward
        """
        self.hidden_size = hidden_size
        self.count_beta = count_beta
        self.rnd_weight = rnd_weight
        self.dynamics_weight = dynamics_weight
        
        # Initialize curiosity mechanisms
        self.count_curiosity = CountBasedCuriosity(beta=count_beta) if enable_count else None
        self.rnd = RandomNetworkDistillation(hidden_size) if enable_rnd else None
        self.dynamics = ForwardDynamicsModel(hidden_size) if enable_dynamics else None
        
    def compute_intrinsic_reward(self, 
                                  state: Tensor,
                                  action: Optional[Tensor] = None,
                                  next_state: Optional[Tensor] = None) -> Tensor:
        """
        Compute combined intrinsic reward.
        
        Args:
            state: [batch, hidden_size]
            action: [batch] or [batch, num_actions] (optional)
            next_state: [batch, hidden_size] (optional)
        Returns:
            intrinsic_reward: [batch]
        """
        total_reward = torch.zeros(state.shape[0], device=state.device)
        
        # Count-based curiosity
        if self.count_curiosity is not None:
            count_bonus = self.count_curiosity.compute_bonus(state)
            total_reward += count_bonus
        
        # RND curiosity
        if self.rnd is not None:
            rnd_bonus = self.rnd(state) * self.rnd_weight
            total_reward += rnd_bonus
        
        # Forward dynamics curiosity
        if self.dynamics is not None and action is not None and next_state is not None:
            # Convert action to one-hot if needed
            if action.dim() == 1:
                action_onehot = torch.nn.functional.one_hot(action.long(), num_classes=2).float()
            else:
                action_onehot = action
            
            dynamics_bonus = self.dynamics.compute_curiosity_reward(state, action_onehot, next_state) * self.dynamics_weight
            total_reward += dynamics_bonus
        
        return total_reward
    
    def get_update_params(self):
        """Get parameters that need to be updated."""
        params = []
        if self.rnd is not None:
            params.extend(self.rnd.predictor_net.parameters())
        if self.dynamics is not None:
            params.extend(self.dynamics.parameters())
        return params

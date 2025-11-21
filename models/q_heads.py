"""
Q-Head architectures for DQN halting control.
Supports: MLP (default), RNN (temporal), Mini-Attention (context-aware)
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from models.layers import CastedLinear, rms_norm


class MLPQHead(nn.Module):
    """Simple MLP Q-head (baseline, fast)."""
    
    def __init__(self, hidden_size: int, num_actions: int = 3):
        super().__init__()
        self.q_head = CastedLinear(hidden_size, num_actions, bias=True)
        
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore
    
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_state: [batch, hidden_size] or [batch, seq_len, hidden_size]
        Returns:
            q_values: [batch, num_actions]
        """
        if hidden_state.dim() == 3:
            hidden_state = hidden_state[:, 0]  # Use first token (puzzle embedding)
        return self.q_head(hidden_state)


class RNNQHead(nn.Module):
    """RNN Q-head for temporal modeling of recursive states."""
    
    def __init__(self, hidden_size: int, num_actions: int = 2, 
                 rnn_hidden_size: int = 128, num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_hidden_size = rnn_hidden_size
        self.num_layers = num_layers
        
        # Project input to RNN hidden size
        self.input_proj = CastedLinear(hidden_size, rnn_hidden_size, bias=False)
        
        # GRU for temporal modeling
        self.rnn = nn.GRU(
            input_size=rnn_hidden_size,
            hidden_size=rnn_hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = CastedLinear(rnn_hidden_size, num_actions, bias=True)
        
        # Hidden state buffer
        self.register_buffer('h_prev', torch.zeros(num_layers, 1, rnn_hidden_size))
        
        # Init output to (almost) zero
        with torch.no_grad():
            self.output_proj.weight.zero_()
            self.output_proj.bias.fill_(-5)  # type: ignore
    
    def reset_hidden(self, batch_size: int, mask: Optional[torch.Tensor] = None):
        """Reset RNN hidden state.
        
        Args:
            batch_size: Size of batch
            mask: Optional boolean mask [batch_size]. If provided, only reset positions where mask=True
        """
        device = self.h_prev.device
        
        if mask is None:
            # Full reset
            self.h_prev = torch.zeros(self.num_layers, batch_size, self.rnn_hidden_size, device=device)
        else:
            # Selective reset for halted sequences
            if self.h_prev.shape[1] != batch_size:
                # Batch size changed, do full reset
                self.h_prev = torch.zeros(self.num_layers, batch_size, self.rnn_hidden_size, device=device)
            else:
                # Reset only masked positions
                reset_mask = mask.view(1, -1, 1).expand(self.num_layers, -1, self.rnn_hidden_size)
                self.h_prev = torch.where(reset_mask, torch.zeros_like(self.h_prev), self.h_prev)
    
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_state: [batch, hidden_size] - current reasoning state
        Returns:
            q_values: [batch, num_actions]
        """
        if hidden_state.dim() == 3:
            hidden_state = hidden_state[:, 0]  # Use first token
        
        batch_size = hidden_state.shape[0]
        
        # Ensure h_prev has correct batch size
        if self.h_prev.shape[1] != batch_size:
            self.reset_hidden(batch_size)
        
        # Project to RNN space
        x = self.input_proj(hidden_state).unsqueeze(1)  # [batch, 1, rnn_hidden]
        
        # Run RNN (maintains temporal context across recursive steps)
        output, h_new = self.rnn(x, self.h_prev)  # output: [batch, 1, rnn_hidden]
        
        # Update hidden state
        self.h_prev = h_new.detach()
        
        # Project to Q-values
        q_values = self.output_proj(output.squeeze(1))  # [batch, num_actions]
        
        return q_values


class MiniAttentionQHead(nn.Module):
    """Mini-attention Q-head for context-aware halting decisions."""
    
    def __init__(self, hidden_size: int, num_actions: int = 2,
                 num_heads: int = 4, context_window: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.context_window = context_window
        self.head_dim = hidden_size // num_heads
        
        # Q, K, V projections
        self.qkv_proj = CastedLinear(hidden_size, 3 * hidden_size, bias=False)
        
        # Output projection
        self.output_proj = CastedLinear(hidden_size, num_actions, bias=True)
        
        # Context buffer (stores past reasoning states)
        self.register_buffer('context_buffer', torch.zeros(1, context_window, hidden_size))
        self.register_buffer('context_ptr', torch.tensor(0, dtype=torch.long))
        
        # Init output to (almost) zero
        with torch.no_grad():
            self.output_proj.weight.zero_()
            self.output_proj.bias.fill_(-5)  # type: ignore
    
    def reset_context(self, batch_size: int):
        """Reset context buffer."""
        device = self.context_buffer.device
        self.context_buffer = torch.zeros(batch_size, self.context_window, self.hidden_size, device=device)
        self.context_ptr.fill_(0)
    
    def reset_hidden(self, batch_size: int, mask: Optional[torch.Tensor] = None):
        """Reset context buffer for specific batch indices.
        
        Args:
            batch_size: Size of batch
            mask: Optional boolean mask [batch_size]. If provided, only reset positions where mask=True
        """
        device = self.context_buffer.device
        
        if mask is None:
            # Full reset
            self.context_buffer = torch.zeros(batch_size, self.context_window, self.hidden_size, device=device)
            self.context_ptr.fill_(0)
        else:
            # Selective reset for halted sequences
            if self.context_buffer.shape[0] != batch_size:
                # Batch size changed, do full reset
                self.context_buffer = torch.zeros(batch_size, self.context_window, self.hidden_size, device=device)
                self.context_ptr.fill_(0)
            else:
                # Reset only masked positions (all timesteps for those sequences)
                reset_mask = mask.view(-1, 1, 1).expand(-1, self.context_window, self.hidden_size)
                self.context_buffer = torch.where(reset_mask, torch.zeros_like(self.context_buffer), self.context_buffer)
                # Note: context_ptr is shared across batch, so we don't reset it selectively
    
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_state: [batch, hidden_size] or [batch, seq_len, hidden_size]
        Returns:
            q_values: [batch, num_actions]
        """
        if hidden_state.dim() == 3:
            hidden_state = hidden_state[:, 0]  # Use first token
        
        batch_size = hidden_state.shape[0]
        
        # Ensure context buffer has correct batch size
        if self.context_buffer.shape[0] != batch_size:
            self.reset_context(batch_size)
        
        # Store current state in context buffer (circular)
        ptr = self.context_ptr.item() % self.context_window
        self.context_buffer[:, ptr] = hidden_state.detach()
        self.context_ptr += 1
        
        # QKV projection
        qkv = self.qkv_proj(hidden_state.unsqueeze(1))  # [batch, 1, 3*hidden]
        qkv = qkv.reshape(batch_size, 1, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # Each: [batch, 1, num_heads, head_dim]
        
        # Get keys and values from context
        context_qkv = self.qkv_proj(self.context_buffer)  # [batch, window, 3*hidden]
        context_qkv = context_qkv.reshape(batch_size, self.context_window, 3, self.num_heads, self.head_dim)
        _, context_k, context_v = context_qkv.unbind(2)
        
        # Concatenate current with context
        k = torch.cat([k, context_k], dim=1)  # [batch, 1+window, num_heads, head_dim]
        v = torch.cat([v, context_v], dim=1)
        
        # Scaled dot-product attention
        q = q.transpose(1, 2)  # [batch, num_heads, 1, head_dim]
        k = k.transpose(1, 2)  # [batch, num_heads, 1+window, head_dim]
        v = v.transpose(1, 2)
        
        attn_output = F.scaled_dot_product_attention(q, k, v)  # [batch, num_heads, 1, head_dim]
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, 1, self.hidden_size)
        
        # Project to Q-values
        q_values = self.output_proj(attn_output.squeeze(1))  # [batch, num_actions]
        
        return q_values



class ConfidenceQHead(nn.Module):
    """
    Tiny MLP Q-head for Confidence Management.
    
    Inputs: [Hidden_State (N), Stability_Delta (1), Loop_Count (1)]
    Actions: LOOP (0), COMMIT (1), REJECT (2)
    """
    
    def __init__(self, hidden_size: int, num_actions: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        # Input dim = hidden_size + stability (1) + loop_count (1)
        input_dim = hidden_size + 2
        
        # Tiny MLP: Input -> 64 -> ReLU -> Actions
        self.net = nn.Sequential(
            CastedLinear(input_dim, 64, bias=True),
            nn.ReLU(),
            CastedLinear(64, num_actions, bias=True)
        )
        
        # Init to near-zero to prevent random actions at start
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.fill_(-0.1) # Slight negative bias
            
    def forward(self, hidden_state: torch.Tensor, stability: torch.Tensor, loop_count: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_state: [batch, hidden_size]
            stability: [batch, 1] (Cosine similarity delta)
            loop_count: [batch, 1] (Normalized loop count 0.0-1.0)
        Returns:
            q_values: [batch, num_actions]
        """
        if hidden_state.dim() == 3:
            hidden_state = hidden_state[:, 0]
            
        # Concatenate inputs
        x = torch.cat([hidden_state, stability, loop_count], dim=-1)
        return self.net(x)


def create_q_head(config) -> nn.Module:
    """Factory function to create Q-head based on config."""
    q_head_type = getattr(config, 'q_head_type', 'mlp')
    hidden_size = config.hidden_size
    num_actions = getattr(config, 'q_head_num_actions', 3)  
    
    if q_head_type == 'mlp':
        return MLPQHead(hidden_size, num_actions)
    elif q_head_type == 'rnn':
        rnn_hidden = getattr(config, 'q_head_hidden_size', 128)
        num_layers = getattr(config, 'q_head_num_layers', 1)
        return RNNQHead(hidden_size, num_actions, rnn_hidden, num_layers)
    elif q_head_type == 'mini_attention':
        num_heads = getattr(config, 'q_head_num_layers', 4)  # Reuse for num_heads
        context_window = getattr(config, 'q_head_hidden_size', 128) // 16  # 8 for default
        return MiniAttentionQHead(hidden_size, num_actions, num_heads, context_window)
    elif q_head_type == 'confidence':
        return ConfidenceQHead(hidden_size, num_actions)
    else:
        raise ValueError(f"Unknown q_head_type: {q_head_type}")

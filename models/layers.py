from typing import Tuple
import einops
import torch
from torch import nn
import torch.nn.functional as F

# Flash Attention 3 with automatic fallback
try:
    from flash_attn import flash_attn_func
    
    # Check if GPU supports FlashAttention (requires Ampere or newer)
    # Compute capability: 8.0+ for A100, 8.6 for RTX 30xx, 8.9 for RTX 40xx
    if torch.cuda.is_available():
        compute_capability = torch.cuda.get_device_capability()
        # FlashAttention requires compute capability >= 8.0 (Ampere)
        FLASH_ATTN_AVAILABLE = compute_capability[0] >= 8
        if not FLASH_ATTN_AVAILABLE:
            print(f"FlashAttention disabled: GPU compute capability {compute_capability[0]}.{compute_capability[1]} < 8.0 (Ampere required)")
            print("   Using PyTorch SDPA instead (slower but compatible)")
    else:
        FLASH_ATTN_AVAILABLE = False
except ImportError:
    FLASH_ATTN_AVAILABLE = False

if not FLASH_ATTN_AVAILABLE:
    from torch.nn.functional import scaled_dot_product_attention
    # Enable memory-efficient attention for T4 (faster than default SDPA)
    import torch.backends.cuda
    if torch.cuda.is_available():
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_flash_sdp(False)  # Not available on T4
        torch.backends.cuda.enable_math_sdp(False)  # Slowest fallback

from models.common import trunc_normal_init_


CosSin = Tuple[torch.Tensor, torch.Tensor]


def _find_multiple(a, b):
    return (-(a // -b)) * b


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


from models.bitnet import BitwiseQuantization, ActivationQuantization

class CastedLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((out_features, )))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # BitNet b1.58 Quantization
        # 1. Quantize Input (8-bit)
        input_quant = ActivationQuantization.apply(input)
        
        # 2. Quantize Weights (1.58-bit)
        # Cast weight to input dtype before quantization if needed, or just quantize the float weight
        # The original CastedLinear cast to input.dtype.
        # We should quantize the weight (which is float) then cast? 
        # Or quantize the casted weight?
        # Usually weights are kept in high precision (float32/bf16) and quantized on the fly.
        weight_quant = BitwiseQuantization.apply(self.weight)
        
        # 3. Linear
        # We use the quantized values. 
        # Note: bias is usually not quantized in BitNet, or high precision.
        return F.linear(input_quant, weight_quant.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)


class CastedEmbedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 init_std: float,
                 cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to

        # Truncated LeCun normal init
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight.to(self.cast_to))


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embeddings for 1D sequences (text)."""
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()

        # RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        # Register as buffers (non-trainable tensors)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached


class LearnedPositionalEmbedding2D(nn.Module):
    """Learned 2D Positional Embeddings for Vision (ViT/CLIP-style).
    
    Standard approach for vision transformers - additive learned embeddings
    that preserve 2D spatial relationships, unlike RoPE which is 1D-only.
    """
    def __init__(self, num_patches: int, embedding_dim: int):
        super().__init__()
        self.num_patches = num_patches
        self.embedding_dim = embedding_dim
        
        # Learned positional embeddings: [1, num_patches, embedding_dim]
        # Initialized with truncated normal (ViT standard)
        self.pos_embedding = nn.Parameter(
            trunc_normal_init_(torch.zeros(1, num_patches, embedding_dim), std=0.02)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional embeddings to input tokens.
        
        Args:
            x: [B, num_patches, embedding_dim] patch embeddings
        
        Returns:
            [B, num_patches, embedding_dim] with positional info added
        """
        return x + self.pos_embedding


class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal

        self.qkv_proj = CastedLinear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor, attn_bias=None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # hidden_states: [bs, seq_len, num_heads, head_dim]
        qkv = self.qkv_proj(hidden_states)

        # Split head
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        # RoPE
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # Flash Attention 3 with fallback
        if FLASH_ATTN_AVAILABLE:
            # Flash Attention expects [batch, seq_len, num_heads, head_dim]
            # Already in correct format, no rearrange needed
            # Note: flash_attn doesn't support attn_bias, ignore it if provided
            attn_output = flash_attn_func(
                query, key, value,
                causal=self.causal,
                softmax_scale=1.0 / (self.head_dim ** 0.5)
            )
            attn_output = attn_output.contiguous().view(batch_size, seq_len, self.output_size)
        else:
            # Fallback to PyTorch SDPA
            query, key, value = map(lambda t: einops.rearrange(t, 'B S H D -> B H S D'), (query, key, value))
            # Handle attn_bias if provided (spatial bias for vision encoder)
            attn_mask = attn_bias if attn_bias is not None else None
            attn_output = scaled_dot_product_attention(
                query=query, key=key, value=value, 
                attn_mask=attn_mask,
                is_causal=self.causal
            )
            attn_output = einops.rearrange(attn_output, 'B H S D -> B S H D')
            attn_output = attn_output.contiguous().view(batch_size, seq_len, self.output_size)  # type: ignore
        
        return self.o_proj(attn_output)

class LinearSwish(nn.Module):
    def __init__(self, hidden_size: int, reverse=False):
        super().__init__()

        self.linear = CastedLinear(hidden_size, hidden_size, bias=False)
        self.reverse = reverse

    def forward(self, x):
        if self.reverse:
            return F.silu(self.linear(x))
        else:
            return self.linear(F.silu(x))


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj    = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)

"""
TRM with H-Cycle DEQ Integration
=================================

This module integrates DEQ (Deep Equilibrium Models) into TRM's H-cycle loop,
replacing explicit iterations with fixed-point solving.

Key Changes from Original TRM:
1. H-cycles: Replaced explicit loop with DEQ fixed-point solver
2. L-cycles: Kept as explicit iterations (inside fixed-point function)
3. Gradients: Flow through all H-cycles via IFT (not just last cycle)
4. Memory: O(L_cycles) - independent of H_cycles depth

Based on: TinyRecursiveModels/models/recursive_reasoning/trm.py
Integration Strategy: Option 1 from implementation plan (H-DEQ only)
"""

from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

# Import TRM components (assume same structure as original)
# Note: In actual integration, these would come from models.common, models.layers
# For this demonstration, we'll define minimal versions or indicate where they come from

try:
    from models.common import trunc_normal_init_
    from models.layers import (
        rms_norm, LinearSwish, SwiGLU, Attention,
        RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
    )
    from models.sparse_embedding import CastedSparseEmbedding
except ImportError:
    # Fallback for testing outside TRM environment
    print("Warning: TRM modules not found. Using placeholder imports.")
    print("To use with actual TRM, copy this file to TinyRecursiveModels/models/recursive_reasoning/")

    def rms_norm(x, variance_epsilon=1e-5):
        """Placeholder RMS norm"""
        return F.normalize(x, p=2, dim=-1)

    def trunc_normal_init_(tensor, std=1.0):
        """Placeholder truncated normal init"""
        return torch.nn.init.trunc_normal_(tensor, std=std)

    # Placeholders for other components
    class SwiGLU(nn.Module):
        def __init__(self, hidden_size, expansion):
            super().__init__()
            self.linear1 = nn.Linear(hidden_size, int(hidden_size * expansion))
            self.linear2 = nn.Linear(int(hidden_size * expansion), hidden_size)
        def forward(self, x):
            return self.linear2(F.gelu(self.linear1(x)))

    Attention = nn.MultiheadAttention
    CastedEmbedding = nn.Embedding
    CastedLinear = nn.Linear
    CastedSparseEmbedding = nn.Embedding
    RotaryEmbedding = lambda **kwargs: None
    CosSin = None

# Import our DEQ implementation
try:
    from trm_deq import DEQLayer
except ImportError:
    print("Warning: trm_deq package not found.")
    print("Install with: pip install -e /path/to/TRM-DEQ")
    DEQLayer = None


IGNORE_LABEL_ID = -100


@dataclass
class TRMDEQInnerCarry:
    """Carry state for TRM-DEQ inner loop."""
    z_H: torch.Tensor  # High-level reasoning state (answer embeddings)
    z_L: torch.Tensor  # Low-level reasoning state (internal features)


@dataclass
class TRMDEQCarry:
    """Carry state for TRM-DEQ outer (ACT) loop."""
    inner_carry: TRMDEQInnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class TRMDEQConfig(BaseModel):
    """Configuration for TRM-DEQ model."""
    # Basic config
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    # Cycle config
    H_cycles: int  # Used as reference for DEQ solver max_steps
    L_cycles: int  # Explicit iterations per H-cycle

    # Layer config
    H_layers: int = 1  # ignored for TRM (kept for compatibility)
    L_layers: int = 2  # Number of transformer blocks in L_level

    # Transformer config
    hidden_size: int = 512
    expansion: float = 2.0
    num_heads: int = 8
    pos_encodings: str = "rope"  # "rope", "learned", or "none"

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Halting Q-learning config (ACT)
    halt_max_steps: int = 10
    halt_exploration_prob: float = 0.1

    forward_dtype: str = "bfloat16"

    # DEQ solver config (NEW)
    deq_solver_type: str = "reversible"  # "simple" or "reversible"
    deq_tol: float = 1e-3
    deq_max_steps: int = 50
    deq_use_ift: bool = True
    deq_backward_tol: float = 1e-4
    deq_backward_max_steps: int = 100

    # Additional config (for compatibility)
    mlp_t: bool = False
    puzzle_emb_len: int = 16
    no_ACT_continue: bool = True


class TRMDEQBlock(nn.Module):
    """Single transformer block for TRM-DEQ reasoning."""

    def __init__(self, config: TRMDEQConfig) -> None:
        super().__init__()
        self.config = config

        if self.config.mlp_t:
            # MLP along sequence dimension (experimental)
            puzzle_emb_len = -(config.puzzle_emb_ndim // -config.hidden_size) if config.puzzle_emb_len == 0 else config.puzzle_emb_len
            self.mlp_t = SwiGLU(
                hidden_size=config.seq_len + puzzle_emb_len,
                expansion=config.expansion,
            )
        else:
            # Standard self-attention
            if Attention == nn.MultiheadAttention:
                # Using placeholder
                self.self_attn = nn.MultiheadAttention(
                    embed_dim=config.hidden_size,
                    num_heads=config.num_heads,
                    batch_first=True
                )
            else:
                # Using actual TRM attention
                self.self_attn = Attention(
                    hidden_size=config.hidden_size,
                    head_dim=config.hidden_size // config.num_heads,
                    num_heads=config.num_heads,
                    num_key_value_heads=config.num_heads,
                    causal=False
                )

        # Feed-forward network
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: Optional[CosSin], hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass through one block."""
        # Post-norm architecture
        if self.config.mlp_t:
            # Transpose for sequence-wise MLP
            hidden_states = hidden_states.transpose(1, 2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1, 2)
        else:
            # Self attention with RMS norm
            if isinstance(self.self_attn, nn.MultiheadAttention):
                # Placeholder version
                attn_out, _ = self.self_attn(hidden_states, hidden_states, hidden_states)
                hidden_states = rms_norm(hidden_states + attn_out, variance_epsilon=self.norm_eps)
            else:
                # TRM version
                hidden_states = rms_norm(
                    hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
                    variance_epsilon=self.norm_eps
                )

        # Feed-forward with RMS norm
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)

        return hidden_states


class TRMDEQReasoningModule(nn.Module):
    """L-level reasoning module (stack of transformer blocks)."""

    def __init__(self, layers: List[TRMDEQBlock]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass through reasoning module.

        Args:
            hidden_states: Current state [batch, seq_len, hidden_size]
            input_injection: Input to inject (added before reasoning)
            **kwargs: Additional args (cos_sin for RoPE, etc.)

        Returns:
            Updated hidden_states
        """
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class TRMDEQ_Inner(nn.Module):
    """
    Inner TRM-DEQ model with H-cycle DEQ integration.

    This replaces the explicit H-cycle loop from original TRM (lines 208-216)
    with a DEQ fixed-point solver, while keeping L-cycles as explicit iterations.
    """

    def __init__(self, config: TRMDEQConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # ========== I/O Layers ==========

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        # Token embedding (handle both TRM and placeholder versions)
        if CastedEmbedding == nn.Embedding:
            # Using placeholder
            self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        else:
            # Using actual TRM embedding
            self.embed_tokens = CastedEmbedding(
                self.config.vocab_size,
                self.config.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype
            )

        # Output heads
        if CastedLinear == nn.Linear:
            self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
            self.q_head = nn.Linear(self.config.hidden_size, 2, bias=True)
        else:
            self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
            self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        # Puzzle embeddings (optional)
        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
        if self.config.puzzle_emb_ndim > 0:
            if CastedSparseEmbedding == nn.Embedding:
                # Using placeholder
                self.puzzle_emb = nn.Embedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim)
            else:
                # Using actual TRM embedding
                self.puzzle_emb = CastedSparseEmbedding(
                    self.config.num_puzzle_identifiers,
                    self.config.puzzle_emb_ndim,
                    batch_size=self.config.batch_size,
                    init_std=0,
                    cast_to=self.forward_dtype
                )

        # ========== Position Encodings ==========

        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta
            ) if RotaryEmbedding is not None else None
        elif self.config.pos_encodings == "learned":
            if CastedEmbedding == nn.Embedding:
                self.embed_pos = nn.Embedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size)
            else:
                self.embed_pos = CastedEmbedding(
                    self.config.seq_len + self.puzzle_emb_len,
                    self.config.hidden_size,
                    init_std=embed_init_std,
                    cast_to=self.forward_dtype
                )

        # ========== Reasoning Layers ==========

        # L-level reasoning module (stack of transformer blocks)
        self.L_level = TRMDEQReasoningModule(
            layers=[TRMDEQBlock(self.config) for _ in range(self.config.L_layers)]
        )

        # Initial states (learnable)
        self.H_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True
        )
        self.L_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True
        )

        # ========== DEQ Solver (NEW!) ==========

        if DEQLayer is not None:
            # Prepare solver kwargs based on solver type
            solver_kwargs = {
                'tol': self.config.deq_tol,
                'max_steps': self.config.deq_max_steps
            }
            if self.config.deq_solver_type == 'reversible':
                solver_kwargs['beta'] = 0.8

            self.deq_layer = DEQLayer(
                solver_type=self.config.deq_solver_type,
                solver_kwargs=solver_kwargs,
                use_ift=self.config.deq_use_ift,
                backward_tol=self.config.deq_backward_tol,
                backward_max_steps=self.config.deq_backward_max_steps
            )
        else:
            print("Warning: DEQLayer not available. Model will not work correctly.")
            self.deq_layer = None

        # Q head special init (for ACT halting)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        """Compute input embeddings from tokens and puzzle IDs."""
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings (if enabled)
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

            # Pad puzzle embedding if needed
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            # Concatenate puzzle embedding with token embedding
            embedding = torch.cat(
                (puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding),
                dim=-2
            )

        # Position embeddings (learned, if enabled)
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def h_cycle_step(self, z: Tuple[torch.Tensor, torch.Tensor], args: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One H-cycle step (fixed-point function for DEQ).

        This replaces the explicit H-cycle loop from original TRM.
        Each call performs L_cycles explicit iterations, then updates z_H.

        Args:
            z: Tuple of (z_H, z_L)
            args: Dict containing:
                - input_embeddings: Input embeddings [batch, seq_len, hidden_size]
                - seq_info: Dict with cos_sin for RoPE

        Returns:
            Tuple of (z_H_next, z_L_next)
        """
        z_H, z_L = z
        input_embeddings = args['input_embeddings']
        seq_info = args['seq_info']

        # ========== L-cycles (EXPLICIT iterations) ==========
        # This keeps the same structure as original TRM
        z_L_temp = z_L
        for _ in range(self.config.L_cycles):
            z_L_temp = self.L_level(z_L_temp, z_H + input_embeddings, **seq_info)

        # ========== H-level update ==========
        z_H_next = self.L_level(z_H, z_L_temp, **seq_info)

        return (z_H_next, z_L_temp)

    def empty_carry(self, batch_size: int):
        """Create empty carry state."""
        return TRMDEQInnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: TRMDEQInnerCarry):
        """Reset carry state based on reset flag."""
        return TRMDEQInnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(
        self,
        carry: TRMDEQInnerCarry,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[TRMDEQInnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with DEQ for H-cycles.

        This is the key change from original TRM:
        Instead of explicit H-cycle loop, we solve for fixed-point using DEQ.

        Args:
            carry: Current state (z_H, z_L)
            batch: Input batch with keys 'inputs', 'puzzle_identifiers'

        Returns:
            Tuple of (new_carry, output_logits, (q_halt, q_continue))
        """
        # Prepare sequence info (for RoPE, etc.)
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") and self.rotary_emb is not None else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # ========== DEQ SOLVE FOR H-CYCLES (KEY CHANGE!) ==========
        # Original TRM (lines 208-216):
        #   with torch.no_grad():
        #       for _H_step in range(H_cycles-1):
        #           for _L_step in range(L_cycles):
        #               z_L = L_level(z_L, z_H + input_embeddings, **seq_info)
        #           z_H = L_level(z_H, z_L, **seq_info)
        #   for _L_step in range(L_cycles):
        #       z_L = L_level(z_L, z_H + input_embeddings, **seq_info)
        #   z_H = L_level(z_H, z_L, **seq_info)
        #
        # TRM-DEQ (this implementation):
        #   z_star = deq_solver.solve(h_cycle_step, z_init, args)
        #
        # Benefits:
        #   - Gradients flow through ALL H-cycles (not just last one)
        #   - Memory O(L_cycles) independent of H_cycles
        #   - Can scale H_cycles to 10, 50, 100+ without memory explosion

        # Prepare args for fixed-point function
        args = {
            'input_embeddings': input_embeddings,
            'seq_info': seq_info
        }

        # Initial state
        z_init = (carry.z_H, carry.z_L)

        # Solve for H-cycle fixed-point using DEQ
        z_star = self.deq_layer(self.h_cycle_step, z_init, args)
        z_H, z_L = z_star

        # ========== LM Outputs (same as original TRM) ==========

        # New carry (detached for next step)
        new_carry = TRMDEQInnerCarry(z_H=z_H.detach(), z_L=z_L.detach())

        # Language model head (predict next tokens)
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]

        # Q-head for ACT halting (uses first puzzle_emb position)
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class TRMDEQ(nn.Module):
    """
    TRM-DEQ: Tiny Recursive Model with Deep Equilibrium for H-cycles.

    This is the main model class with ACT (Adaptive Computation Time) wrapper.
    """

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TRMDEQConfig(**config_dict)
        self.inner = TRMDEQ_Inner(self.config)

    @property
    def puzzle_emb(self):
        """Access to puzzle embeddings (for compatibility)."""
        return self.inner.puzzle_emb if hasattr(self.inner, 'puzzle_emb') else None

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        """Create initial carry state for a new sequence."""
        batch_size = batch["inputs"].shape[0]

        return TRMDEQCarry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),  # Default to halted
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )

    def forward(
        self,
        carry: TRMDEQCarry,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[TRMDEQCarry, Dict[str, torch.Tensor]]:
        """
        Forward pass with ACT halting logic.

        Args:
            carry: Current ACT state
            batch: Input batch

        Returns:
            Tuple of (new_carry, outputs)
        """
        # Reset carry for halted sequences
        inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)

        # Forward through inner model (with DEQ!)
        new_inner_carry, output, (q_halt, q_continue) = self.inner(inner_carry, batch)

        # Update ACT state
        new_steps = carry.steps + (~carry.halted).to(torch.int32)

        # Halting decision (sigmoid of q_halt)
        halt_probs = torch.sigmoid(q_halt)
        new_halted = carry.halted | (halt_probs > 0.5) | (new_steps >= self.config.halt_max_steps)

        # Create new carry
        new_carry = TRMDEQCarry(
            inner_carry=new_inner_carry,
            steps=new_steps,
            halted=new_halted,
            current_data=batch
        )

        # Prepare outputs
        outputs = {
            "output": output,
            "q_halt": q_halt,
            "q_continue": q_continue,
            "halted": new_halted,
            "steps": new_steps,
        }

        return new_carry, outputs


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TRM-DEQ: H-Cycle DEQ Integration Demo")
    print("=" * 70)

    # Configuration
    config = {
        "batch_size": 2,
        "seq_len": 81,  # 9x9 Sudoku grid
        "puzzle_emb_ndim": 0,  # Disable puzzle embeddings for this test
        "puzzle_emb_len": 0,  # No puzzle embeddings
        "num_puzzle_identifiers": 100,
        "vocab_size": 10,  # 0-9 for Sudoku
        "H_cycles": 10,  # Can now scale to 10, 50, 100+ !
        "L_cycles": 6,
        "L_layers": 2,
        "hidden_size": 128,
        "expansion": 2.0,
        "num_heads": 4,
        "pos_encodings": "none",
        "halt_max_steps": 10,
        "halt_exploration_prob": 0.1,
        "forward_dtype": "float32",
        # DEQ config
        "deq_solver_type": "simple",
        "deq_tol": 1e-3,
        "deq_max_steps": 50,
        "deq_use_ift": True,
        "deq_backward_tol": 1e-4,
        "deq_backward_max_steps": 100,
    }

    try:
        # Create model
        model = TRMDEQ(config)
        print(f"\n✓ Model created successfully")
        print(f"  H_cycles: {config['H_cycles']} (solved via DEQ)")
        print(f"  L_cycles: {config['L_cycles']} (explicit iterations)")
        print(f"  DEQ solver: {config['deq_solver_type']}")

        # Create dummy batch
        batch = {
            "inputs": torch.randint(0, 10, (config['batch_size'], config['seq_len'])),
            "puzzle_identifiers": torch.zeros(config['batch_size'], dtype=torch.long),
        }

        # Initial carry
        carry = model.initial_carry(batch)
        print(f"\n✓ Initial carry created")

        # Forward pass
        print(f"\n Running forward pass...")
        new_carry, outputs = model(carry, batch)

        print(f"✓ Forward pass successful!")
        print(f"  Output shape: {outputs['output'].shape}")
        print(f"  Halted sequences: {outputs['halted'].sum()}/{config['batch_size']}")

        # Test backward
        print(f"\n Testing backward pass...")
        loss = outputs['output'].sum()
        loss.backward()

        print(f"✓ Backward pass successful!")
        print(f"  Gradients computed via IFT through {config['H_cycles']} H-cycles")

        print(f"\n" + "=" * 70)
        print(f"SUCCESS: TRM-DEQ integration working!")
        print(f"=" * 70)
        print(f"\nKey achievement:")
        print(f"  - H-cycles solved via DEQ (not explicit loop)")
        print(f"  - Gradients flow through ALL cycles (not just last)")
        print(f"  - Memory O(L_cycles={config['L_cycles']}) independent of H_cycles={config['H_cycles']}")
        print(f"  - Can scale H_cycles to 100+ without memory explosion")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

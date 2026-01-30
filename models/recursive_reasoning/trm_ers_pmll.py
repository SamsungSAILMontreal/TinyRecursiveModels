"""
Hybrid Tiny Recursive Transformer Model with ERS, Topic Integrator, and PMLL

This model combines:
- TRM (Tiny Recursive Model) architecture for efficient recursive reasoning
- ERS (Enhanced Reconsideration System) for persistent memory management
- PMLL (Persistent Memory Logic Loops) for multi-pass validation
- Topic Integrator for knowledge graph integration

Based on research from drqsatoshi.com by Dr. Josef Kurk Edwards
"""

from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
import hashlib
import time

from models.common import trunc_normal_init_
from models.layers import rms_norm, LinearSwish, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding

IGNORE_LABEL_ID = -100


@dataclass
class MemoryBlock:
    """Persistent memory block with temporal decay and consensus tracking"""
    content: torch.Tensor  # Embedding tensor
    confidence: float = 1.0
    timestamp: float = 0.0
    hash_id: str = ""
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.hash_id == "" and self.content is not None:
            # Generate hash from content
            content_bytes = self.content.detach().cpu().numpy().tobytes()
            self.hash_id = hashlib.sha256(content_bytes).hexdigest()[:16]


@dataclass 
class PMLLLatticeState:
    """PMLL Lattice state for tensor routing and processing"""
    routing_weights: torch.Tensor
    commitment_scores: torch.Tensor
    last_update: float = 0.0


@dataclass
class TRM_ERS_PMLL_InnerCarry:
    """Enhanced carry state with persistent memory"""
    z_H: torch.Tensor
    z_L: torch.Tensor
    # ERS persistent memory
    memory_blocks: List[MemoryBlock]
    # PMLL lattice state
    lattice_state: Optional[PMLLLatticeState]
    # Deferred reconsideration queue
    deferred_queue: List[Tuple[MemoryBlock, float]]


@dataclass
class TRM_ERS_PMLL_Carry:
    inner_carry: TRM_ERS_PMLL_InnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class TRM_ERS_PMLL_Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int
    H_layers: int
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # Halting config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    # TRM config
    mlp_t: bool = False
    puzzle_emb_len: int = 16
    no_ACT_continue: bool = True

    # ERS config
    ers_enabled: bool = True
    ers_memory_size: int = 128
    ers_temporal_decay_rate: float = 0.95
    ers_consensus_threshold: float = 0.7
    ers_contradiction_threshold: float = 0.3
    
    # PMLL config
    pmll_enabled: bool = True
    pmll_reconsideration_steps: int = 3
    pmll_commitment_threshold: float = 0.8
    pmll_lattice_dim: int = 64
    
    # Topic Integrator config
    topic_integrator_enabled: bool = True
    topic_integrator_max_topics: int = 16


class PMLLLattice(nn.Module):
    """PMLL Lattice for tensor routing and multi-pass validation"""
    
    def __init__(self, config: TRM_ERS_PMLL_Config):
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)
        
        # Routing network for tensor paths
        self.routing_network = nn.Sequential(
            nn.Linear(config.hidden_size, config.pmll_lattice_dim),
            nn.SiLU(),
            nn.Linear(config.pmll_lattice_dim, config.hidden_size)
        )
        
        # Commitment scoring
        self.commitment_head = nn.Linear(config.hidden_size, 1)
        
        # Multi-petal attention for embedding refinement
        self.attention_petals = nn.ModuleList([
            Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False
            )
            for _ in range(config.pmll_reconsideration_steps)
        ])
        
    def forward(self, hidden_states: torch.Tensor, cos_sin: Optional[CosSin] = None) -> Tuple[torch.Tensor, PMLLLatticeState]:
        """Process through PMLL lattice with multi-pass refinement"""
        B, L, D = hidden_states.shape
        
        # Route through lattice
        routing_output = self.routing_network(hidden_states)
        routing_weights = F.softmax(routing_output, dim=-1)
        
        # Multi-pass attention refinement
        refined_states = hidden_states
        for petal in self.attention_petals:
            refined_states = refined_states + petal(cos_sin=cos_sin, hidden_states=refined_states)
            refined_states = rms_norm(refined_states, variance_epsilon=self.config.rms_norm_eps)
        
        # Compute commitment scores
        commitment_scores = torch.sigmoid(self.commitment_head(refined_states))
        
        lattice_state = PMLLLatticeState(
            routing_weights=routing_weights.detach(),
            commitment_scores=commitment_scores.detach(),
            last_update=time.time()
        )
        
        return refined_states, lattice_state


class TopicIntegrator(nn.Module):
    """Topic Integrator for knowledge graph integration and topic-based memory"""
    
    def __init__(self, config: TRM_ERS_PMLL_Config):
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)
        
        # Topic embedding space
        self.topic_embeddings = nn.Embedding(
            config.topic_integrator_max_topics,
            config.hidden_size
        )
        
        # Topic assignment network
        self.topic_router = nn.Linear(config.hidden_size, config.topic_integrator_max_topics)
        
        # Topic fusion layer
        self.topic_fusion = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion
        )
        
    def forward(self, hidden_states: torch.Tensor, memory_blocks: List[MemoryBlock]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Integrate topics from memory and current state"""
        B, L, D = hidden_states.shape
        
        # Assign topics based on current state
        topic_logits = self.topic_router(hidden_states.mean(dim=1))
        topic_weights = F.softmax(topic_logits, dim=-1)
        
        # Get topic embeddings
        topic_context = torch.matmul(
            topic_weights,
            self.topic_embeddings.weight
        ).unsqueeze(1)
        
        # Fuse with hidden states
        fused_states = hidden_states + topic_context
        fused_states = self.topic_fusion(fused_states)
        
        return fused_states, topic_weights


class EnhancedReconsiderationSystem(nn.Module):
    """ERS for persistent memory management with temporal decay and consensus"""
    
    def __init__(self, config: TRM_ERS_PMLL_Config):
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)
        
        # Memory similarity computation
        self.memory_query = nn.Linear(config.hidden_size, config.hidden_size)
        self.memory_key = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Consensus and contradiction scoring
        self.consensus_head = nn.Linear(config.hidden_size * 2, 1)
        self.contradiction_head = nn.Linear(config.hidden_size * 2, 1)
        
    def temporal_decay(self, memory_blocks: List[MemoryBlock], current_time: float) -> List[MemoryBlock]:
        """Apply temporal decay to memory confidence"""
        decayed_blocks = []
        for block in memory_blocks:
            time_diff = current_time - block.timestamp
            decay_factor = self.config.ers_temporal_decay_rate ** time_diff
            block.confidence *= decay_factor
            decayed_blocks.append(block)
        return decayed_blocks
    
    def find_related(self, query_embedding: torch.Tensor, memory_blocks: List[MemoryBlock], k: int = 5) -> List[MemoryBlock]:
        """Find related memory blocks using embedding similarity"""
        if not memory_blocks:
            return []
        
        # Compute similarities
        similarities = []
        for block in memory_blocks:
            sim = F.cosine_similarity(
                query_embedding.flatten(),
                block.content.flatten(),
                dim=0
            )
            similarities.append((sim.item(), block))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [block for _, block in similarities[:k]]
    
    def compute_consensus(self, target_block: MemoryBlock, related_blocks: List[MemoryBlock]) -> float:
        """Compute consensus score from related memories"""
        if not related_blocks:
            return 0.5
        
        # Weighted average of confidences based on similarity
        total_confidence = 0.0
        total_weight = 0.0
        
        for block in related_blocks:
            similarity = F.cosine_similarity(
                target_block.content.flatten(),
                block.content.flatten(),
                dim=0
            ).item()
            weight = similarity * block.confidence
            total_confidence += weight
            total_weight += similarity
        
        return total_confidence / (total_weight + 1e-8)
    
    def detect_contradiction(self, target_block: MemoryBlock, related_blocks: List[MemoryBlock]) -> float:
        """Detect contradictions in memory"""
        if not related_blocks:
            return 0.0
        
        # Check for semantic contradictions
        contradictions = 0.0
        for block in related_blocks:
            # Negative similarity indicates contradiction
            similarity = F.cosine_similarity(
                target_block.content.flatten(),
                block.content.flatten(),
                dim=0
            ).item()
            
            if similarity < -0.5:  # Strong negative correlation
                contradictions += block.confidence * abs(similarity)
        
        return contradictions / len(related_blocks)
    
    def reconsider_memory(self, memory_blocks: List[MemoryBlock], current_embedding: torch.Tensor, current_time: float) -> List[MemoryBlock]:
        """Main reconsideration flow: decay → consensus → contradiction → update"""
        # Apply temporal decay
        memory_blocks = self.temporal_decay(memory_blocks, current_time)
        
        # Reconsider each block
        reconsidered_blocks = []
        for block in memory_blocks:
            # Find related memories
            related = self.find_related(block.content, memory_blocks, k=5)
            
            # Compute consensus
            consensus_score = self.compute_consensus(block, related)
            
            # Detect contradictions
            contradiction_score = self.detect_contradiction(block, related)
            
            # Update confidence
            if consensus_score > self.config.ers_consensus_threshold:
                block.confidence = min(1.0, block.confidence * 1.1)  # Boost
            
            if contradiction_score > self.config.ers_contradiction_threshold:
                block.confidence *= (1.0 - contradiction_score)  # Penalize
            
            # Keep blocks with sufficient confidence
            if block.confidence > 0.1:
                reconsidered_blocks.append(block)
        
        # Add new memory from current state
        new_block = MemoryBlock(
            content=current_embedding.detach(),
            confidence=1.0,
            timestamp=current_time
        )
        reconsidered_blocks.append(new_block)
        
        # Limit memory size
        if len(reconsidered_blocks) > self.config.ers_memory_size:
            # Sort by confidence and keep top blocks
            reconsidered_blocks.sort(key=lambda x: x.confidence, reverse=True)
            reconsidered_blocks = reconsidered_blocks[:self.config.ers_memory_size]
        
        return reconsidered_blocks


class TRM_ERS_PMLL_Block(nn.Module):
    """Transformer block with ERS and PMLL integration"""
    
    def __init__(self, config: TRM_ERS_PMLL_Config):
        super().__init__()
        self.config = config
        
        if config.mlp_t:
            self.puzzle_emb_len = -(config.puzzle_emb_ndim // -config.hidden_size) if config.puzzle_emb_len == 0 else config.puzzle_emb_len
            self.mlp_t = SwiGLU(
                hidden_size=config.seq_len + self.puzzle_emb_len,
                expansion=config.expansion,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False
            )
        
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Post Norm
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1, 2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1, 2)
        else:
            # Self Attention
            hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        
        # Fully Connected
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states


class TRM_ERS_PMLL_ReasoningModule(nn.Module):
    """Reasoning module with PMLL-enhanced recursive loops"""
    
    def __init__(self, layers: List[TRM_ERS_PMLL_Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class TRM_ERS_PMLL_Inner(nn.Module):
    """Inner model combining TRM, ERS, PMLL, and Topic Integrator"""
    
    def __init__(self, config: TRM_ERS_PMLL_Config):
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)

        # I/O
        self.embed_scale = math.sqrt(config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(config.vocab_size, config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)
        self.q_head = CastedLinear(config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(config.puzzle_emb_ndim // -config.hidden_size) if config.puzzle_emb_len == 0 else config.puzzle_emb_len
        if config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                config.num_puzzle_identifiers,
                config.puzzle_emb_ndim,
                batch_size=config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype
            )

        # Position encodings
        if config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=config.hidden_size // config.num_heads,
                max_position_embeddings=config.seq_len + self.puzzle_emb_len,
                base=config.rope_theta
            )
        elif config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                config.seq_len + self.puzzle_emb_len,
                config.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype
            )

        # Reasoning Layers
        self.L_level = TRM_ERS_PMLL_ReasoningModule(
            layers=[TRM_ERS_PMLL_Block(config) for _ in range(config.L_layers)]
        )

        # ERS Components
        if config.ers_enabled:
            self.ers = EnhancedReconsiderationSystem(config)
        
        # PMLL Components
        if config.pmll_enabled:
            self.pmll_lattice = PMLLLattice(config)
        
        # Topic Integrator
        if config.topic_integrator_enabled:
            self.topic_integrator = TopicIntegrator(config)

        # Initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Q head special init
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        """Generate input embeddings with puzzle context"""
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        """Create empty carry state"""
        return TRM_ERS_PMLL_InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
            memory_blocks=[],
            lattice_state=None,
            deferred_queue=[]
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: TRM_ERS_PMLL_InnerCarry):
        """Reset carry state for new sequences"""
        return TRM_ERS_PMLL_InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
            memory_blocks=[] if reset_flag.any() else carry.memory_blocks,
            lattice_state=None if reset_flag.any() else carry.lattice_state,
            deferred_queue=[] if reset_flag.any() else carry.deferred_queue
        )

    def forward(self, carry: TRM_ERS_PMLL_InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[TRM_ERS_PMLL_InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with ERS, PMLL, and Topic Integrator"""
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # ERS: Reconsider memory before forward pass
        current_time = time.time()
        if self.config.ers_enabled and hasattr(self, 'ers'):
            carry.memory_blocks = self.ers.reconsider_memory(
                carry.memory_blocks,
                input_embeddings.mean(dim=1),
                current_time
            )

        # Topic Integration
        z_H, z_L = carry.z_H, carry.z_L
        if self.config.topic_integrator_enabled and hasattr(self, 'topic_integrator'):
            z_H, topic_weights = self.topic_integrator(z_H, carry.memory_blocks)

        # PMLL-enhanced recursive loops
        # H_cycles-1 without grad (standard TRM approach)
        with torch.no_grad():
            for h_step in range(self.config.H_cycles - 1):
                # PMLL multi-pass reconsideration within each H cycle
                for pmll_pass in range(self.config.pmll_reconsideration_steps if self.config.pmll_enabled else 1):
                    for l_step in range(self.config.L_cycles):
                        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                    
                    # Apply PMLL lattice refinement
                    if self.config.pmll_enabled and hasattr(self, 'pmll_lattice'):
                        z_L, lattice_state = self.pmll_lattice(z_L, seq_info.get('cos_sin'))
                        # Update carry with lattice state
                        carry.lattice_state = lattice_state
                
                z_H = self.L_level(z_H, z_L, **seq_info)
        
        # Final H cycle with grad
        for pmll_pass in range(self.config.pmll_reconsideration_steps if self.config.pmll_enabled else 1):
            for l_step in range(self.config.L_cycles):
                z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
            
            # Apply PMLL lattice refinement
            if self.config.pmll_enabled and hasattr(self, 'pmll_lattice'):
                z_L, lattice_state = self.pmll_lattice(z_L, seq_info.get('cos_sin'))
                carry.lattice_state = lattice_state
        
        z_H = self.L_level(z_H, z_L, **seq_info)

        # LM Outputs
        new_carry = TRM_ERS_PMLL_InnerCarry(
            z_H=z_H.detach(),
            z_L=z_L.detach(),
            memory_blocks=carry.memory_blocks,
            lattice_state=carry.lattice_state,
            deferred_queue=carry.deferred_queue
        )
        
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class TinyRecursiveReasoningModel_ERS_PMLL(nn.Module):
    """Hybrid TRM with ERS, PMLL, and Topic Integrator"""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TRM_ERS_PMLL_Config(**config_dict)
        self.inner = TRM_ERS_PMLL_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        """Initialize carry state"""
        batch_size = batch["inputs"].shape[0]
        return TRM_ERS_PMLL_Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(self, carry: TRM_ERS_PMLL_Carry, batch: Dict[str, torch.Tensor]) -> Tuple[TRM_ERS_PMLL_Carry, Dict[str, torch.Tensor]]:
        """Forward pass with ACT wrapper"""
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v)
            for k, v in carry.current_data.items()
        }

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step

            # ACT halting
            if self.training and (self.config.halt_max_steps > 1):
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    _, _, (next_q_halt_logits, next_q_continue_logits) = self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return TRM_ERS_PMLL_Carry(new_inner_carry, new_steps, halted, new_current_data), outputs

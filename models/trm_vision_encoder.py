"""
TRM Vision Encoder: Use recursive reasoning for vision encoding.

Reuses existing TinyRecursiveReasoningModel_ACTV1 blocks for unified architecture.
Same TRM used for encoding AND reasoning!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1Block,
    TinyRecursiveReasoningModel_ACTV1ReasoningModule,
    TinyRecursiveReasoningModel_ACTV1Config,
)
from models.layers import RotaryEmbedding, LearnedPositionalEmbedding2D
from models.bitnet import BitLinear


class CapsuleAttentionPool(nn.Module):
    """
    Learned attention-weighted pooling for capsule assignment.
    
    Replaces AdaptiveAvgPool2d with learned attention maps that determine
    which spatial regions contribute to each capsule.
    
    Benefits:
    - Preserves spatial information (vs uniform averaging)
    - Learns semantic importance of regions
    - Compatible with existing DQN (fixed capsule count)
    """
    
    def __init__(self, in_channels: int, num_capsules: int):
        super().__init__()
        self.in_channels = in_channels
        self.num_capsules = num_capsules
        
        # 1x1 conv to produce attention logits per capsule
        self.attn_conv = nn.Conv2d(in_channels, num_capsules, kernel_size=1)
    
    def forward(self, feat_map):
        """
        Args:
            feat_map: [B, C, H, W] spatial features
        
        Returns:
            capsules: [B, K, C] weighted pooled features
            attention_maps: [B, K, H, W] attention weights (for children extraction)
        """
        B, C, H, W = feat_map.shape
        
        # Generate attention logits per capsule
        attn_logits = self.attn_conv(feat_map)  # [B, K, H, W]
        
        # Softmax over spatial dimension (each capsule attends to all locations)
        attn = F.softmax(attn_logits.view(B, self.num_capsules, -1), dim=-1)
        attn = attn.view(B, self.num_capsules, H, W)  # [B, K, H, W]
        
        # Weighted pooling: each capsule = weighted sum of spatial features
        # feat_map: [B, C, H, W] -> [B, 1, C, H, W]
        # attn: [B, K, H, W] -> [B, K, 1, H, W]
        feat_expanded = feat_map.unsqueeze(1)  # [B, 1, C, H, W]
        attn_expanded = attn.unsqueeze(2)  # [B, K, 1, H, W]
        
        # Weighted sum over spatial dims
        capsules = (feat_expanded * attn_expanded).sum(dim=(-2, -1))  # [B, K, C]
        
        return capsules, attn
    
    def get_attention_entropy(self, attn_maps):
        """
        Compute attention concentration metric (lower entropy = more focused).
        
        Args:
            attn_maps: [B, K, H, W] attention weights
        
        Returns:
            entropy: [B, K] entropy per capsule
        """
        B, K, H, W = attn_maps.shape
        attn_flat = attn_maps.view(B, K, -1)  # [B, K, H*W]
        
        # Entropy: -sum(p * log(p))
        entropy = -(attn_flat * torch.log(attn_flat + 1e-8)).sum(dim=-1)
        return entropy


class TRMVisionEncoder(nn.Module):
    """
    Vision encoder using TRM recursive reasoning architecture.
    
    Pipeline:
    1. Patch embedding: Image → tokens
    2. TRM blocks: Recursive spatial reasoning (H_cycles × L_cycles)
    3. Spatial pooling: Tokens → capsules (3×4 grid)
    
    Benefits:
    - Unified architecture (same TRM for encoding + reasoning)
    - Trainable on ARC spatial patterns
    - Small size: ~3-5M params (vs 150M CLIP)
    - Native 512D output (no projection loss)
    """
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        hidden_size: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,  # L_layers for encoder
        H_cycles: int = 2,  # High-level cycles
        L_cycles: int = 3,  # Low-level cycles per H-cycle
        target_capsules: int = 12,
        capsule_grid_shape: tuple = (3, 4),
        expansion: int = 4,
        dropout: float = 0.0,
        pooling_method: str = "attention",  # "avg" | "attention"
        real_children: bool = True,
        pretrained_model: str = 'clip',  # Which pretrained model (clip/dinov2/siglip)
        fusion_type: str = 'gated',  # Fusion strategy (gated/attention/learned_avg)
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_patches = (image_size // patch_size) ** 2
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.target_capsules = target_capsules
        self.capsule_grid_shape = capsule_grid_shape
        self.pooling_method = pooling_method
        self.real_children = real_children
        
        # ALWAYS use hybrid encoder (pretrained + trainable + fusion + N2N)
        # Benefits: Generalization + Specialization + No downsides
        from models.hybrid_vision_encoder import HybridVisionEncoder
        self.hybrid_encoder = HybridVisionEncoder(
            pretrained_model=pretrained_model,
            hidden_size=hidden_size,
            fusion_type=fusion_type,
            freeze_pretrained=True
        )
        print(f"   Hybrid encoder: CLIP + ViT + N2N adapter (all mandatory)")
        print(f"   TRM cycles: {H_cycles}H × {L_cycles}L = iterative refinement")
        
        # Create TRM config for encoder
        encoder_config = TinyRecursiveReasoningModel_ACTV1Config(
            batch_size=1,  # Dynamic
            seq_len=self.num_patches,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_key_value_heads=num_heads,  # Full MHA (ViT/CLIP standard, not GQA)
            expansion=expansion,
            H_layers=0,  # Not used (manual cycles)
            L_layers=num_layers,
            H_cycles=H_cycles,  # High-level cycles
            L_cycles=L_cycles,  # Low-level cycles
            vocab_size=2052,  # Dummy
            num_puzzle_identifiers=0,
            puzzle_emb_ndim=0,
            causal=False,  # Non-causal for vision
            rms_norm_eps=1e-6,
            mlp_t=False,
            pos_encodings="none",  # Using learned 2D pos embeddings instead
            halt_max_steps=100,  # Not used in encoder
            halt_exploration_prob=0.0,  # Not used in encoder
        )
        
        # TRM L-level blocks (reuse existing architecture!)
        self.L_blocks = nn.ModuleList([
            TinyRecursiveReasoningModel_ACTV1Block(encoder_config)
            for _ in range(num_layers)
        ])
        
        # Layer normalization before pooling (ViT/CLIP standard)
        self.pre_pool_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        
        # Spatial pooling to capsule grid
        if pooling_method == "attention":
            self.spatial_pool = CapsuleAttentionPool(hidden_size, target_capsules)
            print(f"   Pooling: Learned attention ({target_capsules} attention maps)")
        else:
            self.spatial_pool = nn.AdaptiveAvgPool2d(capsule_grid_shape)
            print(f"   Pooling: Adaptive average (grid {capsule_grid_shape})")
        
        print(f"\nTRM Vision Encoder Initialized:")
        print(f"   Patches: {self.num_patches} ({image_size//patch_size}×{image_size//patch_size})")
        print(f"   Hidden: {hidden_size}")
        print(f"   Layers: {num_layers}")
        print(f"   Cycles: H={H_cycles} × L={L_cycles}")
        print(f"   Output: {target_capsules} capsules (512D)")
        print(f"   Params: ~{self.count_parameters()/1e6:.1f}M")
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, images):
        """
        Encode images to capsules using TRM recursive reasoning.
        
        Args:
            images: [B, 3, 224, 224] or [B, 3, H, W]
        
        Returns:
            capsules: [B, 12, 512]
        """
        B = images.shape[0]
        
        # Get initial features from hybrid encoder
        # Hybrid: CLIP (frozen) + Custom ViT (trainable) → Fusion → N2N Adapter
        tokens = self.hybrid_encoder(images)  # [B, 196, 768]
        # Note: Positional info already embedded by pretrained models
        
        # TRM Recursive Reasoning (H_cycles × L_cycles)
        # No cos_sin needed - using learned positional embeddings
        for h_cycle in range(self.H_cycles):
            for l_cycle in range(self.L_cycles):
                # Apply L-level blocks
                for block in self.L_blocks:
                    tokens = block(
                        cos_sin=None,  # No RoPE for vision
                        hidden_states=tokens
                    )
        
        # Apply layer normalization before pooling (ViT/CLIP standard)
        tokens = self.pre_pool_norm(tokens)  # [B, 196, 768]
        
        # Reshape to spatial grid: [B, 196, 768] -> [B, 768, 14, 14]
        H_patches = W_patches = self.image_size // self.patch_size  # 14x14 for 224/16
        tokens_spatial = tokens.transpose(1, 2).view(
            B, self.hidden_size, H_patches, W_patches
        )
        
        # Spatial pooling to capsules
        if self.pooling_method == "attention":
            # Attention-weighted pooling: [B, 512, 14, 14] -> [B, 12, 512]
            capsules, attention_maps = self.spatial_pool(tokens_spatial)
            # Store attention maps for children extraction
            self._last_attention_maps = attention_maps  # [B, 12, 14, 14]
            self._last_spatial_tokens = tokens  # [B, 196, 512]
        else:
            # Adaptive avg pooling: [B, 512, 14, 14] -> [B, 512, 3, 4]
            capsules_spatial = self.spatial_pool(tokens_spatial)
            # Flatten: [B, 512, 3, 4] -> [B, 12, 512]
            capsules = capsules_spatial.flatten(2).transpose(1, 2)
            self._last_attention_maps = None
            self._last_spatial_tokens = tokens
        
        return capsules


class TRMVisionEncoderWithChecksums(nn.Module):
    """
    TRM Vision Encoder with HESC checksums and children support.
    
    Wrapper around TRMVisionEncoder to match CapsuleEncoder interface.
    """
    
    def __init__(
        self,
        hidden_size: int = 512,
        target_capsules: int = 12,
        children_per_capsule: int = 4,
        checksum_dim: int = 32,
        **trm_kwargs
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.target_capsules = target_capsules
        self.children_per_capsule = children_per_capsule
        self.checksum_dim = checksum_dim
        
        # TRM encoder
        self.encoder = TRMVisionEncoder(
            hidden_size=hidden_size,
            target_capsules=target_capsules,
            **trm_kwargs
        )
        
        # Checksum head (integrity signature)
        self.checksum_head = nn.Sequential(
            BitLinear(hidden_size, 128),
            nn.ReLU(),
            BitLinear(128, checksum_dim)
        )
        
        # Children projection (for hierarchical expansion)
        self.children_projection = BitLinear(hidden_size, hidden_size)
    
    def _extract_topk_children(self, spatial_tokens, attention_maps, m: int = 4):
        """
        Extract top-m spatial patches per capsule based on attention weights.
        
        Args:
            spatial_tokens: [B, H*W, C] flattened patch embeddings
            attention_maps: [B, K, H, W] attention weights per capsule
            m: number of children per capsule
        
        Returns:
            children: [B, K, m, C] top-m patches for each capsule
        """
        B, K, H, W = attention_maps.shape
        _, num_patches, C = spatial_tokens.shape
        
        # Flatten attention maps: [B, K, H*W]
        attn_flat = attention_maps.view(B, K, -1)
        
        # Select top-m patches per capsule
        topk_vals, topk_idx = torch.topk(attn_flat, k=m, dim=-1)  # [B, K, m]
        
        # Gather children embeddings
        # Expand indices for batch gathering
        batch_idx = torch.arange(B, device=spatial_tokens.device)[:, None, None]
        batch_idx = batch_idx.expand(B, K, m)
        
        # Gather: spatial_tokens[batch_idx, topk_idx]
        children = spatial_tokens[batch_idx, topk_idx]  # [B, K, m, C]
        
        return children
    
    def forward(self, images, return_children: bool = True):
        """
        Encode images to HESC capsules.
        
        Args:
            images: [B, 3, 224, 224]
            return_children: Whether to compute children embeddings
        
        Returns:
            dict with 'sketches', 'checksums', 'children'
        """
        # Encode to capsules
        sketches = self.encoder(images)  # [B, 12, 512]
        
        # Compute checksums
        checksums = self.checksum_head(sketches)  # [B, 12, 32]
        
        result = {
            'sketches': sketches,
            'checksums': checksums
        }
        
        # Children: extract top-m spatial patches per capsule via attention
        if return_children:
            if self.encoder.real_children and self.encoder._last_attention_maps is not None:
                # Real children: top-m patches by attention weight
                children = self._extract_topk_children(
                    self.encoder._last_spatial_tokens,
                    self.encoder._last_attention_maps,
                    m=self.children_per_capsule
                )
            else:
                # Fallback: duplicate sketches (legacy behavior)
                children = sketches.unsqueeze(2).repeat(
                    1, 1, self.children_per_capsule, 1
                )  # [B, 12, 4, 512]
            
            children = self.children_projection(children)
            result['children'] = children
        
        return result


if __name__ == "__main__":
    # Test encoder
    print("Testing TRM Vision Encoder...")
    
    encoder = TRMVisionEncoderWithChecksums(
        hidden_size=512,
        target_capsules=12,
        num_layers=2,
        H_cycles=2,
        L_cycles=3
    )
    
    # Dummy image batch
    images = torch.randn(4, 3, 224, 224)
    
    # Encode
    output = encoder(images, return_children=True)
    
    print(f"\nOutput shapes:")
    print(f"  Sketches: {output['sketches'].shape}")  # [4, 12, 512]
    print(f"  Checksums: {output['checksums'].shape}")  # [4, 12, 32]
    print(f"  Children: {output['children'].shape}")  # [4, 12, 4, 512]
    
    print("\nTRM Vision Encoder working!")

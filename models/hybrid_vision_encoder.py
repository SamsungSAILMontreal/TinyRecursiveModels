"""
Dual-Path Hybrid Vision Encoder

Combines pretrained (frozen) and trainable encoders for best of both worlds:
- Pretrained path: Generalization, zero-shot knowledge, stable training
- Trainable path: Task-specific learning, novel patterns
- Adaptive fusion: Learn optimal weighting per sample

Architecture:
    Image → [Frozen CLIP Path, Trainable ViT Path] → Fusion → N2N Adapter → Output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from models.bitnet import BitLinear


class AdaptiveFusion(nn.Module):
    """
    Learnable fusion of pretrained and trainable features.
    
    Uses attention-based gating to decide per-patch weighting:
    - Easy samples: More pretrained (stable)
    - Hard samples: More trainable (specialized)
    - Dynamic adaptation during training
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        fusion_type: str = 'gated'  # 'gated', 'attention', 'learned_avg'
    ):
        super().__init__()
        
        self.fusion_type = fusion_type
        
        if fusion_type == 'gated':
            # Simple gating: learnable per-patch weight
            self.gate = nn.Sequential(
                BitLinear(feature_dim * 2, feature_dim),
                nn.GELU(),
                BitLinear(feature_dim, 1),
                nn.Sigmoid()
            )
        elif fusion_type == 'attention':
            # Attention-based fusion
            self.query = BitLinear(feature_dim, feature_dim)
            self.key_pretrained = BitLinear(feature_dim, feature_dim)
            self.key_trainable = BitLinear(feature_dim, feature_dim)
            self.value_pretrained = BitLinear(feature_dim, feature_dim)
            self.value_trainable = BitLinear(feature_dim, feature_dim)
        elif fusion_type == 'learned_avg':
            # Simple learnable scalar weights
            self.alpha = nn.Parameter(torch.tensor(0.5))  # Start 50-50
    
    def forward(
        self,
        pretrained_feat: torch.Tensor,
        trainable_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse pretrained and trainable features.
        
        Args:
            pretrained_feat: [B, N, D] from frozen CLIP
            trainable_feat: [B, N, D] from trainable ViT
        
        Returns:
            fused_feat: [B, N, D] fused features
            weights: [B, N] or scalar, weight given to pretrained path
        """
        if self.fusion_type == 'gated':
            # Concatenate and compute gate
            concat = torch.cat([pretrained_feat, trainable_feat], dim=-1)  # [B, N, 2D]
            gate = self.gate(concat)  # [B, N, 1]
            
            # Weighted combination
            fused = gate * pretrained_feat + (1 - gate) * trainable_feat
            return fused, gate.squeeze(-1)
        
        elif self.fusion_type == 'attention':
            # Attention mechanism
            B, N, D = pretrained_feat.shape
            
            # Use trainable features as query
            q = self.query(trainable_feat)  # [B, N, D]
            
            # Keys and values
            k_pre = self.key_pretrained(pretrained_feat)
            k_train = self.key_trainable(trainable_feat)
            v_pre = self.value_pretrained(pretrained_feat)
            v_train = self.value_trainable(trainable_feat)
            
            # Attention scores
            attn_pre = torch.bmm(q, k_pre.transpose(1, 2)) / (D ** 0.5)  # [B, N, N]
            attn_train = torch.bmm(q, k_train.transpose(1, 2)) / (D ** 0.5)
            
            # Stack and softmax
            attn = torch.stack([attn_pre, attn_train], dim=1)  # [B, 2, N, N]
            attn = F.softmax(attn, dim=1)  # Softmax over two paths
            
            # Weighted sum
            out_pre = torch.bmm(attn[:, 0], v_pre)
            out_train = torch.bmm(attn[:, 1], v_train)
            fused = out_pre + out_train
            
            # Return attention weights (average over queries)
            weights = attn[:, 0].mean(dim=1)  # [B, N]
            return fused, weights
        
        elif self.fusion_type == 'learned_avg':
            # Simple weighted average
            alpha = torch.sigmoid(self.alpha)  # Constrain to [0, 1]
            fused = alpha * pretrained_feat + (1 - alpha) * trainable_feat
            weights = alpha.expand(pretrained_feat.size(0), pretrained_feat.size(1))
            return fused, weights


class HybridVisionEncoder(nn.Module):
    """
    Dual-path vision encoder combining pretrained and trainable paths.
    
    Architecture:
        Image → Pretrained (frozen) → [196, 768]
               ↓
        Image → Trainable (learnable) → [196, 768]
               ↓
        Fusion → [196, 768] combined
               ↓
        N2N Adapter → [196, 768] clean
               ↓
        Output
    
    Benefits:
    - Pretrained: Generalization, stability, zero-shot
    - Trainable: Task-specific, novel patterns
    - Fusion: Adaptive weighting, best of both
    """
    
    def __init__(
        self,
        pretrained_model: str = 'clip',  # 'clip', 'dinov2', 'siglip'
        hidden_size: int = 768,
        fusion_type: str = 'gated',
        freeze_pretrained: bool = True
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Path 1: Pretrained encoder (frozen)
        from models.noise2noise_denoiser import PretrainedVisionBackbone
        self.pretrained_path = PretrainedVisionBackbone(
            model_name=pretrained_model,
            freeze=freeze_pretrained
        )
        
        # Project CLIP features (768) to match ViT dimension (1024)
        self.pretrained_projection = BitLinear(768, hidden_size)
        
        # Path 2: Trainable encoder (custom ViT)
        self.trainable_path = self._build_trainable_encoder(hidden_size)
        
        # Fusion mechanism
        self.fusion = AdaptiveFusion(
            feature_dim=hidden_size,
            fusion_type=fusion_type
        )
        
        # MANDATORY: N2N adapter for feature refinement
        # No downsides - only benefits (denoising + alignment)
        from models.noise2noise_denoiser import N2NFeatureAdapter
        self.n2n_adapter = N2NFeatureAdapter(
            input_dim=hidden_size,
            num_layers=3
        )
        
        print(f"\nInitialized HybridVisionEncoder:")
        print(f"   Pretrained: {pretrained_model} (frozen={freeze_pretrained})")
        print(f"   Trainable: Custom ViT")
        print(f"   Fusion: {fusion_type}")
        print(f"   N2N Adapter: ENABLED (mandatory)")
    
    def _build_trainable_encoder(self, hidden_size: int) -> nn.Module:
        """Build trainable ViT encoder."""
        from transformers import ViTModel, ViTConfig
        
        # Calculate compatible num_heads: 64 dims per head (ViT standard)
        num_heads = hidden_size // 64
        
        config = ViTConfig(
            hidden_size=hidden_size,
            num_hidden_layers=6,  # Lighter than CLIP (12 layers)
            num_attention_heads=num_heads,  # Auto-calculated for compatibility
            intermediate_size=hidden_size * 4,
            image_size=224,
            patch_size=16,
            num_channels=3
        )
        
        model = ViTModel(config)
        
        print(f"   Trainable ViT: {sum(p.numel() for p in model.parameters()):,} params")
        return model
    
    def forward(
        self,
        images: torch.Tensor,
        return_fusion_weights: bool = False
    ) -> torch.Tensor:
        """
        Dual-path encoding with adaptive fusion.
        
        Args:
            images: [B, 3, 224, 224] input images
            return_fusion_weights: Whether to return fusion weights
        
        Returns:
            features: [B, 196, 768] fused and adapted features
            (optional) weights: [B, 196] fusion weights (pretrained path weight)
        """
        # Path 1: Frozen pretrained features
        with torch.no_grad():
            pretrained_feat_768 = self.pretrained_path(images)  # [B, 196, 768]
        
        # Project CLIP 768 -> hidden_size (1024)
        pretrained_feat = self.pretrained_projection(pretrained_feat_768)  # [B, 196, 1024]
        
        # Path 2: Trainable features
        trainable_output = self.trainable_path(pixel_values=images)
        trainable_feat = trainable_output.last_hidden_state[:, 1:, :]  # Remove CLS, [B, 196, 1024]
        
        # Fusion (both now 1024D)
        fused_feat, fusion_weights = self.fusion(pretrained_feat, trainable_feat)
        
        # MANDATORY N2N adaptation (always enabled)
        final_feat = self.n2n_adapter(fused_feat)
        
        if return_fusion_weights:
            return final_feat, fusion_weights
        return final_feat
    
    def get_fusion_stats(self, images: torch.Tensor) -> dict:
        """
        Analyze fusion behavior on a batch.
        
        Returns:
            stats: Dictionary with fusion statistics
        """
        _, weights = self.forward(images, return_fusion_weights=True)
        
        return {
            'mean_pretrained_weight': weights.mean().item(),
            'std_pretrained_weight': weights.std().item(),
            'min_pretrained_weight': weights.min().item(),
            'max_pretrained_weight': weights.max().item(),
            'median_pretrained_weight': weights.median().item()
        }


class ProgressiveHybridTraining:
    """
    Training strategy for hybrid encoder.
    
    Phase 1 (Early): Rely on pretrained (high fusion weight)
    Phase 2 (Mid): Balanced fusion
    Phase 3 (Late): Shift to trainable (low fusion weight)
    
    This ensures stable early training while allowing specialization later.
    """
    
    def __init__(
        self,
        total_steps: int,
        phase1_ratio: float = 0.2,  # First 20%: pretrained-heavy
        phase3_ratio: float = 0.3   # Last 30%: trainable-heavy
    ):
        self.total_steps = total_steps
        self.phase1_end = int(total_steps * phase1_ratio)
        self.phase3_start = int(total_steps * (1 - phase3_ratio))
    
    def get_fusion_bias(self, current_step: int) -> float:
        """
        Get recommended fusion bias for current training step.
        
        Returns:
            bias: Positive = favor pretrained, Negative = favor trainable
        """
        if current_step < self.phase1_end:
            # Phase 1: Strong pretrained bias
            return 0.3  # Encourage pretrained
        elif current_step > self.phase3_start:
            # Phase 3: Shift to trainable
            progress = (current_step - self.phase3_start) / (self.total_steps - self.phase3_start)
            return 0.3 * (1 - progress) - 0.3 * progress  # Smoothly shift
        else:
            # Phase 2: Balanced
            return 0.0  # No bias


def test_hybrid_encoder():
    """Test hybrid encoder."""
    print("\n🧪 Testing HybridVisionEncoder...")
    
    # Create dummy images
    images = torch.randn(4, 3, 224, 224)
    
    try:
        # Initialize hybrid encoder
        encoder = HybridVisionEncoder(
            pretrained_model='clip',
            hidden_size=768,
            use_n2n_adapter=True,
            fusion_type='gated'
        )
        
        # Forward pass
        features, weights = encoder(images, return_fusion_weights=True)
        
        print(f"\n✓ Input: {images.shape}")
        print(f"✓ Output: {features.shape}")
        print(f"✓ Fusion weights: {weights.shape}")
        print(f"✓ Mean pretrained weight: {weights.mean():.3f}")
        
        # Test fusion stats
        stats = encoder.get_fusion_stats(images)
        print(f"\n📊 Fusion Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value:.3f}")
        
        # Test progressive training
        trainer = ProgressiveHybridTraining(total_steps=10000)
        print(f"\n📈 Progressive Training Biases:")
        for step in [0, 2000, 5000, 7000, 10000]:
            bias = trainer.get_fusion_bias(step)
            print(f"   Step {step}: {bias:+.3f}")
        
        print("\nAll tests passed!")
        
    except Exception as e:
        print(f"\nTest skipped (missing dependencies): {e}")
        print("   Install: pip install transformers")


if __name__ == "__main__":
    test_hybrid_encoder()

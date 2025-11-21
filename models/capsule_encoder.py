"""
Hierarchical Expandable Semantic Capsules (HESC)

Creates capsules with:
- Sketch: coarse semantic embedding (concept-level)
- Checksum: reconstructability signature  
- Children: fine-grained tokens (expandable on-demand)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.trm_vision_encoder import TRMVisionEncoder


class CapsuleEncoder(nn.Module):
    """
    HESC encoder: text → capsules with sketch/checksum/children.
    
    Args:
        hidden_size: TRM hidden dimension (768)
        target_capsules: Number of coarse capsules (k=12)
        children_per_capsule: Fine tokens per capsule (m=4)
        checksum_dim: Reconstructability signature size (32)
    """
    
    def __init__(
        self,
        hidden_size: int = 768,  # Match CLIP/DINOv2 dimension
        target_capsules: int = 12,
        children_per_capsule: int = 4,
        checksum_dim: int = 32,
        num_layers: int = 2,
        H_cycles: int = 2,
        L_cycles: int = 3,
        capsule_grid_shape: tuple = (3, 4),
        pretrained_model: str = 'clip',  # Pretrained backbone (clip/dinov2/siglip)
        fusion_type: str = 'gated',  # Fusion strategy (gated/attention/learned_avg)
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.target_capsules = target_capsules
        self.children_per_capsule = children_per_capsule
        self.checksum_dim = checksum_dim
        self.capsule_grid_shape = capsule_grid_shape
        
        assert capsule_grid_shape[0] * capsule_grid_shape[1] == target_capsules, \
            f"Grid shape {capsule_grid_shape} must multiply to {target_capsules}"
        
        # Always use hybrid encoder (pretrained + trainable + N2N)
        # No downsides - only benefits
        print(f"\nInitializing CapsuleEncoder (HYBRID):")
        print(f"   Pretrained: {pretrained_model} (frozen)")
        print(f"   Trainable: Custom ViT")
        print(f"   Fusion: {fusion_type}")
        print(f"   N2N Adapter: Enabled")
        print(f"   TRM Cycles: H={H_cycles} × L={L_cycles} (iterative refinement)")
        print(f"   Target capsules: {target_capsules}")
        print(f"   Output: {hidden_size}D")
        
        from models.trm_vision_encoder import TRMVisionEncoderWithChecksums
        self.encoder = TRMVisionEncoderWithChecksums(
            hidden_size=hidden_size,
            target_capsules=target_capsules,
            children_per_capsule=children_per_capsule,
            checksum_dim=checksum_dim,
            num_layers=num_layers,
            H_cycles=H_cycles,
            L_cycles=L_cycles,
            pretrained_model=pretrained_model,
            fusion_type=fusion_type,
        )
    
    def forward(self, texts=None, images=None, return_children: bool = True):
        """
        Encode images to capsules using TRM.
        
        Args:
            images: [B, 3, 224, 224] images (text must be pre-rendered to images)
            return_children: Whether to compute children embeddings
        
        Returns:
            dict with 'sketches' [B,k,D], 'checksums' [B,k,R], 'children' [B,k,m,D]
        """
        if images is None:
            raise ValueError("TRM encoder requires images (render text to images first)")
        
        return self.encoder(images=images, return_children=return_children)

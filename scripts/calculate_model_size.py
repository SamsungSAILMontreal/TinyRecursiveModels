"""
Calculate model size, parameters, and memory footprint.

Breaks down:
- Pretrained frozen weights (CLIP)
- Trainable weights (Custom ViT + N2N + TRM + COCONUT)
- Optimizer states (AdamW)
- Activations (forward/backward)
"""

def calculate_clip_params():
    """CLIP ViT-B/16 parameters."""
    return {
        'patch_embedding': 150_528,      # 3*768*16*16 + 768
        'positional_embedding': 151_296,  # 197*768
        'transformer_blocks': 85_054_464, # 12 layers * ~7.1M params/layer
        'layer_norm': 1_536,             # 2*768
        'projection': 590_592,           # 768*768 + 768
        'total': 86_192_896
    }

def calculate_custom_vit_params():
    """Custom ViT (6 layers, 768 hidden) parameters."""
    hidden = 768
    num_layers = 6
    
    # Per-layer params
    attention_params = (
        3 * hidden * hidden +  # QKV projection
        hidden * hidden        # Output projection
    )
    mlp_params = (
        hidden * (hidden * 4) +  # Up projection
        (hidden * 4) * hidden    # Down projection
    )
    norm_params = hidden * 2  # LayerNorm scale + bias
    
    per_layer = attention_params + mlp_params + norm_params
    # per_layer ≈ 7,087,872
    
    return {
        'patch_embedding': 150_528,
        'positional_embedding': 151_296,
        'transformer_blocks': per_layer * num_layers,  # ~42.5M
        'layer_norm': 1_536,
        'total': 150_528 + 151_296 + (per_layer * num_layers) + 1_536
    }

def calculate_fusion_params():
    """Adaptive fusion gating parameters."""
    hidden = 768
    return {
        'gate_network': (
            (hidden * 2) * hidden +  # Linear 1
            hidden * 1               # Linear 2
        ),
        'total': (hidden * 2) * hidden + hidden
    }

def calculate_n2n_adapter_params():
    """N2N Feature Adapter (3 transformer layers)."""
    hidden = 768
    num_layers = 3
    
    # Same structure as transformer layer
    per_layer = (
        4 * hidden * hidden +   # Attention (Q,K,V,O)
        hidden * (hidden * 4) + # MLP up
        (hidden * 4) * hidden + # MLP down
        hidden * 4              # LayerNorms
    )
    
    return {
        'transformer_blocks': per_layer * num_layers,
        'total': per_layer * num_layers
    }

def calculate_trm_params():
    """TRM recursive reasoning parameters."""
    hidden = 768
    num_heads = 12
    num_layers = 2  # L-level blocks
    
    # L-level blocks (similar to transformer)
    per_layer = (
        4 * hidden * hidden +   # Attention
        hidden * (hidden * 4) + # MLP
        (hidden * 4) * hidden +
        hidden * 4              # Norms
    )
    
    # Additional components
    pre_pool_norm = hidden * 2
    spatial_pool = hidden * 12  # CapsuleAttentionPool
    
    return {
        'L_blocks': per_layer * num_layers,
        'pre_pool_norm': pre_pool_norm,
        'spatial_pool': spatial_pool,
        'total': (per_layer * num_layers) + pre_pool_norm + spatial_pool
    }

def calculate_coconut_params():
    """COCONUT latent planning parameters."""
    hidden = 768
    num_paths = 4
    depth = 2
    
    # Path exploration network
    path_network = hidden * hidden * num_paths  # Per path projection
    # Scoring network
    scoring = hidden * num_paths + num_paths
    # Adaptive gate
    gate = hidden * 2
    
    return {
        'path_network': path_network,
        'scoring': scoring,
        'adaptive_gate': gate,
        'total': path_network + scoring + gate
    }

def calculate_total():
    """Calculate total model breakdown."""
    
    clip = calculate_clip_params()
    custom_vit = calculate_custom_vit_params()
    fusion = calculate_fusion_params()
    n2n = calculate_n2n_adapter_params()
    trm = calculate_trm_params()
    coconut = calculate_coconut_params()
    
    # Total parameters
    frozen_params = clip['total']
    trainable_params = (
        custom_vit['total'] +
        fusion['total'] +
        n2n['total'] +
        trm['total'] +
        coconut['total']
    )
    total_params = frozen_params + trainable_params
    
    print("=" * 70)
    print("MODEL PARAMETER BREAKDOWN")
    print("=" * 70)
    
    print("\n📌 FROZEN COMPONENTS (No gradients, no optimizer states):")
    print(f"   CLIP ViT-B/16:              {clip['total']:>12,} params  ({clip['total']*4/1e6:.1f} MB)")
    
    print("\n🔧 TRAINABLE COMPONENTS (Have gradients + optimizer states):")
    print(f"   Custom ViT (6 layers):      {custom_vit['total']:>12,} params  ({custom_vit['total']*4/1e6:.1f} MB)")
    print(f"   Fusion (gated):             {fusion['total']:>12,} params  ({fusion['total']*4/1e6:.1f} MB)")
    print(f"   N2N Adapter (3 layers):     {n2n['total']:>12,} params  ({n2n['total']*4/1e6:.1f} MB)")
    print(f"   TRM (2 layers):             {trm['total']:>12,} params  ({trm['total']*4/1e6:.1f} MB)")
    print(f"   COCONUT (4 paths):          {coconut['total']:>12,} params  ({coconut['total']*4/1e6:.1f} MB)")
    print(f"   {'─' * 68}")
    print(f"   Trainable subtotal:         {trainable_params:>12,} params  ({trainable_params*4/1e6:.1f} MB)")
    
    print(f"\n{'=' * 70}")
    print(f"TOTAL MODEL:                   {total_params:>12,} params  ({total_params*4/1e6:.1f} MB)")
    print(f"{'=' * 70}")
    
    print("\n💾 MEMORY BREAKDOWN (fp16 training, batch_size=192):")
    
    # Model weights
    frozen_mem_fp32 = frozen_params * 4 / 1e9  # CLIP stays fp32
    trainable_mem_fp16 = trainable_params * 2 / 1e9  # Trainable in fp16
    
    # Optimizer states (AdamW: 2 states per param)
    optimizer_mem = trainable_params * 2 * 4 / 1e9  # States in fp32
    
    # Gradients (fp16)
    gradient_mem = trainable_params * 2 / 1e9
    
    # Activations (estimated for batch_size=192)
    # CLIP: frozen, no activations saved
    # Custom ViT: 6 layers × 768 hidden × 196 patches × 192 batch × 2 bytes
    custom_vit_act = 6 * 768 * 196 * 192 * 2 / 1e9
    # N2N: 3 layers
    n2n_act = 3 * 768 * 196 * 192 * 2 / 1e9
    # TRM: 2 layers × 3 L-cycles × 2 H-cycles = 12 passes
    trm_act = 12 * 768 * 196 * 192 * 2 / 1e9
    # COCONUT: 4 paths
    coconut_act = 4 * 768 * 12 * 192 * 2 / 1e9  # 12 capsules
    
    total_activations = custom_vit_act + n2n_act + trm_act + coconut_act
    
    total_mem = (frozen_mem_fp32 + trainable_mem_fp16 + 
                 optimizer_mem + gradient_mem + total_activations)
    
    print(f"   Model weights (frozen, fp32):     {frozen_mem_fp32:.2f} GB")
    print(f"   Model weights (trainable, fp16):  {trainable_mem_fp16:.2f} GB")
    print(f"   Optimizer states (AdamW, fp32):   {optimizer_mem:.2f} GB")
    print(f"   Gradients (fp16):                 {gradient_mem:.2f} GB")
    print(f"   Activations (batch=192, fp16):    {total_activations:.2f} GB")
    print(f"   {'─' * 68}")
    print(f"   TOTAL GPU MEMORY:                 {total_mem:.2f} GB")
    
    print(f"\n✅ Fits in T4 (15GB VRAM): {'YES' if total_mem < 15 else 'NO'}")
    print(f"   Utilization: {total_mem/15*100:.1f}%")
    
    print("\n💿 CHECKPOINT SIZES:")
    # Full checkpoint (fp32)
    full_ckpt = total_params * 4 / 1e9
    # Trainable only (fp16)
    trainable_ckpt = trainable_params * 2 / 1e9
    # With optimizer states
    ckpt_with_opt = (trainable_params * 2 + trainable_params * 2 * 4) / 1e9
    
    print(f"   Full model (fp32):                {full_ckpt:.2f} GB")
    print(f"   Trainable only (fp16):            {trainable_ckpt:.2f} GB")
    print(f"   Trainable + optimizer (fp32):     {ckpt_with_opt:.2f} GB")
    
    print("\n🎯 DEPLOYMENT SIZES (after training):")
    # ONNX export (fp16)
    onnx_size = total_params * 2 / 1e9
    # Quantized int8
    int8_size = total_params * 1 / 1e9
    # BNN (partial - CLIP stays fp16, rest binary)
    bnn_size = (frozen_params * 2 + trainable_params * 0.125) / 1e9
    
    print(f"   ONNX (fp16):                      {onnx_size:.2f} GB")
    print(f"   Quantized (int8):                 {int8_size:.2f} GB")
    print(f"   BNN (CLIP fp16, rest 1-bit):      {bnn_size:.2f} GB")
    
    return {
        'frozen': frozen_params,
        'trainable': trainable_params,
        'total': total_params,
        'memory_gb': total_mem
    }

if __name__ == "__main__":
    calculate_total()

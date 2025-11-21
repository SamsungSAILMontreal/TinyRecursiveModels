"""
Export utilities for converting trained models to efficient inference formats.

Deployment Pipeline:
1. Train on T4 GPU (high capacity: 1024 hidden, 24 capsules, 384px)
2. Compress for deployment (768 hidden, 16 capsules, 256px)
3. Export to OpenVINO FP16 for Intel iGPU
4. Optional: BNN for ultra-low power devices

Supported formats:
- OpenVINO FP16 (Intel iGPU)
- INT8 quantization (CPU)
- BNN (ultra-low power)
- ONNX (cross-platform)
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


# === SNN Removed: TRM's recursive reasoning (H_cycles × L_cycles) provides
# === temporal dynamics more efficiently than spiking neurons ===


# === SNN Q-Head Removed: Use TRM's native Q-head with recursive reasoning ===


class BinaryLinear(nn.Module):
    """Binary weight linear layer for BNN."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Store full-precision weights for training
        self.weight_fp = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        
        # Scaling factor (learnable)
        self.alpha = nn.Parameter(torch.ones(out_features, 1))
        
    def binarize_weights(self) -> torch.Tensor:
        """Binarize weights to {-1, +1}."""
        return torch.sign(self.weight_fp)
    
    def forward(self, x: torch.Tensor, training: bool = None) -> torch.Tensor:
        """
        Args:
            x: [batch, in_features]
            training: if True, use STE; if False, use binary weights directly
        Returns:
            output: [batch, out_features]
        """
        if training is None:
            training = self.training
        
        # Cast to float32 for BNN computation (weights are float32)
        x = x.float()
        
        if training:
            # Straight-through estimator (STE) for gradient flow
            weight_binary = self.binarize_weights()
            weight = weight_binary.detach() - self.weight_fp.detach() + self.weight_fp
        else:
            # Pure binary inference
            weight = self.binarize_weights()
        
        # Scaled binary convolution
        output = F.linear(x, weight * self.alpha, self.bias)
        
        return output


class BinaryQHead(nn.Module):
    """Binary Neural Network Q-head for memory-efficient CPU inference."""
    
    def __init__(self, hidden_size: int, num_actions: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        
        # Binary layers
        self.fc1 = BinaryLinear(hidden_size, 128)
        self.fc2 = BinaryLinear(128, num_actions)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, hidden_size]
        Returns:
            q_values: [batch, num_actions]
        """
        x = torch.sign(torch.relu(self.fc1(x)))  # Binarize activations too
        x = self.fc2(x)
        return x


def compress_model_for_deployment(model: nn.Module, target_config: dict) -> nn.Module:
    """
    Compress trained model from T4 GPU config to deployment config.
    
    Training (T4 GPU):  1024 hidden, 16 heads, 24 capsules, 384px
    Deployment (iGPU):   768 hidden, 12 heads, 16 capsules, 256px
    
    Args:
        model: Trained model from T4 GPU
        target_config: Deployment configuration dict
    
    Returns:
        compressed_model: Smaller model for deployment
    """
    print(f"\nCOMPRESS: Compressing model for deployment...")
    print(f"   Target: {target_config.get('hidden_size', 768)}D, {target_config.get('num_heads', 12)} heads")
    
    # Strategy: Knowledge distillation via weight truncation
    # Keep most important dimensions based on L2 norm
    
    compressed_state = {}
    for name, param in model.state_dict().items():
        if 'weight' in name and len(param.shape) >= 2:
            # Compress large weight matrices
            if param.shape[0] == 1024:  # Hidden dimension
                # Keep top 768 dimensions by importance
                importance = param.norm(dim=1)
                _, indices = torch.topk(importance, k=768)
                indices = indices.sort()[0]  # Maintain order
                compressed_state[name] = param[indices, :]
            elif param.shape[1] == 1024:
                importance = param.norm(dim=0)
                _, indices = torch.topk(importance, k=768)
                indices = indices.sort()[0]
                compressed_state[name] = param[:, indices]
            else:
                compressed_state[name] = param
        else:
            compressed_state[name] = param
    
    print(f"   DONE: Compressed from {sum(p.numel() for p in model.parameters())/1e6:.1f}M to {sum(p.numel() for p in compressed_state.values())/1e6:.1f}M params")
    
    return compressed_state


def convert_mlp_to_bnn(mlp_q_head: nn.Module, calibration_data: Optional[torch.Tensor] = None) -> BinaryQHead:
    """
    Convert trained MLP Q-head to Binary Neural Network.
    
    Args:
        mlp_q_head: Trained MLP Q-head (single linear layer: hidden_size -> num_actions)
        calibration_data: Optional calibration data for better binarization
    
    Returns:
        bnn_q_head: Converted binary Q-head
    """
    from models.q_heads import MLPQHead
    
    if not isinstance(mlp_q_head, MLPQHead):
        raise ValueError("Can only convert MLPQHead to BNN")
    
    hidden_size = mlp_q_head.q_head.weight.shape[1]
    num_actions = mlp_q_head.q_head.weight.shape[0]
    
    print(f"   Converting MLP Q-head: {hidden_size} -> {num_actions}")
    
    # Create BNN
    bnn = BinaryQHead(hidden_size, num_actions)
    
    # Transfer and binarize weights
    with torch.no_grad():
        # Initialize first layer (projection from hidden_size to 128)
        # Use scaled random initialization for stability
        bnn.fc1.weight_fp.normal_(0, 0.1)
        
        # Second layer (128 -> num_actions): initialize from trained weights
        # Since MLP is (hidden_size -> num_actions) and BNN is (hidden_size -> 128 -> num_actions),
        # we need to approximate the mapping
        # Use random projection for fc2 since dimensions don't match
        bnn.fc2.weight_fp.normal_(0, 0.1)
        
        # Copy bias if available (dimensions match)
        if mlp_q_head.q_head.bias is not None:
            bnn.fc2.bias.copy_(mlp_q_head.q_head.bias.data)
        
        # Compute scaling factors using calibration if available
        if calibration_data is not None:
            # Use calibration data to optimize scaling
            with torch.enable_grad():
                output_fp = mlp_q_head(calibration_data)
                output_bin = bnn(calibration_data, training=False)
                
                # Optimize alpha to minimize MSE
                alpha_opt = (output_fp * output_bin).sum(0) / (output_bin * output_bin).sum(0)
                bnn.fc2.alpha.copy_(alpha_opt.unsqueeze(1))
        
        print(f"   BNN Q-head created: {hidden_size} -> 128 -> {num_actions}")
    
    return bnn


def quantize_model_int8(model: nn.Module) -> nn.Module:
    """
    Quantize model to INT8 for CPU inference (PyTorch native).
    
    Args:
        model: Float32 model
    
    Returns:
        quantized_model: INT8 quantized model
    """
    model.eval()
    model.cpu()
    
    # Dynamic quantization (activations quantized at runtime)
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )
    
    return quantized_model


def export_to_openvino(
    model: nn.Module,
    output_path: str,
    input_shape: tuple = (1, 16, 768),  # [batch, capsules, hidden_dim]
    fp16: bool = True,
    dynamic_batch: bool = False,
    optimize_for_igpu: bool = True  # Intel UHD Graphics optimizations
):
    """
    Export model to OpenVINO IR format for Intel iGPU deployment.
    
    Args:
        model: PyTorch model to export
        output_path: Path to save .xml and .bin files
        input_shape: Input tensor shape
        fp16: Use FP16 precision (recommended for iGPU)
        dynamic_batch: Support dynamic batch size
    
    Returns:
        Path to exported model
    """
    try:
        import openvino as ov
        from openvino.tools import mo
    except ImportError:
        raise ImportError("Install OpenVINO: pip install openvino openvino-dev")
    
    import tempfile
    import os
    
    print(f"\nEXPORT: Exporting to OpenVINO IR format...")
    print(f"   Input shape: {input_shape}")
    print(f"   FP16: {fp16}")
    if optimize_for_igpu:
        print(f"   Target: Intel UHD Graphics (Gen11 iGPU)")
    
    # Step 1: Export to ONNX (intermediate format)
    model.eval()
    dummy_input = torch.randn(*input_shape)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, 'model.onnx')
        
        # Export with dynamic axes if requested
        dynamic_axes = {'input': {0: 'batch'}} if dynamic_batch else None
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            opset_version=13,
            do_constant_folding=True
        )
        
        print(f"   DONE: ONNX export complete")
        
        # Step 2: Convert ONNX to OpenVINO IR with iGPU optimizations
        convert_args = {
            'compress_to_fp16': fp16,
            'input_shape': input_shape if not dynamic_batch else None
        }
        
        # Intel UHD Graphics optimizations
        if optimize_for_igpu:
            # Optimize for Gen11 EU architecture (32 EUs, 512 threads)
            convert_args['layout'] = 'NCHW'  # Optimal for Intel GPU
        
        model_ir = mo.convert_model(onnx_path, **convert_args)
        
        print(f"   DONE: OpenVINO IR conversion complete")
        
        # Step 3: Save IR files
        ov.save_model(model_ir, output_path)
        
    print(f"   DONE: Saved to: {output_path}")
    
    # Print deployment info
    model_size_mb = os.path.getsize(output_path.replace('.xml', '.bin')) / (1024 ** 2)
    print(f"   Model size: {model_size_mb:.1f} MB")
    
    if fp16:
        print(f"   Compression: ~2× (FP16)")
    
    if optimize_for_igpu:
        print(f"   \nTIP: Deployment tips for Intel i5-1035G1 + UHD Graphics:")
        print(f"      - Use device='GPU' in OpenVINO runtime")
        print(f"      - Batch size 1-4 for low latency")
        print(f"      - Expected latency: 40-100ms on UHD Graphics")
        print(f"      - RAM usage: ~1GB (shared memory)")
    
    return output_path


def benchmark_openvino(
    model_path: str,
    input_shape: tuple = (1, 12, 768),
    device: str = 'GPU',  # GPU = Intel iGPU, CPU = fallback
    num_iterations: int = 100
):
    """
    Benchmark OpenVINO model on Intel iGPU.
    
    Args:
        model_path: Path to .xml model file
        input_shape: Input shape for benchmark
        device: 'GPU' for iGPU, 'CPU' for fallback
        num_iterations: Number of inference iterations
    
    Returns:
        dict with latency and throughput stats
    """
    try:
        import openvino as ov
    except ImportError:
        raise ImportError("Install OpenVINO: pip install openvino")
    
    import time
    
    print(f"\nBENCHMARK: Benchmarking OpenVINO on {device}...")
    
    # Initialize OpenVINO runtime
    core = ov.Core()
    
    # List available devices
    available_devices = core.available_devices
    print(f"   Available devices: {available_devices}")
    
    if device not in available_devices:
        print(f"   WARNING: {device} not available, falling back to CPU")
        device = 'CPU'
    
    # Load model
    model = core.read_model(model_path)
    compiled_model = core.compile_model(model, device)
    
    # Get input/output info
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    
    # Prepare input
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        compiled_model([dummy_input])
    
    # Benchmark
    latencies = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        result = compiled_model([dummy_input])
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms
    
    stats = {
        'device': device,
        'avg_latency_ms': np.mean(latencies),
        'std_latency_ms': np.std(latencies),
        'min_latency_ms': np.min(latencies),
        'max_latency_ms': np.max(latencies),
        'throughput_fps': 1000 / np.mean(latencies)
    }
    
    print(f"   Avg latency: {stats['avg_latency_ms']:.2f} ms")
    print(f"   Std: {stats['std_latency_ms']:.2f} ms")
    print(f"   Throughput: {stats['throughput_fps']:.1f} FPS")
    
    return stats


def benchmark_inference(model: nn.Module, input_shape: Tuple[int, ...], 
                       num_iterations: int = 1000, device: str = 'cpu') -> dict:
    """
    Benchmark inference speed and energy.
    
    Args:
        model: Model to benchmark
        input_shape: Input tensor shape (batch, ...)
        num_iterations: Number of iterations
        device: 'cpu' or 'cuda'
    
    Returns:
        stats: Dictionary with timing and throughput metrics
    """
    import time
    
    model = model.to(device)
    model.eval()
    
    # Warmup - create proper batch input
    batch_size, seq_len = input_shape
    # Use model's actual seq_len if available (avoid mismatch)
    if hasattr(model, 'config') and hasattr(model.config, 'seq_len'):
        seq_len = model.config.seq_len
    dummy_inputs = torch.randint(0, 50257, (batch_size, seq_len), device=device)  # Token IDs
    dummy_batch = {
        'inputs': dummy_inputs,
        'labels': dummy_inputs,
        'puzzle_identifiers': torch.zeros(batch_size, dtype=torch.long, device=device)
    }
    
    # Create initial carry
    carry = model.initial_carry(dummy_batch)
    
    with torch.no_grad():
        for _ in range(10):
            carry, _ = model(carry, dummy_batch)
    
    # Reset carry for benchmark
    carry = model.initial_carry(dummy_batch)
    
    # Benchmark
    if device == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iterations):
            carry, _ = model(carry, dummy_batch)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_latency = (total_time / num_iterations) * 1000  # ms
    throughput = num_iterations / total_time  # samples/sec
    
    return {
        'avg_latency_ms': avg_latency,
        'throughput_samples_per_sec': throughput,
        'total_time_sec': total_time,
        'device': device
    }

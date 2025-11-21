"""
CPU Export Script - Uses existing export_utils.py infrastructure

Exports trained TRM model for i5-1035G1 CPU inference:
- Target: <4GB model size, 10 tokens/sec
- Uses existing INT8/BNN/SNN quantization from export_utils.py
- Integrates with MTP-enhanced model
"""

import torch
import sys
from pathlib import Path

print("WARNING: This script is deprecated. Please use 'scripts/export_model.py' instead.")
print("Example: python scripts/export_model.py --model checkpoint.pt --output model.pt --target cpu --quantization int8")


# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.export_utils import quantize_model_int8, convert_mlp_to_bnn, benchmark_inference

# Import ModelLoader from same directory
try:
    from load_model import ModelLoader
except ImportError:
    # If running as module
    from scripts.load_model import ModelLoader


def export_cpu_optimized(
    model_path: str,
    output_path: str = 'model_cpu_optimized.pt',
    quantization: str = 'int8'  # 'int8', 'bnn', or 'dynamic'
):
    """
    Export model for CPU using existing infrastructure.
    
    Args:
        model_path: Path to trained model checkpoint
        output_path: Where to save optimized model
        quantization: Type of quantization to apply
    """
    print(f"Loading model from {model_path}...")
    
    # Load model using ModelLoader
    loader = ModelLoader(model_path, device='cpu')
    model = loader.load_model(quantized=False)
    
    print(f"\nApplying {quantization} quantization...")
    
    if quantization == 'int8':
        # Use existing INT8 quantization
        optimized_model = quantize_model_int8(model)
        compression_ratio = 4.0
        
    elif quantization == 'bnn':
        # Convert Q-head to BNN (32× memory reduction)
        q_head_bnn = convert_mlp_to_bnn(model.inner.q_head)
        model.inner.q_head = q_head_bnn
        # Apply INT8 to rest of model
        optimized_model = quantize_model_int8(model)
        compression_ratio = 6.0  # Combined INT8 + BNN
        
    elif quantization == 'dynamic':
        # PyTorch dynamic quantization
        optimized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU},
            dtype=torch.qint8
        )
        compression_ratio = 4.0
    
    else:
        raise ValueError(f"Unknown quantization: {quantization}")
    
    # Calculate model size
    total_params = sum(p.numel() for p in optimized_model.parameters())
    model_size_mb = (total_params * 4 / compression_ratio) / (1024 ** 2)
    model_size_gb = model_size_mb / 1024
    
    print(f"\nModel Statistics:")
    print(f"  Parameters: {total_params:,}")
    print(f"  Size: {model_size_gb:.2f} GB")
    print(f"  Compression: {compression_ratio}×")
    print(f"  Target met: {'✅' if model_size_gb < 4.0 else '❌'} (<4GB)")
    
    # Benchmark on CPU
    print(f"\nBenchmarking on CPU...")
    dummy_input = torch.randn(1, 512)  # Adjust to your model's input
    stats = benchmark_inference(optimized_model, (1, 512), device='cpu', num_iterations=100)
    
    throughput_tps = 32 / (stats['avg_latency_ms'] / 1000)  # Assuming 32 token sequence
    print(f"  Latency: {stats['avg_latency_ms']:.1f} ms")
    print(f"  Throughput: {throughput_tps:.1f} tokens/sec")
    print(f"  Target met: {'✅' if throughput_tps >= 10 else '❌'} (≥10 t/s)")
    
    # Save optimized model with config
    checkpoint = {
        'model_state_dict': optimized_model.state_dict(),
        'config': loader.config  # Include config for loading
    }
    torch.save(checkpoint, output_path)
    print(f"\nSaved to: {output_path}")
    
    # Save deployment info
    deployment_info = {
        'model_size_gb': model_size_gb,
        'compression_ratio': compression_ratio,
        'quantization': quantization,
        'throughput_tokens_per_sec': throughput_tps,
        'latency_ms': stats['avg_latency_ms'],
        'cpu_settings': {
            'threads': 4,  # i5-1035G1 has 4 cores
            'mkldnn': True,
            'batch_size': 1,  # Optimal for latency
        }
    }
    
    import json
    with open(output_path.replace('.pt', '_info.json'), 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    return optimized_model, deployment_info


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Export TRM model for CPU inference')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--output', type=str, default='model_cpu.pt', help='Output path')
    parser.add_argument('--quant', type=str, default='int8', choices=['int8', 'bnn', 'dynamic'],
                       help='Quantization type')
    
    args = parser.parse_args()
    
    # CPU optimizations for i5-1035G1
    torch.set_num_threads(4)
    torch.backends.mkldnn.enabled = True
    
    export_cpu_optimized(args.model, args.output, args.quant)

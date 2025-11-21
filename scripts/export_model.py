import argparse
import torch
import sys
import os
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import model components
try:
    from models.bitnet import BitNetTransformer, BitLinear
    from models.export_utils import quantize_model_int8, convert_mlp_to_bnn, benchmark_inference
    from scripts.load_model import ModelLoader
    from safetensors.torch import save_file as safe_save_file
except ImportError as e:

    print(f"Warning: Some imports failed ({e}). Ensure you are running from project root.")

def export_bitnet(model, output_path):
    """Export BitNet model to JSON/GGUF-compatible format."""
    print(f"Exporting BitNet model to {output_path}...")
    export_data = {}
    
    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            # Get weights
            w = module.weight.detach()
            # Calculate scale
            w_scale = w.abs().mean().clamp_(min=1e-5)
            # Quantize to {-1, 0, 1}
            w_quant = (w / w_scale).round().clamp_(-1, 1)
            
            export_data[name] = {
                "weight_ternary": w_quant.cpu().numpy().tolist(),
                "weight_scale": w_scale.item(),
                "bias": module.bias.detach().cpu().numpy().tolist() if module.bias is not None else None,
                "in_features": module.in_features,
                "out_features": module.out_features
            }
            
    with open(output_path, 'w') as f:
        json.dump(export_data, f)
    print("BitNet export complete.")

def export_cpu(model, output_path, quantization, loader_config):
    """Export optimized CPU model (INT8/BNN)."""
    print(f"Exporting CPU model ({quantization}) to {output_path}...")
    
    if quantization == 'int8':
        optimized_model = quantize_model_int8(model)
        compression_ratio = 4.0
    elif quantization == 'bnn':
        # Assuming model has 'inner.q_head' structure from TRM
        if hasattr(model, 'inner') and hasattr(model.inner, 'q_head'):
            model.inner.q_head = convert_mlp_to_bnn(model.inner.q_head)
        optimized_model = quantize_model_int8(model)
        compression_ratio = 6.0
    elif quantization == 'dynamic':
        optimized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU}, dtype=torch.qint8
        )
        compression_ratio = 4.0
    else:
        optimized_model = model
        compression_ratio = 1.0

    # Save
    checkpoint = {
        'model_state_dict': optimized_model.state_dict(),
        'config': loader_config
    }
    
    if str(output_path).endswith('.safetensors'):
        # For safetensors, we can only save tensors. Config must be separate or flattened.
        # Saving config as a separate json is cleaner.
        config_path = str(output_path).replace('.safetensors', '.json')
        with open(config_path, 'w') as f:
            json.dump(loader_config, f, indent=2)
        safe_save_file(optimized_model.state_dict(), output_path)
        print(f"Saved model to {output_path} and config to {config_path}")
    else:
        torch.save(checkpoint, output_path)

    
    # Benchmark
    print("Benchmarking...")
    try:
        stats = benchmark_inference(optimized_model, (1, 512), device='cpu', num_iterations=50)
        print(f"Latency: {stats['avg_latency_ms']:.2f} ms")
    except Exception as e:
        print(f"Benchmark failed: {e}")

    print("CPU export complete.")

def main():
    parser = argparse.ArgumentParser(description="Unified Model Export Tool")
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, required=True, help='Output path')
    parser.add_argument('--target', type=str, default='cpu', choices=['cpu', 'cuda', 'bitnet'], help='Target platform')
    parser.add_argument('--quantization', type=str, default='int8', choices=['int8', 'bnn', 'dynamic', 'none'], help='Quantization type (for CPU)')
    parser.add_argument('--format', type=str, default='safetensors', choices=['pt', 'json', 'safetensors'], help='Output format')
    
    args = parser.parse_args()

    
    print(f"Loading model from {args.model}...")
    
    # Load model
    if args.target == 'bitnet':
        # BitNet loading logic (simplified)
        # In reality, we'd load the config to know the architecture
        print("Loading as BitNetTransformer...")
        # Dummy load for now as we don't have a BitNetLoader yet
        # Assuming the checkpoint contains args to build the model
        try:
            checkpoint = torch.load(args.model, map_location='cpu')
            # Reconstruct model from config if available, else use defaults/dummy
            # For this task, we assume the user provides a valid checkpoint or we use a dummy for structure
            model = BitNetTransformer(vocab_size=1000, d_model=256, nhead=4, num_layers=2)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"Error loading BitNet checkpoint: {e}. Creating dummy model.")
            model = BitNetTransformer(vocab_size=1000, d_model=256, nhead=4, num_layers=2)
            
        export_bitnet(model, args.output)
        
    else:
        # Standard TRM loading
        loader = ModelLoader(args.model, device='cpu') # Load on CPU first
        model = loader.load_model(quantized=False)
        
        if args.target == 'cpu':
            export_cpu(model, args.output, args.quantization, loader.config)
        elif args.target == 'cuda':
            print("Exporting for CUDA (no quantization applied)...")
        elif args.target == 'cuda':
            print("Exporting for CUDA (no quantization applied)...")
            if args.format == 'safetensors' or args.output.endswith('.safetensors'):
                 safe_save_file(model.state_dict(), args.output)
                 # Save config separately
                 config_path = str(args.output).replace('.safetensors', '.json')
                 with open(config_path, 'w') as f:
                    json.dump(loader.config, f, indent=2)
            else:
                torch.save({'model_state_dict': model.state_dict(), 'config': loader.config}, args.output)

            
    print(f"Done. Saved to {args.output}")

if __name__ == "__main__":
    main()

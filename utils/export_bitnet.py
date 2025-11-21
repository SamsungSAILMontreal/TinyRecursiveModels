import torch
import os
import sys
import argparse
import json

print("WARNING: This script is deprecated. Please use 'scripts/export_model.py' instead.")
print("Example: python scripts/export_model.py --model checkpoint.pt --output model.json --target bitnet")


# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.bitnet import BitNetTransformer, BitLinear

def export_bitnet_model(model_path, output_path):
    """
    Export BitNet model weights to a simplified format for C++ inference.
    
    Extracts:
    - Ternary weights {-1, 0, 1}
    - Weight scales
    - Biases (if any)
    """
    print(f"Loading model from {model_path}...")
    # In a real scenario, we'd load the full checkpoint. 
    # For now, we assume model_path is just a state_dict or we instantiate a dummy one.
    
    # Dummy instantiation for demonstration if file doesn't exist
    if not os.path.exists(model_path):
        print("Model path not found, creating dummy model for export demonstration.")
        model = BitNetTransformer(vocab_size=1000, d_model=256, nhead=4, num_layers=2)
    else:
        # Load actual model
        # model = ...
        pass
        
    export_data = {}
    
    print("Exporting layers...")
    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            print(f"Processing BitLinear: {name}")
            
            # Get weights
            w = module.weight.detach()
            
            # Calculate scale (same logic as forward pass)
            w_scale = w.abs().mean().clamp_(min=1e-5)
            
            # Quantize to {-1, 0, 1}
            w_quant = (w / w_scale).round().clamp_(-1, 1)
            
            # Store in export dict
            export_data[name] = {
                "weight_ternary": w_quant.cpu().numpy().tolist(), # Convert to list for JSON/MsgPack
                "weight_scale": w_scale.item(),
                "bias": module.bias.detach().cpu().numpy().tolist() if module.bias is not None else None,
                "in_features": module.in_features,
                "out_features": module.out_features
            }
            
    # Save to file
    # Using JSON for readability, but binary format (like GGUF) would be better for real use
    print(f"Saving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(export_data, f)
        
    print("Export complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export BitNet model")
    parser.add_argument("--model_path", type=str, default="bitnet_checkpoint.pt", help="Path to model checkpoint")
    parser.add_argument("--output_path", type=str, default="bitnet_exported.json", help="Output file path")
    args = parser.parse_args()
    
    export_bitnet_model(args.model_path, args.output_path)

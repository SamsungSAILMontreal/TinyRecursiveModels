"""
BNN Model Loader - Load quantized models for CPU inference

Based on PyTorch best practices:
https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
"""

import torch
import sys
from pathlib import Path
from typing import Dict, Optional
import json
import json
import yaml
try:
    from safetensors.torch import load_file as safe_load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False


# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1


class ModelLoader:
    """Load and manage TRM models with BNN/INT8 quantization."""
    
    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        """
        Initialize model loader.
        
        Args:
            checkpoint_path: Path to checkpoint (.pt file)
            device: 'cpu' or 'cuda'
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.model = None
        self.config = None
        self.metadata = None
        
        # CPU optimizations
        if device == 'cpu':
            torch.set_num_threads(4)  # i5-1035G1 has 4 cores
            torch.backends.mkldnn.enabled = True
    
    def load_checkpoint(self) -> Dict:
        """
        Load checkpoint file.
        
        Returns:
            checkpoint: Dictionary with model state, config, etc.
        """
        print(f"Loading checkpoint from: {self.checkpoint_path}")
        
        # Load with weights_only=True for security (PyTorch best practice)
        try:
            checkpoint = torch.load(
                self.checkpoint_path, 
                map_location=self.device,
                weights_only=False  # Need False for full checkpoint with config
            )
        except Exception as e:
            # Try safetensors if available and file extension matches or fallback
            if SAFETENSORS_AVAILABLE and str(self.checkpoint_path).endswith('.safetensors'):
                print(f"Loading safetensors from {self.checkpoint_path}")
                return safe_load_file(self.checkpoint_path)
                
            print(f"Warning: Failed to load with weights_only=False, trying pickle_module...")

            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device
            )
        
        return checkpoint
    
    def extract_config(self, checkpoint: Dict) -> Dict:
        """
        Extract model configuration from checkpoint.
        
        Args:
            checkpoint: Loaded checkpoint dictionary
            
        Returns:
            config: Model configuration dict
        """
        # Try different checkpoint formats
        if 'config' in checkpoint:
            config = checkpoint['config']
        elif 'model_config' in checkpoint:
            config = checkpoint['model_config']
        elif 'arch' in checkpoint:
            # Hydra-style config
            config = checkpoint['arch']
        else:
            # Config not in checkpoint, try loading from all_config.yaml in same directory or parent
            config_path = self.checkpoint_path.parent / 'all_config.yaml'
            if not config_path.exists():
                config_path = self.checkpoint_path.parent.parent / 'all_config.yaml'
            
            if config_path.exists():
                print(f"   Loading config from: {config_path}")
                with open(config_path, 'r') as f:
                    full_config = yaml.safe_load(f)
                # Extract arch section (model config)
                if 'arch' in full_config:
                    config = full_config['arch']
                else:
                    config = full_config
            else:
                raise ValueError(
                    f"No config found in checkpoint or {config_path}.\n"
                    f"Available checkpoint keys: {list(checkpoint.keys())[:5]}..."
                )
        
        # Convert OmegaConf to dict if needed
        if hasattr(config, 'to_dict'):
            config = config.to_dict()
        elif hasattr(config, '__dict__'):
            config = vars(config)
        
        return config
    
    def load_model(self, quantized: bool = False) -> TinyRecursiveReasoningModel_ACTV1:
        """
        Load model from checkpoint.
        
        Args:
            quantized: Whether this is a quantized checkpoint (BNN/INT8)
            
        Returns:
            model: Loaded model ready for inference
        """
        # Load checkpoint
        checkpoint = self.load_checkpoint()
        
        # Extract config
        self.config = self.extract_config(checkpoint)
        
        # Inject defaults for missing fields (required by Pydantic model but often dynamic)
        if 'seq_len' not in self.config:
            print("   Injecting default seq_len=1024")
            self.config['seq_len'] = 1024
        if 'vocab_size' not in self.config:
            print("   Injecting default vocab_size=4096")
            self.config['vocab_size'] = 4096
        if 'num_puzzle_identifiers' not in self.config:
            print("   Injecting default num_puzzle_identifiers=0")
            self.config['num_puzzle_identifiers'] = 0
        if 'input_vocab_size' not in self.config:
            self.config['input_vocab_size'] = self.config['vocab_size']
        if 'batch_size' not in self.config:
            print("   Injecting default batch_size=1")
            self.config['batch_size'] = 1
            
        print(f"\n Model Configuration:")
        print(f"   Architecture: {self.config.get('name', 'TRM')}")
        print(f"   Hidden size: {self.config.get('hidden_size')}")
        print(f"   Vocab size: {self.config.get('vocab_size')}")
        print(f"   Sequence length: {self.config.get('seq_len')}")
        print(f"   H cycles: {self.config.get('H_cycles')}")
        print(f"   L layers: {self.config.get('L_layers')}")
        
        # Create model instance
        print("\n Instantiating model...")
        self.model = TinyRecursiveReasoningModel_ACTV1(self.config)
        
        # Load state dict
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # Assume entire checkpoint is state dict
            state_dict = checkpoint
        
        # Clean up state dict keys from various training frameworks
        sample_key = next(iter(state_dict.keys()))
        print(f"   Sample key before cleanup: {sample_key}")
        
        # Remove 'module.' prefix if using DataParallel
        if any(k.startswith('module.') for k in state_dict.keys()):
            print("   Removing 'module.' prefix...")
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
        # Remove 'model.' prefix if present (e.g. from some wrappers)
        if any(k.startswith('model.') for k in state_dict.keys()):
            print("   Removing 'model.' prefix...")
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        
        # Remove '_orig_mod.model.' prefix from torch.compile
        if any(k.startswith('_orig_mod.model.') for k in state_dict.keys()):
            print("   Removing '_orig_mod.model.' prefix from torch.compile...")
            state_dict = {k.replace('_orig_mod.model.', ''): v for k, v in state_dict.items()}
        elif any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            print("   Removing '_orig_mod.' prefix...")
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        
        sample_key_after = next(iter(state_dict.keys()))
        print(f"   Sample key after cleanup: {sample_key_after}")
        
        # Load weights
        print(" Loading weights...")
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"  Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"  Unexpected keys: {unexpected_keys}")
        
        # Set to evaluation mode (CRITICAL for inference)
        self.model.eval()
        self.model = self.model.to(self.device)
        
        # Extract metadata
        self.metadata = {
            'epoch': checkpoint.get('epoch', 0),
            'step': checkpoint.get('step', 0),
            'quantized': quantized,
            'device': self.device
        }
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"\n Model loaded successfully!")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Device: {self.device}")
        
        if quantized:
            print(f"    Quantized model (BNN/INT8)")
        
        return self.model
    
    def save_model_info(self, output_path: str = "model_info.json"):
        """Save model metadata to JSON."""
        if self.model is None:
            raise ValueError("Model not loaded yet. Call load_model() first.")
        
        info = {
            'checkpoint_path': str(self.checkpoint_path),
            'config': self.config,
            'metadata': self.metadata,
            'num_parameters': sum(p.numel() for p in self.model.parameters())
        }
        
        with open(output_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f" Model info saved to: {output_path}")


def load_bnn_model(checkpoint_path: str, device: str = 'cpu') -> TinyRecursiveReasoningModel_ACTV1:
    """
    Convenience function to load BNN quantized model.
    
    Args:
        checkpoint_path: Path to BNN checkpoint
        device: 'cpu' or 'cuda'
        
    Returns:
        model: Loaded model in eval mode
    """
    loader = ModelLoader(checkpoint_path, device)
    model = loader.load_model(quantized=True)
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Load TRM model with BNN quantization')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to checkpoint file')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device to load model on')
    parser.add_argument('--quantized', action='store_true',
                       help='Whether checkpoint is quantized (BNN/INT8)')
    parser.add_argument('--save-info', type=str, default=None,
                       help='Path to save model info JSON')
    
    args = parser.parse_args()
    
    # Load model
    loader = ModelLoader(args.checkpoint, args.device)
    model = loader.load_model(quantized=args.quantized)
    
    # Save info if requested
    if args.save_info:
        loader.save_model_info(args.save_info)
    
    print("\n Model ready for inference!")

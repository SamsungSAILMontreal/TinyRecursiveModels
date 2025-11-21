"""
Unified Model Testing Suite
Tests TRM model forward pass, COCONUT integration, and configuration loading
"""

import torch
import sys
from pathlib import Path
from omegaconf import OmegaConf
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_model_forward(config_path: str, enable_memory: bool = False, verbose: bool = True):
    """
    Test model forward pass with given configuration.
    
    Args:
        config_path: Path to config YAML file
        enable_memory: Enable memory bank for testing
        verbose: Print detailed output
    
    Returns:
        bool: True if test passes
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Testing model: {config_path}")
        print(f"{'='*70}\n")
    
    # Load config
    config = OmegaConf.load(config_path)
    config_dict = OmegaConf.to_container(config, resolve=True)
    config_dict.update({
        'batch_size': 2,
        'seq_len': 12,
        'num_puzzle_identifiers': 100,
        'vocab_size': 2052,
        'enable_memory': enable_memory
    })
    
    # Build model
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
    model = TinyRecursiveReasoningModel_ACTV1(config_dict)
    model.eval()
    
    # Count params
    total = sum(p.numel() for p in model.parameters())
    
    if verbose:
        print(f"Model: {total:,} parameters")
        
        # Check for optional modules
        if model.inner.latent_planner:
            latent = sum(p.numel() for p in model.inner.latent_planner.parameters())
            print(f"  └─ COCONUT Latent Planning: {latent:,} params ({latent/total*100:.1f}%)")
        
        if model.inner.memory:
            memory = sum(p.numel() for p in model.inner.memory.parameters())
            print(f"  └─ Memory Bank: {memory:,} params ({memory/total*100:.1f}%)")
        
        if model.inner.mtp_modules:
            mtp = sum(p.numel() for m in model.inner.mtp_modules for p in m.parameters())
            print(f"  └─ Multi-Token Prediction: {mtp:,} params ({mtp/total*100:.1f}%)")
        
        print()
    
    # Create test batch (vision-unified mode with capsules)
    batch = {
        'inputs': torch.randn(2, 12, 512, dtype=torch.float16),
        'checksums': torch.randn(2, 12, 32, dtype=torch.float16),
        'children': torch.randn(2, 12, 4, 512, dtype=torch.float16),
        'puzzle_identifiers': torch.zeros(2, dtype=torch.long),
        'labels': torch.randint(0, 2048, (2, 12)),
    }
    
    # Test forward pass
    try:
        with torch.no_grad():
            carry = model.initial_carry(batch)
            new_carry, outputs = model(carry, batch)
        
        if verbose:
            print(f"✅ Forward pass successful")
            print(f"   Output shape: {outputs['logits'].shape}")
            print(f"   Q-values: {outputs.get('q_values', 'N/A')}")
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"❌ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
        return False


def test_all_configs(config_dir: str = "config/arch", verbose: bool = True):
    """Test all configuration files in directory."""
    config_path = Path(config_dir)
    configs = sorted(config_path.glob("*.yaml"))
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Testing {len(configs)} configurations")
        print(f"{'='*70}\n")
    
    results = {}
    for config_file in configs:
        config_name = config_file.stem
        try:
            # Skip experimental architectures (use different model files)
            skip_configs = ['baseline', 'transformers_baseline', 'hrm', 'trm_hier6', 'trm_singlez']
            if any(skip in config_name for skip in skip_configs):
                if verbose:
                    print(f"⏭️  Skipping {config_name} (experimental architecture)")
                continue
            
            passed = test_model_forward(
                str(config_file),
                enable_memory=False,  # Disable for speed
                verbose=False
            )
            results[config_name] = passed
            
            if verbose:
                status = "✅" if passed else "❌"
                print(f"{status} {config_name}")
                
        except Exception as e:
            results[config_name] = False
            if verbose:
                print(f"❌ {config_name}: {e}")
    
    # Summary
    passed = sum(results.values())
    total = len(results)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Results: {passed}/{total} configs passed")
        print(f"{'='*70}\n")
    
    return all(results.values())


def main():
    parser = argparse.ArgumentParser(
        description="Test TRM model configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/test.py                              # Test default config
  python scripts/test.py --config multimodal_hesc     # Test specific config
  python scripts/test.py --all                        # Test all configs
  python scripts/test.py --enable-memory              # Test with memory bank
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='multimodal_hesc',
        help='Config name (without .yaml)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Test all configurations'
    )
    
    parser.add_argument(
        '--enable-memory',
        action='store_true',
        help='Enable memory bank for testing'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Minimal output'
    )
    
    args = parser.parse_args()
    
    if args.all:
        success = test_all_configs(verbose=not args.quiet)
    else:
        config_path = f"config/arch/{args.config}.yaml"
        success = test_model_forward(
            config_path,
            enable_memory=args.enable_memory,
            verbose=not args.quiet
        )
    
    if success:
        print("\n✅ All tests passed!\n")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

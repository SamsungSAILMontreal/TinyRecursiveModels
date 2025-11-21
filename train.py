"""Simplified training wrapper - calls pretrain.py with hybrid config.

Usage:
    python train.py                              # Train with hybrid_pretrained.yaml
    python train.py --config other_config.yaml   # Train with custom config
    python train.py [any pretrain.py args]       # Pass through to pretrain.py
"""

import sys
import subprocess


# Default config for hybrid pretrained pipeline
DEFAULT_CONFIG = "config/arch/hybrid_pretrained.yaml"




def main():
    """Simple wrapper: calls pretrain.py with hybrid_pretrained config."""
    
    args = sys.argv[1:]

    # Auto-Sequencing Pipeline
    # 1. Run Pretraining (Teacher)
    print("="*70)
    print("  🚀 Phase 1: TRM Pretraining (Teacher Model)")
    print("="*70)
    
    # Build pretrain command
    # Parse config path for Hydra
    from pathlib import Path
    # Check if user specified custom config
    config = DEFAULT_CONFIG
    for i, arg in enumerate(args):
        if arg == "--config" and i + 1 < len(args):
            config = args[i + 1]
            args = args[:i] + args[i+2:]
            break
            
    config_path_obj = Path(config).resolve()
    config_dir = str(config_path_obj.parent)
    config_name = config_path_obj.stem
    
    pretrain_cmd = ["python", "pretrain.py"]
    if config_dir:
        pretrain_cmd.extend(["--config-path", config_dir])
    pretrain_cmd.extend(["--config-name", config_name])
    pretrain_cmd += args
    
    print(f"Running Pretraining: {' '.join(pretrain_cmd)}\n")
    try:
        subprocess.run(pretrain_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Pretraining failed: {e}")
        sys.exit(1)

    # 2. Run Noise2Noise (Auxiliary)
    print("\n")
    print("="*70)
    print("  🔊 Phase 2: Noise2Noise Training (Auxiliary Denoiser)")
    print("="*70)
    
    n2n_cmd = ["python", "scripts/train_noise2noise.py"]
    # Pass relevant args if needed, or just let it use defaults/global args if appropriate
    # For now, we'll pass the global args as they might contain generic flags like --device
    # But we need to be careful not to pass pretrain-specific flags. 
    # Since train_noise2noise uses argparse, it will error on unknown flags.
    # Safe bet: Run with defaults unless we parse specifically. 
    # For simplicity in this unified pipeline, we run it with defaults.
    
    print(f"Running Noise2Noise: {' '.join(n2n_cmd)}\n")
    try:
        subprocess.run(n2n_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Noise2Noise training failed: {e}")
        sys.exit(1)
        
    # 3. Run Distillation (Student)
    print("\n")
    print("="*70)
    print("  🎓 Phase 3: Distillation (Teacher -> Student)")
    print("  Transferring knowledge to smaller model...")
    print("="*70)
    
    # For now, we assume default distillation config or infer it
    # In a real scenario, we'd pass the teacher checkpoint from Phase 1
    distill_cmd = ["python", "scripts/distill.py"] + args
    print(f"Running Distillation: {' '.join(distill_cmd)}\n")
    try:
        subprocess.run(distill_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Distillation failed: {e}")
        sys.exit(1)
        
    print("\n✅ Full Pipeline Complete!")






if __name__ == "__main__":
    main()

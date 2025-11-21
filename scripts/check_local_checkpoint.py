"""
Check local checkpoint details without transferring.

Usage:
    python scripts/check_local_checkpoint.py path/to/latest.pt
"""

import sys
import torch
import os


def check_checkpoint(checkpoint_path):
    """Check local checkpoint and show all details."""
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ File not found: {checkpoint_path}")
        return
    
    file_size_mb = os.path.getsize(checkpoint_path) / 1024 / 1024
    
    print("="*70)
    print("  📊 LOCAL CHECKPOINT ANALYSIS")
    print("="*70)
    print()
    print(f"File: {checkpoint_path}")
    print(f"Size: {file_size_mb:.2f} MB")
    print()
    
    try:
        # Load checkpoint
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("✅ Checkpoint loaded successfully!\n")
        
        # Extract info
        step = checkpoint.get('step', 0)
        config_epochs = checkpoint.get('config_epochs', 'unknown')
        total_steps = checkpoint.get('total_steps', 'unknown')
        dataset = checkpoint.get('dataset', 'unknown')
        
        # Calculate epochs if possible
        if isinstance(total_steps, int) and total_steps > 0:
            progress_pct = (step / total_steps) * 100
            if isinstance(config_epochs, int):
                approx_epochs = int((step / total_steps) * config_epochs)
            else:
                approx_epochs = "unknown"
        else:
            progress_pct = 0
            approx_epochs = "unknown"
        
        print("Training Progress:")
        print(f"  Current Step: {step:,}")
        print(f"  Total Steps: {total_steps}")
        print(f"  Progress: {progress_pct:.1f}%")
        print()
        print(f"  Target Epochs: {config_epochs}")
        print(f"  Approx Current Epoch: {approx_epochs}")
        print()
        print(f"Dataset: {dataset}")
        print()
        
        # Check what's in the checkpoint
        print("Checkpoint Contents:")
        for key in checkpoint.keys():
            if key == 'model_state_dict':
                num_params = len(checkpoint[key])
                print(f"  ✓ {key} ({num_params} parameters)")
            elif key == 'optimizer_states':
                num_opts = len(checkpoint[key])
                print(f"  ✓ {key} ({num_opts} optimizers)")
            else:
                print(f"  ✓ {key}")
        print()
        
        # Status
        if isinstance(total_steps, int) and total_steps > 0:
            if step >= total_steps:
                print("Status: ✅ COMPLETE")
            else:
                remaining = total_steps - step
                print(f"Status: ⚠️  INCOMPLETE ({remaining:,} steps remaining)")
        else:
            print("Status: ⚠️  Legacy checkpoint (no metadata)")
        
        print()
        print("="*70)
        print("  ✅ This checkpoint is VALID and can be used!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ ERROR loading checkpoint: {e}")
        print("\nThis checkpoint is CORRUPTED!")
        return


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/check_local_checkpoint.py <checkpoint_path>")
        print("\nExample:")
        print("  python scripts/check_local_checkpoint.py checkpoints/text-trm-10m/latest.pt")
        print("  python scripts/check_local_checkpoint.py C:\\Users\\sao\\Downloads\\latest.pt")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    check_checkpoint(checkpoint_path)

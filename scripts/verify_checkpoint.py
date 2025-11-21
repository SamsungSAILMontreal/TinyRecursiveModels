"""
Verify checkpoint integrity after compression/transfer.

Usage:
    # Verify single checkpoint
    python scripts/verify_checkpoint.py checkpoints/text-trm-10m/latest.pt
    
    # Verify all checkpoints in directory
    python scripts/verify_checkpoint.py checkpoints/text-trm-10m/
    
    # After tar.gz compression/decompression
    tar -czf checkpoint.tar.gz checkpoints/text-trm-10m/
    # ... transfer ...
    tar -xzf checkpoint.tar.gz
    python scripts/verify_checkpoint.py checkpoints/text-trm-10m/latest.pt
"""

import sys
import os
import hashlib
import json
from pathlib import Path


def compute_file_checksum(filepath: str) -> str:
    """Compute SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def verify_checkpoint(checkpoint_path: str) -> bool:
    """Verify checkpoint integrity."""
    checksum_file = checkpoint_path + ".sha256"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    if not os.path.exists(checksum_file):
        print(f"⚠️  No checksum file for: {checkpoint_path}")
        print("   (Legacy checkpoint - cannot verify)")
        return True
    
    try:
        # Load expected checksum
        with open(checksum_file, 'r') as f:
            metadata = json.load(f)
        expected_checksum = metadata['checksum']
        expected_size = metadata.get('size_bytes', 0)
        
        # Check file size first (quick check)
        actual_size = os.path.getsize(checkpoint_path)
        if expected_size > 0 and actual_size != expected_size:
            print(f"❌ Size mismatch: {checkpoint_path}")
            print(f"   Expected: {expected_size:,} bytes")
            print(f"   Got:      {actual_size:,} bytes")
            return False
        
        # Compute actual checksum
        print(f"🔍 Verifying: {os.path.basename(checkpoint_path)}...", end=' ')
        actual_checksum = compute_file_checksum(checkpoint_path)
        
        if actual_checksum == expected_checksum:
            print(f"✅")
            print(f"   SHA256: {actual_checksum[:32]}...")
            return True
        else:
            print(f"❌ CORRUPTED!")
            print(f"   Expected: {expected_checksum[:32]}...")
            print(f"   Got:      {actual_checksum[:32]}...")
            return False
            
    except Exception as e:
        print(f"❌ Verification error: {e}")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/verify_checkpoint.py <checkpoint_path>")
        print("\nExamples:")
        print("  python scripts/verify_checkpoint.py checkpoints/text-trm-10m/latest.pt")
        print("  python scripts/verify_checkpoint.py checkpoints/text-trm-10m/")
        sys.exit(1)
    
    path = sys.argv[1]
    
    print("="*70)
    print("  🔒 CHECKPOINT INTEGRITY VERIFICATION")
    print("="*70)
    print()
    
    # Check if directory or file
    if os.path.isdir(path):
        # Verify all .pt files in directory
        checkpoints = list(Path(path).glob("*.pt"))
        if not checkpoints:
            print(f"No checkpoint files found in {path}")
            sys.exit(1)
        
        results = []
        for ckpt in checkpoints:
            results.append(verify_checkpoint(str(ckpt)))
            print()
        
        # Summary
        total = len(results)
        passed = sum(results)
        failed = total - passed
        
        print("="*70)
        print(f"  SUMMARY: {passed}/{total} passed, {failed} failed")
        print("="*70)
        
        if failed > 0:
            print("\n⚠️  Some checkpoints are corrupted!")
            print("   Do NOT use these for training - they may cause errors.")
            sys.exit(1)
        else:
            print("\n✅ All checkpoints verified successfully!")
            print("   Safe to use for training/inference.")
    else:
        # Verify single file
        success = verify_checkpoint(path)
        print()
        
        if success:
            print("✅ Checkpoint verified successfully!")
            sys.exit(0)
        else:
            print("❌ Checkpoint verification failed!")
            sys.exit(1)


if __name__ == "__main__":
    main()

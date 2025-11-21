"""
Storage Manager: Clean up redundant files and manage disk efficiently.

Problems identified:
1. Multiple checkpoint formats (.pt files) scattered everywhere
2. No automatic cleanup of intermediate files
3. Consolidated files duplicate batch files
4. ImageCache never cleans old entries
"""

import os
import glob
import shutil
from pathlib import Path
from typing import List, Dict
import json


class StorageManager:
    """
    Unified storage management for training:
    - Single source of truth for checkpoint locations
    - Automatic cleanup of intermediate files
    - Efficient compression (avoid duplicates)
    """
    
    def __init__(self, base_dir: str = "checkpoints"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Separate directories for different file types
        self.model_dir = self.base_dir / "models"
        self.dataset_dir = self.base_dir / "datasets"
        self.temp_dir = self.base_dir / "temp"
        
        for d in [self.model_dir, self.dataset_dir, self.temp_dir]:
            d.mkdir(exist_ok=True)
    
    def save_checkpoint(self, state_dict: dict, step: int, cleanup_old: bool = True):
        """
        Save checkpoint with automatic cleanup.
        
        Args:
            state_dict: Model state to save
            step: Training step
            cleanup_old: Remove previous checkpoints (keep last 3)
        """
        checkpoint_path = self.model_dir / f"checkpoint_step_{step:06d}.pt"
        
        import torch
        torch.save(state_dict, checkpoint_path)
        
        if cleanup_old:
            self._cleanup_old_checkpoints(keep=3)
        
        return checkpoint_path
    
    def _cleanup_old_checkpoints(self, keep: int = 3):
        """Keep only the N most recent checkpoints."""
        checkpoints = sorted(self.model_dir.glob("checkpoint_step_*.pt"))
        
        if len(checkpoints) > keep:
            for ckpt in checkpoints[:-keep]:
                print(f"🗑️  Removing old checkpoint: {ckpt.name}")
                ckpt.unlink()
    
    def save_dataset_batch(self, batch_data: dict, batch_id: int, temporary: bool = True):
        """
        Save dataset batch to appropriate location.
        
        Args:
            batch_data: Batch data to save
            batch_id: Batch identifier
            temporary: If True, save to temp dir (will be consolidated and deleted)
        """
        target_dir = self.temp_dir if temporary else self.dataset_dir
        batch_file = target_dir / f"batch_{batch_id:04d}.pt"
        
        import torch
        torch.save(batch_data, batch_file)
        
        return batch_file
    
    def consolidate_and_cleanup(self, batch_files: List[Path], output_name: str):
        """
        Consolidate multiple batch files into one and DELETE originals.
        
        This is the KEY fix: no duplicates after consolidation!
        """
        print(f"📦 Consolidating {len(batch_files)} batches...")
        
        import torch
        
        # Load and merge all batches
        merged = []
        for batch_file in batch_files:
            data = torch.load(batch_file)
            merged.append(data)
        
        # Save consolidated
        consolidated_path = self.dataset_dir / f"{output_name}.pt"
        torch.save(merged, consolidated_path)
        
        # DELETE temporary batch files (no duplicates!)
        for batch_file in batch_files:
            if batch_file.parent == self.temp_dir:
                batch_file.unlink()
                print(f"  🗑️  Deleted: {batch_file.name}")
        
        print(f"✅ Consolidated: {consolidated_path.name}")
        return consolidated_path
    
    def cleanup_temp_files(self):
        """Remove ALL temporary files."""
        temp_files = list(self.temp_dir.glob("*"))
        
        for f in temp_files:
            f.unlink()
            print(f"🗑️  Removed temp: {f.name}")
        
        print(f"✅ Cleaned {len(temp_files)} temporary files")
    
    def get_storage_stats(self) -> Dict[str, float]:
        """Get storage usage in GB."""
        def get_dir_size(path: Path) -> float:
            total = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            return total / (1024**3)  # Convert to GB
        
        return {
            'models_gb': get_dir_size(self.model_dir),
            'datasets_gb': get_dir_size(self.dataset_dir),
            'temp_gb': get_dir_size(self.temp_dir),
            'total_gb': get_dir_size(self.base_dir)
        }
    
    def print_storage_report(self):
        """Print storage usage report."""
        stats = self.get_storage_stats()
        
        print("\n💾 Storage Usage:")
        print(f"   Models:   {stats['models_gb']:.2f} GB")
        print(f"   Datasets: {stats['datasets_gb']:.2f} GB")
        print(f"   Temp:     {stats['temp_gb']:.2f} GB")
        print(f"   Total:    {stats['total_gb']:.2f} GB")
        
        if stats['temp_gb'] > 1.0:
            print(f"   ⚠️  Warning: {stats['temp_gb']:.2f} GB in temp (run cleanup!)")


class ImageCacheManager:
    """
    Fixed ImageCache management:
    - LRU eviction (remove old entries)
    - Size limits
    - Periodic cleanup
    """
    
    def __init__(self, cache_dir: str, max_size_gb: float = 5.0):
        self.cache_dir = Path(cache_dir)
        self.max_size_gb = max_size_gb
    
    def cleanup_old_entries(self, keep_ratio: float = 0.7):
        """
        Remove oldest 30% of cached images when size exceeds limit.
        
        Args:
            keep_ratio: Keep this fraction of newest entries (0.7 = keep 70%, remove 30%)
        """
        cache_files = sorted(
            self.cache_dir.rglob('*.npy'),
            key=lambda p: p.stat().st_mtime  # Sort by modification time
        )
        
        # Calculate total size
        total_size = sum(f.stat().st_size for f in cache_files) / (1024**3)
        
        if total_size > self.max_size_gb:
            keep_count = int(len(cache_files) * keep_ratio)
            files_to_remove = cache_files[:len(cache_files) - keep_count]
            
            print(f"🗑️  Cache cleanup: {total_size:.2f} GB > {self.max_size_gb:.2f} GB")
            print(f"   Removing {len(files_to_remove)} oldest entries...")
            
            for f in files_to_remove:
                f.unlink()
            
            new_size = sum(f.stat().st_size for f in cache_files[len(files_to_remove):]) / (1024**3)
            print(f"   ✅ Reduced to {new_size:.2f} GB")


# Global instance
_storage_manager = None

def get_storage_manager() -> StorageManager:
    """Get singleton storage manager."""
    global _storage_manager
    if _storage_manager is None:
        _storage_manager = StorageManager()
    return _storage_manager

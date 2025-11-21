"""
Standalone batch consolidation script.

Run this manually to consolidate completed batches without interrupting training.
Example: If you have 435 batches, this will consolidate batches 0-399 (4 chunks),
leaving batches 400-434 for the next consolidation.

Usage:
    python consolidate_batches.py --checkpoint-dir datasets/vision_unified/stream_checkpoints
    python consolidate_batches.py --checkpoint-dir datasets/vision_unified/stream_checkpoints --drive-dir /content/drive/MyDrive/model_checkpoints
"""

import argparse
import torch
import os
import gc
from pathlib import Path
from tqdm import tqdm


def consolidate_batches(checkpoint_dir, drive_dir=None, chunk_size=100):
    """
    Consolidate batch files into larger chunks using unified progress tracker.
    
    Args:
        checkpoint_dir: Directory containing batch_*.pt files
        drive_dir: Optional drive directory for consolidated files
        chunk_size: Number of batches per consolidated chunk (default: 100)
    """
    from dataset.training_progress import TrainingProgressTracker
    
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return
    
    # Initialize progress tracker
    dataset_name = checkpoint_dir.parent.name  # e.g., "vision_unified"
    tracker = TrainingProgressTracker(
        checkpoint_dir=checkpoint_dir.parent / "checkpoints" if (checkpoint_dir.parent / "checkpoints").exists() else checkpoint_dir,
        dataset_name=dataset_name
    )
    
    # Load existing progress or initialize
    if tracker.has_progress():
        tracker.load()
    else:
        print("ℹ️  No existing progress, initializing...")
    
    # Initialize consolidation tracking
    tracker.init_consolidation(batches_per_chunk=chunk_size)
    
    # Find existing batch files
    batch_files = sorted(checkpoint_dir.glob("batch_*.pt"))
    
    # Update total batches built
    if batch_files:
        max_batch_id = max(int(f.stem.split('_')[1]) for f in batch_files)
        tracker.record_batch_built(max_batch_id)
    
    print(f"\n{'='*70}")
    print(f"📦 BATCH CONSOLIDATION (Unified Progress Tracker)")
    print(f"{'='*70}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Drive dir: {drive_dir or 'None'}")
    print(f"Batch files found: {len(batch_files)}")
    print(f"Total batches built: {tracker.progress.total_batches_built}")
    
    if tracker.progress.consolidation_progress:
        cons = tracker.progress.consolidation_progress
        print(f"Already consolidated: {cons.total_batches_consolidated} batches in {len(cons.consolidation_mapping)} chunks")
        print(f"Last consolidated chunk: {cons.last_consolidated_chunk_id}")
    print(f"{'='*70}\n")
    
    if not batch_files:
        print("✅ No batch files to consolidate")
        return
    
    # Get batches to consolidate from tracker
    start_batch, end_batch, next_chunk_id = tracker.get_batches_to_consolidate()
    
    if end_batch == 0:
        unconsolidated = tracker.get_unconsolidated_batches()
        print(f"ℹ️  Only {len(unconsolidated)} unconsolidated batches (need {chunk_size} for a complete chunk)")
        print(f"   Continue dataset building to reach {chunk_size} batches")
        return
    
    complete_chunks = (end_batch - start_batch) // chunk_size
    
    print(f"📊 Consolidation plan:")
    print(f"   Batches to consolidate: {start_batch} - {end_batch-1}")
    print(f"   Will create {complete_chunks} consolidated chunk(s)")
    print(f"   Remaining batches: {tracker.progress.total_batches_built - end_batch}")
    print(f"")
    
    # Consolidate each chunk
    save_dir = Path(drive_dir) if drive_dir else checkpoint_dir
    if drive_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    
    for chunk_idx in range(complete_chunks):
        chunk_start = start_batch + (chunk_idx * chunk_size)
        chunk_end = chunk_start + chunk_size
        current_chunk_id = next_chunk_id + chunk_idx
        
        print(f"\n📤 Consolidating chunk {current_chunk_id} (batches {chunk_start}-{chunk_end-1})...")
        
        # Load batches in mini-chunks to avoid OOM
        mini_chunk_size = 20
        all_consolidated = {'sketches': [], 'checksums': [], 'children': []}
        corrupted_files = []
        
        batch_files_to_load = [checkpoint_dir / f"batch_{i:05d}.pt" for i in range(chunk_start, chunk_end)]
        
        for mini_start in tqdm(range(0, len(batch_files_to_load), mini_chunk_size), desc="Loading batches"):
            mini_end = min(mini_start + mini_chunk_size, len(batch_files_to_load))
            mini_batch_files = batch_files_to_load[mini_start:mini_end]
            
            chunk_data = {'sketches': [], 'checksums': [], 'children': []}
            
            for batch_file in mini_batch_files:
                if batch_file.exists():
                    try:
                        batch = torch.load(batch_file, map_location='cpu')
                        for key in chunk_data:
                            if key in batch and batch[key] is not None:
                                chunk_data[key].append(batch[key])
                        del batch
                    except Exception as e:
                        print(f"⚠️  Corrupted file: {batch_file.name}")
                        corrupted_files.append(batch_file)
                        continue
            
            # Concatenate this mini chunk
            for key in chunk_data:
                if chunk_data[key]:
                    mini_concat = torch.cat(chunk_data[key], dim=0)
                    all_consolidated[key].append(mini_concat)
                    del mini_concat
            del chunk_data
            gc.collect()
        
        # Final concatenation
        consolidated = {}
        for key in all_consolidated:
            if all_consolidated[key]:
                consolidated[key] = torch.cat(all_consolidated[key], dim=0)
        del all_consolidated
        
        # Save consolidated chunk
        consolidated_file = save_dir / f"consolidated_{current_chunk_id:03d}.pt"
        torch.save(consolidated, consolidated_file)
        del consolidated
        gc.collect()
        
        chunk_size_mb = consolidated_file.stat().st_size / 1024 / 1024
        print(f"✅ Saved {consolidated_file.name} ({chunk_size_mb:.1f}MB)")
        
        if corrupted_files:
            print(f"⚠️  Skipped {len(corrupted_files)} corrupted files")
        
        # Record consolidation in progress tracker
        batch_ids = list(range(chunk_start, chunk_end))
        tracker.record_consolidation(
            chunk_id=current_chunk_id,
            batch_ids=batch_ids
        )
        print(f"📊 Recorded consolidation: chunk {current_chunk_id} <- batches {chunk_start}-{chunk_end-1}")
        
        # Delete individual batch files to free disk space
        deleted_count = 0
        freed_mb = 0
        for batch_file in batch_files_to_load:
            if batch_file.exists() and batch_file not in corrupted_files:
                batch_size_mb = batch_file.stat().st_size / 1024 / 1024
                batch_file.unlink()
                deleted_count += 1
                freed_mb += batch_size_mb
        
        print(f"🧹 Deleted {deleted_count} batch files ({freed_mb:.1f}MB freed)")
    
    # Save progress after consolidation
    tracker.save()
    
    print(f"\n{'='*70}")
    print(f"✅ CONSOLIDATION COMPLETE")
    print(f"{'='*70}")
    print(f"Consolidated chunks: {next_chunk_id} → {next_chunk_id + complete_chunks - 1}")
    print(f"Total batches consolidated: {tracker.progress.consolidation_progress.total_batches_consolidated}")
    print(f"Remaining batches: {len(tracker.get_unconsolidated_batches())}")
    if tracker.progress.total_batches_built >= end_batch + chunk_size:
        print(f"Next consolidation ready at: batch {end_batch + chunk_size}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Consolidate batch checkpoint files")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Directory containing batch_*.pt files"
    )
    parser.add_argument(
        "--drive-dir",
        type=str,
        default=None,
        help="Optional: Drive directory to save consolidated files (default: same as checkpoint-dir)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Number of batches per consolidated chunk (default: 100)"
    )
    
    args = parser.parse_args()
    
    consolidate_batches(
        checkpoint_dir=args.checkpoint_dir,
        drive_dir=args.drive_dir,
        chunk_size=args.chunk_size
    )


if __name__ == "__main__":
    main()
    
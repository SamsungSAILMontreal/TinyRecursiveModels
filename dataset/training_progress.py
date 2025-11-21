"""
Training Progress Tracker for Consolidated Dataset

Tracks exactly which parts of consolidated chunks have been trained,
allowing precise resume without retraining data.

Key Features:
- Records chunk_id, offset, and global_step for each training iteration
- Saves progress in checkpoints
- Enables exact resume from last trained position
- Prevents duplicate training of same data
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import torch


@dataclass
class ChunkProgress:
    """Progress within a single consolidated chunk."""
    chunk_id: int
    samples_trained: int  # How many samples from this chunk have been trained
    last_offset: int  # Last offset position accessed in this chunk
    completed: bool  # Whether this chunk is fully trained
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)


@dataclass
class ConsolidationProgress:
    """Track batch consolidation progress."""
    last_consolidated_chunk_id: int  # Last chunk ID created
    batches_per_chunk: int  # e.g., 100
    total_batches_consolidated: int
    consolidation_mapping: Dict[int, List[int]]  # chunk_id -> [batch_ids]
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data):
        # Convert string keys back to int for consolidation_mapping
        mapping = {int(k): v for k, v in data['consolidation_mapping'].items()}
        return cls(
            last_consolidated_chunk_id=data['last_consolidated_chunk_id'],
            batches_per_chunk=data['batches_per_chunk'],
            total_batches_consolidated=data['total_batches_consolidated'],
            consolidation_mapping=mapping
        )


@dataclass
class TrainingProgress:
    """Complete training progress across all consolidated chunks."""
    global_step: int
    epoch: int
    current_chunk_id: int
    current_offset: int  # Offset within current chunk
    chunk_progress: Dict[int, ChunkProgress]  # chunk_id -> progress
    total_samples_trained: int
    
    # NEW: Batch building and consolidation tracking
    total_batches_built: int  # Total batch_*.pt files created
    consolidation_progress: Optional[ConsolidationProgress]  # Consolidation state
    
    def to_dict(self):
        return {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'current_chunk_id': self.current_chunk_id,
            'current_offset': self.current_offset,
            'chunk_progress': {k: v.to_dict() for k, v in self.chunk_progress.items()},
            'total_samples_trained': self.total_samples_trained,
            'total_batches_built': self.total_batches_built,
            'consolidation_progress': self.consolidation_progress.to_dict() if self.consolidation_progress else None
        }
    
    @classmethod
    def from_dict(cls, data):
        chunk_progress = {
            int(k): ChunkProgress.from_dict(v) 
            for k, v in data['chunk_progress'].items()
        }
        
        # Handle consolidation progress (may be None for old saves)
        consolidation_prog = None
        if 'consolidation_progress' in data and data['consolidation_progress']:
            consolidation_prog = ConsolidationProgress.from_dict(data['consolidation_progress'])
        
        return cls(
            global_step=data['global_step'],
            epoch=data['epoch'],
            current_chunk_id=data['current_chunk_id'],
            current_offset=data['current_offset'],
            chunk_progress=chunk_progress,
            total_samples_trained=data['total_samples_trained'],
            total_batches_built=data.get('total_batches_built', 0),
            consolidation_progress=consolidation_prog
        )


class TrainingProgressTracker:
    """
    Tracks training progress through consolidated dataset chunks.
    
    Enables exact resume by recording:
    - Which consolidated chunk is currently being trained
    - Offset within that chunk
    - Which chunks are fully trained
    - Global training step for synchronization
    
    Usage:
        tracker = TrainingProgressTracker(checkpoint_dir)
        
        # Load existing progress
        if tracker.has_progress():
            tracker.load()
            start_chunk, start_offset = tracker.get_resume_position()
        
        # During training loop
        tracker.update(
            global_step=train_state.step,
            chunk_id=current_chunk_id,
            offset=current_offset,
            samples_in_batch=batch_size
        )
        
        # Save with checkpoint
        tracker.save()
    """
    
    def __init__(self, checkpoint_dir: Path, dataset_name: str = "default"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.dataset_name = dataset_name
        self.progress_file = self.checkpoint_dir / f"training_progress_{dataset_name}.json"
        
        # Initialize empty progress
        self.progress = TrainingProgress(
            global_step=0,
            epoch=0,
            current_chunk_id=0,
            current_offset=0,
            chunk_progress={},
            total_samples_trained=0,
            total_batches_built=0,
            consolidation_progress=None
        )
    
    def has_progress(self) -> bool:
        """Check if training progress exists."""
        return self.progress_file.exists()
    
    def load(self) -> bool:
        """
        Load training progress from disk.
        
        Returns:
            True if progress was loaded successfully, False otherwise
        """
        if not self.progress_file.exists():
            print(f"INFO: No training progress found: {self.progress_file}")
            return False
        
        try:
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
            
            self.progress = TrainingProgress.from_dict(data)
            
            print(f"\n{'='*70}")
            print(f"LOADED TRAINING PROGRESS")
            print(f"{'='*70}")
            print(f"Global step: {self.progress.global_step:,}")
            print(f"Epoch: {self.progress.epoch}")
            print(f"Current chunk: {self.progress.current_chunk_id}")
            print(f"Current offset: {self.progress.current_offset:,}")
            print(f"Chunks trained: {len([c for c in self.progress.chunk_progress.values() if c.completed])}")
            print(f"Total samples trained: {self.progress.total_samples_trained:,}")
            print(f"{'='*70}\n")
            
            return True
        except Exception as e:
            print(f"WARNING: Failed to load training progress: {e}")
            return False
    
    def save(self):
        """Save training progress to disk."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress.to_dict(), f, indent=2)
            
            print(f"SAVED: Training progress saved: {self.progress_file}")
        except Exception as e:
            print(f"WARNING: Failed to save training progress: {e}")
    
    def update(
        self,
        global_step: int,
        epoch: int,
        chunk_id: int,
        offset: int,
        samples_in_batch: int,
        chunk_size: Optional[int] = None
    ):
        """
        Update training progress after processing a batch.
        
        Args:
            global_step: Global training step
            epoch: Current epoch
            chunk_id: ID of consolidated chunk being trained
            offset: Current offset within chunk
            samples_in_batch: Number of samples in this batch
            chunk_size: Total size of chunk (to detect completion)
        """
        self.progress.global_step = global_step
        self.progress.epoch = epoch
        self.progress.current_chunk_id = chunk_id
        self.progress.current_offset = offset
        self.progress.total_samples_trained += samples_in_batch
        
        # Update chunk-specific progress
        if chunk_id not in self.progress.chunk_progress:
            self.progress.chunk_progress[chunk_id] = ChunkProgress(
                chunk_id=chunk_id,
                samples_trained=0,
                last_offset=0,
                completed=False
            )
        
        chunk_prog = self.progress.chunk_progress[chunk_id]
        chunk_prog.samples_trained += samples_in_batch
        chunk_prog.last_offset = offset
        
        # Mark as completed if we've reached the end
        if chunk_size is not None and offset >= chunk_size:
            chunk_prog.completed = True
    
    def get_resume_position(self) -> Tuple[int, int]:
        """
        Get the position to resume training from.
        
        Returns:
            (chunk_id, offset) tuple indicating where to start
        """
        return self.progress.current_chunk_id, self.progress.current_offset
    
    def get_completed_chunks(self) -> List[int]:
        """Get list of chunk IDs that are fully trained."""
        return [
            chunk_id 
            for chunk_id, prog in self.progress.chunk_progress.items() 
            if prog.completed
        ]
    
    def should_skip_chunk(self, chunk_id: int) -> bool:
        """Check if a chunk should be skipped (already fully trained)."""
        if chunk_id not in self.progress.chunk_progress:
            return False
        return self.progress.chunk_progress[chunk_id].completed
    
    def get_chunk_offset(self, chunk_id: int) -> int:
        """
        Get the offset to start from within a chunk.
        
        Returns:
            0 if chunk is new or completed (for next epoch)
            last_offset + 1 if chunk is partially trained
        """
        if chunk_id not in self.progress.chunk_progress:
            return 0
        
        chunk_prog = self.progress.chunk_progress[chunk_id]
        
        # If completed, start from beginning (for next epoch)
        if chunk_prog.completed:
            return 0
        
        # Otherwise continue from last position
        return chunk_prog.last_offset + 1
    
    def mark_epoch_complete(self):
        """Mark current epoch as complete and reset for next epoch."""
        self.progress.epoch += 1
        
        # Reset all chunk completion flags for next epoch
        for chunk_prog in self.progress.chunk_progress.values():
            chunk_prog.completed = False
            chunk_prog.samples_trained = 0
            chunk_prog.last_offset = 0
        
        # Reset to first chunk
        self.progress.current_chunk_id = 0
        self.progress.current_offset = 0
        
        print(f"\nEpoch {self.progress.epoch - 1} complete! Starting epoch {self.progress.epoch}")
    
    def get_statistics(self) -> Dict:
        """Get training progress statistics."""
        completed_chunks = len(self.get_completed_chunks())
        total_chunks = len(self.progress.chunk_progress)
        
        stats = {
            'global_step': self.progress.global_step,
            'epoch': self.progress.epoch,
            'current_chunk': self.progress.current_chunk_id,
            'current_offset': self.progress.current_offset,
            'completed_chunks': completed_chunks,
            'total_chunks': total_chunks,
            'total_samples_trained': self.progress.total_samples_trained,
            'total_batches_built': self.progress.total_batches_built,
            'chunks_progress': {
                chunk_id: {
                    'samples_trained': prog.samples_trained,
                    'completed': prog.completed
                }
                for chunk_id, prog in self.progress.chunk_progress.items()
            }
        }
        
        # Add consolidation info if available
        if self.progress.consolidation_progress:
            stats['consolidation'] = {
                'last_consolidated_chunk': self.progress.consolidation_progress.last_consolidated_chunk_id,
                'batches_per_chunk': self.progress.consolidation_progress.batches_per_chunk,
                'total_batches_consolidated': self.progress.consolidation_progress.total_batches_consolidated,
                'consolidated_chunks': len(self.progress.consolidation_progress.consolidation_mapping)
            }
        
        return stats
    
    # === Batch Building and Consolidation Methods ===
    
    def record_batch_built(self, batch_id: int):
        """Record that a batch file was created."""
        self.progress.total_batches_built = max(self.progress.total_batches_built, batch_id + 1)
    
    def get_next_batch_id(self) -> int:
        """Get the next batch ID to create."""
        return self.progress.total_batches_built
    
    def init_consolidation(self, batches_per_chunk: int = 100):
        """Initialize consolidation tracking."""
        if self.progress.consolidation_progress is None:
            self.progress.consolidation_progress = ConsolidationProgress(
                last_consolidated_chunk_id=-1,
                batches_per_chunk=batches_per_chunk,
                total_batches_consolidated=0,
                consolidation_mapping={}
            )
    
    def record_consolidation(
        self,
        chunk_id: int,
        batch_ids: List[int]
    ):
        """Record that batches were consolidated into a chunk."""
        if self.progress.consolidation_progress is None:
            self.init_consolidation()
        
        cons = self.progress.consolidation_progress
        cons.consolidation_mapping[chunk_id] = batch_ids
        cons.last_consolidated_chunk_id = chunk_id
        cons.total_batches_consolidated += len(batch_ids)
    
    def get_batches_to_consolidate(self) -> Tuple[int, int, int]:
        """Get range of batches that should be consolidated next.
        
        Returns:
            (start_batch, end_batch, next_chunk_id)
            Returns (0, 0, 0) if no batches ready for consolidation
        """
        if self.progress.consolidation_progress is None:
            self.init_consolidation()
        
        cons = self.progress.consolidation_progress
        
        # Calculate starting batch from last consolidated chunk
        start_batch = (cons.last_consolidated_chunk_id + 1) * cons.batches_per_chunk
        
        # Check how many complete chunks can be made
        available_batches = self.progress.total_batches_built - start_batch
        complete_chunks = available_batches // cons.batches_per_chunk
        
        if complete_chunks == 0:
            return (0, 0, 0)  # Not enough batches yet
        
        end_batch = start_batch + (complete_chunks * cons.batches_per_chunk)
        next_chunk_id = cons.last_consolidated_chunk_id + 1
        
        return (start_batch, end_batch, next_chunk_id)
    
    def is_batch_consolidated(self, batch_id: int) -> bool:
        """Check if a batch has been consolidated."""
        if self.progress.consolidation_progress is None:
            return False
        
        cons = self.progress.consolidation_progress
        
        # Check if batch is in any consolidated chunk
        for batch_ids in cons.consolidation_mapping.values():
            if batch_id in batch_ids:
                return True
        
        return False
    
    def get_unconsolidated_batches(self) -> List[int]:
        """Get list of batch IDs that haven't been consolidated yet."""
        if self.progress.consolidation_progress is None:
            return list(range(self.progress.total_batches_built))
        
        # Get all consolidated batch IDs
        consolidated = set()
        for batch_ids in self.progress.consolidation_progress.consolidation_mapping.values():
            consolidated.update(batch_ids)
        
        # Return batches not in consolidated set
        return [i for i in range(self.progress.total_batches_built) if i not in consolidated]


def list_available_chunks(checkpoint_dir: Path) -> List[Tuple[int, Path]]:
    """
    List all available consolidated chunks in order.
    
    Returns:
        List of (chunk_id, file_path) tuples sorted by chunk_id
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    chunks = []
    for file in checkpoint_dir.glob("consolidated_*.pt"):
        try:
            # Extract chunk ID from filename: consolidated_0.pt -> 0
            chunk_id = int(file.stem.split('_')[1])
            chunks.append((chunk_id, file))
        except (IndexError, ValueError):
            continue
    
    return sorted(chunks, key=lambda x: x[0])


def get_training_manifest(checkpoint_dir: Path) -> Dict:
    """
    Generate a manifest of dataset and training status.
    
    Returns:
        Dictionary with:
        - available_chunks: List of chunk IDs
        - chunk_files: Mapping of chunk_id to file path
        - training_progress: Current progress if available
    """
    chunks = list_available_chunks(checkpoint_dir)
    
    manifest = {
        'available_chunks': [chunk_id for chunk_id, _ in chunks],
        'chunk_files': {chunk_id: str(path) for chunk_id, path in chunks},
        'total_chunks': len(chunks)
    }
    
    # Add training progress if available
    tracker = TrainingProgressTracker(checkpoint_dir)
    if tracker.has_progress():
        tracker.load()
        manifest['training_progress'] = tracker.get_statistics()
    else:
        manifest['training_progress'] = None
    
    return manifest


# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Training Progress Tracker")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--action", choices=['show', 'reset'], default='show', help="Action to perform")
    
    args = parser.parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    
    if args.action == 'show':
        print(f"\n{'='*70}")
        print(f"TRAINING PROGRESS REPORT")
        print(f"{'='*70}")
        
        manifest = get_training_manifest(checkpoint_dir)
        
        print(f"\nDataset Info:")
        print(f"   Total consolidated chunks: {manifest['total_chunks']}")
        print(f"   Available chunks: {manifest['available_chunks']}")
        
        if manifest['training_progress']:
            prog = manifest['training_progress']
            print(f"\nTraining Status:")
            print(f"   Global step: {prog['global_step']:,}")
            print(f"   Epoch: {prog['epoch']}")
            print(f"   Current chunk: {prog['current_chunk']}")
            print(f"   Current offset: {prog['current_offset']:,}")
            print(f"   Completed chunks: {prog['completed_chunks']}/{prog['total_chunks']}")
            print(f"   Total samples trained: {prog['total_samples_trained']:,}")
        else:
            print(f"\nINFO: No training progress found (fresh start)")
        
        print(f"\n{'='*70}\n")
    
    elif args.action == 'reset':
        tracker = TrainingProgressTracker(checkpoint_dir)
        if tracker.progress_file.exists():
            tracker.progress_file.unlink()
            print(f"Training progress reset")
        else:
            print(f"INFO: No training progress to reset")

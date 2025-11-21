"""
Streaming dataset builder that overlaps cache population with GPU encoding.

Key insight: Start encoding as soon as some samples are cached,
while CPU continues rendering remaining samples in parallel.
"""

import threading
import queue
import gc
import time
import os
import shutil
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
from safetensors.torch import save_file, load_file
from dataset.image_cache import ImageCache


class CachedDataset(Dataset):
    """Custom dataset that loads from cache."""
    def __init__(self, samples: List[Any], cache: ImageCache):
        self.samples = samples
        self.cache = cache
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Extract text safely
        # Extract cache key (unified logic)
        if hasattr(sample, '_cache_key'):
            key = sample._cache_key
        else:
            # Fallback key generation
            if hasattr(sample, 'image') and sample.image is not None:
                key = f"IMG:{sample.sample_id}"
            elif hasattr(sample, 'grid') and sample.grid is not None:
                key = f"GRID:{sample.sample_id}"
            else:
                if hasattr(sample, 'text') and sample.text:
                    key = sample.text
                elif hasattr(sample, 'question') and sample.question:
                    key = sample.question
                elif isinstance(sample, dict):
                    key = sample.get('text', '') or sample.get('question', '')
                else:
                    key = str(sample)
        
        # Load from cache (all samples should be cached now)
        img = self.cache.get(key, 224, 224)
        if img is None:
            # This should ideally not happen if synchronization is correct
            # But if it does, we might want to retry or raise a more informative error
            raise ValueError(f"Sample {idx} not in cache: {text[:50]}...")
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


class StreamingCacheEncoder:
    """
    Overlap CPU rendering with GPU encoding for maximum throughput.
    
    Architecture:
    - Producer thread: CPU renders samples → saves to cache
    - Consumer thread: GPU encodes cached samples
    - Queue: Coordinates which batches are ready
    """
    
    def __init__(self, cache, encoder, device, batch_size=256, checkpoint_dir=None, drive_dir=None):
        self.cache = cache
        self.encoder = encoder
        self.device = device
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        self.drive_dir = drive_dir  # Google Drive path for progress saves
        
        # Coordination
        self.ready_queue = queue.Queue(maxsize=10)  # Batches ready to encode
        self.stop_flag = threading.Event()
        self.cache_complete = threading.Event()
        
        # Direct-to-disk mode (no RAM accumulation)
        self.batch_files = []  # Track individual batch files
        self.drive_checkpoints = []  # Track Drive-saved chunks
        self.lock = threading.Lock()
        
        # Parallel execution tracking
        self.cached_count = 0  # Number of samples cached so far
        self.cache_threshold_reached = threading.Event()  # Signal when threshold reached
        self.new_batches_start_idx = 0  # Index where NEW batches start (after resume)
        self.new_batches_start_idx = 0  # Index where New batches start (after resume)
        
        # Cleanup stale temp files from previous runs
        self._cleanup_stale_temp_files()

    def _cleanup_stale_temp_files(self):
        """Clean up any stuck temp metadata files from previous runs."""
        import glob
        stale_files = glob.glob("temp_*_meta.json")
        if stale_files:
            print(f"Cleaning up {len(stale_files)} stale temp metadata files...", flush=True)
            for f in stale_files:
                try:
                    os.remove(f)
                except Exception as e:
                    print(f"Warning: Could not remove stale file {f}: {e}", flush=True)
        
    def producer_thread(self, samples, renderer, start_threshold=50000):
        """
        CPU thread: Render samples (text/grid/image) -> ImageCache.
        Ensures ALL modalities are converted to images before GPU encoding.
        """
        print(f"Producer: Processing samples -> Images (parallel CPUs)...")
        
        # Render in batches (skip already cached)
        batch_size = 100
        cached_count = 0
        skipped_count = 0
        
        # Progress bar for caching
        pbar = tqdm(desc="Caching", unit="samples")
        
        # Handle generator or list
        batch_buffer = []
        
        for sample in samples:

            
            batch_buffer.append(sample)
            # print(f"DEBUG: Producer received sample: {str(sample)[:50]}")
            
            if len(batch_buffer) >= batch_size:
                # print(f"DEBUG: Producer processing batch of {len(batch_buffer)}")
                self._process_batch(batch_buffer, renderer, pbar)
                batch_buffer = []
                
        # Process remaining
        if batch_buffer:
            print(f"DEBUG: Producer processing remaining batch of {len(batch_buffer)}")
            self._process_batch(batch_buffer, renderer, pbar)
            
        # Signal completion
        pbar.close()
        self.cache_complete.set()
        self.ready_queue.put(None)  # Sentinel to stop consumer
        print(f"Caching complete!")

    def _process_batch(self, batch, renderer, pbar):
        """Process a batch of samples: cache -> queue."""
        uncached = []
        for sample in batch:
            # Determine cache key based on modality
            if sample.image is not None:
                key = f"IMG:{sample.sample_id}"
            elif sample.grid is not None:
                key = f"GRID:{sample.sample_id}"
            else:
                if hasattr(sample, 'text') and sample.text:
                    key = sample.text
                elif hasattr(sample, 'question') and sample.question:
                    key = sample.question
                else:
                    key = str(sample)
            
            sample._cache_key = key
            
            if not self.cache.has_been_cached(key, 224, 224):
                uncached.append(sample)
        
        if uncached:
            self._process_batch_to_cache(uncached, renderer)
            self.cached_count += len(uncached)
            
            # Check threshold
            if self.cached_count >= 50000 and not self.cache_threshold_reached.is_set():
                self.cache_threshold_reached.set()
                print(f"Cache threshold reached ({self.cached_count}), signaling GPU...")
        
        # Push lightweight samples to queue for consumer
        for sample in batch:
            # Clear heavy data to save RAM
            sample.image = None
            sample.grid = None
            self.ready_queue.put(sample)
            pbar.update(1)

                
    def _process_batch_to_cache(self, samples, renderer):
        """Helper: Convert batch of mixed samples to images and save to cache."""
        import numpy as np
        from PIL import Image
        import cv2
        
        for sample in samples:
            key = getattr(sample, '_cache_key', str(sample))
            img_array = None
            
            try:
                # 1. Existing Image
                if sample.image is not None:
                    if isinstance(sample.image, np.ndarray):
                        img = sample.image
                    elif isinstance(sample.image, Image.Image):
                        img = np.array(sample.image)
                    else:
                        continue # Skip invalid
                        
                    # Resize if needed
                    if img.shape[0] != 224 or img.shape[1] != 224:
                        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)
                    
                    img_array = img
                
                # 2. Grid -> Image
                elif sample.grid is not None:
                    # Simple grid visualization
                    # Normalize to 0-255
                    grid = sample.grid.astype(np.float32)
                    if grid.max() > 0:
                        grid = (grid / grid.max() * 255).astype(np.uint8)
                    else:
                        grid = grid.astype(np.uint8)
                    
                    # Resize nearest neighbor to preserve grid structure
                    img = cv2.resize(grid, (224, 224), interpolation=cv2.INTER_NEAREST)
                    # Convert to RGB
                    img_array = np.stack([img]*3, axis=-1)
                
                # 3. Text -> Image (Renderer)
                else:
                    # Use renderer
                    pil_img = renderer.render_plain_text(key) # key is the text for text samples
                    img_array = np.array(pil_img)
                
                # Save to cache
                if img_array is not None:
                    # print(f"DEBUG: Caching {key[:20]}...")
                    self.cache.put(key, 224, 224, img_array)
                    
            except Exception as e:
                print(f"Failed to process sample {sample.sample_id}: {e}")
    
    def consumer_thread(self, samples=None):
        """
        GPU thread: Pull from queue, encode cached samples.
        """
        print(f"Consumer: Starting GPU encoding loop...") 
        
        # Calculate starting batch number accounting for consolidated files
        num_consolidated_batches = len(self.drive_checkpoints) * 100
        batch_count = num_consolidated_batches
        
        # Fast-forward logic would need to consume queue until match...
        # For now, we assume streaming starts from where we left off or 0
        # If resuming, producer should skip samples before pushing to queue
        
        pbar = tqdm(desc="GPU Encoding", unit="batches")
        
        batch_buffer = []
        
        with torch.no_grad():
            while True:
                # Get sample from queue
                try:
                    sample = self.ready_queue.get(timeout=1.0)
                except queue.Empty:
                    if self.cache_complete.is_set():
                        break
                    continue
                
                if sample is None: # Sentinel
                    break
                
                batch_buffer.append(sample)
                
                if len(batch_buffer) >= self.batch_size:
                    self._encode_batch(batch_buffer, batch_count, pbar)
                    batch_buffer = []
                    batch_count += 1
                    
                    # Cleanup every 10 batches
                    if batch_count % 10 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()
                    
                    # Consolidate every 100 batches
                    if batch_count % 100 == 0 and self.checkpoint_dir:
                        pbar.write(f"Consolidating batches...")
                        self._consolidate_to_drive(batch_count)

            # Process remaining
            if batch_buffer:
                self._encode_batch(batch_buffer, batch_count, pbar)
        
        pbar.close()
        print(f"Consumer: Encoding complete")

    def _encode_batch(self, batch_samples, batch_count, pbar):
        """Helper to encode a batch."""
        batch_images = []
        missing_count = 0
        
        for sample in batch_samples:
            key = getattr(sample, '_cache_key', str(sample))
            img = self.cache.get(key, 224, 224)
            if img is not None:
                batch_images.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
            else:
                print(f"DEBUG: Missing in cache: {key[:20]}")
                missing_count += 1
        
        if not batch_images:
            print("DEBUG: No images in batch, skipping encode")
            return

        # Setup batch file path
        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            batch_file = os.path.join(self.checkpoint_dir, f"batch_{batch_count:05d}.safetensors")
            meta_file = os.path.join(self.checkpoint_dir, f"batch_{batch_count:05d}_meta.json")
            
            if os.path.exists(batch_file):
                pbar.update(1)
                return

        # Encode
        batch_images = torch.stack(batch_images).to(self.device)
        result = self.encoder(images=batch_images, return_children=True)
        
        # Save
        if self.checkpoint_dir:
            tensors = {'sketches': result['sketches'].half().cpu()}
            
            # Sanitize metadata for JSON
            checksums = result.get('checksums', [])
            if hasattr(checksums, 'tolist'):
                checksums = checksums.tolist()
            
            if checksums and isinstance(checksums[0], bytes):
                checksums = [c.decode('utf-8') for c in checksums]
                
            metadata = {
                'checksums': checksums,
                'children': result['children'].tolist() if 'children' in result else []
            }
            
            try:
                save_file(tensors, batch_file)
                with open(meta_file, 'w') as f:
                    json.dump(metadata, f)
            except Exception as e:
                print(f"Error saving batch {batch_count}: {e}")
                # Try to delete corrupted files
                if os.path.exists(batch_file): os.remove(batch_file)
                if os.path.exists(meta_file): os.remove(meta_file)
            
            with self.lock:
                self.batch_files.append(batch_file)
                if hasattr(self, 'progress_tracker'):
                    self.progress_tracker.record_batch_built(batch_count)
        
        pbar.update(1)
    
    def _consolidate_to_drive(self, up_to_batch, final=False):
        """Consolidate processed batches using unified progress tracker."""
        # Use progress tracker to determine what to consolidate
        if not hasattr(self, 'progress_tracker'):
            # Fallback for simple usage
            start_batch = len(self.drive_checkpoints) * 100
            end_batch = up_to_batch
            next_chunk_id = len(self.drive_checkpoints)
        else:
            # Get next chunk to consolidate
            start_batch, end_batch, next_chunk_id = self.progress_tracker.get_batches_to_consolidate()
            if end_batch == 0:
                return
        
        # Ensure we don't go beyond what's available
        end_batch = min(end_batch, up_to_batch)
        
        # Build list of batch files to consolidate
        batch_subset = []
        for batch_num in range(start_batch, end_batch):
            batch_file = os.path.join(self.checkpoint_dir, f"batch_{batch_num:05d}.safetensors")
            if os.path.exists(batch_file):
                batch_subset.append(batch_file)
        
        if not batch_subset:
            return
        
        chunk_size = end_batch - start_batch
        

        
        print(f"Consolidating chunk {next_chunk_id}: batches {start_batch}-{end_batch-1}...")
        print(f"DEBUG: Consolidating batch subset: {len(batch_subset)} files")
        
        # Load and concatenate in smaller chunks to avoid OOM
        # Process 20 batches at a time instead of all 100+
        mini_chunk_size = 20
        all_consolidated = {'sketches': [], 'checksums': [], 'children': []}
        corrupted_files = []
        
        for mini_start in range(0, len(batch_subset), mini_chunk_size):
            mini_end = min(mini_start + mini_chunk_size, len(batch_subset))
            mini_batch_files = batch_subset[mini_start:mini_end]
            
            chunk_data = {'sketches': [], 'checksums': [], 'children': []}
            
            for batch_file in mini_batch_files:
                if os.path.exists(batch_file):
                    try:
                        # Load tensors
                        batch = load_file(batch_file)
                        print(f"DEBUG: Loaded batch {batch_file} keys: {list(batch.keys())}")
                        if 'sketches' in batch:
                            chunk_data['sketches'].append(batch['sketches'])
                        
                        # Load metadata
                        meta_file = batch_file.replace('.safetensors', '_meta.json')
                        if os.path.exists(meta_file):
                            with open(meta_file, 'r') as f:
                                meta = json.load(f)
                                if 'checksums' in meta:
                                    chunk_data['checksums'].extend(meta['checksums'])
                                if 'children' in meta:
                                    chunk_data['children'].extend(meta['children'])
                        del batch
                    except Exception as e:
                        corrupted_files.append(batch_file)
                        continue
            
            # Concatenate this mini chunk
            if chunk_data['sketches']:
                print(f"DEBUG: Concatenating mini-chunk {mini_start}...")
                mini_concat = torch.cat(chunk_data['sketches'], dim=0)
                all_consolidated['sketches'].append(mini_concat)
                del mini_concat
                print(f"DEBUG: Concatenation complete.")
            
            # Extend lists for metadata
            all_consolidated['checksums'].extend(chunk_data['checksums'])
            all_consolidated['children'].extend(chunk_data['children'])
            
            del chunk_data
            print(f"DEBUG: GC collecting...")
            gc.collect()
            print(f"DEBUG: GC done.")
        
        # Final concatenation
        consolidated_tensors = {}
        if all_consolidated['sketches']:
            print(f"DEBUG: Final concatenation...")
            consolidated_tensors['sketches'] = torch.cat(all_consolidated['sketches'], dim=0)
            print(f"DEBUG: Final concatenation done.")
            
        consolidated_meta = {
            'checksums': all_consolidated['checksums'],
            'children': all_consolidated['children']
        }
        del all_consolidated
        
        # Save to drive_dir if available, otherwise checkpoint_dir
        save_dir = self.drive_dir if self.drive_dir else self.checkpoint_dir
        consolidated_file = os.path.join(save_dir, f"consolidated_{next_chunk_id:03d}.safetensors")
        consolidated_meta_file = os.path.join(save_dir, f"consolidated_{next_chunk_id:03d}_meta.json")
        
        if consolidated_tensors:
            print(f"DEBUG: Saving consolidated file {consolidated_file}...", flush=True)
            save_file(consolidated_tensors, consolidated_file)
            print(f"DEBUG: Saved consolidated file.", flush=True)
        
        # Free memory immediately to prevent OOM during JSON dump
        del consolidated_tensors
        print(f"DEBUG: Freed consolidated_tensors memory.", flush=True)
        gc.collect()
        print(f"DEBUG: GC done.", flush=True)

        print(f"DEBUG: Meta sizes - Checksums: {len(consolidated_meta.get('checksums', []))}, Children: {len(consolidated_meta.get('children', []))}", flush=True)
        if consolidated_meta.get('children') and len(consolidated_meta['children']) > 0:
             print(f"DEBUG: First child type: {type(consolidated_meta['children'][0])}", flush=True)

        print(f"DEBUG: Saving metadata to {consolidated_meta_file}...", flush=True)
        temp_meta_file = f"temp_{next_chunk_id:03d}_meta.json"
        try:
            # Write to local temp file first to avoid network drive timeouts/hangs
            print(f"DEBUG: Writing to temp file {temp_meta_file}...", flush=True)
            with open(temp_meta_file, 'w') as f:
                json.dump(consolidated_meta, f)
            print(f"DEBUG: Wrote temp file. Moving to target...", flush=True)
            
            # Move/Copy to target
            import shutil
            shutil.move(temp_meta_file, consolidated_meta_file)
            print(f"DEBUG: Saved metadata file.", flush=True)
        except Exception as e:
            print(f"DEBUG: Error saving metadata: {e}", flush=True)
            raise e
        finally:
            # Clean up temp file if it exists (success or failure)
            if os.path.exists(temp_meta_file):
                try:
                    os.remove(temp_meta_file)
                except Exception as cleanup_error:
                    print(f"DEBUG: Failed to cleanup temp file {temp_meta_file}: {cleanup_error}", flush=True)
            
        self.drive_checkpoints.append(consolidated_file)
        print(f"DEBUG: Appended to drive_checkpoints: {consolidated_file}", flush=True)
        
        del consolidated_meta
        gc.collect()
        
        chunk_size_mb = os.path.getsize(consolidated_file) / 1024 / 1024
        print(f"Saved consolidated chunk {next_chunk_id} ({chunk_size_mb:.1f}MB)")
        
        # Report corrupted files that were skipped
        if corrupted_files:
            print(f"Skipped {len(corrupted_files)} corrupted batch files from previous session")
        
        # Record consolidation in progress tracker
        if hasattr(self, 'progress_tracker'):
            batch_ids = list(range(start_batch, end_batch))
            self.progress_tracker.record_consolidation(
                chunk_id=next_chunk_id,
                batch_ids=batch_ids
            )
            print(f"Recorded consolidation: chunk {next_chunk_id} <- batches {start_batch}-{end_batch-1}")
        
        # Delete individual batch files by batch NUMBER, not list index
        # batch_files list may have gaps from previous consolidations
        deleted_count = 0
        freed_mb = 0
        for batch_num in range(start_batch, end_batch):
            batch_file = os.path.join(self.checkpoint_dir, f"batch_{batch_num:05d}.safetensors")
            meta_file = os.path.join(self.checkpoint_dir, f"batch_{batch_num:05d}_meta.json")
            
            if os.path.exists(batch_file):
                batch_size_mb = os.path.getsize(batch_file) / 1024 / 1024
                os.remove(batch_file)
                if os.path.exists(meta_file):
                    os.remove(meta_file)
                deleted_count += 1
                freed_mb += batch_size_mb
        
        print(f"Deleted {deleted_count} individual batch files ({freed_mb:.1f}MB freed from Colab)")
        print(f"   Drive location: {self.drive_dir}")
        print(f"   Will sync to: C:\\Users\\sao\\Documents\\model-engine")
        
        # Resume producer        # Resume threads
        if not final:
            self.consolidation_pause.set()
        print(f"Resuming encoding and caching...\n")
        
        # Cache management: Keep all cached images for now
        # Can manually clear if disk space critical
    
    def _check_resume_state(self):
        """Check for existing progress using unified tracker.
        
        Returns:
            (samples_already_encoded, existing_batch_files, existing_consolidated_files)
        """
        from dataset.training_progress import TrainingProgressTracker
        
        checkpoint_dir = Path(self.checkpoint_dir)
        if not checkpoint_dir.exists():
            return 0, [], []
        
        # Initialize unified progress tracker
        dataset_name = checkpoint_dir.parent.name
        tracker = TrainingProgressTracker(
            checkpoint_dir=checkpoint_dir.parent / "checkpoints" if (checkpoint_dir.parent / "checkpoints").exists() else checkpoint_dir,
            dataset_name=dataset_name
        )
        
        # Load existing progress
        if tracker.has_progress():
            tracker.load()
        
        # Get actual batch files (for registration)
        batch_files = sorted(checkpoint_dir.glob("batch_*.safetensors"))
        consolidated_files = sorted(checkpoint_dir.glob("consolidated_*.safetensors"))
        
        # Calculate samples from tracker (source of truth)
        if tracker.progress.consolidation_progress:
            cons = tracker.progress.consolidation_progress
            samples_in_consolidated = cons.total_batches_consolidated * self.batch_size
        else:
            samples_in_consolidated = 0
        
        # Unconsolidated batches
        unconsolidated = tracker.get_unconsolidated_batches()
        samples_in_batches = len(unconsolidated) * self.batch_size
        
        total_encoded = samples_in_consolidated + samples_in_batches
        
        # Store tracker for later use
        self.progress_tracker = tracker
        
        return total_encoded, batch_files, consolidated_files
    
    def stream_build(self, samples, renderer, start_threshold=50000, initial_cache_percent=0):
        """
        Main entry point: Start both threads and coordinate.
        Supports resume from existing batch/consolidated files.
        
        Args:
            samples: List of samples to process
            renderer: TextRenderer instance
            start_threshold: Start encoding after this many samples cached
        
        Returns:
            Encoded results dict
        """
        # Check for existing progress
        samples_encoded, existing_batches, existing_consolidated = self._check_resume_state()
        
        if samples_encoded > 0:
            print(f"\nRESUME MODE: Found existing progress")
            print(f"   Consolidated files: {len(existing_consolidated)} ({len(existing_consolidated) * 100} batches)")
            print(f"   Batch files: {len(existing_batches)}")
            print(f"   Samples already encoded: {samples_encoded:,}/{len(samples):,} ({samples_encoded/len(samples)*100:.1f}%)")
            print(f"   Resuming from sample {samples_encoded}...\n")
            
            # Register existing files
            self.batch_files.extend(existing_batches)
            self.drive_checkpoints.extend(existing_consolidated)
            
            # Skip already processed samples
            if samples_encoded > 0:
                print(f"Skipping {samples_encoded} already processed samples...")
                import itertools
                # Consume the generator to skip items
                samples = itertools.islice(samples, samples_encoded, None)
        
        # Adjust threshold based on initial cache
        if initial_cache_percent < 5:
            # If cache is empty/minimal, start GPU after just 1 batch
            actual_threshold = self.batch_size
            if samples_encoded == 0:
                print(f"\nNo cache found - using fast start mode")
        else:
            actual_threshold = start_threshold
        
        print(f"\n{'='*70}")
        print(f"PARALLEL STREAMING PIPELINE")
        print(f"   CPU: Caching samples (parallel workers)")
        print(f"   GPU: Starts after {actual_threshold} cached (batch={self.batch_size})")
        print(f"   Strategy: CPU + GPU run simultaneously")
        print(f"   Memory: Each batch saved directly to disk (minimal RAM)")
        print(f"   Batch files: {self.checkpoint_dir}")
        print(f"{'='*70}\n")
        
        # Start producer thread
        producer = threading.Thread(
            target=self.producer_thread,
            args=(samples, renderer, actual_threshold)
        )
        producer.start()
        
        # Start consumer thread
        consumer = threading.Thread(
            target=self.consumer_thread
        )
        consumer.start()
        
        # Wait for both to complete
        producer.join()
        consumer.join()
        
        # Consolidate any remaining batches
        if self.checkpoint_dir:
            remaining_batches = len(self.batch_files) - (len(self.drive_checkpoints) * 100)
            if remaining_batches > 0:
                print(f"\n📤 Consolidating final {remaining_batches} batches...")
                self._consolidate_to_drive(len(self.batch_files), final=True)
        
        return self._load_existing_results()
    
    def _load_existing_results(self):
        """Load results from existing consolidated files and remaining batches."""
        
        # Load from local consolidated files
        print(f"\n📚 Loading {len(self.drive_checkpoints)} consolidated chunks from local disk...")
        all_results = {'sketches': [], 'checksums': [], 'children': []}
        
        for i, chunk_file in enumerate(self.drive_checkpoints):
            if os.path.exists(chunk_file):
                # Load tensors
                chunk = load_file(chunk_file)
                if 'sketches' in chunk:
                    all_results['sketches'].append(chunk['sketches'])
                
                # Load metadata
                meta_file = str(chunk_file).replace('.safetensors', '_meta.json')
                if os.path.exists(meta_file):
                    with open(meta_file, 'r') as f:
                        meta = json.load(f)
                        if 'checksums' in meta:
                            all_results['checksums'].extend(meta['checksums'])
                        if 'children' in meta:
                            all_results['children'].extend(meta['children'])
                
                del chunk
                gc.collect()
                print(f"  Loaded chunk {i+1}/{len(self.drive_checkpoints)}")
        
        # Final concatenation
        print(f"Final concatenation...")
        result = {}
        
        if all_results['sketches']:
            result['sketches'] = torch.cat(all_results['sketches'], dim=0).float()
        else:
            result['sketches'] = torch.empty(0)
            
        result['checksums'] = all_results['checksums']
        result['children'] = torch.tensor(all_results['children']) if all_results['children'] else torch.empty(0)
        
        # Cleanup any remaining batch files from Colab
        remaining_files = [f for f in self.batch_files if os.path.exists(f)]
        if remaining_files:
            print(f"🧹 Cleaning up {len(remaining_files)} remaining batch files from Colab...")
            for batch_file in remaining_files:
                os.remove(batch_file)
                meta_file = str(batch_file).replace('.safetensors', '_meta.json')
                if os.path.exists(meta_file):
                    os.remove(meta_file)
        
        # Keep Drive chunks - they will sync to local machine
        print(f"💾 Kept {len(self.drive_checkpoints)} chunks in Drive (syncing to local machine)")
        print(f"   Drive location: {self.drive_dir}")
        print(f"   Will sync to: C:\\Users\\sao\\Documents\\model-engine")
        print(f"   You can delete from Drive after confirming sync completed")
        
        print(f"Complete! Final shape: {result['sketches'].shape}")
        return result

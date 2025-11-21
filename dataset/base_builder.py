"""
Unified base class for all dataset builders.
Supports multimodal data: text, images, mazes, puzzles.
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import gc

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.common import PuzzleDatasetMetadata
from dataset.image_cache import ImageCache


class ModalityType:
    """Supported data modalities."""
    TEXT = "text"
    IMAGE = "image"
    MAZE = "maze"
    GRID = "grid"  # Generic 2D grid (ARC, Sudoku, etc.)
    MULTIMODAL = "multimodal"  # Mixed modalities


class DataSample(BaseModel):
    """Single training sample with optional multiple modalities."""
    
    # Unique identifier
    sample_id: str
    
    # Primary modality
    modality: str  # ModalityType
    
    # Data fields (at least one must be present)
    text: Optional[str] = None
    image: Optional[Any] = None  # PIL Image or np.ndarray
    grid: Optional[np.ndarray] = None  # 2D array for mazes/puzzles
    
    # Labels/targets
    label: Optional[Union[str, int, np.ndarray]] = None
    
    # Metadata
    metadata: Dict[str, Any] = {}
    
    class Config:
        arbitrary_types_allowed = True


def collate_images(batch):
    """Convert batch of numpy images [H,W,C] to tensor [B,C,H,W]."""
    batch_array = np.stack(batch, axis=0)  # [B, H, W, C]
    batch_tensor = torch.from_numpy(batch_array).permute(0, 3, 1, 2)  # [B, C, H, W]
    return batch_tensor.float() / 255.0  # Normalize to [0, 1]


class SampleDataset(Dataset):
    """Custom Dataset with cache support."""
    def __init__(self, samples, image_converter, cache):
        self.samples = samples
        self.image_converter = image_converter
        self.cache = cache
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Handle both dict and Pydantic DataSample objects
        if hasattr(sample, 'text'):
            text = sample.text
        elif hasattr(sample, 'question'):
            text = sample.question
        elif isinstance(sample, dict):
            text = sample.get('text', '') or sample.get('question', '') or str(sample)
        else:
            text = str(sample)
        
        # Try cache first (instant)
        cached_img = self.cache.get(text, 224, 224)
        if cached_img is not None:
            return cached_img
        
        # Fallback to rendering (shouldn't happen after pre-population)
        return self.image_converter(sample)


class BaseDatasetBuilder(ABC):
    """
    Abstract base class for vision-unified dataset builders.
    
    Pipeline:
    1. Load raw data (text/images/grids)
    2. Preprocess (render text to images if needed)
    3. Augment (rotations, flips, etc.)
    4. Encode via TRM vision encoder → capsules
    5. Save capsule dataset
    
    All data goes through TRM encoder - no separate text/vision paths.
    """
    
    def __init__(self, config: BaseModel):
        self.config = config
        self.encoder = None
        self.samples: List[DataSample] = []
        self.text_renderer = None
    
    @abstractmethod
    def load_raw_data(self) -> List[DataSample]:
        """Load raw data from source (files, HF datasets, etc.)."""
        pass
    
    @abstractmethod
    def preprocess_sample(self, sample: DataSample) -> DataSample:
        """Apply modality-specific preprocessing."""
        pass
    
    def augment_sample(self, sample: DataSample) -> List[DataSample]:
        """
        Generate augmented versions of sample.
        Override for modality-specific augmentation.
        """
        return [sample]  # No augmentation by default
    
    def build_dataset(self) -> Dict[str, Any]:
        """
        Unified Streaming Pipeline:
        Generator -> Producer (Cache) -> Queue -> Consumer (Encode) -> Safetensors
        
        Uses StreamingCacheEncoder to overlap CPU rendering with GPU encoding.
        """
        print("\nBuilding dataset (Unified Streaming Pipeline)...")
        
        # Imports
        from dataset.streaming_builder import StreamingCacheEncoder
        from models.text_renderer import TextRenderer
        
        # 1. Setup Components
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache
        cache_dir = output_dir / "cache"
        cache = ImageCache(str(cache_dir))
        
        # Renderer
        renderer = TextRenderer(
            font_family=getattr(self.config, 'font_path', "arial.ttf"),
            width=224,
            height=224
        )
        
        # Encoder
        self._init_encoder()
        
        # Streaming Builder
        streamer = StreamingCacheEncoder(
            cache=cache,
            encoder=self.encoder,
            device=next(self.encoder.parameters()).device,
            batch_size=getattr(self.config, 'batch_size', 256),
            checkpoint_dir=str(output_dir / "checkpoints"),
            drive_dir=getattr(self.config, 'drive_dir', None)
        )
        
        # 2. Start Streaming Build
        print("Streaming raw data...")
        sample_generator = self.load_raw_data()
        
        try:
            streamer.stream_build(sample_generator, renderer)
        except KeyboardInterrupt:
            print("\nSTOPPED: Build interrupted by user.")
            return {}
        except Exception as e:
            print(f"\nFAILED: Build failed: {e}")
            import traceback
            traceback.print_exc()
            return {}
            
        # 3. Finalize
        # (Consolidation is handled by streamer)
        
        # Generate metadata
        result = {
            'output_dir': str(output_dir),
            'checkpoints_dir': str(output_dir / "checkpoints"),
            'metadata': {
                'streaming_mode': True,
                'unified_pipeline': True,
                'format': 'safetensors'
            }
        }
        
        return result
    
    def _sample_is_cached(self, sample, cache):
        """Check if sample is already cached."""
        if hasattr(sample, 'text'):
            text = sample.text
        elif hasattr(sample, 'question'):
            text = sample.question
        elif isinstance(sample, dict):
            text = sample.get('text', '') or sample.get('question', '')
        else:
            text = str(sample)
        
        return cache.get(text, 224, 224) is not None
    
    def _init_encoder(self):
        """Initialize CapsuleEncoder (shared by all encoding methods)."""
        if self.encoder is None:
            from models.capsule_encoder import CapsuleEncoder
            self.encoder = CapsuleEncoder(
                hidden_size=getattr(self.config, 'hidden_size', 768),
                target_capsules=getattr(self.config, 'target_capsules', 12),
                children_per_capsule=4,
                checksum_dim=32,
                num_layers=getattr(self.config, 'encoder_num_layers', 2),
                H_cycles=getattr(self.config, 'encoder_H_cycles', 2),
                L_cycles=getattr(self.config, 'encoder_L_cycles', 3),
                capsule_grid_shape=getattr(self.config, 'capsule_grid_shape', (3, 4))
            )
            self.encoder.eval()
            self.encoder.to('cuda' if torch.cuda.is_available() else 'cpu')
    

    
    def encode_to_capsules(self, samples: List[DataSample]) -> Dict[str, torch.Tensor]:
        """
        Unified encoding: text/image/grid → capsules.
        Single pass through CapsuleEncoder.
        """
        # Initialize encoder once (shared method)
        self._init_encoder()
        
        # Process in chunks to prevent RAM accumulation from text list
        batch_size = 4
        chunk_size = 500  # Consolidate every 500 samples
        
        # Initialize result tensors
        result_chunks = {'sketches': [], 'checksums': [], 'children': []}
        temp_data = {'sketches': [], 'checksums': [], 'children': []}
        
        # Process samples without creating full text list (prevents RAM leak)
        total_samples = len(samples)
        for i in tqdm(range(0, total_samples, batch_size), desc="Encoding capsules"):
            # Convert only current batch to images (critical for RAM)
            batch_images = [self._sample_to_image(s) for s in samples[i:i+batch_size]]
            
            with torch.no_grad():
                batch_result = self.encoder(images=batch_images, return_children=True)
            
            # Clear batch images immediately
            del batch_images
            
            # Move to CPU immediately and append
            for key in temp_data:
                if key in batch_result and batch_result[key] is not None:
                    temp_data[key].append(batch_result[key].cpu())
            
            del batch_result
            
            # Concatenate chunks periodically to free intermediate lists
            if len(temp_data['sketches']) >= chunk_size // batch_size:
                for key in temp_data:
                    if temp_data[key]:
                        result_chunks[key].append(torch.cat(temp_data[key], dim=0))
                        temp_data[key] = []  # Clear temp list
                
                gc.collect()  # Force garbage collection
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Final concatenation of remaining data
        for key in temp_data:
            if temp_data[key]:
                result_chunks[key].append(torch.cat(temp_data[key], dim=0))
        
        # Concatenate all chunks into final result
        result = {k: torch.cat(v, dim=0) if v else torch.empty(0) for k, v in result_chunks.items()}
        
        # Final cleanup
        del temp_data, result_chunks
        gc.collect()
        
        # Ensure at least 'sketches' key exists for compatibility
        if 'sketches' not in result or result['sketches'].numel() == 0:
            result['sketches'] = torch.empty(0, getattr(self.config, 'target_capsules', 12), getattr(self.config, 'hidden_size', 768))
        
        return result
    
    def _sample_to_image(self, sample: DataSample):
        """Convert any sample type to numpy array image for vision-unified encoder."""
        # Import TextRenderer lazily
        if self.text_renderer is None:
            from models.text_renderer import TextRenderer
            self.text_renderer = TextRenderer(
                width=getattr(self.config, 'text_image_width', 224),
                height=getattr(self.config, 'text_image_height', 224)
            )
        
        # Text → render to image → convert to numpy
        if sample.text:
            pil_img = self.text_renderer.render_plain_text(sample.text)
            return np.array(pil_img)
        
        # Grid → render to image → convert to numpy
        elif sample.grid is not None:
            grid_text = self._grid_to_text(sample.grid)
            pil_img = self.text_renderer.render_plain_text(grid_text)
            return np.array(pil_img)
        
        # Image → return as numpy array
        elif sample.image is not None:
            # Already numpy array
            if isinstance(sample.image, np.ndarray):
                return sample.image.astype(np.uint8)
            # Convert PIL to numpy
            return np.array(sample.image)
        
        # Fallback: blank image as numpy array
        return np.ones((224, 224, 3), dtype=np.uint8) * 255
    
    def _grid_to_text(self, grid: np.ndarray) -> str:
        """Convert 2D grid to text representation."""
        # Simple row-based serialization
        rows = []
        for row in grid:
            rows.append(' '.join(map(str, row)))
        return ' | '.join(rows)
    
    def create_metadata(self, train_data: Dict) -> PuzzleDatasetMetadata:
        """Generate dataset metadata."""
        num_concepts = getattr(self.config, 'num_concepts', 2048)
        vocab_size = num_concepts + 4  # Concept vocab
        
        if isinstance(train_data, dict):
            if 'sketches' in train_data:
                seq_len = train_data['sketches'].shape[1]
            else:
                seq_len = 1024  # Default
        else:
            seq_len = 1024  # Default
        
        if isinstance(train_data, dict):
            total_samples = len(train_data.get('sketches', []))
        else:
            total_samples = len(train_data) if train_data else 0
        
        return PuzzleDatasetMetadata(
            seq_len=seq_len,
            vocab_size=vocab_size,
            pad_id=0,
            ignore_label_id=-100,
            blank_identifier_id=0,
            num_puzzle_identifiers=0,
            total_groups=total_samples,
            mean_puzzle_examples=1.0,
            total_puzzles=total_samples,
            sets=["all"]
        )
    
    def save_dataset(self, dataset: Dict, output_dir: str):
        """Save dataset to disk with parallel I/O."""
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n💾 Saving dataset to: {output_dir}")
        
        # Prepare compressed datasets
        train_path = os.path.join(output_dir, 'capsule_dataset.pt')
        train_compressed = {
            'sketches': dataset['train']['sketches'].half() if 'sketches' in dataset['train'] else dataset['train'].get('sketches'),
            'checksums': dataset['train'].get('checksums'),
        }
        if 'children' in dataset['train']:
            train_compressed['children'] = dataset['train']['children'].half()
        for key in dataset['train']:
            if key not in train_compressed:
                train_compressed[key] = dataset['train'][key]
        
        test_path = os.path.join(output_dir, 'capsule_dataset_test.pt')
        test_compressed = {
            'sketches': dataset['test']['sketches'].half() if 'sketches' in dataset['test'] else dataset['test'].get('sketches'),
            'checksums': dataset['test'].get('checksums'),
        }
        if 'children' in dataset['test']:
            test_compressed['children'] = dataset['test']['children'].half()
        for key in dataset['test']:
            if key not in test_compressed:
                test_compressed[key] = dataset['test'][key]
        
        # Parallel save with ThreadPoolExecutor + LZ4 compression (C++ backend)
        def save_worker(data, path, name):
            try:
                # Use LZ4 compression for 3-5x faster I/O
                try:
                    import lz4.frame
                    import pickle
                    # Serialize with pickle, compress with LZ4 (C++ backend)
                    with lz4.frame.open(path + '.lz4', 'wb') as f:
                        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                    # Rename to .pt for compatibility
                    import os
                    os.rename(path + '.lz4', path)
                except ImportError:
                    # Fallback to standard torch.save
                    torch.save(data, path)
                return name, path, None
            except Exception as e:
                return name, path, str(e)
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_train = executor.submit(save_worker, train_compressed, train_path, "train")
            future_test = executor.submit(save_worker, test_compressed, test_path, "test")
            
            # Wait for both to complete and check errors
            train_name, train_saved, train_error = future_train.result()
            test_name, test_saved, test_error = future_test.result()
            
            if train_error:
                raise RuntimeError(f"Failed to save {train_name}: {train_error}")
            if test_error:
                raise RuntimeError(f"Failed to save {test_name}: {test_error}")
        
        print(f"Saved {train_name} (float16 compressed): {train_saved}")
        print(f"Saved {test_name} (float16 compressed): {test_saved}")
        
        # Save metadata
        metadata_path = os.path.join(output_dir, 'dataset.json')
        with open(metadata_path, 'w') as f:
            json.dump(dataset.get('metadata', {}).__dict__ if hasattr(dataset.get('metadata'), '__dict__') else dataset.get('metadata', {}), f, indent=2)
        print(f"Saved metadata: {metadata_path}")

import os
import sys
import json
import time
import shutil
import hashlib
import pickle
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import multiprocessing as mp
from typing import List, Optional, Tuple, Union, Any, Dict
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

class ImageCache:
    """
    Persistent disk-based cache for rendered text images.
    
    Features:
    - SHA256 hashing of text content for unique keys
    - 2-level directory structure to avoid filesystem limits (ab/abcdef...)
    - Metadata tracking (hit rates, total size)
    - Multiprocessing support for parallel rendering
    - Automatic denoising of rendered images (optional)
    """
    
    def __init__(self, cache_dir: str = "text_cache", num_workers: int = 10):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.num_workers = num_workers
        
        # Metadata
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
        
        # Denoising (loaded lazily/per-worker)
        self.denoiser = None
        self.device = 'cpu'
        
        # Multiprocessing pool (lazy init)
        self.pool = None
        
        print(f"Image Cache initialized at: {self.cache_dir}")
        print(f"   Entries: {self.metadata.get('count', 0):,}")
        print(f"   Size: {self.metadata.get('size_mb', 0):.2f} MB")

    def _get_cache_key(self, text: str, width: int, height: int) -> str:
        """Generate unique hash for text + dimensions."""
        content = f"{text}_{width}_{height}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _get_cache_path(self, key: str) -> Path:
        """Get file path with 2-level sharding (e.g., ab/abcdef...)."""
        # Use first 2 chars for subdirectory to avoid 100k+ files in one dir
        subdir = self.cache_dir / key[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{key}.npy"

    def put(self, text: str, width: int, height: int, image_array: np.ndarray) -> None:
        """Save image to cache (compressed numpy)."""
        key = self._get_cache_key(text, width, height)
        path = self._get_cache_path(key)
        
        # Save as compressed numpy (uint8) to save space
        # .npy is faster than PNG for loading
        np.save(path, image_array.astype(np.uint8))
        
        # Update metadata (approximate)
        if not path.exists():
            self.metadata['count'] = self.metadata.get('count', 0) + 1

    def get(self, text: str, width: int, height: int) -> Optional[np.ndarray]:
        """Retrieve image from cache if exists."""
        key = self._get_cache_key(text, width, height)
        path = self._get_cache_path(key)
        
        if path.exists():
            try:
                return np.load(path)
            except Exception:
                return None
        return None

    def has_been_cached(self, text: str, width: int, height: int) -> bool:
        """Check if text is already in cache (fast check)."""
        key = self._get_cache_key(text, width, height)
        path = self._get_cache_path(key)
        return path.exists()

    def _init_denoiser_mandatory(self) -> None:
        """Initialize denoiser - MANDATORY for all rendered images."""
        if self.denoiser is None:
            try:
                # Try importing fastdncnn first
                from models.fastdncnn import FastDNCNN
                self.denoiser = FastDNCNN(in_nc=1, out_nc=1, nc=64, nb=17, act_mode='BR')
                
                # Load pretrained weights
                weights_path = Path("weights/fastdncnn_gray.pth")
                if weights_path.exists():
                    self.denoiser.load_state_dict(torch.load(weights_path, map_location='cpu'))
                else:
                    print(f"WARNING: Denoiser weights not found at {weights_path}, using random init (suboptimal)")
                
                self.denoiser.eval()
                # Keep on CPU for workers to avoid CUDA context issues in multiprocessing
                self.denoiser.to('cpu') 
            except ImportError:
                print("ERROR: Could not import FastDNCNN. Denoising disabled (quality may suffer).")
                self.denoiser = None
            except Exception as e:
                print(f"Error initializing denoiser: {e}")
                self.denoiser = None

    def _denoise_image(self, img_array: np.ndarray) -> np.ndarray:
        """Apply denoising to rendered image."""
        if self.denoiser is None:
            self._init_denoiser_mandatory()
            
        if self.denoiser is None:
            return img_array

        # Prepare for model (H,W,C) -> (1,1,H,W)
        # Assuming grayscale/binary rendering for now, but keeping 3 channels
        img_tensor = torch.from_numpy(img_array).float() / 255.0
        if img_tensor.ndim == 3 and img_tensor.shape[2] == 3:
            # Convert to grayscale for denoising
            gray = 0.299 * img_tensor[:,:,0] + 0.587 * img_tensor[:,:,1] + 0.114 * img_tensor[:,:,2]
            gray = gray.unsqueeze(0).unsqueeze(0) # (1,1,H,W)
        else:
            gray = img_tensor.permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            denoised = self.denoiser(gray)
        
        # Convert back
        denoised = denoised.squeeze().clamp(0, 1).numpy()
        denoised = (denoised * 255).astype(np.uint8)
        
        # Restore to 3 channels
        return np.stack([denoised]*3, axis=-1)

    def populate_cache(self, samples: List[Any], renderer_fn, batch_size: int = 100) -> Tuple[int, int]:
        """
        Populate cache for a list of samples.
        Returns (cached_count, newly_rendered_count).
        """
        total = len(samples)
        cached_count = 0
        newly_rendered = 0
        
        # Filter out already cached
        uncached_samples = []
        print(f"🔍 Checking cache for {total} samples...")
        
        for sample in tqdm(samples, desc="Checking cache"):
            # Extract text
            if hasattr(sample, 'text'):
                text = sample.text
            elif hasattr(sample, 'question'):
                text = sample.question
            elif isinstance(sample, dict):
                text = sample.get('text', '') or sample.get('question', '')
            else:
                text = str(sample)
            
            if self.has_been_cached(text, 224, 224):
                cached_count += 1
            else:
                uncached_samples.append(sample)
        
        if not uncached_samples:
            return cached_count, 0
            
        print(f"Rendering {len(uncached_samples)} new samples...")
        
        # Render in parallel
        self._render_samples_parallel(uncached_samples)
        newly_rendered = len(uncached_samples)
        
        self._save_metadata()
        return cached_count, newly_rendered

    def _get_or_create_pool(self) -> mp.Pool:
        """Get existing pool or create new one."""
        if self.pool is None:
            # Use 'spawn' context for compatibility with PyTorch
            ctx = mp.get_context('spawn')
            self.pool = ctx.Pool(processes=self.num_workers)
        return self.pool

    def _render_samples_parallel(self, samples: List[Any], num_workers: Optional[int] = None) -> List[Tuple[Optional[np.ndarray], str]]:
        """Render samples in parallel (helper for streaming)."""
        # For very small batches, multiprocessing overhead exceeds benefit
        if len(samples) < 50:
            # Single-threaded rendering for small batches
            from models.text_renderer import TextRenderer
            renderer = TextRenderer(width=224, height=224)
            
            results = []
            for sample in samples:
                # Extract text
                if hasattr(sample, 'text'):
                    text = sample.text
                elif hasattr(sample, 'question'):
                    text = sample.question
                elif isinstance(sample, dict):
                    text = sample.get('text', '') or sample.get('question', '') or str(sample)
                else:
                    text = str(sample)
                
                # Check if already cached
                key = self._get_cache_key(text, 224, 224)
                path = self._get_cache_path(key)
                
                if path.exists():
                    results.append((None, text))
                else:
                    pil_img = renderer.render_plain_text(text)
                    img_array = np.array(pil_img)
                    results.append((img_array, text))
            
            # Save rendered images
            for img_array, text in results:
                if img_array is not None:
                    self.put(text, 224, 224, img_array)
            
            return results
        
        # For larger batches, use persistent pool (reuse workers)
        args_list = [(sample, 224, 224) for sample in samples]
        pool = self._get_or_create_pool()
        
        # Use map with chunksize for better load balancing
        chunksize = max(1, len(samples) // (self.num_workers * 4))
        results = pool.map(self._render_sample, args_list, chunksize=chunksize)
        
        # CRITICAL: Save rendered images to cache
        for img_array, text in results:
            if img_array is not None:  # None means already cached
                self.put(text, 224, 224, img_array)
        
        return results

    @staticmethod
    def _render_sample(args: Tuple[Any, int, int]) -> Tuple[Optional[np.ndarray], str]:
        """Static worker method for multiprocessing."""
        sample, width, height = args
        
        # Extract text
        if hasattr(sample, 'text'):
            text = sample.text
        elif hasattr(sample, 'question'):
            text = sample.question
        elif isinstance(sample, dict):
            text = sample.get('text', '') or sample.get('question', '') or str(sample)
        else:
            text = str(sample)
            
        # Re-instantiate renderer in worker process
        # This is fast enough and avoids pickling complex objects
        from models.text_renderer import TextRenderer
        renderer = TextRenderer(width=width, height=height)
        
        try:
            pil_img = renderer.render_plain_text(text)
            return np.array(pil_img), text
        except Exception as e:
            # Return blank image on failure
            print(f"Render failed for '{text[:20]}...': {e}")
            return np.ones((height, width, 3), dtype=np.uint8) * 255, text

    def _save_metadata(self) -> None:
        """Save cache metadata to disk."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f)
        except Exception:
            pass

    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {'count': 0, 'size_mb': 0.0}

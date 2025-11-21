import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Dict, Any, Optional

# Pre-initialize transforms to avoid overhead per batch
_TO_TENSOR = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor()
])

_TO_TENSOR_GRID = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor()
])

def process_vision_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a batch of raw samples into tensors for the model.
    Handles images, grids (ARC/Maze), and text placeholders.
    """
    if not isinstance(batch, dict) or 'raw_samples' not in batch:
        return batch
        
    raw_samples = batch['raw_samples']
    
    input_images = []
    output_images = []
    
    for sample in raw_samples:
        # --- Input Processing ---
        img = None
        is_grid = False
        
        # 1. Image
        if sample.image is not None:
            if isinstance(sample.image, np.ndarray):
                img = Image.fromarray(sample.image.astype(np.uint8))
            else:
                img = sample.image
                
        # 2. Grid (ARC/Maze)
        elif sample.grid is not None:
            is_grid = True
            grid = sample.grid
            if grid.dtype == np.int32 or grid.dtype == np.int64:
                # Normalize 0-255
                grid_norm = ((grid - grid.min()) / (grid.max() - grid.min() + 1e-8) * 255).astype(np.uint8)
                img = Image.fromarray(grid_norm).convert('RGB')
            else:
                img = Image.fromarray((grid * 255).astype(np.uint8)).convert('RGB')
                
        # 3. Text / Empty
        else:
            # Placeholder
            img = Image.new('RGB', (224, 224), color=(128, 128, 128) if sample.text else (0, 0, 0))
            
        # Convert to tensor
        if is_grid:
            input_tensor = _TO_TENSOR_GRID(img)
        else:
            input_tensor = _TO_TENSOR(img)
        input_images.append(input_tensor)
        
        # --- Output Processing ---
        if sample.label is not None and isinstance(sample.label, np.ndarray) and sample.label.ndim == 2:
            # Label is a grid
            label_grid = ((sample.label - sample.label.min()) / (sample.label.max() - sample.label.min() + 1e-8) * 255).astype(np.uint8)
            out_img = Image.fromarray(label_grid).convert('RGB')
            output_images.append(_TO_TENSOR_GRID(out_img))
        else:
            # No output image (or not a grid)
            output_images.append(torch.zeros_like(input_tensor))
            
    batch['target_images'] = torch.stack(output_images)
    batch['images'] = torch.stack(input_images)
    
    return batch


def detect_dataset_features(data_path: str) -> Dict[str, Any]:
    """Auto-detect dataset type and features from data."""
    import os
    
    features = {
        'is_capsule': False,
        'is_vision': False,
        'is_text': False,
        'has_checksums': False,
        'has_children': False,
        'enable_dqn': False,
        'enable_expansion': False
    }
    
    # Try loading as capsule dataset
    capsule_paths = [
        data_path.replace('semantic_embeddings', 'capsule_dataset'),
        os.path.join(data_path, 'capsule_dataset.pt')
    ]
    
    for path in capsule_paths:
        if os.path.exists(path) and os.path.isfile(path):
            try:
                # Attempt to load with mmap if possible, or just check file size
                # If file is small (<1GB), try loading keys. If large, assume it has features based on name.
                file_size = os.path.getsize(path)
                if file_size < 1024 * 1024 * 1024:  # < 1GB
                    # Use weights_only=False because it's a dictionary, but map to CPU
                    data = torch.load(path, map_location='cpu', weights_only=False)
                    if isinstance(data, dict) and 'sketches' in data:
                        features['is_capsule'] = True
                        features['has_checksums'] = 'checksums' in data
                        features['has_children'] = 'children' in data
                        features['enable_expansion'] = features['has_children']
                        features['enable_dqn'] = features['enable_expansion']
                        del data # Free memory immediately
                        return features
                else:
                    # Large file, assume standard capsule format if name matches
                    print(f"[INFO] Detected large capsule dataset ({file_size/1e9:.2f} GB). Assuming standard features.")
                    features['is_capsule'] = True
                    features['has_checksums'] = True # Assume standard format has these
                    features['has_children'] = True
                    features['enable_expansion'] = True
                    features['enable_dqn'] = True
                    return features
            except Exception as e:
                print(f"[WARN] Failed to probe dataset features: {e}")
                pass
    
    # Detect from path patterns
    path_lower = data_path.lower()
    if 'arc' in path_lower or 'vision' in path_lower or 'cifar' in path_lower or 'image' in path_lower:
        features['is_vision'] = True
    elif 'text' in path_lower or 'wikitext' in path_lower or 'stories' in path_lower:
        features['is_text'] = True
    
    return features

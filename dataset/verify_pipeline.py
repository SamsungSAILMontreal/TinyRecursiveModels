import sys
import os
import shutil
from pathlib import Path
import numpy as np

# Add parent directory to path
import sys
import os
import shutil
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.build_multimodal_dataset import MultimodalDatasetConfig, MultimodalDatasetBuilder, build
from dataset.image_cache import ImageCache

def test_image_cache():
    print(f"\nTesting ImageCache...")
    cache_dir = "test_cache"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    
    cache = ImageCache(cache_dir=cache_dir)
    
    # Test put/get
    text = "Hello World"
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    cache.put(text, 224, 224, img)
    assert cache.has_been_cached(text, 224, 224)
    
    retrieved = cache.get(text, 224, 224)
    assert retrieved is not None
    assert np.array_equal(img, retrieved)
    
    print(f"ImageCache test passed")
    
    # Cleanup
    shutil.rmtree(cache_dir)

def test_pipeline():
    print(f"\nTesting Dataset Pipeline...")
    output_dir = "test_dataset_output"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    # Create dummy data
    os.makedirs("test_data", exist_ok=True)
    with open("test_data/sample.txt", "w") as f:
        f.write("This is a test sample.\nAnother line.")
        
    config = MultimodalDatasetConfig(
        source_paths=["test_data/sample.txt"],
        output_dir=output_dir,
        include_text=True,
        include_images=False,
        include_grids=False,
        render_text_to_image=True
    )
    
    # Run build
    try:
        build(config)
        print(f"Build function ran successfully")
        
        # Check outputs
        train_meta = os.path.join(output_dir, "train", "dataset.json")
        if os.path.exists(train_meta):
            print(f"Output metadata found")
            
            # Verify Safetensors
            checkpoints_dir = os.path.join(output_dir, "checkpoints")
            if os.path.exists(checkpoints_dir):
                files = list(Path(checkpoints_dir).glob("*.safetensors"))
                if files:
                    print(f"Found {len(files)} safetensors files")
                    
                    # Try loading one
                    from safetensors.torch import load_file
                    import json
                    
                    st_path = files[0]
                    tensors = load_file(st_path)
                    print(f"   Loaded {st_path.name}: {list(tensors.keys())}")
                    
                    # Check sidecar json (format is batch_XXXXX_meta.json)
                    json_path = st_path.parent / f"{st_path.stem}_meta.json"
                    if json_path.exists():
                        try:
                            with open(json_path, 'r') as f:
                                content = f.read()
                                meta = json.loads(content)
                            print(f"   Loaded sidecar metadata: {list(meta.keys())}")
                        except json.JSONDecodeError as e:
                            print(f"JSON Decode Error: {e}")
                            print(f"   File content: {content}")
                    else:
                        print(f"Sidecar JSON missing: {json_path}")
                        
                    # Cleanup tensors to release file handles
                    del tensors
                    import gc
                    gc.collect()
                        
                else:
                    print(f"No safetensors files found in checkpoints dir")
            else:
                print(f"Checkpoints directory missing")
                
        else:
            print(f"Output metadata missing")
            
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    if os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir)
        except PermissionError:
            print(f"Could not remove output directory (files in use)")
    if os.path.exists("test_data"):
        shutil.rmtree("test_data")

if __name__ == "__main__":
    test_image_cache()
    test_pipeline()

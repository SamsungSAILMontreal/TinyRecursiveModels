import tarfile
import os
import shutil
import glob
import sys

# Configuration
BASE_DIR = 'model-dqn-snn'
TARGET_DIR = os.path.join(BASE_DIR, 'datasets', 'vision_unified')
CHECKPOINTS_DIR = os.path.join(TARGET_DIR, 'checkpoints')
TEXT_CACHE_DIR = os.path.join(TARGET_DIR, 'text_cache')

print(f"🚀 Starting selective export for {BASE_DIR}...")
print(f"   Target: {CHECKPOINTS_DIR}")
print(f"   Target: {TARGET_DIR}/consolidated*")

# 1. Cleanup .npy files in text_cache to save space (as requested)
print('\n🧹 Cleaning up .npy files in text_cache...')
npy_pattern = os.path.join(TEXT_CACHE_DIR, '**', '*.npy')
npy_files = glob.glob(npy_pattern, recursive=True)
for f in npy_files:
    try:
        os.remove(f)
    except OSError:
        pass
print(f'   Deleted {len(npy_files)} .npy files')

# 2. Helper functions
def get_all_subdirs(base_path):
    """Get all subdirectories sorted by depth (deepest first)."""
    all_dirs = []
    if not os.path.exists(base_path):
        return []
        
    for root, dirs, files in os.walk(base_path):
        for d in dirs:
            full_path = os.path.join(root, d)
            depth = full_path.count(os.sep)
            all_dirs.append((depth, full_path))
    # Sort by depth (deepest first) so we delete leaf directories first
    all_dirs.sort(reverse=True, key=lambda x: x[0])
    return [path for _, path in all_dirs]

def get_size_mb(path):
    """Get total size of directory or file in MB."""
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024**2)
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total / (1024**2)

# 3. Create Tar
tar_name = 'model-dqn-snn-checkpoints.tar'
print(f'\n📦 Creating {tar_name} with space-saving mode...')

with tarfile.open(tar_name, 'w') as tar:
    # A. Process Checkpoints Directory (Deepest First)
    if os.path.exists(CHECKPOINTS_DIR):
        print(f'\n   Processing Checkpoints: {CHECKPOINTS_DIR}')
        subdirs = get_all_subdirs(CHECKPOINTS_DIR)
        
        # If no subdirs, just add the files in the dir
        if not subdirs:
            # Add files in checkpoints dir
            for item in os.listdir(CHECKPOINTS_DIR):
                item_path = os.path.join(CHECKPOINTS_DIR, item)
                if os.path.isfile(item_path):
                    size = get_size_mb(item_path)
                    rel_path = os.path.relpath(item_path, '.')
                    print(f'   Adding file: {rel_path} ({size:.1f} MB)')
                    tar.add(item_path, arcname=rel_path)
                    os.remove(item_path)
        else:
            # Add subdirectories (checkpoints usually are dirs)
            for i, dir_path in enumerate(subdirs, 1):
                if not os.path.exists(dir_path): continue
                
                rel_path = os.path.relpath(dir_path, '.')
                dir_size = get_size_mb(dir_path)
                
                print(f'   [{i}/{len(subdirs)}] {rel_path} (~{dir_size:.0f}MB)')
                tar.add(dir_path, arcname=rel_path)
                shutil.rmtree(dir_path) # Delete immediately to free space
                print(f'         ✅ Added & Freed')
        
        # Finally remove the empty checkpoints dir itself
        if os.path.exists(CHECKPOINTS_DIR):
            tar.add(CHECKPOINTS_DIR, arcname=os.path.relpath(CHECKPOINTS_DIR, '.'))
            # shutil.rmtree(CHECKPOINTS_DIR) # Optional: keep empty dir structure
    else:
        print(f'   ⚠️ Checkpoints directory not found: {CHECKPOINTS_DIR}')

    # B. Process Consolidated Files
    print(f'\n   Processing Consolidated Files...')
    # Match consolidated* in vision_unified directory
    consolidated_pattern = os.path.join(TARGET_DIR, 'consolidated*')
    consolidated_items = glob.glob(consolidated_pattern)
    
    for item_path in consolidated_items:
        rel_path = os.path.relpath(item_path, '.')
        size = get_size_mb(item_path)
        
        print(f'   Adding: {rel_path} (~{size:.0f}MB)')
        tar.add(item_path, arcname=rel_path)
        
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
        print(f'         ✅ Added & Freed')

print(f'\n✅ Archive created: {tar_name}')
print('   Downloading...')

try:
    from google.colab import files
    files.download(tar_name)
except ImportError:
    print("   (Not running in Colab, skipping download)")

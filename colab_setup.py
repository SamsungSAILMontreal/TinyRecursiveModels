# === Colab Setup with Local Tar Transfer (No Drive) ===
import os, shutil, tarfile
from pathlib import Path
# Handle running locally vs Colab
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    print("⚠️ Not running in Colab - file upload features disabled")

print("="*70)
print("🛠️  Colab Setup - Local Transfer Mode")
print("   No Google Drive needed")
print("   Transfer via tar upload/download (faster, no compression)")
print("="*70 + "\n")

# 1. Clone or update repo
if IN_COLAB:
    os.chdir('/content')
    if os.path.exists('model-dqn-snn'):
        os.chdir('model-dqn-snn')
        os.system('git reset --hard HEAD')
        os.system('git pull origin main')
        print("✅ Repo updated")
    else:
        os.system('git clone https://github.com/saoxdxd2/model-dqn-snn.git')
        os.chdir('model-dqn-snn')
        print("✅ Repo cloned")

# 2. Check for existing data or upload from local machine
checkpoints_dir = Path('model_checkpoints')
dataset_chunks_dir = Path('datasets/vision_unified/stream_checkpoints')

# Check what we already have
has_model_ckpt = checkpoints_dir.exists()
has_dataset_chunks = dataset_chunks_dir.exists() and list(dataset_chunks_dir.glob('consolidated_*.safetensors'))

if not has_model_ckpt and not has_dataset_chunks:
    print("\n📤 No checkpoints or dataset chunks found.")
    print("   Option 1: Upload model-dqn-snn.tar (contains model_checkpoints/ and/or datasets/)")
    print("   Option 2: Upload individual consolidated_*.safetensors and .json files")
    
    if IN_COLAB:
        response = input("Upload tar? (y/n): ").lower()
        
        if response == 'y':
            print("📦 Upload your model-dqn-snn.tar file...")
            uploaded = files.upload()
            
            if uploaded:
                tar_filename = list(uploaded.keys())[0]
                print(f"\n📜 Extracting {tar_filename}...")
                
                with tarfile.open(tar_filename, 'r') as tar:
                    tar.extractall('/content/temp_extract')
                
                temp_base = Path('/content/temp_extract/model-dqn-snn')
                
                # Restore model checkpoints
                extracted_checkpoints = temp_base / 'model_checkpoints'
                if extracted_checkpoints.exists():
                    shutil.copytree(extracted_checkpoints, checkpoints_dir, dirs_exist_ok=True)
                    print("✅ Restored model checkpoints")
                
                # Restore dataset consolidated chunks (pre-encoded capsules)
                extracted_dataset = temp_base / 'datasets'
                if extracted_dataset.exists():
                    Path('datasets').mkdir(exist_ok=True)
                    shutil.copytree(extracted_dataset, 'datasets', dirs_exist_ok=True)
                    
                    # Count consolidated chunks
                    chunks = list(dataset_chunks_dir.glob('consolidated_*.safetensors')) if dataset_chunks_dir.exists() else []
                    if chunks:
                        total_gb = sum(c.stat().st_size for c in chunks) / (1024**3)
                        print(f"✅ Restored {len(chunks)} pre-encoded chunk(s) ({total_gb:.2f}GB)")
                        print(f"   🚀 Training can start immediately on existing data!")
                
                # Cleanup
                shutil.rmtree('/content/temp_extract')
                os.remove(tar_filename)
            else:
                print("ℹ️  No file uploaded")
        else:
            # Option 2: Individual files
            print("\n📤 Upload individual consolidated files? (e.g. consolidated_000.safetensors + .json)")
            response_files = input("Upload files? (y/n): ").lower()
            
            if response_files == 'y':
                print("📦 Upload your consolidated_*.safetensors and _meta.json files...")
                print("   (Select multiple files in the dialog)")
                uploaded = files.upload()
                
                if uploaded:
                    dataset_chunks_dir.mkdir(parents=True, exist_ok=True)
                    count = 0
                    for filename in uploaded.keys():
                        # Move file to dataset directory
                        # In Colab, uploads go to CWD (/content/model-dqn-snn)
                        if filename.startswith("consolidated_") and (filename.endswith(".safetensors") or filename.endswith(".json")):
                            shutil.move(filename, dataset_chunks_dir / filename)
                            count += 1
                            print(f"   Moved {filename}")
                    
                    print(f"✅ Processed {count} files")
                    
                    # Verify
                    chunks = list(dataset_chunks_dir.glob('consolidated_*.safetensors'))
                    if chunks:
                        total_gb = sum(c.stat().st_size for c in chunks) / (1024**3)
                        print(f"✅ Found {len(chunks)} chunk(s) ({total_gb:.2f}GB)")
                        print(f"   🚀 Training can start on uploaded data!")
                else:
                    print("ℹ️  No files uploaded")
            else:
                print("ℹ️  Skipping upload, starting fresh")
    else:
        print("ℹ️  Local run detected, skipping upload prompts")

else:
    if has_model_ckpt:
        print("✅ Found existing model_checkpoints/")
    if has_dataset_chunks:
        chunks = list(dataset_chunks_dir.glob('consolidated_*.safetensors'))
        total_gb = sum(c.stat().st_size for c in chunks) / (1024**3)
        print(f"✅ Found {len(chunks)} pre-encoded chunk(s) ({total_gb:.2f}GB)")
        print(f"   🚀 Training will start on existing data!")

# 3. Install dependencies
print("\n📦 Installing dependencies from requirements.txt...")
print("   This may take 2-3 minutes on first run\n")
if IN_COLAB:
    os.system('pip install --no-cache-dir -r requirements.txt')
else:
    print("   (Skipping pip install in local mode)")
print("\n✅ Dependencies installed")

# 4. Start training
print("\n" + "="*70)
print("🚀 Starting Training - Hybrid Pretrained Pipeline")
print("   Config: hybrid_pretrained.yaml (CLIP+ViT+N2N+TRM+COCONUT)")
print("   Resume: Auto-continues from model_checkpoints/")
print("="*70 + "\n")
# !python train.py
if IN_COLAB:
    os.system('python train.py')

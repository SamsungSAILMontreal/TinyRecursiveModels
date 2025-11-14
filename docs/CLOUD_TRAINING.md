# Cloud Training Guide for TinyRecursiveInference

This guide covers training TRM (Tiny Recursion Model) on various cloud GPU platforms. The 7M parameter model is small enough to train on a single GPU, making it accessible and cost-effective.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Platform-Specific Guides](#platform-specific-guides)
   - [Google Colab](#google-colab)
   - [HuggingFace Spaces/Jobs](#huggingface-spaces)
   - [Prime Intellect](#prime-intellect)
   - [AWS EC2](#aws-ec2)
   - [Google Cloud Platform](#google-cloud-platform)
   - [Azure ML](#azure-ml)
3. [Configuration Guide](#configuration-guide)
4. [GPU Requirements](#gpu-requirements)
5. [Cost Estimates](#cost-estimates)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Automated Setup (Recommended)

```bash
# Clone repository
git clone https://github.com/HarleyCoops/TinyRecursiveInference.git
cd TinyRecursiveInference

# Run automated setup
bash scripts/setup_cloud_training.sh

# Follow prompts to configure your environment
```

The setup script will:
- Detect your GPU and recommend optimal configuration
- Install dependencies
- Prepare datasets
- Configure experiment tracking (W&B)

### Manual Setup

```bash
# Install dependencies
pip install --upgrade pip wheel setuptools
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Prepare dataset
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1-cloud-aug-100 \
  --subsets training evaluation \
  --test-set-name evaluation \
  --num-aug 100

# Start training
python pretrain.py \
  --config-name cfg_cloud_single_gpu \
  arch=trm \
  +run_name="my_experiment"
```

---

## Platform-Specific Guides

### Google Colab

**Best for:** Quick experimentation, learning, prototyping

**GPU Options:**
- Free tier: T4 (16GB) - Limited hours
- Colab Pro: V100 (16GB) or A100 (40GB)
- Colab Pro+: A100 (40GB) - Extended hours

**Setup:**

1. **Upload Notebook**
   - Open [Google Colab](https://colab.research.google.com/)
   - Upload `notebooks/train_colab.ipynb` from this repository
   - Or use this direct link: [Open in Colab](https://colab.research.google.com/github/HarleyCoops/TinyRecursiveInference/blob/main/notebooks/train_colab.ipynb)

2. **Enable GPU**
   - Runtime → Change runtime type → Hardware accelerator → GPU

3. **Run All Cells**
   - The notebook handles everything automatically

**Estimated Training Time:**
- T4 (free): ~6-8 hours for 10K epochs
- A100 (Pro+): ~1-2 hours for 10K epochs

**Tips:**
- Keep browser tab active to prevent disconnection
- Save checkpoints frequently (Colab times out after 12 hours)
- Use reduced augmentation (`--num-aug 100`) for faster setup

**Cost:** $0 (free tier) to $50/month (Pro+)

---

### HuggingFace Spaces

**Best for:** Deployment, sharing demos, inference-only

**Note:** HuggingFace Spaces are primarily for hosting inference applications. For training, use HuggingFace Jobs (see below).

**Setup for Inference Demo:**

```bash
# After training, deploy Gradio app to HF Space
pip install huggingface_hub
huggingface-cli login

# Create Space
python -c "
from huggingface_hub import create_repo
create_repo('my-trm-demo', repo_type='space', space_sdk='gradio')
"

# Push Gradio app
git clone https://huggingface.co/spaces/YOUR_USERNAME/my-trm-demo
cd my-trm-demo
cp -r /path/to/TinyRecursiveInference/tiny_recursive_inference/*.py .
cp /path/to/checkpoints ./checkpoints
git add .
git commit -m "Add TRM inference app"
git push
```

---

### HuggingFace Jobs (Coming Soon)

**Best for:** Managed training, reproducible experiments

HuggingFace is developing a training jobs feature. Check [HuggingFace Jobs documentation](https://huggingface.co/docs/hub/jobs) for updates.

**Expected workflow:**
```bash
# Create training job configuration
huggingface-cli jobs create \
  --name trm-training \
  --gpu a100 \
  --script pretrain.py \
  --config config/cfg_cloud_high_memory.yaml
```

---

### Prime Intellect

**Best for:** Large-scale distributed training, multi-GPU/multi-node

**GPU Options:**
- Single GPU: A100, H100
- Multi-GPU: 2-8 GPUs per node
- Multi-node: Distributed across data centers

**Setup:**

1. **Sign up at [Prime Intellect](https://primeintellect.ai/)**

2. **Install Prime Intellect CLI**
   ```bash
   pip install primeintellect
   primeintellect login
   ```

3. **Create Training Job**
   ```bash
   # Single GPU
   primeintellect submit \
     --gpu a100 \
     --gpus 1 \
     --script pretrain.py \
     --args "--config-name cfg_cloud_high_memory arch=trm +run_name=prime_experiment"

   # Multi-GPU (4x A100)
   primeintellect submit \
     --gpu a100 \
     --gpus 4 \
     --script pretrain.py \
     --args "--config-name cfg_pretrain arch=trm +run_name=prime_multi_gpu" \
     --distributed torchrun
   ```

4. **Monitor Training**
   ```bash
   primeintellect logs <job_id>
   ```

**For Multi-Node Training:**

See [AGENTS.md](../AGENTS.md) for detailed multi-node setup instructions.

**Cost:** Pay-per-use, typically $1-3/GPU-hour for A100

---

### AWS EC2

**Best for:** Full control, custom infrastructure, production deployments

**Recommended Instance Types:**
- `g4dn.xlarge` - T4 GPU, 16GB VRAM (~$0.50/hour)
- `p3.2xlarge` - V100 GPU, 16GB VRAM (~$3/hour)
- `p4d.24xlarge` - 8x A100 80GB (~$32/hour)

**Setup:**

1. **Launch EC2 Instance**
   ```bash
   # Use AWS CLI or Console
   aws ec2 run-instances \
     --image-id ami-0c55b159cbfafe1f0 \  # Deep Learning AMI
     --instance-type g4dn.xlarge \
     --key-name your-key \
     --security-group-ids sg-xxxxx \
     --subnet-id subnet-xxxxx
   ```

2. **Connect and Setup**
   ```bash
   ssh -i your-key.pem ubuntu@<instance-ip>

   # Clone and setup
   git clone https://github.com/HarleyCoops/TinyRecursiveInference.git
   cd TinyRecursiveInference
   bash scripts/setup_cloud_training.sh
   ```

3. **Start Training**
   ```bash
   # Use tmux or screen for persistent sessions
   tmux new -s training

   python pretrain.py \
     --config-name cfg_cloud_single_gpu \
     arch=trm \
     +run_name="aws_experiment"

   # Detach: Ctrl+B, then D
   # Reattach: tmux attach -t training
   ```

4. **Setup Auto-Shutdown (Save Costs)**
   ```bash
   # Add to cron to shutdown after training completes
   echo "0 * * * * [ -f /tmp/training_done ] && sudo shutdown -h now" | crontab -

   # In training script, create flag when done:
   # touch /tmp/training_done
   ```

**Cost:** $0.50-$32/hour depending on instance type

---

### Google Cloud Platform

**Best for:** Preemptible GPUs (cost-effective), integration with GCP services

**Recommended Instance Types:**
- `n1-standard-4` + `nvidia-tesla-t4` (~$0.35/hour)
- `n1-standard-8` + `nvidia-tesla-v100` (~$2.50/hour)
- `a2-highgpu-1g` (1x A100) (~$3.67/hour)

**Setup:**

1. **Create Instance**
   ```bash
   gcloud compute instances create trm-training \
     --zone=us-central1-a \
     --machine-type=n1-standard-8 \
     --accelerator=type=nvidia-tesla-v100,count=1 \
     --image-family=pytorch-latest-gpu \
     --image-project=deeplearning-platform-release \
     --maintenance-policy=TERMINATE \
     --boot-disk-size=100GB
   ```

2. **SSH and Setup**
   ```bash
   gcloud compute ssh trm-training

   git clone https://github.com/HarleyCoops/TinyRecursiveInference.git
   cd TinyRecursiveInference
   bash scripts/setup_cloud_training.sh
   ```

3. **Use Preemptible Instances (Save 70%)**
   ```bash
   # Add --preemptible flag
   gcloud compute instances create trm-training-preemptible \
     --preemptible \
     --zone=us-central1-a \
     --machine-type=n1-standard-8 \
     --accelerator=type=nvidia-tesla-v100,count=1 \
     --image-family=pytorch-latest-gpu \
     --image-project=deeplearning-platform-release

   # Setup automatic checkpoint resumption
   # (Preemptible instances can be terminated anytime)
   ```

**Cost:** $0.35-$3.67/hour (regular), 70% less for preemptible

---

### Azure ML

**Best for:** Enterprise, integration with Azure services, MLOps

**Recommended VM Sizes:**
- `Standard_NC6s_v3` - V100 GPU (~$3.06/hour)
- `Standard_NC24ads_A100_v4` - A100 GPU (~$3.67/hour)

**Setup:**

1. **Create Workspace** (via Azure Portal or CLI)

2. **Create Compute Instance**
   ```bash
   az ml compute create \
     --name trm-training \
     --type ComputeInstance \
     --size Standard_NC6s_v3
   ```

3. **Submit Training Job**
   ```python
   from azure.ai.ml import MLClient, command
   from azure.identity import DefaultAzureCredential

   ml_client = MLClient.from_config(DefaultAzureCredential())

   job = command(
       code="./TinyRecursiveInference",
       command="python pretrain.py --config-name cfg_cloud_single_gpu arch=trm +run_name=azure_experiment",
       environment="azureml:pytorch-1.13-cuda11.7@latest",
       compute="trm-training",
       display_name="TRM Training"
   )

   ml_client.jobs.create_or_update(job)
   ```

**Cost:** $3-4/hour for GPU instances

---

## Configuration Guide

### Choosing the Right Config

| GPU Memory | Config File | Batch Size | Model Size | Expected Time |
|------------|-------------|------------|------------|---------------|
| 12-16GB | `cfg_cloud_single_gpu` + `arch=trm_tiny` | 32 | 2M params | 8-12 hours |
| 16-24GB | `cfg_cloud_single_gpu` + `arch=trm` | 64 | 7M params | 4-8 hours |
| 40-80GB | `cfg_cloud_high_memory` + `arch=trm` | 256 | 7M params | 1-3 hours |

### Custom Configuration

Override any parameter:

```bash
python pretrain.py \
  --config-name cfg_cloud_single_gpu \
  arch=trm \
  global_batch_size=128 \
  arch.hidden_size=384 \
  arch.L_layers=3 \
  lr=5e-5 \
  +run_name="custom_experiment"
```

### Configuration Parameters

**Key parameters to adjust:**

- `global_batch_size`: Total batch size (reduce if OOM)
- `arch.hidden_size`: Model width (256/384/512)
- `arch.L_layers`: Transformer layers (1/2/3)
- `arch.H_cycles`: High-level recursion cycles (2/3/4)
- `arch.L_cycles`: Low-level recursion cycles (3/4/6)
- `epochs`: Total training epochs
- `eval_interval`: Evaluate every N epochs

---

## GPU Requirements

### Minimum Requirements

| Configuration | GPU Memory | Training Time (10K epochs) |
|---------------|-----------|---------------------------|
| Ultra-Tiny (1M params) | 8GB | 12-16 hours |
| Tiny (2M params) | 12GB | 8-12 hours |
| Standard (7M params) | 16GB | 4-8 hours |
| Standard + Large Batch | 40GB | 1-3 hours |

### Memory Usage Breakdown

For standard 7M parameter TRM:

- **Model Parameters**: ~28 MB (7M × 4 bytes)
- **Optimizer States**: ~56 MB (Adam: 2× parameters)
- **Activations**: Varies with batch size (~100-500 MB per sample)
- **Puzzle Embeddings**: ~2-4 MB
- **Total**: 4-12 GB depending on batch size

### Reducing Memory Usage

If you encounter OOM (Out of Memory) errors:

1. **Reduce Batch Size**
   ```bash
   global_batch_size=32  # or even 16
   ```

2. **Use Smaller Model**
   ```bash
   arch=trm_tiny
   ```

3. **Reduce Hidden Size**
   ```bash
   arch.hidden_size=256
   ```

4. **Disable EMA**
   ```bash
   ema=False
   ```

5. **Reduce Sequence Length** (in dataset preparation)
   ```bash
   --max-seq-len 1024  # default is 2048
   ```

---

## Cost Estimates

### Training Cost by Platform

**For 10,000 epochs (initial convergence):**

| Platform | GPU Type | Time | Cost | Notes |
|----------|----------|------|------|-------|
| Colab Free | T4 | 6-8h | $0 | Limited availability |
| Colab Pro+ | A100 | 1-2h | ~$5 | Monthly subscription |
| AWS g4dn.xlarge | T4 | 6-8h | $3-4 | On-demand pricing |
| AWS p3.2xlarge | V100 | 3-4h | $9-12 | On-demand pricing |
| GCP n1+T4 | T4 | 6-8h | $2-3 | Regular pricing |
| GCP preemptible | T4 | 6-8h | $0.60-1 | May be interrupted |
| Azure NC6s_v3 | V100 | 3-4h | $9-12 | On-demand pricing |
| Prime Intellect | A100 | 1-2h | $2-6 | Spot pricing |

**For full 100,000 epochs (paper results):**

Multiply above by ~10x for full training to match paper results.

### Cost Optimization Tips

1. **Use Spot/Preemptible Instances** (save 60-90%)
   - Implement checkpoint resumption
   - Monitor for termination warnings

2. **Train on Smaller Datasets First**
   - Use `--num-aug 100` instead of 1000
   - Validate architecture before full training

3. **Use Free Tiers**
   - Colab free tier: 6-12 hours/day
   - GCP/AWS: Free tier credits for new users

4. **Batch Multiple Experiments**
   - Use hyperparameter search to run multiple configs
   - Maximize GPU utilization

5. **Off-Peak Training**
   - Some providers offer cheaper rates during off-peak hours

---

## Troubleshooting

### Out of Memory (OOM)

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. Reduce `global_batch_size` to 32, 16, or even 8
2. Use `arch=trm_tiny` for smaller model
3. Disable EMA: `ema=False`
4. Check GPU memory: `nvidia-smi`

### Slow Training

**Symptoms:** Training much slower than expected

**Solutions:**
1. Verify GPU is being used:
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.current_device())
   ```

2. Check GPU utilization: `nvidia-smi` (should be 80-100%)

3. Reduce data augmentation: `--num-aug 50`

4. Use `torch.compile()` (already enabled by default)

### CUDA Version Mismatch

**Error:** `CUDA version mismatch` or `cannot find cudnn`

**Solutions:**
1. Check CUDA version:
   ```bash
   nvcc --version
   nvidia-smi  # Driver version
   ```

2. Install matching PyTorch:
   ```bash
   # For CUDA 11.8
   pip install torch --index-url https://download.pytorch.org/whl/cu118

   # For CUDA 12.1
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

### Connection Timeouts (Colab)

**Problem:** Colab disconnects during training

**Solutions:**
1. Keep browser tab active
2. Use Colab Pro for longer runtime
3. Add auto-reconnect script:
   ```javascript
   function KeepAlive() {
     console.log("Keeping alive");
     document.querySelector("colab-toolbar-button#connect").click();
   }
   setInterval(KeepAlive, 60000);
   ```

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'xxx'`

**Solutions:**
1. Reinstall requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Check Python version (needs 3.10+):
   ```bash
   python --version
   ```

3. Install missing packages individually

### Dataset Preparation Fails

**Error:** Issues with `build_arc_dataset.py`

**Solutions:**
1. Check input files exist:
   ```bash
   ls kaggle/combined/arc-agi*.json
   ```

2. Verify JSON format:
   ```bash
   python -c "import json; json.load(open('kaggle/combined/arc-agi_training_challenges.json'))"
   ```

3. Check disk space:
   ```bash
   df -h
   ```

---

## Next Steps

After successful training:

1. **Evaluate Performance**
   ```bash
   # Checkpoints are saved automatically
   ls checkpoints/*/your_run_name/
   ```

2. **Run Inference**
   ```bash
   python -m tiny_recursive_inference.gradio_app \
     checkpoints/*/your_run_name/
   ```

3. **Publish Model**
   ```bash
   python scripts/publish_checkpoint.py \
     --checkpoint-dir checkpoints/*/your_run_name/ \
     --repo-id your-username/trm-model
   ```

4. **Share Your Results**
   - Upload to HuggingFace Model Hub
   - Create Gradio demo
   - Share on social media with #TinyRecursiveInference

---

## Additional Resources

- **Paper**: https://arxiv.org/abs/2510.04871
- **Repository**: https://github.com/HarleyCoops/TinyRecursiveInference
- **Documentation**: See README.md and CLAUDE.md
- **Multi-Node Training**: See AGENTS.md
- **Edge Deployment**: See SNAPDRAGON_NPU_DEPLOYMENT.md

## Support

- **Issues**: https://github.com/HarleyCoops/TinyRecursiveInference/issues
- **Discussions**: https://github.com/HarleyCoops/TinyRecursiveInference/discussions

---

**Happy Training! 🚀**

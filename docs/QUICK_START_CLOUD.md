# Quick Start: Cloud Training

Get TRM training in under 5 minutes on any cloud GPU platform.

## 🚀 Fastest Path: Google Colab (Free!)

1. **Click this link**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HarleyCoops/TinyRecursiveInference/blob/main/notebooks/train_colab.ipynb)

2. **Enable GPU**: Runtime → Change runtime type → GPU

3. **Click "Run all"**

That's it! Training starts automatically. ✨

## ⚡ Quick Start: Any Cloud Platform

```bash
# One-line setup
git clone https://github.com/HarleyCoops/TinyRecursiveInference.git && \
cd TinyRecursiveInference && \
bash scripts/setup_cloud_training.sh

# Follow the prompts, then start training!
```

## 📋 Platform Quick Links

### Google Colab
```bash
# Already in Colab notebook - just run all cells!
```

### AWS / GCP / Azure
```bash
# After launching GPU instance:
git clone https://github.com/HarleyCoops/TinyRecursiveInference.git
cd TinyRecursiveInference
bash scripts/setup_cloud_training.sh

# Then run recommended command shown by setup script
```

### Prime Intellect
```bash
# After installing Prime Intellect CLI:
primeintellect submit \
  --gpu a100 \
  --gpus 1 \
  --script pretrain.py \
  --args "--config-name cfg_cloud_high_memory arch=trm +run_name=my_experiment"
```

## 💾 Minimal Dataset (Fast Setup)

For quick experimentation, use reduced augmentation:

```bash
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1-quick \
  --subsets training evaluation \
  --test-set-name evaluation \
  --num-aug 50  # Fast! (vs 1000 in paper)
```

## 🎛️ Training Command Templates

### T4 GPU (16GB) - Colab Free, AWS g4dn
```bash
python pretrain.py \
  --config-name cfg_cloud_single_gpu \
  arch=trm_tiny \
  global_batch_size=32 \
  +run_name="t4_experiment"
```

### V100 GPU (16-32GB) - AWS p3, GCP, Azure
```bash
python pretrain.py \
  --config-name cfg_cloud_single_gpu \
  arch=trm \
  global_batch_size=64 \
  +run_name="v100_experiment"
```

### A100 GPU (40-80GB) - Colab Pro+, AWS p4, High-end
```bash
python pretrain.py \
  --config-name cfg_cloud_high_memory \
  arch=trm \
  global_batch_size=256 \
  ema=True \
  +run_name="a100_experiment"
```

## 📊 Monitor Training

Training metrics are printed to console. For visual monitoring:

```bash
# Enable Weights & Biases (free account)
wandb login
# Then training dashboard opens automatically
```

## 💰 Cost Estimates (10K epochs)

| Platform | GPU | Time | Cost |
|----------|-----|------|------|
| Colab Free | T4 | 6h | **$0** ⭐ |
| Colab Pro+ | A100 | 1h | ~$5/mo |
| AWS Spot | T4 | 6h | ~$1 |
| GCP Preemptible | T4 | 6h | ~$1 |
| Prime Intellect | A100 | 1h | ~$2-4 |

## 🐛 Quick Troubleshooting

### Out of Memory?
```bash
# Reduce batch size
global_batch_size=16  # or even 8
```

### Too Slow?
```bash
# Check GPU is active
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Missing Dependencies?
```bash
# Reinstall
pip install -r requirements.txt
```

## 📚 Full Documentation

- **Complete Guide**: [docs/CLOUD_TRAINING.md](CLOUD_TRAINING.md)
- **Multi-GPU Setup**: [AGENTS.md](../AGENTS.md)
- **Architecture Details**: [CLAUDE.md](../CLAUDE.md)
- **Main README**: [README.md](../README.md)

## 🎯 What's Next?

After training completes:

1. **Check Results**: `ls checkpoints/*/your_run_name/`
2. **Run Inference**: `python -m tiny_recursive_inference.gradio_app checkpoints/.../your_run_name/`
3. **Share Model**: Upload to HuggingFace Hub

## 🆘 Need Help?

- **Issues**: [GitHub Issues](https://github.com/HarleyCoops/TinyRecursiveInference/issues)
- **Discussions**: [GitHub Discussions](https://github.com/HarleyCoops/TinyRecursiveInference/discussions)

---

**Ready? Let's train! 🚀**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HarleyCoops/TinyRecursiveInference/blob/main/notebooks/train_colab.ipynb)

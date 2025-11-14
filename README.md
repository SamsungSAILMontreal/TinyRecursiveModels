# Less is More: Recursive Reasoning with Tiny Networks

This is the codebase for the paper: "Less is More: Recursive Reasoning with Tiny Networks". TRM is a recursive reasoning approach that achieves amazing scores of 45% on ARC-AGI-1 and 8% on ARC-AGI-2 using a tiny 7M parameters neural network.

[Paper](https://arxiv.org/abs/2510.04871)

### Motivation

Tiny Recursion Model (TRM) is a recursive reasoning model that achieves amazing scores of 45% on ARC-AGI-1 and 8% on ARC-AGI-2 with a tiny 7M parameters neural network. The idea that one must rely on massive foundational models trained for millions of dollars by some big corporation in order to achieve success on hard tasks is a trap. Currently, there is too much focus on exploiting LLMs rather than devising and expanding new lines of direction. With recursive reasoning, it turns out that “less is more”: you don’t always need to crank up model size in order for a model to reason and solve hard problems. A tiny model pretrained from scratch, recursing on itself and updating its answers over time, can achieve a lot without breaking the bank.

This work came to be after I learned about the recent innovative Hierarchical Reasoning Model (HRM). I was amazed that an approach using small models could do so well on hard tasks like the ARC-AGI competition (reaching 40% accuracy when normally only Large Language Models could compete). But I kept thinking that it is too complicated, relying too much on biological arguments about the human brain, and that this recursive reasoning process could be greatly simplified and improved. Tiny Recursion Model (TRM) simplifies recursive reasoning to its core essence, which ultimately has nothing to do with the human brain, does not require any mathematical (fixed-point) theorem, nor any hierarchy.

### How TRM works

<p align="center">
  <img src="https://AlexiaJM.github.io/assets/images/TRM_fig.png" alt="TRM"  style="width: 30%;">
</p>

Tiny Recursion Model (TRM) recursively improves its predicted answer y with a tiny network. It starts with the embedded input question x and initial embedded answer y and latent z. For up to K improvements steps, it tries to improve its answer y. It does so by i) recursively updating n times its latent z given the question x, current answer y, and current latent z (recursive reasoning), and then ii) updating its answer y given the current answer y and current latent z. This recursive process allows the model to progressively improve its answer (potentially addressing any errors from its previous answer) in an extremely parameter-efficient manner while minimizing overfitting.

### Requirements

- Python 3.10 (or similar)
- Cuda 12.6.0 (or similar)

```bash
pip install --upgrade pip wheel setuptools
pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126 # install torch based on your cuda version
pip install -r requirements.txt # install requirements
pip install --no-cache-dir --no-build-isolation adam-atan2
wandb login YOUR-LOGIN # login if you want the logger to sync results to your Weights & Biases (https://wandb.ai/)
```

### 🚀 Cloud Training (New!)

Train TRM on cloud GPUs in under 5 minutes! No local setup required.

**Quick Start Options:**

1. **Google Colab (Free)** - Best for beginners
   - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HarleyCoops/TinyRecursiveInference/blob/main/notebooks/train_colab.ipynb)
   - Just click and run! All setup is automated.

2. **Any Cloud Platform** - One-line setup
   ```bash
   git clone https://github.com/HarleyCoops/TinyRecursiveInference.git && \
   cd TinyRecursiveInference && \
   bash scripts/setup_cloud_training.sh
   ```

**Supported Platforms:**
- Google Colab (Free tier + Pro)
- AWS EC2 (g4dn, p3, p4d instances)
- Google Cloud Platform (with preemptible GPUs)
- Azure ML
- Prime Intellect (distributed training)
- HuggingFace Spaces (inference)

**Documentation:**
- 📖 [Quick Start Guide](docs/QUICK_START_CLOUD.md) - Get training in 5 minutes
- 📚 [Complete Cloud Training Guide](docs/CLOUD_TRAINING.md) - Platform-specific instructions
- 💰 [Cost Estimates](docs/CLOUD_TRAINING.md#cost-estimates) - $0-$4 for initial training

**Training Time Estimates:**
- T4 GPU (Colab Free): 6-8 hours
- A100 GPU (Colab Pro+/Cloud): 1-2 hours
- Multi-GPU: See [AGENTS.md](AGENTS.md)

### Dataset Preparation

```bash
# ARC-AGI-1
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation

# ARC-AGI-2
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2

## Note: You cannot train on both ARC-AGI-1 and ARC-AGI-2 and evaluate them both because ARC-AGI-2 training data contains some ARC-AGI-1 eval data

# Sudoku-Extreme
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000  # 1000 examples, 1000 augments

# Maze-Hard
python dataset/build_maze_dataset.py # 1000 examples, 8 augments
```

## Experiments

### ARC-AGI-1 (assuming 4 H-100 GPUs):

```bash
run_name="pretrain_att_arc1concept_4"
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm \
data_paths="[data/arc1concept-aug-1000]" \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True

```

*Runtime:* ~3 days

### ARC-AGI-2 (assuming 4 H-100 GPUs):

```bash
run_name="pretrain_att_arc2concept_4"
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm \
data_paths="[data/arc2concept-aug-1000]" \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True

```

*Runtime:* ~3 days

### Sudoku-Extreme (assuming 1 L40S GPU):

```bash
run_name="pretrain_mlp_t_sudoku"
python pretrain.py \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True

run_name="pretrain_att_sudoku"
python pretrain.py \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True
```

*Runtime:* < 36 hours

### Maze-Hard (assuming 4 L40S GPUs):

```bash
run_name="pretrain_att_maze30x30"
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm \
data_paths="[data/maze-30x30-hard-1k]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True
```

*Runtime:* < 24 hours

## Reference

If you find our work useful, please consider citing:

```bibtex
@misc{jolicoeurmartineau2025morerecursivereasoningtiny,
      title={Less is More: Recursive Reasoning with Tiny Networks}, 
      author={Alexia Jolicoeur-Martineau},
      year={2025},
      eprint={2510.04871},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.04871}, 
}
```

and the Hierarchical Reasoning Model (HRM):

```bibtex
@misc{wang2025hierarchicalreasoningmodel,
      title={Hierarchical Reasoning Model}, 
      author={Guan Wang and Jin Li and Yuhao Sun and Xing Chen and Changling Liu and Yue Wu and Meng Lu and Sen Song and Yasin Abbasi Yadkori},
      year={2025},
      eprint={2506.21734},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.21734}, 
}
```

This code is based on the Hierarchical Reasoning Model [code](https://github.com/sapientinc/HRM) and the Hierarchical Reasoning Model Analysis [code](https://github.com/arcprize/hierarchical-reasoning-model-analysis).

## TinyRecursiveInference: Publishing and Inference Pipeline

This repository includes **TinyRecursiveInference**, a complete end-to-end system for publishing datasets and models, plus running inference on trained TRM checkpoints.

### Features

- **Dataset Publishing**: Automated upload of prepared datasets (ARC, Sudoku, Maze) to Hugging Face Hub
- **Training Telemetry**: Enhanced Weights & Biases logging with checkpoint artifact tracking
- **Model Publishing**: One-command upload of trained checkpoints and model cards to Hugging Face Model Hub
- **Interactive Inference**: Gradio application for visual puzzle solving with reasoning state visualization

### Quick Start

**1. Publish a Dataset to Hugging Face Hub**

```bash
python -c "
from tiny_recursive_inference import TinyRecursiveInferencePipeline
from tiny_recursive_inference.config import TinyRecursiveInferenceConfig, DatasetPublishConfig

config = TinyRecursiveInferenceConfig(
    dataset=DatasetPublishConfig(
        local_path='data/arc1concept-aug-1000',
        repo_id='your-username/arc-dataset',
        private=True
    )
)

pipeline = TinyRecursiveInferencePipeline(config)
repo_id = pipeline.publish_dataset()
print(f'Published to: {repo_id}')
"
```

**2. Train with Enhanced W&B Telemetry**

```bash
export USE_TRI_CALLBACKS=true
export WANDB_PROJECT=tiny-recursive-models

torchrun --nproc-per-node 4 pretrain.py \
  arch=trm \
  data_paths="[data/arc1concept-aug-1000]" \
  arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 \
  +run_name="arc1-enhanced" \
  ema=True
```

**3. Publish Trained Model to Hugging Face Hub**

```bash
python scripts/publish_checkpoint.py \
  --checkpoint-dir checkpoints/Arc1concept-aug-1000-ACT-torch/arc1-enhanced \
  --repo-id your-username/arc-trm-model \
  --private
```

**4. Launch Interactive Inference App**

```bash
# From local checkpoint
python -m tiny_recursive_inference.gradio_app \
  checkpoints/Arc1concept-aug-1000-ACT-torch/arc1-enhanced

# From Hugging Face Hub
python -m tiny_recursive_inference.gradio_app \
  your-username/arc-trm-model
```

**5. Run Full Pipeline**

```bash
# Edit config/inference_config.yaml first with your settings
python scripts/run_full_pipeline.py \
  --config config/inference_config.yaml
```

### Configuration

Create `config/inference_config.yaml`:

```yaml
project_root: "."

dataset:
  local_path: "data/arc1concept-aug-1000"
  repo_id: "your-username/arc-dataset"
  private: true

training:
  use_torchrun: true
  nproc_per_node: 4
  checkpoint_dir: "checkpoints/arc1-experiment"
  overrides:
    - "arch=trm"
    - "data_paths=[data/arc1concept-aug-1000]"
    - "arch.L_layers=2"
    - "arch.H_cycles=3"
    - "arch.L_cycles=4"
    - "+run_name=arc1-experiment"
    - "ema=True"

model:
  checkpoint_dir: "checkpoints/arc1-experiment"
  repo_id: "your-username/arc-trm-model"
  private: true
```

### Environment Variables

```bash
# Hugging Face (for publishing)
export HUGGINGFACE_TOKEN="your_hf_token"

# Weights & Biases (for training telemetry)
export WANDB_API_KEY="your_wandb_key"
export WANDB_PROJECT="tiny-recursive-models"

# TinyRecursiveInference
export USE_TRI_CALLBACKS=true  # Enable enhanced W&B logging
```

### Installation

```bash
# Core training dependencies
pip install -r requirements.txt

# TinyRecursiveInference extras
pip install huggingface_hub gradio matplotlib Pillow
```

### Documentation

- See [ClaudePlan.md](ClaudePlan.md) for complete technical specifications and implementation details
- See [AGENTS.md](AGENTS.md) for multi-node distributed training setup

## RecursiveInference

- Train a candidate model and log its evaluation metrics (e.g., W&B charts plus ARC evaluators) during or after the run.
- Load the resulting `step_*` checkpoint with `tiny_recursive_inference.model_loader.load_trm_checkpoint` or by passing `load_checkpoint=` into `pretrain.py`, then score it on the same validation suites used by the current best model.
- Promote the new checkpoint only if it outperforms the incumbent; otherwise retain the previous weights and skip publishing.
- Start the next training pass from the promoted checkpoint (`load_checkpoint=<best_step>`) to continue finetuning and repeat until successive runs no longer improve.
- Once improvements plateau, publish the final checkpoint and update inference endpoints (Gradio app, Hugging Face Space, etc.) so downstream users pick up the upgraded model automatically.

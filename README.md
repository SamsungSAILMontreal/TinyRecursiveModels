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

## SETUP

- **Orientation**  
  - Ground yourself in TRM’s architecture and training loop; `config/arch/trm.yaml:15` defines the 512-dim, 8-head transformer core, while recursion depth lives in `config/arch/trm.yaml:9` and `config/arch/trm.yaml:10`.  
  - Skim `config/cfg_pretrain.yaml:18` for the default global batch, EMA toggle, and optimizer settings.  
  - Review the Distributed Data-Parallel mental model (data sharding, replicated weights, all-reduce) and map each `torchrun` flag to its role.  
  - Confirm that `data/` contains ARC-style samples or plan how Stoney/Nakoda tasks will match the JSON schema used in `dataset/build_*`.

- **Environment Setup**  
  - Build a reproducible Python 3.10 + CUDA 12.6 stack; start from the README install commands but pin stable torch wheels if you are not on nightly.  
  - Create and document a dedicated virtualenv or conda environment.  
  - Validate the GPU stack with quick probes such as `python -c "import torch; print(torch.cuda.get_device_name())"`.  
  - Draft a checklist covering drivers, CUDA/NCCL, and `wandb login` so others can replicate the configuration.

- **Data Curation**  
  - Adapt dataset builders (example: `dataset/build_arc_dataset.py`) to ingest Stoney or Nakoda language pairs while preserving the expected schema.  
  - Produce a toy corpus (≈50 samples) under `data/stoney-pilot` to ensure preprocessing works end to end.  
  - Encourage logging dataset statistics (token counts, label distribution) for later scaling discussions.

- **Single-GPU Training**  
  - Start with `python pretrain.py ...` on one GPU; reduce `global_batch_size` in `config/cfg_pretrain.yaml:18` if memory is tight.  
  - Run a short Sudoku experiment to validate the pipeline before larger datasets.  
  - Track GPU memory with `nvidia-smi dmon` and log metrics to Weights & Biases for baseline comparisons.

- **Monitoring & Checkpointing**  
  - Use `checkpoint_every_eval` (`config/cfg_pretrain.yaml:22`) to ensure recoverability.  
  - Demonstrate run interruption and resume flows so longer jobs can survive preemption.  
  - Identify signals (loss curves, gradient norms) that show readiness to scale beyond a single GPU.

- **Scaling Up (Single Node)**  
  - Transition to `torchrun --nproc-per-node 4 --nnodes 1` when multiple GPUs are available.  
  - Explain how `WORLD_SIZE` impacts per-GPU batch: `per_gpu_batch = global_batch_size / WORLD_SIZE`.  
  - Practice diagnosing NCCL setup issues and interpreting torchrun logs.

- **Multi-Node Expansion**  
  - Teach rendezvous concepts: master node (`--node-rank 0`) hosts the rendezvous endpoint and workers join with matching `--rdzv-id`.  
  - Cover networking prerequisites (open port 29500 or chosen rendezvous port, synced clocks, optional shared storage).  
  - Have participants map `RANK`, `LOCAL_RANK`, and `WORLD_SIZE` for a two-node example before launching real jobs.

- **Resource Planning**  
  - Relate architectural knobs to memory: doubling `hidden_size` (~3× VRAM) and `L_layers` (~2×).  
  - Build a worksheet estimating GPU needs using the 4×H100 baseline and alternatives like L40S.  
  - Promote cost controls: short dry runs, logging to wandb, and tuned checkpoint cadence before lengthy training.

- **Next Steps**  
  - Capture a real Stoney/Nakoda pilot run and analyze outputs for dataset quality.  
  - Script a reusable torchrun launcher that parameterizes `nnodes`, `nproc-per-node`, and `run_name`.  
  - Draft public-facing documentation that blends this setup guide with cultural context for the language revitalization effort.

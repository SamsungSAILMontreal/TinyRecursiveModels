# TinyRecursiveInference Pipeline

TinyRecursiveInference (TRI) extends the Tiny Recursive Models training stack with repeatable dataset publishing, run automation, and deployment-grade inference tooling. The goal is to ensure every multi-GPU training run can graduate into a shareable Hugging Face dataset + model pair with an optional Gradio front-end.

## Pipeline Stages

1. **Dataset Publishing**  
   Use `tiny_recursive_inference/publishers.py` to mirror preprocessed ARC datasets (or new puzzle corpora) to the Hugging Face Hub. The helper:
   - Scans the dataset directory, filters auxiliary files, and commits via the Hub API (`publish_dataset`).
   - Auto-generates a dataset card summarising token/sequence stats when no manual README is provided.
   - Respects `.env` credentials (`HUGGINGFACE_TOKEN`, `HUGGINGFACE_DATASET_REPO`, `HUGGINGFACE_DATASET_PRIVATE`).

2. **Training Orchestration**  
   `tiny_recursive_inference/pipeline.py` wraps Hydra-based `pretrain.py` runs so you can:
   - Launch single-node or multi-node jobs (Torch DDP) with a single configuration object.
   - Inject run-specific overrides (e.g. `arch.L_layers=4`, `checkpoint_path=…`) and environment variables (`WANDB_*`).
   - Chain dataset → training → publishing via `TinyRecursiveInferencePipeline.run_all()`.

3. **Model Publishing**  
   The same pipeline uploads checkpoints and code snapshots to a Hugging Face model repo:
   - Commits every file under the checkpoint directory, plus optional extras (charts, evaluation JSON, etc.).
   - Generates a basic model card referencing the dataset repo when available.
   - Supports private or public repos with automated creation.

4. **Inference & Demo UI**  
   - `tiny_recursive_inference/model_loader.py` rebuilds the TRM architecture from `all_config.yaml` and loads the desired `step_*` checkpoint.
   - `tiny_recursive_inference/inference.py` exposes `predict_arc` to run ACT inference loops on ARC-style samples.
   - `tiny_recursive_inference/gradio_app.py` wires everything into a Gradio Blocks app so users can explore predictions interactively.

## Quickstart

1. **Publish Dataset (optional)**
   ```bash
   python -c "from tiny_recursive_inference import TinyRecursiveInferencePipeline, TinyRecursiveInferenceConfig; \
config = TinyRecursiveInferenceConfig(); \
config.dataset.local_path='data/arc1concept-aug-1000'; \
config.dataset.repo_id='username/arc1concept'; \
TinyRecursiveInferencePipeline(config).publish_dataset()"
   ```

2. **Run Training**
   ```bash
   python -c "from tiny_recursive_inference import TinyRecursiveInferencePipeline, TinyRecursiveInferenceConfig; \
config = TinyRecursiveInferenceConfig(); \
config.training.overrides=['arch=trm', 'data_paths=\"[data/arc1concept-aug-1000]\"']; \
config.training.checkpoint_dir='checkpoints/arc1concept/demo'; \
TinyRecursiveInferencePipeline(config).launch_training()"
   ```

3. **Publish Model**
   ```bash
   python -c "from tiny_recursive_inference import TinyRecursiveInferencePipeline, TinyRecursiveInferenceConfig; \
config = TinyRecursiveInferenceConfig(); \
config.model.checkpoint_dir='checkpoints/arc1concept/demo'; \
config.model.repo_id='username/TRM-demo'; \
TinyRecursiveInferencePipeline(config).publish_model()"
   ```

4. **Launch Gradio Demo**
   ```bash
   python -m tiny_recursive_inference.gradio_app
   ```

Set the checkpoint directory and (optionally) dataset path in the UI to browse inference outputs.

## Environment Variables

Copy `.env.example` to `.env` and fill in:

| Variable | Purpose |
| --- | --- |
| `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_RUN_NAME` | Connect training runs to Weights & Biases. |
| `HUGGINGFACE_TOKEN` | Grant write access for dataset/model upload. |
| `HUGGINGFACE_DATASET_REPO`, `HUGGINGFACE_MODEL_REPO` | Default repo slugs consumed by the pipeline. |
| `HUGGINGFACE_DATASET_PRIVATE`, `HUGGINGFACE_MODEL_PRIVATE` | Toggle visibility for dataset/model repos. |

## Suggested Workflow

1. Generate or augment puzzles with existing scripts in `dataset/`.
2. `publish_dataset` once the preprocessed tensors pass sanity checks.
3. Launch multi-GPU training via `TinyRecursiveInferencePipeline.launch_training`; monitor metrics with W&B.
4. Inspect checkpoints (accuracy, pass@K) and optionally publish the best `step_*`.
5. Launch the Gradio UI for qualitative inspection, then share the Hugging Face links.

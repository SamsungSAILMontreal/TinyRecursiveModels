from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import gradio as gr
import numpy as np
import torch

from .inference import get_dataset_example, predict_arc
from .model_loader import load_trm_checkpoint


def _grid_to_text(grid: np.ndarray) -> str:
    return "\n".join(" ".join(f"{int(cell)}" for cell in row) for row in grid)


def _effective_dataset_path(config_data_paths, override: Optional[str]) -> str:
    if override:
        return override
    if not config_data_paths:
        raise ValueError("No dataset paths found in training config.")
    return config_data_paths[0]


@lru_cache(maxsize=2)
def _load_resources(checkpoint_dir: str, dataset_path: Optional[str]) -> Tuple[torch.nn.Module, Dict, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_override = [dataset_path] if dataset_path else None
    model, metadata, config, checkpoint_file = load_trm_checkpoint(
        checkpoint_dir,
        device=device,
        data_paths_override=data_override,
    )
    model.to(device)
    return model, {"metadata": metadata, "config": config}, device


def _dataset_length(dataset_path: str, split: str) -> int:
    inputs_path = Path(dataset_path) / "test" / f"{split}__inputs.npy"
    if not inputs_path.exists():
        raise FileNotFoundError(f"Expected inputs file not found for split '{split}': {inputs_path}")
    arr = np.load(inputs_path, mmap_mode="r")
    length = int(arr.shape[0])
    del arr
    return length


def _predict(
    checkpoint_dir: str,
    dataset_path: Optional[str],
    split: str,
    example_index: int,
) -> Tuple[str, str, str]:
    model, context, device = _load_resources(checkpoint_dir, dataset_path)
    effective_path = _effective_dataset_path(context["config"].data_paths, dataset_path)
    example = get_dataset_example(effective_path, example_index, split=split)
    input_grid, pred_grid = predict_arc(model, example, device=torch.device(device))

    info = (
        f"Checkpoint: {checkpoint_dir}\n"
        f"Dataset: {effective_path}\n"
        f"Split: {split}\n"
        f"Example index: {example_index}"
    )
    return _grid_to_text(input_grid), _grid_to_text(pred_grid), info


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="TinyRecursiveInference Demo") as demo:
        gr.Markdown(
            """
            # TinyRecursiveInference Demo

            Load a trained Tiny Recursive Model checkpoint and explore predictions on ARC-style puzzles.
            """
        )

        with gr.Row():
            checkpoint_dir_input = gr.Textbox(
                label="Checkpoint Directory",
                placeholder="checkpoints/Arc-Run-01",
            )
            dataset_path_input = gr.Textbox(
                label="Dataset Path Override (optional)",
                placeholder="data/arc1concept-aug-1000",
            )

        with gr.Row():
            split_input = gr.Radio(choices=["test", "evaluation"], value="test", label="Dataset split")
            example_index_input = gr.Slider(0, 100, value=0, step=1, label="Example Index")

        run_button = gr.Button("Run Inference")

        with gr.Row():
            input_output = gr.Textbox(label="Input Grid", lines=15)
            prediction_output = gr.Textbox(label="Predicted Grid", lines=15)

        info_box = gr.Textbox(label="Run Info", lines=5)

        def _update_index_bounds(checkpoint_dir: str, dataset_path: Optional[str], split_value: str):
            if not checkpoint_dir:
                return gr.update(), gr.update(), gr.update()
            try:
                _, context, _ = _load_resources(checkpoint_dir, dataset_path)
                effective_path = _effective_dataset_path(context["config"].data_paths, dataset_path)
                length = _dataset_length(effective_path, split_value)
                current_split = (
                    split_value
                    if split_value in context["metadata"].sets
                    else context["metadata"].sets[0]
                )
            except Exception as exc:  # noqa: BLE001
                return gr.update(), gr.update(), gr.update(value=str(exc))

            return (
                gr.update(choices=context["metadata"].sets, value=current_split),
                gr.update(maximum=max(0, length - 1), value=0),
                gr.update(value=f"Loaded dataset with {length} examples for split '{current_split}'."),
            )

        split_input.change(
            _update_index_bounds,
            inputs=[checkpoint_dir_input, dataset_path_input, split_input],
            outputs=[split_input, example_index_input, info_box],
        )
        checkpoint_dir_input.change(
            _update_index_bounds,
            inputs=[checkpoint_dir_input, dataset_path_input, split_input],
            outputs=[split_input, example_index_input, info_box],
        )
        dataset_path_input.change(
            _update_index_bounds,
            inputs=[checkpoint_dir_input, dataset_path_input, split_input],
            outputs=[split_input, example_index_input, info_box],
        )

        run_button.click(
            _predict,
            inputs=[checkpoint_dir_input, dataset_path_input, split_input, example_index_input],
            outputs=[input_output, prediction_output, info_box],
        )

    return demo


if __name__ == "__main__":
    build_interface().launch()

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Optional, Tuple

import numpy as np
import torch

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig


@dataclass
class InferenceExample:
    set_name: str
    inputs: torch.Tensor
    labels: torch.Tensor
    puzzle_identifiers: torch.Tensor


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _arc_crop(tokens: torch.Tensor) -> np.ndarray:
    grid = tokens.cpu().numpy().reshape(30, 30)
    valid_mask = (grid >= 2) & (grid <= 11)
    if not valid_mask.any():
        return np.zeros((1, 1), dtype=np.uint8)

    rows = np.where(valid_mask.any(axis=1))[0]
    cols = np.where(valid_mask.any(axis=0))[0]
    cropped = grid[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    return (cropped - 2).astype(np.uint8)


def iter_dataset_examples(dataset_path: str, split: str = "test") -> Iterator[InferenceExample]:
    """Yield individual examples from the specified dataset split."""
    cfg = PuzzleDatasetConfig(
        seed=0,
        dataset_paths=[dataset_path],
        global_batch_size=1,
        test_set_mode=True,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    )
    dataset = PuzzleDataset(cfg, split=split)
    for set_name, batch, _ in dataset:
        yield InferenceExample(
            set_name=set_name,
            inputs=batch["inputs"][0],
            labels=batch["labels"][0],
            puzzle_identifiers=batch["puzzle_identifiers"][0],
        )


def get_dataset_example(dataset_path: str, index: int, split: str = "test") -> InferenceExample:
    for i, example in enumerate(iter_dataset_examples(dataset_path, split=split)):
        if i == index:
            return example
    raise IndexError(f"Example index {index} out of range for split '{split}'")


def run_model_step(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    return_keys: Optional[Iterable[str]] = None,
) -> Dict[str, torch.Tensor]:
    keys = set(return_keys or [])
    with torch.no_grad():
        carry = model.initial_carry(batch)
        while True:
            carry, _loss, _metrics, outputs, all_finish = model(
                carry=carry,
                batch=batch,
                return_keys=keys,
            )
            if all_finish:
                break
    return outputs


def predict_arc(
    model: torch.nn.Module,
    example: InferenceExample,
    *,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on a single ARC-style example."""
    torch_device = device or next(model.parameters()).device
    batch = {
        "inputs": example.inputs.unsqueeze(0),
        "labels": example.labels.unsqueeze(0),
        "puzzle_identifiers": example.puzzle_identifiers.unsqueeze(0),
    }
    batch = _to_device(batch, torch_device)

    outputs = run_model_step(model, batch, return_keys={"preds"})
    preds = outputs["preds"][0]

    input_grid = _arc_crop(example.inputs)
    pred_grid = _arc_crop(preds)
    return input_grid, pred_grid

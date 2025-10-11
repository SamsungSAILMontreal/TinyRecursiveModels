from __future__ import annotations

import random
import time
from typing import Iterable, List, Sequence

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clone_parameters(params: Iterable[torch.nn.Parameter]) -> List[torch.Tensor]:
    return [p.detach().clone() for p in params]


class Timer:
    """Simple wall-clock timer context manager."""

    def __enter__(self) -> "Timer":
        self.reset()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def reset(self) -> None:
        self._start = time.time()
        self.elapsed: float = 0.0

    def stop(self) -> None:
        if self._start is not None:
            self.elapsed = time.time() - self._start
            self._start = None

    def __repr__(self) -> str:
        return f"Timer(elapsed={self.elapsed:.3f}s)"


def to_device(batch: Sequence[torch.Tensor], device: torch.device) -> List[torch.Tensor]:
    return [tensor.to(device) for tensor in batch]


def acc_on_pair(
    model: torch.nn.Module,
    device: torch.device,
    input_grid,
    output_grid,
) -> float:
    target = np.array(output_grid, dtype=np.int64)

    if hasattr(model, "ttt_predict_grid"):
        pred = model.ttt_predict_grid(input_grid, device=device)
        pred_arr = np.array(pred, dtype=np.int64)
    else:
        from .arc_io import grid_to_tensor, tensor_to_grid

        x = grid_to_tensor(input_grid).unsqueeze(0).to(device)
        logits = model(x)[0]
        pred_arr = tensor_to_grid(logits)

    if pred_arr.shape != target.shape:
        return 0.0
    return float((pred_arr == target).mean())

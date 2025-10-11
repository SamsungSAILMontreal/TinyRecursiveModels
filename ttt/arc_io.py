from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


COLOR_DIM = 10  # ARC uses 10 colors encoded as integers in [0, 9]
ARC_MAX_SIZE = 30
ARC_PAD_TOKEN = 0
ARC_EOS_TOKEN = 1
ARC_COLOR_OFFSET = 2


def load_arc_task(path: str) -> Dict:
    """Load a single ARC puzzle from JSON."""
    with open(path, "r") as handle:
        return json.load(handle)


def iter_arc_tasks(arc_dir: str) -> Iterable[Tuple[str, Dict]]:
    """Yield (filename, task_json) pairs for every ARC JSON in a directory."""
    for filename in sorted(os.listdir(arc_dir)):
        if not filename.endswith(".json"):
            continue
        full_path = os.path.join(arc_dir, filename)
        yield filename, load_arc_task(full_path)


def grid_to_tensor(grid: List[List[int]]) -> torch.Tensor:
    """Convert an integer grid (H, W) into a one-hot tensor (C, H, W)."""
    arr = torch.tensor(grid, dtype=torch.long)
    one_hot = F.one_hot(arr, num_classes=COLOR_DIM).permute(2, 0, 1).float()
    return one_hot


def tensor_to_grid(logits: torch.Tensor) -> np.ndarray:
    """Convert logits (C, H, W) to an integer grid via argmax."""
    if logits.dim() != 3:
        raise ValueError(f"Expected (C,H,W) logits, got shape {tuple(logits.shape)}")
    return logits.argmax(dim=0).cpu().numpy().astype(np.int32)


def grid_to_arc_tokens(grid: List[List[int]]) -> torch.Tensor:
    """Encode an ARC grid into the padded 30x30 token representation used by TRM."""
    arr = np.array(grid, dtype=np.int64)
    if arr.ndim != 2:
        raise ValueError("ARC grid must be 2D")
    h, w = arr.shape
    if h > ARC_MAX_SIZE or w > ARC_MAX_SIZE:
        raise ValueError(f"ARC grid too large ({h}x{w}), max supported {ARC_MAX_SIZE}x{ARC_MAX_SIZE}")

    tokens = np.zeros((ARC_MAX_SIZE, ARC_MAX_SIZE), dtype=np.int64)
    tokens[:h, :w] = arr + ARC_COLOR_OFFSET
    if h < ARC_MAX_SIZE:
        tokens[h, :w] = ARC_EOS_TOKEN
    if w < ARC_MAX_SIZE:
        tokens[:h, w] = ARC_EOS_TOKEN
    return torch.from_numpy(tokens.reshape(-1))


def arc_tokens_to_grid(tokens: torch.Tensor | np.ndarray) -> np.ndarray:
    """Decode a 30x30 token sequence back into an ARC grid."""
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.detach().cpu().numpy()
    tokens = np.asarray(tokens)
    if tokens.size != ARC_MAX_SIZE * ARC_MAX_SIZE:
        raise ValueError("ARC token sequence must have length 900 (30x30)")
    tokens = tokens.reshape(ARC_MAX_SIZE, ARC_MAX_SIZE)
    mask = tokens >= ARC_COLOR_OFFSET
    if not mask.any():
        return np.zeros((0, 0), dtype=np.int64)

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    h = rows[-1] + 1 if rows.size else ARC_MAX_SIZE
    w = cols[-1] + 1 if cols.size else ARC_MAX_SIZE
    grid = tokens[:h, :w] - ARC_COLOR_OFFSET
    return grid.astype(np.int64)

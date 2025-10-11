from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn.functional as F


def grid_ce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute pixel-wise cross entropy between logits (C,H,W) and targets (H,W)."""
    if logits.ndim != 3:
        raise ValueError(f"Expected logits with 3 dims (C,H,W), got {tuple(logits.shape)}")
    if target.ndim != 2:
        raise ValueError(f"Expected target grid with 2 dims (H,W), got {tuple(target.shape)}")
    C, H, W = logits.shape
    if target.shape != (H, W):
        raise ValueError("Target grid must match height/width of logits")
    return F.cross_entropy(logits.view(C, -1).t(), target.view(-1))


def proximal_penalty(
    params: Iterable[torch.Tensor],
    initial_params: Iterable[torch.Tensor],
) -> torch.Tensor:
    """Average squared L2 distance between current and initial adapter parameters."""
    params_list: List[torch.Tensor] = [p for p in params]
    init_list: List[torch.Tensor] = [p0 for p0 in initial_params]

    total = 0.0
    numel = 0
    for p, p0 in zip(params_list, init_list):
        total = total + (p - p0.to(p.device)).pow(2).sum()
        numel += p.numel()
    if numel == 0:
        device = params_list[0].device if params_list else torch.device("cpu")
        return torch.tensor(0.0, device=device)
    return total / numel

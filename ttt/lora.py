from __future__ import annotations
import math
from typing import Iterable, List

import torch
from torch import nn


def _is_embedding_module(module: nn.Module) -> bool:
    name = module.__class__.__name__.lower()
    return isinstance(module, nn.Embedding) or "embedding" in name


def _linear_like_shape(module: nn.Module) -> tuple[int, int]:
    weight = getattr(module, "weight", None)
    if weight is None or not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise TypeError(f"Module {module.__class__.__name__} is not compatible with LoRA (missing 2D weight).")
    out_features, in_features = weight.shape
    return in_features, out_features


class LoRALinear(nn.Module):
    """Low-rank adaptation wrapper for linear-like modules with 2D weights."""

    def __init__(self, base_module: nn.Module, r: int, alpha: float, dropout_p: float = 0.0):
        super().__init__()
        if r < 0:
            raise ValueError("LoRA rank must be non-negative")

        in_features, out_features = _linear_like_shape(base_module)

        self.base = base_module
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 0.0
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
        self.in_features = in_features
        self.out_features = out_features

        if r == 0:
            self.register_parameter("A", None)
            self.register_parameter("B", None)
        else:
            self.A = nn.Parameter(torch.zeros((r, in_features)))
            self.B = nn.Parameter(torch.zeros((out_features, r)))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.r == 0:
            return y

        x_dropped = self.dropout(x)
        delta = (x_dropped @ self.A.t()) @ self.B.t()
        return y + self.scaling * delta

    def lora_parameters(self) -> Iterable[nn.Parameter]:
        if self.r == 0:
            return []
        return [self.A, self.B]


def patch_lora(
    model: nn.Module,
    name_filters: List[str],
    r: int,
    alpha: float,
    dropout: float = 0.0,
) -> List[nn.Parameter]:
    """Wrap selected linear-like modules with LoRA adapters."""

    if r < 0:
        raise ValueError("LoRA rank must be non-negative")
    if alpha <= 0 and r > 0:
        raise ValueError("LoRA alpha must be positive when rank > 0")

    adapter_params: List[nn.Parameter] = []

    def should_patch(qualname: str) -> bool:
        if not name_filters:
            return True
        return any(token in qualname for token in name_filters)

    def recurse(module: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(module.named_children()):
            qualname = f"{prefix}.{child_name}" if prefix else child_name

            if isinstance(child, LoRALinear):
                adapter_params.extend(list(child.lora_parameters()))
                continue

            try:
                _linear_like_shape(child)
                linear_like = True
            except TypeError:
                linear_like = False

            if linear_like and not _is_embedding_module(child) and should_patch(qualname):
                lora_module = LoRALinear(child, r=r, alpha=alpha, dropout_p=dropout)
                setattr(module, child_name, lora_module)
                adapter_params.extend(list(lora_module.lora_parameters()))
                continue

            recurse(child, qualname)

    recurse(model)
    return adapter_params

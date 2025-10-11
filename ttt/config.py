from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TTTConfig:
    """Configuration for per-puzzle test-time training."""

    steps: int = 20
    lr: float = 5e-3
    wd: float = 0.0
    prox_lambda: float = 1e-3
    seed: int = 0

    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_targets: Optional[List[str]] = field(default=None)

    device: str = "cuda"
    max_horizon: int = 1
    gradient_clip_norm: float = 1.0

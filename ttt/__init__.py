"""Test-time training (TTT) scaffold package.

This package provides lightweight utilities to adapt tiny recursive
models (e.g., TRM/HRM) on a per-puzzle basis using LoRA adapters.
"""

from .config import TTTConfig
from .runner import adapt_on_task, adapt_directory
from .trm_adapter import TRMArcAdapter

__all__ = ["TTTConfig", "adapt_on_task", "adapt_directory", "TRMArcAdapter"]

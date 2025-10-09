from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
import yaml

from dataset.common import PuzzleDatasetMetadata
from pretrain import PretrainConfig
from utils.functions import load_model_class


def _select_checkpoint_file(checkpoint_dir: Path, step: Optional[int]) -> Path:
    if step is not None:
        candidate = checkpoint_dir / f"step_{step}"
        if not candidate.exists():
            raise FileNotFoundError(f"Checkpoint step {step} not found in {checkpoint_dir}")
        return candidate

    checkpoints = sorted(
        checkpoint_dir.glob("step_*"),
        key=lambda path: int(path.name.split("_")[-1]),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return checkpoints[-1]


def _load_metadata_from_paths(data_paths: Iterable[str]) -> PuzzleDatasetMetadata:
    metadata = None
    for path in data_paths:
        dataset_path = Path(path)
        if not dataset_path.is_absolute():
            dataset_path = Path(os.getcwd()) / dataset_path
        meta_file = dataset_path / "train" / "dataset.json"
        if meta_file.exists():
            with meta_file.open("r", encoding="utf-8") as f:
                metadata = PuzzleDatasetMetadata(**json.load(f))
            break
    if metadata is None:
        raise FileNotFoundError(
            f"Could not locate dataset metadata. Checked: {[str(Path(p) / 'train/dataset.json') for p in data_paths]}"
        )
    return metadata


def _instantiate_model(
    config: PretrainConfig,
    metadata: PuzzleDatasetMetadata,
    device: torch.device,
) -> torch.nn.Module:
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore[attr-defined]
        batch_size=config.global_batch_size,
        vocab_size=metadata.vocab_size,
        seq_len=metadata.seq_len,
        num_puzzle_identifiers=metadata.num_puzzle_identifiers,
        causal=False,
    )

    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device(device.type):
        base_model = model_cls(model_cfg)
        model = loss_head_cls(base_model, **config.arch.loss.__pydantic_extra__)  # type: ignore[attr-defined]

    model.to(device)
    model.eval()
    return model


def load_trm_checkpoint(
    checkpoint_dir: str,
    *,
    step: Optional[int] = None,
    device: str = "cpu",
    config_path: Optional[str] = None,
    data_paths_override: Optional[List[str]] = None,
) -> Tuple[torch.nn.Module, PuzzleDatasetMetadata, PretrainConfig, Path]:
    """Load a Tiny Recursive Model checkpoint along with metadata and config.

    Returns the initialized model, dataset metadata, original training config,
    and the checkpoint file path that was used.
    """

    checkpoint_root = Path(checkpoint_dir).expanduser().resolve()
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_root}")

    cfg_path = Path(config_path) if config_path else checkpoint_root / "all_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    config = PretrainConfig(**config_dict)
    data_paths = data_paths_override or config.data_paths
    metadata = _load_metadata_from_paths(data_paths)

    torch_device = torch.device(device)
    model = _instantiate_model(config, metadata, torch_device)

    checkpoint_file = _select_checkpoint_file(checkpoint_root, step)
    state_dict = torch.load(checkpoint_file, map_location=torch_device)

    puzzle_emb_name = "_orig_mod.model.inner.puzzle_emb.weights"
    if hasattr(model, "model") and hasattr(model.model, "puzzle_emb"):
        expected_shape: torch.Size = model.model.puzzle_emb.weights.shape  # type: ignore[attr-defined]
        if puzzle_emb_name in state_dict:
            puzzle_emb = state_dict[puzzle_emb_name]
            if puzzle_emb.shape != expected_shape:
                state_dict[puzzle_emb_name] = (
                    torch.mean(puzzle_emb, dim=0, keepdim=True)
                    .expand(expected_shape)
                    .contiguous()
                )

    model.load_state_dict(state_dict, strict=False)
    model.to(torch_device)
    model.eval()

    return model, metadata, config, checkpoint_file

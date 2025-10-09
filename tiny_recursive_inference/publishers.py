from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from huggingface_hub import HfApi

from .config import DatasetPublishConfig, ModelPublishConfig


def _resolve_token(config_token: Optional[str]) -> Optional[str]:
    token = config_token or os.getenv("HUGGINGFACE_TOKEN")
    return token or None


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")


def _iter_files(root: Path, include: Optional[Iterable[str]], ignore: Optional[Iterable[str]]):
    for file_path in root.rglob("*"):
        if file_path.is_dir():
            continue
        relative = file_path.relative_to(root)
        if ignore and any(relative.match(pattern) for pattern in ignore):
            continue
        if include and not any(relative.match(pattern) for pattern in include):
            continue
        yield file_path, relative.as_posix()


def _default_dataset_card(dataset_dir: Path) -> str:
    metadata_paths = list(dataset_dir.glob("*/dataset.json"))
    if not metadata_paths:
        return (
            "# TinyRecursiveInference Dataset\n\n"
            "This dataset was published automatically by the TinyRecursiveInference "
            "pipeline.\n"
        )

    sections = ["# TinyRecursiveInference Dataset"]
    total_examples = 0
    total_puzzles = 0
    seq_len = None
    vocab_size = None
    sets = []

    for meta_path in metadata_paths:
        with meta_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        total_examples += metadata.get("total_groups", 0)
        total_puzzles += metadata.get("total_puzzles", 0)
        if seq_len is None:
            seq_len = metadata.get("seq_len")
        if vocab_size is None:
            vocab_size = metadata.get("vocab_size")
        sets.extend(str(s) for s in metadata.get("sets", []))

    sets = sorted(set(sets))
    sections.append("")
    sections.append(f"- Published at: {_timestamp()}")
    if seq_len:
        sections.append(f"- Sequence length: `{seq_len}` tokens")
    if vocab_size:
        sections.append(f"- Vocabulary size: `{vocab_size}`")
    if total_puzzles:
        sections.append(f"- Total puzzles: `{total_puzzles}`")
    if total_examples:
        sections.append(f"- Total groups: `{total_examples}`")
    if sets:
        sections.append(f"- Splits: `{', '.join(sets)}`")

    sections.append("")
    sections.append(
        "## Overview\n\n"
        "The dataset contains ARC-style puzzles formatted for Tiny Recursive Model "
        "training. Each split directory includes a `dataset.json` metadata file as "
        "well as memory-mapped `inputs`, `labels`, and `puzzle_identifiers` arrays."
    )
    return "\n".join(sections)


def publish_dataset(config: DatasetPublishConfig) -> Optional[str]:
    """Upload a processed dataset directory to the Hugging Face Hub."""

    if not config.local_path or not config.repo_id:
        return None

    dataset_dir = Path(config.local_path).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    api = HfApi(token=_resolve_token(config.token))
    api.create_repo(
        repo_id=config.repo_id,
        repo_type="dataset",
        private=config.private,
        exist_ok=config.allow_create,
    )

    api.upload_folder(
        folder_path=str(dataset_dir),
        repo_id=config.repo_id,
        repo_type="dataset",
        commit_message=config.commit_message,
        ignore_patterns=config.files_ignore,
        allow_patterns=config.files_include,
    )

    if config.add_readme:
        if config.readme_path:
            readme_path = Path(config.readme_path)
            if not readme_path.is_file():
                raise FileNotFoundError(f"Dataset README template not found: {readme_path}")
            with readme_path.open("r", encoding="utf-8") as f:
                readme_content = f.read()
        else:
            readme_content = _default_dataset_card(dataset_dir)

        api.upload_file(
            path_or_fileobj=readme_content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=config.repo_id,
            repo_type="dataset",
            commit_message=f"{config.commit_message} (dataset card)",
        )

    return config.repo_id


def _default_model_card(checkpoint_dir: Path, dataset_repo: Optional[str]) -> str:
    sections = ["# TinyRecursiveInference Model"]
    sections.append("")
    sections.append(f"- Published at: {_timestamp()}")
    if dataset_repo:
        sections.append(f"- Trained on dataset: `{dataset_repo}`")
    config_file = checkpoint_dir / "all_config.yaml"
    if config_file.exists():
        sections.append(f"- Config file: `{config_file.name}`")

    sections.append("")
    sections.append(
        "## Usage\n\n"
        "```\n"
        "from tiny_recursive_inference.model_loader import load_trm_checkpoint\n"
        "\n"
        "model, metadata = load_trm_checkpoint(\n"
        f"    checkpoint_dir=\"{checkpoint_dir}\",\n"
        "    device=\"cuda\"\n"
        ")\n"
        "```\n"
    )
    return "\n".join(sections)


def publish_model(config: ModelPublishConfig, dataset_repo: Optional[str] = None) -> Optional[str]:
    """Upload trained checkpoints, configs, and code snapshots to the Hub."""

    if not config.checkpoint_dir or not config.repo_id:
        return None

    checkpoint_dir = Path(config.checkpoint_dir).expanduser().resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    api = HfApi(token=_resolve_token(config.token))
    api.create_repo(
        repo_id=config.repo_id,
        repo_type="model",
        private=config.private,
        exist_ok=config.allow_create,
    )

    api.upload_folder(
        folder_path=str(checkpoint_dir),
        repo_id=config.repo_id,
        repo_type="model",
        commit_message=config.commit_message,
        ignore_patterns=["*.tmp", "*.lock"],
    )

    for target_path, local_path in config.extra_files.items():
        local_file = Path(local_path)
        if not local_file.exists():
            raise FileNotFoundError(f"Extra file not found: {local_file}")
        api.upload_file(
            path_or_fileobj=str(local_file),
            path_in_repo=target_path,
            repo_id=config.repo_id,
            repo_type="model",
            commit_message=f"{config.commit_message} (extra file)",
        )

    if config.model_card_path:
        model_card_path = Path(config.model_card_path)
        if not model_card_path.is_file():
            raise FileNotFoundError(f"Model card template not found: {model_card_path}")
        with model_card_path.open("r", encoding="utf-8") as f:
            model_card_content = f.read()
    else:
        model_card_content = _default_model_card(checkpoint_dir, dataset_repo)

    api.upload_file(
        path_or_fileobj=model_card_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=config.repo_id,
        repo_type="model",
        commit_message=f"{config.commit_message} (model card)",
    )

    return config.repo_id

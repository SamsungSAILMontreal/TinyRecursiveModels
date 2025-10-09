from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DatasetPublishConfig:
    """Configuration for uploading prepared datasets to the Hugging Face Hub."""

    local_path: Optional[str] = None
    repo_id: Optional[str] = None
    private: bool = True
    token: Optional[str] = None
    add_readme: bool = True
    readme_path: Optional[str] = None
    commit_message: str = "Automated dataset update from TinyRecursiveInference pipeline"
    allow_create: bool = True
    files_include: Optional[List[str]] = None
    files_ignore: Optional[List[str]] = field(default_factory=lambda: [".DS_Store", "Thumbs.db"])


@dataclass
class ModelPublishConfig:
    """Configuration for checkpoint + model card publishing."""

    checkpoint_dir: Optional[str] = None
    repo_id: Optional[str] = None
    private: bool = True
    token: Optional[str] = None
    commit_message: str = "Upload TinyRecursiveInference checkpoint"
    allow_create: bool = True
    config_filename: str = "all_config.yaml"
    include_code_snapshot: bool = True
    model_card_path: Optional[str] = None
    extra_files: Dict[str, str] = field(default_factory=dict)


@dataclass
class TrainingLaunchConfig:
    """Configuration for launching `pretrain.py` with reproducible overrides."""

    overrides: List[str] = field(default_factory=list)
    use_torchrun: bool = False
    nproc_per_node: int = 1
    nnodes: int = 1
    node_rank: int = 0
    rdzv_backend: str = "c10d"
    rdzv_endpoint: str = "localhost:29500"
    rdzv_id: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    checkpoint_dir: Optional[str] = None
    dry_run: bool = False
    python_executable: str = "python"


@dataclass
class TinyRecursiveInferenceConfig:
    """Aggregate configuration for the end-to-end pipeline."""

    dataset: DatasetPublishConfig = field(default_factory=DatasetPublishConfig)
    training: TrainingLaunchConfig = field(default_factory=TrainingLaunchConfig)
    model: ModelPublishConfig = field(default_factory=ModelPublishConfig)
    project_root: str = "."

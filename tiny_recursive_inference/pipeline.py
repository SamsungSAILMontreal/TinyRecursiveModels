from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

from .config import (
    DatasetPublishConfig,
    ModelPublishConfig,
    TinyRecursiveInferenceConfig,
    TrainingLaunchConfig,
)
from .publishers import publish_dataset as _publish_dataset
from .publishers import publish_model as _publish_model


def _resolve_path(project_root: Path, path_value: Optional[str]) -> Optional[str]:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = project_root / path
    return str(path)


class TinyRecursiveInferencePipeline:
    """Coordinate dataset publishing, training, and inference artifact uploads."""

    def __init__(self, config: TinyRecursiveInferenceConfig):
        self.config = config
        self.project_root = Path(config.project_root).expanduser().resolve()
        self._dataset_repo: Optional[str] = None

    # ------------------------------------------------------------------
    # Dataset publishing
    # ------------------------------------------------------------------
    def publish_dataset(self, config: Optional[DatasetPublishConfig] = None) -> Optional[str]:
        dataset_cfg = config or self.config.dataset
        if dataset_cfg.local_path:
            dataset_cfg = DatasetPublishConfig(
                **{
                    **dataset_cfg.__dict__,
                    "local_path": _resolve_path(self.project_root, dataset_cfg.local_path),
                    "readme_path": _resolve_path(self.project_root, dataset_cfg.readme_path),
                }
            )

        repo_id = _publish_dataset(dataset_cfg)
        if repo_id:
            self._dataset_repo = repo_id
        return repo_id

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _build_training_command(self, config: TrainingLaunchConfig) -> Sequence[str]:
        overrides = list(config.overrides)

        # Ensure checkpoint_path override is included if provided explicitly.
        if config.checkpoint_dir and not any(opt.startswith("checkpoint_path=") for opt in overrides):
            overrides.append(f"checkpoint_path={config.checkpoint_dir}")

        if config.use_torchrun:
            command: Sequence[str] = [
                "torchrun",
                f"--nproc-per-node={config.nproc_per_node}",
                f"--nnodes={config.nnodes}",
                f"--node-rank={config.node_rank}",
                f"--rdzv_backend={config.rdzv_backend}",
                f"--rdzv_endpoint={config.rdzv_endpoint}",
            ]
            if config.rdzv_id:
                command = [*command, f"--rdzv-id={config.rdzv_id}"]
            command = [*command, "pretrain.py", *overrides]
        else:
            command = [config.python_executable, "pretrain.py", *overrides]

        return command

    def launch_training(self, config: Optional[TrainingLaunchConfig] = None) -> Optional[int]:
        launch_cfg = config or self.config.training
        command = self._build_training_command(launch_cfg)

        if launch_cfg.dry_run:
            printable = " ".join(shlex.quote(part) for part in command)
            print(f"[TinyRecursiveInference] Dry run training command:\n  {printable}")
            return None

        env: Dict[str, str] = os.environ.copy()
        env.update(launch_cfg.env)

        process = subprocess.run(
            command,
            cwd=str(self.project_root),
            env=env,
            check=True,
        )
        return process.returncode

    # ------------------------------------------------------------------
    # Model publishing
    # ------------------------------------------------------------------
    def publish_model(self, config: Optional[ModelPublishConfig] = None) -> Optional[str]:
        model_cfg = config or self.config.model
        if model_cfg.checkpoint_dir:
            resolved_extra_files = {
                path_in_repo: resolved_path
                for path_in_repo, local_path in model_cfg.extra_files.items()
                if (resolved_path := _resolve_path(self.project_root, local_path)) is not None
            }
            model_cfg = ModelPublishConfig(
                **{
                    **model_cfg.__dict__,
                    "checkpoint_dir": _resolve_path(self.project_root, model_cfg.checkpoint_dir),
                    "model_card_path": _resolve_path(self.project_root, model_cfg.model_card_path),
                    "extra_files": resolved_extra_files,
                }
            )
        return _publish_model(model_cfg, dataset_repo=self._dataset_repo)

    # ------------------------------------------------------------------
    # Convenience execution
    # ------------------------------------------------------------------
    def run_all(self) -> None:
        """Execute dataset publishing, training, and model publishing sequentially."""
        if self.config.dataset.local_path and self.config.dataset.repo_id:
            self.publish_dataset(self.config.dataset)

        self.launch_training(self.config.training)

        if self.config.model.checkpoint_dir and self.config.model.repo_id:
            self.publish_model(self.config.model)

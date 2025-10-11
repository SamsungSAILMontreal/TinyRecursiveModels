from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from .arc_io import grid_to_tensor, tensor_to_grid, iter_arc_tasks
from .config import TTTConfig
from .losses import grid_ce_loss, proximal_penalty
from .lora import patch_lora
from .utils import Timer, clone_parameters, set_seed


@dataclass
class AdapterState:
    """Holds references to LoRA parameters and their frozen copies."""

    params: List[nn.Parameter]
    reference: List[torch.Tensor]

    def reset(self) -> None:
        for param, ref in zip(self.params, self.reference):
            param.data.copy_(ref.to(param.device))


@dataclass
class TaskResult:
    task_name: str
    train_acc: float
    test_preds: List[List[List[int]]]
    adaptation_time: float


def prepare_model_for_ttt(model: nn.Module, cfg: TTTConfig) -> AdapterState:
    adapter_params = patch_lora(
        model=model,
        name_filters=cfg.lora_targets or [],
        r=cfg.lora_r,
        alpha=cfg.lora_alpha,
        dropout=cfg.lora_dropout,
    )
    if not adapter_params:
        raise RuntimeError(
            "patch_lora did not find any linear-like layers matching the provided filters. "
            "Inspect model.named_modules() and adjust --lora-targets."
        )

    for param in model.parameters():
        param.requires_grad_(False)
    for param in adapter_params:
        param.requires_grad_(True)

    reference = clone_parameters(adapter_params)
    return AdapterState(params=list(adapter_params), reference=reference)


def _forward_grid(
    model: nn.Module,
    device: torch.device,
    grid: List[List[int]],
    target_grid: Optional[List[List[int]]] = None,
) -> torch.Tensor:
    target_shape: Optional[Tuple[int, int]] = None
    if target_grid is not None:
        target_arr = np.array(target_grid)
        target_shape = (int(target_arr.shape[0]), int(target_arr.shape[1]))

    if hasattr(model, "ttt_forward_grid"):
        return model.ttt_forward_grid(grid, target_shape=target_shape, device=device)

    x = grid_to_tensor(grid).unsqueeze(0).to(device)
    logits = model(x)[0]
    if target_shape is not None:
        h, w = target_shape
        logits = logits[:, :h, :w]
    return logits


@torch.no_grad()
def _accuracy_on_pairs(model: nn.Module, device: torch.device, pairs: Sequence[Dict]) -> float:
    scores: List[float] = []
    for pair in pairs:
        if hasattr(model, "ttt_predict_grid"):
            pred = model.ttt_predict_grid(pair["input"], device=device)
            pred_arr = np.array(pred, dtype=np.int64)
        else:
            logits = _forward_grid(model, device, pair["input"])
            pred_arr = tensor_to_grid(logits)
        target = np.array(pair["output"], dtype=np.int64)
        if pred_arr.shape != target.shape:
            scores.append(0.0)
        else:
            scores.append(float((pred_arr == target).mean()))
    return float(np.mean(scores)) if scores else 0.0


def adapt_on_task(
    model: nn.Module,
    task_json: Dict,
    cfg: TTTConfig,
    adapter_state: Optional[AdapterState] = None,
    device: Optional[torch.device] = None,
) -> Dict:
    if device is None:
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    if adapter_state is None:
        adapter_state = prepare_model_for_ttt(model, cfg)
    adapter_state.reset()

    train_pairs = task_json.get("train", [])
    if not train_pairs:
        raise ValueError("ARC task JSON contains no training pairs.")

    model.train()
    set_seed(cfg.seed)
    opt = torch.optim.AdamW(adapter_state.params, lr=cfg.lr, weight_decay=cfg.wd)
    reference = adapter_state.reference

    with Timer() as wall_timer:
        for _step in range(cfg.steps):
            opt.zero_grad(set_to_none=True)
            loss_total = 0.0
            for pair in train_pairs:
                logits = _forward_grid(model, device, pair["input"], pair["output"])
                target = torch.tensor(pair["output"], dtype=torch.long, device=device)
                loss = grid_ce_loss(logits, target)
                if cfg.prox_lambda > 0:
                    loss = loss + cfg.prox_lambda * proximal_penalty(adapter_state.params, reference)
                loss.backward()
                loss_total += float(loss.detach().cpu())
            if cfg.gradient_clip_norm is not None and cfg.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(adapter_state.params, cfg.gradient_clip_norm)
            opt.step()
    adaptation_time = wall_timer.elapsed

    model.eval()
    predictions: List[List[List[int]]] = []
    for test_entry in task_json.get("test", []):
        if hasattr(model, "ttt_predict_grid"):
            pred = model.ttt_predict_grid(test_entry["input"], device=device)
            predictions.append(np.array(pred, dtype=np.int64).tolist())
        else:
            logits = _forward_grid(model, device, test_entry["input"])
            pred = tensor_to_grid(logits)
            predictions.append(pred.tolist())

    train_acc = _accuracy_on_pairs(model, device, train_pairs)
    adapter_state.reset()

    return {
        "train_acc": train_acc,
        "test_preds": predictions,
        "adaptation_time": adaptation_time,
    }


def adapt_directory(
    model: nn.Module,
    arc_dir: str,
    cfg: TTTConfig,
    device: Optional[torch.device] = None,
) -> List[TaskResult]:
    if device is None:
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    adapter_state = prepare_model_for_ttt(model, cfg)
    results: List[TaskResult] = []

    for task_name, task_json in iter_arc_tasks(arc_dir):
        out = adapt_on_task(
            model=model,
            task_json=task_json,
            cfg=cfg,
            adapter_state=adapter_state,
            device=device,
        )
        results.append(
            TaskResult(
                task_name=task_name,
                train_acc=out["train_acc"],
                test_preds=out["test_preds"],
                adaptation_time=out["adaptation_time"],
            )
        )

    return results

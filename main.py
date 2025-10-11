from __future__ import annotations

import argparse
from typing import Sequence

import numpy as np

from ttt.arc_io import load_arc_task
from ttt.config import TTTConfig
from ttt.runner import TaskResult, adapt_directory, adapt_on_task
from ttt.trm_adapter import TRMArcAdapter


def _print_results(results: Sequence[TaskResult]) -> None:
    for item in results:
        shapes = [np.array(pred).shape for pred in item.test_preds]
        print(
            f"{item.task_name}: train_acc={item.train_acc:.3f} "
            f"adapt_time={item.adaptation_time:.2f}s "
            f"test_pred_shapes={shapes}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-task TTT scaffold for ARC puzzles.")
    parser.add_argument("--arc-dir", type=str, help="Directory containing ARC JSON tasks.")
    parser.add_argument("--task", type=str, help="Run TTT on a single ARC task JSON.")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional model checkpoint.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device string.")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--prox", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora-targets",
        type=str,
        default="",
        help="Comma-separated substrings used to match nn.Linear module names.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-horizon", type=int, default=1)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> TTTConfig:
    return TTTConfig(
        steps=args.steps,
        lr=args.lr,
        wd=args.wd,
        prox_lambda=args.prox,
        seed=args.seed,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=[tok for tok in args.lora_targets.split(",") if tok],
        device=args.device,
        max_horizon=args.max_horizon,
        gradient_clip_norm=args.clip_norm,
    )


def main() -> None:
    args = parse_args()
    cfg = make_config(args)

    if not args.task and not args.arc_dir:
        raise SystemExit("Provide either --task or --arc-dir.")

    model = TRMArcAdapter.from_checkpoint(args.checkpoint) if args.checkpoint else TRMArcAdapter()

    if args.task:
        task_json = load_arc_task(args.task)
        out = adapt_on_task(model, task_json, cfg)
        print(f"{args.task}: train_acc={out['train_acc']:.3f}")
        print(f"adaptation_time={out['adaptation_time']:.2f}s")
        print(f"test_predictions={out['test_preds']}")
    else:
        results = adapt_directory(model, args.arc_dir, cfg)
        _print_results(results)


if __name__ == "__main__":
    main()

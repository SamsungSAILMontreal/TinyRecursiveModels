from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

from .arc_io import (
    ARC_COLOR_OFFSET,
    ARC_MAX_SIZE,
    grid_to_arc_tokens,
    arc_tokens_to_grid,
)

COLOR_DIM = 10


def _default_trm_config() -> Dict:
    return {
        "halt_exploration_prob": 0.0,
        "halt_max_steps": 16,
        "H_cycles": 3,
        "L_cycles": 6,
        "H_layers": 0,
        "L_layers": 2,
        "hidden_size": 512,
        "num_heads": 8,
        "expansion": 4,
        "puzzle_emb_ndim": 512,
        "pos_encodings": "rope",
        "forward_dtype": "bfloat16",
        "mlp_t": False,
        "puzzle_emb_len": 16,
        "no_ACT_continue": True,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
    }


class TRMArcAdapter(nn.Module):
    """Lightweight wrapper around TinyRecursiveReasoningModel for ARC grids."""

    def __init__(self, config_overrides: Optional[Dict] = None, checkpoint: str = ""):
        super().__init__()
        base_cfg = _default_trm_config()
        if config_overrides:
            base_cfg.update(config_overrides)

        base_cfg.setdefault("batch_size", 1)
        base_cfg.setdefault("seq_len", ARC_MAX_SIZE * ARC_MAX_SIZE)
        base_cfg.setdefault("vocab_size", ARC_COLOR_OFFSET + 10)
        base_cfg.setdefault("num_puzzle_identifiers", 1)

        self.core = TinyRecursiveReasoningModel_ACTV1(base_cfg)
        self.config = base_cfg

        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            missing, unexpected = self.core.load_state_dict(state, strict=False)
            if missing:
                print(f"[TRMArcAdapter] Missing keys when loading checkpoint: {missing}")
            if unexpected:
                print(f"[TRMArcAdapter] Unexpected keys when loading checkpoint: {unexpected}")

    @classmethod
    def from_checkpoint(cls, path: str, config_overrides: Optional[Dict] = None) -> "TRMArcAdapter":
        return cls(config_overrides=config_overrides, checkpoint=path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits: List[torch.Tensor] = []
        device = x.device
        for sample in x:
            grid = sample.argmax(dim=0).to(torch.int64)
            logits.append(self.ttt_forward_grid(grid.cpu().numpy(), device=device))
        return torch.stack(logits, dim=0)

    # ========================
    # TTT helper API
    # ========================

    def ttt_forward_grid(
        self,
        grid,
        target_shape: Optional[Tuple[int, int]] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        device = device or next(self.core.parameters()).device
        tokens = grid_to_arc_tokens(grid).to(torch.int32).to(device)
        seq_logits = self._forward_tokens(tokens, device=device)  # (seq_len, vocab)

        color_logits = seq_logits[:, ARC_COLOR_OFFSET:].to(torch.float32)
        color_logits = color_logits.view(ARC_MAX_SIZE, ARC_MAX_SIZE, COLOR_DIM).permute(2, 0, 1)

        if target_shape:
            h, w = target_shape
            color_logits = color_logits[:, :h, :w]

        return color_logits

    def ttt_predict_grid(self, grid, device: Optional[torch.device] = None):
        device = device or next(self.core.parameters()).device
        tokens = grid_to_arc_tokens(grid).to(torch.int32).to(device)
        seq_logits = self._forward_tokens(tokens, device=device)
        preds = seq_logits.argmax(dim=-1)
        grid_pred = arc_tokens_to_grid(preds)
        return grid_pred.tolist()

    # ========================
    # Internal helpers
    # ========================

    def _build_batch(self, tokens: torch.Tensor, device: torch.device) -> Dict[str, torch.Tensor]:
        tokens = tokens.unsqueeze(0)
        return {
            "inputs": tokens.to(device),
            "labels": tokens.to(device),
            "puzzle_identifiers": torch.zeros(1, dtype=torch.int32, device=device),
        }

    def _forward_tokens(self, tokens: torch.Tensor, device: torch.device) -> torch.Tensor:
        batch = self._build_batch(tokens, device=device)
        carry = self.core.initial_carry(batch)
        carry, outputs = self.core(carry=carry, batch=batch)
        return outputs["logits"][0]  # (seq_len, vocab)

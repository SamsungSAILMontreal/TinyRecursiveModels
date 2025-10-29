import argparse
import json
import torch
import yaml
import numpy as np
from omegaconf import OmegaConf

from utils.functions import load_model_class
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

def grid_to_sequence(grid):
    ARCMaxGridSize = 30
    grid = np.array(grid, dtype=np.uint8)
    nrow, ncol = grid.shape
    grid = np.pad(grid + 2, ((0, ARCMaxGridSize - nrow), (0, ARCMaxGridSize - ncol)), constant_values=0)
    eos_row, eos_col = nrow, ncol
    if eos_row < ARCMaxGridSize:
        grid[eos_row, 0:eos_col] = 1
    if eos_col < ARCMaxGridSize:
        grid[0:eos_row, eos_col] = 1
    return grid.flatten()

def sequence_to_grid(sequence):
    ARCMaxGridSize = 30
    grid = sequence.reshape(ARCMaxGridSize, ARCMaxGridSize)
    eos_indices = np.where(grid == 1)
    if len(eos_indices[0]) > 0 and len(eos_indices[1]) > 0:
        nrow = eos_indices[0].min()
        ncol = eos_indices[1].min()
        grid = grid[:nrow, :ncol]
    return (grid - 2).tolist()

def load_config(config_path="config/cfg_pretrain.yaml", arch_config_path="config/arch/trm.yaml"):
    config = OmegaConf.load(config_path)
    arch_config = OmegaConf.load(arch_config_path)
    config.arch = arch_config
    return config

def load_model(config, checkpoint_path):
    model_cls = load_model_class(config.arch.name)
    model_cfg = dict(
        **config.arch,
        batch_size=1,
        vocab_size=12,
        seq_len=900,
        num_puzzle_identifiers=1001,
        causal=False
    )
    model = model_cls(model_cfg)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="Run inference on ARC-AGI-2 challenges.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input JSON challenge file.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the submission JSON file.")
    parser.add_argument("--config_path", type=str, default="config/cfg_pretrain.yaml", help="Path to the main config file.")
    parser.add_argument("--arch_config_path", type=str, default="config/arch/trm.yaml", help="Path to the architecture config file.")
    args = parser.parse_args()

    config = load_config(args.config_path, args.arch_config_path)
    model = load_model(config, args.checkpoint_path)

    with open(args.input_path, 'r') as f:
        challenges = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    predictions = {}
    for task_id, task in challenges.items():
        task_predictions = []
        for test_case in task['test']:
            test_input = test_case['input']

            input_sequence = grid_to_sequence(test_input)
            input_tensor = torch.tensor(input_sequence, dtype=torch.long).unsqueeze(0).to(device)

            batch = {
                'inputs': input_tensor,
                'labels': torch.zeros_like(input_tensor), # Dummy labels
                'puzzle_identifiers': torch.tensor([1], dtype=torch.long).to(device), # Dummy identifier
                'puzzle_indices': torch.tensor([0, 1], dtype=torch.long).to(device), # Dummy indices
                'group_indices': torch.tensor([0, 1], dtype=torch.long).to(device) # Dummy indices
            }

            with torch.no_grad():
                carry = model.initial_carry(batch)
                _, _, _, preds, _ = model(carry=carry, batch=batch, return_keys=["outputs"])

            output_sequence = preds['outputs'].squeeze(0).cpu().numpy()

            predicted_output = sequence_to_grid(output_sequence)
            task_predictions.append(predicted_output)
        predictions[task_id] = task_predictions

    with open(args.output_path, 'w') as f:
        json.dump(predictions, f)

if __name__ == "__main__":
    main()

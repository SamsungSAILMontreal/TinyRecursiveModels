#!/usr/bin/env python3
"""
RNA 3D Structure Prediction Training Script

This script trains a Tiny Recursive Model (TRM) to predict 3D RNA structures.
The model learns to predict C1' atom coordinates for each nucleotide in an RNA sequence,
generating 5 different conformations as required by the competition format.
"""

import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

# Import TRM components
from models.recursive_reasoning.trm import TinyRecursionModel
from models.common import get_activation


class RNAStructureDataset(Dataset):
    """Dataset for RNA 3D structure prediction."""
    
    def __init__(self, sequences_file: str, labels_file: str, max_length: int = 500):
        """
        Initialize RNA structure dataset.
        
        Args:
            sequences_file: Path to sequences JSON file
            labels_file: Path to labels JSON file
            max_length: Maximum sequence length
        """
        with open(sequences_file, 'r') as f:
            self.sequences = json.load(f)
        
        with open(labels_file, 'r') as f:
            self.labels = json.load(f)
        
        # Create mapping from target_id to labels
        self.label_map = {label['target_id']: label for label in self.labels}
        
        # Filter sequences that have labels
        self.sequences = [seq for seq in self.sequences 
                         if seq['target_id'] in self.label_map]
        
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Returns:
            Tuple of (encoded_sequence, coordinates)
            - encoded_sequence: (max_length,) tensor with nucleotide encoding
            - coordinates: (seq_len, num_structures, 3) tensor with 3D coordinates
        """
        seq_data = self.sequences[idx]
        label_data = self.label_map[seq_data['target_id']]
        
        # Get encoded sequence
        encoded_seq = torch.tensor(seq_data['encoded_sequence'], dtype=torch.long)
        
        # Get coordinates
        coords_list = []
        for residue in label_data['residues']:
            coords = torch.tensor(residue['coordinates'], dtype=torch.float32)
            coords_list.append(coords)
        
        # Stack coordinates: (seq_len, num_structures, 3)
        if coords_list:
            coordinates = torch.stack(coords_list)
        else:
            # Empty coordinates
            coordinates = torch.zeros((self.max_length, 1, 3), dtype=torch.float32)
        
        # Pad coordinates to max_length
        if coordinates.shape[0] < self.max_length:
            padding = torch.zeros(
                (self.max_length - coordinates.shape[0], coordinates.shape[1], 3),
                dtype=torch.float32
            )
            coordinates = torch.cat([coordinates, padding], dim=0)
        else:
            coordinates = coordinates[:self.max_length]
        
        return encoded_seq, coordinates


class RNAStructureModel(nn.Module):
    """TRM-based model for RNA 3D structure prediction."""
    
    def __init__(
        self,
        vocab_size: int = 4,  # A, C, G, U
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_structures: int = 5,
        max_length: int = 500,
        H_cycles: int = 3,
        L_cycles: int = 6,
        L_layers: int = 2,
    ):
        """
        Initialize RNA structure prediction model.
        
        Args:
            vocab_size: Number of nucleotide types
            embed_dim: Embedding dimension
            hidden_dim: Hidden dimension for TRM
            num_structures: Number of structures to predict
            max_length: Maximum sequence length
            H_cycles: Number of high-level reasoning cycles
            L_cycles: Number of low-level reasoning cycles
            L_layers: Number of layers in TRM
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_structures = num_structures
        self.max_length = max_length
        
        # Nucleotide embedding (vocab_size=4 for ACGU, +1 for padding)
        self.nucleotide_embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=vocab_size)
        
        # Position embedding
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        
        # TRM backbone (we'll use a simplified version)
        self.reasoning_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=8,
                dim_feedforward=hidden_dim,
                dropout=0.1,
                batch_first=True
            )
            for _ in range(L_layers)
        ])
        
        # Output heads for each structure
        self.structure_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 3)  # x, y, z coordinates
            )
            for _ in range(num_structures)
        ])
        
        # Recursive refinement
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
    
    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to predict 3D structures.
        
        Args:
            sequences: (batch_size, max_length) tensor of encoded sequences
            
        Returns:
            predictions: (batch_size, max_length, num_structures, 3) tensor
        """
        batch_size, seq_len = sequences.shape
        
        # Embed nucleotides
        # Handle padding index -1 by clamping
        sequences_clamped = torch.clamp(sequences, min=0)
        x = self.nucleotide_embedding(sequences_clamped)  # (batch, seq_len, embed_dim)
        
        # Add position embeddings
        positions = torch.arange(seq_len, device=sequences.device).unsqueeze(0)
        x = x + self.position_embedding(positions)
        
        # Create attention mask for padding (padding index is vocab_size, which is 4)
        padding_mask = sequences == self.vocab_size  # (batch, seq_len)
        
        # Recursive reasoning with multiple cycles
        for h_cycle in range(self.H_cycles):
            for l_cycle in range(self.L_cycles):
                # Apply transformer layers
                for layer in self.reasoning_layers:
                    x = layer(x, src_key_padding_mask=padding_mask)
        
        # Predict structures with each head
        predictions = []
        for head in self.structure_heads:
            coords = head(x)  # (batch, seq_len, 3)
            predictions.append(coords)
        
        # Stack predictions: (batch, seq_len, num_structures, 3)
        predictions = torch.stack(predictions, dim=2)
        
        return predictions


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    clip_coords: bool = True
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for sequences, targets in tqdm(dataloader, desc="Training"):
        sequences = sequences.to(device)
        targets = targets.to(device)
        
        # Forward pass
        predictions = model(sequences)
        
        # Clip coordinates if needed (competition requirement)
        if clip_coords:
            predictions = torch.clamp(predictions, -999.999, 9999.999)
        
        # Compute loss (MSE on coordinates)
        # Only compute loss on non-padding positions (padding index is 4)
        mask = (sequences != 4).unsqueeze(-1).unsqueeze(-1)  # (batch, seq_len, 1, 1)
        
        # If targets have fewer structures, repeat the last one
        if targets.shape[2] < predictions.shape[2]:
            last_structure = targets[:, :, -1:, :]  # (batch, seq_len, 1, 3)
            padding_structures = last_structure.repeat(
                1, 1, predictions.shape[2] - targets.shape[2], 1
            )
            targets = torch.cat([targets, padding_structures], dim=2)
        
        loss = ((predictions - targets) ** 2 * mask).sum() / mask.sum()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> float:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for sequences, targets in tqdm(dataloader, desc="Validating"):
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            # Forward pass
            predictions = model(sequences)
            predictions = torch.clamp(predictions, -999.999, 9999.999)
            
            # Compute loss
            mask = (sequences != 4).unsqueeze(-1).unsqueeze(-1)
            
            # Handle different number of structures
            if targets.shape[2] < predictions.shape[2]:
                last_structure = targets[:, :, -1:, :]
                padding_structures = last_structure.repeat(
                    1, 1, predictions.shape[2] - targets.shape[2], 1
                )
                targets = torch.cat([targets, padding_structures], dim=2)
            
            loss = ((predictions - targets) ** 2 * mask).sum() / mask.sum()
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='Train TRM for RNA 3D structure prediction')
    parser.add_argument('--data-dir', type=str, default='data/rna-structure',
                        help='Directory with processed RNA data')
    parser.add_argument('--output-dir', type=str, default='checkpoints/rna',
                        help='Directory to save model checkpoints')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--max-length', type=int, default=500,
                        help='Maximum sequence length')
    parser.add_argument('--num-structures', type=int, default=5,
                        help='Number of structures to predict')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = RNAStructureDataset(
        f"{args.data_dir}/train_sequences.json",
        f"{args.data_dir}/train_labels.json",
        args.max_length
    )
    
    val_dataset = RNAStructureDataset(
        f"{args.data_dir}/val_sequences.json",
        f"{args.data_dir}/val_labels.json",
        args.max_length
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Create model
    print("Creating model...")
    device = torch.device(args.device)
    model = RNAStructureModel(
        num_structures=args.num_structures,
        max_length=args.max_length
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Train loss: {train_loss:.6f}")
        
        # Validate
        val_loss = validate(model, val_loader, device)
        print(f"Val loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, output_path / 'best_model.pth')
            print(f"Saved best model (val_loss: {val_loss:.6f})")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, output_path / f'checkpoint_epoch_{epoch + 1}.pth')
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == '__main__':
    main()

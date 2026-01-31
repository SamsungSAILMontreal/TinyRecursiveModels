#!/usr/bin/env python3
"""
Kaggle Competition Submission Script for Stanford RNA 3D Folding Part 2

This script is designed to run in a Kaggle notebook environment.
It reads test_sequences.csv and outputs submission.csv with 5 predicted structures.

Competition Requirements:
- Read from: test_sequences.csv
- Output to: submission.csv
- Format: ID, resname, resid, x_1, y_1, z_1, ..., x_5, y_5, z_5
- Must predict 5 structures per sequence
- C1' atom coordinates for each residue
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm


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
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_structures = num_structures
        self.max_length = max_length
        
        # Nucleotide embedding (vocab_size=4 for ACGU, +1 for padding)
        self.nucleotide_embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=vocab_size)
        
        # Position embedding
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        
        # TRM backbone
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
        
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
    
    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = sequences.shape
        
        # Embed nucleotides and add position embeddings
        sequences_clamped = torch.clamp(sequences, min=0)
        x = self.nucleotide_embedding(sequences_clamped)
        positions = torch.arange(seq_len, device=sequences.device).unsqueeze(0)
        x = x + self.position_embedding(positions)
        
        # Create attention mask
        padding_mask = sequences == self.vocab_size
        
        # Recursive reasoning
        for h_cycle in range(self.H_cycles):
            for l_cycle in range(self.L_cycles):
                for layer in self.reasoning_layers:
                    x = layer(x, src_key_padding_mask=padding_mask)
        
        # Predict structures
        predictions = []
        for head in self.structure_heads:
            coords = head(x)
            predictions.append(coords)
        
        predictions = torch.stack(predictions, dim=2)
        return predictions


def encode_rna_sequence(sequence: str, max_length: int = 500) -> np.ndarray:
    """Encode RNA sequence into numerical format."""
    nucleotide_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'N': 4}
    encoded = np.array([nucleotide_map.get(n, 4) for n in sequence.upper()])
    
    if len(encoded) < max_length:
        encoded = np.pad(encoded, (0, max_length - len(encoded)), constant_values=4)
    else:
        encoded = encoded[:max_length]
    
    return encoded


def load_test_sequences(csv_file: str = 'test_sequences.csv') -> pd.DataFrame:
    """Load test sequences from CSV file."""
    return pd.read_csv(csv_file)


def predict_structures(
    model: nn.Module,
    sequences_df: pd.DataFrame,
    device: torch.device,
    max_length: int = 500,
    batch_size: int = 16
) -> List[Dict]:
    """Generate structure predictions."""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences_df), batch_size), desc="Predicting"):
            batch_df = sequences_df.iloc[i:i + batch_size]
            
            # Encode sequences
            encoded_seqs = []
            for _, row in batch_df.iterrows():
                encoded = encode_rna_sequence(row['sequence'], max_length)
                encoded_seqs.append(torch.tensor(encoded, dtype=torch.long))
            
            # Stack and predict
            encoded_batch = torch.stack(encoded_seqs).to(device)
            pred_coords = model(encoded_batch)
            pred_coords = torch.clamp(pred_coords, -999.999, 9999.999)
            pred_coords = pred_coords.cpu().numpy()
            
            # Store predictions
            for j, (idx, row) in enumerate(batch_df.iterrows()):
                seq_len = min(len(row['sequence']), max_length)
                target_id = row['target_id']
                sequence = row['sequence'][:max_length]  # Truncate sequence to max_length
                coords = pred_coords[j, :seq_len, :, :]
                
                predictions.append({
                    'target_id': target_id,
                    'sequence': sequence,
                    'coordinates': coords,
                })
    
    return predictions


def create_submission(predictions: List[Dict], output_file: str = 'submission.csv'):
    """Create Kaggle submission file."""
    rows = []
    
    for pred in predictions:
        target_id = pred['target_id']
        sequence = pred['sequence']
        coords = pred['coordinates']  # (seq_len, num_structures, 3)
        
        for i, nucleotide in enumerate(sequence):
            row = {
                'ID': f"{target_id}_{i + 1}",
                'resname': nucleotide,
                'resid': i + 1,
            }
            
            # Add coordinates for 5 structures
            for struct_idx in range(5):
                if struct_idx < coords.shape[1]:
                    x, y, z = coords[i, struct_idx, :]
                else:
                    x, y, z = coords[i, -1, :]
                
                row[f'x_{struct_idx + 1}'] = x
                row[f'y_{struct_idx + 1}'] = y
                row[f'z_{struct_idx + 1}'] = z
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Ensure correct column order
    coord_cols = []
    for i in range(1, 6):
        coord_cols.extend([f'x_{i}', f'y_{i}', f'z_{i}'])
    columns = ['ID', 'resname', 'resid'] + coord_cols
    df = df[columns]
    
    df.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")
    print(f"Total predictions: {len(df)} residues from {len(predictions)} sequences")
    return df


def main():
    """Main competition submission pipeline."""
    print("=" * 70)
    print("Stanford RNA 3D Folding Part 2 - Kaggle Submission")
    print("=" * 70)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    max_length = 500
    batch_size = 32 if torch.cuda.is_available() else 16
    
    # Load test sequences
    print("\n1. Loading test sequences...")
    test_sequences = load_test_sequences('test_sequences.csv')
    print(f"   Loaded {len(test_sequences)} test sequences")
    
    # Initialize model
    print("\n2. Initializing model...")
    model = RNAStructureModel(
        vocab_size=4,
        embed_dim=256,
        hidden_dim=512,
        num_structures=5,
        max_length=max_length,
        H_cycles=3,
        L_cycles=6,
        L_layers=2,
    ).to(device)
    
    # Load pretrained weights if available
    checkpoint_path = 'rna_model.pth'
    if os.path.exists(checkpoint_path):
        print(f"   Loading pretrained weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"   WARNING: No pretrained model found at {checkpoint_path}")
        print("   Using randomly initialized weights (for testing only)")
    
    # Generate predictions
    print("\n3. Generating predictions...")
    predictions = predict_structures(
        model,
        test_sequences,
        device,
        max_length,
        batch_size
    )
    
    # Create submission file
    print("\n4. Creating submission file...")
    submission_df = create_submission(predictions, 'submission.csv')
    
    print("\n" + "=" * 70)
    print("Submission complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Verify submission.csv format")
    print("  2. Submit to Kaggle competition")
    print(f"  3. Expected TM-score evaluation on {len(test_sequences)} sequences")
    

if __name__ == '__main__':
    main()

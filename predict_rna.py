#!/usr/bin/env python3
"""
RNA 3D Structure Prediction Script

This script generates predictions for RNA 3D structures using a trained TRM model.
It produces 5 different conformations for each RNA sequence as required by the
competition format.
"""

import os
import argparse
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

# Import model
from pretrain_rna import RNAStructureModel, RNAStructureDataset


def predict_structures(
    model: torch.nn.Module,
    sequences_file: str,
    device: torch.device,
    max_length: int = 500,
    batch_size: int = 16
) -> List[Dict]:
    """
    Generate structure predictions for RNA sequences.
    
    Args:
        model: Trained RNA structure prediction model
        sequences_file: Path to sequences JSON file
        device: Device to run predictions on
        max_length: Maximum sequence length
        batch_size: Batch size for inference
        
    Returns:
        List of prediction dictionaries
    """
    model.eval()
    
    # Load sequences
    with open(sequences_file, 'r') as f:
        sequences = json.load(f)
    
    predictions = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="Predicting"):
            batch_sequences = sequences[i:i + batch_size]
            
            # Prepare batch
            encoded_seqs = []
            for seq_data in batch_sequences:
                encoded = torch.tensor(seq_data['encoded_sequence'], dtype=torch.long)
                encoded_seqs.append(encoded)
            
            # Stack into batch
            encoded_batch = torch.stack(encoded_seqs).to(device)
            
            # Predict
            pred_coords = model(encoded_batch)  # (batch, seq_len, num_structures, 3)
            pred_coords = torch.clamp(pred_coords, -999.999, 9999.999)
            
            # Convert to CPU and numpy
            pred_coords = pred_coords.cpu().numpy()
            
            # Store predictions
            for j, seq_data in enumerate(batch_sequences):
                seq_len = seq_data['length']
                target_id = seq_data['target_id']
                sequence = seq_data['sequence']
                
                # Extract coordinates for actual sequence length
                coords = pred_coords[j, :seq_len, :, :]  # (seq_len, num_structures, 3)
                
                predictions.append({
                    'target_id': target_id,
                    'sequence': sequence,
                    'coordinates': coords,
                })
    
    return predictions


def save_kaggle_submission(
    predictions: List[Dict],
    output_file: str,
    num_structures: int = 5
):
    """
    Save predictions in Kaggle submission format.
    
    Args:
        predictions: List of prediction dictionaries
        output_file: Path to save submission CSV
        num_structures: Number of structures in predictions
    """
    rows = []
    
    for pred in predictions:
        target_id = pred['target_id']
        sequence = pred['sequence']
        coords = pred['coordinates']  # (seq_len, num_structures, 3)
        
        # Create row for each residue
        for i, nucleotide in enumerate(sequence):
            row = {
                'ID': f"{target_id}_{i + 1}",  # 1-based indexing
                'resname': nucleotide,
                'resid': i + 1,
            }
            
            # Add coordinates for each structure
            for struct_idx in range(num_structures):
                if struct_idx < coords.shape[1]:
                    x, y, z = coords[i, struct_idx, :]
                else:
                    # If we don't have enough structures, repeat the last one
                    x, y, z = coords[i, -1, :]
                
                row[f'x_{struct_idx + 1}'] = x
                row[f'y_{struct_idx + 1}'] = y
                row[f'z_{struct_idx + 1}'] = z
            
            rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Reorder columns to match expected format
    coord_cols = []
    for i in range(1, num_structures + 1):
        coord_cols.extend([f'x_{i}', f'y_{i}', f'z_{i}'])
    
    columns = ['ID', 'resname', 'resid'] + coord_cols
    df = df[columns]
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Saved submission to {output_file}")
    print(f"Total residues: {len(df)}")


def main():
    parser = argparse.ArgumentParser(description='Generate RNA structure predictions')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--sequences', type=str, required=True,
                        help='Path to sequences JSON file')
    parser.add_argument('--output', type=str, default='submission.csv',
                        help='Output submission file')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for inference')
    parser.add_argument('--max-length', type=int, default=500,
                        help='Maximum sequence length')
    parser.add_argument('--num-structures', type=int, default=5,
                        help='Number of structures to predict')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    device = torch.device(args.device)
    
    model = RNAStructureModel(
        num_structures=args.num_structures,
        max_length=args.max_length
    ).to(device)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Validation loss: {checkpoint['val_loss']:.6f}")
    
    # Generate predictions
    print("Generating predictions...")
    predictions = predict_structures(
        model,
        args.sequences,
        device,
        args.max_length,
        args.batch_size
    )
    
    print(f"Generated predictions for {len(predictions)} sequences")
    
    # Save in Kaggle format
    print("Saving submission...")
    save_kaggle_submission(predictions, args.output, args.num_structures)
    
    print("\nPrediction complete!")
    print(f"Submission saved to {args.output}")


if __name__ == '__main__':
    main()

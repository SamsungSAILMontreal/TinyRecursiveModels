#!/usr/bin/env python3
"""
RNA 3D Structure Dataset Builder

This script creates datasets for RNA 3D structure prediction using the
jrc-rna-structure-pipeline. It processes RNA sequences and structures
for training TinyRecursiveModels on the task of predicting 3D coordinates
of RNA molecules.

Dataset format follows the Stanford RNA 3D Folding competition:
- Input: RNA sequences (ACGU nucleotides)
- Output: 5 predicted 3D structures (x,y,z coordinates for C1' atoms)
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json


def parse_fasta(fasta_string: str) -> Dict[str, str]:
    """
    Parse FASTA-formatted string into a dictionary.
    
    Args:
        fasta_string: FASTA-formatted string with headers and sequences
        
    Returns:
        Dictionary mapping sequence names to sequences
    """
    sequences = {}
    current_name = None
    current_seq = []
    
    for line in fasta_string.strip().split('\n'):
        if line.startswith('>'):
            if current_name is not None:
                sequences[current_name] = ''.join(current_seq)
            current_name = line[1:].split()[0]  # Take first word after >
            current_seq = []
        else:
            current_seq.append(line.strip())
    
    if current_name is not None:
        sequences[current_name] = ''.join(current_seq)
    
    return sequences


def encode_rna_sequence(sequence: str, max_length: int = 500) -> np.ndarray:
    """
    Encode RNA sequence into numerical format.
    
    Args:
        sequence: RNA sequence string (ACGU)
        max_length: Maximum sequence length (pad/truncate)
        
    Returns:
        Numpy array of shape (max_length,) with nucleotide encoding
        A=0, C=1, G=2, U=3, N=4 (also used for padding)
    """
    nucleotide_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'N': 4}
    
    # Encode sequence
    encoded = np.array([nucleotide_map.get(n, 4) for n in sequence.upper()])
    
    # Pad or truncate to max_length
    if len(encoded) < max_length:
        encoded = np.pad(encoded, (0, max_length - len(encoded)), constant_values=4)
    else:
        encoded = encoded[:max_length]
    
    return encoded


def parse_coordinates(row: pd.Series, num_structures: int = 1) -> np.ndarray:
    """
    Parse 3D coordinates from label row.
    
    Args:
        row: DataFrame row with x_i, y_i, z_i columns
        num_structures: Number of structures in the row
        
    Returns:
        Numpy array of shape (num_structures, 3) with coordinates
    """
    coords = []
    for i in range(1, num_structures + 1):
        x_col = f'x_{i}'
        y_col = f'y_{i}'
        z_col = f'z_{i}'
        
        if x_col in row.index and y_col in row.index and z_col in row.index:
            x = row[x_col] if pd.notna(row[x_col]) else 0.0
            y = row[y_col] if pd.notna(row[y_col]) else 0.0
            z = row[z_col] if pd.notna(row[z_col]) else 0.0
            coords.append([x, y, z])
    
    return np.array(coords)


def load_rna_data(
    sequences_file: str,
    labels_file: Optional[str] = None,
    max_length: int = 500
) -> Tuple[List[Dict], List[Dict]]:
    """
    Load RNA sequences and labels from CSV files.
    
    Args:
        sequences_file: Path to sequences CSV file
        labels_file: Optional path to labels CSV file
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (sequences, labels) lists of dictionaries
    """
    # Load sequences
    seq_df = pd.read_csv(sequences_file)
    sequences = []
    
    for idx, row in seq_df.iterrows():
        seq_data = {
            'target_id': row['target_id'],
            'sequence': row['sequence'],
            'encoded_sequence': encode_rna_sequence(row['sequence'], max_length),
            'length': len(row['sequence']),
            'temporal_cutoff': row.get('temporal_cutoff', ''),
            'description': row.get('description', ''),
            'stoichiometry': row.get('stoichiometry', ''),
        }
        sequences.append(seq_data)
    
    # Load labels if provided
    labels = []
    if labels_file and os.path.exists(labels_file):
        label_df = pd.read_csv(labels_file)
        
        # Extract base target_id from ID column (format: target_id_residue_num)
        label_df['base_target_id'] = label_df['ID'].apply(lambda x: '_'.join(x.split('_')[:-1]))
        
        # Group by base target_id to get all residues for each target
        for base_id, group in label_df.groupby('base_target_id'):
            # Get coordinates for each residue in order
            coords_list = []
            for _, residue_row in group.iterrows():
                # Determine how many structures are present
                num_structures = 0
                while f'x_{num_structures + 1}' in residue_row.index:
                    num_structures += 1
                
                if num_structures > 0:
                    coords = parse_coordinates(residue_row, num_structures)
                    coords_list.append({
                        'resid': residue_row['resid'],
                        'resname': residue_row['resname'],
                        'coordinates': coords,
                    })
            
            if coords_list:
                labels.append({
                    'target_id': base_id,
                    'residues': coords_list,
                })
    
    return sequences, labels


def create_dataset_splits(
    sequences: List[Dict],
    labels: List[Dict],
    train_ratio: float = 0.8,
    temporal_cutoff: Optional[str] = None
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """
    Split dataset into train and validation sets.
    
    Args:
        sequences: List of sequence dictionaries
        labels: List of label dictionaries
        train_ratio: Ratio of data to use for training (if no temporal split)
        temporal_cutoff: Date string for temporal split (YYYY-MM-DD)
        
    Returns:
        Tuple of (train_sequences, train_labels, val_sequences, val_labels)
    """
    if temporal_cutoff:
        # Temporal split based on cutoff date
        train_sequences = [s for s in sequences if s['temporal_cutoff'] < temporal_cutoff]
        val_sequences = [s for s in sequences if s['temporal_cutoff'] >= temporal_cutoff]
        
        train_ids = {s['target_id'] for s in train_sequences}
        val_ids = {s['target_id'] for s in val_sequences}
        
        train_labels = [l for l in labels if l['target_id'] in train_ids]
        val_labels = [l for l in labels if l['target_id'] in val_ids]
    else:
        # Random split
        np.random.shuffle(sequences)
        split_idx = int(len(sequences) * train_ratio)
        
        train_sequences = sequences[:split_idx]
        val_sequences = sequences[split_idx:]
        
        train_ids = {s['target_id'] for s in train_sequences}
        val_ids = {s['target_id'] for s in val_sequences}
        
        train_labels = [l for l in labels if l['target_id'] in train_ids]
        val_labels = [l for l in labels if l['target_id'] in val_ids]
    
    return train_sequences, train_labels, val_sequences, val_labels


def save_dataset(
    sequences: List[Dict],
    labels: List[Dict],
    output_dir: str,
    split_name: str
):
    """
    Save dataset to disk in TRM format.
    
    Args:
        sequences: List of sequence dictionaries
        labels: List of label dictionaries
        output_dir: Directory to save dataset
        split_name: Name of the split (train/val/test)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save sequences
    seq_data = []
    for seq in sequences:
        seq_data.append({
            'target_id': seq['target_id'],
            'sequence': seq['sequence'],
            'length': seq['length'],
            'encoded_sequence': seq['encoded_sequence'].tolist(),
        })
    
    with open(output_path / f'{split_name}_sequences.json', 'w') as f:
        json.dump(seq_data, f)
    
    # Save labels
    label_data = []
    for label in labels:
        residue_data = []
        for res in label['residues']:
            residue_data.append({
                'resid': int(res['resid']),
                'resname': res['resname'],
                'coordinates': res['coordinates'].tolist(),
            })
        label_data.append({
            'target_id': label['target_id'],
            'residues': residue_data,
        })
    
    with open(output_path / f'{split_name}_labels.json', 'w') as f:
        json.dump(label_data, f)
    
    print(f"Saved {len(sequences)} sequences and {len(labels)} labels to {output_path}/{split_name}_*.json")


def main():
    parser = argparse.ArgumentParser(description='Build RNA 3D structure prediction dataset')
    parser.add_argument('--sequences', type=str, required=True,
                        help='Path to sequences CSV file (e.g., train_sequences.csv)')
    parser.add_argument('--labels', type=str, default=None,
                        help='Path to labels CSV file (e.g., train_labels.csv)')
    parser.add_argument('--output-dir', type=str, default='data/rna-structure',
                        help='Output directory for processed dataset')
    parser.add_argument('--max-length', type=int, default=500,
                        help='Maximum sequence length (pad/truncate)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Ratio of data for training (if no temporal split)')
    parser.add_argument('--temporal-cutoff', type=str, default=None,
                        help='Temporal cutoff date for train/val split (YYYY-MM-DD)')
    parser.add_argument('--no-split', action='store_true',
                        help='Do not split into train/val, save all as single dataset')
    
    args = parser.parse_args()
    
    print("Loading RNA data...")
    sequences, labels = load_rna_data(
        args.sequences,
        args.labels,
        args.max_length
    )
    
    print(f"Loaded {len(sequences)} sequences and {len(labels)} label sets")
    
    if args.no_split:
        # Save all data as a single dataset
        save_dataset(sequences, labels, args.output_dir, 'all')
    else:
        # Split into train and validation
        print("Splitting dataset...")
        train_seq, train_lab, val_seq, val_lab = create_dataset_splits(
            sequences, labels, args.train_ratio, args.temporal_cutoff
        )
        
        print(f"Train: {len(train_seq)} sequences, {len(train_lab)} labels")
        print(f"Val: {len(val_seq)} sequences, {len(val_lab)} labels")
        
        # Save splits
        save_dataset(train_seq, train_lab, args.output_dir, 'train')
        save_dataset(val_seq, val_lab, args.output_dir, 'val')
    
    print(f"\nDataset saved to {args.output_dir}")
    print("\nNext steps:")
    print("1. Train TRM model: python pretrain_rna.py")
    print("2. Generate predictions: python predict_rna.py")


if __name__ == '__main__':
    main()

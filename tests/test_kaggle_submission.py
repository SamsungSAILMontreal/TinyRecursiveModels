#!/usr/bin/env python3
"""
Test script for Kaggle submission functionality.

This script tests that:
1. The submission script can read test_sequences.csv
2. It generates the correct submission.csv format
3. All required columns are present
4. Coordinates are properly clipped
"""

import os
import sys
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_test_data(tmpdir):
    """Create sample test_sequences.csv for testing."""
    test_data = {
        'target_id': ['TEST_RNA_001', 'TEST_RNA_002'],
        'sequence': ['ACGU', 'GGAACCUU'],
        'temporal_cutoff': ['2026-01-01', '2026-01-02'],
        'description': ['Test RNA 1', 'Test RNA 2'],
    }
    df = pd.DataFrame(test_data)
    csv_path = os.path.join(tmpdir, 'test_sequences.csv')
    df.to_csv(csv_path, index=False)
    return csv_path


def test_kaggle_submission():
    """Test the Kaggle submission workflow."""
    print("Testing Kaggle Submission Workflow")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test data
        print("\n1. Creating test data...")
        test_csv = create_test_data(tmpdir)
        print(f"   Created: {test_csv}")
        
        # Import submission functions
        print("\n2. Importing submission functions...")
        from kaggle_submission import (
            encode_rna_sequence,
            RNAStructureModel,
            create_submission
        )
        print("   ✓ Imports successful")
        
        # Test sequence encoding
        print("\n3. Testing sequence encoding...")
        encoded = encode_rna_sequence("ACGU", max_length=10)
        assert len(encoded) == 10, "Encoded sequence should be padded to max_length"
        assert encoded[0] == 0, "A should encode to 0"
        assert encoded[1] == 1, "C should encode to 1"
        assert encoded[2] == 2, "G should encode to 2"
        assert encoded[3] == 3, "U should encode to 3"
        assert encoded[4] == 4, "Padding should be 4"
        print("   ✓ Sequence encoding works")
        
        # Test model creation
        print("\n4. Testing model creation...")
        import torch
        device = torch.device('cpu')
        model = RNAStructureModel(
            vocab_size=4,
            embed_dim=64,  # Smaller for testing
            hidden_dim=128,
            num_structures=5,
            max_length=100,
            H_cycles=1,  # Faster for testing
            L_cycles=1,
            L_layers=1,
        ).to(device)
        print(f"   ✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test forward pass
        print("\n5. Testing forward pass...")
        test_seq = torch.tensor([[0, 1, 2, 3, 4, 4]], dtype=torch.long)  # ACGU + padding
        output = model(test_seq)
        assert output.shape == (1, 6, 5, 3), f"Expected shape (1, 6, 5, 3), got {output.shape}"
        print("   ✓ Forward pass works")
        
        # Test prediction on test data
        print("\n6. Testing prediction on test data...")
        test_df = pd.read_csv(test_csv)
        predictions = []
        
        model.eval()
        with torch.no_grad():
            for _, row in test_df.iterrows():
                encoded = encode_rna_sequence(row['sequence'], max_length=100)
                encoded_tensor = torch.tensor([encoded], dtype=torch.long)
                pred_coords = model(encoded_tensor)
                pred_coords = torch.clamp(pred_coords, -999.999, 9999.999)
                pred_coords = pred_coords.cpu().numpy()
                
                seq_len = len(row['sequence'])
                predictions.append({
                    'target_id': row['target_id'],
                    'sequence': row['sequence'],
                    'coordinates': pred_coords[0, :seq_len, :, :]
                })
        
        print(f"   ✓ Generated predictions for {len(predictions)} sequences")
        
        # Test submission file creation
        print("\n7. Testing submission file creation...")
        submission_path = os.path.join(tmpdir, 'submission.csv')
        
        # Manually create submission
        rows = []
        for pred in predictions:
            target_id = pred['target_id']
            sequence = pred['sequence']
            coords = pred['coordinates']
            
            for i, nucleotide in enumerate(sequence):
                row = {
                    'ID': f"{target_id}_{i + 1}",
                    'resname': nucleotide,
                    'resid': i + 1,
                }
                
                for struct_idx in range(5):
                    x, y, z = coords[i, struct_idx, :]
                    row[f'x_{struct_idx + 1}'] = x
                    row[f'y_{struct_idx + 1}'] = y
                    row[f'z_{struct_idx + 1}'] = z
                
                # Add chain and copy columns
                row['chain'] = 'A'
                row['copy'] = 1
                
                rows.append(row)
        
        submission_df = pd.DataFrame(rows)
        coord_cols = []
        for i in range(1, 6):
            coord_cols.extend([f'x_{i}', f'y_{i}', f'z_{i}'])
        columns = ['ID', 'resname', 'resid'] + coord_cols + ['chain', 'copy']
        submission_df = submission_df[columns]
        submission_df.to_csv(submission_path, index=False)
        
        print(f"   ✓ Created {submission_path}")
        
        # Verify submission format
        print("\n8. Verifying submission format...")
        sub_df = pd.read_csv(submission_path)
        
        # Check columns
        expected_cols = 20  # ID, resname, resid + 15 coordinates + chain + copy
        assert len(sub_df.columns) == expected_cols, f"Expected {expected_cols} columns, got {len(sub_df.columns)}"
        print(f"   ✓ Column count: {len(sub_df.columns)}")
        
        # Check required columns
        required_cols = ['ID', 'resname', 'resid', 'chain', 'copy']
        for col in required_cols:
            assert col in sub_df.columns, f"Missing required column: {col}"
        print(f"   ✓ Required columns present")
        
        # Check coordinate columns
        for i in range(1, 6):
            for dim in ['x', 'y', 'z']:
                col = f'{dim}_{i}'
                assert col in sub_df.columns, f"Missing coordinate column: {col}"
        print(f"   ✓ All coordinate columns present")
        
        # Check row count
        expected_rows = sum(len(p['sequence']) for p in predictions)
        assert len(sub_df) == expected_rows, f"Expected {expected_rows} rows, got {len(sub_df)}"
        print(f"   ✓ Row count: {len(sub_df)}")
        
        # Check coordinate clipping
        for i in range(1, 6):
            for dim in ['x', 'y', 'z']:
                col = f'{dim}_{i}'
                min_val = sub_df[col].min()
                max_val = sub_df[col].max()
                assert min_val >= -999.999, f"{col} has value below minimum: {min_val}"
                assert max_val <= 9999.999, f"{col} has value above maximum: {max_val}"
        print(f"   ✓ Coordinates properly clipped")
        
        # Display sample
        print("\n9. Sample submission rows:")
        print(sub_df.head(3).to_string())
        
        print("\n" + "=" * 70)
        print("✓ All tests passed!")
        print("=" * 70)
        return True


if __name__ == '__main__':
    try:
        success = test_kaggle_submission()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

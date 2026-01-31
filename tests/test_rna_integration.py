#!/usr/bin/env python3
"""
Integration tests for RNA structure prediction.

This test suite validates the RNA prediction workflow including:
- Dataset building
- Data processing
- Utility functions
"""

import os
import sys
import tempfile
import shutil
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.rna_utils import (
    parse_fasta,
    validate_rna_sequence,
    calculate_rmsd,
    clip_coordinates,
    merge_stoichiometry
)


def test_parse_fasta():
    """Test FASTA parsing."""
    print("Testing parse_fasta...")
    
    fasta = ">Chain_A chain=A\nACGU\n>Chain_B chain=B\nGCAU"
    result = parse_fasta(fasta)
    
    assert 'A' in result, "Should parse chain A"
    assert 'B' in result, "Should parse chain B"
    assert result['A'] == 'ACGU', "Chain A sequence incorrect"
    assert result['B'] == 'GCAU', "Chain B sequence incorrect"
    
    print("✓ parse_fasta test passed")


def test_validate_rna_sequence():
    """Test RNA sequence validation."""
    print("Testing validate_rna_sequence...")
    
    # Valid sequence
    valid, msg = validate_rna_sequence("ACGU")
    assert valid, "Should validate correct RNA sequence"
    
    # Invalid sequence
    valid, msg = validate_rna_sequence("ACGX")
    assert not valid, "Should reject invalid nucleotide"
    
    # Empty sequence
    valid, msg = validate_rna_sequence("")
    assert not valid, "Should reject empty sequence"
    
    print("✓ validate_rna_sequence test passed")


def test_calculate_rmsd():
    """Test RMSD calculation."""
    print("Testing calculate_rmsd...")
    
    coords1 = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    coords2 = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    
    rmsd = calculate_rmsd(coords1, coords2)
    assert np.isclose(rmsd, 0.0), "RMSD of identical coords should be 0"
    
    coords2 = np.array([[1, 0, 0], [2, 1, 1], [3, 2, 2]])
    rmsd = calculate_rmsd(coords1, coords2)
    assert rmsd > 0, "RMSD of different coords should be > 0"
    
    print("✓ calculate_rmsd test passed")


def test_clip_coordinates():
    """Test coordinate clipping."""
    print("Testing clip_coordinates...")
    
    coords = np.array([[10000, 0, -10000], [500, 500, 500]])
    clipped = clip_coordinates(coords)
    
    assert np.isclose(clipped[0, 0], 9999.999), "Should clip to max"
    assert np.isclose(clipped[0, 2], -999.999), "Should clip to min"
    assert clipped[1, 0] == 500, "Should not clip valid values"
    
    print("✓ clip_coordinates test passed")


def test_merge_stoichiometry():
    """Test stoichiometry merging."""
    print("Testing merge_stoichiometry...")
    
    fasta = ">A\nACGU\n>B\nGCAU"
    
    # Single chain
    result = merge_stoichiometry(fasta, "A:1")
    assert result == "ACGU", "Should get chain A once"
    
    # Multiple copies
    result = merge_stoichiometry(fasta, "A:2")
    assert result == "ACGUACGU", "Should get chain A twice"
    
    # Multiple chains
    result = merge_stoichiometry(fasta, "A:1;B:1")
    assert result == "ACGUGCAU", "Should concatenate chains"
    
    print("✓ merge_stoichiometry test passed")


def test_dataset_builder():
    """Test dataset building workflow."""
    print("Testing dataset builder...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test sequences CSV
        sequences_data = {
            'target_id': ['TEST_001', 'TEST_002'],
            'sequence': ['ACGU', 'GCAU'],
            'temporal_cutoff': ['2025-01-01', '2025-02-01'],
            'description': ['Test 1', 'Test 2'],
            'stoichiometry': ['A:1', 'A:1'],
        }
        seq_df = pd.DataFrame(sequences_data)
        seq_file = os.path.join(tmpdir, 'sequences.csv')
        seq_df.to_csv(seq_file, index=False)
        
        # Create test labels CSV
        labels_data = {
            'ID': ['TEST_001_1', 'TEST_001_2', 'TEST_001_3', 'TEST_001_4'],
            'resname': ['A', 'C', 'G', 'U'],
            'resid': [1, 2, 3, 4],
            'x_1': [1.0, 2.0, 3.0, 4.0],
            'y_1': [1.0, 2.0, 3.0, 4.0],
            'z_1': [1.0, 2.0, 3.0, 4.0],
            'chain': ['A', 'A', 'A', 'A'],
            'copy': [1, 1, 1, 1],
        }
        label_df = pd.DataFrame(labels_data)
        label_file = os.path.join(tmpdir, 'labels.csv')
        label_df.to_csv(label_file, index=False)
        
        # Import and run dataset builder
        from dataset.build_rna_dataset import load_rna_data
        
        sequences, labels = load_rna_data(seq_file, label_file, max_length=100)
        
        assert len(sequences) == 2, "Should load 2 sequences"
        assert len(labels) >= 1, "Should load at least 1 label"
        assert sequences[0]['target_id'] == 'TEST_001', "Should have correct target_id"
        assert sequences[0]['sequence'] == 'ACGU', "Should have correct sequence"
        
    print("✓ dataset_builder test passed")


def run_all_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("Running RNA Integration Tests")
    print("=" * 60)
    
    tests = [
        test_parse_fasta,
        test_validate_rna_sequence,
        test_calculate_rmsd,
        test_clip_coordinates,
        test_merge_stoichiometry,
        test_dataset_builder,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

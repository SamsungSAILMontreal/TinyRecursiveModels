#!/usr/bin/env python3
"""
Helper utilities for RNA structure prediction.

This module provides utility functions for:
- Parsing FASTA sequences
- Processing RNA data
- Validating structures
"""

from typing import Dict, List, Tuple
import numpy as np


def parse_fasta(fasta_string: str) -> Dict[str, str]:
    """
    Parse FASTA-formatted string into a dictionary.
    
    This function is compatible with the extra/parse_fasta.py from the
    jrc-rna-structure-pipeline repository.
    
    Args:
        fasta_string: FASTA-formatted string with headers and sequences
        
    Returns:
        Dictionary mapping sequence names to sequences
        
    Example:
        >>> fasta = ">Chain_A\\nACGU\\n>Chain_B\\nGCAU"
        >>> parse_fasta(fasta)
        {'Chain_A': 'ACGU', 'Chain_B': 'GCAU'}
    """
    sequences = {}
    current_name = None
    current_seq = []
    
    for line in fasta_string.strip().split('\n'):
        if line.startswith('>'):
            # Save previous sequence if exists
            if current_name is not None:
                sequences[current_name] = ''.join(current_seq)
            
            # Parse header - extract everything after >
            header = line[1:].strip()
            
            # Extract chain ID if present (format: >... chain=X)
            if 'chain=' in header:
                chain_id = header.split('chain=')[1].split()[0]
                current_name = chain_id
            else:
                # Use first word as name
                current_name = header.split()[0]
            
            current_seq = []
        else:
            current_seq.append(line.strip())
    
    # Save last sequence
    if current_name is not None:
        sequences[current_name] = ''.join(current_seq)
    
    return sequences


def validate_rna_sequence(sequence: str) -> Tuple[bool, str]:
    """
    Validate an RNA sequence.
    
    Args:
        sequence: RNA sequence string
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    valid_nucleotides = set('ACGUN')  # Include N for unknown
    
    if not sequence:
        return False, "Sequence is empty"
    
    invalid_chars = set(sequence.upper()) - valid_nucleotides
    if invalid_chars:
        return False, f"Invalid nucleotides found: {invalid_chars}"
    
    return True, ""


def calculate_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """
    Calculate Root Mean Square Deviation between two sets of coordinates.
    
    Args:
        coords1: First set of coordinates (N, 3)
        coords2: Second set of coordinates (N, 3)
        
    Returns:
        RMSD value
    """
    if coords1.shape != coords2.shape:
        raise ValueError("Coordinate arrays must have the same shape")
    
    diff = coords1 - coords2
    squared_diff = np.sum(diff ** 2, axis=1)
    rmsd = np.sqrt(np.mean(squared_diff))
    
    return rmsd


def clip_coordinates(coords: np.ndarray, min_val: float = -999.999, max_val: float = 9999.999) -> np.ndarray:
    """
    Clip coordinates to valid range for PDB format.
    
    Competition requirements specify that coordinates should be clipped
    to [-999.999, 9999.999] due to legacy PDB format limitations.
    
    Args:
        coords: Coordinate array
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Clipped coordinate array
    """
    return np.clip(coords, min_val, max_val)


def format_pdb_coordinate(value: float) -> str:
    """
    Format a coordinate value for PDB output.
    
    Args:
        value: Coordinate value
        
    Returns:
        Formatted string (8 characters max)
    """
    # Clip to valid range
    value = np.clip(value, -999.999, 9999.999)
    
    # Format with 3 decimal places
    formatted = f"{value:8.3f}"
    
    # Truncate if too long (shouldn't happen with clipping)
    return formatted[:8]


def get_sequence_statistics(sequences: List[str]) -> Dict:
    """
    Calculate statistics for a list of RNA sequences.
    
    Args:
        sequences: List of RNA sequences
        
    Returns:
        Dictionary with statistics
    """
    lengths = [len(seq) for seq in sequences]
    
    # Count nucleotides
    nucleotide_counts = {'A': 0, 'C': 0, 'G': 0, 'U': 0, 'N': 0}
    for seq in sequences:
        for nt in seq.upper():
            if nt in nucleotide_counts:
                nucleotide_counts[nt] += 1
    
    total_nt = sum(nucleotide_counts.values())
    
    return {
        'num_sequences': len(sequences),
        'min_length': min(lengths) if lengths else 0,
        'max_length': max(lengths) if lengths else 0,
        'mean_length': np.mean(lengths) if lengths else 0,
        'median_length': np.median(lengths) if lengths else 0,
        'total_nucleotides': total_nt,
        'nucleotide_counts': nucleotide_counts,
        'nucleotide_frequencies': {
            nt: count / total_nt if total_nt > 0 else 0
            for nt, count in nucleotide_counts.items()
        }
    }


def merge_stoichiometry(all_sequences: str, stoichiometry: str) -> str:
    """
    Merge sequences according to stoichiometry specification.
    
    Args:
        all_sequences: FASTA-formatted string with all chains
        stoichiometry: Stoichiometry specification (e.g., "A:1;B:2")
        
    Returns:
        Concatenated sequence according to stoichiometry
    """
    # Parse FASTA
    sequences = parse_fasta(all_sequences)
    
    # Parse stoichiometry
    result_parts = []
    
    for spec in stoichiometry.split(';'):
        if ':' in spec:
            chain, count = spec.split(':')
            chain = chain.strip()
            count = int(count.strip())
            
            if chain in sequences:
                # Add the sequence 'count' times
                for _ in range(count):
                    result_parts.append(sequences[chain])
    
    return ''.join(result_parts)


if __name__ == '__main__':
    # Test functions
    print("Testing RNA utilities...")
    
    # Test parse_fasta
    test_fasta = ">Chain_A chain=A\nACGU\n>Chain_B chain=B\nGCAU"
    parsed = parse_fasta(test_fasta)
    print(f"Parsed FASTA: {parsed}")
    assert parsed == {'A': 'ACGU', 'B': 'GCAU'}, "FASTA parsing failed"
    
    # Test validate_rna_sequence
    valid, msg = validate_rna_sequence("ACGU")
    print(f"Valid sequence: {valid}")
    assert valid, "Sequence validation failed"
    
    valid, msg = validate_rna_sequence("ACGX")
    print(f"Invalid sequence detected: {not valid}")
    assert not valid, "Should detect invalid sequence"
    
    # Test calculate_rmsd
    coords1 = np.array([[0, 0, 0], [1, 1, 1]])
    coords2 = np.array([[0, 0, 0], [1, 1, 1]])
    rmsd = calculate_rmsd(coords1, coords2)
    print(f"RMSD (identical): {rmsd}")
    assert rmsd == 0, "RMSD of identical coordinates should be 0"
    
    # Test clip_coordinates
    coords = np.array([[10000, 0, -1000], [500, 500, 500]])
    clipped = clip_coordinates(coords)
    print(f"Clipped coordinates: {clipped}")
    assert np.isclose(clipped[0, 0], 9999.999), "Should clip to max"
    assert np.isclose(clipped[0, 2], -999.999), "Should clip to min"
    
    # Test statistics
    seqs = ["ACGU", "GCAU", "AAACCCGGGUUU"]
    stats = get_sequence_statistics(seqs)
    print(f"Sequence statistics: {stats}")
    
    # Test merge_stoichiometry
    merged = merge_stoichiometry(test_fasta, "A:1;B:2")
    print(f"Merged stoichiometry: {merged}")
    assert merged == "ACGUGCAUGCAU", "Stoichiometry merge failed"
    
    print("\nAll tests passed!")

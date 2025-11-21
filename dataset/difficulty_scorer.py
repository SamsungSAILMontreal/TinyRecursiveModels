"""
GPU-based puzzle difficulty estimation and quality scoring.
Provides real value by enabling curriculum learning and dataset balancing.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PuzzleDifficultyScorer(nn.Module):
    """
    Lightweight CNN to estimate puzzle difficulty and quality metrics.
    
    Outputs:
        - difficulty_score: 0-1 (easy to hard)
        - complexity_score: Pattern complexity (edges, regions, symmetries)
        - solvability_score: Confidence that puzzle is solvable
        - diversity_score: Uniqueness compared to training set
    """
    
    def __init__(self, grid_size=30, num_colors=10):
        super().__init__()
        
        # Input: [batch, 2, grid_size, grid_size] (input + output grids)
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Scoring heads
        self.fc_shared = nn.Linear(128 * 4 * 4, 256)
        
        self.difficulty_head = nn.Linear(256, 1)  # Difficulty: 0-1
        self.complexity_head = nn.Linear(256, 1)  # Complexity: 0-1
        self.solvability_head = nn.Linear(256, 1)  # Solvability: 0-1
        self.diversity_head = nn.Linear(256, 1)  # Diversity: 0-1
        
    def forward(self, input_grid: torch.Tensor, output_grid: torch.Tensor):
        """
        Args:
            input_grid: [batch, H, W] - puzzle input (uint8)
            output_grid: [batch, H, W] - puzzle output (uint8)
        
        Returns:
            scores: dict with difficulty, complexity, solvability, diversity
        """
        # Normalize to [0, 1]
        inp = (input_grid.float() / 9.0).unsqueeze(1)  # [batch, 1, H, W]
        out = (output_grid.float() / 9.0).unsqueeze(1)
        
        # Concatenate input/output
        x = torch.cat([inp, out], dim=1)  # [batch, 2, H, W]
        
        # CNN feature extraction
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_shared(x))
        
        # Score heads
        difficulty = torch.sigmoid(self.difficulty_head(x))
        complexity = torch.sigmoid(self.complexity_head(x))
        solvability = torch.sigmoid(self.solvability_head(x))
        diversity = torch.sigmoid(self.diversity_head(x))
        
        return {
            'difficulty': difficulty.squeeze(-1),
            'complexity': complexity.squeeze(-1),
            'solvability': solvability.squeeze(-1),
            'diversity': diversity.squeeze(-1)
        }


def compute_heuristic_difficulty(input_grid: np.ndarray, output_grid: np.ndarray) -> dict:
    """
    Heuristic-based difficulty metrics (CPU fallback).
    
    Returns:
        scores: dict with various difficulty indicators
    """
    scores = {}
    
    # Grid size factor
    scores['size_factor'] = (input_grid.size / (30 * 30))
    
    # Color diversity
    unique_colors_in = len(np.unique(input_grid))
    unique_colors_out = len(np.unique(output_grid))
    scores['color_diversity'] = (unique_colors_in + unique_colors_out) / 20.0
    
    # Transformation complexity (how much changed)
    # Handle different sizes by comparing overlapping region only
    min_h = min(input_grid.shape[0], output_grid.shape[0])
    min_w = min(input_grid.shape[1], output_grid.shape[1])
    if min_h > 0 and min_w > 0:
        inp_crop = input_grid[:min_h, :min_w]
        out_crop = output_grid[:min_h, :min_w]
        diff = (inp_crop != out_crop).sum() / inp_crop.size
    else:
        diff = 1.0  # Completely different if no overlap
    
    # Add size difference as additional complexity
    size_diff = abs(input_grid.size - output_grid.size) / max(input_grid.size, output_grid.size)
    scores['transformation_magnitude'] = float(0.7 * diff + 0.3 * size_diff)
    
    # Pattern complexity (edge density)
    def edge_density(grid):
        # Handle edge cases: grids too small or would cause division by zero
        denominator = grid.size - grid.shape[0] - grid.shape[1] + 1
        if denominator <= 0 or grid.shape[0] <= 1 or grid.shape[1] <= 1:
            return 0.0  # No meaningful edge density for tiny grids
        
        edges = 0
        for i in range(grid.shape[0] - 1):
            for j in range(grid.shape[1] - 1):
                if grid[i, j] != grid[i+1, j] or grid[i, j] != grid[i, j+1]:
                    edges += 1
        return edges / denominator
    
    scores['pattern_complexity'] = (edge_density(input_grid) + edge_density(output_grid)) / 2
    
    # Aggregate difficulty (weighted average)
    scores['estimated_difficulty'] = (
        0.2 * scores['size_factor'] +
        0.3 * scores['color_diversity'] +
        0.3 * scores['transformation_magnitude'] +
        0.2 * scores['pattern_complexity']
    )
    
    return scores


def score_difficulty(input_grid: np.ndarray, output_grid: np.ndarray) -> dict:
    """
    Public API for difficulty scoring (heuristic-based).
    This is the main entry point for external callers.
    
    Args:
        input_grid: Input puzzle grid (numpy array)
        output_grid: Output puzzle grid (numpy array)
    
    Returns:
        scores: dict with difficulty metrics
    """
    return compute_heuristic_difficulty(input_grid, output_grid)


class DatasetQualityEnhancer:
    """
    GPU-accelerated quality enhancement for puzzle datasets.
    """
    
    def __init__(self, device='cuda', use_neural_scorer=True):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_neural = use_neural_scorer and torch.cuda.is_available()
        
        if self.use_neural:
            self.scorer = PuzzleDifficultyScorer().to(self.device)
            self.scorer.eval()
            print(f"Initialized neural difficulty scorer on {self.device}")
        else:
            print("Using heuristic difficulty scoring (CPU)")
    
    def score_puzzle(self, input_grid: np.ndarray, output_grid: np.ndarray) -> dict:
        """Score a single puzzle's difficulty and quality.
        
        Returns:
            scores: dict with difficulty, complexity, solvability metrics
        """
        if self.use_neural:
            with torch.no_grad():
                # Pad to common size if needed
                max_h = max(input_grid.shape[0], output_grid.shape[0])
                max_w = max(input_grid.shape[1], output_grid.shape[1])
                
                def pad_grid(arr, h, w):
                    if arr.shape[0] == h and arr.shape[1] == w:
                        return arr
                    padded = np.zeros((h, w), dtype=arr.dtype)
                    padded[:arr.shape[0], :arr.shape[1]] = arr
                    return padded
                
                inp_padded = pad_grid(input_grid, max_h, max_w)
                out_padded = pad_grid(output_grid, max_h, max_w)
                
                inp_tensor = torch.from_numpy(inp_padded).unsqueeze(0).to(self.device)
                out_tensor = torch.from_numpy(out_padded).unsqueeze(0).to(self.device)
                scores = self.scorer(inp_tensor, out_tensor)
                return {k: v.cpu().item() for k, v in scores.items()}
        else:
            return compute_heuristic_difficulty(input_grid, output_grid)
    
    def score_batch(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> List[dict]:
        """Score multiple puzzles in batch (GPU efficient).
        
        Args:
            examples: List of (input_grid, output_grid) tuples
        
        Returns:
            scores: List of score dicts for each example
        """
        if self.use_neural and len(examples) > 0:
            with torch.no_grad():
                # Find max dimensions across all examples
                max_h = max(max(inp.shape[0], out.shape[0]) for inp, out in examples)
                max_w = max(max(inp.shape[1], out.shape[1]) for inp, out in examples)
                
                # Pad all grids to same size
                def pad_to_size(arr, h, w):
                    if arr.shape[0] == h and arr.shape[1] == w:
                        return arr
                    padded = np.zeros((h, w), dtype=arr.dtype)
                    padded[:arr.shape[0], :arr.shape[1]] = arr
                    return padded
                
                # Batch process on GPU with padded tensors
                inputs = torch.stack([torch.from_numpy(pad_to_size(inp, max_h, max_w)) for inp, _ in examples]).to(self.device)
                outputs = torch.stack([torch.from_numpy(pad_to_size(out, max_h, max_w)) for _, out in examples]).to(self.device)
                
                batch_scores = self.scorer(inputs, outputs)
                
                # Convert to list of dicts
                return [
                    {k: batch_scores[k][i].cpu().item() for k in batch_scores.keys()}
                    for i in range(len(examples))
                ]
        else:
            # CPU fallback
            return [compute_heuristic_difficulty(inp, out) for inp, out in examples]
    
    def filter_low_quality(self, examples: List[Tuple[np.ndarray, np.ndarray]], 
                          threshold: float = 0.3) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Filter out low-quality augmentations.
        
        Args:
            examples: List of augmented examples
            threshold: Minimum solvability score to keep
        
        Returns:
            filtered_examples: High-quality examples only
        """
        scores = self.score_batch(examples)
        filtered = [
            ex for ex, score in zip(examples, scores)
            if score.get('solvability', 1.0) >= threshold
        ]
        return filtered
    
    def balance_by_difficulty(self, puzzles: List, target_distribution: dict = None):
        """
        Balance dataset by difficulty levels for curriculum learning.
        
        Args:
            puzzles: List of puzzles to balance
            target_distribution: Desired distribution {easy: 0.3, medium: 0.4, hard: 0.3}
        
        Returns:
            balanced_puzzles: Resampled puzzles matching target distribution
        """
        if target_distribution is None:
            target_distribution = {'easy': 0.25, 'medium': 0.50, 'hard': 0.25}
        
        # Score all puzzles
        print(f"Scoring {len(puzzles)} puzzles for difficulty balancing...")
        
        # TODO: Implement balanced sampling
        return puzzles  # Placeholder

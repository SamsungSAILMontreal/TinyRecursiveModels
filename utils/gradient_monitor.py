"""
Gradient flow monitoring utilities for debugging and optimization.
Tracks gradient norms, dead neurons, and gradient vanishing/explosion.
"""

import torch
from torch import nn
from typing import Dict, List, Optional
import numpy as np


class GradientFlowMonitor:
    """
    Monitor gradient flow through the network during training.
    
    Tracks:
    - Gradient norms per layer
    - Percentage of dead neurons (zero gradients)
    - Gradient variance
    - Layer-wise gradient statistics
    """
    
    def __init__(self, model: nn.Module, track_every_n_steps: int = 100):
        """
        Initialize gradient flow monitor.
        
        Args:
            model: PyTorch model to monitor
            track_every_n_steps: Frequency of gradient tracking
        """
        self.model = model
        self.track_every_n_steps = track_every_n_steps
        self.step_counter = 0
        
        # Statistics storage (with memory management)
        self.gradient_norms = []
        self.dead_neuron_ratios = []
        self.layer_stats = {}
        self.max_history_size = 1000  # Limit history to prevent memory leaks
        
    def reset(self):
        """Reset all statistics."""
        self.gradient_norms.clear()
        self.dead_neuron_ratios.clear()
        self.layer_stats.clear()
        self.step_counter = 0
    
    def clear_old_data(self, keep_last_n: int = 100):
        """Clear old gradient history, keeping only recent data."""
        if len(self.gradient_norms) > keep_last_n:
            self.gradient_norms = self.gradient_norms[-keep_last_n:]
            self.dead_neuron_ratios = self.dead_neuron_ratios[-keep_last_n:]
        self.layer_stats.clear()  # Layer stats not needed after aggregation
    
    def should_track(self) -> bool:
        """Check if current step should be tracked."""
        return self.step_counter % self.track_every_n_steps == 0
    
    def compute_layer_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Compute gradient statistics for each layer.
        
        Returns:
            Dictionary mapping layer name to gradient statistics
        """
        stats = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                
                # Compute statistics
                grad_norm = grad.norm().item()
                grad_mean = grad.mean().item()
                grad_std = grad.std().item()
                grad_max = grad.abs().max().item()
                
                # Dead neurons (gradient magnitude < 1e-8)
                dead_mask = grad.abs() < 1e-8
                dead_ratio = dead_mask.float().mean().item()
                
                stats[name] = {
                    'grad_norm': grad_norm,
                    'grad_mean': grad_mean,
                    'grad_std': grad_std,
                    'grad_max': grad_max,
                    'dead_ratio': dead_ratio,
                    'param_norm': param.data.norm().item(),
                    'grad_to_param_ratio': grad_norm / (param.data.norm().item() + 1e-8)
                }
        
        return stats
    
    def update(self) -> Optional[Dict[str, float]]:
        """
        Update gradient flow statistics.
        
        Returns:
            Aggregated metrics if tracking this step, None otherwise
        """
        self.step_counter += 1
        
        if not self.should_track():
            return None
        
        # Compute layer-wise statistics
        layer_stats = self.compute_layer_stats()
        
        if not layer_stats:
            return None
        
        # Aggregate statistics
        all_grad_norms = [s['grad_norm'] for s in layer_stats.values()]
        all_dead_ratios = [s['dead_ratio'] for s in layer_stats.values()]
        
        aggregate_stats = {
            'grad_norm_mean': np.mean(all_grad_norms),
            'grad_norm_std': np.std(all_grad_norms),
            'grad_norm_max': np.max(all_grad_norms),
            'grad_norm_min': np.min(all_grad_norms),
            'dead_neuron_ratio_mean': np.mean(all_dead_ratios),
            'dead_neuron_ratio_max': np.max(all_dead_ratios),
        }
        
        # Store for history (with automatic cleanup)
        self.gradient_norms.append(all_grad_norms)
        self.dead_neuron_ratios.append(all_dead_ratios)
        
        # Prevent memory leak: keep only recent history
        if len(self.gradient_norms) > self.max_history_size:
            self.gradient_norms = self.gradient_norms[-self.max_history_size:]
            self.dead_neuron_ratios = self.dead_neuron_ratios[-self.max_history_size:]
        
        # Component-specific tracking (H-level, L-level, Q-head, Memory)
        component_stats = self._aggregate_by_component(layer_stats)
        aggregate_stats.update(component_stats)
        
        return aggregate_stats
    
    def _aggregate_by_component(self, layer_stats: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Aggregate gradient stats by model component."""
        components = {
            'H_level': [],
            'L_level': [],
            'q_head': [],
            'memory': [],
            'embed': [],
            'lm_head': []
        }
        
        for name, stats in layer_stats.items():
            grad_norm = stats['grad_norm']
            
            if 'H_level' in name or 'H_init' in name:
                components['H_level'].append(grad_norm)
            elif 'L_level' in name or 'L_init' in name:
                components['L_level'].append(grad_norm)
            elif 'q_head' in name:
                components['q_head'].append(grad_norm)
            elif 'memory' in name:
                components['memory'].append(grad_norm)
            elif 'embed' in name:
                components['embed'].append(grad_norm)
            elif 'lm_head' in name:
                components['lm_head'].append(grad_norm)
        
        # Compute mean for each component
        result = {}
        for comp_name, grad_norms in components.items():
            if grad_norms:
                result[f'grad_{comp_name}_mean'] = float(np.mean(grad_norms))
                result[f'grad_{comp_name}_max'] = float(np.max(grad_norms))
        
        return result
    
    def diagnose_gradient_flow(self) -> Dict[str, str]:
        """
        Diagnose potential gradient flow issues.
        
        Returns:
            Dictionary of issue type to description
        """
        issues = {}
        
        if not self.gradient_norms:
            return {'warning': 'No gradient statistics collected yet'}
        
        recent_norms = self.gradient_norms[-1]
        recent_dead = self.dead_neuron_ratios[-1]
        
        # Check for vanishing gradients
        if np.mean(recent_norms) < 1e-5:
            issues['vanishing_gradients'] = f'Mean gradient norm very small: {np.mean(recent_norms):.2e}'
        
        # Check for exploding gradients
        if np.max(recent_norms) > 100:
            issues['exploding_gradients'] = f'Max gradient norm very large: {np.max(recent_norms):.2e}'
        
        # Check for dead neurons
        if np.mean(recent_dead) > 0.5:
            issues['dead_neurons'] = f'High proportion of dead neurons: {np.mean(recent_dead)*100:.1f}%'
        
        # Check gradient variance
        if len(self.gradient_norms) > 10:
            recent_variance = np.var([np.mean(norms) for norms in self.gradient_norms[-10:]])
            if recent_variance > 100:
                issues['unstable_training'] = f'High gradient variance: {recent_variance:.2e}'
        
        return issues if issues else {'status': 'Gradient flow appears healthy'}
    
    def get_summary_report(self) -> str:
        """Generate human-readable summary report."""
        if not self.gradient_norms:
            return "No gradient statistics collected yet."
        
        recent_norms = self.gradient_norms[-1]
        recent_dead = self.dead_neuron_ratios[-1]
        
        report = "=== Gradient Flow Summary ===\n"
        report += f"Steps tracked: {len(self.gradient_norms)}\n"
        report += f"Mean gradient norm: {np.mean(recent_norms):.4e}\n"
        report += f"Max gradient norm: {np.max(recent_norms):.4e}\n"
        report += f"Min gradient norm: {np.min(recent_norms):.4e}\n"
        report += f"Dead neuron ratio: {np.mean(recent_dead)*100:.2f}%\n"
        report += "\n"
        
        # Diagnose issues
        issues = self.diagnose_gradient_flow()
        report += "=== Diagnostics ===\n"
        for issue_type, description in issues.items():
            report += f"- {issue_type}: {description}\n"
        
        return report


def analyze_gradient_contributions(model: nn.Module, loss: torch.Tensor) -> Dict[str, float]:
    """
    Analyze gradient contributions from different loss components.
    
    Useful for understanding which loss terms contribute most to learning.
    
    Args:
        model: PyTorch model
        loss: Total loss (should be sum of multiple components)
    
    Returns:
        Dictionary mapping component to gradient magnitude contribution
    """
    # This requires loss components to be tracked separately
    # Implementation would involve computing gradients w.r.t. each loss component
    pass


def check_gradient_sparsity(model: nn.Module, threshold: float = 1e-6) -> Dict[str, float]:
    """
    Check gradient sparsity (percentage of near-zero gradients).
    
    Args:
        model: PyTorch model
        threshold: Threshold below which gradients are considered zero
    
    Returns:
        Dictionary mapping layer name to sparsity ratio
    """
    sparsity_ratios = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            sparse_mask = grad.abs() < threshold
            sparsity_ratio = sparse_mask.float().mean().item()
            sparsity_ratios[name] = sparsity_ratio
    
    return sparsity_ratios

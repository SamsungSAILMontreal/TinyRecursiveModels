"""
SNN-based Parallel Process Coordinator
Uses spiking neural networks for efficient, event-driven task scheduling and load balancing.
"""

from typing import List, Dict, Tuple
import torch
import torch.nn as nn
import numpy as np
from queue import Queue
import time


class LIFSchedulerNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron for task scheduling."""
    
    def __init__(self, num_workers: int, tau: float = 0.9, threshold: float = 1.0):
        super().__init__()
        self.num_workers = num_workers
        self.tau = tau
        self.threshold = threshold
        
        # Worker state potentials (one per worker)
        self.register_buffer('v_mem', torch.zeros(num_workers))
        self.register_buffer('worker_load', torch.zeros(num_workers))
        self.register_buffer('last_spike', torch.zeros(num_workers))
        
    def reset(self):
        """Reset all neuron states."""
        self.v_mem.zero_()
        self.worker_load.zero_()
        self.last_spike.zero_()
    
    def forward(self, task_priority: float, task_complexity: float, timestep: int) -> int:
        """
        Select best worker using spiking dynamics.
        
        Args:
            task_priority: 0-1, urgency of task
            task_complexity: 0-1, estimated compute time
            timestep: Current time step
        
        Returns:
            worker_id: Index of selected worker
        """
        # Input current proportional to task priority
        input_current = task_priority * (1.0 + task_complexity)
        
        # Leak: workers with lower load have higher potential
        self.v_mem = self.tau * self.v_mem + (1 - self.worker_load / (self.worker_load.max() + 1e-6))
        
        # Add input weighted by inverse of load (prefer free workers)
        load_weight = 1.0 / (self.worker_load + 0.1)
        self.v_mem += input_current * load_weight
        
        # Time decay: prefer workers that haven't worked recently
        time_since_spike = timestep - self.last_spike
        self.v_mem += 0.1 * torch.log1p(time_since_spike)
        
        # Select worker with highest potential (first to spike)
        worker_id = self.v_mem.argmax().item()
        
        # Update selected worker
        self.v_mem[worker_id] = 0.0  # Reset after spike
        self.worker_load[worker_id] += task_complexity
        self.last_spike[worker_id] = timestep
        
        return worker_id
    
    def complete_task(self, worker_id: int, actual_time: float):
        """Update worker state after task completion."""
        self.worker_load[worker_id] = max(0, self.worker_load[worker_id] - actual_time)


class SNNParallelCoordinator:
    """
    Event-driven parallel coordinator using SNN for load balancing.
    Prevents GPU conflicts and optimizes CPU-GPU handoffs.
    """
    
    def __init__(self, num_workers: int, use_gpu: bool = False, device='cpu'):
        self.num_workers = num_workers
        self.use_gpu = use_gpu
        self.device = device
        
        # SNN scheduler
        self.scheduler = LIFSchedulerNeuron(num_workers).to(device)
        
        # GPU access control (event-driven)
        self.gpu_available = True
        self.gpu_queue = Queue()
        
        # Task tracking
        self.timestep = 0
        self.completed_tasks = 0
        self.total_time = 0.0
        
        # Performance metrics
        self.worker_utilization = np.zeros(num_workers)
        self.worker_times = np.zeros(num_workers)
        
    def assign_task(self, task_metadata: Dict) -> int:
        """
        Assign task to best worker using SNN scheduling.
        
        Args:
            task_metadata: Dict with 'difficulty', 'complexity', 'priority'
        
        Returns:
            worker_id: Assigned worker
        """
        self.timestep += 1
        
        priority = task_metadata.get('priority', 0.5)
        complexity = task_metadata.get('complexity', 0.5)
        
        # Use SNN to select worker
        worker_id = self.scheduler(
            task_priority=priority,
            task_complexity=complexity,
            timestep=self.timestep
        )
        
        return worker_id
    
    def request_gpu(self, worker_id: int, timeout: float = 1.0) -> bool:
        """
        Request GPU access (event-driven, prevents conflicts).
        
        Args:
            worker_id: Worker requesting GPU
            timeout: Max wait time in seconds
        
        Returns:
            granted: True if GPU access granted
        """
        if not self.use_gpu:
            return False
        
        start_time = time.time()
        
        # Wait in queue
        self.gpu_queue.put(worker_id)
        
        # Spin until granted or timeout
        while time.time() - start_time < timeout:
            if self.gpu_available and self.gpu_queue.queue[0] == worker_id:
                self.gpu_available = False
                self.gpu_queue.get()
                return True
            time.sleep(0.001)  # 1ms polling (very low overhead)
        
        # Timeout - remove from queue
        if worker_id in list(self.gpu_queue.queue):
            temp_queue = []
            while not self.gpu_queue.empty():
                item = self.gpu_queue.get()
                if item != worker_id:
                    temp_queue.append(item)
            for item in temp_queue:
                self.gpu_queue.put(item)
        
        return False
    
    def release_gpu(self, worker_id: int):
        """Release GPU for next worker."""
        self.gpu_available = True
    
    def complete_task(self, worker_id: int, actual_time: float):
        """Update scheduler after task completion."""
        self.scheduler.complete_task(worker_id, actual_time)
        self.completed_tasks += 1
        self.total_time += actual_time
        self.worker_utilization[worker_id] += actual_time
        self.worker_times[worker_id] += 1
    
    def get_stats(self) -> Dict:
        """Get coordination statistics."""
        return {
            'completed_tasks': self.completed_tasks,
            'avg_time': self.total_time / max(1, self.completed_tasks),
            'worker_utilization': self.worker_utilization.tolist(),
            'load_balance': float(self.worker_utilization.std() / (self.worker_utilization.mean() + 1e-6))
        }
    
    def reset(self):
        """Reset coordinator state."""
        self.scheduler.reset()
        self.timestep = 0
        self.completed_tasks = 0
        self.total_time = 0.0
        self.worker_utilization = np.zeros(self.num_workers)
        self.worker_times = np.zeros(self.num_workers)


def create_adaptive_work_schedule(puzzles: List, difficulty_scores: List[Dict], 
                                  num_workers: int) -> List[List]:
    """
    Create optimally balanced work schedule using SNN predictions.
    
    Args:
        puzzles: List of puzzles to process
        difficulty_scores: Difficulty metadata for each puzzle
        num_workers: Number of parallel workers
    
    Returns:
        work_schedule: List of puzzle batches for each worker
    """
    coordinator = SNNParallelCoordinator(num_workers)
    
    # Assign puzzles to workers using SNN
    worker_assignments = [[] for _ in range(num_workers)]
    
    for idx, puzzle in enumerate(puzzles):
        if idx < len(difficulty_scores):
            metadata = difficulty_scores[idx]
        else:
            metadata = {'difficulty': 0.5, 'complexity': 0.5, 'priority': 0.5}
        
        worker_id = coordinator.assign_task(metadata)
        worker_assignments[worker_id].append(puzzle)
    
    stats = coordinator.get_stats()
    print(f"SNN Scheduler: Load balance score: {stats['load_balance']:.3f} (lower is better)")
    
    return worker_assignments

"""
Annealing Schedule Utilities

Provides adaptive annealing schedules that scale with total training steps.
All schedules use ratios (0.0-1.0) instead of hardcoded step counts.
"""

import math
from typing import Literal


def compute_annealing_schedule(
    current_step: int,
    total_steps: int,
    start_value: float,
    end_value: float,
    anneal_ratio: float = 1.0,
    schedule: Literal["linear", "cosine", "exponential"] = "cosine"
) -> float:
    """
    Compute annealed value based on current progress through training.
    
    Args:
        current_step: Current training step
        total_steps: Total training steps
        start_value: Initial value (at step 0)
        end_value: Final value (at anneal_steps)
        anneal_ratio: Fraction of training to anneal over (0.0-1.0)
        schedule: Annealing schedule type
    
    Returns:
        Annealed value at current_step
    
    Examples:
        # Anneal over first 50% of training
        penalty = compute_annealing_schedule(
            current_step=1000,
            total_steps=10000,
            start_value=0.1,
            end_value=0.001,
            anneal_ratio=0.5,
            schedule="cosine"
        )
    """
    # Calculate annealing duration
    anneal_steps = int(total_steps * anneal_ratio)
    
    # If past annealing period, return end value
    if current_step >= anneal_steps:
        return end_value
    
    # Calculate progress (0.0 to 1.0)
    progress = current_step / max(1, anneal_steps)
    
    # Apply schedule
    if schedule == "linear":
        factor = progress
    elif schedule == "cosine":
        # Cosine annealing: smooth transition
        factor = 0.5 * (1.0 - math.cos(math.pi * progress))
    elif schedule == "exponential":
        # Exponential decay: rapid early change, slow later
        factor = 1.0 - math.exp(-5.0 * progress)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    
    # Interpolate between start and end
    return start_value + (end_value - start_value) * factor


def compute_expansion_penalty(
    current_step: int,
    total_steps: int,
    config
) -> float:
    """
    Compute expansion penalty at current step.
    
    High penalty early (discourage expansion) -> Low penalty late (allow expansion)
    """
    if not hasattr(config, 'expansion_penalty_schedule'):
        return config.expansion_penalty_start
    
    # Use ratio if available, otherwise fall back to steps
    if hasattr(config, 'expansion_anneal_ratio'):
        anneal_ratio = config.expansion_anneal_ratio
    else:
        # Fallback: convert steps to ratio
        anneal_ratio = min(1.0, config.expansion_anneal_steps / max(1, total_steps))
    
    return compute_annealing_schedule(
        current_step=current_step,
        total_steps=total_steps,
        start_value=config.expansion_penalty_start,
        end_value=config.expansion_penalty_end,
        anneal_ratio=anneal_ratio,
        schedule=config.expansion_penalty_schedule
    )


def compute_q_temperature(
    current_step: int,
    total_steps: int,
    config
) -> float:
    """
    Compute Q-head temperature at current step.
    
    High temp early (exploration) -> Low temp late (exploitation)
    """
    if not hasattr(config, 'enable_q_temperature_annealing') or not config.enable_q_temperature_annealing:
        return 1.0  # Default temperature
    
    # Use ratio if available
    if hasattr(config, 'q_temperature_anneal_ratio'):
        anneal_ratio = config.q_temperature_anneal_ratio
    else:
        # Fallback: assume full training
        anneal_ratio = 1.0
    
    return compute_annealing_schedule(
        current_step=current_step,
        total_steps=total_steps,
        start_value=config.q_temperature_start,
        end_value=config.q_temperature_end,
        anneal_ratio=anneal_ratio,
        schedule=config.q_temperature_schedule
    )


def compute_warmup_progress(
    current_step: int,
    warmup_steps: int
) -> float:
    """
    Compute linear warmup progress (0.0 to 1.0).
    
    Args:
        current_step: Current training step
        warmup_steps: Number of warmup steps
    
    Returns:
        Warmup factor (0.0 at start, 1.0 at warmup_steps)
    """
    if current_step >= warmup_steps:
        return 1.0
    return current_step / max(1, warmup_steps)


# Example usage and testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Test schedules
    total_steps = 10000
    steps = np.arange(0, total_steps, 10)
    
    # Expansion penalty (anneal over first 50%)
    expansion_cosine = [
        compute_annealing_schedule(s, total_steps, 0.1, 0.001, 0.5, "cosine")
        for s in steps
    ]
    expansion_linear = [
        compute_annealing_schedule(s, total_steps, 0.1, 0.001, 0.5, "linear")
        for s in steps
    ]
    
    # Q-temperature (anneal over full training)
    temp_exponential = [
        compute_annealing_schedule(s, total_steps, 1.0, 0.1, 1.0, "exponential")
        for s in steps
    ]
    temp_cosine = [
        compute_annealing_schedule(s, total_steps, 1.0, 0.1, 1.0, "cosine")
        for s in steps
    ]
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(steps, expansion_cosine, label='Cosine')
    ax1.plot(steps, expansion_linear, label='Linear')
    ax1.axvline(total_steps * 0.5, color='red', linestyle='--', alpha=0.5, label='Anneal end')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Expansion Penalty')
    ax1.set_title('Expansion Penalty Annealing (50% of training)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.plot(steps, temp_exponential, label='Exponential')
    ax2.plot(steps, temp_cosine, label='Cosine')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Q-Temperature')
    ax2.set_title('Q-Temperature Annealing (100% of training)')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('annealing_schedules.png', dpi=150, bbox_inches='tight')
    print("✅ Saved annealing_schedules.png")

#!/usr/bin/env python3
"""
Run TRM agent for ARC-AGI-3 as an experiment.

This script sets up and runs the TRM (Tiny Recursive Model) agent
on ARC-AGI-3 games, demonstrating the integration of recursive reasoning
with the ARC-AGI-3 platform.

Usage:
    python experiments/run_trm_arc_agi_3.py --game=ls20
    python experiments/run_trm_arc_agi_3.py --game=all
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents import TRMAgent, FrameData, GameState


def run_experiment(game_id: str = "all", max_steps: int = 100, recursion_cycles: int = 3):
    """Run the TRM agent experiment.
    
    Args:
        game_id: The game ID to play (or "all" for all games)
        max_steps: Maximum number of steps per game
        recursion_cycles: Number of recursive reasoning cycles
    """
    print("=" * 80)
    print("TRM Agent for ARC-AGI-3 Experiment")
    print("=" * 80)
    print(f"Game ID: {game_id}")
    print(f"Max Steps: {max_steps}")
    print(f"Recursion Cycles: {recursion_cycles}")
    print("=" * 80)
    
    # Create TRM agent
    agent = TRMAgent(max_steps=max_steps, recursion_cycles=recursion_cycles)
    
    print(f"\nInitialized TRM Agent with {recursion_cycles} recursion cycles")
    print(f"Agent type: {type(agent).__name__}")
    
    # Simulate a game run (placeholder)
    print("\n" + "-" * 80)
    print("Running agent simulation...")
    print("-" * 80)
    
    frames = []
    current_frame = FrameData(state=GameState.NOT_PLAYED)
    
    step = 0
    max_sim_steps = min(10, max_steps)  # Limit simulation to 10 steps for demo
    
    while not agent.is_done(frames, current_frame) and step < max_sim_steps:
        print(f"\nStep {step + 1}:")
        
        # Get action from agent
        action = agent.choose_action(frames, current_frame)
        print(f"  Action: {action.value}")
        
        if hasattr(action, 'reasoning') and action.reasoning:
            print(f"  Reasoning: {action.reasoning}")
        
        # Simulate frame update
        frames.append(current_frame)
        current_frame = FrameData(
            state=GameState.PLAYING,
            grid=[[0] * 8 for _ in range(8)],  # Placeholder grid
            score=step + 1
        )
        
        step += 1
    
    print("\n" + "-" * 80)
    print(f"Simulation complete after {step} steps")
    print("-" * 80)
    
    # Print summary
    print("\nExperiment Summary:")
    print(f"  Total frames: {len(frames)}")
    print(f"  Final state: {current_frame.state.value}")
    print(f"  Final score: {current_frame.score}")
    
    return agent


def main():
    """Main entry point for the experiment."""
    parser = argparse.ArgumentParser(
        description="Run TRM agent for ARC-AGI-3 experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on a specific game
  python experiments/run_trm_arc_agi_3.py --game=ls20
  
  # Run on all games
  python experiments/run_trm_arc_agi_3.py --game=all
  
  # Run with custom parameters
  python experiments/run_trm_arc_agi_3.py --game=ls20 --max-steps=200 --cycles=5
        """
    )
    
    parser.add_argument(
        "--game",
        type=str,
        default="all",
        help="Game ID to play (default: all)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum steps per game (default: 100)"
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=3,
        help="Number of recursive reasoning cycles (default: 3)"
    )
    
    args = parser.parse_args()
    
    # Run the experiment
    agent = run_experiment(
        game_id=args.game,
        max_steps=args.max_steps,
        recursion_cycles=args.cycles
    )
    
    print("\n" + "=" * 80)
    print("Experiment completed successfully!")
    print("=" * 80)
    print("\nNote: This is a demonstration of the TRM agent infrastructure.")
    print("For full integration with ARC-AGI-3, install dependencies from:")
    print("  https://github.com/arcprize/ARC-AGI-3-Agents")
    print("\nTo run with the full ARC-AGI-3 API:")
    print("  python main.py --agent=trmagent --game=ls20")


if __name__ == "__main__":
    main()

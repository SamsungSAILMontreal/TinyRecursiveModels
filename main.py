#!/usr/bin/env python3
"""Main entry point for running ARC-AGI-3 game agents."""

import argparse
import os
from agents import AVAILABLE_AGENTS, FrameData, GameState


def main():
    """Run an agent on ARC-AGI-3 games."""
    parser = argparse.ArgumentParser(
        description="Run ARC-AGI-3 game agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run random agent on a specific game
  python main.py --agent=randomagent --game=ls20
  
  # Run random agent on all games
  python main.py --agent=randomagent
  
  # List available agents
  python main.py --list-agents
        """
    )
    parser.add_argument(
        "--agent",
        type=str,
        help="Name of the agent to run (in lowercase)"
    )
    parser.add_argument(
        "--game",
        type=str,
        default="all",
        help="Game to play (default: all)"
    )
    parser.add_argument(
        "--list-agents",
        action="store_true",
        help="List all available agents"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("ARC_API_KEY"),
        help="ARC API key (defaults to ARC_API_KEY environment variable)"
    )
    
    args = parser.parse_args()
    
    # List available agents
    if args.list_agents:
        print("\nAvailable agents:")
        for agent_name in sorted(AVAILABLE_AGENTS.keys()):
            agent_class = AVAILABLE_AGENTS[agent_name]
            print(f"  - {agent_name}: {agent_class.__name__}")
            if agent_class.__doc__:
                doc_first_line = agent_class.__doc__.strip().split('\n')[0]
                print(f"    {doc_first_line}")
        print()
        return
    
    # Validate required arguments
    if not args.agent:
        parser.error("--agent is required (or use --list-agents to see available agents)")
    
    # Get the agent class
    agent_name = args.agent.lower()
    if agent_name not in AVAILABLE_AGENTS:
        available = ", ".join(sorted(AVAILABLE_AGENTS.keys()))
        raise ValueError(
            f"Agent '{args.agent}' not found. "
            f"Available agents: {available}"
        )
    
    agent_class = AVAILABLE_AGENTS[agent_name]
    
    # Check for API key
    if not args.api_key:
        print("Warning: ARC_API_KEY not found in environment variables.")
        print("You can set it with: export ARC_API_KEY=your_api_key")
        print("Or obtain one from: https://three.arcprize.org")
        print()
    
    # Create agent instance
    print(f"Initializing agent: {agent_class.__name__}")
    agent = agent_class()
    
    # Run the agent
    print(f"Running agent on game: {args.game}")
    print("-" * 60)
    
    # Simulate a simple game loop (placeholder for actual game integration)
    # In a real implementation, this would connect to the ARC-AGI-3 API
    frames = []
    current_frame = FrameData(state=GameState.NOT_PLAYED)
    
    max_steps = 10
    for step in range(max_steps):
        print(f"\nStep {step + 1}:")
        
        # Check if game is done
        if agent.is_done(frames, current_frame):
            print("Agent determined game is complete!")
            break
        
        # Get next action
        action = agent.choose_action(frames, current_frame)
        print(f"  Action: {action.value}")
        
        # Display reasoning if available
        if hasattr(action, 'reasoning'):
            print(f"  Reasoning: {action.reasoning}")
        
        # Display data for complex actions
        if action.is_complex():
            data = action.get_data()
            if data:
                print(f"  Data: {data}")
        
        # Add frame to history
        frames.append(current_frame)
        
        # Simulate next frame (in real implementation, this would come from API)
        current_frame = FrameData(state=GameState.PLAYING)
    
    print("\n" + "-" * 60)
    print("Agent run completed!")
    print(f"\nNote: This is a demonstration of the agent infrastructure.")
    print(f"For actual game play, integrate with the ARC-AGI-3 API.")
    print(f"See: https://github.com/arcprize/ARC-AGI-3-Agents for more details.")


if __name__ == "__main__":
    main()

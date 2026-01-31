"""TRM (Tiny Recursive Model) agent for ARC-AGI-3 games.

This agent integrates the TinyRecursiveModels approach with the ARC-AGI-3 game framework,
allowing the recursive reasoning model to play ARC-AGI-3 games.
"""

import os
import sys
from typing import List, Optional
import random

# Add the project root to path to allow imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agents.agent import Agent
from agents.structs import FrameData, GameAction, GameState


class TRMAgent(Agent):
    """TRM (Tiny Recursive Model) agent for ARC-AGI-3.
    
    This agent uses the Tiny Recursive Model approach to play ARC-AGI-3 games.
    The TRM uses recursive reasoning to progressively improve its predictions
    over multiple cycles.
    
    Attributes:
        max_steps: Maximum number of steps before considering the game done
        model_loaded: Whether the TRM model has been loaded
        recursion_cycles: Number of recursive cycles for the model
    """
    
    def __init__(self, max_steps: int = 100, recursion_cycles: int = 3):
        """Initialize the TRM agent.
        
        Args:
            max_steps: Maximum number of steps before considering done
            recursion_cycles: Number of recursive reasoning cycles
        """
        super().__init__()
        self.max_steps = max_steps
        self.recursion_cycles = recursion_cycles
        self.model_loaded = False
        self.step_count = 0
        
    def _load_model(self):
        """Load the TRM model if not already loaded.
        
        This is a placeholder for loading a pre-trained TRM model.
        In a full implementation, this would load weights from a checkpoint.
        """
        if not self.model_loaded:
            # TODO: Load pre-trained TRM model
            # For now, this is a placeholder
            self.model_loaded = True
            
    def _recursive_reasoning(self, grid: Optional[list]) -> GameAction:
        """Apply recursive reasoning to determine the best action.
        
        Args:
            grid: The current game grid
            
        Returns:
            The chosen GameAction
        """
        # This is a simplified version - a full implementation would:
        # 1. Embed the current grid state
        # 2. Run recursive cycles to improve the prediction
        # 3. Decode the prediction to an action
        
        # For now, use a simple heuristic approach as a placeholder
        actions = [
            GameAction.MOVE_UP,
            GameAction.MOVE_DOWN,
            GameAction.MOVE_LEFT,
            GameAction.MOVE_RIGHT,
        ]
        
        return random.choice(actions)

    def is_done(self, frames: List[FrameData], latest_frame: FrameData) -> bool:
        """Determine if the game is finished.
        
        The game is done if:
        - We've won the game
        - We've reached max_steps
        - The game is over
        
        Args:
            frames: List of all previous frames in the game
            latest_frame: The most recent frame
            
        Returns:
            True if the game should stop, False otherwise
        """
        self.step_count = len(frames)
        
        # Check if we've won
        if latest_frame.state == GameState.WIN:
            return True
            
        # Check if we've hit max steps
        if self.step_count >= self.max_steps:
            return True
            
        # Check if game is over
        if latest_frame.state == GameState.GAME_OVER:
            return True
            
        return False

    def choose_action(self, frames: List[FrameData], latest_frame: FrameData) -> GameAction:
        """Choose the next action using TRM recursive reasoning.
        
        Args:
            frames: List of all previous frames in the game
            latest_frame: The most recent frame
            
        Returns:
            The GameAction to execute next
        """
        # Ensure model is loaded
        self._load_model()
        
        # Handle game initialization
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            action = GameAction.RESET
            action.reasoning = "Starting new game"
            return action
        
        # Use recursive reasoning to choose action
        action = self._recursive_reasoning(latest_frame.grid)
        
        # Add reasoning explanation
        if action.is_simple():
            action.reasoning = f"TRM recursive reasoning (cycle {self.recursion_cycles}): {action.value}"
        elif action.is_complex():
            # For complex actions, we'd need to determine coordinates
            # Using 0-63 range as standard ARC-AGI grid size (64x64 max)
            # In production, this should be derived from actual grid dimensions
            action.set_data({
                "x": random.randint(0, 63),
                "y": random.randint(0, 63),
            })
            action.reasoning = {
                "action": action.value,
                "reason": f"TRM prediction with {self.recursion_cycles} cycles"
            }
        
        return action

"""Random agent template for ARC-AGI-3 games."""

import random
from ..agent import Agent
from ..structs import FrameData, GameAction, GameState


class RandomAgent(Agent):
    """A simple agent that chooses random actions.
    
    This is a template agent that demonstrates the basic structure
    of an ARC-AGI-3 agent. It makes random decisions and can be used
    as a starting point for creating more sophisticated agents.
    """

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Check if the game is finished.
        
        Args:
            frames: List of all previous frames
            latest_frame: The most recent frame
            
        Returns:
            True if the game has been won, False otherwise
        """
        return latest_frame.state == GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        """Choose a random action to take.
        
        Args:
            frames: List of all previous frames
            latest_frame: The most recent frame
            
        Returns:
            A randomly selected GameAction
        """
        # Start or restart the game if needed
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            action = GameAction.RESET
        else:
            # Choose a random action (except RESET)
            action = random.choice([a for a in GameAction if a != GameAction.RESET])

        # Add reasoning for simple actions
        if action.is_simple():
            action.reasoning = f"Chose {action.value} randomly"
        # For complex actions, set coordinates
        elif action.is_complex():
            action.set_data({
                "x": random.randint(0, 63),
                "y": random.randint(0, 63),
            })
            action.reasoning = {"action": action.value, "reason": "Random choice"}

        return action

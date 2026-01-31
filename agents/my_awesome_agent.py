"""My awesome agent for ARC-AGI-3 games."""

from .agent import Agent
from .structs import FrameData, GameAction, GameState
import random


class MyAwesomeAgent(Agent):
    """A simple agent that chooses random actions."""

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Check if the game is finished."""
        # Your logic to determine if the game is finished
        return latest_frame.state == GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        """Choose an action to take."""
        # Your custom decision-making logic goes here
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            # Start or restart the game
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

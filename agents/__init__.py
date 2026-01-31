"""Agent registry for ARC-AGI-3 games."""

from .agent import Agent
from .structs import FrameData, GameAction, GameState
from .templates.random_agent import RandomAgent
from .my_awesome_agent import MyAwesomeAgent

# Dictionary mapping agent names to agent classes
# Add your custom agents here
AVAILABLE_AGENTS = {
    "randomagent": RandomAgent,
    "myawesomeagent": MyAwesomeAgent,
}

__all__ = [
    "Agent",
    "FrameData",
    "GameAction",
    "GameState",
    "RandomAgent",
    "MyAwesomeAgent",
    "AVAILABLE_AGENTS",
]

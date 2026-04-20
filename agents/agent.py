"""Base Agent interface for ARC-AGI-3 games."""

from abc import ABC, abstractmethod
from typing import List
from .structs import FrameData, GameAction


class Agent(ABC):
    """Abstract base class for ARC-AGI-3 game agents.
    
    All custom agents must inherit from this class and implement
    the required methods: is_done() and choose_action().
    """
    
    def __init__(self):
        """Initialize the agent."""
        pass
    
    @abstractmethod
    def is_done(self, frames: List[FrameData], latest_frame: FrameData) -> bool:
        """Determine if the game is finished.
        
        Args:
            frames: List of all previous frames in the game
            latest_frame: The most recent frame
            
        Returns:
            True if the game should stop, False otherwise
        """
        pass
    
    @abstractmethod
    def choose_action(self, frames: List[FrameData], latest_frame: FrameData) -> GameAction:
        """Choose the next action to take in the game.
        
        Args:
            frames: List of all previous frames in the game
            latest_frame: The most recent frame
            
        Returns:
            The GameAction to execute next
        """
        pass

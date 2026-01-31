"""Data structures for ARC-AGI-3 game agents."""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


class GameState(Enum):
    """Represents the current state of the game."""
    NOT_PLAYED = "not_played"
    PLAYING = "playing"
    WIN = "win"
    GAME_OVER = "game_over"


class GameAction(Enum):
    """Available actions in ARC-AGI-3 games."""
    RESET = "reset"
    MOVE_UP = "move_up"
    MOVE_DOWN = "move_down"
    MOVE_LEFT = "move_left"
    MOVE_RIGHT = "move_right"
    PLACE = "place"
    REMOVE = "remove"
    
    def is_simple(self) -> bool:
        """Check if action is simple (doesn't require coordinates)."""
        return self in [GameAction.RESET, GameAction.MOVE_UP, GameAction.MOVE_DOWN, 
                       GameAction.MOVE_LEFT, GameAction.MOVE_RIGHT]
    
    def is_complex(self) -> bool:
        """Check if action is complex (requires coordinates)."""
        return self in [GameAction.PLACE, GameAction.REMOVE]
    
    def set_data(self, data: Dict[str, Any]) -> None:
        """Set action data (e.g., coordinates)."""
        self._data = data
    
    def get_data(self) -> Optional[Dict[str, Any]]:
        """Get action data."""
        return getattr(self, '_data', None)


@dataclass
class FrameData:
    """Represents a single frame of game state."""
    state: GameState
    grid: Optional[list] = None
    score: int = 0
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

"""Tests for the TRM agent."""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents import TRMAgent, FrameData, GameAction, GameState


def test_trm_agent_init():
    """Test TRM agent initialization."""
    agent = TRMAgent(max_steps=50, recursion_cycles=3)
    assert agent.max_steps == 50
    assert agent.recursion_cycles == 3
    assert agent.step_count == 0
    assert not agent.model_loaded
    print("✓ TRM agent initialization test passed")


def test_trm_agent_is_done_win():
    """Test that agent recognizes when game is won."""
    agent = TRMAgent()
    frames = []
    win_frame = FrameData(state=GameState.WIN)
    
    assert agent.is_done(frames, win_frame)
    print("✓ TRM agent win detection test passed")


def test_trm_agent_is_done_max_steps():
    """Test that agent stops at max steps."""
    agent = TRMAgent(max_steps=5)
    frames = [FrameData(state=GameState.PLAYING) for _ in range(5)]
    current_frame = FrameData(state=GameState.PLAYING)
    
    assert agent.is_done(frames, current_frame)
    print("✓ TRM agent max steps test passed")


def test_trm_agent_is_done_game_over():
    """Test that agent recognizes game over state."""
    agent = TRMAgent()
    frames = []
    game_over_frame = FrameData(state=GameState.GAME_OVER)
    
    assert agent.is_done(frames, game_over_frame)
    print("✓ TRM agent game over detection test passed")


def test_trm_agent_choose_action_reset():
    """Test that agent resets when game is not started."""
    agent = TRMAgent()
    frames = []
    not_played_frame = FrameData(state=GameState.NOT_PLAYED)
    
    action = agent.choose_action(frames, not_played_frame)
    assert action == GameAction.RESET
    assert hasattr(action, 'reasoning')
    print("✓ TRM agent reset action test passed")


def test_trm_agent_choose_action_playing():
    """Test that agent chooses valid actions during play."""
    agent = TRMAgent()
    frames = []
    playing_frame = FrameData(state=GameState.PLAYING, grid=[[0] * 8 for _ in range(8)])
    
    action = agent.choose_action(frames, playing_frame)
    
    # Action should be one of the valid game actions
    valid_actions = [
        GameAction.MOVE_UP,
        GameAction.MOVE_DOWN,
        GameAction.MOVE_LEFT,
        GameAction.MOVE_RIGHT,
        GameAction.PLACE,
        GameAction.REMOVE,
    ]
    assert action in valid_actions
    assert agent.model_loaded  # Model should be loaded after first action
    print("✓ TRM agent action selection test passed")


def test_trm_agent_multiple_steps():
    """Test agent over multiple steps."""
    agent = TRMAgent(max_steps=10)
    frames = []
    current_frame = FrameData(state=GameState.NOT_PLAYED)
    
    step_count = 0
    max_test_steps = 5
    
    while not agent.is_done(frames, current_frame) and step_count < max_test_steps:
        action = agent.choose_action(frames, current_frame)
        assert isinstance(action, GameAction)
        
        frames.append(current_frame)
        current_frame = FrameData(state=GameState.PLAYING)
        step_count += 1
    
    assert step_count == max_test_steps
    print("✓ TRM agent multiple steps test passed")


def run_all_tests():
    """Run all TRM agent tests."""
    print("\n" + "=" * 80)
    print("Running TRM Agent Tests")
    print("=" * 80 + "\n")
    
    test_trm_agent_init()
    test_trm_agent_is_done_win()
    test_trm_agent_is_done_max_steps()
    test_trm_agent_is_done_game_over()
    test_trm_agent_choose_action_reset()
    test_trm_agent_choose_action_playing()
    test_trm_agent_multiple_steps()
    
    print("\n" + "=" * 80)
    print("All TRM Agent Tests Passed! ✓")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_all_tests()

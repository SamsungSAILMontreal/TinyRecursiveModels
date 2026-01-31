# Create Your Own Agent

> Build a custom agent for ARC-AGI-3 games

Create AI agents that can play ARC-AGI-3 games by implementing the required interface methods. This guide is based on the [ARC-AGI-3 Agents repo](https://github.com/arcprize/ARC-AGI-3-Agents).

## Prerequisites

Make sure you have your `ARC_API_KEY` populated in your environment variables. You can obtain this key by signing up for an account on the [ARC-AGI-3 website](https://three.arcprize.org).

```bash
export ARC_API_KEY=your_api_key_here
```

## Step 1: Create Your Agent File

Create a new Python file for your agent inside the `agents/` directory. For this example, let's copy the `random_agent.py` template.

```bash
cp agents/templates/random_agent.py agents/my_awesome_agent.py
```

Now, modify `agents/my_awesome_agent.py` and rename the class to `MyAwesomeAgent`.

```python
# agents/my_awesome_agent.py

from .agent import Agent  # Make sure to use correct imports
from .structs import FrameData, GameAction, GameState  # Make sure to use correct imports
import random

# Rename the class
class MyAwesomeAgent(Agent):
    """A simple agent that chooses random actions."""

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        # Your logic to determine if the game is finished
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        # Your custom decision-making logic goes here
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            # Start or restart the game
            action = GameAction.RESET
        else:
            # Choose a random action (except RESET)
            action = random.choice([a for a in GameAction if a is not GameAction.RESET])

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
```

## Step 2: Register Your Agent

To make your agent available to run, add an import statement to `agents/__init__.py` and add it to the `AVAILABLE_AGENTS` dictionary:

```python
# agents/__init__.py
# ... existing imports ...
from .my_awesome_agent import MyAwesomeAgent

# Add to AVAILABLE_AGENTS
AVAILABLE_AGENTS = {
    # ... existing agents ...
    "myawesomeagent": MyAwesomeAgent,
}

__all__ = [
    # ... existing agents ...
    "MyAwesomeAgent",
    "AVAILABLE_AGENTS",
]
```

## Step 3: Run Your Agent

Your agent is now registered and ready to run. Use the class name in lower case as the value for the `--agent` argument.

```bash
# Run your custom agent on the 'ls20' game
python main.py --agent=myawesomeagent --game=ls20
```

You can also run it against all available games:

```bash
# Run your agent on all games
python main.py --agent=myawesomeagent
```

### List Available Agents

To see all registered agents:

```bash
python main.py --list-agents
```

## Agent Interface Reference

### Required Methods

Your agent must implement two methods:

#### `is_done(frames, latest_frame) -> bool`

Determines if the game is finished.

**Parameters:**
- `frames`: List of all previous FrameData objects
- `latest_frame`: The most recent FrameData

**Returns:**
- `True` if the game should stop, `False` otherwise

**Example:**
```python
def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
    return latest_frame.state == GameState.WIN
```

#### `choose_action(frames, latest_frame) -> GameAction`

Chooses the next action to take.

**Parameters:**
- `frames`: List of all previous FrameData objects
- `latest_frame`: The most recent FrameData

**Returns:**
- A `GameAction` enum value

**Example:**
```python
def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
    if latest_frame.state == GameState.NOT_PLAYED:
        return GameAction.RESET
    return GameAction.MOVE_UP
```

### Available GameActions

Simple actions (no coordinates required):
- `GameAction.RESET` - Start or restart the game
- `GameAction.MOVE_UP` - Move up
- `GameAction.MOVE_DOWN` - Move down
- `GameAction.MOVE_LEFT` - Move left
- `GameAction.MOVE_RIGHT` - Move right

Complex actions (require coordinates):
- `GameAction.PLACE` - Place an object
- `GameAction.REMOVE` - Remove an object

For complex actions, set coordinates using:
```python
action.set_data({"x": 10, "y": 20})
```

### GameState Enum

- `GameState.NOT_PLAYED` - Game hasn't started
- `GameState.PLAYING` - Game is in progress
- `GameState.WIN` - Game won
- `GameState.GAME_OVER` - Game over (lost)

### FrameData Structure

Each frame contains:
- `state`: Current GameState
- `grid`: Optional grid representation
- `score`: Current score
- `timestamp`: Frame timestamp
- `metadata`: Dictionary of additional metadata

## Troubleshooting

### Relative Import Errors

If you move an agent file or create a new one outside the `agents/` directory, you may encounter `ImportError` exceptions related to relative imports.

**Solution:**
Ensure your import statements use the correct relative pathing. The `..` prefix goes up one directory level.

For example, if your agent is in `agents/my_agents/my_file.py`, the imports should look like this:

```python
# agents/my_agents/my_file.py

# Correct: Go up one level to the 'agents' package root
from ..agent import Agent
from ..structs import FrameData, GameAction, GameState

# Incorrect: Assumes the file is in the 'agents' root
# from .agent import Agent
```

### Agent Not Found Errors

If you see `ValueError: Agent '<your-agent>' not found`, double-check the following:

1. Your agent class is correctly located in the `agents` directory (or a subdirectory).
2. The class name is correctly spelled and matches the name you provided to the `--agent` flag (in lower case).
3. You have saved your changes to your agent file.
4. You have added your agent to the `AVAILABLE_AGENTS` dictionary in `agents/__init__.py`.

## Advanced Topics

### Stateful Agents

You can maintain state across actions by storing information in your agent instance:

```python
class StatefulAgent(Agent):
    def __init__(self):
        super().__init__()
        self.move_count = 0
        
    def choose_action(self, frames, latest_frame):
        self.move_count += 1
        # Use self.move_count in your logic
        return GameAction.MOVE_UP
```

### Using Frame History

Analyze previous frames to make better decisions:

```python
def choose_action(self, frames, latest_frame):
    # Check if we've been moving in circles
    if len(frames) >= 3:
        recent_states = [f.state for f in frames[-3:]]
        # Adjust strategy based on recent history
    return GameAction.MOVE_RIGHT
```

## Integration with ARC-AGI-3 API

For actual game play with the ARC-AGI-3 platform, you'll need to:

1. Install additional dependencies for API communication
2. Replace the simulated game loop in `main.py` with actual API calls
3. Handle game state updates from the API responses

See the [ARC-AGI-3 Agents repository](https://github.com/arcprize/ARC-AGI-3-Agents) for the complete implementation.

## Example Agents

### Random Agent

The template `agents/templates/random_agent.py` provides a basic example that:
- Starts/restarts the game when needed
- Chooses random actions during play
- Adds reasoning to actions
- Handles both simple and complex actions

Use this as a starting point for your own agents!

## Next Steps

1. Study the provided `RandomAgent` template
2. Create your own agent with custom logic
3. Test it using `python main.py --agent=youragent`
4. Iterate and improve based on results
5. Share your agent with the community!

For more information and examples, visit:
- [ARC-AGI-3 Agents](https://github.com/arcprize/ARC-AGI-3-Agents)
- [ARC Prize](https://arcprize.org)
- [ARC-AGI-3 Platform](https://three.arcprize.org)

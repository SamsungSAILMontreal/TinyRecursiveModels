# ARC-AGI-3 Agent Integration Guide

This document describes how to integrate the TinyRecursiveModels (TRM) approach with the ARC-AGI-3 agent framework.

## Overview

The TRM agent brings recursive reasoning capabilities to the ARC-AGI-3 platform, allowing the model to progressively improve its predictions through multiple reasoning cycles.

## Quick Start

### 1. Running the TRM Agent

You can run the TRM agent in two ways:

#### Simple Demo (No External Dependencies)
```bash
# Run the experiment script
python experiments/run_trm_arc_agi_3.py --game=all

# Run with custom parameters
python experiments/run_trm_arc_agi_3.py --game=ls20 --max-steps=200 --cycles=5
```

#### Full ARC-AGI-3 Integration (Requires ARC-AGI-3 Setup)
```bash
# Using the main.py from this repository
python main.py --agent=trmagent --game=ls20
```

### 2. Running Tests

```bash
# Run the TRM agent tests
python tests/test_trm_agent.py
```

## Integration with ARC-AGI-3-Agents

To fully integrate with the official ARC-AGI-3-Agents framework:

### Installation

1. Install the ARC-AGI-3 dependencies:
```bash
# Clone the ARC-AGI-3-Agents repository
git clone https://github.com/arcprize/ARC-AGI-3-Agents.git
cd ARC-AGI-3-Agents

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

2. Set up environment variables:
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your ARC API key
# Get an API key from https://three.arcprize.org/
```

3. Copy the TRM agent to the ARC-AGI-3-Agents repository:
```bash
# From the TinyRecursiveModels directory
cp agents/trm_agent.py /path/to/ARC-AGI-3-Agents/agents/templates/
```

4. Register the TRM agent in ARC-AGI-3-Agents:
```python
# Edit /path/to/ARC-AGI-3-Agents/agents/__init__.py
# Add:
from .templates.trm_agent import TRMAgent

# The agent will be automatically registered via the subclass mechanism
```

### Running with Full ARC-AGI-3 API

Once integrated with the ARC-AGI-3-Agents framework:

```bash
cd /path/to/ARC-AGI-3-Agents
uv run main.py --agent=trmagent --game=ls20
```

## TRM Agent Architecture

The TRM agent uses the following approach:

1. **Initialization**: Sets up the recursive reasoning model with configurable cycles
2. **Recursive Reasoning**: Applies multiple cycles of reasoning to improve predictions
3. **Action Selection**: Converts model predictions to game actions

### Key Parameters

- `max_steps`: Maximum number of steps before the agent stops (default: 100)
- `recursion_cycles`: Number of recursive reasoning cycles (default: 3)

### Example Usage

```python
from agents import TRMAgent, FrameData, GameState

# Create agent with custom parameters
agent = TRMAgent(max_steps=150, recursion_cycles=5)

# Use in game loop
frames = []
current_frame = FrameData(state=GameState.NOT_PLAYED)

while not agent.is_done(frames, current_frame):
    action = agent.choose_action(frames, current_frame)
    # Execute action and get next frame
    # ...
```

## Development Notes

### Current Implementation

The current TRM agent is a demonstration implementation that:
- Shows the integration structure with ARC-AGI-3
- Implements the required Agent interface
- Includes placeholder logic for recursive reasoning

### Future Enhancements

To create a fully functional TRM agent, you would need to:

1. **Load Pre-trained Model**: Integrate a trained TRM checkpoint
2. **Grid Encoding**: Implement proper encoding of game grids
3. **Recursive Cycles**: Implement the full recursive reasoning logic
4. **Action Decoding**: Map model outputs to game actions

Example integration with pre-trained model:

```python
import torch
from models.recursive_reasoning.trm import TRM

class TRMAgent(Agent):
    def __init__(self, checkpoint_path=None, **kwargs):
        super().__init__(**kwargs)
        
        # Load pre-trained model
        if checkpoint_path:
            self.model = TRM(...)
            self.model.load_state_dict(torch.load(checkpoint_path))
            self.model.eval()
```

## Testing

The test suite includes:

- Agent initialization tests
- Game state detection tests (win, game over, max steps)
- Action selection tests
- Multi-step simulation tests

Run all tests:
```bash
python tests/test_trm_agent.py
```

Expected output:
```
================================================================================
Running TRM Agent Tests
================================================================================

✓ TRM agent initialization test passed
✓ TRM agent win detection test passed
✓ TRM agent max steps test passed
✓ TRM agent game over detection test passed
✓ TRM agent reset action test passed
✓ TRM agent action selection test passed
✓ TRM agent multiple steps test passed

================================================================================
All TRM Agent Tests Passed! ✓
================================================================================
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the project root or that the path is set correctly
2. **Missing Dependencies**: Install required packages from requirements.txt
3. **API Key Issues**: Verify your ARC_API_KEY is set in .env

### Getting Help

- TinyRecursiveModels: https://github.com/drqsatoshi/TinyRecursiveModels
- ARC-AGI-3-Agents: https://github.com/arcprize/ARC-AGI-3-Agents
- ARC-AGI-3 Documentation: https://three.arcprize.org/docs

## References

- [TinyRecursiveModels Paper](https://arxiv.org/abs/2510.04871)
- [ARC-AGI-3 Documentation](https://three.arcprize.org/docs)
- [ARC-AGI-3 Tutorial Video](https://youtu.be/xEVg9dcJMkw)

"""
MCP (Model Context Protocol) tool server for the recursive model's memory.

Exposes FloJsonOutputCollector operations as MCP tools so that any
MCP-compatible client (Claude Desktop, Cursor, etc.) can interact with
the persistent memory of a TinyRecursiveModel at runtime.

Run as a standalone MCP server::

    python -m agents.mcp_tool

or register it in your MCP client config pointing at this module.
"""

import json
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

from models.flo_json_output_collector import FloJsonOutputCollector, FloIterator

# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "TRM Memory",
    instructions=(
        "Tools for managing the persistent memory of a TinyRecursiveModel. "
        "Use append_output to record agent outputs, peek/pop/fetch for retrieval, "
        "and rewind/iter_q for Q-promise-style replay of past entries."
    ),
)

# Module-level collector instance (shared across tool calls in a session)
_collector: FloJsonOutputCollector = FloJsonOutputCollector()


def _get_collector() -> FloJsonOutputCollector:
    return _collector


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def append_output(agent_output: str) -> str:
    """Extract JSON from *agent_output* and append it to persistent memory.

    Args:
        agent_output: Raw text from an agent or LLM that contains JSON.

    Returns:
        Confirmation message with the number of entries now stored.
    """
    collector = _get_collector()
    collector.append(agent_output)
    return f"Appended. Total entries in memory: {len(collector.data)}"


@mcp.tool()
def peek_memory() -> Dict[str, Any]:
    """Return the most recent memory entry without removing it.

    Returns:
        The last collected JSON dict, or an empty dict if memory is empty.
    """
    collector = _get_collector()
    result = collector.peek()
    return result if result is not None else {}


@mcp.tool()
def pop_memory() -> Dict[str, Any]:
    """Remove and return the most recent memory entry.

    Returns:
        The removed JSON dict, or an empty dict if memory is empty.
    """
    collector = _get_collector()
    if not collector.data:
        return {}
    return collector.pop()


@mcp.tool()
def fetch_memory() -> Dict[str, Any]:
    """Merge all collected memory entries into a single dict and return it.

    Returns:
        Merged dict of all stored JSON entries (later keys override earlier).
    """
    return _get_collector().fetch()


@mcp.tool()
def rewind_memory(depth: Optional[int] = None) -> str:
    """Replay memory entries newest→oldest as a JSON array string.

    Implements the Q-promise rewind pattern: each entry is visited in
    reverse chronological order, mirroring JS Promise.then chaining.

    Args:
        depth: Maximum number of entries to replay (None = all).

    Returns:
        JSON array of entries replayed, newest first.
    """
    collector = _get_collector()
    replayed: list = []
    collector.rewind(then_callback=replayed.append, depth=depth)
    return json.dumps(replayed)


@mcp.tool()
def iter_q_memory(depth: Optional[int] = None) -> str:
    """Iterate over memory entries using the while-for Q-promise pattern.

    Walks entries newest-first and returns them as a JSON array.

    Args:
        depth: Maximum number of steps to iterate (None = all).

    Returns:
        JSON array of entries in newest-first order.
    """
    collector = _get_collector()
    it = collector.iter_q(depth=depth)
    steps: list = []
    while it.has_next():
        for entry in it.next():
            steps.append(entry)
    return json.dumps(steps)


@mcp.tool()
def memory_size() -> int:
    """Return the number of entries currently stored in memory."""
    return len(_get_collector().data)


@mcp.tool()
def clear_memory() -> str:
    """Clear all entries from persistent memory.

    Returns:
        Confirmation message.
    """
    global _collector
    _collector = FloJsonOutputCollector()
    return "Memory cleared."


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()

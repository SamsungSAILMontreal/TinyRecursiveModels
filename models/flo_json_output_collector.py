"""
FloJsonOutputCollector — collects JSON payloads from LLM/agent outputs,
gracefully handles comments, and offers Q-promise looping for memory replay.

Based on the PMLL (Persistent Memory Logic Loop) pattern from:
https://github.com/rootflo/flo-ai/pull/102 by Dr. Josef Kurk Edwards

Key Features:
  - Strips out // and /* … */ comments before parsing
  - Uses recursive regex to find balanced { … } blocks
  - Strict mode: raises ValueError if no JSON found
  - peek, pop, fetch to manage collected data
  - rewind(): recursive promise-then replay, newest-first
  - iter_q(): while–for hybrid iterator over memory steps
"""

import json
import re
from typing import Any, Callable, Dict, List, Optional


class FloJsonOutputCollector:
    """Collects JSON payloads from agent/LLM outputs with Q-promise replay.

    Attributes:
        strict: If True, raises ValueError when no JSON is found in input.
        data: List of collected JSON dicts, in insertion order.
    """

    def __init__(self, strict: bool = False) -> None:
        self.strict = strict
        self.data: List[Dict[str, Any]] = []

    # ——————————————————————————————————————————————————————————————
    # Internal helpers
    # ——————————————————————————————————————————————————————————————

    def _strip_comments(self, json_str: str) -> str:
        """Remove JS-style // and /* … */ comments so json.loads() will succeed."""
        cleaned: List[str] = []
        length = len(json_str)
        i = 0

        while i < length:
            char = json_str[i]

            # Inside a string literal — copy verbatim until closing quote
            if char == '"':
                cleaned.append(char)
                i += 1
                while i < length:
                    char = json_str[i]
                    cleaned.append(char)
                    i += 1
                    if char == '"' and (i < 2 or json_str[i - 2] != '\\'):
                        break
                continue

            # Possible comment start
            if char == '/' and i + 1 < length:
                next_char = json_str[i + 1]

                if next_char == '/':  # single-line comment
                    i += 2
                    while i < length and json_str[i] != '\n':
                        i += 1
                    continue

                if next_char == '*':  # block comment
                    i += 2
                    while i < length - 1 and not (json_str[i] == '*' and json_str[i + 1] == '/'):
                        i += 1
                    i += 2  # skip closing */
                    continue

            cleaned.append(char)
            i += 1

        return ''.join(cleaned)

    def _extract_jsons(self, llm_response: str) -> Dict[str, Any]:
        """Find all balanced { … } blocks, strip comments, parse and merge.

        Args:
            llm_response: Raw agent/LLM output string.

        Returns:
            Merged dict of all JSON objects found.

        Raises:
            ValueError: If strict=True and no JSON is found.
        """
        # Simple brace-depth scan to collect top-level { … } blocks
        merged: Dict[str, Any] = {}
        depth = 0
        start = -1
        matches: List[str] = []

        for idx, ch in enumerate(llm_response):
            if ch == '{':
                if depth == 0:
                    start = idx
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start != -1:
                    matches.append(llm_response[start:idx + 1])
                    start = -1

        for json_str in matches:
            try:
                cleaned = self._strip_comments(json_str)
                obj = json.loads(cleaned)
                merged.update(obj)
            except json.JSONDecodeError:
                pass  # partial/corrupt blocks are silently skipped

        if self.strict and not matches:
            raise ValueError(
                f'No JSON found in strict mode. Input: {llm_response!r}'
            )

        return merged

    # ——————————————————————————————————————————————————————————————
    # Standard data management
    # ——————————————————————————————————————————————————————————————

    def append(self, agent_output: str) -> None:
        """Extract JSON from *agent_output* and append the resulting dict."""
        self.data.append(self._extract_jsons(agent_output))

    def pop(self) -> Dict[str, Any]:
        """Remove and return the last collected JSON dict."""
        return self.data.pop()

    def peek(self) -> Optional[Dict[str, Any]]:
        """View the last collected JSON dict without removing it."""
        return self.data[-1] if self.data else None

    def fetch(self) -> Dict[str, Any]:
        """Merge all collected dicts into one and return it."""
        merged: Dict[str, Any] = {}
        for d in self.data:
            merged.update(d)
        return merged

    # ——————————————————————————————————————————————————————————————
    # Flo Q-Promise looping methods
    # ——————————————————————————————————————————————————————————————

    def rewind(
        self,
        then_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        depth: Optional[int] = None,
    ) -> None:
        """Recursively replay memory entries newest→oldest.

        Mirrors JS Promise.then chaining in reverse chronological order.

        Args:
            then_callback: Function called for each entry.
            depth: Maximum number of entries to process (None = all).
        """
        if not self.data:
            return

        entries = self.data[::-1]  # reverse: newest first
        if depth is not None:
            entries = entries[:depth]

        def _recursive(idx: int) -> None:
            if idx >= len(entries):
                return
            entry = entries[idx]
            if then_callback:
                then_callback(entry)
            _recursive(idx + 1)

        _recursive(0)

    def iter_q(self, depth: Optional[int] = None) -> "FloIterator":
        """Return a :class:`FloIterator` for a while–for hybrid loop.

        Args:
            depth: Maximum number of entries to iterate (None = all).

        Returns:
            A FloIterator over this collector's data, newest first.
        """
        return FloIterator(self, depth)


class FloIterator:
    """Hybrid while–for iterator over FloJsonOutputCollector data.

    Yields entries newest-first, depth-limited if requested.

    Usage::

        it = collector.iter_q(depth=5)
        while it.has_next():
            for step in it.next():
                print("Q-step:", step)
    """

    def __init__(self, collector: FloJsonOutputCollector, depth: Optional[int] = None) -> None:
        self.entries: List[Dict[str, Any]] = collector.data[::-1]
        self.limit: int = (
            min(depth, len(self.entries)) if depth is not None else len(self.entries)
        )
        self.index: int = 0

    def has_next(self) -> bool:
        """Return True if more entries remain."""
        return self.index < self.limit

    def next(self) -> List[Dict[str, Any]]:
        """Return the next entry wrapped in a list (for inner for-loop compatibility).

        Returns an empty list when exhausted.
        """
        if not self.has_next():
            return []
        entry = self.entries[self.index]
        self.index += 1
        return [entry]

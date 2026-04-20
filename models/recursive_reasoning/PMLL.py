"""
PMLL.py — Persistent Memory Logic Loop (PMLL) runtime for TinyRecursiveModels

Goal:
- Provide a compact, dependency-light persistent memory loop that can be used by
  TRM / hybrid TRM-ERS-PMLL logic.
- Integrates the “new tools” pattern you’ve been using:
    * JSON extraction from LLM/agent outputs (comment-stripping + balanced {} capture)
    * Memory stack primitives: append / peek / pop / fetch
    * Replay/rewind iteration: rewind(callback=...) and iter_q(...)
    * Deterministic hashing for memory blocks (stable IDs)
    * JSON persistence to disk (append-only log + snapshot)
    * Optional topic integration hook + ERS hook points (pluggable)

This module is designed to drop into:
  models/recursive_reasoning/PMLL.py

No external dependencies required.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union, AsyncIterator


# ----------------------------
# Exceptions
# ----------------------------

class PMLLError(Exception):
    pass


class PMLLNoJSONError(PMLLError):
    pass


class PMLLPersistenceError(PMLLError):
    pass


# ----------------------------
# Deterministic hashing
# ----------------------------

def _stable_json_dumps(obj: Any) -> str:
    """Stable JSON string (sorted keys, compact) for hashing / persistence."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def deterministic_hash(payload: Any, salt: str = "") -> str:
    """
    Deterministic ID for a memory block.
    Uses SHA-256 over stable JSON + optional salt.
    """
    s = _stable_json_dumps(payload)
    h = hashlib.sha256()
    h.update(salt.encode("utf-8"))
    h.update(s.encode("utf-8"))
    return h.hexdigest()


# ----------------------------
# JSON extraction (comment stripping + {} capture)
# ----------------------------

_COMMENT_LINE = re.compile(r"//.*?$", re.MULTILINE)
_COMMENT_BLOCK = re.compile(r"/\*.*?\*/", re.DOTALL)

# Balanced-brace extraction (good-enough approach):
# Find candidate '{' and then scan to matching brace accounting for strings/escapes.
def _extract_balanced_json_objects(text: str, max_objects: int = 32) -> List[str]:
    objs: List[str] = []
    n = len(text)
    i = 0

    while i < n and len(objs) < max_objects:
        if text[i] != "{":
            i += 1
            continue

        start = i
        depth = 0
        in_str = False
        escape = False

        while i < n:
            ch = text[i]

            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        # Found a complete balanced object
                        objs.append(text[start : i + 1])
                        i += 1
                        break
            i += 1
        else:
            # No closing brace found; stop scanning further to avoid infinite loops
            break

    return objs


def strip_js_style_comments(s: str) -> str:
    """Remove //... and /*...*/ comments."""
    s = _COMMENT_BLOCK.sub("", s)
    s = _COMMENT_LINE.sub("", s)
    return s


def parse_json_objects_from_text(text: str, strict: bool = False) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Extract all balanced JSON dict objects in text.
    Returns:
      (merged_dict, list_of_dicts)
    Merge policy:
      Later objects override earlier keys.
    """
    cleaned = strip_js_style_comments(text)
    candidates = _extract_balanced_json_objects(cleaned)
    parsed: List[Dict[str, Any]] = []
    merged: Dict[str, Any] = {}

    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                parsed.append(obj)
                merged.update(obj)
        except Exception:
            # ignore candidate parse failures
            continue

    if strict and not parsed:
        raise PMLLNoJSONError("No JSON dict object found in text (strict=True).")

    return merged, parsed


# ----------------------------
# Memory block + persistence
# ----------------------------

@dataclass
class MemoryBlock:
    """
    A single memory item stored by PMLL.

    payload: the dict we store (semantic content)
    mid: deterministic ID (hash)
    ts: unix seconds
    topic: optional topic label / routing key
    meta: auxiliary info (source, agent, score, etc.)
    """
    payload: Dict[str, Any]
    mid: str
    ts: float
    topic: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


class JSONLStore:
    """
    Append-only JSONL store (log) + optional snapshot file.

    Files:
      - <root>/pmll_log.jsonl   (append-only)
      - <root>/pmll_snapshot.json  (latest snapshot of blocks)
    """

    def __init__(self, root: str):
        self.root = root
        self.log_path = os.path.join(root, "pmll_log.jsonl")
        self.snapshot_path = os.path.join(root, "pmll_snapshot.json")
        os.makedirs(root, exist_ok=True)

    def append_block(self, block: MemoryBlock) -> None:
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(_stable_json_dumps(asdict(block)) + "\n")
        except Exception as e:
            raise PMLLPersistenceError(f"Failed to append to JSONL log: {e}") from e

    def save_snapshot(self, blocks: List[MemoryBlock]) -> None:
        try:
            with open(self.snapshot_path, "w", encoding="utf-8") as f:
                f.write(_stable_json_dumps([asdict(b) for b in blocks]))
        except Exception as e:
            raise PMLLPersistenceError(f"Failed to write snapshot: {e}") from e

    def load(self) -> List[MemoryBlock]:
        # Prefer snapshot for speed; replay log if no snapshot exists.
        if os.path.exists(self.snapshot_path):
            try:
                with open(self.snapshot_path, "r", encoding="utf-8") as f:
                    arr = json.loads(f.read())
                return [MemoryBlock(**x) for x in arr]
            except Exception:
                # fall back to log replay
                pass

        blocks: List[MemoryBlock] = []
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        x = json.loads(line)
                        blocks.append(MemoryBlock(**x))
            except Exception as e:
                raise PMLLPersistenceError(f"Failed to replay JSONL log: {e}") from e
        return blocks


# ----------------------------
# Topic integration + ERS hooks (pluggable)
# ----------------------------

class TopicIntegrator:
    """
    Minimal interface: derive a topic label from payload/context.

    Replace or extend this with your project’s integrator.
    """
    def infer_topic(self, payload: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        # Default: try common keys; otherwise None
        for k in ("topic", "domain", "category"):
            v = payload.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None


class ERSAdapter:
    """
    Enhanced Reconsideration System hook points:
    - score(payload, ctx) -> float
    - reconsider(memory, new_payload, ctx) -> revised_payload

    Default behavior: no-op.
    """
    def score(self, payload: Dict[str, Any], ctx: Optional[Dict[str, Any]] = None) -> float:
        return 0.0

    def reconsider(
        self,
        memory: List[MemoryBlock],
        new_payload: Dict[str, Any],
        ctx: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return new_payload


# ----------------------------
# PMLL core
# ----------------------------

class PMLL:
    """
    Persistent Memory Logic Loop

    Key behaviors:
    - append_text(): parse JSON dict(s) from agent/LLM output text, store as blocks
    - append_payload(): store dict directly
    - peek()/pop(): stack-like access (newest-first)
    - fetch(): merge memory payloads into one dict (newest overrides oldest or vice versa)
    - rewind(): replay blocks newest-to-oldest (or reverse) with a callback
    - iter_q(): async iterator over blocks (hybrid while/for style)
    """

    def __init__(
        self,
        *,
        strict_json: bool = False,
        store_dir: Optional[str] = None,
        topic_integrator: Optional[TopicIntegrator] = None,
        ers: Optional[ERSAdapter] = None,
        hash_salt: str = "",
        snapshot_every: int = 50,
        max_blocks: int = 10_000
    ):
        self.strict_json = strict_json
        self.hash_salt = hash_salt
        self.snapshot_every = max(1, snapshot_every)
        self.max_blocks = max_blocks

        self.topic_integrator = topic_integrator or TopicIntegrator()
        self.ers = ers or ERSAdapter()

        self._blocks: List[MemoryBlock] = []
        self._append_count = 0

        self._store: Optional[JSONLStore] = JSONLStore(store_dir) if store_dir else None
        if self._store:
            self._blocks = self._store.load()[-self.max_blocks :]

    # -------------
    # Basic access
    # -------------

    def __len__(self) -> int:
        return len(self._blocks)

    def blocks(self) -> List[MemoryBlock]:
        """Return a copy of blocks (oldest-to-newest)."""
        return list(self._blocks)

    def peek(self) -> Optional[MemoryBlock]:
        """Newest block (stack top)."""
        return self._blocks[-1] if self._blocks else None

    def pop(self) -> Optional[MemoryBlock]:
        """Remove and return newest block."""
        if not self._blocks:
            return None
        return self._blocks.pop()

    def fetch(
        self,
        *,
        newest_overrides: bool = True,
        include_meta: bool = False
    ) -> Dict[str, Any]:
        """
        Merge payloads into a single dict.
        If newest_overrides=True: older merged first, then newer overrides.
        """
        merged: Dict[str, Any] = {}
        blocks = self._blocks if newest_overrides else list(reversed(self._blocks))

        for b in blocks:
            merged.update(b.payload)

        if include_meta:
            merged["_pmll"] = {
                "count": len(self._blocks),
                "newest_mid": self._blocks[-1].mid if self._blocks else None,
                "oldest_mid": self._blocks[0].mid if self._blocks else None,
                "ts_newest": self._blocks[-1].ts if self._blocks else None,
                "ts_oldest": self._blocks[0].ts if self._blocks else None,
            }
        return merged

    # -------------
    # Append APIs
    # -------------

    def append_payload(
        self,
        payload: Dict[str, Any],
        *,
        topic: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        ctx: Optional[Dict[str, Any]] = None
    ) -> MemoryBlock:
        """
        Append a single dict as a memory block (sync).
        Applies ERS reconsideration hook and topic inference if not provided.
        """
        revised = self.ers.reconsider(self._blocks, payload, ctx)
        if not isinstance(revised, dict):
            raise PMLLError("ERS reconsider() must return a dict payload.")

        if topic is None:
            topic = self.topic_integrator.infer_topic(revised, ctx)

        mid = deterministic_hash(revised, salt=self.hash_salt)
        block = MemoryBlock(payload=revised, mid=mid, ts=time.time(), topic=topic, meta=meta)

        self._blocks.append(block)
        if len(self._blocks) > self.max_blocks:
            self._blocks = self._blocks[-self.max_blocks :]

        self._append_count += 1

        if self._store:
            self._store.append_block(block)
            if self._append_count % self.snapshot_every == 0:
                self._store.save_snapshot(self._blocks)

        return block

    def append_text(
        self,
        agent_output: str,
        *,
        meta: Optional[Dict[str, Any]] = None,
        ctx: Optional[Dict[str, Any]] = None
    ) -> List[MemoryBlock]:
        """
        Parse JSON dict(s) from raw agent output and append each dict as a block.
        Returns the blocks appended (in parse order).
        """
        merged, objs = parse_json_objects_from_text(agent_output, strict=self.strict_json)
        if not objs and merged:
            objs = [merged]

        blocks: List[MemoryBlock] = []
        for obj in objs:
            blocks.append(self.append_payload(obj, meta=meta, ctx=ctx))
        return blocks

    async def append_text_async(
        self,
        agent_output: str,
        *,
        meta: Optional[Dict[str, Any]] = None,
        ctx: Optional[Dict[str, Any]] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> List[MemoryBlock]:
        """
        Async version of append_text(), offloads parsing to thread executor.
        """
        loop = loop or asyncio.get_event_loop()
        merged, objs = await loop.run_in_executor(
            None,
            lambda: parse_json_objects_from_text(agent_output, strict=self.strict_json)
        )
        if not objs and merged:
            objs = [merged]

        blocks: List[MemoryBlock] = []
        for obj in objs:
            blocks.append(self.append_payload(obj, meta=meta, ctx=ctx))
        return blocks

    # -------------
    # Rewind / Q-promise iteration
    # -------------

    def rewind(
        self,
        callback: Optional[Callable[[MemoryBlock], Any]] = None,
        *,
        newest_first: bool = True,
        depth: Optional[int] = None
    ) -> List[MemoryBlock]:
        """
        Replay blocks in time order.
        - newest_first=True: newest -> oldest
        - depth: limit number of steps
        - callback: called for each block
        Returns the replayed list (in replay order).
        """
        if not self._blocks:
            return []

        seq = reversed(self._blocks) if newest_first else iter(self._blocks)
        out: List[MemoryBlock] = []

        for b in seq:
            out.append(b)
            if callback:
                callback(b)
            if depth is not None and len(out) >= depth:
                break

        return out

    async def iter_q(
        self,
        *,
        newest_first: bool = True,
        depth: Optional[int] = None,
        sleep_s: float = 0.0
    ) -> AsyncIterator[MemoryBlock]:
        """
        Async iterator over memory blocks (Q-promise style).
        Useful for hybrid agent loops:
            async for step in pmll.iter_q(depth=5):
                ...
        """
        seq = list(reversed(self._blocks)) if newest_first else list(self._blocks)
        if depth is not None:
            seq = seq[:depth]

        for b in seq:
            yield b
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)

    # -------------
    # Convenience: memory views
    # -------------

    def memory_stack_view(self, *, newest_first: bool = True) -> List[Dict[str, Any]]:
        """
        Return list of payload dicts (for quick integration into prompts / context).
        """
        seq = reversed(self._blocks) if newest_first else iter(self._blocks)
        return [b.payload for b in seq]

    def topic_index(self) -> Dict[str, List[str]]:
        """
        Simple topic -> list[mid] index.
        """
        idx: Dict[str, List[str]] = {}
        for b in self._blocks:
            if not b.topic:
                continue
            idx.setdefault(b.topic, []).append(b.mid)
        return idx


# ----------------------------
# Optional: Drop-in collector wrapper (so TRM can call it like a tool)
# ----------------------------

class FloJsonOutputCollectorPMLLAdapter:
    """
    Adapter that mirrors the “FloJsonOutputCollector” style API but stores to PMLL.

    Methods:
      - append(agent_output: str)  (async)
      - peek/pop/fetch
      - rewind/iter_q

    Use this when existing TRM code expects a collector-like object.
    """

    def __init__(self, pmll: PMLL):
        self.pmll = pmll

    async def append(self, agent_output: str, meta: Optional[Dict[str, Any]] = None, ctx: Optional[Dict[str, Any]] = None) -> None:
        await self.pmll.append_text_async(agent_output, meta=meta, ctx=ctx)

    def peek(self) -> Optional[Dict[str, Any]]:
        b = self.pmll.peek()
        return b.payload if b else None

    def pop(self) -> Optional[Dict[str, Any]]:
        b = self.pmll.pop()
        return b.payload if b else None

    def fetch(self) -> Dict[str, Any]:
        return self.pmll.fetch(newest_overrides=True)

    def rewind(self, callback: Optional[Callable[[Dict[str, Any]], Any]] = None, depth: Optional[int] = None) -> List[Dict[str, Any]]:
        def _cb(b: MemoryBlock) -> None:
            if callback:
                callback(b.payload)
        blocks = self.pmll.rewind(callback=_cb if callback else None, newest_first=True, depth=depth)
        return [b.payload for b in blocks]

    async def iter_q(self, depth: Optional[int] = None) -> AsyncIterator[List[Dict[str, Any]]]:
        """
        For parity with earlier “wrapped as list” yield style:
          yields [payload] so callers can do: for step in it: ...
        """
        async for b in self.pmll.iter_q(depth=depth):
            yield [b.payload]


# ----------------------------
# Quick self-test (optional)
# ----------------------------

if __name__ == "__main__":
    pmll = PMLL(strict_json=False, store_dir=None)

    sample = """
    // comment
    {"a": 1, "topic": "alpha"}
    some text
    /* block comment */ {"b": 2}
    """

    pmll.append_text(sample, meta={"source": "demo"})
    print("peek:", pmll.peek())
    print("fetch:", pmll.fetch(include_meta=True))

    print("rewind:")
    pmll.rewind(lambda b: print(" -", b.mid, b.payload), depth=10)

    async def _run():
        print("iter_q:")
        async for b in pmll.iter_q(depth=10):
            print(" *", b.mid, b.payload)

    asyncio.run(_run())

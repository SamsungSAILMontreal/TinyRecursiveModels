"""Tests for FloJsonOutputCollector and FloIterator."""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.flo_json_output_collector import FloJsonOutputCollector, FloIterator


# ---------------------------------------------------------------------------
# FloJsonOutputCollector tests
# ---------------------------------------------------------------------------


def test_append_basic_json():
    """Test that basic JSON is parsed and appended correctly."""
    collector = FloJsonOutputCollector()
    collector.append('{"a": 1}')
    assert collector.data == [{"a": 1}]
    print("✓ append basic JSON test passed")


def test_append_json_with_line_comment():
    """Test that // comments are stripped before parsing."""
    collector = FloJsonOutputCollector()
    collector.append('{"x": 2} // ignore this')
    assert collector.data == [{"x": 2}]
    print("✓ append JSON with line comment test passed")


def test_append_json_with_block_comment():
    """Test that /* ... */ comments are stripped before parsing."""
    collector = FloJsonOutputCollector()
    collector.append('{"y": 3} /* ignore this too */')
    assert collector.data == [{"y": 3}]
    print("✓ append JSON with block comment test passed")


def test_append_multiple_json_objects():
    """Test that multiple JSON objects in one string are merged."""
    collector = FloJsonOutputCollector()
    collector.append('{"a": 1} {"b": 2}')
    assert collector.data == [{"a": 1, "b": 2}]
    print("✓ append multiple JSON objects test passed")


def test_append_non_json_text():
    """Test that non-JSON text produces an empty dict."""
    collector = FloJsonOutputCollector()
    collector.append("no json here at all")
    assert collector.data == [{}]
    print("✓ append non-JSON text test passed")


def test_strict_mode_raises_on_no_json():
    """Test that strict mode raises ValueError when no JSON is found."""
    collector = FloJsonOutputCollector(strict=True)
    raised = False
    try:
        collector.append("no json here")
    except ValueError:
        raised = True
    assert raised, "Expected ValueError in strict mode"
    print("✓ strict mode raises ValueError test passed")


def test_peek_returns_last_without_removing():
    """Test that peek returns the last entry without removing it."""
    collector = FloJsonOutputCollector()
    collector.append('{"a": 1}')
    collector.append('{"b": 2}')
    peeked = collector.peek()
    assert peeked == {"b": 2}
    assert len(collector.data) == 2  # nothing removed
    print("✓ peek test passed")


def test_peek_empty_returns_none():
    """Test that peek on empty collector returns None."""
    collector = FloJsonOutputCollector()
    assert collector.peek() is None
    print("✓ peek empty returns None test passed")


def test_pop_removes_and_returns_last():
    """Test that pop removes and returns the last entry."""
    collector = FloJsonOutputCollector()
    collector.append('{"a": 1}')
    collector.append('{"b": 2}')
    popped = collector.pop()
    assert popped == {"b": 2}
    assert len(collector.data) == 1
    assert collector.data[0] == {"a": 1}
    print("✓ pop test passed")


def test_fetch_merges_all():
    """Test that fetch merges all entries into one dict."""
    collector = FloJsonOutputCollector()
    collector.append('{"a": 1}')
    collector.append('{"b": 2}')
    merged = collector.fetch()
    assert merged == {"a": 1, "b": 2}
    # Original data unchanged
    assert len(collector.data) == 2
    print("✓ fetch merges all test passed")


def test_rewind_newest_first():
    """Test that rewind replays entries newest-first."""
    collector = FloJsonOutputCollector()
    collector.append('{"a": 1}')
    collector.append('{"b": 2}')
    replayed = []
    collector.rewind(then_callback=replayed.append)
    assert replayed == [{"b": 2}, {"a": 1}]
    print("✓ rewind newest-first test passed")


def test_rewind_with_depth():
    """Test that rewind respects the depth limit."""
    collector = FloJsonOutputCollector()
    for i in range(4):
        collector.append(f'{{"i": {i}}}')
    replayed = []
    collector.rewind(then_callback=replayed.append, depth=2)
    assert len(replayed) == 2
    assert replayed[0] == {"i": 3}
    assert replayed[1] == {"i": 2}
    print("✓ rewind with depth test passed")


def test_rewind_empty_does_not_raise():
    """Test that rewind on empty collector is a no-op."""
    collector = FloJsonOutputCollector()
    collector.rewind(then_callback=lambda e: None)
    print("✓ rewind empty no-op test passed")


# ---------------------------------------------------------------------------
# FloIterator tests
# ---------------------------------------------------------------------------


def test_iter_q_basic():
    """Test the while-for hybrid iterator returns all entries newest-first."""
    collector = FloJsonOutputCollector()
    collector.append('{"a": 1}')
    collector.append('{"b": 2}')

    it = collector.iter_q()
    steps = []
    while it.has_next():
        for entry in it.next():
            steps.append(entry)

    assert steps == [{"b": 2}, {"a": 1}]
    print("✓ iter_q basic test passed")


def test_iter_q_depth_limit():
    """Test that iter_q respects depth."""
    collector = FloJsonOutputCollector()
    for i in range(5):
        collector.append(f'{{"i": {i}}}')

    it = collector.iter_q(depth=3)
    steps = []
    while it.has_next():
        for entry in it.next():
            steps.append(entry)

    assert len(steps) == 3
    assert steps[0] == {"i": 4}
    print("✓ iter_q depth limit test passed")


def test_iter_q_empty():
    """Test that iter_q on empty collector terminates immediately."""
    collector = FloJsonOutputCollector()
    it = collector.iter_q()
    assert not it.has_next()
    assert it.next() == []
    print("✓ iter_q empty test passed")


def test_iter_q_next_exhausted_returns_empty():
    """Test that calling next() after exhaustion returns []."""
    collector = FloJsonOutputCollector()
    collector.append('{"a": 1}')
    it = collector.iter_q()
    it.next()  # consume the only entry
    assert not it.has_next()
    assert it.next() == []
    print("✓ iter_q exhausted returns [] test passed")


# ---------------------------------------------------------------------------
# Q-promise demo (matches the PR example)
# ---------------------------------------------------------------------------


def test_pr_example():
    """Reproduce the example from rootflo/flo-ai PR #102."""
    collector = FloJsonOutputCollector(strict=False)
    collector.append('{"a":1} // ignore this')
    collector.append('{"b":2} /* ignore this too */')

    # Q-promise rewind
    promise_q = []
    collector.rewind(lambda entry: promise_q.append(entry))
    assert promise_q == [{"b": 2}, {"a": 1}]

    # While-for hybrid
    flo_q = []
    it = collector.iter_q()
    while it.has_next():
        for entry in it.next():
            flo_q.append(entry)
    assert flo_q == [{"b": 2}, {"a": 1}]

    print("✓ PR example test passed")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_all_tests():
    print("\n" + "=" * 80)
    print("Running FloJsonOutputCollector Tests")
    print("=" * 80 + "\n")

    test_append_basic_json()
    test_append_json_with_line_comment()
    test_append_json_with_block_comment()
    test_append_multiple_json_objects()
    test_append_non_json_text()
    test_strict_mode_raises_on_no_json()
    test_peek_returns_last_without_removing()
    test_peek_empty_returns_none()
    test_pop_removes_and_returns_last()
    test_fetch_merges_all()
    test_rewind_newest_first()
    test_rewind_with_depth()
    test_rewind_empty_does_not_raise()
    test_iter_q_basic()
    test_iter_q_depth_limit()
    test_iter_q_empty()
    test_iter_q_next_exhausted_returns_empty()
    test_pr_example()

    print("\n" + "=" * 80)
    print("All FloJsonOutputCollector Tests Passed! ✓")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_all_tests()

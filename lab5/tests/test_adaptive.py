import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import SlidingStats


def test_adaptive_reduces_concurrency_on_errors():
    stats = SlidingStats(initial_concurrency=3)
    for _ in range(5):
        stats.record(False, 300)
    assert stats.adapt(global_limit=5) == 2


def test_adaptive_reduces_concurrency_on_slow_responses():
    stats = SlidingStats(initial_concurrency=3)
    for _ in range(5):
        stats.record(True, 2500)
    assert stats.adapt(global_limit=5) == 2


def test_adaptive_increases_concurrency_on_fast_successes():
    stats = SlidingStats(initial_concurrency=2)
    for _ in range(5):
        stats.record(True, 120)
    assert stats.adapt(global_limit=5) == 3

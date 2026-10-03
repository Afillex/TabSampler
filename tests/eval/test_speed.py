"""Tests for setting E3's speed limit from observed hand moves (ADR 0031)."""

from __future__ import annotations

import pytest

from tabsampler.eval.speed import speed_limit_for


def test_the_limit_is_the_slowest_speed_that_reaches_the_pass_rate() -> None:
    # 10 transitions, 5 free, moves at 1..5 frets/s: 90% needs 4 of the moves to pass.
    assert speed_limit_for(0.9, [5.0, 1.0, 4.0, 2.0, 3.0], free=5, transitions=10) == 4.0


def test_free_transitions_alone_can_reach_the_rate() -> None:
    assert speed_limit_for(0.5, [9.0], free=5, transitions=10) == 0.0


def test_a_rate_no_finite_limit_reaches_is_refused() -> None:
    # One zero-time move fails at any limit, so 100% is out of reach.
    with pytest.raises(ValueError, match="no finite"):
        speed_limit_for(1.0, [1.0, 2.0, 3.0, 4.0], free=5, transitions=10)

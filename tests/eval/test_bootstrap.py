"""Tests for the song-level paired bootstrap (ADR 0028's lesson)."""

from __future__ import annotations

import pytest

from tabsampler.eval.bootstrap import paired_bootstrap


def test_identical_decoders_differ_by_exactly_zero() -> None:
    counts = {"a": (5, 10), "b": (7, 10), "c": (1, 4)}
    result = paired_bootstrap(counts, counts)
    assert (result.delta, result.low, result.high) == (0.0, 0.0, 0.0)


def test_a_uniform_gain_has_a_degenerate_interval() -> None:
    a = {"a": (5, 10), "b": (3, 10)}
    b = {"a": (6, 10), "b": (4, 10)}
    result = paired_bootstrap(a, b)
    assert result.delta == pytest.approx(0.1)
    assert result.low == pytest.approx(0.1)
    assert result.high == pytest.approx(0.1)


def test_the_interval_contains_the_estimate_and_is_reproducible() -> None:
    a = {f"s{i}": (i % 7, 10) for i in range(40)}
    b = {f"s{i}": ((i * 3) % 9, 10) for i in range(40)}
    first, second = paired_bootstrap(a, b, seed=1), paired_bootstrap(a, b, seed=1)
    assert first == second
    assert first.low <= first.delta <= first.high
    assert (first.n_songs, first.n_notes) == (40, 400)


def test_the_bootstrap_refuses_different_songs() -> None:
    with pytest.raises(ValueError, match="same songs"):
        paired_bootstrap({"a": (1, 2)}, {"b": (1, 2)})


def test_the_bootstrap_refuses_different_note_counts() -> None:
    with pytest.raises(ValueError, match="note counts"):
        paired_bootstrap({"a": (1, 2)}, {"a": (1, 3)})

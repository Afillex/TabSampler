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


def test_the_ratio_bootstrap_is_the_paired_one_when_the_counts_agree() -> None:
    from tabsampler.eval.bootstrap import paired_ratio_bootstrap

    a = {"x": (3, 10), "y": (5, 8), "z": (0, 4)}
    b = {"x": (6, 10), "y": (4, 8), "z": (2, 4)}
    assert paired_ratio_bootstrap(a, b, seed=3) == paired_bootstrap(a, b, seed=3)


def test_the_ratio_bootstrap_pools_each_side_over_its_own_counts() -> None:
    from tabsampler.eval.bootstrap import paired_ratio_bootstrap

    # End to end, a transcriber that writes fewer notes changes the denominator too.
    a = {"x": (4, 20), "y": (6, 20)}  # 10 of 40
    b = {"x": (4, 10), "y": (6, 15)}  # 10 of 25
    result = paired_ratio_bootstrap(a, b)
    assert result.delta == pytest.approx(10 / 25 - 10 / 40)
    assert result.low <= result.delta <= result.high
    assert result.n_songs == 2


def test_the_ratio_bootstrap_refuses_different_songs() -> None:
    from tabsampler.eval.bootstrap import paired_ratio_bootstrap

    with pytest.raises(ValueError, match="same songs"):
        paired_ratio_bootstrap({"a": (1, 2)}, {"b": (1, 2)})

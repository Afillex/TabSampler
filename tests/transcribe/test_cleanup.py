"""Filters on transcribed notes (plan 2026-10-06-optimise-current, Task 3)."""

from __future__ import annotations

from tabsampler.transcribe.cleanup import drop_octave_ghosts, drop_quiet
from tabsampler.types import NoteEvent


def n(onset: float, pitch: int, confidence: float = 0.6) -> NoteEvent:
    return NoteEvent(onset=onset, offset=onset + 0.3, pitch=pitch, confidence=confidence)


def test_drop_quiet_keeps_notes_at_or_above_the_floor() -> None:
    notes = [n(0.0, 40, 0.2), n(0.5, 45, 0.3), n(1.0, 50, 0.29)]
    assert drop_quiet(notes, 0.3) == [notes[1]]


def test_drop_quiet_with_no_floor_is_the_identity() -> None:
    notes = [n(0.0, 40, 0.01)]
    assert drop_quiet(notes, 0.0) == notes


def test_an_octave_above_a_louder_note_at_the_same_time_is_dropped() -> None:
    root, ghost = n(1.0, 45, 0.7), n(1.03, 57, 0.5)
    assert drop_octave_ghosts([root, ghost]) == [root]


def test_an_octave_louder_than_the_note_below_is_kept() -> None:
    # A played octave shape: the upper note can be the louder one.
    root, upper = n(1.0, 45, 0.4), n(1.0, 57, 0.6)
    assert drop_octave_ghosts([root, upper]) == [root, upper]


def test_an_octave_that_starts_later_than_50_ms_is_kept() -> None:
    root, later = n(1.0, 45, 0.7), n(1.06, 57, 0.5)
    assert drop_octave_ghosts([root, later]) == [root, later]


def test_only_exactly_twelve_semitones_counts() -> None:
    root, fifth, two_octaves = n(1.0, 45, 0.7), n(1.0, 52, 0.5), n(1.0, 69, 0.5)
    assert drop_octave_ghosts([root, fifth, two_octaves]) == [root, fifth, two_octaves]


def test_the_order_of_the_notes_is_kept() -> None:
    a, b, ghost = n(2.0, 50, 0.6), n(0.0, 40, 0.6), n(0.0, 52, 0.5)
    assert drop_octave_ghosts([a, b, ghost]) == [a, b]

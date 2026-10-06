"""Filters on the transcriber's notes, applied before grouping (plan 2026-10-06-optimise-current,
Task 3). Pure: notes in, notes out, order kept.

On electric guitar Basic Pitch writes notes nobody played -- 566 of 1,951 on Guitar-TECHS's player
3 at its chosen thresholds, 109 of them an octave from a played note. Whether either filter is
used is decided by measurement, not here.
"""

from __future__ import annotations

from collections.abc import Sequence

from tabsampler.types import NoteEvent

#: Two onsets this close start together: E2's onset tolerance.
TOGETHER_S = 0.05


def drop_quiet(notes: Sequence[NoteEvent], floor: float) -> list[NoteEvent]:
    """The notes whose ``confidence`` (Basic Pitch's amplitude) is at least ``floor``."""
    return [note for note in notes if note.confidence >= floor]


def drop_octave_ghosts(notes: Sequence[NoteEvent]) -> list[NoteEvent]:
    """Without notes exactly an octave above a note at least as loud that starts with them.

    A string's second harmonic is an octave above it, and a transcriber can hear it as a note.
    A played octave whose upper note is the louder one is kept.
    """
    return [
        note
        for note in notes
        if not any(
            other.pitch == note.pitch - 12
            and abs(other.onset - note.onset) <= TOGETHER_S
            and other.confidence >= note.confidence
            for other in notes
        )
    ]

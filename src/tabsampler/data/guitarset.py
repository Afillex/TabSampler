"""GuitarSet reference extraction (spec 3.3).

GUARDED AREA -- this is data loading for evaluation. What these functions return *is*
the definition of "correct" for E1 and E2. GuitarSet is test-only (ADR 0003).

Two reference views, because E1 and E2 need different things (ADR 0006):

- :func:`reference_tab` and :func:`reference_notes` round mirdata's float MIDI values
  to integers, because a (string, fret) pair can only sound an integer pitch and E2
  requires exact pitch equality.
- :func:`reference_note_arrays` keeps the fractional value and returns Hz, because
  ``mir_eval`` compares pitch *ratios* against a 50-cent tolerance and can absorb
  vibrato that rounding would throw away.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import NDArray

from tabsampler.errors import ReferenceInconsistencyError
from tabsampler.types import NoteEvent, Position, Tuning

if TYPE_CHECKING:
    from mirdata.core import Dataset

#: mirdata builds ``notes[_GUITAR_STRINGS[i]]`` from ``load_notes(jams_path, i)`` with
#: ``_GUITAR_STRINGS = ["E", "A", "D", "G", "B", "e"]``. Dict index i therefore *is*
#: string index i, and 0 is the low E -- the same convention as ``Position.string``.
GUITAR_STRINGS = ("E", "A", "D", "G", "B", "e")

#: Only these two archives are needed. The hex-pickup archives are several GB and are
#: forbidden as transcriber input anyway (ADR 0005), since one channel per string is
#: the E2 ground truth.
REQUIRED_REMOTES = ("annotations", "audio_mic")


class NoteAnnotation(Protocol):
    """Structural view of ``mirdata.annotations.NoteData``."""

    intervals: NDArray[np.float64]
    pitches: NDArray[np.float64]


class GuitarSetTrack(Protocol):
    """Structural view of the parts of mirdata's GuitarSet Track that we use."""

    track_id: str
    # Values are Optional: mirdata's NoteData rejects empty arrays, so a string with
    # no annotated notes comes through as None rather than an empty annotation.
    notes: dict[str, NoteAnnotation | None]
    notes_all: NoteAnnotation | None


def load_dataset(data_home: Path | str) -> Dataset:
    """Initialise the GuitarSet loader rooted at ``data_home``.

    The single place the untyped mirdata entry point is called, so the ``Dataset``
    annotation here is what the rest of the package sees.
    """
    import mirdata

    dataset = mirdata.initialize(  # pyright: ignore[reportUnknownMemberType]
        "guitarset", data_home=str(data_home)
    )
    return dataset


def _iter_string_notes(
    track: GuitarSetTrack,
) -> list[tuple[int, float, float, float]]:
    """(string_index, onset, offset, float_midi) for every annotated note."""
    out: list[tuple[int, float, float, float]] = []
    for index, name in enumerate(GUITAR_STRINGS):
        ann = track.notes.get(name)
        if ann is None:
            continue
        intervals = np.asarray(ann.intervals, dtype=float).reshape(-1, 2)
        pitches = np.asarray(ann.pitches, dtype=float).reshape(-1)
        for (onset, offset), pitch in zip(intervals, pitches, strict=True):
            out.append((index, float(onset), float(offset), float(pitch)))
    return out


def reference_tab(
    track: GuitarSetTrack, tuning: Tuning = Tuning.STANDARD
) -> list[tuple[NoteEvent, Position]]:
    """The reference fingering: every note with the string it was actually played on.

    This is the ground truth for E2 (Exact Tab F1). Sorted by (onset, string) so the
    output is deterministic.

    Raises:
        ReferenceInconsistencyError: if an annotated pitch cannot be produced on the
            string it is annotated against. Raised rather than dropping the note,
            because silently discarding unplaceable reference notes inflates precision.
    """
    placed: list[tuple[NoteEvent, Position]] = []

    for string, onset, offset, raw_pitch in _iter_string_notes(track):
        pitch = int(np.rint(raw_pitch))  # ADR 0006: round, never truncate
        fret = tuning.fret_for(string, pitch)
        if fret is None:
            raise ReferenceInconsistencyError(
                f"track {track.track_id}: pitch {pitch} (annotated {raw_pitch:.3f}) "
                f"is not reachable on string {string} "
                f"(open {tuning.open_pitches[string]}, capo {tuning.capo}, "
                f"frets 0-{tuning.max_fret})"
            )
        note = NoteEvent(onset=onset, offset=offset, pitch=pitch, confidence=1.0)
        placed.append((note, Position(string=string, fret=fret)))

    placed.sort(key=lambda pair: (pair[0].onset, pair[1].string))
    return placed


def reference_notes(track: GuitarSetTrack) -> list[NoteEvent]:
    """Every reference note, ignoring which string it was played on.

    Prefers ``notes_all`` and falls back to merging the per-string annotations.
    Pitches are rounded to integers (ADR 0006); for E1 use
    :func:`reference_note_arrays` instead, which keeps the fractional value.
    """
    ann = track.notes_all
    if ann is not None:
        intervals = np.asarray(ann.intervals, dtype=float).reshape(-1, 2)
        pitches = np.asarray(ann.pitches, dtype=float).reshape(-1)
        rows = [
            (float(a), float(b), float(p)) for (a, b), p in zip(intervals, pitches, strict=True)
        ]
    else:
        rows = [(on, off, p) for _, on, off, p in _iter_string_notes(track)]

    notes = [
        NoteEvent(onset=on, offset=off, pitch=int(np.rint(p)), confidence=1.0)
        for on, off, p in rows
    ]
    notes.sort(key=lambda n: (n.onset, n.pitch))
    return notes


def reference_note_arrays(
    track: GuitarSetTrack,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Reference notes as ``(intervals_seconds, pitches_hz)`` for ``mir_eval``.

    Keeps the fractional MIDI value and converts to Hz, because mir_eval computes its
    pitch tolerance in cents from a frequency ratio (ADR 0006).
    """
    ann = track.notes_all
    if ann is not None:
        intervals = np.asarray(ann.intervals, dtype=float).reshape(-1, 2)
        midi = np.asarray(ann.pitches, dtype=float).reshape(-1)
    else:
        rows = _iter_string_notes(track)
        intervals = np.array([[on, off] for _, on, off, _ in rows], dtype=float).reshape(-1, 2)
        midi = np.array([p for _, _, _, p in rows], dtype=float)

    order = np.lexsort((midi, intervals[:, 0])) if len(midi) else np.array([], dtype=int)
    intervals = intervals[order]
    midi = midi[order]

    # librosa.midi_to_hz rather than a hand-rolled formula, so the A440 reference is
    # the library's and not ours to get wrong.
    import librosa

    hz = (
        np.asarray(librosa.midi_to_hz(midi), dtype=float) if len(midi) else np.zeros(0, dtype=float)
    )
    return intervals, hz

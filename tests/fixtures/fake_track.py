"""A stand-in for mirdata's GuitarSet Track, so tests never need the dataset.

The annotations themselves are real ``mirdata.annotations.NoteData`` objects rather
than fakes, so these tests exercise the actual annotation type the loader will see.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from mirdata.annotations import NoteData

#: mirdata builds notes[_GUITAR_STRINGS[i]] from load_notes(jams, i), with
#: _GUITAR_STRINGS = ["E", "A", "D", "G", "B", "e"]. So dict index i IS string index i,
#: and 0 is the low E -- matching Position.string in spec 2.1.
GUITAR_STRINGS = ("E", "A", "D", "G", "B", "e")

# (onset_s, offset_s, midi_pitch_as_float)
Triple = tuple[float, float, float]


@dataclass
class FakeGuitarSetTrack:
    track_id: str
    # mirdata's NoteData rejects empty arrays ("Object should not be empty, use None
    # instead"), so a string with no annotated notes is None rather than an empty
    # annotation. The loader must tolerate that.
    notes: dict[str, NoteData | None]
    notes_all: NoteData | None


def _note_data(triples: list[Triple]) -> NoteData:
    intervals = np.array([[s, e] for s, e, _ in triples], dtype=float).reshape(-1, 2)
    pitches = np.array([p for _, _, p in triples], dtype=float)
    return NoteData(intervals, "s", pitches, "midi")


def fake_track(
    notes_per_string: dict[str, list[Triple]],
    track_id: str = "00_BN1-129-Eb_comp",
) -> FakeGuitarSetTrack:
    """Build a track from a {string name: [(onset, offset, midi)]} mapping."""
    notes: dict[str, NoteData | None] = {
        name: (_note_data(triples) if triples else None)
        for name, triples in notes_per_string.items()
    }

    merged: list[Triple] = [t for triples in notes_per_string.values() for t in triples]
    merged.sort(key=lambda t: (t[0], t[2]))
    notes_all = _note_data(merged) if merged else None

    return FakeGuitarSetTrack(track_id=track_id, notes=notes, notes_all=notes_all)

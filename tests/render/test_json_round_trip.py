"""ADR 0013's document read back, so the page can send a tab to the exporters."""

from __future__ import annotations

import json

import pytest

from tabsampler.render.json_out import render_json, tab_from_dict, tab_to_dict
from tabsampler.types import NoteEvent, Position, TabNote, Tuning

TAB = [
    TabNote(
        note=NoteEvent(onset=0.1, offset=0.6, pitch=45, confidence=0.8),
        position=Position(1, 0),
        posterior=0.55,
        alternatives=((Position(0, 5), 0.45),),
    ),
    TabNote(
        note=NoteEvent(onset=0.7, offset=1.0, pitch=50, confidence=0.9),
        position=Position(2, 0),
        posterior=1.0,
    ),
]


def test_a_tab_survives_the_round_trip_through_json() -> None:
    tuning = Tuning(capo=2)
    tab = [
        TabNote(
            note=NoteEvent(
                onset=t.note.onset,
                offset=t.note.offset,
                pitch=t.note.pitch + 2,
                confidence=t.note.confidence,
            ),
            position=t.position,
            posterior=t.posterior,
            alternatives=t.alternatives,
        )
        for t in TAB
    ]
    doc = json.loads(render_json(tab, tuning))
    assert tab_from_dict(doc) == (tab, tuning)


def test_the_server_response_extras_are_ignored() -> None:
    doc = tab_to_dict(TAB) | {"degradation": {}, "metrics": {}, "uncertainty_threshold": 0.6}
    assert tab_from_dict(doc)[0] == TAB


def test_another_schema_version_is_refused() -> None:
    doc = tab_to_dict(TAB) | {"schema_version": 2}
    with pytest.raises(ValueError, match="schema_version"):
        tab_from_dict(doc)


@pytest.mark.parametrize(
    "broken",
    [
        {},
        {"schema_version": 1, "tuning": {"open_pitches": [40], "n_frets": 22, "capo": 0}},
        {"schema_version": 1, "tuning": "standard", "notes": []},
    ],
)
def test_a_malformed_document_raises_value_error(broken: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        tab_from_dict(broken)


def test_a_note_whose_fret_does_not_sound_its_pitch_is_refused() -> None:
    # The exporters write the pitch from the string and fret; a document that disagrees with
    # itself must not become a file that quietly plays something else.
    doc = tab_to_dict(TAB)
    doc["notes"][0]["fret"] = 3
    with pytest.raises(ValueError, match="does not sound"):
        tab_from_dict(doc)

"""MusicXML export (ADRs 0017, 0059)."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from tabsampler.render.grid import DISCLAIMER
from tabsampler.render.musicxml import render_musicxml
from tabsampler.types import NoteEvent, Position, TabNote, Tuning


def note(onset: float, offset: float, string: int, fret: int, tuning: Tuning) -> TabNote:
    return TabNote(
        note=NoteEvent(
            onset=onset, offset=offset, pitch=tuning.pitch_at(string, fret), confidence=0.9
        ),
        position=Position(string, fret),
        posterior=0.9,
    )


TAB = [
    note(0.0, 0.5, 0, 3, Tuning()),
    note(0.01, 0.5, 2, 2, Tuning()),
    note(1.75, 2.25, 5, 12, Tuning()),
]


def parse(data: bytes) -> ET.Element:
    return ET.fromstring(data)


def sounded(root: ET.Element) -> list[ET.Element]:
    """Notes that are struck: not rests, not tie continuations."""
    out = []
    for n in root.iter("note"):
        if n.find("rest") is not None:
            continue
        if any(t.get("type") == "stop" for t in n.findall("tie")):
            continue
        out.append(n)
    return out


def test_the_disclaimer_is_in_the_bytes_of_the_file() -> None:
    data = render_musicxml(TAB, Tuning())
    assert DISCLAIMER.encode() in data
    assert b"Rhythm is NOT transcribed" in data


def test_it_is_well_formed_partwise_musicxml() -> None:
    root = parse(render_musicxml(TAB, Tuning()))
    assert root.tag == "score-partwise"
    assert root.find("part-list/score-part") is not None
    assert root.find("part/measure/attributes/divisions").text == "32"


def test_string_and_fret_survive_the_round_trip() -> None:
    root = parse(render_musicxml(TAB, Tuning()))
    got = [
        (
            int(n.find("notations/technical/string").text),
            int(n.find("notations/technical/fret").text),
        )
        for n in sounded(root)
    ]
    # MusicXML counts strings from the highest: our string 0 (low E) is its string 6.
    assert got == [(6, 3), (4, 2), (1, 12)]


def test_the_pitch_written_is_the_pitch_sounded() -> None:
    steps = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    root = parse(render_musicxml(TAB, Tuning()))
    for n, t in zip(sounded(root), TAB, strict=True):
        p = n.find("pitch")
        alter = int(p.findtext("alter", "0"))
        midi = 12 * (int(p.findtext("octave")) + 1) + steps[p.findtext("step")] + alter
        assert midi == t.note.pitch


def test_a_chord_note_carries_the_chord_mark() -> None:
    root = parse(render_musicxml(TAB, Tuning()))
    first, second = sounded(root)[:2]
    assert first.find("chord") is None and second.find("chord") is not None


def test_a_note_across_the_bar_line_is_tied() -> None:
    root = parse(render_musicxml(TAB, Tuning()))
    ties = [t.get("type") for n in root.iter("note") for t in n.findall("tie")]
    assert ties.count("start") == ties.count("stop") >= 1


def test_every_measure_adds_up_to_four_quarters() -> None:
    root = parse(render_musicxml(TAB, Tuning()))
    for measure in root.iter("measure"):
        total = sum(
            int(n.findtext("duration")) for n in measure.iter("note") if n.find("chord") is None
        )
        assert total == 4 * 32


def test_a_capo_is_a_capo_and_frets_stay_relative() -> None:
    capo = Tuning(capo=2)
    root = parse(render_musicxml([note(0.0, 0.5, 1, 0, capo)], capo))
    assert root.find("part/measure/attributes/staff-details/capo").text == "2"
    (struck,) = sounded(root)
    assert struck.findtext("notations/technical/fret") == "0"
    assert struck.findtext("pitch/step") == "B"  # A2 + 2 semitones


def test_the_tempo_and_metre_are_stated() -> None:
    root = parse(render_musicxml(TAB, Tuning()))
    assert root.find("part/measure/direction/sound").get("tempo") == "120"
    assert root.findtext("part/measure/attributes/time/beats") == "4"


def test_the_tab_staff_has_one_line_per_string_tuned_as_ours() -> None:
    root = parse(render_musicxml(TAB, Tuning()))
    details = root.find("part/measure/attributes/staff-details")
    assert details.findtext("staff-lines") == "6"
    lowest = details.find("staff-tuning[@line='1']")
    assert (lowest.findtext("tuning-step"), lowest.findtext("tuning-octave")) == ("E", "2")


def test_an_empty_tab_exports_one_bar_of_rest() -> None:
    root = parse(render_musicxml([], Tuning()))
    assert len(root.findall("part/measure")) == 1
    assert sounded(root) == []

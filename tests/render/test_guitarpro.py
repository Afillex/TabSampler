"""Guitar Pro 5 export (ADRs 0017, 0059), read back with pyguitarpro."""

from __future__ import annotations

import io

import guitarpro

from tabsampler.render.grid import DISCLAIMER, UNITS_PER_BAR
from tabsampler.render.guitarpro import render_guitarpro
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


def read(data: bytes) -> guitarpro.Song:
    return guitarpro.parse(io.BytesIO(data))


def struck(song: guitarpro.Song) -> list[list[tuple[int, int]]]:
    """(GP string, fret) per struck beat, skipping rests and tie continuations."""
    out = []
    for measure in song.tracks[0].measures:
        for beat in measure.voices[0].beats:
            notes = [n for n in beat.notes if n.type == guitarpro.NoteType.normal]
            if notes:
                out.append(sorted((n.string, n.value) for n in notes))  # stored by string
    return out


def test_the_disclaimer_is_in_the_file() -> None:
    data = render_guitarpro(TAB, Tuning())
    assert b"Rhythm is NOT transcribed" in data
    song = read(data)
    assert "".join(song.notice) == DISCLAIMER
    first_beat = song.tracks[0].measures[0].voices[0].beats[0]
    assert "NOT transcribed" in first_beat.text


def test_string_and_fret_survive_the_round_trip() -> None:
    # Guitar Pro counts strings from the highest, as MusicXML does.
    assert struck(read(render_guitarpro(TAB, Tuning()))) == [[(4, 2), (6, 3)], [(1, 12)]]


def test_the_tuning_is_ours() -> None:
    strings = read(render_guitarpro(TAB, Tuning())).tracks[0].strings
    assert [(s.number, s.value) for s in strings] == [
        (1, 64),
        (2, 59),
        (3, 55),
        (4, 50),
        (5, 45),
        (6, 40),
    ]


def test_a_capo_is_a_capo_and_frets_stay_relative() -> None:
    # pyguitarpro's Track.offset is the capo ("Height of the capo", gp5.py's track reader).
    capo = Tuning(capo=2)
    song = read(render_guitarpro([note(0.0, 0.5, 1, 0, capo)], capo))
    assert song.tracks[0].offset == 2
    assert struck(song) == [[(5, 0)]]


def test_every_bar_adds_up_to_four_quarters() -> None:
    song = read(render_guitarpro(TAB, Tuning()))
    for measure in song.tracks[0].measures:
        total = sum(beat.duration.time for beat in measure.voices[0].beats)
        assert total == 4 * guitarpro.Duration.quarterTime
    assert len(song.tracks[0].measures) == 2  # the last note crosses into bar 2


def test_a_note_across_the_bar_line_is_tied() -> None:
    song = read(render_guitarpro(TAB, Tuning()))
    second_bar = song.tracks[0].measures[1].voices[0].beats[0]
    assert [n.type for n in second_bar.notes] == [guitarpro.NoteType.tie]


def test_the_tempo_is_declared() -> None:
    assert read(render_guitarpro(TAB, Tuning())).tempo == 120


def test_an_empty_tab_exports_one_bar_of_rest() -> None:
    song = read(render_guitarpro([], Tuning()))
    assert len(song.tracks[0].measures) == 1
    assert struck(song) == []
    assert UNITS_PER_BAR == 128

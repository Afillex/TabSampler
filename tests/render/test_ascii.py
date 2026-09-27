"""Tests for ASCII tab rendering (ADR 0013)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tabsampler.decode.forward_backward import decode
from tabsampler.fingering.candidates import group_notes
from tabsampler.fingering.costs import HandSetScorer
from tabsampler.render.ascii import render_ascii, render_ascii_with_legend
from tabsampler.render.json_out import render_json, tab_to_dict
from tabsampler.types import Context, NoteEvent, Position, TabNote, Tuning

STANDARD = Tuning.STANDARD


def tabnote(onset: float, string: int, fret: int, posterior: float = 0.9) -> TabNote:
    pitch = STANDARD.pitch_at(string, fret)
    return TabNote(
        note=NoteEvent(onset=onset, offset=onset + 0.2, pitch=pitch, confidence=1.0),
        position=Position(string, fret),
        posterior=posterior,
    )


def lines(text: str) -> list[str]:
    return text.splitlines()


# ------------------------------------------------------------------ layout


def test_six_lines_with_high_e_on_top_and_low_e_at_the_bottom() -> None:
    out = lines(render_ascii([tabnote(0.0, 0, 3)]))
    assert len(out) == 6
    assert out[0].startswith("e|")
    assert out[-1].startswith("E|")


def test_the_note_appears_on_its_own_string_and_nowhere_else() -> None:
    out = lines(render_ascii([tabnote(0.0, 0, 3)]))
    assert "3" in out[-1]  # low E is the bottom line
    assert all("3" not in line for line in out[:-1])


def test_every_line_has_the_same_length() -> None:
    tab = [tabnote(0.0, 0, 3), tabnote(0.5, 5, 12), tabnote(1.0, 2, 7)]
    assert len({len(line) for line in lines(render_ascii(tab))}) == 1


def test_spacing_is_proportional_to_onset_time() -> None:
    close = render_ascii([tabnote(0.0, 0, 3), tabnote(0.2, 1, 3)])
    far = render_ascii([tabnote(0.0, 0, 3), tabnote(2.0, 1, 3)])
    assert len(lines(far)[0]) > len(lines(close)[0])


def test_a_chord_renders_in_one_column_across_several_strings() -> None:
    tab = [tabnote(0.0, 0, 0), tabnote(0.0, 1, 2), tabnote(0.0, 2, 2)]
    out = lines(render_ascii(tab))
    # Every token sits at the same character offset.
    offsets = [line.index(tok) for line, tok in ((out[-1], "0"), (out[-2], "2"), (out[-3], "2"))]
    assert len(set(offsets)) == 1


def test_two_digit_frets_do_not_break_column_alignment() -> None:
    # Fret 12 is two characters wide. A naive renderer shifts every later column.
    tab = [tabnote(0.0, 0, 12), tabnote(0.0, 1, 3), tabnote(0.6, 2, 5)]
    assert len({len(line) for line in lines(render_ascii(tab))}) == 1


def test_an_uncertain_note_is_parenthesised() -> None:
    out = render_ascii([tabnote(0.0, 0, 5, posterior=0.2)], uncertainty_threshold=0.6)
    assert "(5)" in out


def test_a_confident_note_is_not_parenthesised() -> None:
    out = render_ascii([tabnote(0.0, 0, 5, posterior=0.95)], uncertainty_threshold=0.6)
    assert "(5)" not in out
    assert "5" in out


def test_parenthesised_markers_keep_the_lines_the_same_length() -> None:
    tab = [tabnote(0.0, 0, 11, posterior=0.1), tabnote(0.0, 1, 3, posterior=0.99)]
    assert len({len(line) for line in lines(render_ascii(tab))}) == 1


def test_an_empty_tab_renders_six_empty_strings_rather_than_raising() -> None:
    out = lines(render_ascii([]))
    assert len(out) == 6
    assert all(set(line[2:-1]) <= {"-"} for line in out)


def test_a_non_standard_tuning_gets_the_right_number_of_lines() -> None:
    seven = Tuning(open_pitches=(35, 40, 45, 50, 55, 59, 64))
    assert len(lines(render_ascii([], tuning=seven))) == 7


def test_a_non_positive_seconds_per_char_is_refused() -> None:
    with pytest.raises(ValueError):
        render_ascii([tabnote(0.0, 0, 3)], seconds_per_char=0.0)


def test_the_legend_states_the_v1_limitations() -> None:
    out = render_ascii_with_legend([tabnote(0.0, 0, 3, posterior=0.2)])
    assert "proportional to time" in out
    assert "no bar lines" in out
    assert "1 of 1" in out


def test_rendering_is_deterministic() -> None:
    tab = [tabnote(0.0, 0, 3), tabnote(0.4, 2, 7)]
    assert render_ascii(tab) == render_ascii(tab)


# ------------------------------------------------------------------ JSON


def test_json_carries_posteriors_and_alternatives() -> None:
    tab = [
        TabNote(
            note=NoteEvent(onset=0.0, offset=0.2, pitch=52, confidence=0.8),
            position=Position(2, 2),
            posterior=0.7,
            alternatives=((Position(1, 7), 0.3),),
        )
    ]
    doc = tab_to_dict(tab)
    (note,) = doc["notes"]
    assert note["posterior"] == 0.7
    assert note["alternatives"] == [{"string": 1, "fret": 7, "posterior": 0.3}]
    assert note["transcriber_confidence"] == 0.8


def test_json_records_the_tuning_and_that_there_is_no_rhythm() -> None:
    doc = tab_to_dict([], Tuning.STANDARD)
    assert doc["tuning"]["open_pitches"] == [40, 45, 50, 55, 59, 64]
    assert doc["rhythm"] == "time_positioned_only"


def test_json_is_valid_and_byte_stable() -> None:
    tab = [tabnote(0.0, 0, 3), tabnote(0.5, 2, 7)]
    first = render_json(tab)
    assert json.loads(first)["schema_version"] == 1
    assert first == render_json(tab)


def test_json_round_trips_positions_and_pitches() -> None:
    tab = [tabnote(0.0, 0, 3), tabnote(0.5, 5, 12)]
    doc = json.loads(render_json(tab))
    got = [(n["string"], n["fret"], n["pitch"]) for n in doc["notes"]]
    assert got == [(0, 3, 43), (5, 12, 76)]


# ------------------------------------------------------------------ regression


GOLDEN = Path(__file__).parent.parent / "fixtures" / "golden_clip.txt"


def golden_tab() -> list[TabNote]:
    """A fixed ascending phrase, decoded with the committed baseline settings."""
    pitches = [52, 55, 57, 59, 60, 64, 67, 72]
    notes = [
        NoteEvent(onset=i * 0.3, offset=i * 0.3 + 0.25, pitch=p, confidence=0.9)
        for i, p in enumerate(pitches)
    ]
    ctx = Context(tuning=STANDARD, max_span=4)
    return decode(group_notes(notes), HandSetScorer(), ctx)


def test_golden_clip_output_is_unchanged() -> None:
    # A fixed input whose output must not change unless we intend it to. If this fails,
    # either the cost model or the decoder changed, and you should know which.
    expected = GOLDEN.read_text()
    assert render_ascii(golden_tab()) == expected


def test_the_golden_clip_is_pitch_valid() -> None:
    from tabsampler.eval.metrics import pitch_validity_rate, tab_notes_to_placed

    assert pitch_validity_rate(tab_notes_to_placed(golden_tab()), STANDARD) == 1.0

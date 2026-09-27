"""Tests for candidate generation and note grouping."""

from __future__ import annotations

from dataclasses import replace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tabsampler.fingering.candidates import (
    DEFAULT_GROUP_WINDOW_S,
    candidates,
    group_notes,
)
from tabsampler.types import NoteEvent, Position, Tuning


def n(onset: float, pitch: int, dur: float = 0.4) -> NoteEvent:
    return NoteEvent(onset=onset, offset=onset + dur, pitch=pitch, confidence=1.0)


# ------------------------------------------------------------------ candidates


def test_midi_64_has_exactly_the_expected_positions() -> None:
    # MIDI 64 (high E) in standard tuning, 22 frets:
    #   string 5 (e,  open 64) -> fret 0
    #   string 4 (B,  open 59) -> fret 5
    #   string 3 (G,  open 55) -> fret 9
    #   string 2 (D,  open 50) -> fret 14
    #   string 1 (A,  open 45) -> fret 19
    #   string 0 (E,  open 40) -> fret 24 -> past 22, excluded
    assert candidates(64, Tuning.STANDARD) == (
        Position(1, 19),
        Position(2, 14),
        Position(3, 9),
        Position(4, 5),
        Position(5, 0),
    )


def test_candidates_are_sorted_by_string_for_determinism() -> None:
    got = candidates(52, Tuning.STANDARD)
    assert list(got) == sorted(got, key=lambda p: (p.string, p.fret))


@given(pitch=st.integers(min_value=0, max_value=127))
def test_every_candidate_reproduces_the_requested_pitch(pitch: int) -> None:
    # The invariant E4 later measures across the whole pipeline.
    for p in candidates(pitch, Tuning.STANDARD):
        assert Tuning.STANDARD.pitch_at(p.string, p.fret) == pitch


def test_pitch_below_the_lowest_open_string_has_no_candidates() -> None:
    assert candidates(39, Tuning.STANDARD) == ()


def test_pitch_above_the_highest_fret_has_no_candidates() -> None:
    # Highest reachable in standard tuning with 22 frets is high e + 22 = 86.
    assert candidates(86, Tuning.STANDARD) != ()
    assert candidates(87, Tuning.STANDARD) == ()


def test_capo_removes_positions_below_the_capo() -> None:
    capo5 = replace(Tuning.STANDARD, capo=5)
    # MIDI 40 is the open low E, which a capo at fret 5 makes unreachable.
    assert candidates(40, capo5) == ()
    # MIDI 45 is now the capo'd low E at relative fret 0.
    assert Position(0, 0) in candidates(45, capo5)


def test_capo_shortens_the_reachable_neck() -> None:
    capo2 = replace(Tuning.STANDARD, capo=2)
    assert capo2.max_fret == 20
    assert candidates(86, capo2) == (Position(5, 20),)
    assert candidates(87, capo2) == ()


def test_a_drop_d_tuning_changes_the_candidate_set() -> None:
    drop_d = Tuning(open_pitches=(38, 45, 50, 55, 59, 64))
    assert Position(0, 0) in candidates(38, drop_d)
    assert candidates(38, Tuning.STANDARD) == ()


def test_a_seven_string_tuning_is_supported_without_special_casing() -> None:
    seven = Tuning(open_pitches=(35, 40, 45, 50, 55, 59, 64))
    assert Position(0, 0) in candidates(35, seven)


# ------------------------------------------------------------------ grouping


def test_notes_inside_the_window_group_together() -> None:
    groups = group_notes([n(0.0, 40), n(0.01, 47), n(0.02, 52)], window_s=0.03)
    assert len(groups) == 1
    assert [note.pitch for note in groups[0].notes] == [40, 47, 52]


def test_notes_outside_the_window_do_not_group() -> None:
    groups = group_notes([n(0.0, 40), n(0.5, 47)], window_s=0.03)
    assert len(groups) == 2


def test_a_note_exactly_at_the_window_edge_joins_the_group() -> None:
    # Inclusive, matching the mir_eval convention used by the metrics.
    groups = group_notes([n(0.0, 40), n(0.03, 47)], window_s=0.03)
    assert len(groups) == 1


def test_a_note_just_past_the_window_starts_a_new_group() -> None:
    groups = group_notes([n(0.0, 40), n(0.0301, 47)], window_s=0.03)
    assert len(groups) == 2


def test_a_long_chain_of_close_notes_does_not_collapse_into_one_group() -> None:
    # Notes 20 ms apart for a full second. Transitive chaining would swallow all 50
    # into a single "chord"; windows anchored at the first onset of each group must not.
    notes = [n(i * 0.02, 40 + (i % 3)) for i in range(50)]
    groups = group_notes(notes, window_s=0.03)
    assert len(groups) > 10
    assert all(len(g) <= 2 for g in groups)


def test_grouping_is_not_order_dependent() -> None:
    notes = [n(0.5, 52), n(0.0, 40), n(0.01, 47)]
    a = group_notes(notes, window_s=0.03)
    b = group_notes(list(reversed(notes)), window_s=0.03)
    assert a == b


def test_groups_are_returned_in_time_order() -> None:
    groups = group_notes([n(1.0, 40), n(0.0, 45), n(0.5, 50)], window_s=0.03)
    assert [g.onset for g in groups] == [0.0, 0.5, 1.0]


def test_grouping_an_empty_list_gives_no_groups() -> None:
    assert group_notes([], window_s=0.03) == []


def test_default_window_is_thirty_milliseconds() -> None:
    assert DEFAULT_GROUP_WINDOW_S == 0.03


def test_a_non_positive_window_is_refused() -> None:
    with pytest.raises(ValueError):
        group_notes([n(0.0, 40)], window_s=0.0)


def test_duplicate_pitches_at_the_same_onset_are_kept_as_separate_notes() -> None:
    # Two strings can sound the same pitch (a unison). The group must keep both so
    # state enumeration can decide whether it is fingerable at all.
    groups = group_notes([n(0.0, 52), n(0.0, 52)], window_s=0.03)
    assert len(groups) == 1
    assert len(groups[0]) == 2

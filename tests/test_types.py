"""Tests for the data contracts in spec 2.1.

These pin the invariants everything downstream relies on: that a (string, fret) pair
can only ever mean one pitch, and that the contracts are genuinely immutable.
"""

from dataclasses import FrozenInstanceError, replace

import pytest

from tabsampler.types import (
    ChordState,
    Context,
    CostWeights,
    NoteEvent,
    NoteGroup,
    Position,
    TabNote,
    Tuning,
    assert_state_matches_group,
)


def note(onset: float = 0.0, pitch: int = 40, offset: float | None = None) -> NoteEvent:
    return NoteEvent(
        onset=onset,
        offset=onset + 0.5 if offset is None else offset,
        pitch=pitch,
        confidence=1.0,
    )


# --------------------------------------------------------------------------- immutability


def test_frozen_dataclasses_reject_mutation() -> None:
    p = Position(string=0, fret=3)
    with pytest.raises(FrozenInstanceError):
        p.fret = 4  # type: ignore[misc]


def test_contracts_are_hashable_so_they_can_be_used_as_dict_keys() -> None:
    # ADR 0007: collections in the contracts are tuples, not lists, so "frozen"
    # actually means immutable. A list field would make these unhashable.
    assert len({Position(0, 3), Position(0, 3), Position(1, 3)}) == 2
    assert len({note(pitch=40), note(pitch=40)}) == 1


def test_bend_is_a_tuple_not_a_list() -> None:
    n = NoteEvent(onset=0.0, offset=0.1, pitch=40, confidence=1.0, bend=(0.0, 0.5))
    assert isinstance(n.bend, tuple)
    with pytest.raises(TypeError):
        NoteEvent(onset=0.0, offset=0.1, pitch=40, confidence=1.0, bend=[0.0, 0.5])  # type: ignore[arg-type]


def test_tabnote_alternatives_is_a_tuple() -> None:
    t = TabNote(
        note=note(),
        position=Position(0, 0),
        posterior=0.9,
        alternatives=((Position(1, 5), 0.1),),
    )
    assert isinstance(t.alternatives, tuple)


# --------------------------------------------------------------------------- pitch maths


def test_standard_tuning_pitches() -> None:
    assert Tuning.STANDARD.open_pitches == (40, 45, 50, 55, 59, 64)
    assert Tuning.STANDARD.pitch_at(string=0, fret=0) == 40  # low E
    assert Tuning.STANDARD.pitch_at(string=5, fret=0) == 64  # high e
    assert Tuning.STANDARD.pitch_at(string=1, fret=5) == 50  # A string, 5th fret == D
    assert Tuning.STANDARD.pitch_at(string=2, fret=12) == 62  # octave above open D


def test_capo_shifts_every_pitch_and_fret_is_relative_to_capo() -> None:
    capo2 = replace(Tuning.STANDARD, capo=2)
    assert capo2.pitch_at(string=0, fret=0) == 42
    assert capo2.pitch_at(string=0, fret=3) == 45


def test_highest_fret_shrinks_by_the_capo_position() -> None:
    assert Tuning.STANDARD.max_fret == 22
    assert replace(Tuning.STANDARD, capo=2).max_fret == 20


def test_pitch_at_rejects_a_string_or_fret_out_of_range() -> None:
    with pytest.raises(IndexError):
        Tuning.STANDARD.pitch_at(string=6, fret=0)
    with pytest.raises(ValueError):
        Tuning.STANDARD.pitch_at(string=0, fret=23)
    with pytest.raises(ValueError):
        Tuning.STANDARD.pitch_at(string=0, fret=-1)


def test_tuning_rejects_a_capo_beyond_the_neck() -> None:
    with pytest.raises(ValueError):
        Tuning(n_frets=22, capo=23)


# --------------------------------------------------------------------------- validation


def test_note_event_rejects_an_offset_before_its_onset() -> None:
    with pytest.raises(ValueError):
        NoteEvent(onset=1.0, offset=0.5, pitch=40, confidence=1.0)


def test_note_event_rejects_a_confidence_outside_zero_one() -> None:
    with pytest.raises(ValueError):
        NoteEvent(onset=0.0, offset=0.1, pitch=40, confidence=1.5)


def test_note_event_rejects_a_pitch_outside_midi_range() -> None:
    with pytest.raises(ValueError):
        NoteEvent(onset=0.0, offset=0.1, pitch=128, confidence=1.0)


def test_position_rejects_a_negative_string_or_fret() -> None:
    with pytest.raises(ValueError):
        Position(string=-1, fret=0)
    with pytest.raises(ValueError):
        Position(string=0, fret=-1)


# --------------------------------------------------------------------------- groups/states


def test_notegroup_onset_is_the_earliest_member_onset() -> None:
    g = NoteGroup.of([note(onset=0.02, pitch=52), note(onset=0.01, pitch=40)])
    assert g.onset == pytest.approx(0.01)


def test_notegroup_ordering_is_deterministic() -> None:
    a = NoteGroup.of([note(onset=0.0, pitch=52), note(onset=0.0, pitch=40)])
    b = NoteGroup.of([note(onset=0.0, pitch=40), note(onset=0.0, pitch=52)])
    assert a == b
    assert [n.pitch for n in a.notes] == [40, 52]


def test_notegroup_rejects_being_empty() -> None:
    with pytest.raises(ValueError):
        NoteGroup.of([])


def test_notegroup_constructor_rejects_unsorted_notes() -> None:
    # NoteGroup.of sorts; the constructor requires already-sorted input so that a
    # group built by hand cannot silently differ from one built by grouping.
    with pytest.raises(ValueError):
        NoteGroup(notes=(note(pitch=52), note(pitch=40)))


def test_chordstate_rejects_two_notes_on_one_string() -> None:
    with pytest.raises(ValueError):
        ChordState(positions=(Position(0, 3), Position(0, 5)))


def test_chordstate_rejects_being_empty() -> None:
    with pytest.raises(ValueError):
        ChordState(positions=())


def test_chordstate_length_must_match_its_group() -> None:
    g = NoteGroup.of([note(pitch=40), note(pitch=45)])
    assert_state_matches_group(g, ChordState(positions=(Position(0, 0), Position(1, 0))))
    with pytest.raises(ValueError):
        assert_state_matches_group(g, ChordState(positions=(Position(0, 0),)))


def test_fretted_positions_excludes_open_strings() -> None:
    s = ChordState(positions=(Position(0, 0), Position(1, 5), Position(2, 7)))
    assert s.fretted_frets == (5, 7)
    assert s.span == 2


def test_span_of_an_all_open_state_is_zero() -> None:
    s = ChordState(positions=(Position(0, 0), Position(1, 0)))
    assert s.fretted_frets == ()
    assert s.span == 0


def test_hand_position_is_the_lowest_fretted_fret_and_none_when_all_open() -> None:
    assert ChordState(positions=(Position(0, 7), Position(1, 5))).hand_position == 5
    assert ChordState(positions=(Position(0, 0),)).hand_position is None


# --------------------------------------------------------------------------- context


def test_context_defaults_are_usable_without_arguments() -> None:
    ctx = Context(tuning=Tuning.STANDARD)
    assert ctx.max_span == 4
    assert ctx.weights == CostWeights()


def test_cost_weights_acoustic_term_defaults_to_zero_until_phase_3() -> None:
    # The acoustic term exists in the contract from day one so the interface stays
    # stable, but it contributes nothing until Phase 3 supplies audio evidence.
    assert CostWeights().acoustic == 0.0

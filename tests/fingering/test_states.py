"""Tests for chord-state enumeration (ADR 0010)."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from tabsampler.fingering.candidates import candidates
from tabsampler.fingering.states import enumerate_states, state_count_stats
from tabsampler.types import Context, NoteEvent, NoteGroup, Tuning

STANDARD = Tuning.STANDARD


def n(pitch: int, onset: float = 0.0) -> NoteEvent:
    return NoteEvent(onset=onset, offset=onset + 0.4, pitch=pitch, confidence=1.0)


def group(*pitches: int) -> NoteGroup:
    return NoteGroup.of([n(p) for p in pitches])


def ctx(max_span: int = 4) -> Context:
    return Context(tuning=STANDARD, max_span=max_span)


def test_a_single_note_gives_one_state_per_candidate() -> None:
    states = enumerate_states(group(64), STANDARD, max_span=4)
    assert len(states) == len(candidates(64, STANDARD))
    assert all(len(s) == 1 for s in states)


def test_no_state_assigns_two_notes_to_one_string() -> None:
    for state in enumerate_states(group(40, 45, 50), STANDARD, max_span=4):
        strings = [p.string for p in state.positions]
        assert len(set(strings)) == len(strings)


@given(
    pitches=st.lists(st.integers(min_value=40, max_value=76), min_size=1, max_size=4, unique=True)
)
def test_every_state_reproduces_every_input_pitch(pitches: list[int]) -> None:
    g = group(*pitches)
    for state in enumerate_states(g, STANDARD, max_span=4):
        for note, pos in zip(g.notes, state.positions, strict=True):
            assert STANDARD.pitch_at(pos.string, pos.fret) == note.pitch


def test_positions_are_parallel_to_the_group_notes() -> None:
    g = group(40, 47)
    for state in enumerate_states(g, STANDARD, max_span=4):
        assert len(state.positions) == len(g.notes)


def test_span_pruning_removes_wide_shapes() -> None:
    narrow = enumerate_states(group(40, 45, 50), STANDARD, max_span=4)
    assert all(s.span <= 4 for s in narrow)
    wider = enumerate_states(group(40, 45, 50), STANDARD, max_span=12)
    assert len(wider) >= len(narrow)


def test_open_strings_do_not_count_towards_the_span() -> None:
    # Open low E (40) plus D string fret 12 (62) plus G string fret 12 (67).
    # Fretted frets are 12 and 12, so span 0 -- not 12.
    states = enumerate_states(group(40, 62, 67), STANDARD, max_span=1)
    assert states != ()
    assert any(s.span == 0 for s in states)


def test_an_unfingerable_group_yields_no_states_without_raising() -> None:
    # Seven simultaneous notes on a six-string guitar.
    g = group(40, 41, 42, 43, 44, 45, 46)
    assert enumerate_states(g, STANDARD, max_span=12) == ()


def test_a_unison_needing_two_strings_is_handled() -> None:
    # The same pitch twice must use two different strings, which is possible for 52.
    g = NoteGroup.of([n(52), n(52)])
    states = enumerate_states(g, STANDARD, max_span=12)
    assert states != ()
    for s in states:
        assert s.positions[0].string != s.positions[1].string


def test_a_unison_with_only_one_possible_string_is_unfingerable() -> None:
    # MIDI 86 is reachable only at high e fret 22, so it cannot sound twice.
    assert candidates(86, STANDARD) == (candidates(86, STANDARD)[0],)
    g = NoteGroup.of([n(86), n(86)])
    assert enumerate_states(g, STANDARD, max_span=12) == ()


def test_a_note_with_no_candidates_makes_the_whole_group_unfingerable() -> None:
    # 39 is below the open low E, so no assignment of the group can exist.
    g = group(39, 45)
    assert enumerate_states(g, STANDARD, max_span=12) == ()


def test_enumeration_is_deterministic_and_sorted() -> None:
    a = enumerate_states(group(40, 47), STANDARD, max_span=4)
    b = enumerate_states(group(40, 47), STANDARD, max_span=4)
    assert a == b
    keys = [tuple((p.string, p.fret) for p in s.positions) for s in a]
    assert keys == sorted(keys)


def test_max_span_zero_allows_only_single_fret_and_open_shapes() -> None:
    for state in enumerate_states(group(40, 45, 50), STANDARD, max_span=0):
        assert state.span == 0


def test_state_count_stays_bounded_for_a_dense_six_note_chord() -> None:
    # Guards spec 7's chord-state explosion risk with a number rather than a hope.
    # An E-shape barre chord: E A D G B e.
    g = group(40, 47, 52, 56, 59, 64)
    states = enumerate_states(g, STANDARD, max_span=4)
    assert len(states) <= 64


def test_stats_report_the_state_count_per_group() -> None:
    groups = [group(40), group(40, 47), group(40, 47, 52)]
    stats = state_count_stats(groups, STANDARD, max_span=4)
    assert stats.n_groups == 3
    assert stats.max_states >= stats.mean_states > 0
    assert stats.n_unfingerable == 0


def test_stats_count_unfingerable_groups_separately() -> None:
    groups = [group(40), group(39, 45)]
    stats = state_count_stats(groups, STANDARD, max_span=4)
    assert stats.n_unfingerable == 1


def test_stats_on_no_groups_do_not_divide_by_zero() -> None:
    stats = state_count_stats([], STANDARD, max_span=4)
    assert stats.n_groups == 0
    assert stats.mean_states == 0.0


def test_context_max_span_is_what_callers_pass_through() -> None:
    assert ctx(5).max_span == 5


# ------------------------------------------------------------------ the hand window (ADR 0025)

from tabsampler.fingering.states import HAND_WINDOW, shift_window  # noqa: E402


def test_the_first_fretted_shape_places_the_hand_at_its_lowest_fret() -> None:
    assert shift_window(None, (5, 7)) == (5, 0)


def test_notes_inside_the_window_do_not_move_the_hand() -> None:
    # Fret 5 -> 7 -> 9 is a finger reaching inside one position, not the hand moving.
    assert shift_window(5, (7,)) == (5, 0)
    assert shift_window(5, (9,)) == (5, 0)


def test_a_note_above_the_window_moves_it_up_just_enough() -> None:
    assert HAND_WINDOW == 4
    assert shift_window(5, (11,)) == (7, 2)


def test_a_note_below_the_window_moves_it_down_to_that_note() -> None:
    assert shift_window(5, (3,)) == (3, 2)


def test_an_all_open_shape_leaves_the_hand_where_it_was() -> None:
    assert shift_window(5, ()) == (5, 0)
    assert shift_window(None, ()) == (None, 0)


def test_a_shape_wider_than_the_window_anchors_at_its_lowest_fret() -> None:
    # A 5-fret span at fret 12 is legal (ADR 0011) but wider than 4: anchor low, never negative.
    assert shift_window(2, (12, 17)) == (12, 10)
    assert shift_window(14, (12, 17)) == (12, 2)


def test_the_distance_is_never_negative() -> None:
    for prev in (None, 0, 3, 9, 20):
        for fretted in ((), (1,), (4, 8), (10, 15), (19,)):
            _, moved = shift_window(prev, fretted)
            assert moved >= 0

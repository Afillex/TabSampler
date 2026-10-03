"""Tests for E3 playability (ADR 0011)."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tabsampler.eval.playability import (
    PlayabilityRules,
    fingers_needed,
    group_is_playable,
    hand_move_is_playable,
    judge_transitions,
    playability_rate,
)
from tabsampler.types import NoteEvent, Position, TabNote

RULES = PlayabilityRules()


def pos(*pairs: tuple[int, int]) -> tuple[Position, ...]:
    return tuple(Position(s, f) for s, f in pairs)


def tabnote(onset: float, string: int, fret: int, pitch: int = 52) -> TabNote:
    return TabNote(
        note=NoteEvent(onset=onset, offset=onset + 0.2, pitch=pitch, confidence=1.0),
        position=Position(string, fret),
        posterior=0.9,
    )


# ------------------------------------------------------------------ group rules


def test_a_comfortable_shape_passes() -> None:
    assert group_is_playable(pos((0, 3), (1, 5), (2, 5)), RULES) is None


def test_a_six_fret_stretch_below_fret_twelve_fails() -> None:
    reason = group_is_playable(pos((0, 2), (1, 8)), RULES)
    assert reason is not None
    assert "span 6 exceeds 4" in reason


def test_a_five_fret_stretch_at_fret_twelve_passes() -> None:
    # ADR 0011: frets narrow up the neck, so the limit rises from 4 to 5 at fret 12.
    assert group_is_playable(pos((0, 12), (1, 17)), RULES) is None


def test_the_same_five_fret_stretch_low_down_fails() -> None:
    assert group_is_playable(pos((0, 2), (1, 7)), RULES) is not None


def test_open_strings_do_not_count_towards_the_span() -> None:
    # Open low E plus frets 12 and 13: fretted span is 1, not 13.
    assert group_is_playable(pos((0, 0), (1, 12), (2, 13)), RULES) is None


def test_an_all_open_shape_passes() -> None:
    assert group_is_playable(pos((0, 0), (1, 0), (2, 0)), RULES) is None


def test_two_notes_on_one_string_fails() -> None:
    reason = group_is_playable(pos((0, 3), (0, 5)), RULES)
    assert reason is not None
    assert "one string" in reason


def test_a_full_barre_chord_is_playable() -> None:
    # E-shape barre at fret 5: notes at 5,7,7,6,5,5 -> distinct frets {5,6,7} = 3 fingers.
    shape = pos((0, 5), (1, 7), (2, 7), (3, 6), (4, 5), (5, 5))
    assert group_is_playable(shape, RULES) is None


def test_five_distinct_fretted_frets_is_still_impossible() -> None:
    # Five different frets needs five fingers; a hand has four.
    shape = pos((0, 5), (1, 6), (2, 7), (3, 8), (4, 9))
    reason = group_is_playable(shape, RULES)
    assert reason is not None and "fingers" in reason


def test_a_barre_above_the_lowest_fret_does_not_get_the_discount() -> None:
    # Review Focus 5: two notes at fret 9 cannot be barred while fret 5 is held below them.
    shape = pos((0, 5), (1, 9), (2, 9), (3, 7), (4, 6))
    assert group_is_playable(shape, RULES) is not None


def test_barre_modelling_can_be_switched_off() -> None:
    strict = PlayabilityRules(allow_barre=False)
    shape = pos((0, 5), (1, 7), (2, 7), (3, 6), (4, 5), (5, 5))
    assert group_is_playable(shape, strict) is not None


def test_fingers_needed_barres_the_lowest_fret_and_charges_a_finger_above_it() -> None:
    # One finger covers every string at the *lowest* fretted fret, however many that is.
    # Every note above it needs a finger of its own, even if two share a fret: the other
    # fingers are already committed and cannot also lie flat across strings.
    assert fingers_needed(pos((0, 5), (1, 5), (2, 5), (3, 5)), RULES) == 1
    assert fingers_needed(pos((0, 0), (1, 0)), RULES) == 0
    assert fingers_needed(pos((0, 5), (1, 7), (2, 7), (3, 6)), RULES) == 4  # 1 barre + 3
    # Without the barre discount it is one finger per fretted note.
    strict = PlayabilityRules(allow_barre=False)
    assert fingers_needed(pos((0, 5), (1, 5), (2, 5), (3, 5)), strict) == 4


def test_modelling_barres_can_only_ever_relax_the_finger_count() -> None:
    # Whatever the shape, allowing a barre must not make E3 stricter, or the metric
    # would move for a reason the change did not claim.
    strict = PlayabilityRules(allow_barre=False)
    shapes = (
        pos((0, 5), (1, 7), (2, 7), (3, 6), (4, 5), (5, 5)),
        pos((0, 0), (1, 12), (2, 13)),
        pos(
            (0, 3),
        ),
        pos((0, 5), (1, 9), (2, 9), (3, 7), (4, 6)),
        pos((0, 0), (1, 0), (2, 0)),
    )
    for shape in shapes:
        assert fingers_needed(shape, RULES) <= fingers_needed(shape, strict)


def test_a_barre_cannot_reach_across_a_string_that_must_ring_open() -> None:
    # The barre finger lies flat across a contiguous run of strings, so it would press
    # any string inside that run. An open string strictly between two lowest-fret notes
    # therefore rules the barre out: frets 5 and 5 on the low E and D strings cannot be
    # one finger while the A string rings open. That shape needs five fingers, not four.
    shape = pos((0, 5), (1, 0), (2, 5), (3, 6), (4, 7), (5, 8))
    assert fingers_needed(shape, RULES) == 5
    reason = group_is_playable(shape, RULES)
    assert reason is not None and "fingers" in reason


def test_a_barre_still_reaches_across_strings_fretted_higher_up() -> None:
    # The E-shape barre: strings 1, 2 and 3 sit inside the barred run at frets 7, 7 and 6.
    # The barre presses them too, but each sounds at its own higher fret, which is exactly
    # how the chord is played. Only an *open* string inside the run is a conflict.
    shape = pos((0, 5), (1, 7), (2, 7), (3, 6), (4, 5), (5, 5))
    assert fingers_needed(shape, RULES) == 4
    assert group_is_playable(shape, RULES) is None


def test_a_barre_may_span_strings_that_are_not_sounded_at_all() -> None:
    # Nothing is played on strings 1-4, so a barre across them presses only silence.
    shape = pos((0, 5), (5, 5))
    assert fingers_needed(shape, RULES) == 1


def test_an_open_string_outside_the_barred_run_is_not_a_conflict() -> None:
    # The open high e is above both lowest-fret notes, so the barre never reaches it.
    shape = pos((0, 5), (1, 5), (5, 0))
    assert fingers_needed(shape, RULES) == 1


def test_a_barre_plus_four_fingers_across_nine_frets_is_refused() -> None:
    # Review Focus 5 again, the shape the relaxation must not let through: a barre at
    # fret 5 plus notes at 9, 10, 11, 14 is five distinct frets and a 9-fret stretch.
    shape = pos((0, 5), (1, 5), (2, 9), (3, 10), (4, 11), (5, 14))
    assert group_is_playable(shape, RULES) is not None


def test_rules_are_data_and_can_be_relaxed() -> None:
    loose = PlayabilityRules(max_span_low=8)
    assert group_is_playable(pos((0, 2), (1, 8)), RULES) is not None
    assert group_is_playable(pos((0, 2), (1, 8)), loose) is None


# ------------------------------------------------------------------ transition rules


def test_a_reach_inside_one_position_is_not_a_hand_move() -> None:
    # 5 -> 7 in a sixteenth note used to read as 16 frets/s (ADR 0022). Now: no move.
    reason, hand = hand_move_is_playable((5, 9), pos((0, 7)), 0.125, RULES)
    assert reason is None and hand == (5, 9)


def test_a_slow_long_jump_passes() -> None:
    reason, hand = hand_move_is_playable((2, 6), pos((1, 12)), 2.0, RULES)
    assert reason is None and hand == (8, 12)


def test_the_same_jump_played_fast_fails() -> None:
    reason, _ = hand_move_is_playable((2, 6), pos((1, 12)), 0.1, RULES)
    assert reason is not None and "frets/s" in reason


def test_a_jump_with_no_time_between_groups_fails() -> None:
    reason, _ = hand_move_is_playable((2, 6), pos((1, 12)), 0.0, RULES)
    assert reason is not None


def test_no_move_with_no_time_between_groups_passes() -> None:
    # Distance is judged before time: a reach inside the window needs no time at all.
    reason, hand = hand_move_is_playable((5, 9), pos((0, 7)), 0.0, RULES)
    assert reason is None and hand == (5, 9)


def test_a_jump_across_an_open_string_is_still_a_jump() -> None:
    # fret 2 -> open -> fret 20 in 0.1 s: the open string must not reset the hand.
    tab = [tabnote(0.00, 0, 2), tabnote(0.05, 1, 0), tabnote(0.10, 0, 20)]
    report = playability_rate(tab, RULES)
    assert (report.n_transitions_pass, report.n_transitions) == (1, 2)


def test_a_move_across_an_open_string_is_timed_from_the_last_fretted_note() -> None:
    # fret 2 -> open -> fret 10 at 0.25 s steps: the hand had 0.5 s, not 0.25 s (ADR 0029).
    tab = [tabnote(0.0, 0, 2), tabnote(0.25, 1, 0), tabnote(0.5, 0, 10)]
    assert playability_rate(tab, RULES).n_transitions_pass == 2


def test_a_piece_that_starts_with_open_strings_has_no_moves_to_time() -> None:
    shapes = [(0.0, pos((1, 0))), (0.1, pos((2, 0))), (0.2, pos((0, 5)))]
    assert judge_transitions(shapes, RULES) == [None, None]


@given(
    a=st.integers(1, 15),
    b=st.integers(1, 20),
    gap=st.floats(0.02, 1.0),
    cuts=st.lists(st.floats(0.01, 0.99), min_size=1, max_size=3),
)
def test_open_strings_between_two_fretted_shapes_never_change_the_verdict(
    a: int, b: int, gap: float, cuts: list[float]
) -> None:
    first, last = (0.0, pos((0, a))), (gap, pos((1, b)))
    opens = [(gap * c, pos((2, 0))) for c in sorted(cuts)]
    plain = judge_transitions([first, last], RULES)[-1]
    assert judge_transitions([first, *opens, last], RULES)[-1] == plain


def test_a_shape_with_two_notes_on_one_string_still_moves_the_hand() -> None:
    # Frets 3 and 12 cannot fit one rest window: the index goes to 3, stretched to 12.
    reason, hand = hand_move_is_playable((5, 9), pos((0, 3), (0, 12)), 5.0, RULES)
    assert hand == (3, 12) and reason is None


# ------------------------------------------------------------------ whole tab


def test_a_playable_passage_scores_one() -> None:
    tab = [tabnote(i * 0.25, string=1, fret=5 + (i % 2)) for i in range(6)]
    report = playability_rate(tab, RULES)
    assert report.group_rate == 1.0
    assert report.transition_rate == 1.0
    assert report.overall_rate == 1.0
    assert report.failures == ()


def test_groups_and_transitions_are_reported_separately() -> None:
    # A single conflated rate would hide which rule failed.
    tab = [tabnote(0.0, 0, 2), tabnote(0.0, 1, 8), tabnote(0.05, 2, 3)]
    report = playability_rate(tab, RULES)
    assert report.n_groups == 2
    assert report.n_transitions == 1
    assert report.group_rate != report.transition_rate


def test_failures_name_what_went_wrong_and_when() -> None:
    tab = [tabnote(0.0, 0, 2), tabnote(0.0, 1, 8)]
    report = playability_rate(tab, RULES)
    assert len(report.failures) == 1
    assert "group @0.000s" in report.failures[0]


def test_e3_needs_no_reference_tab() -> None:
    # The whole point: this call passes nothing but our own output.
    assert playability_rate([tabnote(0.0, 0, 3)], RULES).overall_rate == 1.0


def test_a_single_group_has_no_transitions_and_does_not_divide_by_zero() -> None:
    report = playability_rate([tabnote(0.0, 0, 3)], RULES)
    assert report.n_transitions == 0
    assert report.transition_rate == 1.0


def test_an_empty_tab_is_vacuously_playable() -> None:
    report = playability_rate([], RULES)
    assert report.n_groups == 0
    assert report.overall_rate == 1.0


def test_simultaneous_notes_group_into_one_shape() -> None:
    tab = [tabnote(0.0, 0, 3, pitch=43), tabnote(0.01, 1, 5, pitch=50)]
    assert playability_rate(tab, RULES).n_groups == 1


def test_a_tab_placing_two_notes_on_one_string_is_reported_not_crashed() -> None:
    # ChordState refuses to hold this, but E3 must still be able to score a tab
    # that contains it -- including one from somewhere other than our decoder.
    tab = [tabnote(0.0, 0, 3, pitch=43), tabnote(0.0, 0, 5, pitch=45)]
    report = playability_rate(tab, RULES)
    assert report.group_rate == 0.0
    assert "one string" in report.failures[0]


def test_the_default_rules_are_adr_0011s_numbers() -> None:
    assert (RULES.max_span_low, RULES.max_span_high, RULES.high_neck_fret) == (4, 5, 12)
    assert RULES.max_frets_per_second == 12.0
    # max_fingers replaces ADR 0011's max_fretted_notes; barres are modelled (ADR 0019).
    assert RULES.max_fingers == 4
    assert RULES.allow_barre is True


def test_max_span_at_switches_at_the_high_neck_fret() -> None:
    assert RULES.max_span_at(11) == 4
    assert RULES.max_span_at(12) == 5


def test_rate_properties_agree_with_the_counts() -> None:
    tab = [tabnote(0.0, 0, 2), tabnote(0.0, 1, 8), tabnote(0.5, 2, 3)]
    r = playability_rate(tab, RULES)
    assert r.group_rate == pytest.approx(r.n_groups_pass / r.n_groups)

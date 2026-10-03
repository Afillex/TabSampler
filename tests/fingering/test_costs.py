"""Tests for the hand-set cost model (spec 2.2, ADR 0012).

Costs are a modelling choice, so these test *ordering properties* -- which is where the
guitar knowledge lives -- rather than specific magic numbers. A test asserting
`emission == 0.37` would break on every reweighting and prove nothing.
"""

from __future__ import annotations

import math

import pytest

from tabsampler.fingering.costs import HandSetScorer
from tabsampler.types import (
    ChordState,
    Context,
    CostWeights,
    FingeringScorer,
    NoteEvent,
    NoteGroup,
    Position,
    Tuning,
)

STANDARD = Tuning.STANDARD


def n(pitch: int) -> NoteEvent:
    return NoteEvent(onset=0.0, offset=0.4, pitch=pitch, confidence=1.0)


def state(*pairs: tuple[int, int]) -> ChordState:
    return ChordState(positions=tuple(Position(s, f) for s, f in pairs))


def ctx(**kw: float | int) -> Context:
    max_span = int(kw.pop("max_span", 12))
    return Context(tuning=STANDARD, max_span=max_span, weights=CostWeights(**kw))  # type: ignore[arg-type]


def scorer(**kw: float) -> HandSetScorer:
    return HandSetScorer(weights=CostWeights(**kw))  # type: ignore[arg-type]


def test_it_satisfies_the_fingering_scorer_protocol() -> None:
    assert isinstance(scorer(), FingeringScorer)


# ------------------------------------------------------------------ emission ordering


def test_a_narrower_shape_costs_no_more_than_a_wider_one() -> None:
    s = scorer()
    g = NoteGroup.of([n(45), n(50)])
    narrow = s.emission_cost(g, state((0, 5), (1, 5)), ctx())
    wide = s.emission_cost(g, state((0, 5), (2, 12)), ctx())
    assert narrow <= wide


def test_a_low_neck_position_costs_no_more_than_the_same_shape_high_up() -> None:
    s = scorer()
    g = NoteGroup.of([n(43), n(48)])
    low = s.emission_cost(g, state((0, 3), (1, 3)), ctx())
    high = s.emission_cost(g, state((0, 15), (1, 15)), ctx())
    assert low <= high


def test_open_strings_are_rewarded_relative_to_the_same_pitch_fretted() -> None:
    # MIDI 64 is the open high e, or the B string at fret 5. Open should be cheaper.
    s = scorer()
    g = NoteGroup.of([n(64)])
    assert s.emission_cost(g, state((5, 0)), ctx()) < s.emission_cost(g, state((4, 5)), ctx())


def test_more_open_strings_is_cheaper_than_fewer() -> None:
    s = scorer()
    g = NoteGroup.of([n(45), n(50)])
    two_open = s.emission_cost(g, state((1, 0), (2, 0)), ctx())
    one_open = s.emission_cost(g, state((1, 0), (0, 10)), ctx())
    assert two_open < one_open


def test_an_all_open_shape_has_no_span_and_no_height_penalty() -> None:
    s = scorer()
    g = NoteGroup.of([n(40), n(45)])
    cost = s.emission_cost(g, state((0, 0), (1, 0)), ctx())
    # Only the open reward contributes, so the cost is negative.
    assert cost < 0.0


# ------------------------------------------------------------------ transition ordering


def test_staying_in_position_costs_less_than_jumping() -> None:
    s = scorer()
    stay = s.transition_cost(state((0, 5), (1, 5)), state((2, 5), (3, 6)))
    jump = s.transition_cost(state((0, 5), (1, 5)), state((2, 15), (3, 16)))
    assert stay < jump


def test_transition_cost_grows_with_distance() -> None:
    s = scorer()
    d1 = s.transition_cost(state((0, 5)), state((1, 6)))
    d5 = s.transition_cost(state((0, 5)), state((1, 10)))
    d10 = s.transition_cost(state((0, 5)), state((1, 15)))
    assert d1 < d5 < d10


def test_the_window_makes_up_and_down_moves_differ() -> None:
    # A hand is placed at its lowest fretted note and covers 4 frets above it (ADR 0025).
    # From fret 3 (frets 3-7), reaching fret 9 moves the window up 2, to 5-9. From fret 9
    # (frets 9-13), reaching fret 3 moves it down 6. The old point model was symmetric.
    s = scorer()
    a, b = state((0, 3)), state((1, 9))
    assert s.transition_cost(a, b) == pytest.approx(2.0 * s.weights.move)
    assert s.transition_cost(b, a) == pytest.approx(6.0 * s.weights.move)


def test_transition_to_or_from_an_all_open_state_is_free() -> None:
    # An all-open shape has no hand position of its own, so the *stateless* view of the
    # move is free. That is not the whole story: the decoder carries the hand across an
    # all-open shape and charges the movement, which is what transition_cost_from is
    # for. See ADR 0018 and the module docstring.
    s = scorer()
    assert s.transition_cost(state((0, 2)), state((1, 0))) == 0.0
    assert s.transition_cost(state((1, 0)), state((0, 10))) == 0.0
    assert s.transition_cost(state((1, 0)), state((2, 0))) == 0.0


def test_movement_is_charged_across_an_intervening_open_chord() -> None:
    s = scorer()
    # fret 2 -> all-open -> fret 10. A hand with its index on 2 covers frets 2-6, so reaching
    # fret 10 moves it 4, to 6-10 (ADR 0025); the open chord in between changes nothing.
    assert s.transition_cost_from((2, 6), state((1, 0))) == 0.0  # nothing to move to
    assert s.transition_cost_from((2, 6), state((0, 10))) == pytest.approx(4.0 * s.weights.move)
    # An all-open shape carries the previous hand forward rather than resetting it.
    assert s.carry(previous_hand=(2, 6), curr=state((1, 0))) == (2, 6)
    assert s.carry(previous_hand=(2, 6), curr=state((0, 10))) == (6, 10)
    assert s.carry(previous_hand=None, curr=state((1, 0))) is None


def test_a_reach_inside_the_window_costs_nothing() -> None:
    s = scorer()
    assert s.transition_cost_from((5, 9), state((0, 8))) == 0.0


def test_transition_cost_is_transition_cost_from_the_hand_the_previous_shape_leaves() -> None:
    # The stateless method stays on the FingeringScorer Protocol and must agree with the
    # hand-aware one, so the two cannot drift apart.
    from tabsampler.fingering.states import carry_hand

    s = scorer()
    for prev in (state((0, 2)), state((1, 0)), state((0, 5), (1, 7)), state((0, 12), (1, 17))):
        for curr in (state((2, 0)), state((0, 10)), state((3, 4), (4, 6)), state((1, 17))):
            expected = s.transition_cost_from(carry_hand(None, prev), curr)
            assert s.transition_cost(prev, curr) == expected


# ------------------------------------------------------------------ robustness


def test_costs_are_finite_for_every_legal_state() -> None:
    # A -inf or nan here would poison Viterbi and forward-backward silently.
    from tabsampler.fingering.states import enumerate_states

    s = scorer()
    g = NoteGroup.of([n(40), n(47), n(52)])
    states = enumerate_states(g, STANDARD, max_span=4)
    assert states != ()
    for st in states:
        assert math.isfinite(s.emission_cost(g, st, ctx(max_span=4)))
        for other in states:
            assert math.isfinite(s.transition_cost(st, other))


def test_every_weight_changes_the_cost() -> None:
    # Guards a dead lambda: a weight with no effect would make tuning meaningless.
    g = NoteGroup.of([n(40), n(52), n(62)])
    st = state((0, 0), (1, 7), (2, 12))  # one open, fretted 7 and 12 -> span 5
    base = CostWeights()
    reference = HandSetScorer(weights=base).emission_cost(g, st, ctx())

    for field, bumped in (
        ("span", CostWeights(span=base.span + 1.0)),
        ("high", CostWeights(high=base.high + 1.0)),
        ("open_reward", CostWeights(open_reward=base.open_reward + 1.0)),
    ):
        other = HandSetScorer(weights=bumped).emission_cost(g, st, ctx())
        assert other != pytest.approx(reference), f"weight {field} has no effect"

    moved = HandSetScorer(weights=CostWeights(move=base.move + 1.0))
    assert moved.transition_cost(state((0, 3)), state((1, 9))) != pytest.approx(
        HandSetScorer(weights=base).transition_cost(state((0, 3)), state((1, 9)))
    )


def test_the_acoustic_weight_contributes_nothing_before_phase_3() -> None:
    g = NoteGroup.of([n(40)])
    st = state((0, 0))
    a = HandSetScorer(weights=CostWeights(acoustic=0.0)).emission_cost(g, st, ctx())
    b = HandSetScorer(weights=CostWeights(acoustic=5.0)).emission_cost(g, st, ctx())
    assert a == pytest.approx(b)


def test_a_state_that_does_not_match_its_group_is_refused() -> None:
    s = scorer()
    g = NoteGroup.of([n(40), n(45)])
    with pytest.raises(ValueError):
        s.emission_cost(g, state((0, 0)), ctx())

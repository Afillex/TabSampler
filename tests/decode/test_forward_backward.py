"""Tests for forward-backward posteriors.

This is where log space earns its keep. The oracle here computes marginals in plain
probability space by enumerating paths, which underflows on anything long -- exactly the
failure the implementation must not have.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings

from tabsampler.decode.forward_backward import (
    _cost_arrays,
    decode,
    forward_backward,
    log_partition,
    note_posteriors,
)
from tabsampler.decode.viterbi import (
    LatticeNode,
    build_lattice,
    node_transition_cost,
    viterbi,
)
from tabsampler.errors import UnfingerableGroupError
from tabsampler.fingering.costs import HandSetScorer
from tabsampler.types import (
    ChordState,
    Context,
    CostWeights,
    NoteEvent,
    NoteGroup,
    Position,
    Tuning,
)

from .brute_force import brute_force_marginals
from .strategies import group_sequence

STANDARD = Tuning.STANDARD
CTX = Context(tuning=STANDARD, max_span=4)
SCORER = HandSetScorer()

SLOW = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


def n(pitch: int, onset: float = 0.0) -> NoteEvent:
    return NoteEvent(onset=onset, offset=onset + 0.4, pitch=pitch, confidence=1.0)


def group(*pitches: int, onset: float = 0.0) -> NoteGroup:
    return NoteGroup.of([n(p, onset) for p in pitches])


# ============================================================ normalisation


@SLOW
@given(groups=group_sequence(max_groups=5, max_notes=2))
def test_state_posteriors_sum_to_one_at_every_timestep(groups: list[NoteGroup]) -> None:
    for level in forward_backward(groups, SCORER, CTX):
        assert float(level.sum()) == pytest.approx(1.0, abs=1e-9)


def test_a_node_transition_that_would_teleport_the_hand_costs_infinity() -> None:
    # The guard the node lattice rests on. A node's carried hand must be exactly what its
    # predecessor's hand becomes after playing that shape; without the check an all-open
    # shape could inherit a hand position no path ever put it in, and forward-backward
    # would give that fiction weight. build_lattice never emits such a node, so the pair
    # is hand-built here -- which is the only way to exercise the check directly.
    open_shape = ChordState(positions=(Position(5, 0),))
    prior = LatticeNode(ChordState(positions=(Position(0, 5),)), (5, 9))
    inherits = LatticeNode(open_shape, (5, 9))  # carries the hand it was handed: legal
    teleports = LatticeNode(open_shape, (11, 15))  # a hand nothing put there
    assert node_transition_cost(prior, inherits, SCORER) == 0.0
    assert node_transition_cost(prior, teleports, SCORER) == math.inf
    # A note outside the window (frets 5-9) moves it just enough: fret 12 -> frets 8-12.
    fretted = LatticeNode(ChordState(positions=(Position(0, 12),)), (8, 12))
    assert node_transition_cost(prior, fretted, SCORER) == pytest.approx(3.0)
    # ... and a node claiming the old point-model hand (index on 12) is unreachable here.
    point = LatticeNode(ChordState(positions=(Position(0, 12),)), (12, 16))
    assert node_transition_cost(prior, point, SCORER) == math.inf


def test_infinite_transitions_do_not_poison_the_posteriors() -> None:
    # An all-open group between two fretted ones puts math.inf into the transition matrix,
    # so -inf appears inside the logsumexp. If that were mishandled the posteriors would
    # come back nan rather than a distribution. MIDI 64 is the open high e, so level 1
    # carries one all-open node per hand position reachable from level 0.
    groups = [group(52, onset=0.0), group(64, onset=0.5), group(80, onset=1.0)]
    transitions = _cost_arrays(groups, build_lattice(groups, CTX), SCORER, CTX)[1]
    assert any(bool(np.isinf(matrix).any()) for matrix in transitions), "no forbidden moves"
    for level in forward_backward(groups, SCORER, CTX):
        assert bool(np.all(np.isfinite(level)))
        assert float(level.sum()) == pytest.approx(1.0, abs=1e-9)


@SLOW
@given(groups=group_sequence(max_groups=4, max_notes=2))
def test_note_position_posteriors_sum_to_one_per_note(groups: list[NoteGroup]) -> None:
    for per_group in note_posteriors(groups, SCORER, CTX):
        for per_note in per_group:
            assert sum(per_note.values()) == pytest.approx(1.0, abs=1e-9)


def test_posteriors_are_all_non_negative() -> None:
    groups = [group(45, 52), group(60, onset=0.5)]
    for level in forward_backward(groups, SCORER, CTX):
        assert bool(np.all(level >= 0.0))


# ============================================================ the oracle tests


@pytest.mark.oracle
@SLOW
@given(groups=group_sequence(max_groups=5, max_notes=3))
def test_marginals_equal_brute_force_marginals(groups: list[NoteGroup]) -> None:
    # The definitive correctness test: log-space forward-backward against a plain
    # sum over every path. A lattice node is a shape plus the hand carried into it
    # (ADR 0018), so a *shape's* marginal is the sum over the nodes holding it.
    expected = brute_force_marginals(groups, SCORER, CTX, temperature=1.0)
    got = forward_backward(groups, SCORER, CTX, temperature=1.0)
    lattice = build_lattice(groups, CTX)

    for nodes, level_got, level_expected in zip(lattice, got, expected, strict=True):
        per_state: dict[ChordState, float] = {}
        for node, weight in zip(nodes, level_got, strict=True):
            per_state[node.state] = per_state.get(node.state, 0.0) + float(weight)
        for state in set(per_state) | set(level_expected):
            assert per_state.get(state, 0.0) == pytest.approx(
                level_expected.get(state, 0.0), abs=1e-9
            )


@pytest.mark.oracle
@SLOW
@given(groups=group_sequence(max_groups=4, max_notes=3))
def test_marginals_equal_brute_force_with_open_strings_up_the_neck_charged(
    groups: list[NoteGroup],
) -> None:
    # ADR 0039's term rides on the move into a shape; the oracle charges it independently.
    from tabsampler.types import CostWeights

    scorer = HandSetScorer(weights=CostWeights(open_up_neck=0.7))
    expected = brute_force_marginals(groups, scorer, CTX, temperature=1.0)
    got = forward_backward(groups, scorer, CTX, temperature=1.0)
    for nodes, level_got, level_expected in zip(
        build_lattice(groups, CTX), got, expected, strict=True
    ):
        per_state: dict[ChordState, float] = {}
        for node, weight in zip(nodes, level_got, strict=True):
            per_state[node.state] = per_state.get(node.state, 0.0) + float(weight)
        for state in set(per_state) | set(level_expected):
            assert per_state.get(state, 0.0) == pytest.approx(
                level_expected.get(state, 0.0), abs=1e-9
            )


@pytest.mark.oracle
@SLOW
@given(groups=group_sequence(max_groups=4, max_notes=2))
def test_log_partition_matches_a_direct_sum_over_paths(groups: list[NoteGroup]) -> None:
    import math

    from .brute_force import all_paths, path_cost

    direct = sum(math.exp(-path_cost(groups, p, SCORER, CTX) / 1.0) for p in all_paths(groups, CTX))
    assert log_partition(groups, SCORER, CTX, temperature=1.0) == pytest.approx(
        math.log(direct), abs=1e-9
    )


@pytest.mark.oracle
@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ],
)
@given(groups=group_sequence(max_groups=4, max_notes=2))
def test_log_partition_matches_a_direct_sum_when_shapes_are_wider_than_the_hand(
    groups: list[NoteGroup],
) -> None:
    # As above, at span 6: some shapes do not fit in the 4-fret hand (ADR 0025).
    from tabsampler.fingering.states import HAND_WINDOW, enumerate_states

    from .brute_force import all_paths, path_cost

    wide = Context(tuning=STANDARD, max_span=6)
    assume(
        any(
            state.span > HAND_WINDOW
            for g in groups
            for state in enumerate_states(g, STANDARD, wide.max_span)
        )
    )
    direct = sum(math.exp(-path_cost(groups, p, SCORER, wide)) for p in all_paths(groups, wide))
    assert log_partition(groups, SCORER, wide, temperature=1.0) == pytest.approx(
        math.log(direct), abs=1e-9
    )


# ============================================================ numerics


def test_no_underflow_on_a_long_sequence() -> None:
    # 500 groups. A naive product of probabilities underflows to zero and every
    # posterior becomes nan. In log space they must stay finite and normalised.
    groups = [group(45 + (i % 12), onset=i * 0.2) for i in range(500)]
    levels = forward_backward(groups, SCORER, CTX)
    assert len(levels) == 500
    for level in levels:
        assert bool(np.all(np.isfinite(level)))
        assert float(level.sum()) == pytest.approx(1.0, abs=1e-8)


def test_a_very_low_temperature_does_not_produce_nan() -> None:
    groups = [group(45, 52), group(60, onset=0.5), group(64, onset=1.0)]
    levels = forward_backward(groups, SCORER, CTX, temperature=1e-3)
    for level in levels:
        assert bool(np.all(np.isfinite(level)))
        assert float(level.sum()) == pytest.approx(1.0, abs=1e-8)


def test_low_temperature_concentrates_posteriors_on_the_viterbi_path() -> None:
    # As T -> 0 the marginals must agree with the MAP path. Ties the two decoders
    # together, and would catch a sign error in either.
    groups = [group(52, onset=0.0), group(57, onset=0.5), group(62, onset=1.0)]
    path, _ = viterbi(groups, SCORER, CTX)
    lattice = build_lattice(groups, CTX)
    levels = forward_backward(groups, SCORER, CTX, temperature=0.01)

    for nodes, level, chosen in zip(lattice, levels, path, strict=True):
        assert nodes[int(np.argmax(level))].state == chosen


def test_high_temperature_approaches_uniform() -> None:
    groups = [group(52, onset=0.0), group(57, onset=0.5)]
    cold = forward_backward(groups, SCORER, CTX, temperature=0.05)
    hot = forward_backward(groups, SCORER, CTX, temperature=500.0)
    # Spread shrinks as temperature rises.
    assert float(hot[0].max() - hot[0].min()) < float(cold[0].max() - cold[0].min())


def test_temperature_defaults_to_the_context_weights() -> None:
    groups = [group(52), group(57, onset=0.5)]
    ctx = Context(tuning=STANDARD, max_span=4, weights=CostWeights(temperature=0.01))
    from_ctx = forward_backward(groups, SCORER, ctx)
    explicit = forward_backward(groups, SCORER, ctx, temperature=0.01)
    for a, b in zip(from_ctx, explicit, strict=True):
        np.testing.assert_allclose(a, b)


# ============================================================ decode -> TabNote


def test_decode_returns_one_tabnote_per_input_note() -> None:
    groups = [group(45, 52, onset=0.0), group(60, onset=0.5)]
    tab = decode(groups, SCORER, CTX)
    assert len(tab) == 3


def test_decoded_positions_come_from_the_viterbi_path() -> None:
    # The chosen position is the MAP joint assignment, not a per-note argmax, which
    # could otherwise pick a combination that is not a legal chord state at all.
    groups = [group(45, 52, onset=0.0), group(60, onset=0.5), group(64, onset=1.0)]
    path, _ = viterbi(groups, SCORER, CTX)
    tab = decode(groups, SCORER, CTX)
    expected = [p for state in path for p in state.positions]
    assert [t.position for t in tab] == expected


def test_decoded_pitch_always_equals_input_pitch() -> None:
    groups = [group(45, 52, onset=0.0), group(60, 64, onset=0.5)]
    for t in decode(groups, SCORER, CTX):
        assert STANDARD.pitch_at(t.position.string, t.position.fret) == t.note.pitch


def test_posteriors_are_probabilities() -> None:
    groups = [group(52, onset=0.0), group(57, onset=0.5)]
    for t in decode(groups, SCORER, CTX):
        assert 0.0 <= t.posterior <= 1.0


def test_a_note_with_a_single_candidate_has_posterior_one() -> None:
    # MIDI 86 is reachable only at high e fret 22.
    (t,) = decode([group(86)], SCORER, CTX)
    assert t.position == Position(5, 22)
    assert t.posterior == pytest.approx(1.0)
    assert t.alternatives == ()


def test_alternatives_exclude_the_chosen_position_and_descend() -> None:
    groups = [group(52, onset=0.0), group(57, onset=0.5)]
    for t in decode(groups, SCORER, CTX):
        assert t.position not in [p for p, _ in t.alternatives]
        probs = [p for _, p in t.alternatives]
        assert probs == sorted(probs, reverse=True)


def test_alternatives_can_be_capped() -> None:
    tab = decode([group(60)], SCORER, CTX, max_alternatives=1)
    assert all(len(t.alternatives) <= 1 for t in tab)


def test_posterior_and_alternatives_sum_to_one() -> None:
    for t in decode([group(60, onset=0.0)], SCORER, CTX):
        assert t.posterior + sum(p for _, p in t.alternatives) == pytest.approx(1.0)


# ============================================================ edges


def test_no_groups_gives_no_posteriors_and_no_tab() -> None:
    assert forward_backward([], SCORER, CTX) == []
    assert decode([], SCORER, CTX) == []


def test_an_unfingerable_group_raises_rather_than_producing_nan() -> None:
    # An all -inf level makes log Z = -inf and every posterior nan. Must fail loudly,
    # with the same error type Viterbi raises.
    with pytest.raises(UnfingerableGroupError):
        forward_backward([group(39)], SCORER, CTX)
    with pytest.raises(UnfingerableGroupError):
        decode([group(39)], SCORER, CTX)

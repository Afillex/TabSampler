"""Tests for fitting the cost weights by maximum likelihood (ADR 0023).

The fitter re-expresses the hand-set cost model as features times weights so a gradient
exists. Each test pins that re-expression to something already trusted: the scorer's own
costs, the oracle-verified ``log_partition``, a brute-force sum over every path, and a
finite-difference gradient.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings

from tabsampler.decode.forward_backward import log_partition
from tabsampler.fingering.costs import HandSetScorer
from tabsampler.fingering.fit import (
    FEATURE_GROUPS,
    WEIGHT_NAMES,
    HumanSequence,
    fit_weights,
    human_sequences,
    nll_and_gradient,
    path_features,
    sequence_features,
    weights_from_vector,
    weights_to_vector,
)
from tabsampler.fingering.states import enumerate_states
from tabsampler.types import (
    ChordState,
    Context,
    CostWeights,
    NoteEvent,
    NoteGroup,
    Position,
    Tuning,
)

from ..decode.brute_force import path_cost
from ..decode.strategies import group_sequence

STANDARD = Tuning.STANDARD
CTX = Context(tuning=STANDARD, max_span=4)

SLOW = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


def n(pitch: int, onset: float = 0.0) -> NoteEvent:
    return NoteEvent(onset=onset, offset=onset + 0.4, pitch=pitch, confidence=1.0)


def group(*pitches: int, onset: float = 0.0) -> NoteGroup:
    return NoteGroup.of([n(p, onset) for p in pitches])


def human(groups: list[NoteGroup]) -> HumanSequence:
    """A 'human' fingering: the first legal state of each group. Any legal path will do."""
    states = [enumerate_states(g, STANDARD, CTX.max_span)[0] for g in groups]
    return HumanSequence(
        groups=tuple(groups), states=tuple(states), spans=(CTX.max_span,) * len(groups)
    )


WEIGHTS = (
    CostWeights(),
    CostWeights(move=0.3, span=1.7, high=2.5, open_reward=-0.4),
    # Every feature group non-zero (ADR 0034); the low E's bias is the pinned reference.
    CostWeights(
        move=0.3,
        span=1.7,
        high=2.5,
        open_reward=-0.4,
        string_bias=(0.0, 0.2, -0.1, 0.4, 0.05, -0.3),
        low_region=0.6,
        high_region=-0.2,
    ),
)


# ------------------------------------------------------------------ re-expression is exact


@pytest.mark.parametrize("weights", WEIGHTS)
def test_features_times_weights_is_the_scorers_path_cost(weights: CostWeights) -> None:
    groups = [group(52, onset=0.0), group(64, onset=0.5), group(57, 62, onset=1.0)]
    seq = human(groups)
    expected = path_cost(groups, seq.states, HandSetScorer(weights=weights), CTX)
    got = float(weights_to_vector(weights) @ path_features(seq.states))
    assert got == pytest.approx(expected, abs=1e-9)


def test_the_weight_vector_round_trips() -> None:
    for w in WEIGHTS:
        assert weights_from_vector(weights_to_vector(w), temperature=w.temperature) == w
    assert WEIGHT_NAMES == (
        "move",
        "span",
        "high",
        "open_reward",
        "string_1",
        "string_2",
        "string_3",
        "string_4",
        "string_5",
        "low_region",
        "high_region",
    )


def test_the_low_e_string_is_the_pinned_reference() -> None:
    # Six string counts always sum to the group size, so one bias cannot be fitted.
    with pytest.raises(ValueError, match="low E"):
        weights_to_vector(CostWeights(string_bias=(0.5, 0.0, 0.0, 0.0, 0.0, 0.0)))


@pytest.mark.oracle
@SLOW
@given(groups=group_sequence(max_groups=5, max_notes=2))
def test_log_partition_matches_the_oracle_verified_decoder(groups: list[NoteGroup]) -> None:
    # The decoder's log_partition is itself checked against brute force (make oracle).
    for weights in WEIGHTS:
        feats = sequence_features(human(groups), CTX)
        nll, _ = nll_and_gradient(weights_to_vector(weights), [feats])
        observed_cost = float(weights_to_vector(weights) @ feats.observed)
        ctx = Context(tuning=STANDARD, max_span=CTX.max_span, weights=weights)
        log_z = log_partition(groups, HandSetScorer(weights=weights), ctx, temperature=1.0)
        assert nll == pytest.approx(observed_cost + log_z, abs=1e-8)


@pytest.mark.oracle
@SLOW
@given(groups=group_sequence(max_groups=4, max_notes=2))
def test_expected_features_match_a_brute_force_sum_over_paths(groups: list[NoteGroup]) -> None:
    # The gradient is observed minus expected features. Expected features by enumerating
    # every state path is obviously right and hopelessly slow, which is the point.
    weights = WEIGHTS[2]
    w = weights_to_vector(weights)
    feats = sequence_features(human(groups), CTX)
    _, gradient = nll_and_gradient(w, [feats])

    lattice = [enumerate_states(g, STANDARD, CTX.max_span) for g in groups]
    scorer = HandSetScorer(weights=weights)
    total = 0.0
    expected = np.zeros(len(WEIGHT_NAMES))
    for path in itertools.product(*lattice):
        weight = math.exp(-path_cost(groups, list(path), scorer, CTX))
        total += weight
        expected += weight * path_features(path)
    expected /= total
    assert gradient == pytest.approx(feats.observed - expected, abs=1e-8)


WIDE = Context(tuning=STANDARD, max_span=6)


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
def test_expected_features_match_brute_force_when_shapes_are_wider_than_the_hand(
    groups: list[NoteGroup],
) -> None:
    # As above, at span 6, where a shape can stretch the hand (ADR 0030): the fitter's
    # features must still agree with the oracle's independent reading of the ADR.
    lattice = [enumerate_states(g, STANDARD, WIDE.max_span) for g in groups]
    assume(any(state.span > 4 for level in lattice for state in level))
    weights = WEIGHTS[2]
    w = weights_to_vector(weights)
    states = tuple(level[0] for level in lattice)
    seq = HumanSequence(groups=tuple(groups), states=states, spans=(6,) * len(groups))
    feats = sequence_features(seq, WIDE)
    _, gradient = nll_and_gradient(w, [feats])

    scorer = HandSetScorer(weights=weights)
    total = 0.0
    expected = np.zeros(len(WEIGHT_NAMES))
    for path in itertools.product(*lattice):
        weight = math.exp(-path_cost(groups, list(path), scorer, WIDE))
        total += weight
        expected += weight * path_features(path)
    expected /= total
    assert gradient == pytest.approx(feats.observed - expected, abs=1e-8)


# ------------------------------------------------------------------ the optimisation


def test_the_gradient_matches_finite_differences() -> None:
    groups = [group(52, onset=0.0), group(64, onset=0.5), group(80, onset=1.0), group(55, 59)]
    feats = [sequence_features(human(groups), CTX)]
    w = weights_to_vector(WEIGHTS[2])
    _, gradient = nll_and_gradient(w, feats)
    step = 1e-6
    for i in range(len(WEIGHT_NAMES)):
        up, down = w.copy(), w.copy()
        up[i] += step
        down[i] -= step
        numeric = (nll_and_gradient(up, feats)[0] - nll_and_gradient(down, feats)[0]) / (2 * step)
        assert gradient[i] == pytest.approx(numeric, rel=1e-5, abs=1e-6)


def test_fitting_lowers_the_nll_and_lands_on_a_stationary_point() -> None:
    groups = [group(52 + (i % 7), onset=i * 0.5) for i in range(12)]
    feats = [sequence_features(human(groups), CTX)]
    start = weights_to_vector(CostWeights())
    result = fit_weights(feats, CostWeights())
    assert result.nll <= nll_and_gradient(start, feats)[0] + 1e-9
    _, gradient = nll_and_gradient(weights_to_vector(result.weights), feats)
    assert float(np.abs(gradient[:4]).max()) < 1e-3  # the four base weights were fitted


def test_fitting_a_subset_leaves_the_other_weights_untouched() -> None:
    # Started away from zero, so a fitter that reset the inactive weights would be caught.
    groups = [group(52 + (i % 7), onset=i * 0.5) for i in range(12)]
    feats = [sequence_features(human(groups), CTX)]
    initial = CostWeights(
        string_bias=(0.0, 0.3, -0.2, 0.1, 0.0, 0.4), low_region=0.25, high_region=-0.15
    )
    base_only = fit_weights(feats, initial, active=FEATURE_GROUPS["base"])
    assert base_only.weights.string_bias == initial.string_bias
    assert base_only.weights.low_region == initial.low_region
    assert base_only.weights.high_region == initial.high_region
    with_strings = fit_weights(
        feats, initial, active=FEATURE_GROUPS["base"] + FEATURE_GROUPS["string"]
    )
    assert with_strings.weights.string_bias != initial.string_bias
    assert with_strings.weights.low_region == initial.low_region
    assert with_strings.weights.high_region == initial.high_region


def test_a_human_state_missing_from_the_lattice_is_refused() -> None:
    # The likelihood of a path the lattice cannot express is zero, so its NLL is infinite.
    # That must fail loudly, not quietly drag every weight towards infinity.
    g = group(45, 52)
    # Low E fret 5 and D fret 2: a legal shape, but a 3-fret span cannot fit a 0-fret bound.
    wide = ChordState(positions=(Position(0, 5), Position(2, 2)))
    assert wide not in enumerate_states(g, STANDARD, 0)
    seq = HumanSequence(groups=(g,), states=(wide,), spans=(0,))
    with pytest.raises(ValueError, match="not in the lattice"):
        sequence_features(seq, CTX)


# ------------------------------------------------------------------ from human tab to sequences


def step(onset: float, *pairs: tuple[int, int]) -> tuple[NoteGroup, ChordState]:
    positions = sorted(
        (Position(s, f) for s, f in pairs),
        key=lambda p: STANDARD.pitch_at(p.string, p.fret) if p.fret <= STANDARD.max_fret else 999,
    )
    pitches = [STANDARD.open_pitches[p.string] + p.fret for p in positions]
    notes = tuple(
        NoteEvent(onset=onset, offset=onset + 0.2, pitch=x, confidence=1.0) for x in pitches
    )
    return NoteGroup(notes=notes), ChordState(positions=tuple(positions))


def test_a_chord_wider_than_the_bound_widens_only_its_own_group() -> None:
    ctx = Context(tuning=STANDARD, max_span=5)
    steps = [step(0.0, (0, 3)), step(0.5, (0, 2), (1, 8)), step(1.0, (1, 5))]  # middle span 6
    (seq,) = human_sequences(steps, ctx)
    assert seq.spans == (5, 6, 5)
    sequence_features(seq, ctx)  # the human path is in the lattice, so this does not raise


def test_a_group_the_lattice_cannot_express_splits_the_sequence() -> None:
    # Fret 23 is past a 22-fret neck, and span 9 is past the widest relaxation. Each is
    # dropped and the sequence split there, rather than chaining a hand movement across it.
    ctx = Context(tuning=STANDARD, max_span=5)
    steps = [
        step(0.0, (0, 3)),
        step(0.5, (5, 23)),
        step(1.0, (1, 5)),
        step(1.5, (2, 7)),
        step(2.0, (0, 1), (1, 10)),
        step(2.5, (3, 4)),
    ]
    sequences = human_sequences(steps, ctx, max_relaxed_span=8)
    assert [len(s.groups) for s in sequences] == [1, 2, 1]


def test_the_fitter_refuses_a_guitar_with_more_than_six_strings() -> None:
    # The per-string features are five biases against the low E: a six-string model.
    seven = Tuning(open_pitches=(35, 40, 45, 50, 55, 59, 64))
    g = group(67)
    on_seventh = next(s for s in enumerate_states(g, seven, 4) if s.positions[0].string == 6)
    sequence = HumanSequence(groups=(g,), states=(on_seventh,), spans=(4,))
    with pytest.raises(ValueError, match="six"):
        sequence_features(sequence, Context(tuning=seven, max_span=4))

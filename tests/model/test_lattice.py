"""The lattice as arrays for the learned model (ADR 0043).

Each test pins the arrays to something already trusted: the fitter's features, which the
brute-force oracle checks, and the oracle's own path cost.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings

from tabsampler.fingering.costs import HandSetScorer
from tabsampler.fingering.fit import (
    HumanSequence,
    path_features,
    sequence_features,
    weights_to_vector,
)
from tabsampler.fingering.states import enumerate_states
from tabsampler.model.lattice import (
    DESCRIPTOR_SIZE,
    GROUP_INPUT_SIZE,
    group_inputs,
    lattice_arrays,
)
from tabsampler.types import Context, CostWeights, NoteEvent, NoteGroup, Tuning

from ..decode.brute_force import path_cost
from ..decode.strategies import group_sequence

STANDARD = Tuning.STANDARD
CTX = Context(tuning=STANDARD, max_span=4)
WEIGHTS = CostWeights(move=0.3, span=1.7, high=2.5, open_reward=-0.4, open_up_neck=0.7)
SLOW = settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


def human(groups: list[NoteGroup], choice: int = 0) -> HumanSequence:
    """A 'human' fingering: one legal state per group. Any legal path will do."""
    states = []
    for g in groups:
        options = enumerate_states(g, STANDARD, CTX.max_span)
        states.append(options[choice % len(options)])
    return HumanSequence(tuple(groups), tuple(states), (CTX.max_span,) * len(groups))


@SLOW
@given(groups=group_sequence(max_groups=5, max_notes=2))
def test_the_arrays_hold_the_fitters_features(groups: list[NoteGroup]) -> None:
    seq = human(groups)
    arrays = lattice_arrays(seq.groups, CTX, seq.spans, seq.states)
    fitter = sequence_features(seq, CTX)
    for ours, theirs in zip(arrays.features, fitter.emission, strict=True):
        assert np.array_equal(ours, theirs)
    for ours, theirs in zip(arrays.movement, fitter.movement, strict=True):
        assert np.array_equal(ours, theirs)
    for ours, theirs in zip(arrays.allowed, fitter.allowed, strict=True):
        assert np.array_equal(ours, theirs)


@SLOW
@given(groups=group_sequence(max_groups=5, max_notes=2))
def test_the_human_nodes_cost_what_the_oracle_says_the_human_path_costs(
    groups: list[NoteGroup],
) -> None:
    seq = human(groups, choice=1)
    arrays = lattice_arrays(seq.groups, CTX, seq.spans, seq.states)
    assert arrays.human is not None
    w = weights_to_vector(WEIGHTS)
    energy = sum(float(arrays.features[t][j] @ w) for t, j in enumerate(arrays.human))
    for t in range(1, len(arrays.human)):
        previous, current = arrays.human[t - 1], arrays.human[t]
        assert arrays.allowed[t - 1][previous, current]
        energy += WEIGHTS.move * float(arrays.movement[t - 1][previous, current])
    assert energy == pytest.approx(float(w @ path_features(seq.states)), abs=1e-9)
    oracle = path_cost(list(seq.groups), seq.states, HandSetScorer(weights=WEIGHTS), CTX)
    assert energy == pytest.approx(oracle, abs=1e-9)
    for t, j in enumerate(arrays.human):
        assert arrays.states[t][j] == seq.states[t]


def test_without_a_human_path_there_are_no_human_nodes() -> None:
    groups = [NoteGroup.of([note(52)]), NoteGroup.of([note(55, onset=0.5)])]
    assert lattice_arrays(groups, CTX).human is None


def test_descriptors_and_group_inputs_have_their_documented_shapes() -> None:
    groups = [NoteGroup.of([note(52), note(59)]), NoteGroup.of([note(64, onset=0.5)])]
    arrays = lattice_arrays(groups, CTX)
    for nodes, descriptors in zip(arrays.states, arrays.descriptors, strict=True):
        assert descriptors.shape == (len(nodes), DESCRIPTOR_SIZE)
        assert descriptors.min() >= 0.0 and descriptors.max() <= 1.0
    assert arrays.group_inputs.shape == (2, GROUP_INPUT_SIZE)


def test_group_inputs_mark_each_pitch_and_the_time_since_the_last_group() -> None:
    first = NoteGroup.of([note(52), note(59)])
    second = NoteGroup.of([note(64, onset=0.5)])
    inputs = group_inputs([first, second])
    assert inputs[0, 52 - 28] == 1.0 and inputs[0, 59 - 28] == 1.0
    assert inputs[0, :73].sum() == 2.0 and inputs[1, :73].sum() == 1.0
    assert inputs[0, 74] == 0.0  # nothing came before the first group
    assert inputs[1, 74] == pytest.approx(np.log1p(0.5))


def test_a_pitch_outside_the_input_range_is_refused() -> None:
    with pytest.raises(ValueError, match="pitch"):
        group_inputs([NoteGroup.of([note(110)])])


def note(pitch: int, onset: float = 0.0) -> NoteEvent:
    return NoteEvent(onset=onset, offset=onset + 0.4, pitch=pitch, confidence=1.0)

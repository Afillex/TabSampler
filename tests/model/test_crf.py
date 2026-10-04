"""The CRF in torch (ADR 0043), pinned to what is already trusted.

With the learned term off, the torch log-likelihood and its gradient must be the fitter's,
which the brute-force oracle checks, and torch Viterbi's cost the decoder's. Float64 throughout,
so agreement can be asked to 1e-9.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from hypothesis import HealthCheck, given, settings

from tabsampler.decode.viterbi import viterbi as decoder_viterbi
from tabsampler.fingering.costs import HandSetScorer
from tabsampler.fingering.fit import (
    HumanSequence,
    nll_and_gradient,
    sequence_features,
    weights_to_vector,
)
from tabsampler.fingering.states import enumerate_states
from tabsampler.model.batch import make_batch
from tabsampler.model.crf import cost_energies, log_partition, path_energy, viterbi
from tabsampler.model.lattice import lattice_arrays
from tabsampler.types import Context, CostWeights, NoteEvent, NoteGroup, Tuning

from ..decode.strategies import group_sequence

STANDARD = Tuning.STANDARD
CTX = Context(tuning=STANDARD, max_span=4)
WEIGHTS = CostWeights(move=0.3, span=1.7, high=2.5, open_reward=-0.4, open_up_neck=0.7)
SLOW = settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


def human(groups: list[NoteGroup], choice: int) -> HumanSequence:
    states = []
    for g in groups:
        options = enumerate_states(g, STANDARD, CTX.max_span)
        states.append(options[choice % len(options)])
    return HumanSequence(tuple(groups), tuple(states), (CTX.max_span,) * len(groups))


def torch_nll(
    sequences: list[HumanSequence], weights: CostWeights
) -> tuple[torch.Tensor, torch.Tensor]:
    """The torch negative log-likelihood of each sequence, and the weights it was taken at."""
    items = [lattice_arrays(s.groups, CTX, s.spans, s.states) for s in sequences]
    batch = make_batch(items, dtype=torch.float64)
    w = torch.tensor(weights_to_vector(weights), dtype=torch.float64, requires_grad=True)
    energies, transitions = cost_energies(batch, w)
    nll = path_energy(energies, transitions, batch.lengths, batch.human) + log_partition(
        energies, transitions, batch.lengths
    )
    return nll, w


@SLOW
@given(
    first=group_sequence(max_groups=5, max_notes=2),
    second=group_sequence(max_groups=3, max_notes=2),
)
def test_the_nll_and_its_gradient_are_the_fitters(
    first: list[NoteGroup], second: list[NoteGroup]
) -> None:
    # Two sequences of different lengths in one padded batch: padding must change nothing.
    sequences = [human(first, 0), human(second, 1)]
    nll, w = torch_nll(sequences, WEIGHTS)
    nll.sum().backward()
    feats = [sequence_features(s, CTX) for s in sequences]
    vector = weights_to_vector(WEIGHTS)
    for got, feat in zip(nll.tolist(), feats, strict=True):
        expected, _ = nll_and_gradient(vector, [feat])
        assert got == pytest.approx(expected, abs=1e-9)
    _, gradient = nll_and_gradient(vector, feats)
    assert w.grad is not None
    assert np.allclose(w.grad.numpy(), gradient, atol=1e-8)


@SLOW
@given(groups=group_sequence(max_groups=6, max_notes=3))
def test_viterbi_costs_what_the_decoder_finds(groups: list[NoteGroup]) -> None:
    items = [lattice_arrays(groups, CTX)]
    batch = make_batch(items, dtype=torch.float64)
    w = torch.tensor(weights_to_vector(WEIGHTS), dtype=torch.float64)
    energies, transitions = cost_energies(batch, w)
    path, cost = viterbi(energies, transitions, batch.lengths)
    ctx = Context(tuning=STANDARD, max_span=CTX.max_span, weights=WEIGHTS)
    _, expected = decoder_viterbi(groups, HandSetScorer(weights=WEIGHTS), ctx)
    assert float(cost[0]) == pytest.approx(expected, abs=1e-9)
    assert float(path_energy(energies, transitions, batch.lengths, path)[0]) == pytest.approx(
        float(cost[0]), abs=1e-9
    )


def test_a_sequence_scores_the_same_alone_and_in_a_padded_batch() -> None:
    long = [NoteGroup.of([_n(52 + i, 0.5 * i)]) for i in range(6)]
    short = [NoteGroup.of([_n(55, 0.0), _n(59, 0.0)])]
    together, _ = torch_nll([human(long, 2), human(short, 0)], WEIGHTS)
    alone, _ = torch_nll([human(short, 0)], WEIGHTS)
    assert float(together[1].detach()) == pytest.approx(float(alone[0].detach()), abs=1e-12)


def _n(pitch: int, onset: float) -> NoteEvent:
    return NoteEvent(onset=onset, offset=onset + 0.4, pitch=pitch, confidence=1.0)

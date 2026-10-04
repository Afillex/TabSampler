"""The learned model's network (ADR 0043): an encoder over the groups and a scorer per node."""

from __future__ import annotations

import pytest
import torch

from tabsampler.fingering.fit import weights_to_vector
from tabsampler.model.batch import make_batch
from tabsampler.model.crf import PAD, cost_energies
from tabsampler.model.lattice import lattice_arrays
from tabsampler.model.net import LearnedModel
from tabsampler.types import Context, CostWeights, NoteEvent, NoteGroup, Tuning

CTX = Context(tuning=Tuning.STANDARD, max_span=4)
DEFAULT = CostWeights(open_up_neck=0.7615, temperature=1.2934)


def note(pitch: int, onset: float) -> NoteEvent:
    return NoteEvent(onset=onset, offset=onset + 0.4, pitch=pitch, confidence=1.0)


LONG = [NoteGroup.of([note(52 + i, 0.5 * i), note(59 + i, 0.5 * i)]) for i in range(7)]
SHORT = [
    NoteGroup.of([note(55, 0.0)]),
    NoteGroup.of([note(64, 0.4)]),
    NoteGroup.of([note(57, 0.9)]),
]


def batch(*sequences: list[NoteGroup]):  # type: ignore[no-untyped-def]
    return make_batch([lattice_arrays(s, CTX) for s in sequences], dtype=torch.float64)


def model(seed: int = 0, learned: bool = True) -> LearnedModel:
    torch.manual_seed(seed)
    return LearnedModel(DEFAULT, learned=learned, dtype=torch.float64)


def nudge(m: LearnedModel) -> LearnedModel:
    """Move the scorer's last layer off zero, so the learned term is not identically zero."""
    with torch.no_grad():
        for parameter in m.scorer[-1].parameters():
            parameter.add_(0.1)
    return m


def test_with_the_learned_term_off_the_energies_are_the_cost_models() -> None:
    b = batch(LONG, SHORT)
    energies, transitions = model(learned=False).energies(b)
    w = torch.tensor(weights_to_vector(DEFAULT), dtype=torch.float64)
    expected_energies, expected_transitions = cost_energies(b, w)
    assert torch.equal(energies, expected_energies)
    assert torch.equal(transitions, expected_transitions)


def test_a_new_model_starts_exactly_at_the_default() -> None:
    # The scorer's last layer starts at zero, so the learned term starts at zero (ADR 0043).
    b = batch(LONG, SHORT)
    energies, _ = model(learned=True).energies(b)
    w = torch.tensor(weights_to_vector(DEFAULT), dtype=torch.float64)
    assert torch.equal(energies, cost_energies(b, w)[0])


def test_the_learned_term_touches_real_nodes_only() -> None:
    b = batch(LONG, SHORT)
    energies, _ = nudge(model()).energies(b)
    assert energies.shape == b.nodes.shape
    assert bool((energies[~b.nodes] == PAD).all())
    w = torch.tensor(weights_to_vector(DEFAULT), dtype=torch.float64)
    assert not torch.allclose(energies[b.nodes], cost_energies(b, w)[0][b.nodes])


def test_a_sequence_scores_the_same_alone_and_beside_a_longer_one() -> None:
    # The encoder reads each sequence at its own length: padding must not reach a shorter one,
    # in either direction of the bidirectional GRU.
    m = nudge(model())
    together, _ = m.energies(batch(LONG, SHORT))
    alone, _ = m.energies(batch(SHORT))
    levels, width = alone.shape[1], alone.shape[2]
    assert torch.allclose(together[1, :levels, :width], alone[0], atol=1e-12)


def test_a_fixed_seed_gives_the_same_model_twice() -> None:
    b = batch(SHORT)
    first, _ = nudge(model(seed=7)).energies(b)
    second, _ = nudge(model(seed=7)).energies(b)
    assert torch.equal(first, second)
    third, _ = nudge(model(seed=8)).energies(b)
    assert not torch.equal(first, third)


def test_the_cost_weights_start_at_the_given_decoders() -> None:
    m = model()
    assert m.weights.detach().tolist() == pytest.approx(weights_to_vector(DEFAULT).tolist())

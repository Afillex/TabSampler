"""Fit the cost weights by maximum likelihood on human fingerings (ADR 0023).

Pure: no I/O, no global state.

The hand-set cost model is **linear in its weights**::

    C(path) = move * sum|dhand| + span * sum span
            + high * sum(mean fret / 12) - open_reward * sum n_open
            + string biases and fret-region weights (ADR 0034)
            = w . Phi(path)

and the decoder scores a path as ``exp(-C / T)``, so the decoder is already a linear-chain
conditional random field. Its negative log-likelihood is convex in ``w``, and its gradient
is the classic one::

    NLL(w)      = sum over sequences of  w . Phi(human)  +  log Z_w
    dNLL / dw   = sum over sequences of  Phi(human)  -  E_w[Phi]

The expectation needs node marginals for the three shape features and **edge** marginals
for movement, which the decoder's own forward-backward does not return; this module
computes both on the same node lattice (ADR 0018) in log space.

The temperature is held at 1 while fitting. ``w`` and ``T`` are not separately identifiable
-- only ``w / T`` matters -- so the scale lives in ``w`` here, and calibrating ``T`` is a
separate, later step on held-out data.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize  # pyright: ignore[reportUnknownVariableType]
from scipy.special import logsumexp  # pyright: ignore[reportUnknownVariableType]

from tabsampler.decode.viterbi import build_lattice
from tabsampler.fingering.costs import (
    FRETS_PER_OCTAVE,
    count_high_region,
    count_low_region,
    count_open,
    mean_fretted_fret,
)
from tabsampler.fingering.states import shift_window
from tabsampler.types import ChordState, Context, CostWeights, Hand, NoteGroup

#: Order of the weight vector, and of every feature vector. The low E string's bias is not
#: here: a group's notes always number the same, so the six string counts sum to a constant
#: and only five biases can be fitted -- the low E is the reference, pinned at zero
#: (ADR 0034). The middle fret region (5-11) is the reference for the same reason.
WEIGHT_NAMES = (
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

#: The weights each experiment can switch on; the rest stay at their starting values.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "base": WEIGHT_NAMES[:4],
    "string": WEIGHT_NAMES[4:9],
    "region": WEIGHT_NAMES[9:],
}

Vector = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class HumanSequence:
    """A run of note groups, the shapes a human played them with, and each group's span bound.

    ``spans`` is per group so the lattice can always contain the human's shape: a human
    chord wider than the decoder's usual bound is legal tab, and a lattice that cannot
    express the truth gives it zero likelihood.
    """

    groups: tuple[NoteGroup, ...]
    states: tuple[ChordState, ...]
    spans: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SequenceFeatures:
    """One sequence's lattice as feature arrays: any weights score it without a rebuild."""

    emission: tuple[Vector, ...]  # per level: (nodes, len(WEIGHT_NAMES)), move column zero
    movement: tuple[Vector, ...]  # per transition: (previous nodes, nodes) of |dhand|
    allowed: tuple[NDArray[np.bool_], ...]  # per transition: False where the lattice forbids it
    observed: Vector  # the human path's features, shape (len(WEIGHT_NAMES),)

    @property
    def n_groups(self) -> int:
        return len(self.emission)


@dataclass(frozen=True, slots=True)
class FitResult:
    weights: CostWeights
    nll: float
    n_groups: int
    iterations: int
    converged: bool
    gradient_norm: float

    @property
    def nll_per_group(self) -> float:
        return self.nll / self.n_groups if self.n_groups else 0.0


def weights_to_vector(weights: CostWeights) -> Vector:
    """The weights in ``WEIGHT_NAMES`` order.

    Raises:
        ValueError: if the low E string's bias is not zero; it is the fixed reference.
    """
    if weights.string_bias[0] != 0.0:
        raise ValueError("the low E string's bias is the fitter's reference and must be zero")
    return np.array(
        [
            weights.move,
            weights.span,
            weights.high,
            weights.open_reward,
            *weights.string_bias[1:],
            weights.low_region,
            weights.high_region,
        ],
        dtype=float,
    )


def weights_from_vector(vector: Vector, temperature: float = 1.0) -> CostWeights:
    v = [float(x) for x in vector]
    return CostWeights(
        move=v[0],
        span=v[1],
        high=v[2],
        open_reward=v[3],
        string_bias=(0.0, *v[4:9]),
        low_region=v[9],
        high_region=v[10],
        temperature=temperature,
    )


def _shape_features(state: ChordState) -> Vector:
    """Emission features of one shape. ``w . this`` is ``HandSetScorer.emission_cost``."""
    strings = [0.0] * 5
    for position in state.positions:
        if position.string > 5:
            raise ValueError(
                f"a note is on string {position.string}: the fitter's per-string features "
                "assume a six-string guitar, as DadaGP's cleared songs are (ADR 0034)"
            )
        if position.string > 0:
            strings[position.string - 1] += 1.0
    return np.array(
        [
            0.0,
            float(state.span),
            mean_fretted_fret(state) / FRETS_PER_OCTAVE,
            -float(count_open(state)),
            *strings,
            float(count_low_region(state)),
            float(count_high_region(state)),
        ]
    )


def path_features(states: Sequence[ChordState]) -> Vector:
    """``Phi`` of one path: the shape features summed, plus the hand window's movement."""
    total = np.zeros(len(WEIGHT_NAMES))
    hand: Hand | None = None
    for state in states:
        total += _shape_features(state)
        hand, moved = shift_window(hand, state.fretted_frets)
        total[0] += moved
    return total


def sequence_features(sequence: HumanSequence, ctx: Context) -> SequenceFeatures:
    """Build the lattice once and keep only what scoring it under new weights needs.

    Raises:
        ValueError: if a human shape is not in its group's lattice. Its likelihood would be
            zero and its NLL infinite, which would quietly drag every weight to infinity.
    """
    lattice = build_lattice(sequence.groups, ctx, sequence.spans)
    for index, (nodes, state) in enumerate(zip(lattice, sequence.states, strict=True)):
        if all(node.state != state for node in nodes):
            raise ValueError(
                f"human shape {state.positions} at group {index} is not in the lattice "
                f"(span bound {sequence.spans[index]}); widen the bound or drop the group"
            )

    emission = tuple(np.array([_shape_features(node.state) for node in nodes]) for nodes in lattice)
    movement: list[Vector] = []
    allowed: list[NDArray[np.bool_]] = []
    for previous, current in itertools.pairwise(lattice):
        move = np.zeros((len(previous), len(current)))
        ok = np.ones((len(previous), len(current)), dtype=bool)
        for i, prior in enumerate(previous):
            for j, node in enumerate(current):
                new, moved = shift_window(prior.carried_hand, node.state.fretted_frets)
                if new != node.carried_hand:
                    ok[i, j] = False
                    continue
                move[i, j] = moved
        movement.append(move)
        allowed.append(ok)
    return SequenceFeatures(
        emission=emission,
        movement=tuple(movement),
        allowed=tuple(allowed),
        observed=path_features(sequence.states),
    )


def _lse(values: Vector, axis: int | None = None) -> Vector:
    """Typed boundary around scipy's untyped ``logsumexp``."""
    result = logsumexp(values, axis=axis)  # pyright: ignore[reportUnknownVariableType]
    return np.asarray(result, dtype=float)  # pyright: ignore[reportUnknownArgumentType]


def _sequence_nll_and_gradient(w: Vector, feats: SequenceFeatures) -> tuple[float, Vector]:
    energies = [emission @ w for emission in feats.emission]
    transitions = [
        np.where(ok, w[0] * move, np.inf)
        for move, ok in zip(feats.movement, feats.allowed, strict=True)
    ]

    log_alpha = [-energies[0]]
    for t in range(1, len(energies)):
        log_alpha.append(-energies[t] + _lse(log_alpha[-1][:, None] - transitions[t - 1], axis=0))
    log_beta = [np.zeros_like(energies[-1])]
    for t in range(len(energies) - 2, -1, -1):
        log_beta.insert(
            0, _lse(-transitions[t] - energies[t + 1][None, :] + log_beta[0][None, :], axis=1)
        )
    log_z = float(_lse(log_alpha[-1]))

    expected = np.zeros(len(WEIGHT_NAMES))
    for t, emission in enumerate(feats.emission):
        expected += np.exp(log_alpha[t] + log_beta[t] - log_z) @ emission
    for t, move in enumerate(feats.movement):
        log_xi = (
            log_alpha[t][:, None]
            - transitions[t]
            - energies[t + 1][None, :]
            + log_beta[t + 1][None, :]
            - log_z
        )
        expected[0] += float(np.sum(np.exp(log_xi) * move))
    return float(w @ feats.observed) + log_z, feats.observed - expected


def nll_and_gradient(w: Vector, sequences: Sequence[SequenceFeatures]) -> tuple[float, Vector]:
    """Total negative log-likelihood at temperature 1, and its gradient in ``w``."""
    total = 0.0
    gradient = np.zeros(len(WEIGHT_NAMES))
    for feats in sequences:
        nll, grad = _sequence_nll_and_gradient(w, feats)
        total += nll
        gradient += grad
    return total, gradient


def _minimise(
    objective: Callable[[Vector], tuple[float, Vector]], start: Vector
) -> tuple[Vector, int, bool]:
    """Typed boundary around scipy's untyped L-BFGS. Returns (x, iterations, converged)."""
    result = minimize(objective, start, jac=True, method="L-BFGS-B")  # pyright: ignore[reportUnknownVariableType]
    x: Vector = np.asarray(result.x, dtype=float)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    return x, int(result.nit), bool(result.success)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]


def fit_weights(
    sequences: Sequence[SequenceFeatures],
    initial: CostWeights,
    active: Sequence[str] = FEATURE_GROUPS["base"],
) -> FitResult:
    """Maximum-likelihood weights, starting from ``initial``. Only the weights named in
    ``active`` move; every other weight keeps its starting value, so each feature group is
    an experiment of its own (ADR 0034). The temperature is carried through unchanged: it is
    not fitted here (see the module docstring)."""
    mask = np.array([name in active for name in WEIGHT_NAMES])
    start = weights_to_vector(initial)

    def objective(x: Vector) -> tuple[float, Vector]:
        w = start.copy()
        w[mask] = x
        nll, gradient = nll_and_gradient(w, sequences)
        return nll, gradient[mask]

    x, iterations, converged = _minimise(objective, start[mask])
    w = start.copy()
    w[mask] = x
    nll, gradient = nll_and_gradient(w, sequences)
    return FitResult(
        weights=weights_from_vector(w, temperature=initial.temperature),
        nll=nll,
        n_groups=sum(s.n_groups for s in sequences),
        iterations=iterations,
        converged=converged,
        gradient_norm=float(np.linalg.norm(gradient[mask])),
    )


#: Widest chord a human shape may have before its group is dropped instead of having its
#: span bound widened. The same ceiling ``decode_best_effort`` uses.
DEFAULT_MAX_RELAXED_SPAN = 8


def human_sequences(
    steps: Sequence[tuple[NoteGroup, ChordState]],
    ctx: Context,
    max_relaxed_span: int = DEFAULT_MAX_RELAXED_SPAN,
) -> list[HumanSequence]:
    """Cut one human track into sequences whose every shape the lattice can express.

    A group's span bound is widened to its human shape's span when that exceeds
    ``ctx.max_span``, so a legal wide chord stays in. A shape the lattice cannot express at
    all -- a fret past the neck, or wider than ``max_relaxed_span`` -- is dropped and the
    sequence is split there, rather than charging a hand movement across a gap.
    """
    sequences: list[HumanSequence] = []
    groups: list[NoteGroup] = []
    states: list[ChordState] = []
    spans: list[int] = []

    def close() -> None:
        if groups:
            sequences.append(HumanSequence(tuple(groups), tuple(states), tuple(spans)))
            groups.clear()
            states.clear()
            spans.clear()

    for group, state in steps:
        on_neck = all(p.fret <= ctx.tuning.max_fret for p in state.positions)
        if not on_neck or state.span > max_relaxed_span:
            close()
            continue
        groups.append(group)
        states.append(state)
        spans.append(max(ctx.max_span, state.span))
    close()
    return sequences

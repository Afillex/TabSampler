"""Forward-backward posteriors over chord states (spec 2.2).

Pure: no I/O, no global state.

Viterbi answers "what is the single best fingering". This answers "how sure are we about
each note", which is what makes the output honest about uncertainty (spec 1, goal 4).

Everything is computed on log-potentials, with ``scipy.special.logsumexp``. The
potentials are ``exp(-C / T)``, so::

    log alpha_1(s) = -E_1(s) / T
    log alpha_t(s) = -E_t(s) / T + logsumexp_{s'} [ log alpha_{t-1}(s') - T(s',s) / T ]
    log beta_N(s)  = 0
    log beta_t(s)  = logsumexp_{s'} [ -T(s,s')/T - E_{t+1}(s')/T + log beta_{t+1}(s') ]
    log Z          = logsumexp_s log alpha_N(s)
    log gamma_t(s) = log alpha_t(s) + log beta_t(s) - log Z

A raw cost is never exponentiated. A 500-group sequence would underflow a plain product
of probabilities to zero and turn every posterior into a nan; there is a test for that.

The lattice is the node lattice of :func:`tabsampler.decode.viterbi.build_lattice`: a
shape plus the hand position carried into it (ADR 0018). Two nodes can hold the same
shape with different carried hands, and a transition the lattice forbids gets an infinite
cost, hence a ``-inf`` log-potential and zero weight. Every per-shape quantity here --
:func:`note_posteriors` above all -- therefore sums over nodes, which marginalises the
carried hand out again.

**Design point worth knowing.** :func:`decode` takes the *position* from the Viterbi
path and the *confidence* from these marginals. Picking each note's position by its own
marginal argmax would be wrong: the per-note winners need not form a legal chord state,
so the output could place two notes on one string. The two can therefore disagree --
a note can sit on the MAP path while its marginal favours somewhere else -- and that
disagreement is real information, not a bug.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.special import (
    logsumexp,  # pyright: ignore[reportUnknownVariableType]
)

from tabsampler.decode.viterbi import LatticeNode, build_lattice, node_transition_cost, viterbi
from tabsampler.types import (
    Context,
    FingeringScorer,
    NoteGroup,
    Position,
    TabNote,
)


def _lse(a: NDArray[np.float64], axis: int | None = None) -> NDArray[np.float64]:
    """Typed boundary around scipy's untyped ``logsumexp``.

    Its inferred return type is a union including a tuple, because of the
    ``return_sign`` option we never use. Confining that to one function keeps the rest
    of this module fully checked.
    """
    result = logsumexp(  # pyright: ignore[reportUnknownVariableType]
        a, axis=axis
    )
    return np.asarray(result, dtype=float)  # pyright: ignore[reportUnknownArgumentType]


def _lse_scalar(a: NDArray[np.float64]) -> float:
    return float(_lse(a))


def _cost_arrays(
    groups: Sequence[NoteGroup],
    lattice: Sequence[tuple[LatticeNode, ...]],
    scorer: FingeringScorer,
    ctx: Context,
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """Emission costs per level, and transition costs between consecutive levels.

    A transition the node lattice forbids costs ``inf``, so its log-potential is ``-inf``
    and it carries no weight. Every node has at least one legal predecessor and at least
    one legal successor by construction, so no level ever sums to ``-inf``.
    """
    emissions = [
        np.array([scorer.emission_cost(group, node.state, ctx) for node in nodes], dtype=float)
        for group, nodes in zip(groups, lattice, strict=True)
    ]
    transitions = [
        np.array(
            [
                [node_transition_cost(prior, node, scorer) for node in lattice[t]]
                for prior in lattice[t - 1]
            ],
            dtype=float,
        )
        for t in range(1, len(lattice))
    ]
    return emissions, transitions


def _log_alpha_beta(
    emissions: Sequence[NDArray[np.float64]],
    transitions: Sequence[NDArray[np.float64]],
    temperature: float,
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    n = len(emissions)
    log_alpha: list[NDArray[np.float64]] = [-emissions[0] / temperature]
    for t in range(1, n):
        # shape (prev, curr): alpha_{t-1} broadcast down the rows, transitions scaled.
        contributions = log_alpha[t - 1][:, None] - transitions[t - 1] / temperature
        log_alpha.append(-emissions[t] / temperature + _lse(contributions, axis=0))

    log_beta: list[NDArray[np.float64]] = [np.zeros_like(emissions[-1])] * n
    log_beta[n - 1] = np.zeros_like(emissions[-1])
    for t in range(n - 2, -1, -1):
        contributions = (
            -transitions[t] / temperature
            + (-emissions[t + 1] / temperature + log_beta[t + 1])[None, :]
        )
        log_beta[t] = _lse(contributions, axis=1)
    return log_alpha, log_beta


def log_partition(
    groups: Sequence[NoteGroup],
    scorer: FingeringScorer,
    ctx: Context,
    temperature: float | None = None,
    spans: Sequence[int] | None = None,
) -> float:
    """log Z: the log total weight of every path. 0.0 for no groups (one empty path)."""
    if not groups:
        return 0.0
    t = ctx.weights.temperature if temperature is None else temperature
    lattice = build_lattice(groups, ctx, spans)
    emissions, transitions = _cost_arrays(groups, lattice, scorer, ctx)
    log_alpha, _ = _log_alpha_beta(emissions, transitions, t)
    return _lse_scalar(log_alpha[-1])


def forward_backward(
    groups: Sequence[NoteGroup],
    scorer: FingeringScorer,
    ctx: Context,
    temperature: float | None = None,
    spans: Sequence[int] | None = None,
) -> list[NDArray[np.float64]]:
    """Per-group **node** posteriors, parallel to :func:`build_lattice`. Each array sums to 1.

    A shape's own posterior is the sum over the nodes holding it, which is what
    :func:`note_posteriors` does.

    Args:
        temperature: Defaults to ``ctx.weights.temperature``. Lower concentrates the
            posteriors on the Viterbi path; higher flattens them toward uniform.

    Raises:
        UnfingerableGroupError: if any group has no legal state.
    """
    if not groups:
        return []
    t = ctx.weights.temperature if temperature is None else temperature
    if t <= 0.0:
        raise ValueError(f"temperature must be positive, got {t}")

    lattice = build_lattice(groups, ctx, spans)
    emissions, transitions = _cost_arrays(groups, lattice, scorer, ctx)
    log_alpha, log_beta = _log_alpha_beta(emissions, transitions, t)
    log_z = _lse_scalar(log_alpha[-1])

    posteriors: list[NDArray[np.float64]] = []
    for a, b in zip(log_alpha, log_beta, strict=True):
        log_gamma = a + b - log_z
        # Renormalise explicitly: log_z is computed from the final level, and floating
        # point leaves each level a few ulps off 1 otherwise.
        gamma = np.exp(log_gamma - _lse_scalar(log_gamma))
        posteriors.append(np.asarray(gamma, dtype=float))
    return posteriors


def note_posteriors(
    groups: Sequence[NoteGroup],
    scorer: FingeringScorer,
    ctx: Context,
    temperature: float | None = None,
    spans: Sequence[int] | None = None,
) -> list[list[dict[Position, float]]]:
    """For each group, for each note, the probability of each candidate position.

    A note's position posterior marginalises over the states that place it there:
    ``P(note i at p) = sum of gamma_t(s) over states s with s.positions[i] == p``.
    """
    if not groups:
        return []
    lattice = build_lattice(groups, ctx, spans)
    levels = forward_backward(groups, scorer, ctx, temperature, spans)

    out: list[list[dict[Position, float]]] = []
    for group, nodes, gamma in zip(groups, lattice, levels, strict=True):
        per_note: list[dict[Position, float]] = [{} for _ in group.notes]
        for node, weight in zip(nodes, gamma, strict=True):
            for index, position in enumerate(node.state.positions):
                per_note[index][position] = per_note[index].get(position, 0.0) + float(weight)
        out.append(per_note)
    return out


def decode(
    groups: Sequence[NoteGroup],
    scorer: FingeringScorer,
    ctx: Context,
    temperature: float | None = None,
    max_alternatives: int | None = None,
    spans: Sequence[int] | None = None,
) -> list[TabNote]:
    """Place every note, with a posterior and ranked alternatives.

    Positions come from the Viterbi path (a jointly legal assignment); confidences come
    from the forward-backward marginals. See the module docstring for why the two are
    taken from different places.

    Raises:
        UnfingerableGroupError: if any group has no legal state.
    """
    if not groups:
        return []

    path, _ = viterbi(groups, scorer, ctx, spans)
    marginals = note_posteriors(groups, scorer, ctx, temperature, spans)

    tab: list[TabNote] = []
    for group, state, per_group in zip(groups, path, marginals, strict=True):
        for index, note in enumerate(group.notes):
            chosen = state.positions[index]
            distribution = per_group[index]
            others = sorted(
                ((p, w) for p, w in distribution.items() if p != chosen),
                key=lambda item: (-item[1], item[0].string, item[0].fret),
            )
            if max_alternatives is not None:
                others = others[:max_alternatives]
            tab.append(
                TabNote(
                    note=note,
                    position=chosen,
                    posterior=min(max(distribution.get(chosen, 0.0), 0.0), 1.0),
                    alternatives=tuple(others),
                )
            )
    return tab

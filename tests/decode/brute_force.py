"""Brute-force reference implementations. **This file is the specification.**

Obviously correct and hopelessly slow: it enumerates every path through the state
lattice and scores each one from first principles. The decoder in
``src/tabsampler/decode/`` must agree with it exactly on short inputs.

Nothing here is clever on purpose. If you cannot read a function in this file and see
that it is right, it is not doing its job.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator, Sequence

from tabsampler.fingering.states import enumerate_states
from tabsampler.types import ChordState, Context, FingeringScorer, NoteGroup


def lattice(groups: Sequence[NoteGroup], ctx: Context) -> list[tuple[ChordState, ...]]:
    """Legal states for each group, in order."""
    return [enumerate_states(g, ctx.tuning, ctx.max_span) for g in groups]


def all_paths(groups: Sequence[NoteGroup], ctx: Context) -> Iterator[tuple[ChordState, ...]]:
    """Every combination of one state per group."""
    return itertools.product(*lattice(groups, ctx))


def path_cost(
    groups: Sequence[NoteGroup],
    path: Sequence[ChordState],
    scorer: FingeringScorer,
    ctx: Context,
) -> float:
    """Total cost of one complete path, straight from spec 2.2.

    C = sum of emission costs + sum of movement costs, where movement is charged against
    **where the hand was**, not against the previous shape.

    The hand starts nowhere, so the first group is charged no movement. A shape with a
    fretted note puts the hand at its lowest fretted fret (spec 2.2); an all-open shape
    has no hand position of its own and leaves the hand where it was, so the movement
    across it is charged to the next fretted shape (ADR 0018).

    The carry is spelled out here rather than imported from
    ``tabsampler.fingering.states``, so that this stays an independent check on it.
    """
    total = 0.0
    for group, state in zip(groups, path, strict=True):
        total += scorer.emission_cost(group, state, ctx)

    hand: int | None = None
    for state in path:
        total += scorer.transition_cost_from(hand, state)
        if state.hand_position is not None:
            hand = state.hand_position
    return total


def brute_force_min(
    groups: Sequence[NoteGroup], scorer: FingeringScorer, ctx: Context
) -> tuple[tuple[ChordState, ...], float]:
    """The lowest-cost path, found by trying all of them."""
    best_path: tuple[ChordState, ...] = ()
    best_cost = math.inf
    for path in all_paths(groups, ctx):
        cost = path_cost(groups, path, scorer, ctx)
        if cost < best_cost:
            best_path, best_cost = path, cost
    return best_path, best_cost


def brute_force_marginals(
    groups: Sequence[NoteGroup],
    scorer: FingeringScorer,
    ctx: Context,
    temperature: float,
) -> list[dict[ChordState, float]]:
    """Per-group state marginals, by summing the weight of every path.

    Weight of a path is exp(-C / T). The marginal of state s at position t is the total
    weight of paths whose t-th state is s, divided by the total weight of all paths.

    Computed in plain probability space, which is exactly why it is only usable on tiny
    inputs -- and exactly what makes it a fair check on the log-space implementation.
    """
    weights: list[dict[ChordState, float]] = [dict() for _ in groups]
    partition = 0.0
    for path in all_paths(groups, ctx):
        weight = math.exp(-path_cost(groups, path, scorer, ctx) / temperature)
        partition += weight
        for index, state in enumerate(path):
            weights[index][state] = weights[index].get(state, 0.0) + weight

    if partition == 0.0:  # pragma: no cover - guarded by callers
        raise ZeroDivisionError("no paths carry any weight")
    return [{s: w / partition for s, w in level.items()} for level in weights]

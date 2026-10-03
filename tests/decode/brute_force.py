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

    The hand covers a range of frets, written out here from ADR 0030's text (which keeps
    ADR 0025's window) and not imported, so this stays an independent check on the decoder
    *and* on the cost model's movement term:

    - the first fretted shape places the hand, for free;
    - a fretted shape whose notes all lie inside the range the hand covers costs nothing,
      and the index finger stays where it was;
    - otherwise the index moves to the start, among those whose rest window (start to
      start + 4) holds the whole shape, closest to where it was -- and a shape wider than
      the rest window can only be held stretched, with the index on its own lowest fret;
    - after every fretted shape the hand covers from its index to the higher of index + 4
      and the shape's highest fret: stretched only while a shape needs it;
    - an all-open shape leaves the hand as it was, stretch included;
    - movement is the distance the index moves, charged at ``move`` per fret.
    """
    total = 0.0
    for group, state in zip(groups, path, strict=True):
        total += scorer.emission_cost(group, state, ctx)

    hand: tuple[int, int] | None = None  # (index fret, highest fret the hand covers)
    for state in path:
        fretted = sorted(p.fret for p in state.positions if p.fret > 0)
        if not fretted:
            continue  # open strings need no hand; it stays as it was, stretch and all
        low, high = fretted[0], fretted[-1]
        if hand is None:
            hand = (low, max(low + 4, high))  # the first fretted shape places it, free
            continue
        index, covered_to = hand
        if index <= low and high <= covered_to:
            new_index = index  # inside what the hand covers: a finger reaches
        elif high - low > 4:
            new_index = low  # only a stretch holds it, from its own lowest fret
        else:
            holding = range(high - 4, low + 1)  # every rest window that holds the shape
            new_index = min(holding, key=lambda start: abs(start - index))
        total += scorer.weights.move * abs(new_index - index)  # type: ignore[attr-defined]
        hand = (new_index, max(new_index + 4, high))
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

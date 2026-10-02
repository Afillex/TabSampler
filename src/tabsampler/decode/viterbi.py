"""Viterbi: the single lowest-cost fingering (spec 2.2).

Pure: no I/O, no global state.

Recurrence, minimising cost directly::

    delta_1(s) = E_1(s)
    delta_t(s) = E_t(s) + min over s' of [ delta_{t-1}(s') + T(s', s) ]
    psi_t(s)   = the s' attaining that minimum
    best       = argmin over s of delta_T(s), then follow psi backwards

Costs are added, never multiplied, so there is nothing to underflow here. Log space
matters in :mod:`tabsampler.decode.forward_backward`, which exponentiates.

**The states above are lattice nodes, not bare chord shapes (ADR 0018).** A node is a
shape plus the hand position in force while it is played, because an all-open shape has
no hand position of its own and must carry the previous one forward -- otherwise
fret 2 -> open chord -> fret 10 is charged no movement at all. Only all-open shapes need
the augmentation: any shape with a fretted note determines its own hand position, so it
has exactly one node. See :func:`build_lattice`.

Written as plain loops. Spec 2.2 says to optimise only after measuring, and the state
counts are small enough (see
:func:`tabsampler.fingering.states.state_count_stats`) that this has not been needed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

from tabsampler.errors import UnfingerableGroupError
from tabsampler.fingering.states import carry_hand, enumerate_states
from tabsampler.types import ChordState, Context, FingeringScorer, NoteGroup


class LatticeNode(NamedTuple):
    """One decoder state: a chord shape, plus where the hand is while it is played.

    ``carried_hand`` equals ``state.hand_position`` for any shape with a fretted note.
    For an all-open shape it is the position inherited from earlier in the piece, and
    None only when nothing fretted has been played yet.
    """

    state: ChordState
    carried_hand: int | None


def node_transition_cost(prior: LatticeNode, node: LatticeNode, scorer: FingeringScorer) -> float:
    """Cost of moving between two lattice nodes, or ``inf`` if the lattice forbids it.

    A node's ``carried_hand`` must be exactly what the predecessor's hand becomes after
    playing this shape. Anything else is a bookkeeping inconsistency, not a move a hand
    could make, so it gets infinite cost rather than being silently allowed.
    """
    if carry_hand(prior.carried_hand, node.state) != node.carried_hand:
        return math.inf
    return scorer.transition_cost_from(prior.carried_hand, node.state)


def build_lattice(
    groups: Sequence[NoteGroup],
    ctx: Context,
    spans: Sequence[int] | None = None,
) -> list[tuple[LatticeNode, ...]]:
    """Legal nodes for every group: each legal shape, paired with the hand it is played with.

    A shape with any fretted note fixes its own hand position and yields exactly one
    node. An all-open shape yields one node per distinct hand position reachable at the
    previous level, because that is the information it has to carry forward (ADR 0018).
    Augmenting *every* node with a hand position instead would multiply the state space
    by ``max_fret + 1`` and Viterbi is ``O(T * S^2)``; the measured cost of doing it only
    where it is needed is in the ADR.

    Args:
        spans: Optional per-group span bound, overriding ``ctx.max_span``. Used by
            :mod:`tabsampler.decode.robust`, which relaxes the bound for the occasional
            chord that genuinely needs a wider stretch.

    Raises:
        UnfingerableGroupError: if any group has no legal state. Raised rather than
            returning an empty level, because a level with no states would make every
            path cost infinite and the result would look like a valid but terrible tab.
    """
    if spans is not None and len(spans) != len(groups):
        raise ValueError(f"{len(spans)} spans for {len(groups)} groups")
    lattice: list[tuple[LatticeNode, ...]] = []
    # Hand positions the previous level can leave behind. Nothing precedes group 0, so
    # an all-open opening shape starts from no hand position at all -- which is what
    # keeps the first group's cost unchanged from M1.
    carried: tuple[int | None, ...] = (None,)
    for index, group in enumerate(groups):
        span = ctx.max_span if spans is None else spans[index]
        states = enumerate_states(group, ctx.tuning, span)
        if not states:
            pitches = [n.pitch for n in group.notes]
            raise UnfingerableGroupError(
                f"group at index {index} (onset {group.onset:.3f}s, pitches {pitches}) "
                f"has no legal chord state with tuning {ctx.tuning.open_pitches}, "
                f"capo {ctx.tuning.capo}, max_span {span}"
            )
        level: list[LatticeNode] = []
        for state in states:
            # The window a shape is played in depends on where the hand came from, for
            # fretted shapes as well as all-open ones (ADR 0025), so each state gets one
            # node per distinct window it can be reached in.
            hands = dict.fromkeys(carry_hand(hand, state) for hand in carried)
            level.extend(LatticeNode(state, hand) for hand in hands)
        lattice.append(tuple(level))
        # dict.fromkeys dedups while preserving order, and the order it sees is the
        # sorted state order, so the lattice stays deterministic run to run.
        carried = tuple(dict.fromkeys(node.carried_hand for node in level))
    return lattice


def viterbi(
    groups: Sequence[NoteGroup],
    scorer: FingeringScorer,
    ctx: Context,
    spans: Sequence[int] | None = None,
) -> tuple[list[ChordState], float]:
    """The lowest-cost state sequence and its total cost.

    Returns ``([], 0.0)`` for no groups.

    Raises:
        UnfingerableGroupError: if a group has no legal state.
    """
    if not groups:
        return [], 0.0

    lattice = build_lattice(groups, ctx, spans)

    # delta[i] is the cost of the best path ending in lattice[0][i].
    delta: list[float] = [scorer.emission_cost(groups[0], node.state, ctx) for node in lattice[0]]
    backpointers: list[list[int]] = []

    for t in range(1, len(groups)):
        previous, current = lattice[t - 1], lattice[t]
        new_delta: list[float] = []
        psi: list[int] = []
        for node in current:
            emission = scorer.emission_cost(groups[t], node.state, ctx)
            best_cost = math.inf
            best_index: int | None = None
            for index, prior in enumerate(previous):
                candidate = delta[index] + node_transition_cost(prior, node, scorer)
                if candidate < best_cost:
                    best_cost, best_index = candidate, index
            if best_index is None:  # pragma: no cover - build_lattice makes this impossible
                raise AssertionError(
                    f"lattice node {node} at group {t} has no reachable predecessor; "
                    f"build_lattice is supposed to guarantee one"
                )
            new_delta.append(emission + best_cost)
            psi.append(best_index)
        delta = new_delta
        backpointers.append(psi)

    # Ties are common with hand-set weights; min() takes the first, which combined with
    # the sorted lattice makes the choice deterministic.
    final = min(range(len(delta)), key=lambda i: delta[i])
    total = delta[final]

    path_indices = [final]
    for psi in reversed(backpointers):
        path_indices.append(psi[path_indices[-1]])
    path_indices.reverse()

    return [lattice[t][i].state for t, i in enumerate(path_indices)], total

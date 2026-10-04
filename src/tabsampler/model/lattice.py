"""The decoder's lattice as arrays, for the learned model (ADR 0043).

Pure: no I/O, no global state, no torch. For one sequence of note groups it lays out what the
model reads and scores: the decoder's own lattice (ADRs 0018, 0030), each node's cost-model
features exactly as :mod:`tabsampler.fingering.fit` computes them, a descriptor of each node
for the learned term, the groups' own inputs, the transitions, and -- for training -- the node
the human path passes through at each level.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tabsampler.decode.viterbi import LatticeNode, build_lattice
from tabsampler.fingering.fit import node_features, transition_arrays
from tabsampler.fingering.states import carry_hand
from tabsampler.types import ChordState, Context, Hand, NoteGroup

Vector = NDArray[np.float64]

#: The pitches a group's input marks, as MIDI numbers: below a seven-string's low B (35) to
#: well above a 24-fret high E (88).
LOWEST_PITCH = 28
HIGHEST_PITCH = 100
N_PITCHES = HIGHEST_PITCH - LOWEST_PITCH + 1

#: A group's input: a mark for each pitch it holds, its note count over six, and the log of
#: one plus the seconds since the previous group's onset (zero for the first group).
GROUP_INPUT_SIZE = N_PITCHES + 2

STRINGS = 6

#: Frets and spans are divided by this, so a descriptor stays within [0, 1] on a 24-fret neck.
FRET_SCALE = 24.0

#: A node's descriptor: for each of six strings whether it is played, whether open, and its
#: fret over 24; then the hand's index fret over 24 (zero before anything is fretted) and the
#: shape's span over 24.
DESCRIPTOR_SIZE = 3 * STRINGS + 2


@dataclass(frozen=True, slots=True)
class LatticeArrays:
    """One sequence's lattice, laid out for the learned model."""

    states: tuple[tuple[ChordState, ...], ...]  # per level, each node's shape
    features: tuple[Vector, ...]  # per level: (nodes, len(WEIGHT_NAMES)), the cost model's
    descriptors: tuple[Vector, ...]  # per level: (nodes, DESCRIPTOR_SIZE)
    group_inputs: Vector  # (levels, GROUP_INPUT_SIZE)
    movement: tuple[Vector, ...]  # per transition: (previous nodes, nodes) of |dhand|
    allowed: tuple[NDArray[np.bool_], ...]  # per transition: False where the lattice forbids it
    human: tuple[int, ...] | None  # the human path's node at each level, when there is one


def group_inputs(groups: Sequence[NoteGroup]) -> Vector:
    """Each group's input to the encoder: what was played and when, not where.

    Raises:
        ValueError: if a pitch lies outside ``LOWEST_PITCH``-``HIGHEST_PITCH``.
    """
    out = np.zeros((len(groups), GROUP_INPUT_SIZE))
    previous: float | None = None
    for t, group in enumerate(groups):
        for note in group.notes:
            if not LOWEST_PITCH <= note.pitch <= HIGHEST_PITCH:
                raise ValueError(
                    f"pitch {note.pitch} lies outside the inputs' range "
                    f"{LOWEST_PITCH}-{HIGHEST_PITCH}"
                )
            out[t, note.pitch - LOWEST_PITCH] = 1.0
        out[t, N_PITCHES] = len(group.notes) / STRINGS
        if previous is not None:
            out[t, N_PITCHES + 1] = float(np.log1p(max(group.onset - previous, 0.0)))
        previous = group.onset
    return out


def descriptor(node: LatticeNode) -> Vector:
    """What the learned term sees of a candidate: its strings and frets, the hand, the span.

    Raises:
        ValueError: if a note lies beyond six strings; the model is trained on six-string
            guitar, as DadaGP's cleared songs are (ADR 0021).
    """
    out = np.zeros(DESCRIPTOR_SIZE)
    for position in node.state.positions:
        if position.string >= STRINGS:
            raise ValueError(f"a note is on string {position.string}; the model has six")
        base = 3 * position.string
        out[base] = 1.0
        if position.fret == 0:
            out[base + 1] = 1.0
        else:
            out[base + 2] = min(position.fret / FRET_SCALE, 1.0)
    hand = node.carried_hand
    out[3 * STRINGS] = 0.0 if hand is None else min(hand[0] / FRET_SCALE, 1.0)
    out[3 * STRINGS + 1] = min(node.state.span / FRET_SCALE, 1.0)
    return out


def lattice_arrays(
    groups: Sequence[NoteGroup],
    ctx: Context,
    spans: Sequence[int] | None = None,
    human_states: Sequence[ChordState] | None = None,
) -> LatticeArrays:
    """The lattice of ``groups`` as arrays, with the human path's nodes if its shapes are given.

    Raises:
        ValueError: if a human shape is not in its group's lattice, or the shapes and the groups
            differ in number.
        UnfingerableGroupError: if a group has no legal shape (from :func:`build_lattice`).
    """
    lattice = build_lattice(groups, ctx, spans)
    movement, allowed = transition_arrays(lattice)
    return LatticeArrays(
        states=tuple(tuple(node.state for node in nodes) for nodes in lattice),
        features=tuple(
            np.array([node_features(node, first=level == 0) for node in nodes])
            for level, nodes in enumerate(lattice)
        ),
        descriptors=tuple(np.array([descriptor(node) for node in nodes]) for nodes in lattice),
        group_inputs=group_inputs(groups),
        movement=tuple(movement),
        allowed=tuple(allowed),
        human=None if human_states is None else _human_nodes(lattice, human_states),
    )


def _human_nodes(
    lattice: Sequence[Sequence[LatticeNode]], states: Sequence[ChordState]
) -> tuple[int, ...]:
    """At each level, the node holding the human shape with the hand the human path carries."""
    if len(states) != len(lattice):
        raise ValueError(f"{len(states)} human shapes for {len(lattice)} groups")
    hand: Hand | None = None
    out: list[int] = []
    for level, (nodes, state) in enumerate(zip(lattice, states, strict=True)):
        hand = carry_hand(hand, state)
        found = next(
            (
                j
                for j, node in enumerate(nodes)
                if node.state == state and node.carried_hand == hand
            ),
            None,
        )
        if found is None:
            raise ValueError(
                f"human shape {state.positions} at group {level} is not in the lattice"
            )
        out.append(found)
    return tuple(out)

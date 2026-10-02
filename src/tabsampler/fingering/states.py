"""Chord-state enumeration with pruning (ADR 0010).

Pure: no I/O, no global state.

A :class:`ChordState` assigns every note in a group to a **distinct** string. Two hard
constraints are enforced *during* the search rather than by filtering afterwards, which
is what keeps the state count bounded (spec 7 names the explosion as a risk):

- one note per string;
- the span of **fretted** notes is at most ``max_span``. Open strings need no finger and
  are excluded, so open-E plus two notes at fret 12 has span 0, not 12.

An unfingerable group returns an empty tuple. The caller decides what that means:
:mod:`tabsampler.decode.viterbi` raises :class:`UnfingerableGroupError` rather than
returning a path of infinite cost that would later read as a valid tab.

:func:`carry_hand` lives here rather than in the cost model because where the hand ends
up is geometry, not a weight: spec 2.2 defines it, every scorer has to agree with it,
and the decoder's lattice is built on it (ADR 0018).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tabsampler.fingering.candidates import candidates
from tabsampler.types import ChordState, NoteGroup, Position, Tuning


@dataclass(frozen=True, slots=True)
class StateStats:
    """Instrumentation for the state-space size (spec 7's explosion risk)."""

    n_groups: int
    n_unfingerable: int
    total_states: int
    max_states: int
    mean_states: float


#: Frets one hand position covers, in span units: fret h to fret h + 4, as ADR 0011's span
#: limit below fret 12. Fixed by decision, not fitted (ADR 0025).
HAND_WINDOW = 4


def shift_window(
    previous_hand: int | None, fretted: Sequence[int], window: int = HAND_WINDOW
) -> tuple[int | None, int]:
    """Where the hand's window starts after a shape, and how far it had to move (ADR 0025).

    The hand covers frets ``hand`` to ``hand + window`` and moves only when a fretted note
    falls outside that range, by the least distance that brings the shape inside it. A
    shape wider than the window anchors at its lowest fret. Open strings need no hand, so
    an all-open shape (``fretted`` empty) leaves it where it was. ``fretted`` is ascending.
    """
    if not fretted:
        return previous_hand, 0
    low, high = fretted[0], fretted[-1]
    if previous_hand is None:
        return low, 0
    if low >= previous_hand and high <= previous_hand + window:
        return previous_hand, 0
    new = low if low < previous_hand else min(low, high - window)
    return new, abs(new - previous_hand)


def carry_hand(previous_hand: int | None, state: ChordState) -> int | None:
    """Where the hand's window starts after playing ``state``, given where it was before.

    The hand is a 4-fret window (ADR 0025, see :func:`shift_window`): a finger reaching
    inside it does not move it, and an all-open shape **inherits** the previous position
    rather than resetting to fret 0 (ADR 0018). ``None`` means the hand has not been
    anywhere yet.
    """
    return shift_window(previous_hand, state.fretted_frets)[0]


def enumerate_states(group: NoteGroup, tuning: Tuning, max_span: int) -> tuple[ChordState, ...]:
    """Every legal chord state for ``group``, sorted for determinism.

    Returns an empty tuple when the group cannot be fingered at all: more notes than
    strings, a note with no legal position, a unison whose pitch has only one position,
    or a shape that cannot fit inside ``max_span``.
    """
    if len(group) > tuning.n_strings:
        return ()

    per_note: list[tuple[Position, ...]] = [candidates(note.pitch, tuning) for note in group.notes]
    if any(not options for options in per_note):
        return ()

    found: list[ChordState] = []
    chosen: list[Position] = []
    used: set[int] = set()

    def fretted_bounds(positions: Sequence[Position]) -> tuple[int, int] | None:
        frets = [p.fret for p in positions if p.fret != 0]
        return (min(frets), max(frets)) if frets else None

    def search(index: int) -> None:
        if index == len(per_note):
            found.append(ChordState(positions=tuple(chosen)))
            return
        for option in per_note[index]:
            if option.string in used:
                continue
            chosen.append(option)
            bounds = fretted_bounds(chosen)
            # Prune inside the search: a partial shape already too wide can never
            # become narrower by adding notes.
            if bounds is None or bounds[1] - bounds[0] <= max_span:
                used.add(option.string)
                search(index + 1)
                used.discard(option.string)
            chosen.pop()

    search(0)
    found.sort(key=lambda s: tuple((p.string, p.fret) for p in s.positions))
    return tuple(found)


def state_count_stats(groups: Sequence[NoteGroup], tuning: Tuning, max_span: int) -> StateStats:
    """Measure the state space over a sequence of groups.

    Spec 7 lists chord-state explosion as a risk and says to measure the state counts.
    This is that measurement; the decoder's complexity claim rests on it.
    """
    counts = [len(enumerate_states(g, tuning, max_span)) for g in groups]
    total = sum(counts)
    return StateStats(
        n_groups=len(counts),
        n_unfingerable=sum(1 for c in counts if c == 0),
        total_states=total,
        max_states=max(counts, default=0),
        mean_states=(total / len(counts)) if counts else 0.0,
    )

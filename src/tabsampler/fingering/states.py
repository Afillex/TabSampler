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
from tabsampler.types import ChordState, Hand, NoteGroup, Position, Tuning


@dataclass(frozen=True, slots=True)
class StateStats:
    """Instrumentation for the state-space size (spec 7's explosion risk)."""

    n_groups: int
    n_unfingerable: int
    total_states: int
    max_states: int
    mean_states: float


#: Frets a hand at rest covers above its index finger, in span units: fret h to fret h + 4,
#: as ADR 0011's span limit below fret 12. Fixed by decision, not fitted (ADR 0025).
HAND_WINDOW = 4


def shift_window(
    previous: Hand | None, fretted: Sequence[int], window: int = HAND_WINDOW
) -> tuple[Hand | None, int]:
    """The frets the hand covers after a shape, and how far its index finger moved.

    At rest the hand covers its index fret and the ``window`` frets above it (ADR 0025). A
    shape inside what the hand covers costs nothing. Otherwise the index moves by the least
    distance that lets a rest window hold the shape; a shape wider than that can only be
    held stretched, from its own lowest fret, and the hand stays stretched only while a
    shape needs it (ADR 0030). Open strings need no hand, so an all-open shape (``fretted``
    empty) leaves it as it was. ``fretted`` is ascending.
    """
    if not fretted:
        return previous, 0
    low, high = fretted[0], fretted[-1]
    if previous is None:
        return (low, max(low + window, high)), 0
    start, end = previous
    if start <= low and high <= end:
        new_start = start
    elif low < start or high - low > window:
        new_start = low
    else:
        new_start = high - window
    return (new_start, max(new_start + window, high)), abs(new_start - start)


def carry_hand(previous_hand: Hand | None, state: ChordState) -> Hand | None:
    """The frets the hand covers after playing ``state``, given what it covered before.

    See :func:`shift_window`: a finger reaching inside the hand does not move it, a wide
    chord stretches it (ADR 0030), and an all-open shape **inherits** the previous hand
    rather than resetting it (ADR 0018). ``None`` means the hand has not been anywhere yet.
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

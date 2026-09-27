"""Stage 2: note events to legal (string, fret) candidates, and notes to groups.

Pure: no I/O, no global state.

Two jobs, both small and both load-bearing:

- :func:`candidates` enumerates every position that can sound a pitch in a tuning.
  A pitch with no legal position returns an **empty tuple** rather than raising, so the
  caller decides what an unreachable note means; :mod:`tabsampler.fingering.states` then
  surfaces it as an unfingerable group rather than dropping the note.
- :func:`group_notes` collects notes that sound together into a :class:`NoteGroup`.
  Windows are anchored at the first onset of each group, **not** chained transitively:
  a run of notes 20 ms apart would otherwise collapse into one enormous "chord".
"""

from __future__ import annotations

from collections.abc import Sequence

from tabsampler.types import NoteEvent, NoteGroup, Position, Tuning

#: Spec 2.2's grouping window. Notes within this of the group's first onset sound
#: together as a chord.
DEFAULT_GROUP_WINDOW_S = 0.03


def candidates(pitch: int, tuning: Tuning) -> tuple[Position, ...]:
    """Every legal position that sounds ``pitch``, sorted by (string, fret).

    Returns an empty tuple when the pitch is below the lowest open string, above the
    last reachable fret, or blocked by a capo. Sorted so downstream enumeration is
    deterministic.
    """
    found: list[Position] = []
    for string in range(tuning.n_strings):
        fret = tuning.fret_for(string, pitch)
        if fret is not None:
            found.append(Position(string=string, fret=fret))
    found.sort(key=lambda p: (p.string, p.fret))
    return tuple(found)


def group_notes(
    notes: Sequence[NoteEvent], window_s: float = DEFAULT_GROUP_WINDOW_S
) -> list[NoteGroup]:
    """Collect simultaneous notes into groups, in time order.

    A group starts at the earliest ungrouped onset and takes every note whose onset is
    within ``window_s`` of it, inclusive. The window is then re-anchored at the next
    ungrouped note, so groups cannot chain: 50 notes spaced 20 ms apart give many small
    groups, not one group of 50.

    Sorting first makes the result independent of input order, which spec 4's
    determinism requirement needs.

    Raises:
        ValueError: if ``window_s`` is not positive.
    """
    if window_s <= 0:
        raise ValueError(f"window_s must be positive, got {window_s}")
    if not notes:
        return []

    ordered = sorted(notes, key=lambda note: (note.onset, note.pitch))
    groups: list[NoteGroup] = []
    index = 0
    while index < len(ordered):
        anchor = ordered[index].onset
        end = index
        while end < len(ordered) and ordered[end].onset - anchor <= window_s:
            end += 1
        groups.append(NoteGroup.of(ordered[index:end]))
        index = end
    return groups

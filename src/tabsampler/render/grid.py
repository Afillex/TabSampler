"""The time grid shared by the MusicXML and Guitar Pro exporters (ADRs 0017, 0059).

Pure: no I/O. A *unit* is a 1/128 note -- 15.625 ms at the declared 120 BPM. Rhythm is not
transcribed (ADR 0009): the grid is a container for measured onsets, not a beat estimate.

One voice. Notes within the decoder's grouping window form one chord at the first note's
onset, with the same anchored rule as ``group_notes``. A chord lasts until its longest note
ends or the next chord starts, whichever is first, so a note held under the next one is cut
where the next starts. Gaps become rests. Each stretch is split at bar lines and into plain
note values (powers of two), tied, largest first.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tabsampler.types import Position, TabNote

TEMPO_BPM = 120
BEATS_PER_BAR = 4
UNITS_PER_QUARTER = 32
UNITS_PER_BAR = BEATS_PER_BAR * UNITS_PER_QUARTER
UNITS_PER_SECOND = UNITS_PER_QUARTER * TEMPO_BPM / 60

#: ADR 0017's text, with ADR 0059's sentence. Written into every exported file.
DISCLAIMER = (
    "Rhythm is NOT transcribed. Note positions encode measured onset times on a fixed "
    "120 BPM grid; the note values are an artefact of the file format. Bar lines are "
    "arbitrary. Positions are rounded to a 1/128 note (about 16 ms), and a note held under "
    "the next is cut where the next starts. Produced by Tab Sampler."
)


@dataclass(frozen=True, slots=True)
class GridEvent:
    """One piece of the timeline: a chord (or a rest, when ``positions`` is empty)."""

    start: int  # units from the beginning
    length: int  # units; always a power of two, never crossing a bar line
    positions: tuple[Position, ...]
    tie_start: bool = False  # tied to the next piece
    tie_stop: bool = False  # continues the previous piece


def _units(seconds: float) -> int:
    return round(seconds * UNITS_PER_SECOND)


def _chords(tab: Sequence[TabNote], window_s: float) -> list[tuple[int, int, tuple[Position, ...]]]:
    """(start, end, positions) per chord, starts strictly increasing."""
    ordered = sorted(tab, key=lambda t: (t.note.onset, t.note.pitch))
    out: list[tuple[int, int, tuple[Position, ...]]] = []
    index = 0
    while index < len(ordered):
        anchor = ordered[index].note.onset
        end = index
        while end < len(ordered) and ordered[end].note.onset - anchor <= window_s:
            end += 1
        members = ordered[index:end]
        start = _units(anchor)
        stop = max(_units(max(t.note.offset for t in members)), start + 1)
        positions = tuple(t.position for t in members)
        if out and start <= out[-1][0]:
            # Two windows rounded to one unit; only when window_s is under a unit.
            prev_start, prev_stop, prev_positions = out[-1]
            out[-1] = (prev_start, max(prev_stop, stop), prev_positions + positions)
        else:
            out.append((start, stop, positions))
        index = end
    return out


def _pieces(start: int, length: int) -> list[tuple[int, int]]:
    """Split [start, start+length) at bar lines, then into powers of two, largest first."""
    out: list[tuple[int, int]] = []
    cursor, end = start, start + length
    while cursor < end:
        bar_end = (cursor // UNITS_PER_BAR + 1) * UNITS_PER_BAR
        remaining = min(end, bar_end) - cursor
        while remaining:
            piece = 1 << (remaining.bit_length() - 1)
            out.append((cursor, piece))
            cursor += piece
            remaining -= piece
    return out


def _stretch(start: int, length: int, positions: tuple[Position, ...]) -> list[GridEvent]:
    pieces = _pieces(start, length)
    tied = bool(positions) and len(pieces) > 1
    last = len(pieces) - 1
    return [
        GridEvent(
            start=s,
            length=n,
            positions=positions,
            tie_start=tied and i < last,
            tie_stop=tied and i > 0,
        )
        for i, (s, n) in enumerate(pieces)
    ]


def to_grid(tab: Sequence[TabNote], window_s: float = 0.03) -> list[GridEvent]:
    """The tab as tied pieces that tile whole bars; at least one bar."""
    chords = _chords(tab, window_s)
    events: list[GridEvent] = []
    cursor = 0
    for i, (start, stop, positions) in enumerate(chords):
        if start > cursor:
            events += _stretch(cursor, start - cursor, ())
        end = min(stop, chords[i + 1][0]) if i + 1 < len(chords) else stop
        events += _stretch(start, end - start, positions)
        cursor = end
    total = max(-(-cursor // UNITS_PER_BAR), 1) * UNITS_PER_BAR
    if total > cursor:
        events += _stretch(cursor, total - cursor, ())
    return events

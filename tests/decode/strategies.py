"""Hypothesis strategies for short, fingerable note-group sequences."""

from __future__ import annotations

from hypothesis import strategies as st

from tabsampler.fingering.states import enumerate_states
from tabsampler.types import Context, NoteEvent, NoteGroup, Tuning

STANDARD = Tuning.STANDARD
CTX = Context(tuning=STANDARD, max_span=4)

# A range where every pitch has at least two candidate positions, so the lattice is
# wide enough for the oracle comparison to be meaningful.
PITCH = st.integers(min_value=45, max_value=72)


def _group(pitches: list[int], onset: float) -> NoteGroup:
    return NoteGroup.of(
        [NoteEvent(onset=onset, offset=onset + 0.4, pitch=p, confidence=1.0) for p in pitches]
    )


@st.composite
def fingerable_group(draw: st.DrawFn, max_notes: int = 3) -> NoteGroup:
    """A group guaranteed to have at least one legal chord state."""
    pitches = draw(st.lists(PITCH, min_size=1, max_size=max_notes, unique=True))
    group = _group(pitches, 0.0)
    if enumerate_states(group, STANDARD, CTX.max_span) == ():
        # Fall back to a single note, which always has candidates in this pitch range.
        group = _group(pitches[:1], 0.0)
    return group


@st.composite
def group_sequence(draw: st.DrawFn, max_groups: int = 6, max_notes: int = 3) -> list[NoteGroup]:
    """1..max_groups fingerable groups, spaced well apart in time."""
    count = draw(st.integers(min_value=1, max_value=max_groups))
    groups: list[NoteGroup] = []
    for index in range(count):
        base = draw(fingerable_group(max_notes=max_notes))
        onset = index * 0.5
        groups.append(
            NoteGroup.of(
                [
                    NoteEvent(onset=onset, offset=onset + 0.4, pitch=n.pitch, confidence=1.0)
                    for n in base.notes
                ]
            )
        )
    return groups

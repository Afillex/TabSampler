"""Best-effort decoding for real audio, with the degradation reported.

Pure: no I/O, no global state.

:func:`tabsampler.decode.forward_backward.decode` raises on a group it cannot finger,
which is right: silently producing nonsense would be worse. But on real input that is
too brittle. Running the very first GuitarSet track surfaced the chord
``[51, 55, 58, 62, 67, 70]`` -- an Eb voicing whose only string assignment uses frets
6 to 11, a span of 5. At ``max_span = 4`` it has no legal state, and one chord killed the
whole file.

Three distinct causes, handled differently:

1. **A note is outside the instrument entirely.** The same track produced MIDI 39, a
   semitone below the open low E, which no string and fret can sound. That is a
   transcriber error about a note the guitar cannot play, so the note is dropped. If
   that empties the group, the group goes too.
2. **The shape needs a wider stretch than the enumeration bound allows.** Legitimate;
   ADR 0011 expects the bound to be permissive and leaves the finer 4-below-12 rule to
   the E3 metric. So the span is relaxed for that group only, and the relaxation is
   counted.
3. **The group cannot be fingered at any stretch** -- more notes than strings, or a
   unison whose pitch has one position. On real audio this is usually the transcriber
   hallucinating a harmonic. The lowest-confidence note is dropped and the group retried.

Nothing here is silent. Every relaxation and every dropped note lands in
:class:`Degradation`, which the caller must report. A dropped note lowers recall, and a
metric quoted without the degradation that produced it is not a measurement.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from tabsampler.decode.forward_backward import decode
from tabsampler.errors import UnfingerableGroupError
from tabsampler.fingering.candidates import candidates
from tabsampler.fingering.states import enumerate_states
from tabsampler.types import Context, FingeringScorer, NoteGroup, TabNote

#: How far the span bound may be relaxed for a single group before we start dropping
#: notes. 8 frets is already beyond what a hand can do; past this the group is not a
#: chord, it is a transcription error.
DEFAULT_MAX_RELAXED_SPAN = 8


@dataclass(frozen=True, slots=True)
class Degradation:
    """What had to be given up to decode this input. Must be reported, not hidden."""

    n_groups: int = 0
    n_groups_relaxed: int = 0
    n_groups_dropped: int = 0
    #: Notes the instrument cannot sound at all (below the lowest string, above the
    #: last fret, or blocked by a capo). A transcriber error, not a fingering problem.
    n_notes_out_of_range: int = 0
    #: Notes removed to make an over-full group fingerable.
    n_notes_dropped: int = 0
    max_span_used: int = 0
    details: tuple[str, ...] = field(default=())

    @property
    def n_notes_lost(self) -> int:
        return self.n_notes_out_of_range + self.n_notes_dropped

    @property
    def is_clean(self) -> bool:
        return self.n_groups_relaxed == 0 and self.n_groups_dropped == 0 and self.n_notes_lost == 0


def prepare_groups(
    groups: Sequence[NoteGroup],
    ctx: Context,
    max_relaxed_span: int = DEFAULT_MAX_RELAXED_SPAN,
) -> tuple[list[NoteGroup], list[int], Degradation]:
    """Make every group fingerable, relaxing the span and then dropping notes.

    Returns the (possibly reduced) groups, the span to use for each, and a record of
    what was given up.
    """
    prepared: list[NoteGroup] = []
    spans: list[int] = []
    details: list[str] = []
    relaxed = 0
    dropped = 0
    out_of_range = 0
    groups_dropped = 0

    for index, group in enumerate(groups):
        # 1. Notes the instrument cannot sound at all.
        playable = [n for n in group.notes if candidates(n.pitch, ctx.tuning)]
        for note in group.notes:
            if not candidates(note.pitch, ctx.tuning):
                out_of_range += 1
                details.append(
                    f"group @{group.onset:.3f}s: dropped pitch {note.pitch}, outside the "
                    f"instrument (open strings {ctx.tuning.open_pitches}, capo "
                    f"{ctx.tuning.capo}, frets 0-{ctx.tuning.max_fret})"
                )
        if not playable:
            groups_dropped += 1
            details.append(f"group @{group.onset:.3f}s: dropped entirely, no note is playable")
            continue

        current = NoteGroup.of(playable)
        span = ctx.max_span
        while not enumerate_states(current, ctx.tuning, span):
            # 2. Relax the stretch before giving anything up.
            if span < max_relaxed_span:
                span += 1
                continue
            # 3. Out of stretch: drop the least confident note and start over.
            if len(current) <= 1:
                raise UnfingerableGroupError(
                    f"group at index {index} (onset {group.onset:.3f}s, pitch "
                    f"{current.notes[0].pitch}) has candidate positions but no legal "
                    f"state up to span {max_relaxed_span}. This should not happen; a "
                    f"single note always fits."
                )
            weakest = min(current.notes, key=lambda n: (n.confidence, -n.pitch))
            remaining = [n for n in current.notes if n is not weakest]
            details.append(
                f"group @{group.onset:.3f}s: dropped pitch {weakest.pitch} "
                f"(confidence {weakest.confidence:.3f}) to make it fingerable"
            )
            dropped += 1
            current = NoteGroup.of(remaining)
            span = ctx.max_span

        if span > ctx.max_span:
            relaxed += 1
            details.append(f"group @{group.onset:.3f}s: span relaxed {ctx.max_span} -> {span}")
        prepared.append(current)
        spans.append(span)

    return (
        prepared,
        spans,
        Degradation(
            n_groups=len(groups),
            n_groups_relaxed=relaxed,
            n_groups_dropped=groups_dropped,
            n_notes_out_of_range=out_of_range,
            n_notes_dropped=dropped,
            max_span_used=max(spans, default=0),
            details=tuple(details),
        ),
    )


def decode_best_effort(
    groups: Sequence[NoteGroup],
    scorer: FingeringScorer,
    ctx: Context,
    temperature: float | None = None,
    max_alternatives: int | None = None,
    max_relaxed_span: int = DEFAULT_MAX_RELAXED_SPAN,
) -> tuple[list[TabNote], Degradation]:
    """Decode real input, reporting what had to be relaxed or dropped.

    Prefer :func:`tabsampler.decode.forward_backward.decode` when the input is known
    fingerable and you want a hard failure instead.
    """
    if not groups:
        return [], Degradation()
    prepared, spans, degradation = prepare_groups(groups, ctx, max_relaxed_span)
    tab = decode(
        prepared,
        scorer,
        ctx,
        temperature=temperature,
        max_alternatives=max_alternatives,
        spans=spans,
    )
    return tab, degradation

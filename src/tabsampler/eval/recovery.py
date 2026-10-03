"""How much of the human fingering a decoder recovers on held-out human tab (ADR 0023).

Pure: no I/O, no global state. Counts are kept per song and per part, so two decoders
scored on the same songs can be compared with
:func:`tabsampler.eval.bootstrap.paired_bootstrap`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from tabsampler.decode.viterbi import viterbi
from tabsampler.eval.playability import PlayabilityRules, group_is_playable
from tabsampler.fingering.candidates import candidates
from tabsampler.fingering.fit import HumanSequence
from tabsampler.types import Context, FingeringScorer


@dataclass(frozen=True, slots=True)
class PartSequence:
    """One human sequence, its song, and whether its part is ``clean`` or ``distorted``."""

    song: str
    part: str
    sequence: HumanSequence


@dataclass(slots=True)
class RecoveryReport:
    """Per song and part, [notes placed where the human placed them, notes]; per part,
    [decoded shapes that pass E3's chord rules, decoded shapes]."""

    per_song: dict[str, dict[str, list[int]]] = field(
        default_factory=dict[str, dict[str, list[int]]]
    )
    shapes: dict[str, list[int]] = field(default_factory=dict[str, list[int]])
    single_candidate: int = 0

    def counts(self, part: str | None = None) -> tuple[int, int]:
        pairs = self.song_counts(part).values()
        return sum(h for h, _ in pairs), sum(n for _, n in pairs)

    def share(self, part: str | None = None) -> float:
        hits, notes = self.counts(part)
        return hits / notes if notes else math.nan

    def song_counts(self, part: str | None = None) -> dict[str, tuple[int, int]]:
        """song -> (recovered, notes), for one part or all parts pooled. A song with no
        notes in that part is left out rather than counted as zero of zero."""
        out: dict[str, tuple[int, int]] = {}
        for song, parts in self.per_song.items():
            chosen = [c for name, c in parts.items() if part is None or name == part]
            if chosen:
                out[song] = (sum(c[0] for c in chosen), sum(c[1] for c in chosen))
        return out

    def chord_shape_rate(self, part: str | None = None) -> float:
        chosen = [c for name, c in self.shapes.items() if part is None or name == part]
        total = sum(c[1] for c in chosen)
        return sum(c[0] for c in chosen) / total if total else math.nan

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_song": self.per_song,
            "shapes": self.shapes,
            "single_candidate": self.single_candidate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecoveryReport:
        per_song: dict[str, dict[str, list[Any]]] = data["per_song"]
        shapes: dict[str, list[Any]] = data["shapes"]
        return cls(
            per_song={
                song: {part: [int(c[0]), int(c[1])] for part, c in parts.items()}
                for song, parts in per_song.items()
            },
            shapes={part: [int(c[0]), int(c[1])] for part, c in shapes.items()},
            single_candidate=int(data["single_candidate"]),
        )


def recover(
    items: Sequence[PartSequence],
    scorer: FingeringScorer,
    ctx: Context,
    rules: PlayabilityRules | None = None,
) -> RecoveryReport:
    """Decode every sequence with Viterbi and count the notes placed as the human did."""
    active = rules or PlayabilityRules()
    report = RecoveryReport()
    for item in items:
        seq = item.sequence
        path, _ = viterbi(seq.groups, scorer, ctx, seq.spans)
        notes = report.per_song.setdefault(item.song, {}).setdefault(item.part, [0, 0])
        shapes = report.shapes.setdefault(item.part, [0, 0])
        for group, truth, guess in zip(seq.groups, seq.states, path, strict=True):
            shapes[0] += group_is_playable(guess.positions, active) is None
            shapes[1] += 1
            for note, t, g in zip(group.notes, truth.positions, guess.positions, strict=True):
                notes[0] += t == g
                notes[1] += 1
                report.single_candidate += len(candidates(note.pitch, ctx.tuning)) == 1
    return report

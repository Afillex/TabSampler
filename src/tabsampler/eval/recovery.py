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
from tabsampler.eval.metrics import PRF
from tabsampler.eval.playability import (
    PlayabilityReport,
    PlayabilityRules,
    group_is_playable,
)
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
    [decoded shapes that pass E3's chord rules, decoded shapes]; and, where they were
    recorded, the same shape counts per song and part, which a paired interval on the
    chord-shape rate needs (ADR 0039)."""

    per_song: dict[str, dict[str, list[int]]] = field(
        default_factory=dict[str, dict[str, list[int]]]
    )
    shapes: dict[str, list[int]] = field(default_factory=dict[str, list[int]])
    single_candidate: int = 0
    per_song_shapes: dict[str, dict[str, list[int]]] = field(
        default_factory=dict[str, dict[str, list[int]]]
    )

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

    def song_shape_counts(self, part: str | None = None) -> dict[str, tuple[int, int]]:
        """song -> (decoded shapes passing E3's chord rules, decoded shapes), for one part
        or all pooled; empty for a report that did not record them per song."""
        out: dict[str, tuple[int, int]] = {}
        for song, parts in self.per_song_shapes.items():
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
            "per_song_shapes": self.per_song_shapes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecoveryReport:
        per_song: dict[str, dict[str, list[Any]]] = data["per_song"]
        shapes: dict[str, list[Any]] = data["shapes"]
        # Written only since ADR 0039; older reports load with none.
        song_shapes: dict[str, dict[str, list[Any]]] = data.get("per_song_shapes", {})
        return cls(
            per_song={
                song: {part: [int(c[0]), int(c[1])] for part, c in parts.items()}
                for song, parts in per_song.items()
            },
            shapes={part: [int(c[0]), int(c[1])] for part, c in shapes.items()},
            single_candidate=int(data["single_candidate"]),
            per_song_shapes={
                song: {part: [int(c[0]), int(c[1])] for part, c in parts.items()}
                for song, parts in song_shapes.items()
            },
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


def add_track(
    report: RecoveryReport, track_id: str, mode: str, e2: PRF, e3: PlayabilityReport
) -> None:
    """One GuitarSet track's E2 and E3 counts, in the report's per-song form (ADR 0037).

    E2 is an F1, so a track contributes ``2 x matches`` over ``estimated + reference``
    notes: pooled over tracks, that ratio is exactly the micro-averaged F1 the harness
    reports. The mode (``oracle`` or ``e2e``) takes the place of a part.
    """
    report.per_song.setdefault(track_id, {})[mode] = [2 * e2.n_match, e2.n_est + e2.n_ref]
    shapes = report.shapes.setdefault(mode, [0, 0])
    shapes[0] += e3.n_groups_pass
    shapes[1] += e3.n_groups
    report.per_song_shapes.setdefault(track_id, {})[mode] = [e3.n_groups_pass, e3.n_groups]

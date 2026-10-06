"""JSON rendering (ADR 0013).

Carries everything the ASCII form cannot: exact times, full posteriors and the ranked
alternatives. This is what the later web UI needs to show uncertainty on hover (spec
D14), and what makes a regression test on a fixed clip possible.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from tabsampler.types import NoteEvent, Position, TabNote, Tuning

#: Bumped when the shape of the document changes, so a consumer can tell.
SCHEMA_VERSION = 1


def tab_to_dict(tab: Sequence[TabNote], tuning: Tuning = Tuning.STANDARD) -> dict[str, Any]:
    """A JSON-ready document for a decoded tab."""
    return {
        "schema_version": SCHEMA_VERSION,
        "tuning": {
            "open_pitches": list(tuning.open_pitches),
            "n_frets": tuning.n_frets,
            "capo": tuning.capo,
        },
        # ADR 0009: no rhythmic notation in v1. Stated in the document so a consumer
        # does not infer a beat grid that is not there.
        "rhythm": "time_positioned_only",
        "notes": [
            {
                "onset": t.note.onset,
                "offset": t.note.offset,
                "pitch": t.note.pitch,
                "transcriber_confidence": t.note.confidence,
                "string": t.position.string,
                "fret": t.position.fret,
                "posterior": t.posterior,
                "alternatives": [
                    {"string": p.string, "fret": p.fret, "posterior": w} for p, w in t.alternatives
                ],
            }
            for t in tab
        ],
    }


def render_json(
    tab: Sequence[TabNote], tuning: Tuning = Tuning.STANDARD, indent: int | None = 2
) -> str:
    """Serialise a decoded tab. ``sort_keys`` makes the output byte-stable."""
    return json.dumps(tab_to_dict(tab, tuning), indent=indent, sort_keys=True)


def tab_from_dict(doc: Mapping[str, Any]) -> tuple[list[TabNote], Tuning]:
    """Read :func:`tab_to_dict`'s document back -- what the page sends to the exporters.

    Keys the server adds (``degradation``, ``metrics``) are ignored.

    Raises:
        ValueError: on another ``schema_version``, a missing or mistyped field, or a note
            whose string and fret do not sound its pitch.
    """
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version {doc.get('schema_version')!r} is not {SCHEMA_VERSION}, "
            f"the only one this version reads"
        )
    try:
        t = doc["tuning"]
        tuning = Tuning(
            open_pitches=tuple(int(p) for p in t["open_pitches"]),
            n_frets=int(t["n_frets"]),
            capo=int(t["capo"]),
        )
        tab = [
            TabNote(
                note=NoteEvent(
                    onset=float(n["onset"]),
                    offset=float(n["offset"]),
                    pitch=int(n["pitch"]),
                    confidence=float(n["transcriber_confidence"]),
                ),
                position=Position(int(n["string"]), int(n["fret"])),
                posterior=float(n["posterior"]),
                alternatives=tuple(
                    (Position(int(a["string"]), int(a["fret"])), float(a["posterior"]))
                    for a in n["alternatives"]
                ),
            )
            for n in doc["notes"]
        ]
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError(f"not a tab document: {exc!r}") from exc
    for i, note in enumerate(tab):
        try:
            sounded = tuning.pitch_at(note.position.string, note.position.fret)
        except IndexError as exc:
            raise ValueError(f"note {i}: {exc}") from exc
        if sounded != note.note.pitch:
            raise ValueError(
                f"note {i}: string {note.position.string} fret {note.position.fret} "
                f"does not sound pitch {note.note.pitch}"
            )
    return tab, tuning

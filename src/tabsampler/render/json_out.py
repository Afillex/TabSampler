"""JSON rendering (ADR 0013).

Carries everything the ASCII form cannot: exact times, full posteriors and the ranked
alternatives. This is what the later web UI needs to show uncertainty on hover (spec
D14), and what makes a regression test on a fixed clip possible.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from tabsampler.types import TabNote, Tuning

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

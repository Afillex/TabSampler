"""Record each DadaGP song's guitar tunings and capos, read from the original GP files.

The DadaGP token files are E-standardised: a fret token is ``GP value + capo - drop shift``,
so pitch survives but the *physical* fingering of a capo'd or drop-tuned track does not, and
the tokens do not say which tracks those are (ADR 0021). The original GuitarPro files do.
This reads them once and writes the per-song verdict the loader filters on.

PyGuitarPro is deliberately not a project dependency; it is only needed for this one-off
pass. Run it in a throwaway environment:

    uv run --no-project --python 3.13 --with pyguitarpro==0.11 \\
        python scripts/dadagp_track_meta.py data/dadagp/DadaGP-v1.1.zip \\
        data/dadagp/track_meta.json

About 18 ms per song, so roughly 8 minutes for all 26,181.
"""

from __future__ import annotations

import io
import itertools
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import guitarpro  # pyright: ignore[reportMissingImports]

ROOT = "DadaGP-v1.1/"
#: General MIDI programs the DadaGP encoder groups as "clean" (24-28) and "distorted"
#: (29-31) guitar. Everything else is bass, drums, leads or pads, which we do not use.
GUITAR_PROGRAMS = range(24, 32)
#: Pitch steps between adjacent strings, high to low, for a standard-tuned 6-string.
#: Compared as differences so that a uniform downtune still counts as standard.
STANDARD_6 = [-5, -4, -5, -5, -5]


def classify(gp_bytes: bytes) -> dict[str, object]:
    song = guitarpro.parse(io.BytesIO(gp_bytes))
    tracks: list[dict[str, object]] = []
    for track in song.tracks:
        if track.isPercussionTrack or track.channel.instrument not in GUITAR_PROGRAMS:
            continue
        values = [s.value for s in track.strings]
        diffs = [b - a for a, b in itertools.pairwise(values)]
        if len(values) == 6 and diffs == STANDARD_6:
            tuning = "standard"
        elif len(values) in (6, 7) and -7 in diffs:
            tuning = "drop"
        else:
            tuning = "other"
        tracks.append({"strings": len(values), "tuning": tuning, "capo": int(track.offset)})

    if not tracks:
        reason = "no guitar track"
    elif any(t["strings"] != 6 for t in tracks):
        reason = "7-string guitar"
    elif any(t["tuning"] != "standard" for t in tracks):
        reason = "drop or other tuning"
    elif any(t["capo"] != 0 for t in tracks):
        reason = "capo"
    else:
        reason = "clean"
    return {"clean": reason == "clean", "reason": reason, "tracks": tracks}


def main() -> None:
    zip_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    archive = zipfile.ZipFile(zip_path)
    names = set(archive.namelist())
    keys: list[str] = []
    for split in ("training", "validation"):
        keys += [e["tokens.txt"] for e in json.loads(archive.read(f"{ROOT}_DadaGP_{split}.json"))]

    meta: dict[str, dict[str, object]] = {}
    reasons: Counter[str] = Counter()
    for index, key in enumerate(keys, 1):
        if ROOT + key not in names:
            meta[key] = {"clean": False, "reason": "missing from archive", "tracks": []}
        else:
            try:
                meta[key] = classify(archive.read(ROOT + key[: -len(".tokens.txt")]))
            except Exception as exc:  # a corrupt file is a data fact, not a crash
                meta[key] = {"clean": False, "reason": f"parse error: {type(exc).__name__}"}
        reasons[str(meta[key]["reason"]).split(":")[0]] += 1
        if index % 2000 == 0:
            print(f"{index}/{len(keys)}", flush=True)

    out_path.write_text(json.dumps(meta, indent=0, sort_keys=True))
    print(f"wrote {out_path}")
    for reason, count in reasons.most_common():
        print(f"  {count:6d}  {reason}")


if __name__ == "__main__":
    main()

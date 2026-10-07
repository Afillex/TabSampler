"""Build full-song mixes of the validation takes over BabySlakh backing (Phase 6, Task 1).

    uv run python -u scripts/build_mixes.py --split validation

For every labelled validation take -- EGSet12, IDMT-SMT-Guitar's licks (electric), GuitarSet's
player 00 (acoustic microphone) -- the backing song chosen by a hash of the take's name among
BabySlakh's validation songs (1-10), its guitars removed, started ``OFFSET_S`` in, resampled to
44.1 kHz and mixed under the take at each level in ``LEVELS_DB`` (guitar against backing, RMS).
Writes ``cache/mixes/<split>/<level>/<set>/<take>.wav`` and an index of each take's backing and
gain. Labels stay the take's own: a mix scales the guitar by one gain and adds other instruments.
The test split -- EGDB, GuitarSet's players 01-05 -- is built only at its pre-registered look,
which is logged before any test audio is read.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

from tabsampler.data.babyslakh import backing, songs
from tabsampler.data.electric import load_egset12, load_idmt_licks
from tabsampler.data.mixes import backing_for, mix

RATE = 44100
LEVELS_DB = (0.0, -6.0)
#: Backing starts this far into its song, past most intros.
OFFSET_S = 30.0
SLAKH = Path("data/babyslakh/babyslakh_16k")


def validation_takes() -> Iterator[tuple[str, str, Path]]:
    """(set, take name, audio path) for every validation take."""
    for take in load_egset12(Path("data/egset12")):
        yield "egset12", take.name, take.audio
    for take in load_idmt_licks(Path("data/idmt-smt-guitar")):
        yield "idmt", take.name, take.audio
    from tabsampler.data.guitarset import load_dataset
    from tabsampler.data.splits import guitarset_validation_ids

    dataset: Any = load_dataset(Path("data/guitarset"))
    for track_id in guitarset_validation_ids():
        yield "guitarset00", track_id, Path(dataset.track(track_id).audio_mic_path)


def test_takes() -> Iterator[tuple[str, str, Path]]:
    """(set, take name, audio path) for every test take: EGDB's direct input, GuitarSet's players
    01-05 (microphone). Only at the pre-registered look; the caller logs it."""
    from tabsampler.data.egdb import load_clips

    clips, _ = load_clips(Path("data/egdb"))
    for clip in clips:
        yield "egdb", clip.clip_id, clip.direct_input
    from tabsampler.data.guitarset import load_dataset
    from tabsampler.data.splits import guitarset_test_ids

    dataset: Any = load_dataset(Path("data/guitarset"))
    for track_id in guitarset_test_ids():
        yield "guitarset_test", track_id, Path(dataset.track(track_id).audio_mic_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--note", default="", help="The pre-registration, for the access log.")
    parser.add_argument("--out", type=Path, default=Path("cache/mixes"))
    args = parser.parse_args()
    backing_songs = songs(SLAKH, args.split)
    names = [s.name for s in backing_songs]
    cache: dict[str, np.ndarray] = {}
    index: list[dict[str, Any]] = []
    if args.split == "test":
        from tabsampler.data.splits import record_test_set_access

        record_test_set_access(
            "build_mixes.py --split test: EGDB's 240 clips and GuitarSet's players 01-05 mixed "
            f"over BabySlakh songs 11-20 (ADR 0066); a pre-registered look. {args.note}".strip()
        )
    takes = test_takes() if args.split == "test" else validation_takes()
    for group, name, audio in takes:
        guitar = np.asarray(librosa.load(audio, sr=RATE, mono=True)[0], np.float32)
        song = backing_for(name, names)
        if song not in cache:
            raw, sr = backing(SLAKH / song)
            raw = raw[int(OFFSET_S * sr) :]
            cache[song] = np.asarray(librosa.resample(raw, orig_sr=sr, target_sr=RATE), np.float32)
        for level in LEVELS_DB:
            mixed, _, gain = mix(guitar, cache[song], level)
            path = args.out / args.split / f"{level:+g}dB" / group / f"{name.replace(' ', '_')}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, mixed, RATE)
            index.append(
                {
                    "set": group,
                    "take": name,
                    "level_db": level,
                    "backing": song,
                    "gain": gain,
                    "mix": str(path),
                    "source": str(audio),
                }
            )
    (args.out / args.split / "index.json").write_text(json.dumps(index, indent=1))
    counts: dict[str, int] = {}
    for row in index:
        counts[row["set"]] = counts.get(row["set"], 0) + 1
    used = sorted({r["backing"] for r in index})
    print(f"{len(index)} mixes written: {counts}; backing songs used {used}")


if __name__ == "__main__":
    main()

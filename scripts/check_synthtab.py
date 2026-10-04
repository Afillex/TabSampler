"""Check that SynthTab's converted note onsets line up with its audio (ADR 0046).

For every usable track of the development set: the lag, within +-0.3 s, at which the audio's
onset strength is highest on average at the labelled onsets. The onset-strength measure itself
reads late: on GuitarSet's player 00, whose onsets are trusted, it reads one 23 ms frame late, so
SynthTab is compared with that, not with zero. A data check, not a metric: no
results row.

    uv run python scripts/check_synthtab.py data/synthtab/SynthTab_Dev
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from tabsampler.data.synthtab import load_tracks

RATE = 22050
MAX_LAG_SECONDS = 0.3


def best_lag(audio: Path, onsets: list[float], hop: int = 512) -> int:
    """The frame lag at which the onset strength is highest on average at ``onsets``."""
    signal, rate = sf.read(audio, dtype="float32")
    envelope = librosa.onset.onset_strength(y=signal, sr=rate, hop_length=hop)
    frames = np.round(np.asarray(onsets) * rate / hop).astype(int)
    scores: dict[int, float] = {}
    reach = round(MAX_LAG_SECONDS * rate / hop)
    for lag in range(-reach, reach + 1):
        shifted = frames + lag
        shifted = shifted[(shifted >= 0) & (shifted < len(envelope))]
        scores[lag] = float(envelope[shifted].mean()) if len(shifted) else -1.0
    return max(scores, key=lambda lag: scores[lag])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--hop", type=int, default=512, help="Samples per frame.")
    args = parser.parse_args()
    tracks, skipped = load_tracks(args.root)
    print(f"{len(tracks)} usable tracks; {len(skipped)} skipped")
    lags: dict[str, list[int]] = defaultdict(list)
    for track in tracks:
        onsets = sorted({note.onset for note, _ in track.notes})
        lags[track.family].append(best_lag(track.audio, onsets, args.hop))
    frame_ms = 1000 * args.hop / RATE
    for family, values in sorted(lags.items()):
        near = sum(abs(v) <= 1 for v in values)
        print(
            f"  {family:22s} {len(values):3d} tracks: median best lag "
            f"{statistics.median(values) * frame_ms:+6.1f} ms; within one frame: {near}"
        )


if __name__ == "__main__":
    main()

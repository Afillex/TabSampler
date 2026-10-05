"""Check Guitar-TECHS's pickup MIDI against its direct-input audio (Phase 4, Task 2).

Per take: the notes dropped as unplayable on their string; notes that start before the previous
note on the same string has ended; the onset lag, within +-0.3 s, at which the audio's onset
strength is highest on average at the labelled onsets; and the share of labelled pitches the
audio confirms -- over the 30-150 ms after the onset (lag corrected), the constant-Q energy at the
pitch is above that a semitone either side. GuitarSet's player 00 (``audio_mic``, its trusted
onsets) is the control for both measures, since onset strength reads a frame late and not every
true note is a local maximum. A data check, not a metric: no results row.

    uv run python scripts/check_guitartechs.py data/guitar-techs
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from tabsampler.audio.windows import BINS_PER_SEMITONE, HOP, LOWEST_MIDI, RATE, track_cqt
from tabsampler.data.guitartechs import load_takes, parse_midi
from tabsampler.types import NoteEvent

MAX_LAG_SECONDS = 0.3
FRAME = HOP / RATE


def load_audio(path: Path | str) -> np.ndarray:
    signal, _ = librosa.load(path, sr=RATE, mono=True)
    return np.asarray(signal, dtype=np.float32)


def best_lag(signal: np.ndarray, onsets: list[float]) -> int:
    """The frame lag at which the onset strength is highest on average at ``onsets``."""
    envelope = librosa.onset.onset_strength(y=signal, sr=RATE, hop_length=HOP)
    frames = np.round(np.asarray(onsets) / FRAME).astype(int)
    reach = round(MAX_LAG_SECONDS / FRAME)
    scores: dict[int, float] = {}
    for lag in range(-reach, reach + 1):
        shifted = frames + lag
        shifted = shifted[(shifted >= 0) & (shifted < len(envelope))]
        scores[lag] = float(envelope[shifted].mean()) if len(shifted) else -1.0
    return max(scores, key=lambda lag: scores[lag])


def confirmed(cqt: np.ndarray, notes: list[NoteEvent], lag: int) -> tuple[int, int]:
    """How many of ``notes`` the audio confirms, of how many could be checked."""
    hits = checked = 0
    for note in notes:
        row = (note.pitch - LOWEST_MIDI) * BINS_PER_SEMITONE
        start = round(note.onset / FRAME) + lag + round(0.03 / FRAME)
        stop = round(note.onset / FRAME) + lag + round(0.15 / FRAME)
        low, high = row - BINS_PER_SEMITONE, row + BINS_PER_SEMITONE
        if low < 0 or high >= cqt.shape[0] or start < 0 or stop > cqt.shape[1]:
            continue
        energy = cqt[:, start:stop].mean(axis=1)
        checked += 1
        hits += bool(energy[row] > energy[low] and energy[row] > energy[high])
    return hits, checked


def overlaps(placed: tuple[Any, ...]) -> int:
    """Notes that start before the previous note on their string has ended."""
    last: dict[int, float] = {}
    count = 0
    for note, position in placed:
        if note.onset < last.get(position.string, -1.0) - 1e-6:
            count += 1
        last[position.string] = max(last.get(position.string, -1.0), note.offset)
    return count


def control() -> None:
    from tabsampler.data.guitarset import load_dataset, reference_notes
    from tabsampler.data.splits import guitarset_validation_ids

    dataset: Any = load_dataset(Path("data/guitarset"))
    lags: list[int] = []
    hits = checked = 0
    for track_id in guitarset_validation_ids():
        track = dataset.track(track_id)
        notes = reference_notes(track)
        signal = load_audio(track.audio_mic_path)
        lags.append(best_lag(signal, sorted({n.onset for n in notes})))
        h, c = confirmed(track_cqt(signal), notes, 0)
        hits, checked = hits + h, checked + c
    print(
        f"control, GuitarSet player 00: median best lag {np.median(lags) * FRAME * 1000:+.1f} ms;"
        f" pitches confirmed {hits}/{checked} ({hits / checked:.3f})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--no-control", action="store_true", help="Skip GuitarSet's player 00.")
    args = parser.parse_args()
    if not args.no_control:
        control()
    takes, skipped = load_takes(args.root)
    print(f"{len(takes)} takes; skipped: {skipped or 'none'}")
    for take in takes:
        midi = take.direct_input.parent.parent.parent / "midi" / f"midi_{take.name}.mid"
        _, dropped = parse_midi(midi)
        signal = load_audio(take.direct_input)
        lag = best_lag(signal, sorted({n.onset for n, _ in take.notes}))
        hits, checked = confirmed(track_cqt(signal), [n for n, _ in take.notes], lag)
        reasons = Counter(reason.split(": ")[1].split(" ")[0] for reason in dropped)
        print(
            f"  P{take.player} {take.category:11s} {take.name:24s} {len(take.notes):5d} notes,"
            f" {len(signal) / RATE / 60:5.1f} min; dropped {len(dropped)} {dict(reasons)};"
            f" overlapping {overlaps(take.notes)}; lag {lag * FRAME * 1000:+6.1f} ms;"
            f" confirmed {hits}/{checked} ({hits / max(checked, 1):.3f})"
        )


if __name__ == "__main__":
    main()

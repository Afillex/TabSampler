"""Guitar-TECHS as Basic Pitch's training data (Phase 5, Task 4; ADR 0056).

    ~/.local/share/tabsampler/bp-train/bin/python scripts/finetune_basic_pitch.py grids \\
        --out cache/transcriber/grids.npz
    uv run python scripts/prepare_transcriber_data.py --grids cache/transcriber/grids.npz \\
        --out cache/transcriber/data

Per usable take (ADR 0051): its direct input at 22,050 Hz, and its cleaned notes with the label
delay taken off (ADR 0055), as Basic Pitch's three targets -- notes and contours over each note's
duration, onsets at its first frame -- built by mirdata's ``NoteData.to_sparse_index`` on Basic
Pitch's own grids (read from the installed package by ``finetune_basic_pitch.py grids``), as its
MAESTRO pipeline builds them. Players 1 and 2 go to ``<out>/train``, player 3 to
``<out>/validation``; one ``.npz`` per take.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import librosa
import numpy as np
from mirdata.annotations import NoteData

from tabsampler.data.guitartechs import aligned, load_takes, usable
from tabsampler.types import NoteEvent, Position

VALIDATION_PLAYER = 3  # ADR 0051


def targets(
    notes: list[tuple[NoteEvent, Position]], duration: float, grids: Any
) -> dict[str, np.ndarray]:
    """The onset, note and contour matrices (frames x bins, uint8) Basic Pitch trains on."""
    hop = float(grids["annotation_hop"])
    time_scale = np.arange(0, duration + hop, hop)
    shapes = {
        "onsets": (len(time_scale), len(grids["freq_bins_notes"])),
        "notes": (len(time_scale), len(grids["freq_bins_notes"])),
        "contours": (len(time_scale), len(grids["freq_bins_contours"])),
    }
    out = {name: np.zeros(shape, dtype=np.uint8) for name, shape in shapes.items()}
    if not notes:
        return out
    data = NoteData(
        np.array([[max(n.onset, 0.0), max(n.offset, n.onset + hop)] for n, _ in notes]),
        "s",
        np.array([float(n.pitch) for n, _ in notes]),
        "midi",
    )
    for name, bins, onsets_only in (
        ("onsets", grids["freq_bins_notes"], True),
        ("notes", grids["freq_bins_notes"], False),
        ("contours", grids["freq_bins_contours"], False),
    ):
        index, _ = data.to_sparse_index(time_scale, "s", bins, "hz", onsets_only=onsets_only)
        if len(index):
            out[name][index[:, 0], index[:, 1]] = 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grids", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("data/guitar-techs"))
    args = parser.parse_args()
    grids = np.load(args.grids)
    rate = int(grids["audio_sample_rate"])
    seconds = {"train": 0.0, "validation": 0.0}
    count = {"train": 0, "validation": 0}
    for take in load_takes(args.root)[0]:
        if not usable(take):
            continue
        side = "validation" if take.player == VALIDATION_PLAYER else "train"
        audio, _ = librosa.load(take.direct_input, sr=rate, mono=True)
        audio = np.asarray(audio, dtype=np.float32)
        notes = list(aligned(take))
        folder = args.out / side
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            folder / f"P{take.player}_{take.category}_{take.name}.npz",
            audio=audio,
            **targets(notes, len(audio) / rate, grids),
        )
        seconds[side] += len(audio) / rate
        count[side] += 1
    for side in ("train", "validation"):
        print(f"{side}: {count[side]} takes, {seconds[side] / 3600:.2f} hours")


if __name__ == "__main__":
    main()

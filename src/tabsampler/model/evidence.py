"""What the string classifier hears, for the decoder's acoustic term (ADRs 0046, 0047, 0062).

One window per note, cut at its onset from the recording's constant-Q transform, through the
classifier and its temperature: six log-probabilities per note, minus infinity on strings that
cannot sound it. Needs the ``model`` dependency group (PyTorch); nothing else imports it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from tabsampler.audio.windows import note_window, possible_strings, track_cqt
from tabsampler.model.strings import StringClassifier, tempered
from tabsampler.types import NoteEvent, Tuning


def load_classifier(run: Path) -> StringClassifier:
    """A trained run's best weights (``<run>/best.pt``), ready to listen."""
    model = StringClassifier()
    model.load_state_dict(torch.load(run / "best.pt", weights_only=True))
    return model.eval()


def string_evidence(
    model: StringClassifier,
    signal: NDArray[np.float32],
    notes: Sequence[NoteEvent],
    tuning: Tuning,
    temperature: float,
) -> dict[NoteEvent, tuple[float, ...]]:
    """Each note's string log-probabilities, heard from ``signal`` (mono, at the windows' rate)
    around its onset. Notes no string can sound are left out."""
    kept: list[NoteEvent] = []
    possible: list[tuple[bool, ...]] = []
    for note in notes:
        mask = possible_strings(note.pitch - tuning.capo, tuning.open_pitches, tuning.max_fret)
        if any(mask):
            kept.append(note)
            possible.append(mask)
    if not kept:
        return {}
    cqt = track_cqt(signal)
    windows = np.stack([note_window(cqt, note.onset, note.pitch) for note in kept])
    with torch.no_grad():
        log_probs = tempered(
            model(
                torch.from_numpy(windows),  # pyright: ignore[reportUnknownMemberType]
                torch.tensor([note.pitch for note in kept]),
                torch.tensor(possible),
            ),
            temperature,
        )
    return {note: tuple(float(v) for v in row) for note, row in zip(kept, log_probs, strict=True)}

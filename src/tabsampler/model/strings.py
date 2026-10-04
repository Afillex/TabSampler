"""The acoustic model's string classifier (ADR 0046).

A small CNN over a note's window (:mod:`tabsampler.audio.windows`), with the note's pitch beside
it, gives six logits; a softmax over only the strings that can sound the pitch turns them into
P(string | audio, pitch). A string that cannot sound the pitch gets a logit so low that its
probability is zero in float32, and a note with one possible string is certain.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

STRINGS = 6

#: The logit of a string that cannot sound the pitch: exp(-1e4) is zero in float32.
IMPOSSIBLE = -1.0e4

#: Pitches are divided by this before entering the head, to sit near [0, 1].
PITCH_SCALE = 100.0


class StringClassifier(nn.Module):
    """``(windows (B, BINS, FRAMES), pitches (B,), possible (B, 6)) -> log P (B, 6)``."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(32 * 16 + 1, 64), nn.ReLU(), nn.Linear(64, STRINGS))

    def forward(self, windows: Tensor, pitches: Tensor, possible: Tensor) -> Tensor:
        seen: Tensor = self.features(windows.unsqueeze(1))
        pitch = (pitches.to(seen.dtype) / PITCH_SCALE).unsqueeze(1)
        logits: Tensor = self.head(torch.cat([seen, pitch], dim=1))
        return torch.log_softmax(logits.masked_fill(~possible, IMPOSSIBLE), dim=1)

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


def tempered(log_probs: Tensor, temperature: float) -> Tensor:
    """``log_probs`` recalibrated at ``temperature``: above 1 flattens them, below sharpens.

    The classifier's log-probabilities differ from its logits by a constant per note, so dividing
    them is temperature scaling. Impossible strings, near -1e4, are left unscaled, so they stay
    impossible at any temperature.
    """
    impossible = log_probs < IMPOSSIBLE / 2
    return torch.log_softmax(torch.where(impossible, log_probs, log_probs / temperature), dim=-1)


def fit_temperature(log_probs: Tensor, labels: Tensor) -> float:
    """The temperature in [0.5, 50] minimising the labels' negative log-likelihood (Task 6)."""
    from scipy.optimize import minimize_scalar  # pyright: ignore[reportUnknownVariableType]

    def nll(log_temperature: float) -> float:
        scaled = tempered(log_probs, float(torch.exp(torch.tensor(log_temperature))))
        return float(-scaled.gather(1, labels.unsqueeze(1)).mean())

    low, high = float(torch.log(torch.tensor(0.5))), float(torch.log(torch.tensor(50.0)))
    result = minimize_scalar(nll, bounds=(low, high), method="bounded")  # pyright: ignore[reportUnknownVariableType]
    best = float(result.x)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownArgumentType, reportUnknownMemberType]
    return float(torch.exp(torch.tensor(best)))

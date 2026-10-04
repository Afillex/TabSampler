"""The acoustic model's string classifier (ADR 0046)."""

from __future__ import annotations

import torch

from tabsampler.audio.windows import WINDOW_BINS, WINDOW_FRAMES
from tabsampler.model.strings import StringClassifier


def inputs(size: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    windows = torch.randn(size, WINDOW_BINS, WINDOW_FRAMES, generator=generator)
    pitches = torch.randint(40, 80, (size,), generator=generator)
    return windows, pitches


def test_probabilities_cover_only_the_strings_that_can_sound_the_pitch() -> None:
    torch.manual_seed(0)
    model = StringClassifier()
    windows, pitches = inputs(3)
    possible = torch.tensor(
        [
            [True, False, False, False, False, False],
            [False, True, True, True, True, True],
            [True, True, False, False, False, False],
        ]
    )
    with torch.no_grad():
        log_probs = model(windows, pitches, possible)
    assert log_probs.shape == (3, 6)
    probs = log_probs.exp()
    assert torch.allclose(probs.sum(dim=1), torch.ones(3))
    assert bool((probs[~possible] < 1e-6).all())
    assert float(probs[0, 0]) == 1.0  # one possible string: certain


def test_a_fixed_seed_gives_the_same_classifier_twice() -> None:
    windows, pitches = inputs(4)
    possible = torch.ones(4, 6, dtype=torch.bool)
    with torch.no_grad():
        torch.manual_seed(3)
        first = StringClassifier()(windows, pitches, possible)
        torch.manual_seed(3)
        second = StringClassifier()(windows, pitches, possible)
    assert torch.equal(first, second)

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


def test_tempering_at_one_changes_nothing_and_high_flattens_the_possible_strings() -> None:
    from tabsampler.model.strings import tempered

    possible = torch.tensor([[True, True, True, False, False, False]])
    with torch.no_grad():
        torch.manual_seed(1)
        log_probs = StringClassifier()(*inputs(1, seed=4), possible)
        assert torch.allclose(tempered(log_probs, 1.0), log_probs, atol=1e-6)
        flat = tempered(log_probs, 1000.0).exp()
    assert torch.allclose(flat[0, :3], torch.full((3,), 1 / 3), atol=1e-2)
    assert bool((flat[0, 3:] < 1e-6).all())


def test_the_fitted_temperature_undoes_overconfidence() -> None:
    # Labels drawn from p; the model reports p sharpened as if at temperature 1/4. Scaling
    # by about 4 restores it.
    from tabsampler.model.strings import fit_temperature

    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(4000, 6, generator=generator)
    labels = torch.multinomial(torch.softmax(logits, dim=1), 1, generator=generator).squeeze(1)
    overconfident = torch.log_softmax(4.0 * logits, dim=1)
    assert abs(fit_temperature(overconfident, labels) - 4.0) < 0.4

"""Estimating and correcting a recording's offset from A440 (plan Task 6)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from tabsampler.audio.tuning import CORRECTION_THRESHOLD, estimate_offset, retune

SR = 22050


def plucks(offset: float, seconds: float = 3.0) -> NDArray[np.float32]:
    """An A minor arpeggio of decaying harmonic tones, every pitch ``offset`` semitones off."""
    t = np.arange(int(SR * 0.5)) / SR
    out: list[NDArray[np.float64]] = []
    for k in range(int(seconds / 0.5)):
        midi = (57, 60, 64, 69)[k % 4] + offset
        f0 = 440.0 * 2 ** ((midi - 69) / 12)
        tone = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in (1, 2, 3, 4))
        out.append(tone * np.exp(-3 * t))
    audio = np.concatenate(out)
    return (0.3 * audio / np.abs(audio).max()).astype(np.float32)


@pytest.mark.parametrize("offset", [0.0, 0.2, -0.3, 0.45])
def test_the_offset_is_recovered(offset: float) -> None:
    assert estimate_offset(plucks(offset), SR) == pytest.approx(offset, abs=0.05)


@pytest.mark.parametrize("offset", [0.3, -0.45])
def test_a_retuned_recording_is_at_a440(offset: float) -> None:
    corrected = retune(plucks(offset), SR, estimate_offset(plucks(offset), SR))
    assert estimate_offset(corrected, SR) == pytest.approx(0.0, abs=0.05)
    assert corrected.dtype == np.float32 and len(corrected) == len(plucks(offset))


def test_an_offset_under_the_threshold_leaves_the_audio_untouched() -> None:
    audio = plucks(0.0)
    assert retune(audio, SR, CORRECTION_THRESHOLD / 2) is audio


def test_the_threshold_is_ten_cents() -> None:
    assert CORRECTION_THRESHOLD == 0.1  # fixed before the measurement (plan Task 6)


def test_silence_is_in_tune() -> None:
    assert estimate_offset(np.zeros(SR, dtype=np.float32), SR) == 0.0

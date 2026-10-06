"""Full-song mixes from a labelled guitar take and a backing song (Phase 6, Task 1)."""

from __future__ import annotations

import numpy as np
import pytest

from tabsampler.data.mixes import backing_for, fit_length, mix


def tone(n: int, freq: float, amp: float, sr: int = 16000) -> np.ndarray:
    return (amp * np.sin(2 * np.pi * freq * np.arange(n) / sr)).astype(np.float32)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


@pytest.mark.parametrize("level_db", [0.0, -6.0, 6.0])
def test_the_guitar_sits_at_the_stated_level_against_the_backing(level_db: float) -> None:
    guitar, backing = tone(16000, 220, 0.3), tone(16000, 55, 0.05)
    mixed, scaled_backing, gain = mix(guitar, backing, level_db)
    assert 20 * np.log10(rms(guitar * gain) / rms(scaled_backing)) == pytest.approx(
        level_db, abs=1e-3
    )
    np.testing.assert_allclose(mixed, guitar * gain + scaled_backing, atol=1e-6)


def test_the_mix_never_clips() -> None:
    mixed, _, gain = mix(tone(16000, 220, 0.9), tone(16000, 330, 0.9), 0.0)
    assert np.abs(mixed).max() <= 0.99 + 1e-6
    assert 0 < gain <= 1


def test_a_short_backing_is_looped_and_a_long_one_cut() -> None:
    backing = np.arange(5, dtype=np.float32)
    assert fit_length(backing, 12).tolist() == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1]
    assert fit_length(backing, 3).tolist() == [0, 1, 2]


def test_a_silent_backing_is_refused() -> None:
    with pytest.raises(ValueError, match="silent"):
        mix(tone(100, 220, 0.3), np.zeros(100, dtype=np.float32), 0.0)


def test_the_backing_song_depends_on_the_take_alone_not_on_order() -> None:
    songs = [f"Track{i:05d}" for i in range(1, 11)]
    picks = {name: backing_for(name, songs) for name in ("egset12 01", "idmt AR_Lick1_FN", "x")}
    assert all(p in songs for p in picks.values())
    assert picks == {name: backing_for(name, list(reversed(songs))) for name in picks}

"""The acoustic model's note windows (ADR 0046): constant-Q, centred on each note's pitch."""

from __future__ import annotations

import numpy as np

from tabsampler.audio.windows import (
    BELOW,
    BINS_PER_SEMITONE,
    FRAMES_BEFORE,
    HOP,
    RATE,
    WINDOW_BINS,
    WINDOW_FRAMES,
    note_window,
    onset_lag,
    possible_strings,
    track_cqt,
)

STANDARD = (40, 45, 50, 55, 59, 64)


def plucked(pitch: int, onset: float, seconds: float = 1.5) -> np.ndarray:
    """A decaying harmonic tone at ``pitch`` from ``onset``, silence before."""
    t = np.arange(int(seconds * RATE)) / RATE
    f0 = 440.0 * 2 ** ((pitch - 69) / 12)
    tone = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 6))
    tone = tone * np.exp(-3 * np.clip(t - onset, 0, None)) * (t >= onset)
    return tone.astype(np.float32)


def test_a_window_has_the_documented_shape() -> None:
    cqt = track_cqt(plucked(52, 0.3))
    assert note_window(cqt, 0.3, 52).shape == (WINDOW_BINS, WINDOW_FRAMES)


def test_the_fundamental_sits_on_the_same_row_whatever_the_pitch() -> None:
    # Centring on the pitch is what lets the network learn timbre rather than pitch.
    for pitch in (45, 52, 64, 76):
        window = note_window(track_cqt(plucked(pitch, 0.3)), 0.3, pitch)
        after = window[:, FRAMES_BEFORE + 2 :]
        row = int(np.argmax(after.mean(axis=1)))
        assert abs(row - BELOW * BINS_PER_SEMITONE) <= 1, (pitch, row)


def test_the_note_starts_after_the_windows_lead_in() -> None:
    window = note_window(track_cqt(plucked(57, 0.5)), 0.5, 57)
    energy = window[BELOW * BINS_PER_SEMITONE]
    assert energy[: FRAMES_BEFORE - 1].mean() < energy[FRAMES_BEFORE + 1 : FRAMES_BEFORE + 6].mean()


def test_windows_at_the_edges_are_padded_not_refused() -> None:
    cqt = track_cqt(plucked(88, 0.0, seconds=1.0))  # long enough for the lowest octaves
    assert note_window(cqt, 0.0, 88).shape == (WINDOW_BINS, WINDOW_FRAMES)
    assert note_window(cqt, 0.99, 39).shape == (WINDOW_BINS, WINDOW_FRAMES)


def test_a_pitch_can_be_sounded_only_on_strings_that_reach_it() -> None:
    assert possible_strings(40, STANDARD, 22) == (True, False, False, False, False, False)
    assert possible_strings(64, STANDARD, 22) == (False, True, True, True, True, True)
    assert possible_strings(64, STANDARD, 24) == (True, True, True, True, True, True)
    assert not any(possible_strings(39, STANDARD, 22))


def test_frames_are_the_documented_length() -> None:
    assert HOP / RATE == 256 / 22050


def test_the_onset_lag_finds_labels_that_run_early_or_late() -> None:
    onsets = [0.5 + 0.7 * i for i in range(8)]
    signal = sum(plucked(45 + 3 * i, t, seconds=6.5) for i, t in enumerate(onsets))
    signal = np.asarray(signal, dtype=np.float32)
    truth = onset_lag(signal, onsets)
    early = onset_lag(signal, [t - 0.06 for t in onsets])
    late = onset_lag(signal, [t + 0.04 for t in onsets])
    frame = 64 / RATE
    assert abs((early - truth) - 0.06) <= frame
    assert abs((late - truth) + 0.04) <= frame

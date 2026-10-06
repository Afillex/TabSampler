"""A recording's offset from A440, and its correction (plan Task 6).

Basic Pitch assumes standard pitch and rounds each note to the nearest semitone, so a guitar
tuned a quarter-tone off has its notes dropped or snapped to the wrong neighbour. The offset is
estimated from the recording's harmonic part, in semitones in [-0.5, 0.5); the correction
pitch-shifts the audio by its negative. Pure: arrays in, arrays out.

Near +-0.5 the direction is ambiguous: a guitar a quarter-tone sharp of E is a quarter-tone flat
of F. A wrong guess moves every note a whole semitone.
"""

from __future__ import annotations

import librosa
import numpy as np
from numpy.typing import NDArray

#: Offsets smaller than this are left alone: 10 cents, fixed before the measurement.
#: ``retune`` is not used by the pipeline: the correction was measured and not adopted (ADR 0060).
CORRECTION_THRESHOLD = 0.1

#: Offsets at least this large are reported to the user as a reason to re-tune (Ege, after ADR
#: 0060). On player 3, Basic Pitch lost nothing at +0.25 semitone and a quarter of E2 at +0.45.
#: A warning only: the tab is unchanged.
WARNING_THRESHOLD = 0.25


def estimate_offset(audio: NDArray[np.float32], sr: int) -> float:
    """Semitones from A440, in [-0.5, 0.5). Silence is in tune."""
    if not np.any(audio):
        return 0.0
    harmonic = librosa.effects.harmonic(audio)  # pyright: ignore[reportUnknownMemberType]
    offset = librosa.estimate_tuning(y=harmonic, sr=sr)  # pyright: ignore[reportUnknownMemberType]
    return float(offset)  # pyright: ignore[reportUnknownArgumentType]


def retune(audio: NDArray[np.float32], sr: int, offset: float) -> NDArray[np.float32]:
    """The audio shifted by ``-offset`` semitones; unchanged under the threshold."""
    if abs(offset) < CORRECTION_THRESHOLD:
        return audio
    shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-offset)  # pyright: ignore[reportUnknownMemberType]
    return np.asarray(shifted, dtype=np.float32)


def tuning_warning(offset: float) -> str:
    """What the user is told about an offset at or over ``WARNING_THRESHOLD``."""
    direction = "sharp" if offset > 0 else "flat"
    return (
        f"The recording is about {abs(offset):.2f} semitone {direction} of standard pitch (A440). "
        f"The transcriber assumes standard pitch, so many notes may be missed or written a "
        f"semitone off. Tune to A440 and record again for a better tab."
    )

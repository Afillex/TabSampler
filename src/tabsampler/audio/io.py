"""Audio loading: stage 0 of the pipeline (spec 2).

Deliberately boring, and deliberately deterministic. Spec 4 requires byte-identical
output for the same input in Phases 0-1, and the transcriber cache keys off the audio,
so every parameter that could change a sample value is pinned rather than left to a
library default that may shift between releases.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from numpy.typing import NDArray

#: basic-pitch operates at 22.05 kHz. Matching it keeps our own analysis aligned with
#: the transcriber's view of the signal, which starts to matter at Phase 3.
DEFAULT_SAMPLE_RATE = 22050

#: Pinned rather than left default: librosa's default resampler has changed across
#: releases, and a change would silently alter every sample we compute.
RESAMPLE_TYPE = "soxr_hq"


def load_audio(path: Path | str, sr: int = DEFAULT_SAMPLE_RATE) -> tuple[NDArray[np.float32], int]:
    """Load ``path`` as mono float32 at ``sr`` Hz.

    Returns the samples and the sample rate actually used, which is always ``sr``.

    Multi-channel input is downmixed by averaging channels. Degenerate input -- an
    empty file, or one shorter than a single analysis frame -- comes back as a short
    array rather than raising, because a caller evaluating a corpus should see a
    legitimate "no notes here" rather than a crash.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"no audio file at {p}")

    data, file_sr = sf.read(p, dtype="float32", always_2d=True)
    mono: NDArray[np.float32] = np.asarray(data, dtype=np.float32).mean(axis=1)

    if file_sr != sr and mono.size > 0:
        mono = librosa.resample(mono, orig_sr=file_sr, target_sr=sr, res_type=RESAMPLE_TYPE).astype(
            np.float32
        )

    return np.ascontiguousarray(mono, dtype=np.float32), sr


def duration_seconds(audio: NDArray[np.float32], sr: int) -> float:
    """Length of ``audio`` in seconds. Used for the E7 runtime metric."""
    return len(audio) / sr

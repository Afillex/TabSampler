"""Full-song mixes: a labelled guitar take over a backing song with its guitars removed (Phase 6).

Pure: arrays in, arrays out, at one sample rate the caller chooses. The guitar keeps its labels,
because a mix only adds other instruments and scales everything by one gain. The guitar and its
backing share no key or tempo -- a limit of building mixes, stated wherever results are.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

#: The mix's peak after scaling, so nothing clips.
PEAK = 0.99

Audio = NDArray[np.float32]


def _rms(x: Audio) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def fit_length(backing: Audio, n: int) -> Audio:
    """``backing`` cut to ``n`` samples, or looped until it is that long."""
    if len(backing) >= n:
        return backing[:n]
    return np.tile(backing, -(-n // len(backing)))[:n]


def mix(guitar: Audio, backing: Audio, level_db: float) -> tuple[Audio, Audio, float]:
    """The mix, the backing as it sits in it, and the gain applied to the guitar.

    The backing is scaled so the guitar is ``level_db`` louder than it (RMS over the take; 0 dB
    equal, -6 dB the guitar half as loud), then both by one gain so the mix peaks at most at
    :data:`PEAK`.

    Raises:
        ValueError: if the backing is silent, so no level can be set.
    """
    backing = fit_length(backing, len(guitar))
    loudness = _rms(backing)
    if loudness == 0.0:
        raise ValueError("the backing is silent over this take")
    scaled = backing * (_rms(guitar) / loudness / 10 ** (level_db / 20))
    mixed = guitar + scaled
    peak = float(np.abs(mixed).max())
    gain = min(1.0, PEAK / peak) if peak > 0 else 1.0
    return (mixed * gain).astype(np.float32), (scaled * gain).astype(np.float32), gain


def backing_for(take: str, songs: Sequence[str]) -> str:
    """The backing song for a take: by a hash of the take's name, so it never depends on order."""
    ordered = sorted(songs)
    return ordered[hashlib.sha1(take.encode()).digest()[0] % len(ordered)]

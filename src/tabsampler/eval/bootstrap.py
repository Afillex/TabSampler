"""Song-level paired bootstrap: is one decoder better than another on the same songs?

Pure: no I/O, no global state. Notes inside one song are not independent -- a decoder that
mis-places a riff mis-places every repeat of it -- so the song is the unit resampled, and
both decoders are always scored on the same resampled songs (ADR 0028).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PairedDifference:
    """``b``'s recovery minus ``a``'s, pooled over notes, with a 95% bootstrap interval."""

    delta: float
    low: float
    high: float
    n_songs: int
    n_notes: int


def paired_bootstrap(
    a: Mapping[str, tuple[int, int]],
    b: Mapping[str, tuple[int, int]],
    n_resamples: int = 2000,
    seed: int = 0,
) -> PairedDifference:
    """Compare two decoders scored on the same songs: song -> (notes recovered, notes).

    Raises:
        ValueError: if the two cover different songs, or a song has different note counts --
            then they were not scored on the same notes and a paired comparison is void.
    """
    if set(a) != set(b):
        raise ValueError("a paired comparison needs the same songs on both sides")
    songs = sorted(a)
    if any(a[s][1] != b[s][1] for s in songs):
        raise ValueError("a song has different note counts on the two sides")
    hits_a = np.array([a[s][0] for s in songs], dtype=float)
    hits_b = np.array([b[s][0] for s in songs], dtype=float)
    notes = np.array([a[s][1] for s in songs], dtype=float)
    delta = float((hits_b.sum() - hits_a.sum()) / notes.sum())
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(songs), size=(n_resamples, len(songs)))
    deltas = (hits_b[picks].sum(axis=1) - hits_a[picks].sum(axis=1)) / notes[picks].sum(axis=1)
    low, high = (float(v) for v in np.percentile(deltas, [2.5, 97.5]))
    return PairedDifference(delta, low, high, len(songs), int(notes.sum()))

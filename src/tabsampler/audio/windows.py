"""The acoustic model's note windows (ADR 0046): constant-Q, centred on each note's pitch.

Pure: no I/O. A track's constant-Q transform is taken once (22,050 Hz, hop 256 -- 11.6 ms; two
bins per semitone from MIDI 28 to 124); each note's window is cropped from it to the 48 semitones
from an octave below its pitch to three octaves above, and to the frames from 35 ms before its
onset to 300 ms after. Centring on the pitch keeps what differs between strings -- the harmonics'
balance, the attack -- in the same place for every note. Log magnitude, normalised per window.
"""

from __future__ import annotations

from collections.abc import Sequence

import librosa
import numpy as np
from numpy.typing import NDArray

RATE = 22050
HOP = 256
BINS_PER_SEMITONE = 2
LOWEST_MIDI = 28
HIGHEST_MIDI = 124
N_BINS = (HIGHEST_MIDI - LOWEST_MIDI) * BINS_PER_SEMITONE

BELOW = 12  # semitones of the window below the note's pitch
ABOVE = 36  # and above it: up to the eighth harmonic
WINDOW_BINS = (BELOW + ABOVE) * BINS_PER_SEMITONE
FRAMES_BEFORE = round(0.035 * RATE / HOP)
FRAMES_AFTER = round(0.300 * RATE / HOP)
WINDOW_FRAMES = FRAMES_BEFORE + FRAMES_AFTER

Matrix = NDArray[np.float32]

#: The transform's lowest octave is computed at a 64th of the rate, with 512-point frames, so a
#: signal needs 2 ** 15 samples (about 1.5 s); a shorter one is padded with silence, which
#: changes no window within it.
MIN_SAMPLES = 2**15


def track_cqt(signal: NDArray[np.float32]) -> Matrix:
    """Log-magnitude constant-Q transform of a mono track at ``RATE``: (N_BINS, frames)."""
    if len(signal) < MIN_SAMPLES:
        signal = np.pad(signal, (0, MIN_SAMPLES - len(signal)))
    spectrum = librosa.cqt(  # pyright: ignore[reportUnknownMemberType]
        signal,
        sr=RATE,
        hop_length=HOP,
        fmin=float(librosa.midi_to_hz(LOWEST_MIDI)),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        n_bins=N_BINS,
        bins_per_octave=12 * BINS_PER_SEMITONE,
    )
    magnitude: NDArray[np.float64] = np.abs(np.asarray(spectrum))
    return np.log1p(100.0 * magnitude).astype(np.float32)


def note_window(cqt: Matrix, onset: float, pitch: int) -> Matrix:
    """The window of one note, ``(WINDOW_BINS, WINDOW_FRAMES)``: zero-padded where it runs past
    the transform's edges, then normalised to zero mean and unit variance."""
    window = np.zeros((WINDOW_BINS, WINDOW_FRAMES), dtype=np.float32)
    first_bin = (pitch - BELOW - LOWEST_MIDI) * BINS_PER_SEMITONE
    first_frame = round(onset * RATE / HOP) - FRAMES_BEFORE
    bins = slice(max(first_bin, 0), min(first_bin + WINDOW_BINS, cqt.shape[0]))
    frames = slice(max(first_frame, 0), min(first_frame + WINDOW_FRAMES, cqt.shape[1]))
    if bins.start < bins.stop and frames.start < frames.stop:
        window[
            bins.start - first_bin : bins.stop - first_bin,
            frames.start - first_frame : frames.stop - first_frame,
        ] = cqt[bins, frames]
    return (window - window.mean()) / (window.std() + 1e-6)


def possible_strings(pitch: int, open_pitches: Sequence[int], max_fret: int) -> tuple[bool, ...]:
    """Which strings, low first, can sound ``pitch`` within ``max_fret`` frets."""
    return tuple(base <= pitch <= base + max_fret for base in open_pitches)

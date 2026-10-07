"""The guitar stem of a full song, by ``htdemucs_6s`` (Phase 6; spec §6).

The separator returns the guitar stem as a file, because Basic Pitch reads files, cached by a hash
of the audio's bytes, the model and the share of the mix added back, as transcriptions are cached
(ADR 0001). ``add_mix`` adds that share of the original mixture to the stem -- a little of it can
cover separation artefacts (Phase 6, Task 3). Demucs is imported only when a real separation runs.
"""

from __future__ import annotations

import functools
import hashlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

#: htdemucs_6s works at 44.1 kHz; the mixture is resampled to it and the stem written at it.
SEPARATED_RATE = 44100
MODEL = "htdemucs_6s"
DEFAULT_CACHE_DIR = Path("cache/stems")

#: (channels, samples) mixture at SEPARATED_RATE -> (channels, samples) guitar stem.
Separate = Callable[[NDArray[np.float32]], NDArray[np.float32]]


@functools.cache
def _model() -> Any:
    """``htdemucs_6s``, loaded once per process: loading it checks its files online every time."""
    from demucs.pretrained import (
        get_model,  # pyright: ignore[reportMissingImports, reportUnknownVariableType]
    )

    model: Any = get_model(MODEL)
    model.eval()
    return model


def demucs_guitar(mixture: NDArray[np.float32], device: str = "cpu") -> NDArray[np.float32]:
    """``htdemucs_6s``'s guitar stem, computed on ``device`` -- "cpu", or "mps" for the Mac's GPU,
    about three times faster on Phase 6's test mixes. Needs the ``separate`` dependency group."""
    import torch
    from demucs.apply import (
        apply_model,  # pyright: ignore[reportMissingImports, reportUnknownVariableType]
    )

    model = _model()
    stereo = mixture if mixture.shape[0] == 2 else np.repeat(mixture[:1], 2, axis=0)
    with torch.no_grad():
        sources = apply_model(  # pyright: ignore[reportUnknownVariableType]
            model,
            torch.from_numpy(stereo)[None],  # pyright: ignore[reportUnknownMemberType]
            device=device,
            split=True,
            overlap=0.25,
        )[0]
    guitar = sources[list(model.sources).index("guitar")]  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    return np.asarray(guitar.numpy(), dtype=np.float32)  # pyright: ignore[reportUnknownMemberType]


class GuitarSeparator:
    """Separates a song's guitar and keeps the stem on disk."""

    def __init__(
        self,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        separate: Separate | None = None,
        device: str = "cpu",
    ):
        """``device`` is where Demucs runs; it cannot be given with a ``separate`` of one's own.

        Raises:
            ValueError: if both ``separate`` and a device other than the CPU are given.
        """
        if separate is not None and device != "cpu":
            raise ValueError("a device is for Demucs; a separator of one's own runs where it runs")
        self.cache_dir = cache_dir
        self.device = device
        self.separate: Separate = separate or functools.partial(demucs_guitar, device=device)

    def stem(self, path: Path, add_mix: float = 0.0) -> Path:
        """The guitar stem of ``path`` as a WAV file, with ``add_mix`` of the mixture added back.

        Raises:
            ValueError: if ``add_mix`` is negative.
        """
        if add_mix < 0:
            raise ValueError(f"add_mix {add_mix} must not be negative")
        key = hashlib.sha256(path.read_bytes() + MODEL.encode()).hexdigest()[:32]
        stem_path = self.cache_dir / f"{key}.wav"
        if not stem_path.is_file():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            _write_whole(stem_path, self.separate(_stereo(path)).T)
        if add_mix == 0:
            return stem_path
        blended_path = self.cache_dir / f"{key}_mix{add_mix:g}.wav"
        if not blended_path.is_file():
            stem, _ = sf.read(stem_path, dtype="float32", always_2d=True)
            mixture = _stereo(path).T[: len(stem)]
            _write_whole(blended_path, stem + add_mix * mixture)
        return blended_path


def _write_whole(path: Path, audio: NDArray[np.floating[Any]]) -> None:
    """Write ``audio`` so that ``path`` exists only once complete: a write cut short (the machine
    lost power during Phase 6's test look) leaves a ``.partial`` file the cache never reads."""
    partial = path.with_name(f"{path.stem}.partial.wav")
    sf.write(partial, audio, SEPARATED_RATE)
    os.replace(partial, path)


def _stereo(path: Path) -> NDArray[np.float32]:
    """The file as (2, samples) at SEPARATED_RATE; a mono file is doubled."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    channels = np.asarray(data, dtype=np.float32).T
    if sr != SEPARATED_RATE:
        channels = np.stack([load_audio_channel(c, int(sr)) for c in channels])
    if channels.shape[0] == 1:
        channels = np.repeat(channels, 2, axis=0)
    return np.ascontiguousarray(channels[:2])


def load_audio_channel(channel: NDArray[np.float32], sr: int) -> NDArray[np.float32]:
    import librosa

    out = librosa.resample(channel, orig_sr=sr, target_sr=SEPARATED_RATE)  # pyright: ignore[reportUnknownMemberType]
    return np.asarray(out, dtype=np.float32)

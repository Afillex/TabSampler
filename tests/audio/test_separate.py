"""The guitar stem of a full song, cached (Phase 6, Task 2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tabsampler.audio.separate import SEPARATED_RATE, GuitarSeparator


def song(tmp_path: Path, seconds: float = 1.0) -> Path:
    t = np.arange(int(SEPARATED_RATE * seconds)) / SEPARATED_RATE
    path = tmp_path / "song.wav"
    sf.write(path, (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), SEPARATED_RATE)
    return path


class Fake:
    """Stands in for htdemucs_6s: the 'guitar' is half the mix."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, mixture: np.ndarray) -> np.ndarray:
        self.calls += 1
        return 0.5 * mixture


def test_the_stem_is_written_and_reused_from_the_cache(tmp_path: Path) -> None:
    fake = Fake()
    separator = GuitarSeparator(cache_dir=tmp_path / "cache", separate=fake)
    first = separator.stem(song(tmp_path))
    second = separator.stem(song(tmp_path))
    assert first == second and first.is_file()
    assert fake.calls == 1


def test_stem_plus_mix_adds_a_share_of_the_mixture_back(tmp_path: Path) -> None:
    path = song(tmp_path)
    separator = GuitarSeparator(cache_dir=tmp_path / "cache", separate=Fake())
    mixture, _ = sf.read(path, dtype="float32", always_2d=True)
    blended, _ = sf.read(separator.stem(path, add_mix=0.25), dtype="float32", always_2d=True)
    # stem 0.5 x mix, plus 0.25 x mix
    np.testing.assert_allclose(blended[:, 0], 0.75 * mixture[:, 0], atol=1e-3)


def test_a_different_share_is_a_different_file(tmp_path: Path) -> None:
    path = song(tmp_path)
    separator = GuitarSeparator(cache_dir=tmp_path / "cache", separate=Fake())
    assert separator.stem(path) != separator.stem(path, add_mix=0.1)


def test_a_negative_share_is_refused(tmp_path: Path) -> None:
    separator = GuitarSeparator(cache_dir=tmp_path / "cache", separate=Fake())
    with pytest.raises(ValueError):
        separator.stem(song(tmp_path), add_mix=-0.1)


def test_the_model_is_loaded_once_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    # Loading htdemucs_6s checks its files online; once per song made a 414-mix run slow and
    # dependent on the network (Phase 6, Task 2).
    pretrained = pytest.importorskip("demucs.pretrained")
    from tabsampler.audio import separate

    loads: list[str] = []

    class Model:
        def eval(self) -> None:
            pass

    def get_model(name: str) -> Model:
        loads.append(name)
        return Model()

    monkeypatch.setattr(pretrained, "get_model", get_model)
    separate._model.cache_clear()  # pyright: ignore[reportPrivateUsage]
    first = separate._model()  # pyright: ignore[reportPrivateUsage]
    assert separate._model() is first  # pyright: ignore[reportPrivateUsage]
    assert loads == ["htdemucs_6s"]
    separate._model.cache_clear()  # pyright: ignore[reportPrivateUsage]

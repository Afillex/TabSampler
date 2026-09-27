"""Tests for audio loading.

Determinism matters here beyond tidiness: spec 4 requires that Phases 0-1 produce
byte-identical output for the same input, and the transcriber cache keys off audio bytes.
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from numpy.typing import NDArray

from tabsampler.audio.io import DEFAULT_SAMPLE_RATE, load_audio

# soundfile writes 16-bit PCM for .wav by default, which quantises a float32 fixture on
# the way to disk and puts ~1/32768 of noise on everything read back. Fixtures therefore
# write float WAVs, so a round trip is exact and these tests measure the loader rather
# than the storage format. One test below uses PCM_16 on purpose.
FLOAT_WAV = "FLOAT"


def write_wav(
    path: Path, data: NDArray[np.float32], sr: int = 44100, subtype: str = FLOAT_WAV
) -> Path:
    sf.write(path, data, sr, subtype=subtype)
    return path


def sine(seconds: float, sr: int = 44100, freq: float = 440.0) -> NDArray[np.float32]:
    t = np.arange(int(seconds * sr)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_mono_file_loads_at_the_requested_rate(tmp_path: Path) -> None:
    p = write_wav(tmp_path / "m.wav", sine(1.0))
    audio, sr = load_audio(p, sr=22050)
    assert sr == 22050
    assert audio.ndim == 1
    assert len(audio) == pytest.approx(22050, rel=0.01)


def test_stereo_is_downmixed_to_mono(tmp_path: Path) -> None:
    left = sine(0.5)
    right = -left  # cancels exactly on a mean downmix
    p = write_wav(tmp_path / "s.wav", np.stack([left, right], axis=1))
    audio, _ = load_audio(p)
    assert audio.ndim == 1
    assert np.abs(audio).max() < 1e-6


def test_resample_length_matches_duration_ratio(tmp_path: Path) -> None:
    p = write_wav(tmp_path / "r.wav", sine(2.0, sr=44100), sr=44100)
    audio, _ = load_audio(p, sr=16000)
    assert len(audio) == pytest.approx(32000, rel=0.01)


def test_no_resampling_happens_when_rates_already_match(tmp_path: Path) -> None:
    data = sine(0.25, sr=22050)
    p = write_wav(tmp_path / "n.wav", data, sr=22050)
    audio, _ = load_audio(p, sr=22050)
    assert len(audio) == len(data)
    np.testing.assert_array_equal(audio, data)  # exact: no resampling, no quantisation


def test_loading_is_deterministic(tmp_path: Path) -> None:
    p = write_wav(tmp_path / "d.wav", sine(0.7, sr=44100), sr=44100)
    a, _ = load_audio(p, sr=22050)
    b, _ = load_audio(p, sr=22050)
    assert np.array_equal(a, b)  # byte-identical, not merely close


def test_silent_audio_loads_as_zeros_not_an_error(tmp_path: Path) -> None:
    p = write_wav(tmp_path / "z.wav", np.zeros(4410, dtype=np.float32))
    audio, _ = load_audio(p)
    assert len(audio) > 0
    assert not np.any(audio)


def test_audio_shorter_than_one_frame_does_not_raise(tmp_path: Path) -> None:
    # A three-sample file is degenerate but legal input; it must come back as a short
    # array rather than blowing up inside the resampler.
    p = write_wav(tmp_path / "t.wav", np.zeros(3, dtype=np.float32), sr=44100)
    audio, sr = load_audio(p, sr=22050)
    assert sr == 22050
    assert audio.ndim == 1
    assert len(audio) <= 3


def test_an_empty_file_returns_an_empty_array(tmp_path: Path) -> None:
    p = write_wav(tmp_path / "e.wav", np.zeros(0, dtype=np.float32), sr=44100)
    audio, _ = load_audio(p)
    assert len(audio) == 0


def test_dtype_is_float32_and_range_is_bounded(tmp_path: Path) -> None:
    p = write_wav(tmp_path / "f.wav", sine(0.3))
    audio, _ = load_audio(p)
    assert audio.dtype == np.float32
    assert np.abs(audio).max() <= 1.0


def test_sixteen_bit_input_loads_and_stays_bounded(tmp_path: Path) -> None:
    # Real recordings are usually 16-bit PCM. Loading one must work and stay in range;
    # values differ from the float source by up to one LSB (~3e-5), which is the
    # storage format's doing, not the loader's.
    data = sine(0.25, sr=22050)
    p = write_wav(tmp_path / "pcm16.wav", data, sr=22050, subtype="PCM_16")
    audio, _ = load_audio(p, sr=22050)
    assert audio.dtype == np.float32
    assert len(audio) == len(data)
    np.testing.assert_allclose(audio, data, atol=1 / 32768)


def test_missing_file_raises_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_audio(tmp_path / "nope.wav")


def test_default_sample_rate_matches_basic_pitch(tmp_path: Path) -> None:
    # basic-pitch operates at 22050 Hz; matching it keeps our own analysis aligned
    # with the transcriber's view of the signal (relevant from Phase 3).
    assert DEFAULT_SAMPLE_RATE == 22050
    p = write_wav(tmp_path / "def.wav", sine(0.2))
    _, sr = load_audio(p)
    assert sr == DEFAULT_SAMPLE_RATE

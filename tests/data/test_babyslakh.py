"""BabySlakh as backing for full-song mixes (Phase 6, Task 1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

from tabsampler.data.babyslakh import backing, songs


def make_song(root: Path, name: str, stems: dict[str, tuple[str, float]]) -> None:
    song = root / name
    (song / "stems").mkdir(parents=True)
    meta = {"stems": {k: {"inst_class": cls} for k, (cls, _) in stems.items()}}
    (song / "metadata.yaml").write_text(yaml.safe_dump(meta))
    for key, (_, level) in stems.items():
        if level >= 0:  # a negative level: listed in the metadata, never rendered
            sf.write(song / "stems" / f"{key}.wav", np.full(1600, level, np.float32), 16000)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    make_song(
        tmp_path,
        "Track00001",
        {
            "S00": ("Guitar", 0.5),
            "S01": ("Drums", 0.1),
            "S02": ("Bass", 0.2),
            "S03": ("Ethnic", 0.4),  # banjo, sitar and the like: plucked strings
            "S04": ("Piano", -1.0),
        },
    )
    make_song(tmp_path, "Track00011", {"S00": ("Bass", 0.3)})
    return tmp_path


def test_the_backing_leaves_out_guitars_and_plucked_strings(root: Path) -> None:
    audio, sr = backing(root / "Track00001")
    assert sr == 16000
    np.testing.assert_allclose(audio, np.full(1600, 0.1 + 0.2, np.float32), atol=1e-4)


def test_songs_1_to_10_are_validation_and_11_to_20_test(root: Path) -> None:
    assert [s.name for s in songs(root, "validation")] == ["Track00001"]
    assert [s.name for s in songs(root, "test")] == ["Track00011"]


def test_a_song_with_nothing_but_guitars_is_refused(tmp_path: Path) -> None:
    make_song(tmp_path, "Track00002", {"S00": ("Guitar", 0.5)})
    with pytest.raises(ValueError, match="Track00002"):
        backing(tmp_path / "Track00002")

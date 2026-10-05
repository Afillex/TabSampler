"""EGDB's labels (ADR 0050): one MIDI track per string, numbered 1 (high e) to 6 (low E)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from tabsampler.data.egdb import load_clips, parse_labels
from tabsampler.types import Position


def write_labels(path: Path, strings: dict[str, list[tuple[float, float, int]]]) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=119.0)
    for name, notes in strings.items():
        track = pretty_midi.Instrument(program=24, name=name)
        for start, end, pitch in notes:
            track.notes.append(pretty_midi.Note(velocity=90, pitch=pitch, start=start, end=end))
        midi.instruments.append(track)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))


def test_track_one_is_the_high_string_and_six_the_low(tmp_path: Path) -> None:
    path = tmp_path / "1.midi"
    write_labels(path, {"1": [(2.0, 2.5, 69)], "5": [(1.0, 1.5, 52)], "6": [(1.0, 1.5, 40)]})
    notes, dropped = parse_labels(path)
    assert dropped == []
    assert [(round(n.onset, 2), n.pitch, p) for n, p in notes] == [
        (1.0, 40, Position(string=0, fret=0)),
        (1.0, 52, Position(string=1, fret=7)),
        (2.0, 69, Position(string=5, fret=5)),
    ]


def test_a_pitch_the_string_cannot_sound_is_dropped_with_its_reason(tmp_path: Path) -> None:
    path = tmp_path / "2.midi"
    write_labels(path, {"2": [(0.0, 0.5, 58), (1.0, 1.5, 60)]})
    notes, dropped = parse_labels(path)
    assert [(n.pitch, p) for n, p in notes] == [(60, Position(string=4, fret=1))]
    assert dropped == ["2: pitch 58 below the open string"]


def test_a_track_that_names_no_string_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "3.midi"
    write_labels(path, {"7": [(0.0, 0.5, 40)]})
    with pytest.raises(ValueError, match="'7'"):
        parse_labels(path)


def test_clips_are_named_by_their_test_id_and_need_their_audio(tmp_path: Path) -> None:
    root = tmp_path / "egdb"
    for number in (1, 17):
        write_labels(root / "audio_label" / f"{number}.midi", {"6": [(0.1, 0.4, 40)]})
    (root / "audio_DI").mkdir()
    sf.write(root / "audio_DI" / "17.wav", np.zeros(100), 44100)
    clips, skipped = load_clips(root)
    assert [(c.clip_id, len(c.notes), c.direct_input.name) for c in clips] == [
        ("egdb_017", 1, "17.wav")
    ]
    assert skipped == [("egdb_001", "no direct-input audio")]

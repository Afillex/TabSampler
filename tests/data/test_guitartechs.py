"""Guitar-TECHS's labels (ADR 0049): one MIDI track per string, named e B G D A E."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from tabsampler.data.guitartechs import STANDARD, load_takes, parse_midi
from tabsampler.types import Position


def write_midi(path: Path, strings: dict[str, list[tuple[float, float, int]]]) -> None:
    """A MIDI file with one named track per string: name -> [(start s, end s, pitch)]."""
    midi = pretty_midi.PrettyMIDI(initial_tempo=60.0)
    for name, notes in strings.items():
        track = pretty_midi.Instrument(program=0, name=name)
        for start, end, pitch in notes:
            track.notes.append(pretty_midi.Note(velocity=90, pitch=pitch, start=start, end=end))
        midi.instruments.append(track)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))


def test_each_track_is_a_string_and_the_fret_follows_from_the_pitch(tmp_path: Path) -> None:
    path = tmp_path / "take.mid"
    write_midi(path, {"e": [(1.0, 1.5, 67)], "E": [(0.5, 1.0, 40)], "G": [(1.0, 2.0, 57)]})
    notes, dropped = parse_midi(path)
    assert dropped == []
    assert [(round(n.onset, 3), n.pitch, p) for n, p in notes] == [
        (0.5, 40, Position(string=0, fret=0)),
        (1.0, 57, Position(string=3, fret=2)),
        (1.0, 67, Position(string=5, fret=3)),
    ]
    assert notes[0][0].offset == pytest.approx(1.0, abs=1e-3)


def test_a_pitch_the_string_cannot_sound_is_dropped_with_its_reason(tmp_path: Path) -> None:
    path = tmp_path / "take.mid"
    write_midi(path, {"A": [(0.0, 0.5, 44), (1.0, 1.5, 45)], "B": [(0.0, 0.5, 59 + 25)]})
    notes, dropped = parse_midi(path)
    assert [(n.pitch, p.string) for n, p in notes] == [(45, 1)]
    assert sorted(dropped) == ["A: pitch 44 below the open string", "B: fret 25 above 24"]


def test_a_track_that_names_no_string_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "take.mid"
    write_midi(path, {"Bass": [(0.0, 0.5, 40)]})
    with pytest.raises(ValueError, match="Bass"):
        parse_midi(path)


def test_standard_tuning_low_string_first() -> None:
    assert STANDARD == (40, 45, 50, 55, 59, 64)


def test_takes_are_found_by_player_category_and_name(tmp_path: Path) -> None:
    root = tmp_path / "guitar-techs"
    for player, category, name in [(1, "singlenotes", "allsinglenotes"), (3, "music", "song1")]:
        folder = root / f"P{player}_{category}"
        write_midi(folder / "midi" / f"midi_{name}.mid", {"E": [(0.1, 0.4, 40)]})
        for kind in ("directinput", "micamp"):
            (folder / "audio" / kind).mkdir(parents=True)
            sf.write(folder / "audio" / kind / f"{kind}_{name}.wav", np.zeros(100), 22050)
    (root / "P3_music" / "midi" / "midi_unpaired.mid").write_bytes(b"")
    takes, skipped = load_takes(root)
    assert [(t.player, t.category, t.name, len(t.notes)) for t in takes] == [
        (1, "singlenotes", "allsinglenotes", 1),
        (3, "music", "song1", 1),
    ]
    assert takes[0].direct_input.name == "directinput_allsinglenotes.wav"
    assert takes[0].mic_amp is not None and takes[0].mic_amp.name == "micamp_allsinglenotes.wav"
    assert skipped == [("P3_music/unpaired", "no direct-input audio")]

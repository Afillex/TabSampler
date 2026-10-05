"""EGDB's labels (ADRs 0050, 0053): the MIDI channel is the string, 0 the high e to 5 the low E."""

from __future__ import annotations

from pathlib import Path

import mido
import numpy as np
import soundfile as sf

from tabsampler.data.egdb import load_clips, parse_labels
from tabsampler.types import Position

TICKS = 480  # per beat, at 120 bpm: 960 ticks to the second


def write_labels(path: Path, tracks: list[tuple[str, list[tuple[float, float, int, int]]]]) -> None:
    """Named tracks of (start s, end s, pitch, channel), at 120 bpm."""
    midi = mido.MidiFile(ticks_per_beat=TICKS)
    midi.tracks.append(mido.MidiTrack([mido.MetaMessage("set_tempo", tempo=500000, time=0)]))
    for name, notes in tracks:
        events = []
        for start, end, pitch, channel in notes:
            events.append((round(start * 960), "note_on", pitch, channel, 90))
            events.append((round(end * 960), "note_off", pitch, channel, 0))
        track = mido.MidiTrack([mido.MetaMessage("track_name", name=name, time=0)])
        now = 0
        for tick, kind, pitch, channel, velocity in sorted(events):
            track.append(
                mido.Message(kind, note=pitch, channel=channel, velocity=velocity, time=tick - now)
            )
            now = tick
        midi.tracks.append(track)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(str(path))


def test_the_channel_is_the_string_whatever_the_track_is_named(tmp_path: Path) -> None:
    path = tmp_path / "1.midi"
    write_labels(
        path,
        [("", [(2.0, 2.5, 69, 0)]), ("6i", [(1.0, 1.5, 52, 4)]), ("5", [(1.0, 1.5, 40, 5)])],
    )
    notes, dropped = parse_labels(path)
    assert dropped == []
    assert [(round(n.onset, 3), round(n.offset, 3), n.pitch, p) for n, p in notes] == [
        (1.0, 1.5, 40, Position(string=0, fret=0)),
        (1.0, 1.5, 52, Position(string=1, fret=7)),
        (2.0, 2.5, 69, Position(string=5, fret=5)),
    ]


def test_a_pitch_its_string_cannot_sound_is_dropped_with_its_reason(tmp_path: Path) -> None:
    path = tmp_path / "2.midi"
    write_labels(path, [("", [(0.0, 0.5, 58, 1), (1.0, 1.5, 60, 1), (2.0, 2.5, 40, 9)])])
    notes, dropped = parse_labels(path)
    assert [(n.pitch, p) for n, p in notes] == [(60, Position(string=4, fret=1))]
    assert dropped == ["channel 1: pitch 58 below the open string", "channel 9: no string"]


def test_clips_are_named_by_their_test_id_and_need_their_audio(tmp_path: Path) -> None:
    root = tmp_path / "egdb"
    for number in (1, 17):
        write_labels(root / "audio_label" / f"{number}.midi", [("", [(0.1, 0.4, 40, 5)])])
    (root / "audio_DI").mkdir()
    sf.write(root / "audio_DI" / "17.wav", np.zeros(100), 44100)
    clips, skipped = load_clips(root)
    assert [(c.clip_id, len(c.notes), c.direct_input.name) for c in clips] == [
        ("egdb_017", 1, "17.wav")
    ]
    assert skipped == [("egdb_001", "no direct-input audio")]

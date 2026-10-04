"""SynthTab's labels (ADR 0046): Guitar Pro ticks, strings numbered from the top, any tuning."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tabsampler.data.synthtab import load_tracks, parse_jams, ticks_to_seconds
from tabsampler.types import Position


def jams(
    strings: list[tuple[int, int, list[tuple[float, float, int]]]], tempo: list[tuple[float, float]]
) -> dict:  # type: ignore[type-arg]
    """A JAMS dict: per string (Guitar Pro number, open pitch, [(tick, ticks long, fret)])."""
    annotations = [
        {
            "namespace": "tempo",
            "data": [
                {"time": t, "duration": 0.0, "value": bpm, "confidence": 1.0} for t, bpm in tempo
            ],
        }
    ]
    for number, open_pitch, notes in strings:
        annotations.append(
            {
                "namespace": "note_tab",
                "sandbox": {"string_index": number, "open_tuning": open_pitch},
                "data": [
                    {
                        "time": t,
                        "duration": d,
                        "value": {"fret": f, "velocity": 95},
                        "confidence": None,
                    }
                    for t, d, f in notes
                ],
            }
        )
    return {"annotations": annotations, "file_metadata": {"duration": 0.0}, "sandbox": {}}


def test_ticks_are_960_to_the_quarter_at_the_files_tempo() -> None:
    assert ticks_to_seconds(960, [(0.0, 120.0)]) == pytest.approx(0.5)
    # 1,920 ticks at 120 BPM is one second; 960 more at 60 BPM is another.
    assert ticks_to_seconds(2880, [(0.0, 120.0), (1920.0, 60.0)]) == pytest.approx(2.0)


def test_strings_count_from_the_top_and_pitch_is_the_open_string_plus_the_fret() -> None:
    notes, open_pitches = parse_jams(
        jams([(6, 40, [(960, 480, 3)]), (1, 64, [(0, 960, 0)])], tempo=[(0.0, 120.0)])
    )
    by_string = {position.string: (note, position) for note, position in notes}
    low, high = by_string[0], by_string[5]
    assert low[1] == Position(0, 3) and low[0].pitch == 43
    assert low[0].onset == pytest.approx(0.5) and low[0].offset == pytest.approx(0.75)
    assert high[1] == Position(5, 0) and high[0].pitch == 64 and high[0].onset == 0.0
    assert [note.onset for note, _ in notes] == sorted(note.onset for note, _ in notes)
    assert open_pitches[0] == 40 and open_pitches[5] == 64


def test_another_tuning_comes_through_from_the_file() -> None:
    notes, open_pitches = parse_jams(jams([(1, 63, [(0, 960, 2)])], tempo=[(0.0, 120.0)]))
    assert notes[0][0].pitch == 65 and open_pitches[5] == 63


def test_a_seventh_string_is_refused() -> None:
    with pytest.raises(ValueError, match="string"):
        parse_jams(jams([(7, 35, [(0, 960, 0)])], tempo=[(0.0, 120.0)]))


def test_unusable_tracks_are_skipped_with_their_reason(tmp_path: Path) -> None:
    for name in ("Labelled Song", "Unlabelled Song", "Seven String Song"):
        folder = tmp_path / "acoustic" / "taylor_pick" / name / name
        folder.mkdir(parents=True)
        sf.write(folder / "taylor_pick_body.flac", np.zeros(2205), 22050)
    for name, strings in (
        ("Labelled Song", [(6, 40, [(960, 480, 3)])]),
        ("Seven String Song", [(7, 35, [(0, 960, 0)])]),
    ):
        folder = tmp_path / "jams" / name
        folder.mkdir(parents=True)
        (folder / "1 - Guitar.jams").write_text(json.dumps(jams(strings, tempo=[(0.0, 120.0)])))
    tracks, skipped = load_tracks(tmp_path)
    assert [t.name for t in tracks] == ["Labelled Song"]
    assert tracks[0].family == "acoustic" and tracks[0].tone == "taylor_pick"
    reasons = dict(skipped)
    assert (
        reasons["Unlabelled Song"] == "no label file" and "string 7" in reasons["Seven String Song"]
    )

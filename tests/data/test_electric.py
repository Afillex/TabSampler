"""Loaders for EGFxSet, EGSet12 and IDMT-SMT-Guitar (plan 2026-10-06-electric-audio-evidence)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tabsampler.data.electric import egfxset_notes, load_egset12, load_idmt_licks
from tabsampler.types import Position


def idmt_xml(events: list[tuple[int, float, float, int, int, str]]) -> str:
    body = "".join(
        f"<event><pitch>{p}</pitch><onsetSec>{on}</onsetSec><offsetSec>{off}</offsetSec>"
        f"<fretNumber>{fret}</fretNumber><stringNumber>{s}</stringNumber>"
        f"<excitationStyle>PK</excitationStyle><expressionStyle>{style}</expressionStyle></event>"
        for p, on, off, fret, s, style in events
    )
    return f"<instrumentRecording><transcription>{body}</transcription></instrumentRecording>"


@pytest.fixture
def idmt(tmp_path: Path) -> Path:
    root = tmp_path / "IDMT-SMT-GUITAR_V2" / "dataset2"
    (root / "annotation").mkdir(parents=True)
    (root / "audio").mkdir()
    takes = {
        # IDMT counts strings from the low E: pitch 50 on string 1 is fret 10.
        "AR_Lick1_FN": [(50, 0.2, 0.5, 10, 1, "NO"), (64, 0.6, 0.9, 0, 6, "NO")],
        "AR_Lick2_FVSH": [(76, 0.2, 0.5, 12, 6, "HA")],  # a harmonic: not a fretted pitch
        "AR_E_fret_0-20": [(40, 0.2, 0.5, 0, 1, "NO")],  # a drill, not a lick
    }
    for name, events in takes.items():
        (root / "annotation" / f"{name}.xml").write_text(idmt_xml(events))
        (root / "audio" / f"{name}.wav").write_bytes(b"")
    return tmp_path


def test_idmt_licks_count_strings_from_the_low_e(idmt: Path) -> None:
    (take,) = load_idmt_licks(idmt)
    assert take.name == "idmt AR_Lick1_FN"
    assert [p for _, p in take.notes] == [Position(0, 10), Position(5, 0)]
    assert take.audio.name == "AR_Lick1_FN.wav"


def test_idmt_takes_with_any_technique_or_drills_are_left_out(idmt: Path) -> None:
    assert [t.name for t in load_idmt_licks(idmt)] == ["idmt AR_Lick1_FN"]


def test_an_idmt_note_whose_fret_does_not_sound_its_pitch_is_refused(idmt: Path) -> None:
    bad = idmt / "IDMT-SMT-GUITAR_V2" / "dataset2" / "annotation" / "LP_Lick3_FN.xml"
    bad.write_text(idmt_xml([(51, 0.1, 0.4, 10, 1, "NO")]))
    with pytest.raises(ValueError, match="LP_Lick3_FN"):
        load_idmt_licks(idmt)


def test_egset12_reads_one_string_per_note_midi_annotation(tmp_path: Path) -> None:
    annotations = [
        {
            "namespace": "note_midi",
            "annotation_metadata": {"data_source": str(s)},
            "data": [
                {"time": 0.5 + s, "duration": 0.25, "value": 40 + 5 * s + 2, "confidence": None}
            ]
            if s in (0, 5)
            else [],
        }
        for s in range(6)
    ]
    (tmp_path / "01.jams").write_text(json.dumps({"annotations": annotations}))
    (tmp_path / "01.wav").write_bytes(b"")
    (take,) = load_egset12(tmp_path)
    assert take.name == "egset12 01"
    # String 0 is the low E: 42 on it is fret 2; on string 5 (high e, 64), 67 is fret 3.
    assert [(n.pitch, p) for n, p in take.notes] == [(42, Position(0, 2)), (67, Position(5, 3))]


def test_egfxset_counts_strings_from_the_high_e(tmp_path: Path) -> None:
    for name in ("1-0.wav", "6-5.wav", "notes.txt"):
        (tmp_path / "Clean" / "Neck").mkdir(parents=True, exist_ok=True)
        (tmp_path / "Clean" / "Neck" / name).write_bytes(b"")
    found = sorted((p.name, pitch, pos) for p, pitch, pos in egfxset_notes(tmp_path))
    # Measured from the audio: EGFxSet's string 1 sounds MIDI 64 open, string 6 MIDI 40.
    assert found == [("1-0.wav", 64, Position(5, 0)), ("6-5.wav", 45, Position(0, 5))]

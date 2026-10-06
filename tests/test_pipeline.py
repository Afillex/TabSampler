"""The one pipeline both front ends call (ADR 0058)."""

from __future__ import annotations

from pathlib import Path

from tabsampler.config import load_phase1_config
from tabsampler.decode.robust import decode_best_effort
from tabsampler.fingering.candidates import group_notes
from tabsampler.fingering.costs import HandSetScorer
from tabsampler.pipeline import transcribe_path
from tabsampler.types import NoteEvent

CFG = load_phase1_config(Path("configs/decoder_clean.yaml"))

# E2, G3 and B3, one after another: an easy line with a unique best fingering per note.
NOTES = [
    NoteEvent(onset=0.0, offset=0.4, pitch=40, confidence=0.9),
    NoteEvent(onset=0.5, offset=0.9, pitch=55, confidence=0.9),
    NoteEvent(onset=1.0, offset=1.4, pitch=59, confidence=0.9),
]


class Fake:
    def __init__(self, notes: list[NoteEvent]) -> None:
        self.notes = notes
        self.calls: list[Path] = []

    def transcribe_file(self, path: Path) -> list[NoteEvent]:
        self.calls.append(path)
        return list(self.notes)


def test_the_tab_is_what_the_decoder_gives_directly() -> None:
    result = transcribe_path(Path("x.wav"), CFG, Fake(NOTES))
    groups = group_notes(NOTES, window_s=CFG.group_window_s)
    expected, _ = decode_best_effort(groups, HandSetScorer(weights=CFG.weights), CFG.context)
    assert list(result.tab) == expected
    assert result.n_notes_detected == 3
    assert result.degradation.is_clean
    assert result.pitch_validity == 1.0
    assert 0.0 <= result.group_rate <= 1.0 and 0.0 <= result.transition_rate <= 1.0


def test_the_file_is_handed_to_the_transcriber() -> None:
    fake = Fake(NOTES)
    transcribe_path(Path("take.wav"), CFG, fake)
    assert fake.calls == [Path("take.wav")]


def test_a_note_the_guitar_cannot_sound_is_counted_not_hidden() -> None:
    low = NoteEvent(onset=2.0, offset=2.3, pitch=30, confidence=0.9)  # below the low E
    result = transcribe_path(Path("x.wav"), CFG, Fake([*NOTES, low]))
    assert result.degradation.n_notes_out_of_range == 1
    assert result.n_notes_detected == 4
    assert len(result.tab) == 3


def test_no_notes_gives_an_empty_tab() -> None:
    result = transcribe_path(Path("x.wav"), CFG, Fake([]))
    assert result.tab == ()
    assert result.n_notes_detected == 0
    assert result.degradation.is_clean

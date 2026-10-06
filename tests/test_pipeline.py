"""The one pipeline both front ends call (ADR 0058)."""

from __future__ import annotations

from pathlib import Path

import pytest

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


def test_the_tuning_offset_is_reported_and_the_tab_is_unchanged() -> None:
    # ADR 0060: the offset is a warning, never a correction.
    plain = transcribe_path(Path("x.wav"), CFG, Fake(NOTES), estimate=lambda _: None)
    sharp = transcribe_path(Path("x.wav"), CFG, Fake(NOTES), estimate=lambda _: 0.45)
    assert sharp.tuning_offset == 0.45 and plain.tuning_offset is None
    assert sharp.tab == plain.tab


def test_an_unreadable_file_still_transcribes_with_the_offset_unknown(tmp_path: Path) -> None:
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"not audio")
    result = transcribe_path(broken, CFG, Fake(NOTES))
    assert result.tuning_offset is None
    assert len(result.tab) == 3


def test_the_offset_is_estimated_from_the_file() -> None:
    from tabsampler.pipeline import offset_from_file

    take = Path("tests/fixtures/sharp_take.wav")
    assert offset_from_file(take) == pytest.approx(0.4, abs=0.05)


def test_the_tuning_check_runs_while_the_transcriber_works() -> None:
    # The check takes about as long as Basic Pitch on a long file; run one after the other
    # they doubled the wait (plan 2026-10-06-optimise-current, Task 1).
    import threading

    started = threading.Event()

    def estimate(_: Path) -> float:
        started.set()
        return 0.0

    class WaitsForTheCheck(Fake):
        def transcribe_file(self, path: Path, /) -> list[NoteEvent]:
            assert started.wait(timeout=5), "the tuning check did not start alongside"
            return super().transcribe_file(path)

    result = transcribe_path(Path("x.wav"), CFG, WaitsForTheCheck(NOTES), estimate=estimate)
    assert result.tuning_offset == 0.0 and len(result.tab) == 3


def test_audio_evidence_moves_notes_to_the_strings_it_hears() -> None:
    # A config with an evidence block decodes with what the classifier hears (plan Task 3).
    from dataclasses import replace

    from tabsampler.config import EvidenceConfig

    g3 = [NoteEvent(onset=0.0, offset=0.4, pitch=55, confidence=0.9)]  # open G, or D fret 5
    plain = transcribe_path(Path("x.wav"), CFG, Fake(g3), estimate=lambda _: None)
    assert plain.tab[0].position.string == 3  # the decoder alone plays it open

    def hear(_: Path, notes: list[NoteEvent]) -> dict[NoteEvent, tuple[float, ...]]:
        sure_d = (-30.0, -30.0, 0.0, -30.0, -30.0, -30.0)  # certain it is the D string
        return dict.fromkeys(notes, sure_d)

    cfg = replace(
        CFG,
        weights=replace(CFG.weights, acoustic=1.0),
        evidence=EvidenceConfig(run=Path("unused"), temperature=1.0),
    )
    heard = transcribe_path(Path("x.wav"), cfg, Fake(g3), estimate=lambda _: None, hear=hear)
    assert heard.tab[0].position.string == 2


def test_without_an_evidence_block_nothing_is_heard() -> None:
    def hear(_: Path, __: list[NoteEvent]) -> dict[NoteEvent, tuple[float, ...]]:
        raise AssertionError("no evidence was asked for")

    transcribe_path(Path("x.wav"), CFG, Fake(NOTES), estimate=lambda _: None, hear=hear)

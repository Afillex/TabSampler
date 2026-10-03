"""Tests for per-song, per-part recovery of human fingerings."""

from __future__ import annotations

from tabsampler.eval.recovery import PartSequence, RecoveryReport, recover
from tabsampler.fingering.costs import HandSetScorer
from tabsampler.fingering.fit import HumanSequence
from tabsampler.types import ChordState, Context, NoteEvent, NoteGroup, Position, Tuning

CTX = Context(tuning=Tuning.STANDARD, max_span=5)


def one_note(pitch: int, human: Position) -> HumanSequence:
    group = NoteGroup.of([NoteEvent(onset=0.0, offset=0.4, pitch=pitch, confidence=1.0)])
    return HumanSequence((group,), (ChordState(positions=(human,)),), (5,))


def test_recovery_counts_hits_per_song_and_part() -> None:
    # Hand-set weights reward open strings, so the decoder plays E4 (64) on the open e.
    items = [
        PartSequence("song1", "clean", one_note(64, Position(5, 0))),  # recovered
        PartSequence("song1", "distorted", one_note(64, Position(4, 5))),  # not
        PartSequence("song2", "clean", one_note(64, Position(4, 5))),  # not
    ]
    report = recover(items, HandSetScorer(), CTX)
    assert report.counts() == (1, 3)
    assert report.counts("clean") == (1, 2)
    assert report.song_counts("clean") == {"song1": (1, 1), "song2": (0, 1)}
    assert report.song_counts("distorted") == {"song1": (0, 1)}
    assert report.chord_shape_rate() == 1.0


def test_a_report_survives_a_round_trip_through_plain_data() -> None:
    items = [PartSequence("s", "clean", one_note(64, Position(5, 0)))]
    report = recover(items, HandSetScorer(), CTX)
    assert RecoveryReport.from_dict(report.to_dict()) == report

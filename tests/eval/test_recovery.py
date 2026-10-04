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


def test_per_track_counts_pool_to_the_reported_f1() -> None:
    # GuitarSet tracks are compared by E2's F1: per track, 2 x matches over estimated plus
    # reference notes, so the pooled ratio is exactly the micro-averaged F1 (ADR 0037).
    import pytest

    from tabsampler.eval.metrics import PRF
    from tabsampler.eval.playability import PlayabilityReport
    from tabsampler.eval.recovery import add_track

    report = RecoveryReport()
    shapes = PlayabilityReport(n_groups=10, n_groups_pass=9, n_transitions=9, n_transitions_pass=9)
    add_track(report, "t0", "e2e", PRF(0.82, 0.8, 0.81, n_ref=100, n_est=98, n_match=80), shapes)
    add_track(report, "t1", "e2e", PRF(0.6, 0.6, 0.6, n_ref=50, n_est=50, n_match=30), shapes)
    assert report.share("e2e") == pytest.approx(2 * (80 + 30) / ((100 + 98) + (50 + 50)))
    assert report.chord_shape_rate("e2e") == pytest.approx(18 / 20)
    assert report.song_counts("e2e") == {"t0": (160, 198), "t1": (60, 100)}


def test_per_track_counts_keep_each_tracks_chord_shapes() -> None:
    # A paired interval on the chord-shape rate needs each track's counts, not the pool
    # (ADR 0039's rule, Ege's decision of 2026-10-04: no clear drop).
    from tabsampler.eval.metrics import PRF
    from tabsampler.eval.playability import PlayabilityReport
    from tabsampler.eval.recovery import add_track

    report = RecoveryReport()
    prf = PRF(1.0, 1.0, 1.0, n_ref=10, n_est=10, n_match=10)
    add_track(report, "t0", "oracle", prf, PlayabilityReport(10, 9, 9, 9))
    add_track(report, "t1", "oracle", prf, PlayabilityReport(5, 5, 4, 4))
    assert report.song_shape_counts("oracle") == {"t0": (9, 10), "t1": (5, 5)}
    assert RecoveryReport.from_dict(report.to_dict()) == report


def test_a_report_written_before_per_track_shapes_still_loads() -> None:
    old = {"per_song": {"t0": {"oracle": [8, 10]}}, "shapes": {"oracle": [9, 10]}}
    report = RecoveryReport.from_dict({**old, "single_candidate": 0})
    assert report.song_shape_counts("oracle") == {}

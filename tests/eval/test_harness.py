"""Tests for the evaluation harness.

The harness is pure: loaders and the transcriber are injected, so these tests need
neither the dataset nor a subprocess. I/O -- reading config, writing results.csv,
logging test-set access -- belongs to the CLI, so that eval/ stays free of I/O and
global state.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from tabsampler.decode.robust import Degradation, decode_best_effort
from tabsampler.errors import TranscriberFailedError
from tabsampler.eval.harness import FullReport, evaluate_full, evaluate_notes
from tabsampler.eval.playability import PlayabilityRules
from tabsampler.fingering.candidates import group_notes
from tabsampler.fingering.costs import HandSetScorer
from tabsampler.types import Context, NoteEvent, Position, TabNote, Tuning

STANDARD = Tuning.STANDARD


def ref_arrays(
    midi: list[float], start: float = 0.0
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    intervals = np.array([[start + i * 1.0, start + i * 1.0 + 0.5] for i in range(len(midi))])
    hz = 440.0 * 2 ** ((np.array(midi, dtype=float) - 69.0) / 12.0)
    return intervals, hz


def est_notes(midi: list[int], start: float = 0.0) -> list[NoteEvent]:
    return [
        NoteEvent(onset=start + i * 1.0, offset=start + i * 1.0 + 0.5, pitch=p, confidence=0.9)
        for i, p in enumerate(midi)
    ]


def test_a_perfect_run_scores_one() -> None:
    report = evaluate_notes(
        track_ids=["t1"],
        reference=lambda _: ref_arrays([40, 45, 50]),
        estimate=lambda _: est_notes([40, 45, 50]),
        audio_duration=lambda _: 30.0,
    )
    assert report.e1_onset.f1 == 1.0
    assert len(report.tracks) == 1


def test_corpus_score_is_micro_averaged_over_notes_not_a_mean_of_track_f1() -> None:
    # t_long: 10 reference notes, all found            -> per-track f1 = 1.0
    # t_short: 1 reference note, missed entirely       -> per-track f1 = 0.0
    # Mean of per-track f1  = 0.5   <- wrong; a 1-note excerpt outweighs a 10-note one
    # Micro-average         = precision 10/10, recall 10/11, f1 = 2*1*(10/11)/(1+10/11)
    #                       = 20/21 ~= 0.952
    refs = {"t_long": ref_arrays([40] * 10), "t_short": ref_arrays([60])}
    ests = {"t_long": est_notes([40] * 10), "t_short": []}
    report = evaluate_notes(
        track_ids=["t_long", "t_short"],
        reference=lambda t: refs[t],
        estimate=lambda t: ests[t],
        audio_duration=lambda _: 30.0,
    )
    assert report.e1_onset.n_ref == 11
    assert report.e1_onset.n_match == 10
    assert report.e1_onset.f1 == pytest.approx(20 / 21)
    assert report.e1_onset.f1 != pytest.approx(0.5)


def test_per_track_results_are_kept_alongside_the_corpus_total() -> None:
    refs = {"a": ref_arrays([40]), "b": ref_arrays([45])}
    ests = {"a": est_notes([40]), "b": []}
    report = evaluate_notes(
        track_ids=["a", "b"],
        reference=lambda t: refs[t],
        estimate=lambda t: ests[t],
        audio_duration=lambda _: 10.0,
    )
    by_id = {t.track_id: t for t in report.tracks}
    assert by_id["a"].e1_onset.f1 == 1.0
    assert by_id["b"].e1_onset.f1 == 0.0


def test_both_onset_only_and_onset_offset_variants_are_reported() -> None:
    # Same onsets and pitches, badly wrong durations: onset-only forgives, the
    # offset variant does not. Spec 3.1 asks for both.
    intervals = np.array([[0.0, 3.0]])
    hz = np.array([440.0])
    est = [NoteEvent(onset=0.0, offset=0.1, pitch=69, confidence=1.0)]
    report = evaluate_notes(
        track_ids=["t"],
        reference=lambda _: (intervals, hz),
        estimate=lambda _: est,
        audio_duration=lambda _: 5.0,
    )
    assert report.e1_onset.f1 == 1.0
    assert report.e1_onset_offset.f1 == 0.0


def test_a_track_with_no_reference_notes_is_reported_not_skipped() -> None:
    empty = (np.zeros((0, 2)), np.zeros(0))
    report = evaluate_notes(
        track_ids=["silent"],
        reference=lambda _: empty,
        estimate=lambda _: [],
        audio_duration=lambda _: 30.0,
    )
    assert len(report.tracks) == 1
    assert report.tracks[0].n_reference_notes == 0
    # Both sides empty is the perfect-empty-match convention from metrics.PRF.
    assert report.tracks[0].e1_onset.f1 == 1.0


def test_transcriber_failure_propagates_rather_than_scoring_zero() -> None:
    # The single most important behaviour in the harness. Swallowing this would turn
    # a broken transcriber into a plausible low F1.
    def boom(_: str) -> list[NoteEvent]:
        raise TranscriberFailedError("basic-pitch exit 1")

    with pytest.raises(TranscriberFailedError):
        evaluate_notes(
            track_ids=["t"],
            reference=lambda _: ref_arrays([40]),
            estimate=boom,
            audio_duration=lambda _: 30.0,
        )


def test_runtime_is_reported_as_seconds_per_minute_of_audio() -> None:
    # E7 per spec 3.1. 120 s of audio processed in a measurable time must give a
    # positive, finite rate.
    report = evaluate_notes(
        track_ids=["a", "b"],
        reference=lambda _: ref_arrays([40]),
        estimate=lambda _: est_notes([40]),
        audio_duration=lambda _: 60.0,
    )
    assert report.total_audio_seconds == pytest.approx(120.0)
    assert report.runtime_seconds_per_audio_minute >= 0.0
    assert np.isfinite(report.runtime_seconds_per_audio_minute)


def test_zero_length_audio_does_not_divide_by_zero() -> None:
    report = evaluate_notes(
        track_ids=["t"],
        reference=lambda _: (np.zeros((0, 2)), np.zeros(0)),
        estimate=lambda _: [],
        audio_duration=lambda _: 0.0,
    )
    assert report.runtime_seconds_per_audio_minute == 0.0


def test_an_empty_track_list_is_refused() -> None:
    # Evaluating nothing would print a vacuous 1.0 and look like success.
    with pytest.raises(ValueError, match="no tracks"):
        evaluate_notes(
            track_ids=[],
            reference=lambda _: ref_arrays([40]),
            estimate=lambda _: est_notes([40]),
            audio_duration=lambda _: 30.0,
        )


def test_tracks_are_evaluated_in_the_given_order() -> None:
    seen: list[str] = []

    def record(track_id: str) -> list[NoteEvent]:
        seen.append(track_id)
        return est_notes([40])

    evaluate_notes(
        track_ids=["c", "a", "b"],
        reference=lambda _: ref_arrays([40]),
        estimate=record,
        audio_duration=lambda _: 1.0,
    )
    assert seen == ["c", "a", "b"]


# ------------------------------------------------- evaluate_full: E1 at two points


def n(onset: float, pitch: int) -> NoteEvent:
    return NoteEvent(onset=onset, offset=onset + 0.5, pitch=pitch, confidence=0.9)


def full_report(
    notes_in: Callable[[str], list[NoteEvent]],
    ref_midi: list[float],
    mode: str = "e2e",
) -> FullReport:
    """Run the real fingering pipeline over one synthetic track."""
    ctx = Context(tuning=STANDARD, max_span=5)
    scorer = HandSetScorer()

    def place(notes: list[NoteEvent]) -> tuple[list[TabNote], Degradation]:
        groups = group_notes(notes)
        if not groups:
            return [], Degradation()
        return decode_best_effort(groups, scorer, ctx)

    def reference_tab(_: str) -> list[tuple[NoteEvent, Position]]:
        # MIDI 40 and 45 are the open low E and open A.
        return [(n(0.0, 40), Position(0, 0)), (n(1.0, 45), Position(1, 0))]

    return evaluate_full(
        track_ids=["t"],
        reference=lambda _: ref_arrays(ref_midi),
        reference_tab=reference_tab,
        notes_in=notes_in,
        place=place,
        audio_duration=lambda _: 30.0,
        tuning=STANDARD,
        rules=PlayabilityRules(),
        mode=mode,
    )


def test_e1_is_reported_before_and_after_the_fingering_stage() -> None:
    # Two reference notes. The estimator returns three, one of which is unplayable
    # (MIDI 39, below the open low E) and is dropped during placement. Reporting one
    # number for both hides that the drop raises precision.
    report = full_report(lambda _: [*est_notes([40, 45]), n(2.0, 39)], [40, 45])
    assert report.e1_incoming.n_est == 3  # what the transcriber said
    assert report.e1_onset.n_est == 2  # what survived placement
    assert report.e1_onset.f1 >= report.e1_incoming.f1


def test_the_two_e1_rows_agree_when_nothing_is_dropped() -> None:
    # In oracle mode every note is playable by construction, so the transcriber's score
    # and the pipeline's are the same number -- which is itself a check on the plumbing.
    report = full_report(lambda _: est_notes([40, 45]), [40, 45], mode="oracle")
    assert report.e1_incoming == report.e1_onset
    assert report.e1_onset.f1 == 1.0


def test_dropping_an_unplayable_note_raises_precision_and_leaves_recall_alone() -> None:
    # The exact reason the two numbers differ, stated as the property rather than as
    # a pair of magic values.
    report = full_report(lambda _: [*est_notes([40, 45]), n(2.0, 39)], [40, 45])
    assert report.e1_onset.precision > report.e1_incoming.precision
    assert report.e1_onset.recall == report.e1_incoming.recall

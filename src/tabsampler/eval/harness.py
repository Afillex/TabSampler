"""The evaluation runner (spec 3.2).

Pure by construction: the reference loader, the estimator and the duration lookup are
injected, so this module does no I/O and holds no state. Reading config, writing
``experiments/results.csv`` and logging test-set access all live in the CLI: ``eval/``
stays free of I/O so that a metric can never depend on where a file happens to be.

Corpus scores are **micro-averaged over notes**, not averaged over per-track F1 values.
GuitarSet excerpts are all about 30 s, but the note counts still vary several-fold, and
a mean of per-track F1 lets a sparse excerpt count for as much as a dense one.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tabsampler.decode.robust import Degradation
from tabsampler.eval.calibration import expected_calibration_error
from tabsampler.eval.metrics import (
    DEFAULT_ONSET_TOLERANCE,
    PRF,
    exact_tab_f1_with_matches,
    note_f1_arrays,
    tab_notes_to_placed,
)
from tabsampler.eval.playability import PlayabilityReport, PlayabilityRules, playability_rate
from tabsampler.types import NoteEvent, Position, TabNote, Tuning

#: Reference notes for a track, as ``(intervals_seconds, pitches_hz)``. Hz because
#: mir_eval derives its cent tolerance from a frequency ratio (ADR 0006).
ReferenceLoader = Callable[[str], tuple[NDArray[np.float64], NDArray[np.float64]]]

#: Predicted notes for a track. May raise: a transcriber failure must propagate.
Estimator = Callable[[str], list[NoteEvent]]

#: Audio duration in seconds, for the E7 runtime metric.
DurationLoader = Callable[[str], float]

#: Called after each track. Lets a caller report progress without this module doing
#: I/O itself: the harness stays pure, the CLI decides how to display it.
ProgressCallback = Callable[["TrackResult", int, int], None]


@dataclass(frozen=True, slots=True)
class TrackResult:
    """What one track contributed."""

    track_id: str
    e1_onset: PRF
    e1_onset_offset: PRF
    n_reference_notes: int
    n_estimated_notes: int
    audio_seconds: float
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class EvalReport:
    """Corpus-level result, with the per-track detail kept."""

    mode: str
    tracks: tuple[TrackResult, ...]
    e1_onset: PRF
    e1_onset_offset: PRF
    total_audio_seconds: float
    total_elapsed_seconds: float

    @property
    def runtime_seconds_per_audio_minute(self) -> float:
        """E7: seconds of processing per minute of audio. 0.0 for zero-length input."""
        if self.total_audio_seconds <= 0.0:
            return 0.0
        return self.total_elapsed_seconds / (self.total_audio_seconds / 60.0)


def _micro_average(parts: Sequence[PRF]) -> PRF:
    """Combine per-track PRFs by summing their counts, not their F1 values."""
    return PRF.from_counts(
        n_ref=sum(p.n_ref for p in parts),
        n_est=sum(p.n_est for p in parts),
        n_match=sum(p.n_match for p in parts),
    )


def evaluate_notes(
    track_ids: Sequence[str],
    reference: ReferenceLoader,
    estimate: Estimator,
    audio_duration: DurationLoader,
    onset_tolerance: float = DEFAULT_ONSET_TOLERANCE,
    mode: str = "e2e",
    on_track_done: ProgressCallback | None = None,
) -> EvalReport:
    """E1 over a corpus, in both the onset-only and onset+offset variants.

    Args:
        track_ids: Tracks to evaluate, in order.
        reference: Ground-truth notes per track, pitches in Hz.
        estimate: Predicted notes per track.
        audio_duration: Audio length in seconds per track, for E7.
        onset_tolerance: Seconds. 50 ms by default (MIREX, spec 3.1).
        mode: ``"oracle"`` or ``"e2e"``; recorded, not acted on.
        on_track_done: Optional progress hook, called with
            ``(result, index, total)`` after each track.

    Raises:
        ValueError: if ``track_ids`` is empty. Evaluating nothing would report a
            vacuous perfect score.
        Exception: anything ``estimate`` raises propagates unchanged. A transcriber
            failure must not be scored as zero -- see ADR 0001.
    """
    if not track_ids:
        raise ValueError("no tracks to evaluate; an empty corpus reports a vacuous score")

    results: list[TrackResult] = []
    for track_id in track_ids:
        ref_intervals, ref_hz = reference(track_id)

        started = time.perf_counter()
        estimated = estimate(track_id)
        elapsed = time.perf_counter() - started

        est_intervals, est_hz = _note_arrays(estimated)
        results.append(
            TrackResult(
                track_id=track_id,
                e1_onset=note_f1_arrays(
                    ref_intervals,
                    ref_hz,
                    est_intervals,
                    est_hz,
                    onset_tolerance=onset_tolerance,
                    match_offsets=False,
                ),
                e1_onset_offset=note_f1_arrays(
                    ref_intervals,
                    ref_hz,
                    est_intervals,
                    est_hz,
                    onset_tolerance=onset_tolerance,
                    match_offsets=True,
                ),
                n_reference_notes=int(ref_intervals.shape[0]),
                n_estimated_notes=len(estimated),
                audio_seconds=audio_duration(track_id),
                elapsed_seconds=elapsed,
            )
        )
        if on_track_done is not None:
            on_track_done(results[-1], len(results), len(track_ids))

    return EvalReport(
        mode=mode,
        tracks=tuple(results),
        e1_onset=_micro_average([r.e1_onset for r in results]),
        e1_onset_offset=_micro_average([r.e1_onset_offset for r in results]),
        total_audio_seconds=sum(r.audio_seconds for r in results),
        total_elapsed_seconds=sum(r.elapsed_seconds for r in results),
    )


def _note_arrays(
    notes: Sequence[NoteEvent],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(intervals, pitches in Hz) for a list of NoteEvents."""
    if not notes:
        return np.zeros((0, 2), dtype=float), np.zeros(0, dtype=float)
    intervals = np.array([[n.onset, n.offset] for n in notes], dtype=float)
    midi = np.array([n.pitch for n in notes], dtype=float)
    # Same A440 reference as metrics; kept explicit rather than importing librosa here.
    hz = 440.0 * np.power(2.0, (midi - 69.0) / 12.0)
    return intervals, hz


# =====================================================================================
# M1: the full evaluation, in both modes (spec 3.2)
# =====================================================================================


@dataclass(frozen=True, slots=True)
class FullTrackResult:
    """One track's contribution to the M1 table."""

    track_id: str
    #: E1 on the notes handed *to* the fingering stage: the transcriber's own score.
    e1_incoming: PRF
    #: E1 on the notes that survived placement: the pipeline's score. Placement drops
    #: notes the guitar cannot sound, which raises precision, so the two differ.
    e1_onset: PRF
    e2: PRF
    e3: PlayabilityReport
    n_pitch_valid: int
    n_placed: int
    posteriors: tuple[float, ...]
    correct: tuple[bool, ...]
    audio_seconds: float
    elapsed_seconds: float
    degradation: Degradation


@dataclass(frozen=True, slots=True)
class FullReport:
    """E1-E5 and E7 over a corpus, in one mode.

    ``mode`` is ``"oracle"`` (reference notes fed to the fingering stage, which measures
    fingering quality alone) or ``"e2e"`` (transcriber notes, which is what a user gets).
    Spec 3.2 requires reporting both: the gap between them is what the transcriber costs.

    **E1 is reported at two points, and they are different questions.** ``e1_incoming``
    scores the notes handed to the fingering stage; that is the *transcriber's* number and
    the one comparable with Phase 0. ``e1_onset`` scores the notes that came out; that is
    the *pipeline's*. They differ because ``decode_best_effort`` drops notes the guitar
    cannot sound, which raises precision without touching recall. M1 reported only the
    second and compared it with Phase 0's first (0.7452 against 0.7437), which was
    comparing two different measurements.
    """

    mode: str
    tracks: tuple[FullTrackResult, ...]
    #: E1 on the transcriber's raw output, before the fingering stage sees it.
    e1_incoming: PRF
    #: E1 on the notes that survived placement.
    e1_onset: PRF
    e2: PRF
    e3_group_rate: float
    e3_transition_rate: float
    e4: float
    e5_ece: float
    total_audio_seconds: float
    total_elapsed_seconds: float
    n_groups_relaxed: int
    n_notes_out_of_range: int
    n_notes_dropped: int
    n_groups_dropped: int

    @property
    def runtime_seconds_per_audio_minute(self) -> float:
        if self.total_audio_seconds <= 0.0:
            return 0.0
        return self.total_elapsed_seconds / (self.total_audio_seconds / 60.0)


#: Reference fingering for a track: the E2 ground truth.
TabLoader = Callable[[str], list[tuple[NoteEvent, Position]]]

#: Notes to feed the fingering stage. Reference notes in oracle mode, transcriber
#: notes end to end.
NoteLoader = Callable[[str], list[NoteEvent]]


def evaluate_full(
    track_ids: Sequence[str],
    reference: ReferenceLoader,
    reference_tab: TabLoader,
    notes_in: NoteLoader,
    place: Callable[[list[NoteEvent]], tuple[list[TabNote], Degradation]],
    audio_duration: DurationLoader,
    tuning: Tuning,
    rules: PlayabilityRules,
    onset_tolerance: float = DEFAULT_ONSET_TOLERANCE,
    mode: str = "e2e",
    group_window_s: float = 0.03,
    on_track_done: Callable[[FullTrackResult, int, int], None] | None = None,
) -> FullReport:
    """Run E1-E5 and E7 over a corpus.

    Everything is injected, so this stays pure. ``place`` is the whole fingering
    pipeline -- group, enumerate, score, decode -- as one callable, which keeps the
    harness ignorant of how the tab is produced.

    Raises:
        ValueError: if ``track_ids`` is empty.
    """
    if not track_ids:
        raise ValueError("no tracks to evaluate; an empty corpus reports a vacuous score")

    results: list[FullTrackResult] = []
    for track_id in track_ids:
        ref_intervals, ref_hz = reference(track_id)
        ref_tab = reference_tab(track_id)

        started = time.perf_counter()
        incoming = notes_in(track_id)
        tab, degradation = place(incoming)
        elapsed = time.perf_counter() - started

        in_intervals, in_hz = _note_arrays(incoming)
        est_intervals, est_hz = _note_arrays([t.note for t in tab])
        est_placed = tab_notes_to_placed(tab)
        e2, correct = exact_tab_f1_with_matches(ref_tab, est_placed, onset_tolerance)

        results.append(
            FullTrackResult(
                track_id=track_id,
                e1_incoming=note_f1_arrays(
                    ref_intervals,
                    ref_hz,
                    in_intervals,
                    in_hz,
                    onset_tolerance=onset_tolerance,
                    match_offsets=False,
                ),
                e1_onset=note_f1_arrays(
                    ref_intervals,
                    ref_hz,
                    est_intervals,
                    est_hz,
                    onset_tolerance=onset_tolerance,
                    match_offsets=False,
                ),
                e2=e2,
                e3=playability_rate(tab, rules, window_s=group_window_s),
                n_pitch_valid=sum(
                    1
                    for note, position in est_placed
                    if 0 <= position.string < tuning.n_strings
                    and 0 <= position.fret <= tuning.max_fret
                    and tuning.pitch_at(position.string, position.fret) == note.pitch
                ),
                n_placed=len(tab),
                posteriors=tuple(t.posterior for t in tab),
                correct=correct,
                audio_seconds=audio_duration(track_id),
                elapsed_seconds=elapsed,
                degradation=degradation,
            )
        )
        if on_track_done is not None:
            on_track_done(results[-1], len(results), len(track_ids))

    # E5 is pooled over the whole corpus: per-track ECE on 30-second excerpts would be
    # dominated by binning noise.
    all_posteriors = [p for r in results for p in r.posteriors]
    all_correct = [c for r in results for c in r.correct]
    total_placed = sum(r.n_placed for r in results)

    return FullReport(
        mode=mode,
        tracks=tuple(results),
        e1_incoming=_micro_average([r.e1_incoming for r in results]),
        e1_onset=_micro_average([r.e1_onset for r in results]),
        e2=_micro_average([r.e2 for r in results]),
        e3_group_rate=_pooled_rate([(r.e3.n_groups_pass, r.e3.n_groups) for r in results]),
        e3_transition_rate=_pooled_rate(
            [(r.e3.n_transitions_pass, r.e3.n_transitions) for r in results]
        ),
        e4=(sum(r.n_pitch_valid for r in results) / total_placed) if total_placed else 1.0,
        e5_ece=expected_calibration_error(all_posteriors, all_correct),
        total_audio_seconds=sum(r.audio_seconds for r in results),
        total_elapsed_seconds=sum(r.elapsed_seconds for r in results),
        n_groups_relaxed=sum(r.degradation.n_groups_relaxed for r in results),
        n_notes_out_of_range=sum(r.degradation.n_notes_out_of_range for r in results),
        n_notes_dropped=sum(r.degradation.n_notes_dropped for r in results),
        n_groups_dropped=sum(r.degradation.n_groups_dropped for r in results),
    )


def _pooled_rate(parts: Sequence[tuple[int, int]]) -> float:
    """Sum numerators and denominators rather than averaging rates."""
    passed = sum(p for p, _ in parts)
    total = sum(t for _, t in parts)
    return passed / total if total else 1.0

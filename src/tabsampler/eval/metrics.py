"""E1 note F1, E2 Exact Tab F1 and E4 pitch validity (spec 3.1).

GUARDED AREA -- this is metric code. Every number the project reports comes through
here. Pure: no I/O, no global state, no printing.

Two things in here are easy to get subtly and silently wrong.

**Pitch units.** ``mir_eval.transcription`` derives its pitch tolerance in cents from a
frequency *ratio* -- ``1200 * log2(ref / est)`` -- so it requires **Hz**. Handing it MIDI
numbers returns a plausible, wrong answer: a semitone error at MIDI 40 vs 41 computes as
1200*log2(41/40) = 42.7 cents, inside the 50-cent tolerance, and counts as a hit. In Hz
the same error is exactly 100 cents and correctly misses. Conversion happens once, here.

**Matching.** E2 pairs reference against estimated notes, and a note may be used only
once. The maximum number of pairs is a maximum bipartite matching, not a greedy sweep:
greedy undercounts, and its answer depends on input order, which would make the metric
non-deterministic with respect to how notes happen to be listed. ``mir_eval`` matches
optimally for E1, so a greedy E2 would also disagree with E1 about the same audio.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import librosa
import numpy as np
from mir_eval import transcription as mir_transcription
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import (
    maximum_bipartite_matching,  # pyright: ignore[reportUnknownVariableType]
)

from tabsampler.types import NoteEvent, Position, TabNote, Tuning

#: A note together with where it was played. The shape both E2 inputs take.
PlacedNote = tuple[NoteEvent, Position]

#: MIREX's note-tracking tolerance, and mir_eval's default (spec 3.1).
DEFAULT_ONSET_TOLERANCE = 0.05

#: A quarter tone, in cents. mir_eval's default and MIREX's criterion.
PITCH_TOLERANCE_CENTS = 50.0

#: mir_eval's onset+offset variant: within 20% of the reference duration, or 50 ms,
#: whichever is larger.
DEFAULT_OFFSET_RATIO = 0.2
DEFAULT_OFFSET_MIN_TOLERANCE = 0.05


@dataclass(frozen=True, slots=True)
class PRF:
    """Precision, recall and F-measure, with the counts they came from.

    The counts are kept because a corpus score must be micro-averaged over notes;
    averaging per-track F1 values would let a 3-note excerpt outweigh a 200-note one.
    """

    precision: float
    recall: float
    f1: float
    n_ref: int
    n_est: int
    n_match: int

    @classmethod
    def from_counts(cls, n_ref: int, n_est: int, n_match: int) -> PRF:
        """Build a PRF from counts, applying the empty-input conventions.

        Conventions, which are choices rather than derivations:

        - both sides empty: 1.0. Predicting nothing where there is nothing is correct,
          and silent clips are legitimate input.
        - exactly one side empty: 0.0, never NaN.
        """
        if n_ref == 0 and n_est == 0:
            return cls(1.0, 1.0, 1.0, 0, 0, 0)
        precision = n_match / n_est if n_est else 0.0
        recall = n_match / n_ref if n_ref else 0.0
        denominator = precision + recall
        f1 = (2 * precision * recall / denominator) if denominator else 0.0
        return cls(precision, recall, f1, n_ref, n_est, n_match)


def _to_arrays(
    notes: Sequence[NoteEvent],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(intervals in seconds, pitches in Hz) for mir_eval."""
    if not notes:
        return np.zeros((0, 2), dtype=float), np.zeros(0, dtype=float)
    intervals = np.array([[n.onset, n.offset] for n in notes], dtype=float)
    midi = np.array([n.pitch for n in notes], dtype=float)
    hz = np.asarray(librosa.midi_to_hz(midi), dtype=float)
    return intervals, hz


def note_f1(
    ref: Sequence[NoteEvent],
    est: Sequence[NoteEvent],
    onset_tolerance: float = DEFAULT_ONSET_TOLERANCE,
    match_offsets: bool = False,
) -> PRF:
    """E1: did we hear the right notes?

    A hit needs the onset within ``onset_tolerance`` and the pitch within a quarter
    tone. With ``match_offsets`` the offset must also land within 20% of the reference
    duration, or 50 ms, whichever is larger.

    The onset boundary is **inclusive**: mir_eval's default ``strict=False`` compares
    with ``<=``, having first rounded the distance to 4 decimal places, so a difference
    of exactly 50 ms is a hit.
    """
    return note_f1_arrays(
        *_to_arrays(ref),
        *_to_arrays(est),
        onset_tolerance=onset_tolerance,
        match_offsets=match_offsets,
    )


def note_f1_arrays(
    ref_intervals: NDArray[np.float64],
    ref_pitches_hz: NDArray[np.float64],
    est_intervals: NDArray[np.float64],
    est_pitches_hz: NDArray[np.float64],
    onset_tolerance: float = DEFAULT_ONSET_TOLERANCE,
    match_offsets: bool = False,
) -> PRF:
    """E1 from raw arrays, with **pitches in Hz**.

    The entry point for reference annotations whose pitch is fractional -- GuitarSet
    annotates continuous pitch, and rounding it to a semitone before scoring would
    throw away information mir_eval's 50-cent tolerance can use (ADR 0006).
    """
    n_ref = int(ref_intervals.shape[0])
    n_est = int(est_intervals.shape[0])
    if n_ref == 0 or n_est == 0:
        return PRF.from_counts(n_ref, n_est, 0)

    n_match = _count_mir_eval_matches(
        ref_intervals,
        ref_pitches_hz,
        est_intervals,
        est_pitches_hz,
        onset_tolerance=onset_tolerance,
        offset_ratio=DEFAULT_OFFSET_RATIO if match_offsets else None,
    )
    return PRF.from_counts(n_ref, n_est, n_match)


def exact_tab_f1(
    ref: Sequence[PlacedNote],
    est: Sequence[PlacedNote],
    onset_tolerance: float = DEFAULT_ONSET_TOLERANCE,
) -> PRF:
    """E2. See :func:`exact_tab_f1_with_matches` for the per-estimate detail."""
    return exact_tab_f1_with_matches(ref, est, onset_tolerance)[0]


def exact_tab_f1_with_matches(
    ref: Sequence[PlacedNote],
    est: Sequence[PlacedNote],
    onset_tolerance: float = DEFAULT_ONSET_TOLERANCE,
) -> tuple[PRF, tuple[bool, ...]]:
    """E2: did we recover the reference string and fret?

    A hit requires all three of: onset within ``onset_tolerance``, identical integer
    pitch, and identical string. Fret is implied -- given the pitch and the string, the
    fret is determined (and E4 checks that invariant separately).

    The hit count is a maximum bipartite matching over admissible pairs, so no note is
    used twice and the result does not depend on input order.

    Returns:
        The PRF, and a flag per *estimated* note saying whether it was matched. The
        flags are what E5 calibration scores posteriors against: "was this placement
        actually right".
    """
    n_ref, n_est = len(ref), len(est)
    if n_ref == 0 or n_est == 0:
        return PRF.from_counts(n_ref, n_est, 0), tuple([False] * n_est)

    rows: list[int] = []
    cols: list[int] = []
    for i, (ref_note, ref_pos) in enumerate(ref):
        for j, (est_note, est_pos) in enumerate(est):
            if (
                ref_note.pitch == est_note.pitch
                and ref_pos.string == est_pos.string
                # Rounded to 4 decimals before comparing, matching mir_eval's
                # N_DECIMALS, so E1 and E2 agree on the tolerance boundary.
                and round(abs(ref_note.onset - est_note.onset), 4) <= onset_tolerance
            ):
                rows.append(i)
                cols.append(j)

    if not rows:
        return PRF.from_counts(n_ref, n_est, 0), tuple([False] * n_est)

    matched_columns = _bipartite_matched_columns(rows, cols, n_ref, n_est)
    flags = tuple(index in matched_columns for index in range(n_est))
    return PRF.from_counts(n_ref, n_est, len(matched_columns)), flags


def pitch_validity_rate(tab: Sequence[PlacedNote], tuning: Tuning) -> float:
    """E4: does every (string, fret) actually produce the pitch we claimed?

    Should be exactly 1.0. Anything lower is a bug in candidate generation, the
    decoder or the renderer -- not a result to report.

    An out-of-range string or fret counts as invalid rather than raising, so that one
    bad note is a visible score rather than an aborted corpus run. An empty tab is
    vacuously 1.0.
    """
    if not tab:
        return 1.0

    valid = 0
    for note, pos in tab:
        if not 0 <= pos.string < tuning.n_strings:
            continue
        if not 0 <= pos.fret <= tuning.max_fret:
            continue
        if tuning.pitch_at(pos.string, pos.fret) == note.pitch:
            valid += 1
    return valid / len(tab)


def tab_notes_to_placed(tab: Sequence[TabNote]) -> list[PlacedNote]:
    """Adapt the decoder's output to the shape the tab metrics take."""
    return [(t.note, t.position) for t in tab]


# --------------------------------------------------------------------------------
# Typed boundaries around untyped libraries.
#
# mir_eval and scipy.sparse.csgraph ship no annotations, so calls into them are
# "partially unknown" under pyright strict. Confining each to one small function with
# an explicit return type keeps the suppressions in two places instead of seven, and
# leaves the rest of this module fully checked.
# --------------------------------------------------------------------------------


def _count_mir_eval_matches(
    ref_intervals: NDArray[np.float64],
    ref_pitches_hz: NDArray[np.float64],
    est_intervals: NDArray[np.float64],
    est_pitches_hz: NDArray[np.float64],
    onset_tolerance: float,
    offset_ratio: float | None,
) -> int:
    """How many reference notes mir_eval matches to an estimate.

    ``offset_ratio=None`` means "ignore offsets", which mir_eval documents as valid
    ("float > 0 or None") but does not express in its signature -- hence an argument
    suppression rather than a cast, which would hide a genuine mismatch later.
    """
    matches = mir_transcription.match_notes(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        ref_intervals,
        ref_pitches_hz,
        est_intervals,
        est_pitches_hz,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=PITCH_TOLERANCE_CENTS,
        offset_ratio=offset_ratio,  # pyright: ignore[reportArgumentType]
        offset_min_tolerance=DEFAULT_OFFSET_MIN_TOLERANCE,
    )
    return len(matches)  # pyright: ignore[reportUnknownArgumentType]


def _bipartite_matched_columns(
    rows: list[int], cols: list[int], n_ref: int, n_est: int
) -> set[int]:
    """Estimate indices used by a maximum matching over the admissible pairs."""
    graph = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n_ref, n_est))
    matching = maximum_bipartite_matching(  # pyright: ignore[reportUnknownVariableType]
        graph, perm_type="column"
    )
    matched: NDArray[np.int64] = np.asarray(matching, dtype=np.int64)  # pyright: ignore[reportUnknownArgumentType]
    return {int(column) for column in matched if int(column) >= 0}

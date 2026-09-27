"""Tests for GuitarSet reference extraction.

GUARDED AREA: this is data loading for evaluation. What these functions return *is*
the definition of "correct" for E1 and E2, so a bug here silently moves every number
in the project. Everything runs on a fake track; no download, no network.
"""

from __future__ import annotations

import numpy as np
import pytest

from tabsampler.data.guitarset import (
    GUITAR_STRINGS,
    reference_note_arrays,
    reference_notes,
    reference_tab,
)
from tabsampler.errors import ReferenceInconsistencyError
from tabsampler.types import Tuning

from ..fixtures.fake_track import fake_track


def test_string_index_zero_is_low_e() -> None:
    # The convention that ties mirdata's dict keys to Position.string.
    track = fake_track({"E": [(0.0, 0.5, 40.0)]})
    ((_, pos),) = reference_tab(track)
    assert pos.string == 0
    assert pos.fret == 0


def test_string_index_five_is_high_e() -> None:
    track = fake_track({"e": [(0.0, 0.5, 64.0)]})
    ((_, pos),) = reference_tab(track)
    assert pos.string == 5
    assert pos.fret == 0


def test_guitar_string_order_matches_mirdata() -> None:
    assert GUITAR_STRINGS == ("E", "A", "D", "G", "B", "e")


def test_float_midi_is_rounded_not_truncated() -> None:
    # ADR 0006. int() would give 45 here and bias every pitch downward.
    track = fake_track({"A": [(0.0, 0.4, 45.48)], "D": [(1.0, 1.4, 50.62)]})
    pitches = sorted(n.pitch for n, _ in reference_tab(track))
    assert pitches == [45, 51]


def test_reference_tab_fret_is_consistent_with_pitch() -> None:
    # E4 must be 1.0 on the reference itself, or our own ground truth is wrong.
    track = fake_track(
        {
            "E": [(0.0, 0.3, 43.0)],  # low E, 3rd fret
            "A": [(0.1, 0.4, 52.0)],  # A string, 7th fret
            "e": [(0.2, 0.5, 76.0)],  # high e, 12th fret
        }
    )
    for note, pos in reference_tab(track):
        assert Tuning.STANDARD.pitch_at(pos.string, pos.fret) == note.pitch


def test_a_note_below_its_open_string_is_reported_not_dropped() -> None:
    # MIDI 38 cannot be played on the low E string (open = 40). Dropping it silently
    # would inflate precision; the track id must appear so it can be investigated.
    track = fake_track({"E": [(0.0, 0.3, 38.0)]}, track_id="00_bad_comp")
    with pytest.raises(ReferenceInconsistencyError, match="00_bad_comp"):
        reference_tab(track)


def test_a_note_above_the_last_fret_is_reported_not_dropped() -> None:
    track = fake_track({"E": [(0.0, 0.3, 99.0)]})
    with pytest.raises(ReferenceInconsistencyError):
        reference_tab(track)


def test_reference_tab_is_sorted_by_onset_then_string() -> None:
    track = fake_track({"D": [(0.5, 0.9, 50.0)], "E": [(0.1, 0.4, 40.0)], "A": [(0.1, 0.4, 45.0)]})
    got = [(round(n.onset, 3), p.string) for n, p in reference_tab(track)]
    assert got == [(0.1, 0), (0.1, 1), (0.5, 2)]


def test_reference_notes_are_sorted_by_onset() -> None:
    track = fake_track({"E": [(0.9, 1.0, 40.0), (0.1, 0.2, 41.0)]})
    onsets = [n.onset for n in reference_notes(track)]
    assert onsets == sorted(onsets)


def test_reference_notes_have_full_confidence() -> None:
    # A reference annotation is ground truth; there is no uncertainty to express.
    track = fake_track({"E": [(0.0, 0.5, 40.0)]})
    assert all(n.confidence == 1.0 for n in reference_notes(track))


def test_empty_annotation_yields_empty_results() -> None:
    track = fake_track({})
    assert reference_notes(track) == []
    assert reference_tab(track) == []
    intervals, pitches = reference_note_arrays(track)
    assert intervals.shape == (0, 2)
    assert pitches.shape == (0,)


def test_a_string_with_no_notes_is_skipped_without_error() -> None:
    track = fake_track({"E": [(0.0, 0.5, 40.0)], "A": []})
    assert len(reference_tab(track)) == 1


# ------------------------------------------------------- the float path used by E1


def test_reference_note_arrays_are_in_hz_not_midi() -> None:
    # ADR 0006 and mir_eval's cents computation: mir_eval compares pitch *ratios*,
    # so it needs Hz. MIDI 69 is A440 by definition.
    track = fake_track({"A": [(0.0, 0.5, 69.0)]})
    _, pitches = reference_note_arrays(track)
    assert pitches[0] == pytest.approx(440.0)


def test_reference_note_arrays_preserve_the_fractional_midi_value() -> None:
    # The reference tab rounds (E2 needs integers); the E1 path must not, so that
    # mir_eval's 50-cent tolerance can absorb vibrato rather than us discarding it.
    track = fake_track({"A": [(0.0, 0.5, 69.5)]})
    _, pitches = reference_note_arrays(track)
    assert pitches[0] > 440.0 * 2 ** (0.4 / 12)
    assert pitches[0] < 440.0 * 2 ** (0.6 / 12)


def test_reference_note_arrays_intervals_are_seconds_shaped_n_by_2() -> None:
    track = fake_track({"E": [(0.25, 0.75, 40.0)], "A": [(1.0, 1.5, 45.0)]})
    intervals, pitches = reference_note_arrays(track)
    assert intervals.shape == (2, 2)
    assert pitches.shape == (2,)
    np.testing.assert_allclose(intervals[0], [0.25, 0.75])


def test_a_track_without_notes_all_falls_back_to_the_per_string_notes() -> None:
    track = fake_track({"E": [(0.0, 0.5, 40.0)], "A": [(0.1, 0.6, 45.0)]})
    track.notes_all = None
    assert len(reference_notes(track)) == 2

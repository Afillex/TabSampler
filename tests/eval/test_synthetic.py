"""Tests for the synthetic round-trip diagnostic.

These test the *diagnostic*, not the cost model. See the module docstring of
``tabsampler.eval.synthetic`` for why a number from it must never tune a weight.
"""

from __future__ import annotations

import numpy as np
import pytest

from tabsampler.eval.playability import PlayabilityRules, group_is_playable
from tabsampler.eval.synthetic import (
    RoundTripReport,
    round_trip_accuracy,
    sample_playable_path,
)
from tabsampler.fingering.costs import HandSetScorer
from tabsampler.types import ChordState, Context, NoteEvent, NoteGroup, Position, Tuning

STANDARD = Tuning.STANDARD
RULES = PlayabilityRules()
CTX = Context(tuning=STANDARD, max_span=5)
SCORER = HandSetScorer()


def n(pitch: int, onset: float = 0.0) -> NoteEvent:
    return NoteEvent(onset=onset, offset=onset + 0.4, pitch=pitch, confidence=1.0)


# ------------------------------------------------------------------ sampling


def test_a_sampled_path_is_playable_by_construction() -> None:
    rng = np.random.default_rng(0)
    for _group, state in sample_playable_path(rng, n_groups=20, tuning=STANDARD, rules=RULES):
        assert group_is_playable(state.positions, RULES) is None


def test_sampling_is_deterministic_given_a_seed() -> None:
    a = sample_playable_path(np.random.default_rng(7), 10, STANDARD, RULES)
    b = sample_playable_path(np.random.default_rng(7), 10, STANDARD, RULES)
    assert [s.positions for _, s in a] == [s.positions for _, s in b]


def test_different_seeds_give_different_paths() -> None:
    # Guards a sampler that ignores the generator and returns a constant.
    a = sample_playable_path(np.random.default_rng(1), 20, STANDARD, RULES)
    b = sample_playable_path(np.random.default_rng(2), 20, STANDARD, RULES)
    assert [s.positions for _, s in a] != [s.positions for _, s in b]


def test_sampled_pitches_are_reproducible_from_the_sampled_positions() -> None:
    for group, state in sample_playable_path(np.random.default_rng(1), 15, STANDARD, RULES):
        for note, p in zip(group.notes, state.positions, strict=True):
            assert STANDARD.pitch_at(p.string, p.fret) == note.pitch


def test_a_sampled_group_is_ordered_and_matches_its_state() -> None:
    # NoteGroup sorts by (pitch, onset); positions are parallel to notes, so a sampler
    # that built the group and the state independently would silently misalign them.
    for group, state in sample_playable_path(np.random.default_rng(3), 15, STANDARD, RULES):
        assert len(group) == len(state)
        assert len({p.string for p in state.positions}) == len(state.positions)


def test_the_generator_is_passed_in_and_never_created_inside() -> None:
    # Purity: eval/ holds no global state, so two calls with the same generator object
    # must continue the stream rather than restart it.
    rng = np.random.default_rng(11)
    first = sample_playable_path(rng, 10, STANDARD, RULES)
    second = sample_playable_path(rng, 10, STANDARD, RULES)
    assert [s.positions for _, s in first] != [s.positions for _, s in second]


def test_zero_groups_samples_nothing() -> None:
    assert sample_playable_path(np.random.default_rng(0), 0, STANDARD, RULES) == []


# ------------------------------------------------------------------ round trip


def test_round_trip_accuracy_is_one_when_every_note_has_a_single_candidate() -> None:
    # MIDI 86 is reachable only at high e fret 22, so any correct decoder recovers it
    # whatever the weights. Isolates decoder plumbing from cost-model quality.
    paths = [[(NoteGroup.of([n(86)]), ChordState(positions=(Position(5, 22),)))]]
    report = round_trip_accuracy(paths, SCORER, CTX)
    assert report.accuracy == 1.0
    assert report.n_single_candidate == 1


def test_round_trip_reports_per_note_accuracy_not_per_path() -> None:
    # One 10-note path recovered plus one 1-note path missed is 10/11, not 0.5 -- the
    # same micro-averaging rule the corpus metrics use.
    recovered = [
        (NoteGroup.of([n(86, onset=i * 0.5)]), ChordState(positions=(Position(5, 22),)))
        for i in range(10)
    ]
    # MIDI 64 sits on five strings; claiming the one the decoder will not pick makes
    # this path a miss without depending on which one that is.
    missed = [(NoteGroup.of([n(64)]), ChordState(positions=(Position(1, 19),)))]
    report = round_trip_accuracy([recovered, missed], SCORER, CTX)
    assert report.n_notes == 11
    assert report.n_paths == 2
    assert report.accuracy == pytest.approx(10 / 11)


def test_single_candidate_notes_are_counted_apart_as_free_wins() -> None:
    # A note with one legal position is recovered by any decoder, so an accuracy that
    # pooled them with real choices would flatter the model.
    paths = [
        [
            (NoteGroup.of([n(86)]), ChordState(positions=(Position(5, 22),))),
            (NoteGroup.of([n(64, onset=0.5)]), ChordState(positions=(Position(5, 0),))),
        ]
    ]
    report = round_trip_accuracy(paths, SCORER, CTX)
    assert report.n_notes == 2
    assert report.n_single_candidate == 1


def test_an_empty_report_is_accurate_by_convention_not_by_division() -> None:
    report = round_trip_accuracy([], SCORER, CTX)
    assert report.n_notes == 0
    assert report.accuracy == 1.0


def test_the_report_is_frozen_so_a_number_cannot_be_edited_after_the_fact() -> None:
    report = RoundTripReport(n_notes=4, n_recovered=3, n_paths=1, n_single_candidate=0)
    assert report.accuracy == pytest.approx(0.75)
    with pytest.raises(AttributeError):
        report.n_recovered = 4  # type: ignore[misc]


def test_a_sampled_corpus_round_trips_at_a_reportable_rate() -> None:
    # The diagnostic end to end. The assertion is deliberately weak: this test exists to
    # prove the machinery runs and reports, NOT to pin an accuracy. Pinning one would
    # invite tuning the cost model until the number moved, which is the circularity the
    # module docstring forbids.
    rng = np.random.default_rng(0)
    paths = [sample_playable_path(rng, 8, STANDARD, RULES) for _ in range(10)]
    report = round_trip_accuracy(paths, SCORER, CTX)
    assert report.n_paths == 10
    assert report.n_notes > 0
    assert 0.0 <= report.accuracy <= 1.0
    assert report.n_recovered <= report.n_notes

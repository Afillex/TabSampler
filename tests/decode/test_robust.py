"""Tests for best-effort decoding.

The case that motivated this module is real: GuitarSet's first track contains the chord
[51, 55, 58, 62, 67, 70], whose only string assignment spans 5 frets, so max_span=4
made it unfingerable and one chord killed the whole file.
"""

from __future__ import annotations

from tabsampler.decode.robust import Degradation, decode_best_effort, prepare_groups
from tabsampler.fingering.costs import HandSetScorer
from tabsampler.types import Context, NoteEvent, NoteGroup, Tuning

STANDARD = Tuning.STANDARD
SCORER = HandSetScorer()
CTX4 = Context(tuning=STANDARD, max_span=4)

#: The real chord from GuitarSet 00_BN1-129-Eb_comp.
REAL_CHORD = (51, 55, 58, 62, 67, 70)


def group(*pitches: int, onset: float = 0.0, confidence: float = 0.9) -> NoteGroup:
    return NoteGroup.of(
        [
            NoteEvent(onset=onset, offset=onset + 0.4, pitch=p, confidence=confidence)
            for p in pitches
        ]
    )


def test_the_real_chord_is_unfingerable_at_span_four() -> None:
    from tabsampler.fingering.states import enumerate_states

    assert enumerate_states(group(*REAL_CHORD), STANDARD, 4) == ()
    assert enumerate_states(group(*REAL_CHORD), STANDARD, 5) != ()


def test_the_span_is_relaxed_just_enough_and_the_relaxation_is_reported() -> None:
    _, spans, degradation = prepare_groups([group(*REAL_CHORD)], CTX4)
    assert spans == [5]  # relaxed to exactly what the shape needs, not to the maximum
    assert degradation.n_groups_relaxed == 1
    assert degradation.n_notes_dropped == 0
    assert degradation.max_span_used == 5
    assert "span relaxed 4 -> 5" in degradation.details[0]


def test_a_fingerable_group_is_left_alone() -> None:
    prepared, spans, degradation = prepare_groups([group(45, 52)], CTX4)
    assert spans == [4]
    assert degradation.is_clean
    assert len(prepared[0]) == 2


def test_too_many_notes_drops_the_least_confident_one() -> None:
    # Seven simultaneous notes on six strings cannot be fingered at any span.
    notes = [
        NoteEvent(onset=0.0, offset=0.4, pitch=p, confidence=c)
        for p, c in zip(
            (52, 54, 56, 57, 59, 61, 63), (0.9, 0.9, 0.9, 0.1, 0.9, 0.9, 0.9), strict=True
        )
    ]
    prepared, _, degradation = prepare_groups([NoteGroup.of(notes)], CTX4)
    assert degradation.n_notes_dropped >= 1
    assert 57 not in [n.pitch for n in prepared[0].notes]  # the 0.1-confidence note
    assert "dropped pitch 57" in degradation.details[0]


def test_dropping_is_reported_so_the_recall_cost_is_visible() -> None:
    notes = [
        NoteEvent(onset=0.0, offset=0.4, pitch=p, confidence=0.5)
        for p in (52, 54, 56, 57, 59, 61, 63)
    ]
    _, _, degradation = prepare_groups([NoteGroup.of(notes)], CTX4)
    assert not degradation.is_clean
    assert degradation.n_notes_dropped >= 1
    assert degradation.details != ()


def test_a_note_outside_the_instrument_is_dropped_not_raised() -> None:
    # Real case: basic-pitch predicted MIDI 39 on GuitarSet's first track, a semitone
    # below the open low E. No string and fret can sound it, so it is a transcriber
    # error about an unplayable note -- drop it and say so.
    prepared, spans, degradation = prepare_groups([group(39)], CTX4)
    assert prepared == []
    assert spans == []
    assert degradation.n_notes_out_of_range == 1
    assert degradation.n_groups_dropped == 1
    assert "outside the instrument" in degradation.details[0]


def test_an_out_of_range_note_beside_playable_ones_loses_only_itself() -> None:
    prepared, _, degradation = prepare_groups([group(39, 45, 52)], CTX4)
    assert [n.pitch for n in prepared[0].notes] == [45, 52]
    assert degradation.n_notes_out_of_range == 1
    assert degradation.n_groups_dropped == 0


def test_out_of_range_and_fingering_drops_are_counted_separately() -> None:
    # They have different causes and different meanings for the metrics.
    _, _, degradation = prepare_groups([group(39)], CTX4)
    assert degradation.n_notes_out_of_range == 1
    assert degradation.n_notes_dropped == 0
    assert degradation.n_notes_lost == 1


def test_decode_best_effort_places_the_real_chord() -> None:
    tab, degradation = decode_best_effort([group(*REAL_CHORD)], SCORER, CTX4)
    assert len(tab) == 6
    assert degradation.n_groups_relaxed == 1
    for t in tab:
        assert STANDARD.pitch_at(t.position.string, t.position.fret) == t.note.pitch


def test_decode_best_effort_matches_strict_decode_on_clean_input() -> None:
    from tabsampler.decode.forward_backward import decode

    groups = [group(45, 52), group(60, onset=0.5), group(64, onset=1.0)]
    strict = decode(groups, SCORER, CTX4)
    best, degradation = decode_best_effort(groups, SCORER, CTX4)
    assert degradation.is_clean
    assert [t.position for t in best] == [t.position for t in strict]


def test_no_groups_gives_no_tab_and_a_clean_report() -> None:
    tab, degradation = decode_best_effort([], SCORER, CTX4)
    assert tab == []
    assert degradation == Degradation()
    assert degradation.is_clean


def test_only_the_offending_group_is_relaxed() -> None:
    groups = [group(45, 52), group(*REAL_CHORD, onset=0.5), group(60, onset=1.0)]
    _, spans, degradation = prepare_groups(groups, CTX4)
    assert spans == [4, 5, 4]
    assert degradation.n_groups_relaxed == 1

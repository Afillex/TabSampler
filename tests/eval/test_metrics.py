"""Tests for E1 (note F1), E2 (Exact Tab F1) and E4 (pitch validity).

GUARDED AREA: metric code. Every number this project reports comes through here, so
each expected value below is hand-computed and the arithmetic is written out.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tabsampler.eval.metrics import (
    exact_tab_f1,
    note_f1,
    note_f1_arrays,
    pitch_validity_rate,
    tab_notes_to_placed,
)
from tabsampler.types import NoteEvent, Position, TabNote, Tuning


def n(onset: float, pitch: int, dur: float = 0.4) -> NoteEvent:
    return NoteEvent(onset=onset, offset=onset + dur, pitch=pitch, confidence=1.0)


def placed(onset: float, pitch: int, string: int, fret: int) -> tuple[NoteEvent, Position]:
    return (n(onset, pitch), Position(string=string, fret=fret))


# =========================================================== E1: note F1


def test_perfect_match_is_f1_one() -> None:
    notes = [n(0.0, 40), n(0.5, 45), n(1.0, 52)]
    r = note_f1(notes, notes)
    assert r.f1 == 1.0
    assert (r.n_ref, r.n_est, r.n_match) == (3, 3, 3)


def test_pitch_is_converted_to_hz_not_passed_as_midi() -> None:
    # The real trap. mir_eval computes cents as 1200*log2(ref/est), so it needs Hz.
    #   Correct (Hz):  1200*log2(87.31/82.41) = 100.0 cents -> outside 50 -> no match
    #   Wrong  (MIDI): 1200*log2(41/40)      =  42.7 cents -> inside  50 -> false match
    # So a semitone error distinguishes the two implementations exactly.
    assert note_f1([n(0.0, 40)], [n(0.0, 41)]).f1 == 0.0


def test_a_quarter_tone_error_still_matches() -> None:
    # 50 cents is the documented tolerance, so a half-semitone deviation is a hit.
    # Expressed through the float-Hz entry point, since NoteEvent.pitch is an int.
    ref_iv = np.array([[0.0, 0.4]])
    est_iv = np.array([[0.0, 0.4]])
    hz_ref = np.array([440.0])
    hz_est = np.array([440.0 * 2 ** (0.4 / 12)])  # 40 cents sharp
    assert note_f1_arrays(ref_iv, hz_ref, est_iv, hz_est).f1 == 1.0


def test_onset_exactly_at_the_tolerance_matches() -> None:
    # Pinned by experiment against mir_eval 0.8.2, not assumed: the default
    # strict=False compares with <=, and distances are rounded to N_DECIMALS=4
    # (0.1 ms) first. So a 50.00 ms offset is a hit.
    assert note_f1([n(0.0, 40)], [n(0.05, 40)]).f1 == 1.0


def test_onset_beyond_the_tolerance_does_not_match() -> None:
    # 50.1 ms does not round down to 50.0 ms, so it misses.
    assert note_f1([n(0.0, 40)], [n(0.0501, 40)]).f1 == 0.0


def test_offset_matching_is_stricter_than_onset_only() -> None:
    # Same onset and pitch, wildly wrong duration: onset-only forgives it, the
    # onset+offset variant does not.
    ref = [NoteEvent(onset=0.0, offset=2.0, pitch=40, confidence=1.0)]
    est = [NoteEvent(onset=0.0, offset=0.1, pitch=40, confidence=1.0)]
    assert note_f1(ref, est, match_offsets=False).f1 == 1.0
    assert note_f1(ref, est, match_offsets=True).f1 == 0.0


def test_partial_match_precision_and_recall() -> None:
    # 2 of 3 estimated notes are correct; 2 of 2 reference notes are found.
    #   precision = 2/3, recall = 2/2 = 1, f1 = 2*(2/3*1)/(2/3+1) = 0.8
    ref = [n(0.0, 40), n(1.0, 45)]
    est = [n(0.0, 40), n(1.0, 45), n(2.0, 52)]
    r = note_f1(ref, est)
    assert r.precision == pytest.approx(2 / 3)
    assert r.recall == pytest.approx(1.0)
    assert r.f1 == pytest.approx(0.8)


def test_empty_ref_and_empty_est_is_one_by_convention() -> None:
    # Documented convention, not a derivation: predicting nothing where there is
    # nothing is a perfect result. Matters because silent clips exist.
    r = note_f1([], [])
    assert r.f1 == 1.0
    assert (r.n_ref, r.n_est, r.n_match) == (0, 0, 0)


def test_empty_est_against_nonempty_ref_is_zero_not_nan() -> None:
    r = note_f1([n(0.0, 40)], [])
    assert r.f1 == 0.0
    assert r.precision == 0.0


def test_empty_ref_against_nonempty_est_is_zero_not_nan() -> None:
    assert note_f1([], [n(0.0, 40)]).f1 == 0.0


# =========================================================== E2: Exact Tab F1


def test_e2_perfect_match() -> None:
    tab = [placed(0.0, 40, 0, 0), placed(0.5, 45, 1, 0)]
    assert exact_tab_f1(tab, tab).f1 == 1.0


def test_e2_requires_string_to_match_not_only_pitch() -> None:
    # Same onset, same pitch, different string. This is the entire point of E2:
    # MIDI 45 is both the open A string and the low E at fret 5.
    ref = [placed(0.0, 45, 1, 0)]
    est = [placed(0.0, 45, 0, 5)]
    assert exact_tab_f1(ref, est).f1 == 0.0
    assert note_f1([p[0] for p in ref], [p[0] for p in est]).f1 == 1.0  # E1 forgives it


def test_e2_requires_pitch_to_match_not_only_string() -> None:
    ref = [placed(0.0, 40, 0, 0)]
    est = [placed(0.0, 41, 0, 1)]
    assert exact_tab_f1(ref, est).f1 == 0.0


def test_e2_uses_optimal_not_greedy_matching() -> None:
    # Constructed so a greedy sweep in input order undercounts.
    #   e0 at 0.100, e1 at 0.149 (both pitch 40, string 0)
    #   r0 at 0.100 -> |0.000| and |0.049| both within 0.05  => admissible with e0, e1
    #   r1 at 0.055 -> |0.045| within, |0.094| outside       => admissible with e0 only
    # Greedy takes r0-e0 first and leaves r1 unmatched: 1 match.
    # Optimal pairs r0-e1 and r1-e0:                        2 matches.
    ref = [placed(0.100, 40, 0, 0), placed(0.055, 40, 0, 0)]
    est = [placed(0.100, 40, 0, 0), placed(0.149, 40, 0, 0)]
    assert exact_tab_f1(ref, est).n_match == 2


def test_e2_is_independent_of_input_order() -> None:
    # Optimal matching is order-free; a greedy one is not. Guards determinism.
    ref = [placed(0.100, 40, 0, 0), placed(0.055, 40, 0, 0)]
    est = [placed(0.100, 40, 0, 0), placed(0.149, 40, 0, 0)]
    a = exact_tab_f1(ref, est).n_match
    b = exact_tab_f1(list(reversed(ref)), list(reversed(est))).n_match
    assert a == b == 2


def test_e2_does_not_match_one_reference_note_twice() -> None:
    # Two identical estimates against one reference: exactly one may be a hit,
    # so precision is 1/2, not 1.
    ref = [placed(0.0, 40, 0, 0)]
    est = [placed(0.0, 40, 0, 0), placed(0.01, 40, 0, 0)]
    r = exact_tab_f1(ref, est)
    assert r.n_match == 1
    assert r.precision == pytest.approx(0.5)
    assert r.recall == pytest.approx(1.0)


def test_e2_onset_boundary_is_inclusive_like_e1() -> None:
    assert exact_tab_f1([placed(0.0, 40, 0, 0)], [placed(0.05, 40, 0, 0)]).n_match == 1
    assert exact_tab_f1([placed(0.0, 40, 0, 0)], [placed(0.0501, 40, 0, 0)]).n_match == 0


def test_e2_empty_conventions_match_e1() -> None:
    assert exact_tab_f1([], []).f1 == 1.0
    assert exact_tab_f1([placed(0.0, 40, 0, 0)], []).f1 == 0.0
    assert exact_tab_f1([], [placed(0.0, 40, 0, 0)]).f1 == 0.0


def test_tab_notes_convert_to_placed_pairs() -> None:
    tab = [
        TabNote(note=n(0.0, 40), position=Position(0, 0), posterior=0.9),
        TabNote(note=n(0.5, 45), position=Position(1, 0), posterior=0.8),
    ]
    pairs = tab_notes_to_placed(tab)
    assert [p.string for _, p in pairs] == [0, 1]
    assert exact_tab_f1(pairs, pairs).f1 == 1.0


# =========================================================== E4: pitch validity


def test_e4_is_one_for_a_consistent_tab() -> None:
    tab = [placed(0.0, 40, 0, 0), placed(0.1, 52, 1, 7), placed(0.2, 76, 5, 12)]
    assert pitch_validity_rate(tab, Tuning.STANDARD) == 1.0


def test_e4_detects_a_single_bad_fret() -> None:
    # Three notes, one claiming fret 3 on the low E for a pitch that needs fret 5.
    tab = [placed(0.0, 40, 0, 0), placed(0.1, 45, 0, 3), placed(0.2, 64, 5, 0)]
    assert pitch_validity_rate(tab, Tuning.STANDARD) == pytest.approx(2 / 3)


def test_e4_is_one_with_a_capo() -> None:
    # Frets are relative to the capo: capo 2, fret 3 on the low E sounds 40+2+3 = 45.
    capo2 = replace(Tuning.STANDARD, capo=2)
    assert pitch_validity_rate([placed(0.0, 45, 0, 3)], capo2) == 1.0
    assert pitch_validity_rate([placed(0.0, 43, 0, 3)], capo2) == 0.0


def test_e4_counts_an_out_of_range_string_as_invalid_rather_than_raising() -> None:
    # A 7-string position against a 6-string tuning is a bug to surface as a score
    # of 0, not an exception that aborts a corpus run.
    assert pitch_validity_rate([placed(0.0, 40, 9, 0)], Tuning.STANDARD) == 0.0


def test_e4_counts_a_fret_past_the_neck_as_invalid() -> None:
    assert pitch_validity_rate([placed(0.0, 99, 0, 59)], Tuning.STANDARD) == 0.0


def test_e4_of_an_empty_tab_is_one_by_convention() -> None:
    # Vacuously valid: there is nothing inconsistent in an empty tab.
    assert pitch_validity_rate([], Tuning.STANDARD) == 1.0


# =========================================================== E2 match detail (for E5)


def test_match_flags_mark_which_estimates_were_correct() -> None:
    from tabsampler.eval.metrics import exact_tab_f1_with_matches

    ref = [placed(0.0, 40, 0, 0), placed(1.0, 45, 1, 0)]
    est = [placed(0.0, 40, 0, 0), placed(1.0, 45, 0, 5), placed(2.0, 52, 1, 7)]
    prf, flags = exact_tab_f1_with_matches(ref, est)
    # First matches; second is the right pitch on the wrong string; third is spurious.
    assert flags == (True, False, False)
    assert prf.n_match == 1


def test_match_flags_are_all_false_when_there_is_no_reference() -> None:
    from tabsampler.eval.metrics import exact_tab_f1_with_matches

    _, flags = exact_tab_f1_with_matches([], [placed(0.0, 40, 0, 0)])
    assert flags == (False,)


def test_match_flags_length_always_equals_the_estimate_count() -> None:
    from tabsampler.eval.metrics import exact_tab_f1_with_matches

    est = [placed(0.0, 40, 0, 0), placed(5.0, 45, 1, 0)]
    for ref in ([], [placed(0.0, 40, 0, 0)], [placed(9.0, 60, 3, 5)]):
        _, flags = exact_tab_f1_with_matches(ref, est)
        assert len(flags) == len(est)


def test_exact_tab_f1_still_agrees_with_the_detailed_version() -> None:
    from tabsampler.eval.metrics import exact_tab_f1_with_matches

    ref = [placed(0.0, 40, 0, 0), placed(1.0, 45, 1, 0)]
    est = [placed(0.0, 40, 0, 0)]
    assert exact_tab_f1(ref, est).f1 == exact_tab_f1_with_matches(ref, est)[0].f1

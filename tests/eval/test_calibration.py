"""Tests for E5 calibration."""

from __future__ import annotations

import pytest

from tabsampler.eval.calibration import (
    expected_calibration_error,
    reliability_curve,
)


def test_perfectly_calibrated_input_has_ece_zero() -> None:
    # Ten predictions at confidence 0.8, eight of them correct. They all land in bin
    # [0.8, 0.9), where mean confidence is 0.8 and accuracy is 8/10 = 0.8, so the gap
    # is zero and so is the weighted sum.
    confidences = [0.8] * 10
    correctness = [True] * 8 + [False] * 2
    assert expected_calibration_error(confidences, correctness, n_bins=10) == pytest.approx(0.0)


def test_always_confident_always_wrong_has_ece_one() -> None:
    assert expected_calibration_error([1.0] * 5, [False] * 5, n_bins=10) == pytest.approx(1.0)


def test_always_confident_always_right_has_ece_zero() -> None:
    assert expected_calibration_error([1.0] * 5, [True] * 5, n_bins=10) == pytest.approx(0.0)


def test_ece_matches_a_hand_computed_three_bin_example() -> None:
    # n_bins=10. Three populated bins:
    #   bin [0.2,0.3): confidences 0.25, 0.25    accuracy 0/2 = 0.0   gap |0.0-0.25|=0.25
    #   bin [0.5,0.6): confidences 0.5,  0.5     accuracy 1/2 = 0.5   gap |0.5-0.5| =0.0
    #   bin [0.9,1.0]: confidences 0.9,0.9,0.9,0.9 accuracy 3/4=0.75  gap |0.75-0.9|=0.15
    # N = 8, so ECE = (2/8)(0.25) + (2/8)(0.0) + (4/8)(0.15)
    #               = 0.0625 + 0 + 0.075 = 0.1375
    confidences = [0.25, 0.25, 0.5, 0.5, 0.9, 0.9, 0.9, 0.9]
    correctness = [False, False, True, False, True, True, True, False]
    assert expected_calibration_error(confidences, correctness, n_bins=10) == pytest.approx(0.1375)


def test_empty_bins_are_skipped_not_counted_as_zero_error() -> None:
    # The classic ECE bug: dividing by n_bins instead of by N. With 100 bins and two
    # predictions, 98 bins are empty. Dividing by the bin count would give ~0.005
    # instead of the correct 1.0.
    assert expected_calibration_error([1.0, 1.0], [False, False], n_bins=100) == pytest.approx(1.0)


def test_bin_count_does_not_change_a_single_valued_result() -> None:
    for n in (2, 5, 10, 50):
        assert expected_calibration_error([1.0] * 4, [False] * 4, n_bins=n) == pytest.approx(1.0)


def test_empty_input_is_zero_not_nan() -> None:
    assert expected_calibration_error([], [], n_bins=10) == 0.0


def test_confidence_of_exactly_one_lands_in_the_last_bin() -> None:
    curve = reliability_curve([1.0], [True], n_bins=10)
    assert len(curve) == 1
    assert curve[0].lower == pytest.approx(0.9)
    assert curve[0].upper == pytest.approx(1.0)


def test_reliability_curve_reports_only_populated_bins() -> None:
    curve = reliability_curve([0.05, 0.95], [False, True], n_bins=10)
    assert len(curve) == 2
    assert [b.count for b in curve] == [1, 1]


def test_reliability_curve_bin_contents_are_right() -> None:
    curve = reliability_curve([0.42, 0.44], [True, False], n_bins=10)
    (bin_,) = curve
    assert bin_.count == 2
    assert bin_.mean_confidence == pytest.approx(0.43)
    assert bin_.accuracy == pytest.approx(0.5)
    assert bin_.gap == pytest.approx(0.07)


def test_mismatched_lengths_are_refused() -> None:
    with pytest.raises(ValueError, match="correctness"):
        expected_calibration_error([0.5, 0.5], [True], n_bins=10)


def test_a_confidence_outside_zero_one_is_refused() -> None:
    with pytest.raises(ValueError, match="confidence"):
        expected_calibration_error([1.5], [True], n_bins=10)


def test_zero_bins_is_refused() -> None:
    with pytest.raises(ValueError, match="n_bins"):
        reliability_curve([0.5], [True], n_bins=0)

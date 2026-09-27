"""E5: calibration of the decoder's posteriors (spec 3.1).

Pure: no I/O, no global state.

When the decoder says 0.8, is it right about 80% of the time? Expected calibration error
bins predictions by confidence and measures the gap between average confidence and
observed accuracy in each bin, weighted by how many predictions fall in it::

    ECE = sum over bins of (n_b / N) * | accuracy_b - mean_confidence_b |

**The classic bug this avoids:** dividing by the number of bins rather than by the total
number of predictions. An empty bin must contribute nothing, not a zero-error term that
drags the average down. Weighting by ``n_b / N`` does that by construction, and there is
a test for it.

Reference: Guo et al. (2017), "On Calibration of Modern Neural Networks".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

DEFAULT_N_BINS = 10


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    """One bin of the reliability curve."""

    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float

    @property
    def gap(self) -> float:
        return abs(self.accuracy - self.mean_confidence)


def _validate(confidences: Sequence[float], correctness: Sequence[bool]) -> None:
    if len(confidences) != len(correctness):
        raise ValueError(f"{len(confidences)} confidences but {len(correctness)} correctness flags")
    for c in confidences:
        if not 0.0 <= c <= 1.0:
            raise ValueError(f"confidence {c} outside [0, 1]")


def reliability_curve(
    confidences: Sequence[float],
    correctness: Sequence[bool],
    n_bins: int = DEFAULT_N_BINS,
) -> list[CalibrationBin]:
    """Bin predictions by confidence. Empty bins are omitted.

    Bins are ``[0, 1/n)``, ``[1/n, 2/n)``, ..., ``[(n-1)/n, 1]`` -- the last is closed
    so a confidence of exactly 1.0 lands in it rather than falling off the end.
    """
    _validate(confidences, correctness)
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1, got {n_bins}")

    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for confidence, correct in zip(confidences, correctness, strict=True):
        index = min(int(confidence * n_bins), n_bins - 1)
        buckets[index].append((confidence, correct))

    out: list[CalibrationBin] = []
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        out.append(
            CalibrationBin(
                lower=index / n_bins,
                upper=(index + 1) / n_bins,
                count=len(bucket),
                mean_confidence=sum(c for c, _ in bucket) / len(bucket),
                accuracy=sum(1 for _, ok in bucket if ok) / len(bucket),
            )
        )
    return out


def expected_calibration_error(
    confidences: Sequence[float],
    correctness: Sequence[bool],
    n_bins: int = DEFAULT_N_BINS,
) -> float:
    """E5. 0.0 is perfectly calibrated; 1.0 is maximally wrong.

    Returns 0.0 for no predictions: there is no miscalibration to measure. Defined
    rather than NaN, because a corpus can legitimately contain a silent track.
    """
    _validate(confidences, correctness)
    total = len(confidences)
    if total == 0:
        return 0.0
    return sum(
        (b.count / total) * b.gap for b in reliability_curve(confidences, correctness, n_bins)
    )

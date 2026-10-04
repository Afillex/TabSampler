"""Writing rows to ``experiments/results.csv``, the project's ledger of executed runs.

One row per run. Two rules are enforced here rather than trusted:

- **A metric that was not computed is blank, never 0.0.** Writing a zero for something
  we did not measure is inventing a result.
- **The hypothesis is required**, and it is written before the run, not after.
"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

from tabsampler.types import CostWeights

#: The results schema. Fixed: every row ever written uses these columns, in this order.
RESULTS_COLUMNS = (
    "date",
    "commit",
    "config",
    "hypothesis",
    "dataset",
    "split",
    "mode",
    "E1",
    "E2",
    "E3",
    "E5",
    "E7",
    "seed",
    "notes",
)

VALID_MODES = ("oracle", "e2e")


def _fmt(value: float | None) -> str:
    """A measured number, or blank when it was not measured."""
    return "" if value is None else f"{value:.4g}"


def results_row(
    commit: str,
    config: str,
    hypothesis: str,
    dataset: str,
    split: str,
    mode: str,
    seed: int,
    e1: float | None = None,
    e2: float | None = None,
    e3: float | None = None,
    e5: float | None = None,
    e7: float | None = None,
    notes: str = "",
    date: str | None = None,
) -> dict[str, str]:
    """Build one results row.

    Raises:
        ValueError: if the hypothesis is blank, or the mode is not oracle/e2e.
    """
    if not hypothesis.strip():
        raise ValueError(
            "a results row needs a hypothesis, and it is written before the run, not after"
        )
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")

    return {
        "date": date or dt.datetime.now(dt.UTC).date().isoformat(),
        "commit": commit,
        "config": config,
        "hypothesis": hypothesis.strip(),
        "dataset": dataset,
        "split": split,
        "mode": mode,
        "E1": _fmt(e1),
        "E2": _fmt(e2),
        "E3": _fmt(e3),
        "E5": _fmt(e5),
        "E7": _fmt(e7),
        "seed": str(seed),
        "notes": notes,
    }


def describe_weights(weights: CostWeights) -> str:
    """A decoder's cost weights and temperature, stated rather than labelled.

    Rows once said "hand-set weights" whatever the decoder was, which mislabelled the runs
    with fitted weights; the values themselves cannot be wrong. The acoustic term is listed
    only once it is in use (Phase 3).
    """
    parts = [
        f"move {weights.move:g}",
        f"span {weights.span:g}",
        f"high {weights.high:g}",
        f"open_reward {weights.open_reward:g}",
    ]
    if any(weights.string_bias):
        parts.append("string_bias (" + ", ".join(f"{b:g}" for b in weights.string_bias) + ")")
    if weights.low_region:
        parts.append(f"low_region {weights.low_region:g}")
    if weights.high_region:
        parts.append(f"high_region {weights.high_region:g}")
    if weights.open_up_neck:
        parts.append(f"open_up_neck {weights.open_up_neck:g}")
    if weights.acoustic:
        parts.append(f"acoustic {weights.acoustic:g}")
    parts.append(f"temperature {weights.temperature:g}")
    return ", ".join(parts)


def m1_row_notes(
    *,
    channel: str,
    decoder: Path,
    weights: CostWeights,
    e1_raw: float,
    e3_transitions: float,
    e4: float,
    n_tracks: int,
    cache_hits: int,
    cache_misses: int,
) -> str:
    """The notes column of an ``eval-m1`` row, naming the decoder that produced it."""
    return (
        f"{channel}; decoder {decoder}: {describe_weights(weights)}; "
        f"E1 is the pipeline's (placed) figure, transcriber raw {e1_raw:.4f}; "
        f"E3 transitions {e3_transitions:.4f}; E4 {e4:.4f}; {n_tracks} tracks; "
        f"E7 is decode-only ({cache_hits} cache hits, {cache_misses} misses) and too noisy "
        f"to compare across runs (ADR 0018)"
    )


def append_row(path: Path | str, row: dict[str, str]) -> None:
    """Append ``row`` to the results CSV, writing the header if the file is new."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not target.exists() or target.stat().st_size == 0

    fieldnames: list[str] = list(RESULTS_COLUMNS)
    with target.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)

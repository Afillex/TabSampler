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

"""Tests for the experiments/results.csv writer."""

from __future__ import annotations

from pathlib import Path

import pytest

from tabsampler.results import RESULTS_COLUMNS, append_row, results_row


def test_columns_match_the_committed_schema() -> None:
    assert RESULTS_COLUMNS == (
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


def test_a_row_has_a_value_for_every_column() -> None:
    row = results_row(
        commit="abc1234",
        config="configs/phase0_eval_notes.yaml",
        hypothesis="Basic Pitch note F1 on GuitarSet is in a plausible range",
        dataset="guitarset",
        split="test",
        mode="e2e",
        e1=0.61,
        seed=0,
        notes="audio_mic; onset 0.5",
    )
    assert set(row) == set(RESULTS_COLUMNS)


def test_unmeasured_metrics_are_blank_not_zero() -> None:
    # Writing 0.0 for a metric we did not compute would be inventing a result.
    row = results_row(
        commit="abc1234",
        config="c.yaml",
        hypothesis="h",
        dataset="guitarset",
        split="test",
        mode="e2e",
        e1=0.61,
        seed=0,
    )
    assert row["E1"] == "0.61"
    assert row["E2"] == ""
    assert row["E3"] == ""
    assert row["E5"] == ""


def test_append_writes_a_header_once_then_data_rows(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    common = dict(
        commit="abc1234",
        config="c.yaml",
        hypothesis="h",
        dataset="guitarset",
        split="test",
        mode="e2e",
        seed=0,
    )
    append_row(path, results_row(e1=0.5, **common))
    append_row(path, results_row(e1=0.6, **common))

    lines = path.read_text().strip().splitlines()
    assert lines[0].startswith("date,commit,config")
    assert len(lines) == 3


def test_append_preserves_an_existing_header(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    path.write_text(",".join(RESULTS_COLUMNS) + "\n")
    append_row(
        path,
        results_row(
            commit="a",
            config="c",
            hypothesis="h",
            dataset="d",
            split="test",
            mode="oracle",
            e1=0.1,
            seed=0,
        ),
    )
    assert len(path.read_text().strip().splitlines()) == 2


def test_a_row_refuses_an_empty_hypothesis() -> None:
    # The hypothesis is written before the run, not after.
    with pytest.raises(ValueError, match="hypothesis"):
        results_row(
            commit="a",
            config="c",
            hypothesis="  ",
            dataset="d",
            split="test",
            mode="e2e",
            e1=0.1,
            seed=0,
        )


def test_mode_must_be_oracle_or_e2e() -> None:
    with pytest.raises(ValueError, match="mode"):
        results_row(
            commit="a",
            config="c",
            hypothesis="h",
            dataset="d",
            split="test",
            mode="whatever",
            e1=0.1,
            seed=0,
        )

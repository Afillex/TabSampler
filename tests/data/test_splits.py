"""Tests for the sacred split.

GUARDED AREA: this is the data split. ADR 0003 makes GuitarSet test-only. The point
of these tests is that the rule is mechanical -- a mistake fails loudly rather than
producing a number that looks fine.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tabsampler.data.splits import (
    EXPECTED_GUITARSET_TRACKS,
    VALIDATION_PLAYER,
    Split,
    assert_tuning_allowed,
    guitarset_test_ids,
    guitarset_track_ids,
    guitarset_validation_ids,
    record_test_set_access,
    split_by_player,
)
from tabsampler.errors import TestSetMisuseError


def snapshot(tmp_path: Path, ids: list[str], name: str = "ids.txt") -> Path:
    p = tmp_path / name
    p.write_text("\n".join(ids) + "\n")
    return p


# --------------------------------------------------------------- the split itself


def test_the_validation_player_is_fixed_by_adr_0037() -> None:
    # Chosen by rule, the lowest ID, before any per-player figure was seen.
    assert VALIDATION_PLAYER == "00"


def test_validation_is_player_00s_sixty_tracks() -> None:
    ids = guitarset_validation_ids()
    assert len(ids) == 60
    assert all(i.startswith("00_") for i in ids)


def test_the_test_set_is_the_other_five_players() -> None:
    ids = guitarset_test_ids()
    assert len(ids) == 300
    assert not any(i.startswith("00_") for i in ids)


def test_validation_and_test_partition_the_corpus() -> None:
    validation, test = set(guitarset_validation_ids()), set(guitarset_test_ids())
    assert not validation & test
    assert validation | test == set(guitarset_track_ids())


def test_a_player_with_the_wrong_track_count_is_refused() -> None:
    # A partial download would otherwise split unevenly and silently shrink one side.
    ids = sorted([f"00_t{i:02d}" for i in range(59)] + [f"01_t{i:02d}" for i in range(60)])
    with pytest.raises(ValueError, match="60"):
        split_by_player(ids, "00")


def test_ids_are_sorted_unique_and_non_empty(tmp_path: Path) -> None:
    p = snapshot(tmp_path, ["00_a_comp", "00_b_solo", "01_a_comp"])
    ids = guitarset_track_ids(p, expected_count=3)
    assert list(ids) == sorted(set(ids))
    assert len(ids) == 3


def test_split_is_deterministic_across_calls(tmp_path: Path) -> None:
    p = snapshot(tmp_path, ["00_a_comp", "01_b_solo"])
    assert guitarset_track_ids(p, expected_count=2) == guitarset_track_ids(p, expected_count=2)


def test_returns_a_tuple_so_a_caller_cannot_mutate_the_split(tmp_path: Path) -> None:
    p = snapshot(tmp_path, ["00_a_comp"])
    assert isinstance(guitarset_track_ids(p, expected_count=1), tuple)


def test_comments_and_blank_lines_are_ignored(tmp_path: Path) -> None:
    p = tmp_path / "c.txt"
    p.write_text("# GuitarSet test ids\n\n00_a_comp\n\n01_b_solo\n")
    assert guitarset_track_ids(p, expected_count=2) == ("00_a_comp", "01_b_solo")


# --------------------------------------------------------------- snapshot integrity


def test_a_missing_snapshot_raises_with_instructions(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="download_guitarset"):
        guitarset_track_ids(tmp_path / "absent.txt")


def test_an_unsorted_snapshot_is_rejected(tmp_path: Path) -> None:
    p = snapshot(tmp_path, ["01_b_solo", "00_a_comp"])
    with pytest.raises(ValueError, match="sorted"):
        guitarset_track_ids(p, expected_count=2)


def test_a_snapshot_with_duplicates_is_rejected(tmp_path: Path) -> None:
    p = snapshot(tmp_path, ["00_a_comp", "00_a_comp"])
    with pytest.raises(ValueError, match="duplicate"):
        guitarset_track_ids(p, expected_count=2)


def test_an_empty_snapshot_is_rejected(tmp_path: Path) -> None:
    # An empty test set would evaluate nothing and report a vacuous score.
    p = tmp_path / "e.txt"
    p.write_text("# nothing here\n")
    with pytest.raises(ValueError, match="empty"):
        guitarset_track_ids(p)


def test_a_truncated_snapshot_is_rejected(tmp_path: Path) -> None:
    # The realistic failure: a partial download produces fewer tracks than GuitarSet
    # has, and every metric is then computed over a silently different corpus.
    p = snapshot(tmp_path, ["00_a_comp", "01_b_solo"])
    with pytest.raises(ValueError, match="expected 360"):
        guitarset_track_ids(p, expected_count=EXPECTED_GUITARSET_TRACKS)


def test_guitarset_has_360_excerpts() -> None:
    # 6 players x 2 styles x 5 progressions x 3 tempi, per mirdata's loader docstring.
    assert EXPECTED_GUITARSET_TRACKS == 360


# --------------------------------------------------------------- the misuse guard


def test_tuning_against_the_test_split_is_refused() -> None:
    # ADR 0003. There is no legal way to tune on GuitarSet, because there is no
    # GuitarSet training or validation split in this project at all.
    with pytest.raises(TestSetMisuseError, match="ADR 0003"):
        assert_tuning_allowed(Split.TEST)


def test_tuning_is_allowed_on_validation_and_train() -> None:
    assert_tuning_allowed(Split.VALIDATION)
    assert_tuning_allowed(Split.TRAIN)


def test_split_has_exactly_the_three_expected_members() -> None:
    assert {s.name for s in Split} == {"TRAIN", "VALIDATION", "TEST"}


# --------------------------------------------------------------- the audit log


def test_recording_test_set_access_appends_a_timestamped_line(tmp_path: Path) -> None:
    log = tmp_path / "access.log"
    record_test_set_access("M1 milestone evaluation", log_path=log)
    record_test_set_access("Phase 0 sanity check", log_path=log)

    lines = [ln for ln in log.read_text().splitlines() if ln and not ln.startswith("#")]
    assert len(lines) == 2
    assert "M1 milestone evaluation" in lines[0]
    assert "Phase 0 sanity check" in lines[1]
    # An ISO-8601 date so the log is sortable and greppable.
    assert lines[0].startswith("20")


def test_the_access_log_records_the_commit_so_a_look_is_attributable(tmp_path: Path) -> None:
    log = tmp_path / "access.log"
    record_test_set_access("checking", log_path=log, commit="abc1234")
    assert "abc1234" in log.read_text()


def test_recording_refuses_an_empty_reason(tmp_path: Path) -> None:
    # A log entry with no reason defeats the purpose of having the log.
    with pytest.raises(ValueError):
        record_test_set_access("", log_path=tmp_path / "access.log")


def test_the_log_is_created_with_a_header_explaining_what_it_is(tmp_path: Path) -> None:
    log = tmp_path / "nested" / "access.log"
    record_test_set_access("first look", log_path=log)
    assert log.read_text().startswith("#")

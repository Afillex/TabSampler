"""Smoke tests for the DadaGP scripts' main paths, on synthetic sequences: no dataset.

The scripts read DadaGP through split files frozen by hash, so they cannot be pointed at a
synthetic archive; instead the loading function is replaced and everything after it runs.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

from tabsampler.fingering.fit import HumanSequence
from tabsampler.fingering.states import enumerate_states
from tabsampler.types import NoteEvent, NoteGroup, Position, TabNote, Tuning

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sequence(pitches: list[int], choice: int) -> HumanSequence:
    tuning = Tuning(n_frets=24)
    groups = tuple(
        NoteGroup.of([NoteEvent(onset=i * 0.5, offset=i * 0.5 + 0.4, pitch=p, confidence=1.0)])
        for i, p in enumerate(pitches)
    )
    states = tuple(
        (options := enumerate_states(g, tuning, 5))[choice % len(options)] for g in groups
    )
    return HumanSequence(groups, states, (5,) * len(groups))


def test_the_fit_script_runs_end_to_end_with_a_feature_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fit = load("fit_cost_weights")

    def fake_sequences(*_: object) -> list[tuple[str, str, HumanSequence]]:
        return [
            (f"song{i}", "clean" if i % 2 else "distorted", sequence([52 + i, 55, 59, 62 - i], i))
            for i in range(6)
        ]

    monkeypatch.setattr(fit, "sequences_for", fake_sequences)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fit_cost_weights.py",
            "archive.zip",
            "meta.json",
            "--split",
            "artist",
            "--skip-halves",
            "--features",
            "string",
            "--weights-out",
            str(tmp_path / "weights.json"),
            "--per-song-out",
            str(tmp_path / "songs"),
        ],
    )
    fit.main()
    weights = json.loads((tmp_path / "weights.json").read_text())
    assert len(weights["string_bias"]) == 6 and weights["string_bias"][0] == 0.0
    assert (tmp_path / "songs" / "hand-set.json").exists()
    assert (tmp_path / "songs" / "fitted.json").exists()


def test_the_comparison_prints_exact_chord_shape_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A pre-registered rule with a 0.0005 threshold cannot be judged from four decimals.
    compare = load("compare_validation")
    for name, playable in (("a", 58148), ("b", 58142)):
        (tmp_path / f"{name}.json").write_text(
            json.dumps(
                {
                    "per_song": {"s": {"clean": [5, 10]}},
                    "shapes": {"clean": [playable, 58163]},
                    "single_candidate": 0,
                }
            )
        )
    monkeypatch.setattr(
        sys, "argv", ["compare", str(tmp_path / "a.json"), str(tmp_path / "b.json")]
    )
    compare.main()
    out = capsys.readouterr().out
    assert "58148/58163 -> 58142/58163" in out
    assert "drop +0.000103" in out


def test_the_speed_estimate_compares_against_the_limit_it_replaced(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # After ADR 0031 the default limit is 48; the baseline this experiment reports
    # against must stay ADR 0011's 12, not whatever the default is today.
    speed = load("estimate_speed_limit")

    def fake_collect(*_: object) -> object:
        side = speed.Side()
        side.add([(0, 0.5)] * 900 + [(2, 0.5)] * 90 + [(4, 0.05)] * 10)
        return side

    monkeypatch.setattr(speed, "collect", fake_collect)
    monkeypatch.setattr(sys, "argv", ["estimate_speed_limit.py", "archive.zip", "meta.json"])
    speed.main()
    out = capsys.readouterr().out
    assert "at 12 frets/s" in out


def test_the_comparison_reads_modes_when_the_parts_are_oracle_and_e2e(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    compare = load("compare_validation")
    for name in ("a", "b"):
        (tmp_path / f"{name}.json").write_text(
            json.dumps(
                {
                    "per_song": {"00_x": {"oracle": [5, 10], "e2e": [8, 20]}},
                    "shapes": {"oracle": [9, 10], "e2e": [9, 10]},
                    "single_candidate": 0,
                }
            )
        )
    monkeypatch.setattr(
        sys, "argv", ["compare", str(tmp_path / "a.json"), str(tmp_path / "b.json")]
    )
    compare.main()
    lines = capsys.readouterr().out.splitlines()
    starts = [line.split(":")[0].strip() for line in lines if not line.startswith(" ")]
    assert starts == ["e2e", "oracle"]


def test_the_comparison_refuses_a_report_holding_a_test_track(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Comparing decoders on test-player tracks would choose on the test set (ADR 0003,
    # ADR 0037); the 2026-10-04 review found such counts beside the validation ones.
    from tabsampler.data.splits import guitarset_test_ids

    compare = load("compare_validation")
    test_track = guitarset_test_ids()[0]
    for name in ("a", "b"):
        (tmp_path / f"{name}.json").write_text(
            json.dumps(
                {
                    "per_song": {"00_x": {"oracle": [5, 10]}, test_track: {"oracle": [6, 10]}},
                    "shapes": {"oracle": [9, 10]},
                    "single_candidate": 0,
                }
            )
        )
    monkeypatch.setattr(
        sys, "argv", ["compare", str(tmp_path / "a.json"), str(tmp_path / "b.json")]
    )
    with pytest.raises(SystemExit, match=re.escape(test_track)):
        compare.main()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    ("part", "label"),
    [("clean", "1 songs, 10 notes"), ("oracle", "1 tracks, 10 estimated + reference notes")],
)
def test_the_comparison_names_what_it_counted(
    part: str,
    label: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A GuitarSet count is E2's 2 x matches over estimated + reference notes (add_track), so
    # it once printed "26446 notes" for player 00's 13,223.
    compare = load("compare_validation")
    for name in ("a", "b"):
        (tmp_path / f"{name}.json").write_text(
            json.dumps(
                {
                    "per_song": {"00_x": {part: [5, 10]}},
                    "shapes": {part: [9, 10]},
                    "single_candidate": 0,
                }
            )
        )
    monkeypatch.setattr(
        sys, "argv", ["compare", str(tmp_path / "a.json"), str(tmp_path / "b.json")]
    )
    compare.main()
    assert f"({label})" in capsys.readouterr().out


def test_the_lattice_measure_refuses_more_tracks_than_the_validation_player_has(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # It once measured player 00's 60 tracks and labelled them with whatever N was asked for.
    lattice = load("measure_lattice")

    def unreachable(*_: object) -> None:
        raise AssertionError("measured although more tracks were asked for than exist")

    monkeypatch.setattr(lattice, "measure_guitarset", unreachable)
    monkeypatch.setattr(sys, "argv", ["measure_lattice", "--guitarset", "61"])
    with pytest.raises(SystemExit) as stopped:
        lattice.main()
    assert stopped.value.code == 2
    assert "60" in capsys.readouterr().err


# ------------------------------------------- scripts/analyse_errors.py (C3, 2026-10-04 Task 1)


def placed(onset: float, pitch: int, string: int, fret: int) -> tuple[NoteEvent, Position]:
    note = NoteEvent(onset=onset, offset=onset + 0.4, pitch=pitch, confidence=1.0)
    return note, Position(string=string, fret=fret)


def tab(onset: float, pitch: int, string: int, fret: int) -> TabNote:
    note, position = placed(onset, pitch, string, fret)
    return TabNote(note=note, position=position, posterior=1.0)


def test_the_error_analysis_reads_style_and_mode_from_the_track_id() -> None:
    errors = load("analyse_errors")
    assert errors.track_style_and_mode("00_BN1-129-Eb_comp") == ("BN", "comp")
    assert errors.track_style_and_mode("00_Funk2-108-Eb_solo") == ("Funk", "solo")
    assert errors.track_style_and_mode("00_SS3-84-Bb_comp") == ("SS", "comp")
    with pytest.raises(ValueError, match="not a GuitarSet track id"):
        errors.track_style_and_mode("Some Artist - Some Song")


def test_the_error_analysis_pairs_notes_by_pitch_and_onset_not_by_order() -> None:
    errors = load("analyse_errors")
    reference = [placed(0.0, 59, 4, 0), placed(0.0, 64, 5, 0), placed(1.0, 69, 5, 5)]
    reference.append(placed(2.0, 71, 5, 7))  # the decoder dropped this one
    decoded = [tab(0.0, 64, 4, 5), tab(0.0, 59, 3, 4), tab(1.03, 69, 4, 10)]
    assert errors.pair(reference, decoded) == [
        Position(3, 4),
        Position(4, 5),
        Position(4, 10),
        None,
    ]


def test_the_error_analysis_prefers_the_players_string_among_equal_notes() -> None:
    # A unison on two strings: each reference note must find the decoded copy on its own
    # string, as E2's matching would, rather than the first copy in the list.
    errors = load("analyse_errors")
    reference = [placed(0.0, 64, 5, 0), placed(0.0, 64, 4, 5)]
    decoded = [tab(0.0, 64, 4, 5), tab(0.0, 64, 5, 0)]
    assert errors.pair(reference, decoded) == [Position(5, 0), Position(4, 5)]


def test_the_error_analysis_marks_notes_that_sound_together() -> None:
    errors = load("analyse_errors")
    reference = [placed(0.0, 59, 4, 0), placed(0.01, 64, 5, 0), placed(1.0, 69, 5, 5)]
    assert errors.chord_flags(reference, window_s=0.03) == [True, True, False]


def test_errors_count_as_a_run_only_from_four_in_a_row() -> None:
    errors = load("analyse_errors")
    assert errors.errors_in_long_runs([False, True, True, True, True, False, True]) == 4
    assert errors.errors_in_long_runs([True] * 4) == 4  # a run at the end still counts
    assert errors.errors_in_long_runs([True, True, True, False, True]) == 0


def test_the_error_report_answers_each_question_from_the_outcomes() -> None:
    errors = load("analyse_errors")

    def outcome(onset: float, human: Position, decoded: Position | None) -> object:
        return errors.Outcome("00_BN1-129-Eb_comp", "BN", "comp", onset, False, human, decoded)

    outcomes = [
        outcome(0.0, Position(5, 0), Position(5, 0)),  # right
        outcome(0.5, Position(2, 2), Position(1, 7)),  # a string lower, up the neck
        outcome(1.0, Position(3, 0), Position(2, 5)),  # the player's open string, fretted
        outcome(1.5, Position(1, 5), Position(2, 0)),  # fretted by the player, open here
        outcome(2.0, Position(0, 3), None),  # dropped
    ]
    lines = errors.report(outcomes)
    assert lines[0] == "notes 5, right 1, misplaced 3, dropped 1"
    text = "\n".join(lines)
    assert f"  {'1 string':20s} {errors.share(3, 3)}" in text
    assert f"  {'higher on the neck':20s} {errors.share(2, 3)}" in text
    assert f"misplaced and in another region: {errors.share(3, 3)}" in text
    assert f"player open, decoder fretted: {errors.share(1, 3)}" in text
    assert f"decoder open, player fretted: {errors.share(1, 3)}" in text
    assert f"errors in runs of 4 or more: {errors.share(4, 4)}" in text

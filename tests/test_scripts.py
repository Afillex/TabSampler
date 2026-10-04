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


def test_the_fit_script_can_hold_the_base_weights_and_fit_one_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ADR 0039's experiment: one weight fitted, every other held at its hand-set value.
    from tabsampler.config import load_phase1_config

    fit = load("fit_cost_weights")

    def fake_sequences(*_: object) -> list[tuple[str, str, HumanSequence]]:
        return [(f"song{i}", "clean", sequence([52 + i, 55, 59, 62 - i], i)) for i in range(6)]

    monkeypatch.setattr(fit, "sequences_for", fake_sequences)
    out = tmp_path / "weights.json"
    argv = ["fit_cost_weights.py", "archive.zip", "meta.json", "--split", "artist"]
    argv += ["--skip-halves", "--hold-base"]
    monkeypatch.setattr(sys, "argv", [*argv, "--features", "open", "--weights-out", str(out)])
    fit.main()
    weights = json.loads(out.read_text())
    hand_set = load_phase1_config(Path("configs/phase1_baseline.yaml")).weights
    for name in ("move", "span", "high", "open_reward"):
        assert weights[name] == getattr(hand_set, name)
    assert "open_up_neck" in weights

    monkeypatch.setattr(sys, "argv", argv)  # holding the base and naming no group fits nothing
    with pytest.raises(SystemExit):
        fit.main()


def test_the_fit_script_can_start_from_the_default_and_hold_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A group fitted on top of today's default must keep every other weight at the default's
    # value, including ADR 0039's, which the hand-set config does not have.
    from tabsampler.config import load_phase1_config

    fit = load("fit_cost_weights")

    def fake_sequences(*_: object) -> list[tuple[str, str, HumanSequence]]:
        return [(f"song{i}", "clean", sequence([52 + i, 55, 59, 62 - i], i)) for i in range(6)]

    monkeypatch.setattr(fit, "sequences_for", fake_sequences)
    out, songs = tmp_path / "weights.json", tmp_path / "songs"
    argv = ["fit_cost_weights.py", "archive.zip", "meta.json", "--split", "artist"]
    argv += ["--skip-halves", "--hold-base", "--base-config", "configs/decoder_clean.yaml"]
    argv += ["--features", "region", "--weights-out", str(out), "--per-song-out", str(songs)]
    monkeypatch.setattr(sys, "argv", argv)
    fit.main()
    weights = json.loads(out.read_text())
    default = load_phase1_config(Path("configs/decoder_clean.yaml")).weights
    for name in ("move", "span", "high", "open_reward", "open_up_neck", "temperature"):
        assert weights[name] == getattr(default, name)
    assert (songs / "base.json").exists() and (songs / "fitted.json").exists()


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


def test_the_error_analysis_pairs_a_unison_as_e2_matches_it() -> None:
    # 00_Jazz1-200-B_solo: the player's D on the low E string and on the A string, 3 ms
    # apart. Taken in order, the first note took the copy on the A string and neither
    # counted as right; E2's matching counts one.
    errors = load("analyse_errors")
    reference = [placed(3.3245, 50, 0, 10), placed(3.3278, 50, 1, 5)]
    decoded = [tab(3.3245, 50, 1, 5), tab(3.3278, 50, 2, 0)]
    assert errors.pair(reference, decoded) == [Position(2, 0), Position(1, 5)]


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


def test_the_error_analysis_follows_each_hand_from_group_to_group() -> None:
    errors = load("analyse_errors")
    # Fret 5 puts the hand at 5 (frets 5-9); an open string carries it; fret 10 moves it up
    # one, to 6; a dropped note moves nothing.
    groups = [0, 0, 1, 2, 3]
    positions = [Position(1, 5), Position(2, 0), Position(3, 0), Position(2, 10), None]
    assert errors.hand_indices(groups, positions) == [5, 5, 5, 6, 6]
    assert errors.hand_indices([0, 1], [Position(5, 0), Position(4, 0)]) == [None, None]


def test_question_seven_tables_the_decoders_open_strings_by_both_hands() -> None:
    errors = load("analyse_errors")

    def opened(player: int | None, decoder: int | None) -> object:
        human, decoded = Position(2, 5), Position(3, 0)  # the same D, fretted and open
        return errors.Outcome(
            "00_BN1-129-Eb_solo", "BN", "solo", 0.0, False, human, decoded, player, decoder
        )

    outcomes = [opened(5, 5), opened(5, 2), opened(2, None)]
    text = "\n".join(errors.report(outcomes))

    def row(name: str, cells: tuple[int, int, int]) -> str:
        return f"  {name:12s}" + "".join(f"{cell:10d}" for cell in cells)

    assert row("none yet", (0, 1, 0)) in text
    assert row("1-4", (0, 0, 1)) in text
    assert row("5+", (0, 0, 1)) in text


@pytest.mark.parametrize(("passing", "verdict"), [(95, "a clear drop"), (100, "no clear drop")])
def test_the_comparison_gives_the_chord_shape_change_an_interval(
    passing: int,
    verdict: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # ADR 0039's rule, Ege's decision of 2026-10-04: a chord-shape drop refuses a challenger
    # only when its whole 95% track-level interval is below zero.
    compare = load("compare_validation")
    tracks = [f"00_t{i}" for i in range(5)]
    for name, passed in (("a", 100), ("b", passing)):
        (tmp_path / f"{name}.json").write_text(
            json.dumps(
                {
                    "per_song": {t: {"oracle": [8, 10]} for t in tracks},
                    "shapes": {"oracle": [5 * passed, 500]},
                    "single_candidate": 0,
                    "per_song_shapes": {t: {"oracle": [passed, 100]} for t in tracks},
                }
            )
        )
    monkeypatch.setattr(
        sys, "argv", ["compare", str(tmp_path / "a.json"), str(tmp_path / "b.json")]
    )
    compare.main()
    out = capsys.readouterr().out
    assert "chord-shape change" in out and out.rstrip().endswith(verdict)


def test_training_runs_checkpoints_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ADR 0041: background jobs here die at 30 minutes and when the machine sleeps, so a run
    # must pick up where it stopped rather than start over.
    train = load("train_model")

    def fake_sequences(*_: object) -> list[HumanSequence]:
        return [sequence([52 + i % 5, 55, 59, 62 - i % 3, 57, 64], i) for i in range(6)]

    monkeypatch.setattr(train, "clean_sequences", fake_sequences)
    run = tmp_path / "run"
    argv = ["train_model.py", "archive.zip", "meta.json", "--run", str(run)]
    argv += ["--chunk", "4", "--batch-size", "2"]
    monkeypatch.setattr(sys, "argv", [*argv, "--max-epochs", "1"])
    train.main()
    history = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines()]
    assert [entry["epoch"] for entry in history] == [1]
    assert (run / "checkpoint.pt").exists() and (run / "best.pt").exists()
    assert history[0]["device"] == "cpu" and history[0]["validation_nll_per_group"] > 0

    monkeypatch.setattr(sys, "argv", [*argv, "--max-epochs", "2"])
    train.main()  # resumes at epoch 2 rather than repeating epoch 1
    history = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines()]
    assert [entry["epoch"] for entry in history] == [1, 2]


def test_a_resumed_run_trains_exactly_as_an_uninterrupted_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    train = load("train_model")
    monkeypatch.setattr(
        train,
        "clean_sequences",
        lambda *_: [sequence([52 + i % 5, 55, 59, 62 - i % 3, 57, 64], i) for i in range(6)],
    )

    def run(directory: Path, *epochs: int) -> dict[str, torch.Tensor]:
        argv = ["train_model.py", "a.zip", "m.json", "--run", str(directory)]
        argv += ["--chunk", "4", "--batch-size", "2", "--patience", "99"]
        for limit in epochs:
            monkeypatch.setattr(sys, "argv", [*argv, "--max-epochs", str(limit)])
            train.main()
        return torch.load(directory / "checkpoint.pt", weights_only=True)["model"]

    straight = run(tmp_path / "straight", 3)
    interrupted = run(tmp_path / "interrupted", 1, 3)
    assert straight.keys() == interrupted.keys()
    assert all(torch.equal(straight[k], interrupted[k]) for k in straight)


def test_the_model_evaluation_reads_only_the_validation_player(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Player 00 only (ADR 0037); and the untrained model, whose learned term is zero, must place
    # the notes as the default decoder does -- the plumbing check on real data, here on fakes.
    from tabsampler.fingering.states import enumerate_states as states_for

    evaluate = load("evaluate_model")
    validation = ["00_BN1-129-Eb_comp", "00_Funk2-108-Eb_solo"]
    asked: list[str] = []

    class Dataset:
        def track(self, track_id: str) -> str:
            asked.append(track_id)
            return track_id

    def reference(track: str, tuning: Tuning) -> list[tuple[NoteEvent, Position]]:
        shift = len(track) % 3
        out = []
        for i, pitch in enumerate([52 + shift, 55, 59, 64, 57 + shift, 60]):
            group = NoteGroup.of([NoteEvent(0.5 * i, 0.5 * i + 0.4, pitch, 1.0)])
            out.append((group.notes[0], states_for(group, tuning, 5)[0].positions[0]))
        return out

    monkeypatch.setattr(evaluate, "guitarset_validation_ids", lambda: tuple(validation))
    monkeypatch.setattr(evaluate, "load_dataset", lambda *_: Dataset())
    monkeypatch.setattr(evaluate, "reference_tab", reference)
    monkeypatch.setattr(
        evaluate, "reference_notes", lambda track: [n for n, _ in reference(track, Tuning())]
    )
    looks: list[str] = []
    monkeypatch.setattr(evaluate, "record_test_set_access", looks.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_model.py",
            "--split",
            "validation",
            "--untrained",
            "--out",
            str(tmp_path / "out"),
        ],
    )
    evaluate.main()
    assert asked == validation and looks == []
    scores = {name: json.loads((tmp_path / "out" / f"{name}.json").read_text()) for name in "abc"}
    assert scores["c"]["per_song"] == scores["a"]["per_song"]
    assert all(score["split"] == "validation" for score in scores.values())


def test_the_model_evaluation_on_the_test_players_logs_the_look_and_writes_no_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Task C6's one look (ADR 0003, ADR 0037): logged with its reason before anything is read,
    # and no per-track counts, which are what decoders are chosen with.
    from tabsampler.fingering.states import enumerate_states as states_for

    evaluate = load("evaluate_model")
    test = ["01_BN1-129-Eb_comp", "02_Jazz1-130-D_solo"]
    order: list[str] = []

    class Dataset:
        def track(self, track_id: str) -> str:
            order.append(f"read {track_id}")
            return track_id

    def reference(track: str, tuning: Tuning) -> list[tuple[NoteEvent, Position]]:
        out = []
        for i, pitch in enumerate([52, 55, 59, 64]):
            group = NoteGroup.of([NoteEvent(0.5 * i, 0.5 * i + 0.4, pitch, 1.0)])
            out.append((group.notes[0], states_for(group, tuning, 5)[0].positions[0]))
        return out

    monkeypatch.setattr(evaluate, "guitarset_test_ids", lambda: tuple(test))
    monkeypatch.setattr(evaluate, "load_dataset", lambda *_: Dataset())
    monkeypatch.setattr(evaluate, "reference_tab", reference)
    monkeypatch.setattr(
        evaluate, "reference_notes", lambda track: [n for n, _ in reference(track, Tuning())]
    )
    monkeypatch.setattr(evaluate, "record_test_set_access", lambda reason: order.append("look"))
    monkeypatch.setattr(sys, "argv", ["evaluate_model.py", "--split", "test", "--untrained"])
    evaluate.main()
    assert order == ["look", *(f"read {t}" for t in test)]
    assert "(c) E2" in capsys.readouterr().out
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_model.py", "--split", "test", "--untrained", "--out", str(tmp_path / "x")],
    )
    with pytest.raises(SystemExit):
        evaluate.main()
    assert not (tmp_path / "x").exists()


def test_the_string_training_runs_checkpoints_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np

    from tabsampler.audio.windows import WINDOW_BINS, WINDOW_FRAMES

    train = load("train_strings")
    rng = np.random.default_rng(0)

    def side(count: int) -> dict[str, np.ndarray]:
        return {
            "windows": rng.standard_normal((count, WINDOW_BINS, WINDOW_FRAMES)).astype(np.float16),
            "pitches": rng.integers(45, 70, count).astype(np.int16),
            "possible": np.ones((count, 6), dtype=bool),
            "strings": rng.integers(0, 6, count).astype(np.int8),
        }

    monkeypatch.setattr(train, "examples_for", lambda *_: (side(12), side(6)))
    run = tmp_path / "run"
    argv = ["train_strings.py", "root", "--run", str(run), "--batch-size", "4"]
    monkeypatch.setattr(sys, "argv", [*argv, "--max-epochs", "1"])
    train.main()
    monkeypatch.setattr(sys, "argv", [*argv, "--max-epochs", "2"])
    train.main()  # resumes from the checkpoint, reusing the cached examples
    history = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines()]
    assert [entry["epoch"] for entry in history] == [1, 2]
    assert history[0]["chance"] == pytest.approx(1 / 6)
    assert (run / "best.pt").exists() and (run / "examples.npz").exists()


def test_the_string_evaluation_reads_only_the_validation_player(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import numpy as np
    import torch

    from tabsampler.model.strings import StringClassifier

    evaluate = load("evaluate_strings")
    validation = ["00_BN1-129-Eb_comp", "00_Jazz1-130-D_solo"]
    asked: list[str] = []

    class Track:
        def __init__(self, track_id: str) -> None:
            self.audio_mic_path = track_id

    class Dataset:
        def track(self, track_id: str) -> Track:
            asked.append(track_id)
            return Track(track_id)

    def reference(track: Track, tuning: Tuning) -> list[tuple[NoteEvent, Position]]:
        return [
            (NoteEvent(0.5 * i, 0.5 * i + 0.4, pitch, 1.0), Position(string, fret))
            for i, (pitch, string, fret) in enumerate([(55, 3, 0), (57, 2, 7), (64, 4, 5)])
        ]

    monkeypatch.setattr(evaluate, "guitarset_validation_ids", lambda: tuple(validation))
    monkeypatch.setattr(evaluate, "load_dataset", lambda *_: Dataset())
    monkeypatch.setattr(evaluate, "reference_tab", reference)
    monkeypatch.setattr(
        evaluate, "reference_notes", lambda t: [n for n, _ in reference(t, Tuning())]
    )
    monkeypatch.setattr(evaluate, "load_audio", lambda _: np.zeros(44100, dtype=np.float32))
    torch.manual_seed(0)
    (tmp_path / "run").mkdir()
    torch.save(StringClassifier().state_dict(), tmp_path / "run" / "best.pt")
    monkeypatch.setattr(sys, "argv", ["evaluate_strings.py", "--run", str(tmp_path / "run")])
    evaluate.main()
    assert asked == validation
    out = capsys.readouterr().out
    assert "the classifier" in out and "the default decoder" in out and "chance" in out


def test_the_acoustic_ablation_reads_only_the_validation_player(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # And with the weight at zero, (c) is (a): the ablation's two arms differ only by the term.
    import numpy as np
    import torch

    from tabsampler.model.strings import StringClassifier

    ablation = load("evaluate_acoustic")
    validation = ["00_BN1-129-Eb_comp", "00_Jazz1-130-D_solo"]
    asked: list[str] = []

    class Track:
        def __init__(self, track_id: str) -> None:
            self.audio_mic_path = track_id

    class Dataset:
        def track(self, track_id: str) -> Track:
            asked.append(track_id)
            return Track(track_id)

    def reference(track: Track, tuning: Tuning) -> list[tuple[NoteEvent, Position]]:
        return [
            (NoteEvent(0.5 * i, 0.5 * i + 0.4, pitch, 1.0), Position(string, fret))
            for i, (pitch, string, fret) in enumerate(
                [(55, 3, 0), (57, 2, 7), (64, 4, 5), (52, 1, 7)]
            )
        ]

    monkeypatch.setattr(ablation, "guitarset_validation_ids", lambda: tuple(validation))
    monkeypatch.setattr(ablation, "load_dataset", lambda *_: Dataset())
    monkeypatch.setattr(ablation, "reference_tab", reference)
    monkeypatch.setattr(
        ablation, "reference_notes", lambda t: [n for n, _ in reference(t, Tuning())]
    )
    monkeypatch.setattr(ablation, "load_audio", lambda _: np.zeros(44100, dtype=np.float32))
    torch.manual_seed(0)
    (tmp_path / "run").mkdir()
    torch.save(StringClassifier().state_dict(), tmp_path / "run" / "best.pt")
    looks: list[str] = []
    monkeypatch.setattr(ablation, "record_test_set_access", looks.append)
    argv = ["evaluate_acoustic.py", "--split", "validation", "--run", str(tmp_path / "run")]
    argv += ["--out", str(tmp_path / "out")]
    monkeypatch.setattr(sys, "argv", [*argv, "--weight", "0.0"])
    ablation.main()
    assert asked == validation and looks == []
    a, c = (json.loads((tmp_path / "out" / f"{x}.json").read_text()) for x in "ac")
    assert a["per_song"] == c["per_song"] and a["split"] == c["split"] == "validation"


def test_the_calibration_keeps_the_best_weight_and_the_smaller_on_a_tie() -> None:
    calibrate = load("calibrate_acoustic")
    assert calibrate.choose_weight({0.0: 10, 0.1: 12, 0.25: 11}) == 0.1
    assert calibrate.choose_weight({0.0: 12, 0.5: 12, 1.0: 3}) == 0.0


def test_the_acoustic_ablation_on_the_test_players_logs_the_look_and_writes_no_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import numpy as np
    import torch

    from tabsampler.model.strings import StringClassifier

    ablation = load("evaluate_acoustic")
    test = ["01_BN1-129-Eb_comp", "02_Jazz1-130-D_solo"]
    order: list[str] = []

    class Track:
        def __init__(self, track_id: str) -> None:
            self.audio_mic_path = track_id

    class Dataset:
        def track(self, track_id: str) -> Track:
            order.append(f"read {track_id}")
            return Track(track_id)

    def reference(track: Track, tuning: Tuning) -> list[tuple[NoteEvent, Position]]:
        return [
            (NoteEvent(0.0, 0.4, 55, 1.0), Position(3, 0)),
            (NoteEvent(0.5, 0.9, 57, 1.0), Position(2, 7)),
        ]

    monkeypatch.setattr(ablation, "guitarset_test_ids", lambda: tuple(test))
    monkeypatch.setattr(ablation, "load_dataset", lambda *_: Dataset())
    monkeypatch.setattr(ablation, "reference_tab", reference)
    monkeypatch.setattr(
        ablation, "reference_notes", lambda t: [n for n, _ in reference(t, Tuning())]
    )
    monkeypatch.setattr(ablation, "load_audio", lambda _: np.zeros(44100, dtype=np.float32))
    monkeypatch.setattr(ablation, "record_test_set_access", lambda reason: order.append("look"))
    torch.manual_seed(0)
    (tmp_path / "run").mkdir()
    torch.save(StringClassifier().state_dict(), tmp_path / "run" / "best.pt")
    argv = [
        "evaluate_acoustic.py",
        "--split",
        "test",
        "--run",
        str(tmp_path / "run"),
        "--weight",
        "0.25",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    ablation.main()
    assert order == ["look", *(f"read {t}" for t in test)]
    assert "(c)" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", [*argv, "--out", str(tmp_path / "x")])
    with pytest.raises(SystemExit):
        ablation.main()
    assert not (tmp_path / "x").exists()

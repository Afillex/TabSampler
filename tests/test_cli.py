"""The CLI's defaults (ADR 0032) and its guards around GuitarSet's test players (ADR 0037)."""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tabsampler import cli


def plain(output: str) -> str:
    """CLI output without colour codes. Typer colours its errors when GITHUB_ACTIONS,
    FORCE_COLOR or PY_COLORS is set, which splits '--split' into '-' and '-split'."""
    return re.sub(r"\x1b\[[0-9;]*m", "", output)


class Stopped(Exception):
    """Raised where GuitarSet would be loaded: a command gets that far and no further."""


@pytest.fixture
def looks(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The test-set looks a command logs, with GuitarSet's loading replaced by Stopped."""
    logged: list[str] = []

    def stop(*_: object, **__: object) -> None:
        raise Stopped

    monkeypatch.setattr(cli, "record_test_set_access", logged.append)
    monkeypatch.setattr(cli, "load_dataset", stop)  # stop before any audio is read
    return logged


def test_every_command_defaults_to_the_clean_guitar_decoder() -> None:
    # Clean or acoustic guitar is the typical recording, so its decoder is the default;
    # distorted guitar passes configs/decoder_distorted.yaml (ADR 0032).
    clean = Path("configs/decoder_clean.yaml")
    assert inspect.signature(cli.eval_m1).parameters["decoder"].default == clean
    assert inspect.signature(cli.transcribe).parameters["config"].default == clean
    assert inspect.signature(cli.diagnose).parameters["decoder"].default == clean


def make_recipe(target: str) -> str:
    makefile = (Path(__file__).resolve().parents[1] / "Makefile").read_text()
    return makefile.split(f"\n{target}:", 1)[1].split("\n\n", 1)[0]


def test_make_eval_m1_names_the_decoder_its_hypothesis_describes() -> None:
    # configs/m1_full_eval.yaml's hypothesis is about a hand-set cost model. Without an
    # explicit --decoder-config the target would run the CLI default -- since ADR 0032 the
    # clean-fitted decoder -- and write a row whose hypothesis describes another model.
    recipe = make_recipe("eval-m1")
    assert "--config configs/m1_full_eval.yaml" in recipe
    assert "--decoder-config configs/phase1_baseline.yaml" in recipe


def test_eval_m1_runs_only_on_a_split_named_on_the_command_line(looks: list[str]) -> None:
    # Choices are made with eval-m1 --split validation (ADR 0037). Were test the default, one
    # forgotten flag would read the test players; DadaGP's scripts likewise refuse to run
    # without --split (ADR 0024).
    result = CliRunner().invoke(cli.app, ["eval-m1", "--dry-run"])
    assert result.exit_code == 2 and "--split" in plain(result.output)
    assert looks == []
    assert "--split test" in make_recipe("eval-m1")


def test_a_validation_run_does_not_touch_the_test_set_log(looks: list[str]) -> None:
    # Player 00 is validation data (ADR 0037): reading it is not a look at the test set.
    runner = CliRunner()
    result = runner.invoke(cli.app, ["eval-m1", "--split", "validation", "--dry-run"])
    assert isinstance(result.exception, Stopped)
    assert looks == []
    result = runner.invoke(cli.app, ["eval-m1", "--split", "test", "--dry-run"])
    assert isinstance(result.exception, Stopped)
    assert len(looks) == 1 and "300 GuitarSet test tracks" in looks[0]


def test_per_track_counts_are_refused_on_the_test_split(looks: list[str], tmp_path: Path) -> None:
    # Per-track counts are what scripts/compare_validation.py chooses a decoder with; written
    # for the test players, they invite a choice made on the test set (ADR 0003, ADR 0037).
    out = tmp_path / "counts.json"
    args = ["eval-m1", "--per-track-out", str(out), "--dry-run"]
    runner = CliRunner()
    result = runner.invoke(cli.app, [*args, "--split", "test"])
    assert result.exit_code == 2 and "--per-track-out" in plain(result.output)
    assert looks == []  # refused before the test set is looked at
    assert not out.exists()
    result = runner.invoke(cli.app, [*args, "--split", "validation"])
    assert isinstance(result.exception, Stopped)  # the validation player is free to use


def test_a_test_set_look_names_the_decoder(looks: list[str]) -> None:
    # The two looks of 2026-10-03 logged the same reason though each ran another decoder.
    decoder = "configs/phase1_baseline.yaml"
    result = CliRunner().invoke(
        cli.app, ["eval-m1", "--split", "test", "--decoder-config", decoder, "--dry-run"]
    )
    assert isinstance(result.exception, Stopped)
    assert len(looks) == 1 and decoder in looks[0]


def test_per_track_counts_record_the_split_and_the_weights() -> None:
    # A config path alone can name other weights after the run: configs/decoder_clean.yaml
    # held the clean fit when cache/validation/p00-clean-fit.json was written, and the
    # hand-set weights from ADR 0038 on.
    from types import SimpleNamespace

    from tabsampler.eval.metrics import PRF
    from tabsampler.eval.playability import PlayabilityReport
    from tabsampler.eval.recovery import RecoveryReport
    from tabsampler.results import describe_weights
    from tabsampler.types import CostWeights

    track = SimpleNamespace(
        track_id="00_a",
        e2=PRF(0.8, 0.8, 0.8, n_ref=10, n_est=10, n_match=8),
        e3=PlayabilityReport(n_groups=5, n_groups_pass=5, n_transitions=4, n_transitions_pass=4),
    )
    weights = CostWeights(temperature=1.5728)
    payload = cli._per_track_payload(  # pyright: ignore[reportPrivateUsage]
        {"oracle": SimpleNamespace(tracks=(track,))},
        split="validation",
        decoder=Path("configs/decoder_clean.yaml"),
        weights=weights,
    )
    assert payload["split"] == "validation"
    assert payload["weights"] == describe_weights(weights)
    assert "temperature 1.5728" in payload["weights"]
    # Still what scripts/compare_validation.py reads.
    assert RecoveryReport.from_dict(payload).song_counts("oracle") == {"00_a": (16, 20)}


def test_transcribe_uses_the_thresholds_phase_5_adopted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ADR 0057: a user's tab comes from Basic Pitch at onset 0.7 / frame 0.4 / min 58 ms.
    from tabsampler.transcribe.basic_pitch_cli import CHOSEN_PARAMS

    built: list[object] = []

    class Fake:
        def __init__(self, exe: str, params: object = None) -> None:
            built.append(params)

        def transcribe_file(self, path: Path) -> list[object]:
            return []

    monkeypatch.setattr(cli, "BasicPitchCLITranscriber", Fake)
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")
    result = CliRunner().invoke(cli.app, ["transcribe", str(audio)])
    assert result.exit_code == 0, result.output
    assert built == [CHOSEN_PARAMS]
    assert (CHOSEN_PARAMS.onset_threshold, CHOSEN_PARAMS.frame_threshold) == (0.7, 0.4)
    assert CHOSEN_PARAMS.minimum_note_length_ms == 58.0

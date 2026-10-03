"""The CLI's defaults (ADR 0032): what a user gets without passing a decoder config."""

from __future__ import annotations

import inspect
from pathlib import Path

from tabsampler import cli


def test_every_command_defaults_to_the_clean_guitar_decoder() -> None:
    # Clean or acoustic guitar is the typical recording, so its decoder is the default;
    # distorted guitar passes configs/decoder_distorted.yaml (ADR 0032).
    clean = Path("configs/decoder_clean.yaml")
    assert inspect.signature(cli.eval_m1).parameters["decoder"].default == clean
    assert inspect.signature(cli.transcribe).parameters["config"].default == clean
    assert inspect.signature(cli.diagnose).parameters["decoder"].default == clean


def test_make_eval_m1_names_the_decoder_its_hypothesis_describes() -> None:
    # configs/m1_full_eval.yaml's hypothesis is about a hand-set cost model. Without an
    # explicit --decoder-config the target would run the CLI default -- since ADR 0032 the
    # clean-fitted decoder -- and write a row whose hypothesis describes another model.
    makefile = (Path(__file__).resolve().parents[1] / "Makefile").read_text()
    recipe = makefile.split("\neval-m1:", 1)[1].split("\n\n", 1)[0]
    assert "--config configs/m1_full_eval.yaml" in recipe
    assert "--decoder-config configs/phase1_baseline.yaml" in recipe

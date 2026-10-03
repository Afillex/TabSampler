"""Smoke tests for the DadaGP scripts' main paths, on synthetic sequences: no dataset.

The scripts read DadaGP through split files frozen by hash, so they cannot be pointed at a
synthetic archive; instead the loading function is replaced and everything after it runs.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from tabsampler.fingering.fit import HumanSequence
from tabsampler.fingering.states import enumerate_states
from tabsampler.types import NoteEvent, NoteGroup, Tuning

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

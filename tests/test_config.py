"""Tests for config loading.

The loaders exist to turn a typo into an error instead of a silently-default value, so
what matters here is that an unknown key is refused and that every field a config is
supposed to reach is actually reachable. The second is the one that catches a key missing
from a ``_require_known_keys`` tuple -- a mistake that rejects every user config while CI,
whose committed configs do not set the key, stays green.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml

from tabsampler.config import Phase1Config, load_eval_config, load_phase1_config
from tabsampler.eval.playability import PlayabilityRules
from tabsampler.types import CostWeights, Tuning

BASELINE = Path("configs/phase1_baseline.yaml")
M1_EVAL = Path("configs/m1_full_eval.yaml")

#: Field values that are legal for every field of the type, used only to prove the key
#: is accepted by the loader. Not defaults and not meaningful settings.
PROBE: dict[str, object] = {
    "open_pitches": [40, 45, 50, 55, 59, 64],
    "n_frets": 22,
    "capo": 0,
    "move": 1.0,
    "span": 1.0,
    "high": 0.1,
    "open_reward": 0.25,
    "acoustic": 0.0,
    "temperature": 1.0,
    "max_span_low": 4,
    "max_span_high": 5,
    "high_neck_fret": 12,
    "max_fingers": 4,
    "allow_barre": True,
    "max_frets_per_second": 48.0,
}


def write(tmp_path: Path, extra: dict[str, object]) -> Path:
    raw: dict[str, object] = {"hypothesis": "a test config, not an experiment", **extra}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


# ------------------------------------------------------------------ committed configs


def test_the_committed_baseline_config_loads() -> None:
    cfg = load_phase1_config(BASELINE)
    assert isinstance(cfg, Phase1Config)
    assert cfg.max_span == 5  # the value the decoder actually runs with
    assert cfg.tuning.open_pitches == (40, 45, 50, 55, 59, 64)


def test_the_committed_eval_config_loads() -> None:
    cfg = load_eval_config(M1_EVAL)
    assert cfg.dataset.channel == "audio_mic"  # ADR 0005
    assert cfg.hypothesis.strip()


# ------------------------------------------------------------------ unknown keys


def test_an_unknown_root_key_is_refused(tmp_path: Path) -> None:
    path = write(tmp_path, {"max_spn": 4})
    with pytest.raises(ValueError, match="unknown key"):
        load_phase1_config(path)


def test_the_renamed_finger_rule_key_is_refused_by_name(tmp_path: Path) -> None:
    # ADR 0019 renamed max_fretted_notes to max_fingers. A config still naming the old key
    # must fail loudly: the rule it set no longer exists, so keeping the default silently
    # would make the run unreproducible.
    path = write(tmp_path, {"rules": {"max_fretted_notes": 4}})
    with pytest.raises(ValueError, match="max_fretted_notes"):
        load_phase1_config(path)


def test_an_unknown_key_is_refused_in_every_nested_section(tmp_path: Path) -> None:
    for section in ("tuning", "grouping", "weights", "rules"):
        path = write(tmp_path, {section: {"definitely_not_a_key": 1}})
        with pytest.raises(ValueError, match="unknown key"):
            load_phase1_config(path)


def test_an_unknown_key_is_refused_in_the_eval_config(tmp_path: Path) -> None:
    path = write(tmp_path, {"onset_tolerence": 0.05})
    with pytest.raises(ValueError, match="unknown key"):
        load_eval_config(path)


# ------------------------------------------------------------------ reachability


@pytest.mark.parametrize("field", [f.name for f in dataclasses.fields(PlayabilityRules)])
def test_every_playability_rule_is_reachable_from_config(tmp_path: Path, field: str) -> None:
    # A field missing from _require_known_keys("rules", ...) would reject every config
    # that sets it, and no committed config sets any of them.
    path = write(tmp_path, {"rules": {field: PROBE[field]}})
    assert load_phase1_config(path).rules == PlayabilityRules()


@pytest.mark.parametrize("field", [f.name for f in dataclasses.fields(CostWeights)])
def test_every_cost_weight_is_reachable_from_config(tmp_path: Path, field: str) -> None:
    path = write(tmp_path, {"weights": {field: PROBE[field]}})
    assert load_phase1_config(path).weights == CostWeights()


@pytest.mark.parametrize("field", [f.name for f in dataclasses.fields(Tuning)])
def test_every_tuning_field_is_reachable_from_config(tmp_path: Path, field: str) -> None:
    path = write(tmp_path, {"tuning": {field: PROBE[field]}})
    assert load_phase1_config(path).tuning == Tuning()


# ------------------------------------------------------------------ values arrive


def test_the_finger_rule_and_barre_switch_arrive_from_config(tmp_path: Path) -> None:
    path = write(tmp_path, {"rules": {"max_fingers": 3, "allow_barre": False}})
    rules = load_phase1_config(path).rules
    assert rules.max_fingers == 3
    assert rules.allow_barre is False


def test_a_config_without_a_hypothesis_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"seed": 0}))
    with pytest.raises(ValueError, match="hypothesis"):
        load_eval_config(path)


def test_the_hex_pickup_channels_are_refused_as_transcriber_input(tmp_path: Path) -> None:
    # ADR 0005: one channel per string IS the E2 ground truth, so it would leak the label.
    path = write(tmp_path, {"dataset": {"channel": "audio_hex"}})
    with pytest.raises(ValueError, match=r"hex-pickup|not allowed"):
        load_eval_config(path)

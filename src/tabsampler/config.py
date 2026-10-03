"""Experiment configuration (spec 4: config file + git commit + seed).

One YAML file per experiment, loaded into frozen dataclasses so a typo becomes an
error rather than a silently-default value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from tabsampler.eval.metrics import DEFAULT_ONSET_TOLERANCE
from tabsampler.eval.playability import PlayabilityRules
from tabsampler.fingering.candidates import DEFAULT_GROUP_WINDOW_S
from tabsampler.transcribe.basic_pitch_cli import DEFAULT_CACHE_DIR, BasicPitchParams
from tabsampler.types import Context, CostWeights, Tuning

#: Channels that may be fed to a transcriber under evaluation (ADR 0005).
#: The hex-pickup channels are excluded deliberately: one channel per string IS the
#: E2 ground truth, so using them as input would leak the label.
ALLOWED_CHANNELS = ("audio_mic", "audio_mix")


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    name: str = "guitarset"
    data_home: Path = Path("data/guitarset")
    channel: str = "audio_mic"  # ADR 0005

    def __post_init__(self) -> None:
        if self.channel not in ALLOWED_CHANNELS:
            raise ValueError(
                f"channel {self.channel!r} is not allowed as transcriber input; "
                f"pick one of {ALLOWED_CHANNELS}. The hex-pickup channels carry "
                f"per-string ground truth and would leak the E2 label (ADR 0005)."
            )

    @property
    def path_attribute(self) -> str:
        """The mirdata Track attribute holding this channel's file path."""
        return f"{self.channel}_path"


@dataclass(frozen=True, slots=True)
class TranscriberConfig:
    exe: str = "basic-pitch"
    cache_dir: Path = DEFAULT_CACHE_DIR
    params: BasicPitchParams = field(default_factory=BasicPitchParams)


@dataclass(frozen=True, slots=True)
class EvalConfig:
    hypothesis: str
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    transcriber: TranscriberConfig = field(default_factory=TranscriberConfig)
    onset_tolerance: float = DEFAULT_ONSET_TOLERANCE
    seed: int = 0
    #: Evaluate only the first N tracks. For smoke-testing the pipeline; a real run
    #: leaves it unset, and the results row records it in `notes`.
    limit: int | None = None

    def __post_init__(self) -> None:
        if not self.hypothesis.strip():
            raise ValueError(
                "every config needs a hypothesis, and it is written before the run, not after"
            )


def _require_known_keys(section: str, given: dict[str, Any], known: tuple[str, ...]) -> None:
    unknown = sorted(set(given) - set(known))
    if unknown:
        raise ValueError(
            f"unknown key(s) {unknown} in config section {section!r}; known keys are {list(known)}"
        )


def load_eval_config(path: Path | str) -> EvalConfig:
    """Load an evaluation config from YAML.

    Unknown keys are an error: a misspelled ``onset_threshold`` that silently keeps
    the default would make an experiment unreproducible and its results misleading.
    """
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
    _require_known_keys(
        "<root>", raw, ("hypothesis", "dataset", "transcriber", "onset_tolerance", "seed", "limit")
    )

    dataset_raw: dict[str, Any] = raw.get("dataset") or {}
    _require_known_keys("dataset", dataset_raw, ("name", "data_home", "channel"))
    dataset = DatasetConfig(
        name=dataset_raw.get("name", "guitarset"),
        data_home=Path(dataset_raw.get("data_home", "data/guitarset")),
        channel=dataset_raw.get("channel", "audio_mic"),
    )

    transcriber_raw: dict[str, Any] = raw.get("transcriber") or {}
    _require_known_keys(
        "transcriber",
        transcriber_raw,
        (
            "exe",
            "cache_dir",
            "onset_threshold",
            "frame_threshold",
            "minimum_note_length_ms",
            "model_serialization",
            "version_tag",
        ),
    )
    defaults = BasicPitchParams()
    transcriber = TranscriberConfig(
        exe=transcriber_raw.get("exe", "basic-pitch"),
        cache_dir=Path(transcriber_raw.get("cache_dir", DEFAULT_CACHE_DIR)),
        params=BasicPitchParams(
            onset_threshold=float(transcriber_raw.get("onset_threshold", defaults.onset_threshold)),
            frame_threshold=float(transcriber_raw.get("frame_threshold", defaults.frame_threshold)),
            minimum_note_length_ms=float(
                transcriber_raw.get("minimum_note_length_ms", defaults.minimum_note_length_ms)
            ),
            model_serialization=transcriber_raw.get(
                "model_serialization", defaults.model_serialization
            ),
            version_tag=transcriber_raw.get("version_tag", defaults.version_tag),
        ),
    )

    limit_raw = raw.get("limit")
    return EvalConfig(
        hypothesis=raw.get("hypothesis", ""),
        dataset=dataset,
        transcriber=transcriber,
        onset_tolerance=float(raw.get("onset_tolerance", DEFAULT_ONSET_TOLERANCE)),
        seed=int(raw.get("seed", 0)),
        limit=None if limit_raw is None else int(limit_raw),
    )


@dataclass(frozen=True, slots=True)
class Phase1Config:
    """Decoder settings: tuning, grouping, cost weights, rendering."""

    hypothesis: str
    tuning: Tuning = field(default_factory=Tuning)
    group_window_s: float = DEFAULT_GROUP_WINDOW_S
    max_span: int = 4
    weights: CostWeights = field(default_factory=CostWeights)
    rules: PlayabilityRules = field(default_factory=PlayabilityRules)
    uncertainty_threshold: float = 0.6
    seed: int = 0

    @property
    def context(self) -> Context:
        return Context(tuning=self.tuning, max_span=self.max_span, weights=self.weights)


def load_phase1_config(path: Path | str) -> Phase1Config:
    """Load the decoder config from YAML. Unknown keys are an error."""
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
    _require_known_keys(
        "<root>",
        raw,
        (
            "hypothesis",
            "tuning",
            "grouping",
            "max_span",
            "weights",
            "rules",
            "uncertainty_threshold",
            "seed",
        ),
    )

    tuning_raw: dict[str, Any] = raw.get("tuning") or {}
    _require_known_keys("tuning", tuning_raw, ("open_pitches", "n_frets", "capo"))
    tuning = Tuning(
        open_pitches=tuple(
            int(p) for p in tuning_raw.get("open_pitches", (40, 45, 50, 55, 59, 64))
        ),
        n_frets=int(tuning_raw.get("n_frets", 22)),
        capo=int(tuning_raw.get("capo", 0)),
    )

    grouping_raw: dict[str, Any] = raw.get("grouping") or {}
    _require_known_keys("grouping", grouping_raw, ("window_s",))

    weights_raw: dict[str, Any] = raw.get("weights") or {}
    _require_known_keys(
        "weights",
        weights_raw,
        (
            "move",
            "span",
            "high",
            "open_reward",
            "acoustic",
            "temperature",
            "string_bias",
            "low_region",
            "high_region",
        ),
    )
    defaults = CostWeights()
    weights = CostWeights(
        move=float(weights_raw.get("move", defaults.move)),
        span=float(weights_raw.get("span", defaults.span)),
        high=float(weights_raw.get("high", defaults.high)),
        open_reward=float(weights_raw.get("open_reward", defaults.open_reward)),
        acoustic=float(weights_raw.get("acoustic", defaults.acoustic)),
        temperature=float(weights_raw.get("temperature", defaults.temperature)),
        string_bias=tuple(float(b) for b in weights_raw.get("string_bias", defaults.string_bias)),
        low_region=float(weights_raw.get("low_region", defaults.low_region)),
        high_region=float(weights_raw.get("high_region", defaults.high_region)),
    )

    rules_raw: dict[str, Any] = raw.get("rules") or {}
    _require_known_keys(
        "rules",
        rules_raw,
        (
            "max_span_low",
            "max_span_high",
            "high_neck_fret",
            # ADR 0019 renamed max_fretted_notes to max_fingers. A config still naming
            # the old key now fails as an unknown key, which is the loader working: the
            # rule it set no longer exists.
            "max_fingers",
            "allow_barre",
            "max_frets_per_second",
        ),
    )
    rule_defaults = PlayabilityRules()
    rules = PlayabilityRules(
        max_span_low=int(rules_raw.get("max_span_low", rule_defaults.max_span_low)),
        max_span_high=int(rules_raw.get("max_span_high", rule_defaults.max_span_high)),
        high_neck_fret=int(rules_raw.get("high_neck_fret", rule_defaults.high_neck_fret)),
        max_fingers=int(rules_raw.get("max_fingers", rule_defaults.max_fingers)),
        allow_barre=bool(rules_raw.get("allow_barre", rule_defaults.allow_barre)),
        max_frets_per_second=float(
            rules_raw.get("max_frets_per_second", rule_defaults.max_frets_per_second)
        ),
    )

    return Phase1Config(
        hypothesis=raw.get("hypothesis", ""),
        tuning=tuning,
        group_window_s=float(grouping_raw.get("window_s", DEFAULT_GROUP_WINDOW_S)),
        max_span=int(raw.get("max_span", 4)),
        weights=weights,
        rules=rules,
        uncertainty_threshold=float(raw.get("uncertainty_threshold", 0.6)),
        seed=int(raw.get("seed", 0)),
    )

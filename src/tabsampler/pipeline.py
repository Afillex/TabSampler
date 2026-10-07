"""audio -> tab, the one function both front ends call (ADR 0058).

The CLI's ``transcribe`` and the web server's ``/api/transcribe`` both go through
:func:`transcribe_path`, so the page and the command line agree note for note by
construction. It reads the file twice: through the transcriber it is given, and once more to
estimate the recording's offset from A440, which is reported and never corrected (ADR 0060).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from tabsampler.audio.io import load_audio
from tabsampler.audio.tuning import estimate_offset
from tabsampler.config import Phase1Config
from tabsampler.decode.robust import Degradation, decode_best_effort
from tabsampler.eval.metrics import pitch_validity_rate, tab_notes_to_placed
from tabsampler.eval.playability import playability_rate
from tabsampler.fingering.candidates import group_notes
from tabsampler.fingering.costs import HandSetScorer
from tabsampler.types import NoteEvent, TabNote


class FileTranscriber(Protocol):
    """A transcriber that reads its audio from a file, as Basic Pitch's CLI does."""

    def transcribe_file(self, path: Path, /) -> list[NoteEvent]: ...


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """A decoded tab and what the user must be told about it."""

    tab: tuple[TabNote, ...]
    #: What had to be relaxed or dropped. A dropped note lowers recall; report it.
    degradation: Degradation
    #: E3: shares of chord shapes and hand moves that are playable.
    group_rate: float
    transition_rate: float
    #: E4. Anything below 1.0 is a bug.
    pitch_validity: float
    n_notes_detected: int
    #: Semitones from A440, in [-0.5, 0.5); None when the audio could not be read for it.
    #: Reported, never corrected (ADR 0060).
    tuning_offset: float | None = None


def offset_from_file(path: Path) -> float | None:
    """The recording's offset from A440, or None if the file cannot be read as audio."""
    try:
        audio, sr = load_audio(path)
    except Exception:  # an unreadable file must not stop the transcription
        return None
    return estimate_offset(audio, sr)


Hearing = Callable[[Path, list[NoteEvent]], Mapping[NoteEvent, tuple[float, ...]]]


def classifier_hearing(cfg: Phase1Config) -> Hearing:
    """The string classifier ``cfg.evidence`` names, listening to a file (needs PyTorch)."""
    from tabsampler.audio.windows import RATE
    from tabsampler.model.evidence import load_classifier, string_evidence

    assert cfg.evidence is not None
    model = load_classifier(cfg.evidence.run)
    temperature = cfg.evidence.temperature

    def hear(path: Path, notes: list[NoteEvent]) -> Mapping[NoteEvent, tuple[float, ...]]:
        signal, _ = load_audio(path, sr=RATE)
        return string_evidence(model, signal, notes, cfg.tuning, temperature)

    return hear


def electric_option(config: Path) -> tuple[tuple[Phase1Config, Hearing] | None, str]:
    """The electric-guitar decoder at ``config`` with its classifier loaded (ADR 0063), or None
    and why not: no evidence block, no trained weights, or no PyTorch."""
    from tabsampler.config import load_phase1_config

    cfg = load_phase1_config(config)
    if cfg.evidence is None:
        return None, f"{config} names no evidence"
    weights = cfg.evidence.run / "best.pt"
    if not weights.is_file():
        return None, f"the trained classifier is missing ({weights})"
    try:
        hear = classifier_hearing(cfg)
    except ImportError as exc:
        return None, f"PyTorch is missing ({exc.name}): run `uv sync --all-groups`"
    return (cfg, hear), "ready"


def separation_option() -> tuple[Callable[[Path], Path] | None, str]:
    """A song's path to its htdemucs_6s guitar stem's (Phase 6, ADR 0065), or None and why not."""
    import importlib.util

    if importlib.util.find_spec("demucs") is None:
        return None, "Demucs is missing: run `uv sync --all-groups` (the `separate` group)"
    from tabsampler.audio.separate import GuitarSeparator

    return GuitarSeparator().stem, "ready"


def transcribe_path(
    path: Path,
    cfg: Phase1Config,
    transcriber: FileTranscriber,
    estimate: Callable[[Path], float | None] = offset_from_file,
    hear: Hearing | None = None,
) -> PipelineResult:
    """Transcribe ``path`` and decode it with the decoder ``cfg`` describes -- with what the
    string classifier hears when ``cfg.evidence`` names one (ADR 0062)."""
    # The tuning check takes about as long as Basic Pitch on a long file, which runs in its own
    # process, so the two overlap: 9.05 s -> 5.17 s on a 3-minute file, same output.
    with ThreadPoolExecutor(max_workers=1) as pool:
        offset = pool.submit(estimate, path)
        notes = transcriber.transcribe_file(path)
        tuning_offset = offset.result()
    evidence: Mapping[NoteEvent, tuple[float, ...]] = {}
    if cfg.evidence is not None and notes:
        evidence = (hear or classifier_hearing(cfg))(path, notes)
    groups = group_notes(notes, window_s=cfg.group_window_s)
    scorer = HandSetScorer(weights=cfg.weights, evidence=evidence)
    tab, degradation = decode_best_effort(groups, scorer, cfg.context)
    play = playability_rate(tab, cfg.rules, window_s=cfg.group_window_s)
    return PipelineResult(
        tab=tuple(tab),
        degradation=degradation,
        group_rate=play.group_rate,
        transition_rate=play.transition_rate,
        pitch_validity=pitch_validity_rate(tab_notes_to_placed(tab), cfg.tuning),
        n_notes_detected=len(notes),
        tuning_offset=tuning_offset,
    )

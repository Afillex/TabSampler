"""audio -> tab, the one function both front ends call (ADR 0058).

The CLI's ``transcribe`` and the web server's ``/api/transcribe`` both go through
:func:`transcribe_path`, so the page and the command line agree note for note by
construction. It does I/O only through the transcriber it is given.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

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


def transcribe_path(path: Path, cfg: Phase1Config, transcriber: FileTranscriber) -> PipelineResult:
    """Transcribe ``path`` and decode it with the decoder ``cfg`` describes."""
    notes = transcriber.transcribe_file(path)
    groups = group_notes(notes, window_s=cfg.group_window_s)
    tab, degradation = decode_best_effort(groups, HandSetScorer(weights=cfg.weights), cfg.context)
    play = playability_rate(tab, cfg.rules, window_s=cfg.group_window_s)
    return PipelineResult(
        tab=tuple(tab),
        degradation=degradation,
        group_rate=play.group_rate,
        transition_rate=play.transition_rate,
        pitch_validity=pitch_validity_rate(tab_notes_to_placed(tab), cfg.tuning),
        n_notes_detected=len(notes),
    )

"""String evidence from audio, for the pipeline (electric plan, Task 3)."""

from __future__ import annotations

import math

import numpy as np
import torch

from tabsampler.audio.windows import RATE
from tabsampler.model.evidence import string_evidence
from tabsampler.model.strings import StringClassifier
from tabsampler.types import NoteEvent, Tuning


def test_every_note_a_string_can_sound_gets_six_log_probabilities() -> None:
    torch.manual_seed(0)
    model = StringClassifier().eval()
    notes = [
        NoteEvent(onset=0.2, offset=0.6, pitch=57, confidence=0.9),
        NoteEvent(onset=0.8, offset=1.2, pitch=30, confidence=0.9),  # below the low E
    ]
    signal = np.zeros(RATE * 2, dtype=np.float32)
    heard = string_evidence(model, signal, notes, Tuning(), temperature=2.0)
    assert set(heard) == {notes[0]}
    row = heard[notes[0]]
    assert len(row) == 6
    # A57 can be played on the low E (fret 17), A, D and G strings, not on B or e: those carry
    # the classifier's mark for impossible, about -1e4 (model/strings.py), at any temperature.
    assert row[4] < -1000 and row[5] < -1000
    assert abs(sum(math.exp(v) for v in row) - 1.0) < 1e-4


def test_no_notes_hear_nothing() -> None:
    model = StringClassifier().eval()
    assert string_evidence(model, np.zeros(RATE, dtype=np.float32), [], Tuning(), 1.0) == {}

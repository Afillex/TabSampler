"""Phase 3's ablation: the decoder with and without the audio evidence (ADRs 0046, 0047).

    uv run python scripts/evaluate_acoustic.py --run cache/acoustic/dev --weight 1.0 \\
        --out cache/validation/p3

Oracle mode on GuitarSet's player 00: every track decoded from its reference notes twice by the
default decoder -- (a) with the acoustic weight at zero, which is Phase 2's decoder exactly, and
(c) with ``--weight`` and, for every note, the string classifier's log-probabilities heard
through ``audio_mic``. Per-track counts for ``scripts/compare_validation.py``. Reads player 00
only (ADR 0037).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch

from tabsampler.audio.windows import RATE, note_window, possible_strings, track_cqt
from tabsampler.config import load_phase1_config
from tabsampler.data.guitarset import load_dataset, reference_notes, reference_tab
from tabsampler.data.splits import guitarset_validation_ids
from tabsampler.decode.robust import decode_best_effort
from tabsampler.eval.metrics import exact_tab_f1, tab_notes_to_placed
from tabsampler.eval.playability import playability_rate
from tabsampler.eval.recovery import RecoveryReport, add_track
from tabsampler.fingering.candidates import group_notes
from tabsampler.fingering.costs import HandSetScorer
from tabsampler.model.strings import StringClassifier, tempered
from tabsampler.types import NoteEvent

ONSET_TOLERANCE = 0.05  # E2's


def load_audio(path: str) -> np.ndarray:
    signal, _ = librosa.load(path, sr=RATE, mono=True)
    return np.asarray(signal, dtype=np.float32)


def heard(
    model: StringClassifier,
    signal: np.ndarray,
    notes: list[NoteEvent],
    open_pitches: Any,
    max_fret: int,
    temperature: float = 1.0,
) -> dict[NoteEvent, tuple[float, ...]]:
    """Each note's six string log-probabilities from the audio around its onset."""
    cqt = track_cqt(signal)
    out: dict[NoteEvent, tuple[float, ...]] = {}
    for note in notes:
        possible = possible_strings(note.pitch, open_pitches, max_fret)
        if not any(possible):
            continue  # the guitar cannot sound it; the decoder drops it anyway
        with torch.no_grad():
            log_probs = model(
                torch.from_numpy(note_window(cqt, note.onset, note.pitch)).unsqueeze(0),
                torch.tensor([note.pitch]),
                torch.tensor([possible]),
            )
        out[note] = tuple(float(v) for v in tempered(log_probs, temperature)[0])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="The classifier's run (best.pt).")
    parser.add_argument("--weight", type=float, required=True, help="The acoustic weight for (c).")
    parser.add_argument("--temperature", type=float, default=1.0, help="The classifier's (Task 6).")
    parser.add_argument("--decoder-config", type=Path, default=Path("configs/decoder_clean.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    dec = load_phase1_config(args.decoder_config)
    model = StringClassifier()
    model.load_state_dict(torch.load(args.run / "best.pt", weights_only=True))
    model.eval()
    dataset: Any = load_dataset(Path("data/guitarset"))
    variants = {
        "a": replace(dec.weights, acoustic=0.0),
        "c": replace(dec.weights, acoustic=args.weight),
    }
    reports = {name: RecoveryReport() for name in variants}
    for track_id in guitarset_validation_ids():
        track = dataset.track(track_id)
        reference = reference_tab(track, dec.tuning)
        notes = reference_notes(track)
        evidence = heard(
            model,
            load_audio(track.audio_mic_path),
            notes,
            dec.tuning.open_pitches,
            dec.tuning.max_fret,
            args.temperature,
        )
        groups = group_notes(notes, window_s=dec.group_window_s)
        for name, weights in variants.items():
            scorer = HandSetScorer(weights=weights, evidence=evidence if name == "c" else {})
            tab, _ = decode_best_effort(groups, scorer, dec.context)
            e2 = exact_tab_f1(reference, tab_notes_to_placed(tab), ONSET_TOLERANCE)
            e3 = playability_rate(tab, dec.rules, window_s=dec.group_window_s)
            add_track(reports[name], track_id, "oracle", e2, e3)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, report in reports.items():
        payload = {"split": "validation", "acoustic": variants[name].acoustic, **report.to_dict()}
        (args.out / f"{name}.json").write_text(json.dumps(payload))
        right, total = report.counts("oracle")
        passed, shapes = report.shapes["oracle"]
        print(
            f"({name}) acoustic {variants[name].acoustic:g}: E2 {right / total:.4f}   "
            f"chord shapes {passed}/{shapes}"
        )


if __name__ == "__main__":
    main()

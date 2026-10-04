"""The string classifier on GuitarSet's validation player: per-note string accuracy (ADR 0046).

    uv run python scripts/evaluate_strings.py --run cache/acoustic/dev

For every reference note of player 00 that more than one string can sound: the classifier's most
probable string, heard through ``audio_mic`` at 22,050 Hz with the reference onset (GuitarSet's
onsets are trusted, so no latency is applied). Two floors beside it, on the same notes: chance
among the possible strings, and the default decoder's own choice in oracle mode. Reads player 00
only (ADR 0037); writes nothing but what it prints.
"""

from __future__ import annotations

import argparse
from collections import Counter
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
from tabsampler.fingering.candidates import group_notes
from tabsampler.fingering.costs import HandSetScorer
from tabsampler.model.strings import StringClassifier


def load_audio(path: str) -> np.ndarray:
    signal, _ = librosa.load(path, sr=RATE, mono=True)
    return np.asarray(signal, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="The run whose best.pt to score.")
    parser.add_argument("--decoder-config", type=Path, default=Path("configs/decoder_clean.yaml"))
    args = parser.parse_args()
    dec = load_phase1_config(args.decoder_config)
    model = StringClassifier()
    model.load_state_dict(torch.load(args.run / "best.pt", weights_only=True))
    model.eval()
    dataset: Any = load_dataset(Path("data/guitarset"))
    open_pitches, max_fret = dec.tuning.open_pitches, dec.tuning.max_fret

    notes = heard = decoder = chance = 0.0
    for track_id in guitarset_validation_ids():
        track = dataset.track(track_id)
        reference = reference_tab(track, dec.tuning)
        tab, _ = decode_best_effort(
            group_notes(reference_notes(track), window_s=dec.group_window_s),
            HandSetScorer(weights=dec.weights),
            dec.context,
        )
        decoded = Counter((round(t.note.onset, 4), t.note.pitch, t.position.string) for t in tab)
        cqt = track_cqt(load_audio(track.audio_mic_path))
        for note, position in reference:
            possible = possible_strings(note.pitch, open_pitches, max_fret)
            if sum(possible) < 2:
                continue
            with torch.no_grad():
                log_probs = model(
                    torch.from_numpy(note_window(cqt, note.onset, note.pitch)).unsqueeze(0),
                    torch.tensor([note.pitch]),
                    torch.tensor([possible]),
                )
            key = (round(note.onset, 4), note.pitch, position.string)
            notes += 1
            heard += int(log_probs.argmax(dim=1)) == position.string
            decoder += decoded[key] > 0
            decoded[key] -= decoded[key] > 0
            chance += 1.0 / sum(possible)
    print(
        f"player 00: {int(notes)} notes with more than one possible string\n"
        f"  the classifier, from audio_mic  {heard / notes:.4f}\n"
        f"  the default decoder, oracle     {decoder / notes:.4f}\n"
        f"  chance among possible strings   {chance / notes:.4f}"
    )


if __name__ == "__main__":
    main()

"""Calibrate the audio evidence on SynthTab's held-out tracks (Phase 3, Task 6).

    caffeinate -i uv run python -u scripts/calibrate_acoustic.py data/synthtab/SynthTab_Dev \\
        --run cache/acoustic/dev

The rule was fixed before this ran (plan, Task 6). Step 1: the classifier's temperature, the one
minimising the held-out notes' NLL (the run's cached held-out examples). Step 2: the acoustic
weight among ``WEIGHTS``: the default decoder decodes each held-out track in its own tuning
(24 frets) from its labelled notes, with the tempered evidence -- windows cut with the rendering
latency corrected, as in training -- and the weight placing the most notes on their labelled
string is kept, the smaller on a tie. Nothing here reads GuitarSet. Writes
``<run>/calibration.json``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from tabsampler.audio.windows import note_window, possible_strings, track_cqt
from tabsampler.config import load_phase1_config
from tabsampler.data.synthtab import RENDER_LATENCY, SynthTabTrack, is_held_out, load_tracks
from tabsampler.decode.robust import decode_best_effort
from tabsampler.fingering.candidates import group_notes
from tabsampler.fingering.costs import HandSetScorer
from tabsampler.model.strings import StringClassifier, fit_temperature, tempered
from tabsampler.types import Context, CostWeights, NoteEvent, Tuning

WEIGHTS = (0.0, 0.1, 0.25, 0.5, 1.0)
MAX_FRET = 24


def choose_weight(recovered: dict[float, int]) -> float:
    """The weight that recovered the most notes, the smaller on a tie."""
    best = max(recovered.values())
    return min(weight for weight, count in recovered.items() if count == best)


def evidence_for(
    model: StringClassifier, track: SynthTabTrack, temperature: float
) -> dict[NoteEvent, tuple[float, ...]]:
    signal, _ = sf.read(track.audio, dtype="float32")
    cqt = track_cqt(signal)
    notes, windows, pitches, possible = [], [], [], []
    for note, _ in track.notes:
        mask = possible_strings(note.pitch, track.open_pitches, MAX_FRET)
        if not any(mask):
            continue
        notes.append(note)
        windows.append(note_window(cqt, note.onset + RENDER_LATENCY[track.family], note.pitch))
        pitches.append(note.pitch)
        possible.append(mask)
    if not notes:
        return {}
    with torch.no_grad():
        log_probs = tempered(
            model(
                torch.from_numpy(np.stack(windows)), torch.tensor(pitches), torch.tensor(possible)
            ),
            temperature,
        )
    return {note: tuple(float(v) for v in row) for note, row in zip(notes, log_probs, strict=True)}


def recovered(
    track: SynthTabTrack, weights: CostWeights, evidence: dict[NoteEvent, tuple[float, ...]]
) -> int:
    """How many of the track's notes the decoder places on their labelled string."""
    ctx = Context(tuning=Tuning(open_pitches=track.open_pitches, n_frets=MAX_FRET), max_span=5)
    groups = group_notes([note for note, _ in track.notes], window_s=0.03)
    tab, _ = decode_best_effort(groups, HandSetScorer(weights=weights, evidence=evidence), ctx)
    labelled = Counter((round(n.onset, 4), n.pitch, p.string) for n, p in track.notes)
    decoded = Counter((round(t.note.onset, 4), t.note.pitch, t.position.string) for t in tab)
    return sum((labelled & decoded).values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--decoder-config", type=Path, default=Path("configs/decoder_clean.yaml"))
    args = parser.parse_args()
    model = StringClassifier()
    model.load_state_dict(torch.load(args.run / "best.pt", weights_only=True))
    model.eval()

    saved = np.load(args.run / "examples.npz")
    with torch.no_grad():
        log_probs = model(
            torch.from_numpy(saved["held_windows"].astype(np.float32)),
            torch.from_numpy(saved["held_pitches"].astype(np.int64)),
            torch.from_numpy(saved["held_possible"]),
        )
    labels = torch.from_numpy(saved["held_strings"].astype(np.int64))
    before = float(-log_probs.gather(1, labels.unsqueeze(1)).mean())
    temperature = fit_temperature(log_probs, labels)
    after = float(-tempered(log_probs, temperature).gather(1, labels.unsqueeze(1)).mean())
    print(f"temperature {temperature:.4f}: held-out NLL {before:.4f} -> {after:.4f}", flush=True)

    tracks = [t for t in load_tracks(args.root)[0] if is_held_out(t.name)]
    default = load_phase1_config(args.decoder_config).weights
    counts = dict.fromkeys(WEIGHTS, 0)
    total = 0
    for track in tracks:
        evidence = evidence_for(model, track, temperature)
        total += len(track.notes)
        for weight in WEIGHTS:
            counts[weight] += recovered(track, replace(default, acoustic=weight), evidence)
    for weight in WEIGHTS:
        print(
            f"  weight {weight:<5g} recovers {counts[weight]} of {total}"
            f" ({counts[weight] / total:.4f})"
        )
    weight = choose_weight(counts)
    print(f"chosen weight {weight:g} on {len(tracks)} held-out tracks")
    (args.run / "calibration.json").write_text(
        json.dumps(
            {
                "temperature": temperature,
                "weight": weight,
                "nll_before": before,
                "nll_after": after,
                "recovered": {str(w): c for w, c in counts.items()},
                "notes": total,
                "tracks": len(tracks),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

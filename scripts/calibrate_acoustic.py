"""Calibrate the audio evidence on held-out audio (Phase 3, Task 6; Phase 4, Task 6).

    caffeinate -i uv run python -u scripts/calibrate_acoustic.py data/synthtab/SynthTab_Dev \\
        --run cache/acoustic/dev
    caffeinate -i uv run python -u scripts/calibrate_acoustic.py data/guitar-techs \\
        --corpus guitartechs --run cache/acoustic/gt-ft

The rule was fixed before this ran (plan, Task 6). Step 1: the classifier's temperature, the one
minimising the held-out notes' NLL (the run's cached held-out examples). Step 2: the acoustic
weight among ``WEIGHTS``: the default decoder decodes each held-out track in its own tuning
(24 frets) from its labelled notes, with the tempered evidence -- windows cut with the rendering
latency corrected, as in training -- and the weight placing the most notes on their labelled
string is kept, the smaller on a tie. ``--corpus guitartechs`` does the same on Guitar-TECHS's
player 3 (ADR 0051): its cleaned notes, standard tuning, each take's label delay measured from
its audio (ADR 0052). ``--corpus idmt`` calibrates on IDMT-SMT-Guitar's licks instead, both the
temperature (by their notes' NLL) and the weight, and ``--judge`` then scores that calibration on
EGSet12 (ADR 0062). Nothing here reads GuitarSet. Writes ``<run>/calibration.json``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import librosa
import numpy as np
import soundfile as sf
import torch

from tabsampler.audio.windows import (
    MEASURE_LAG,
    RATE,
    note_window,
    onset_lag,
    possible_strings,
    track_cqt,
)
from tabsampler.config import load_phase1_config
from tabsampler.data.electric import aligned, load_egset12, load_idmt_licks
from tabsampler.data.guitartechs import STANDARD, clean, load_takes, usable
from tabsampler.data.synthtab import RENDER_LATENCY, is_held_out, load_tracks
from tabsampler.decode.robust import decode_best_effort
from tabsampler.fingering.candidates import group_notes
from tabsampler.fingering.costs import HandSetScorer
from tabsampler.model.strings import StringClassifier, fit_temperature, tempered
from tabsampler.types import Context, CostWeights, NoteEvent, Position, Tuning

WEIGHTS = (0.0, 0.1, 0.25, 0.5, 1.0)
MAX_FRET = 24
HELD_OUT_PLAYER = 3  # Guitar-TECHS's validation player (ADR 0051)


def choose_weight(recovered: dict[float, int]) -> float:
    """The weight that recovered the most notes, the smaller on a tie."""
    best = max(recovered.values())
    return min(weight for weight, count in recovered.items() if count == best)


class Piece(NamedTuple):
    """One held-out track: its labelled notes, tuning, audio, and where its windows are cut
    relative to the labelled onsets (seconds, added)."""

    notes: tuple[tuple[NoteEvent, Position], ...]
    open_pitches: tuple[int, ...]
    signal: np.ndarray
    shift: float


def synthtab_pieces(root: Path) -> Iterator[Piece]:
    for track in load_tracks(root)[0]:
        if is_held_out(track.name):
            signal, _ = sf.read(track.audio, dtype="float32")
            yield Piece(track.notes, track.open_pitches, signal, RENDER_LATENCY[track.family])


def guitartechs_pieces(root: Path) -> Iterator[Piece]:
    for take in load_takes(root)[0]:
        if take.player != HELD_OUT_PLAYER or not usable(take):
            continue
        notes, _ = clean(take.notes)
        signal, _ = librosa.load(take.direct_input, sr=RATE, mono=True)
        signal = np.asarray(signal, dtype=np.float32)
        lag = onset_lag(signal, sorted({n.onset for n, _ in notes}))
        yield Piece(notes, STANDARD, signal, lag - MEASURE_LAG)


def electric_pieces(which: str) -> Iterator[Piece]:
    """ADR 0062's validation takes, each with its label delay taken off (``aligned``)."""
    takes = (
        load_idmt_licks(Path("data/idmt-smt-guitar"))
        if which == "idmt"
        else load_egset12(Path("data/egset12"))
    )
    for take in takes:
        take = aligned(take)
        signal, _ = librosa.load(take.audio, sr=RATE, mono=True)
        yield Piece(take.notes, STANDARD, np.asarray(signal, dtype=np.float32), 0.0)


def piece_log_probs(
    model: StringClassifier, pieces: list[Piece]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Untempered log-probabilities and labels for every note more than one string can sound."""
    windows, pitches, possible, labels = [], [], [], []
    for piece in pieces:
        cqt = track_cqt(piece.signal)
        for note, position in piece.notes:
            mask = possible_strings(note.pitch, piece.open_pitches, MAX_FRET)
            if sum(mask) < 2:
                continue
            windows.append(note_window(cqt, note.onset + piece.shift, note.pitch))
            pitches.append(note.pitch)
            possible.append(mask)
            labels.append(position.string)
    with torch.no_grad():
        log_probs = model(
            torch.from_numpy(np.stack(windows)), torch.tensor(pitches), torch.tensor(possible)
        )
    return log_probs, torch.tensor(labels)


def judge(model: StringClassifier, temperature: float, weight: float, default: CostWeights) -> None:
    """EGSet12's string recovery (oracle E2) without and with the evidence, per take, with a
    bootstrap interval over takes for the change."""
    rows: list[tuple[int, int, int]] = []
    for piece in electric_pieces("egset12"):
        evidence = evidence_for(model, piece, temperature)
        without = recovered(piece.notes, piece.open_pitches, replace(default, acoustic=0.0), {})
        with_ = recovered(
            piece.notes, piece.open_pitches, replace(default, acoustic=weight), evidence
        )
        rows.append((len(piece.notes), without, with_))
    n = np.array(rows)
    base, new = n[:, 1].sum() / n[:, 0].sum(), n[:, 2].sum() / n[:, 0].sum()
    rng = np.random.default_rng(0)
    deltas = []
    for _ in range(10000):
        pick = n[rng.integers(0, len(n), len(n))]
        deltas.append((pick[:, 2].sum() - pick[:, 1].sum()) / pick[:, 0].sum())
    low, high = np.percentile(deltas, [2.5, 97.5])
    print(
        f"EGSet12, {len(n)} takes, {n[:, 0].sum()} notes: oracle string recovery {base:.4f} -> "
        f"{new:.4f} (weight {weight:g}, T {temperature:.4f}), change {new - base:+.4f} "
        f"[{low:+.4f}, {high:+.4f}]"
    )


def evidence_for(
    model: StringClassifier, piece: Piece, temperature: float
) -> dict[NoteEvent, tuple[float, ...]]:
    cqt = track_cqt(piece.signal)
    notes, windows, pitches, possible = [], [], [], []
    for note, _ in piece.notes:
        mask = possible_strings(note.pitch, piece.open_pitches, MAX_FRET)
        if not any(mask):
            continue
        notes.append(note)
        windows.append(note_window(cqt, note.onset + piece.shift, note.pitch))
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
    notes: Sequence[tuple[NoteEvent, Position]],
    open_pitches: tuple[int, ...],
    weights: CostWeights,
    evidence: dict[NoteEvent, tuple[float, ...]],
) -> int:
    """How many of the notes the decoder places on their labelled string."""
    ctx = Context(tuning=Tuning(open_pitches=open_pitches, n_frets=MAX_FRET), max_span=5)
    groups = group_notes([note for note, _ in notes], window_s=0.03)
    tab, _ = decode_best_effort(groups, HandSetScorer(weights=weights, evidence=evidence), ctx)
    labelled = Counter((round(n.onset, 4), n.pitch, p.string) for n, p in notes)
    decoded = Counter((round(t.note.onset, 4), t.note.pitch, t.position.string) for t in tab)
    return sum((labelled & decoded).values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--corpus", choices=("synthtab", "guitartechs", "idmt"), default="synthtab")
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Score the run's calibration on EGSet12 instead of calibrating (ADR 0062).",
    )
    parser.add_argument("--decoder-config", type=Path, default=Path("configs/decoder_clean.yaml"))
    args = parser.parse_args()
    model = StringClassifier()
    model.load_state_dict(torch.load(args.run / "best.pt", weights_only=True))
    model.eval()

    default = load_phase1_config(args.decoder_config).weights
    if args.judge:
        calibration = json.loads((args.run / "calibration.json").read_text())
        judge(model, calibration["temperature"], calibration["weight"], default)
        return
    if args.corpus == "idmt":
        idmt = list(electric_pieces("idmt"))
        log_probs, labels = piece_log_probs(model, idmt)
    else:
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
    print(f"temperature {temperature:.4f}: NLL {before:.4f} -> {after:.4f}", flush=True)

    pieces = {
        "synthtab": synthtab_pieces,
        "guitartechs": guitartechs_pieces,
        "idmt": lambda _: iter(idmt),
    }[args.corpus]
    counts = dict.fromkeys(WEIGHTS, 0)
    total = tracks = 0
    for piece in pieces(args.root):
        evidence = evidence_for(model, piece, temperature)
        total += len(piece.notes)
        tracks += 1
        for weight in WEIGHTS:
            weights = replace(default, acoustic=weight)
            counts[weight] += recovered(piece.notes, piece.open_pitches, weights, evidence)
    for weight in WEIGHTS:
        print(
            f"  weight {weight:<5g} recovers {counts[weight]} of {total}"
            f" ({counts[weight] / total:.4f})"
        )
    weight = choose_weight(counts)
    print(f"chosen weight {weight:g} on {tracks} held-out tracks")
    (args.run / "calibration.json").write_text(
        json.dumps(
            {
                "temperature": temperature,
                "weight": weight,
                "nll_before": before,
                "nll_after": after,
                "recovered": {str(w): c for w, c in counts.items()},
                "notes": total,
                "tracks": tracks,
                "corpus": args.corpus,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

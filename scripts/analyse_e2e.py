"""Where the end-to-end loss goes: the transcriber's note errors against context (Phase 5, Task 2).

    uv run python -u scripts/analyse_e2e.py --corpus guitartechs
    uv run python -u scripts/analyse_e2e.py --corpus guitarset

Validation data only: Guitar-TECHS's player 3 (ADR 0051), each take's label delay subtracted from
its reference onsets (ADR 0055), or GuitarSet's player 00, which Basic Pitch has most likely heard
in training, so its figures are labelled and choose nothing (ADR 0055). Per piece: Basic Pitch's
notes (the canonical eval config's settings), the default decoder end to end and in oracle mode,
and E1's matching of reference to transcribed notes -- the same pitch, onsets within 50 ms. Every
reference note is then missed or heard; a missed note is split by its length and by what the
transcriber put near it, and a heard one by whether it is strung right end to end and in oracle
mode. The E2 gap splits into **note errors** -- oracle E2 down to the E2 of the transcriber's notes
strung as oracle mode strings them -- and **context**, from there to the end-to-end E2. A
diagnostic: no results row, nothing chosen.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import librosa
import mir_eval
import numpy as np

from tabsampler.audio.windows import MEASURE_LAG, RATE, onset_lag
from tabsampler.config import load_eval_config, load_phase1_config
from tabsampler.data.guitarset import load_dataset, reference_tab
from tabsampler.data.guitartechs import clean, load_takes, usable
from tabsampler.data.splits import guitarset_validation_ids
from tabsampler.decode.robust import decode_best_effort
from tabsampler.eval.metrics import exact_tab_f1, tab_notes_to_placed
from tabsampler.fingering.candidates import group_notes
from tabsampler.fingering.costs import HandSetScorer
from tabsampler.transcribe.basic_pitch_cli import BasicPitchCLITranscriber
from tabsampler.types import NoteEvent, Position, TabNote

TOLERANCE = 0.05  # E1's and E2's onset tolerance
SHORT = 0.1277  # Basic Pitch's minimum note length, 127.70 ms
VALIDATION_PLAYER = 3  # Guitar-TECHS (ADR 0051)

Placed = tuple[NoteEvent, Position]


class Piece(NamedTuple):
    name: str
    reference: list[Placed]
    audio: Path


def pairs(reference: Sequence[NoteEvent], estimated: Sequence[NoteEvent]) -> list[tuple[int, int]]:
    """E1's matching: (reference index, estimated index), the same pitch, onsets within 50 ms."""
    if not reference or not estimated:
        return []

    def arrays(notes: Sequence[NoteEvent]) -> tuple[np.ndarray, np.ndarray]:
        intervals = np.array([[n.onset, max(n.offset, n.onset + 1e-3)] for n in notes])
        return intervals, librosa.midi_to_hz(np.array([n.pitch for n in notes], dtype=float))

    matched: list[tuple[int, int]] = mir_eval.transcription.match_notes(
        *arrays(reference), *arrays(estimated), onset_tolerance=TOLERANCE, offset_ratio=None
    )
    return matched


def near(note: NoteEvent, others: Sequence[NoteEvent]) -> str:
    """What sounds near ``note`` among ``others``: an octave off, a semitone off, other, none."""
    gaps = {abs(o.pitch - note.pitch) for o in others if abs(o.onset - note.onset) <= TOLERANCE}
    if gaps & {12, 24}:
        return "octave"
    if 1 in gaps:
        return "semitone"
    return "other" if gaps else "none"


def strings(tab: Sequence[TabNote]) -> dict[tuple[float, int], int]:
    return {(round(t.note.onset, 4), t.note.pitch): t.position.string for t in tab}


def tally(
    reference: Sequence[Placed],
    estimated: Sequence[NoteEvent],
    oracle: Sequence[TabNote],
    e2e: Sequence[TabNote],
) -> Counter[str]:
    """The counts for one piece."""
    refs = [note for note, _ in reference]
    matched = pairs(refs, estimated)
    heard = dict(matched)
    used = {j for _, j in matched}
    loose = [n for j, n in enumerate(estimated) if j not in used]
    oracle_strings, e2e_strings = strings(oracle), strings(e2e)
    counts: Counter[str] = Counter(ref=len(reference), est=len(estimated), placed=len(e2e))
    for i, (note, position) in enumerate(reference):
        oracle_right = oracle_strings.get((round(note.onset, 4), note.pitch)) == position.string
        counts["oracle right"] += oracle_right
        if i not in heard:
            counts["missed"] += 1
            counts["missed short" if note.offset - note.onset < SHORT else "missed long"] += 1
            counts[f"missed, {near(note, loose)} near"] += 1
            continue
        found = estimated[heard[i]]
        e2e_right = e2e_strings.get((round(found.onset, 4), found.pitch)) == position.string
        counts["heard"] += 1
        counts[
            {
                (True, True): "heard, both right",
                (True, False): "heard, oracle only",
                (False, True): "heard, end to end only",
                (False, False): "heard, both wrong",
            }[(oracle_right, e2e_right)]
        ] += 1
    for note in loose:
        counts["extra"] += 1
        counts[f"extra, {near(note, refs)} near"] += 1
    return counts


def split(counts: Counter[str], oracle_e2: float, e2e_e2: float) -> dict[str, float]:
    """The E2 gap, oracle to end to end, as note errors and context."""
    denominator = counts["ref"] + counts["placed"]
    notes_only = 2 * (counts["heard, both right"] + counts["heard, oracle only"]) / denominator
    return {
        "oracle E2": oracle_e2,
        "E2 of the heard notes strung as in oracle mode": notes_only,
        "end-to-end E2": e2e_e2,
        "note errors": oracle_e2 - notes_only,
        "context": notes_only - e2e_e2,
    }


def guitartechs(root: Path) -> Iterator[Piece]:
    for take in load_takes(root)[0]:
        if take.player != VALIDATION_PLAYER or not usable(take):
            continue
        notes, _ = clean(take.notes)
        signal, _ = librosa.load(take.direct_input, sr=RATE, mono=True)
        delay = MEASURE_LAG - onset_lag(
            np.asarray(signal, dtype=np.float32), sorted({n.onset for n, _ in notes})
        )
        shifted = [
            (NoteEvent(n.onset - delay, n.offset - delay, n.pitch, n.confidence), p)
            for n, p in notes
        ]
        yield Piece(f"P3 {take.name}", shifted, take.direct_input)


def guitarset(tuning: Any) -> Iterator[Piece]:
    dataset: Any = load_dataset(Path("data/guitarset"))
    for track_id in guitarset_validation_ids():
        track = dataset.track(track_id)
        yield Piece(track_id, list(reference_tab(track, tuning)), Path(track.audio_mic_path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=("guitartechs", "guitarset"), required=True)
    parser.add_argument("--root", type=Path, default=Path("data/guitar-techs"))
    parser.add_argument("--config", type=Path, default=Path("configs/m1_full_eval.yaml"))
    parser.add_argument("--decoder-config", type=Path, default=Path("configs/decoder_clean.yaml"))
    args = parser.parse_args()
    cfg = load_eval_config(args.config)
    dec = load_phase1_config(args.decoder_config)
    transcriber = BasicPitchCLITranscriber(
        exe=cfg.transcriber.exe, params=cfg.transcriber.params, cache_dir=cfg.transcriber.cache_dir
    )
    scorer = HandSetScorer(weights=dec.weights)

    def place(notes: list[NoteEvent]) -> list[TabNote]:
        groups = group_notes(notes, window_s=dec.group_window_s)
        return decode_best_effort(groups, scorer, dec.context)[0] if groups else []

    pieces = guitartechs(args.root) if args.corpus == "guitartechs" else guitarset(dec.tuning)
    counts: Counter[str] = Counter()
    hits = {"oracle": [0, 0], "e2e": [0, 0]}
    for piece in pieces:
        estimated = transcriber.transcribe_file(piece.audio)
        oracle = place([note for note, _ in piece.reference])
        e2e = place(estimated)
        counts += tally(piece.reference, estimated, oracle, e2e)
        for mode, tab in (("oracle", oracle), ("e2e", e2e)):
            prf = exact_tab_f1(piece.reference, tab_notes_to_placed(tab), TOLERANCE)
            hits[mode][0] += 2 * prf.n_match
            hits[mode][1] += prf.n_ref + prf.n_est

    label = (
        "Guitar-TECHS player 3"
        if args.corpus == "guitartechs"
        else ("GuitarSet player 00 -- Basic Pitch most likely trained on it (ADR 0055)")
    )
    print(f"{label}: {counts['ref']} reference notes, {counts['est']} transcribed")
    for key in sorted(counts):
        if key not in ("ref", "est"):
            print(f"  {key:32s} {counts[key]:6d}")
    oracle_e2, e2e_e2 = (hits[m][0] / hits[m][1] for m in ("oracle", "e2e"))
    for name, value in split(counts, oracle_e2, e2e_e2).items():
        print(f"  {name:48s} {value:.4f}")


if __name__ == "__main__":
    main()

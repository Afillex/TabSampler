"""A pre-registered look at EGDB, the second test set: clean electric guitar (ADRs 0049, 0050).

    uv run python -u scripts/evaluate_egdb.py --run cache/acoustic/gt-ft --weight 0.25 \\
        --temperature 0.6957 --config configs/p4_test_eval.yaml
    uv run python -u scripts/evaluate_egdb.py --config configs/p5_test_eval.yaml

Every one of EGDB's 240 clips, direct input, against its own labels: (a) the default decoder in
oracle mode, Phase 2's exactly; (c) the same with the string classifier's evidence at ``--weight``
and ``--temperature``, heard from the direct input at the labelled onsets -- EGDB's annotations
were aligned to the audio by onset detection, so, like GuitarSet's, they are taken as they are; and
the default decoder end to end, on Basic Pitch's notes from the same audio -- with the config's
transcriber settings -- and the transcriber's E1. Without ``--run``, (c) is left out. The look is
logged before anything is read; a partial download is refused, since a subset is never quoted;
nothing is written per clip.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch

from tabsampler.audio.windows import RATE, note_window, possible_strings, track_cqt
from tabsampler.config import load_eval_config, load_phase1_config
from tabsampler.data.egdb import load_clips
from tabsampler.data.splits import egdb_clip_id, egdb_test_ids, record_test_set_access
from tabsampler.decode.robust import decode_best_effort
from tabsampler.eval.metrics import exact_tab_f1, note_f1, tab_notes_to_placed
from tabsampler.eval.playability import playability_rate
from tabsampler.eval.recovery import RecoveryReport, add_track
from tabsampler.fingering.candidates import group_notes
from tabsampler.fingering.costs import HandSetScorer
from tabsampler.model.strings import StringClassifier, tempered
from tabsampler.transcribe.basic_pitch_cli import BasicPitchCLITranscriber
from tabsampler.types import NoteEvent


def heard(
    model: StringClassifier,
    signal: np.ndarray,
    notes: list[NoteEvent],
    open_pitches: Any,
    max_fret: int,
    temperature: float,
) -> dict[NoteEvent, tuple[float, ...]]:
    """Each note's six string log-probabilities from the audio around its onset."""
    cqt = track_cqt(signal)
    out: dict[NoteEvent, tuple[float, ...]] = {}
    for note in notes:
        possible = possible_strings(note.pitch, open_pitches, max_fret)
        if not any(possible):
            continue
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
    parser.add_argument("--run", type=Path, help="The classifier's run (best.pt); none: no (c).")
    parser.add_argument("--weight", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--config", type=Path, required=True, help="Evaluation config YAML.")
    parser.add_argument("--decoder-config", type=Path, default=Path("configs/decoder_clean.yaml"))
    parser.add_argument("--root", type=Path, default=Path("data/egdb"))
    parser.add_argument("--note", default="", help="Added to the access log's reason.")
    args = parser.parse_args()
    cfg = load_eval_config(args.config)
    dec = load_phase1_config(args.decoder_config)

    present = sorted(
        egdb_clip_id(int(label.stem))
        for label in (args.root / "audio_label").glob("*.midi")
        if (args.root / "audio_DI" / f"{label.stem}.wav").exists()
    )
    if present != list(egdb_test_ids()):
        raise SystemExit(
            f"EGDB is incomplete ({len(present)} clips with labels and audio): finish "
            f"scripts/download_egdb.py first -- a subset is never quoted"
        )
    evidence_part = (
        f"oracle with acoustic 0 and {args.weight:g} at temperature {args.temperature:g} "
        f"(classifier {args.run}), and end to end"
        if args.run is not None
        else "oracle and end to end"
    )
    record_test_set_access(
        f"evaluate_egdb.py, {evidence_part}, on {len(present)} EGDB clips: decoder "
        f"{args.decoder_config}, config {args.config}; a pre-registered look"
        + (f"; {args.note}" if args.note else "")
    )

    clips, _ = load_clips(args.root)  # complete: checked above by name
    model: StringClassifier | None = None
    if args.run is not None:
        model = StringClassifier()
        model.load_state_dict(torch.load(args.run / "best.pt", weights_only=True))
        model.eval()
    transcriber = BasicPitchCLITranscriber(
        exe=cfg.transcriber.exe,
        params=cfg.transcriber.params,
        cache_dir=cfg.transcriber.cache_dir,
        model=cfg.transcriber.model_path,
    )
    plain = HandSetScorer(weights=replace(dec.weights, acoustic=0.0))
    report = RecoveryReport()
    e1 = [0, 0, 0]  # matches, estimated, reference
    placed = dict.fromkeys(("oracle", "audio", "e2e"), 0)
    given = dict.fromkeys(("oracle", "audio", "e2e"), 0)
    for clip in clips:
        reference = list(clip.notes)
        notes = [note for note, _ in reference]
        signal = np.asarray(librosa.load(clip.direct_input, sr=RATE, mono=True)[0], np.float32)
        modes = [("oracle", plain, notes)]
        if model is not None:
            evidence = heard(
                model, signal, notes, dec.tuning.open_pitches, dec.tuning.max_fret, args.temperature
            )
            with_audio = HandSetScorer(
                weights=replace(dec.weights, acoustic=args.weight), evidence=evidence
            )
            modes.append(("audio", with_audio, notes))
        estimated = transcriber.transcribe_file(clip.direct_input)
        heard_notes = note_f1(notes, estimated, cfg.onset_tolerance)
        e1 = [e1[0] + heard_notes.n_match, e1[1] + heard_notes.n_est, e1[2] + heard_notes.n_ref]
        for mode, scorer, given_notes in [*modes, ("e2e", plain, estimated)]:
            groups = group_notes(given_notes, window_s=dec.group_window_s)
            tab = decode_best_effort(groups, scorer, dec.context)[0] if groups else []
            placed[mode] += len(tab)
            given[mode] += len(given_notes)
            e2 = exact_tab_f1(reference, tab_notes_to_placed(tab), cfg.onset_tolerance)
            e3 = playability_rate(tab, dec.rules, window_s=dec.group_window_s)
            add_track(report, clip.clip_id, mode, e2, e3)

    params = cfg.transcriber.params
    print(
        f"EGDB, {len(clips)} clips, direct input; Basic Pitch onset {params.onset_threshold:g},"
        f" frame {params.frame_threshold:g}, min {params.minimum_note_length_ms:g} ms"
    )
    print(f"  E1 transcriber (raw): {2 * e1[0] / (e1[1] + e1[2]):.4f}")
    names = {"oracle": "(a) oracle", "audio": f"(c) oracle, acoustic {args.weight:g}", "e2e": "e2e"}
    for mode, name in names.items():
        if mode not in report.shapes:
            continue
        right, total = report.counts(mode)
        passed, shapes = report.shapes[mode]
        print(
            f"  {name}: E2 {right / total:.4f}   chord shapes {passed}/{shapes}"
            f"   placed {placed[mode]} of {given[mode]} notes"
        )


if __name__ == "__main__":
    main()

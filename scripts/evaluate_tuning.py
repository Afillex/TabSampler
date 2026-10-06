"""The tuning correction, scored end to end on Guitar-TECHS's player 3 (app plan, Task 6).

    caffeinate -i uv run python -u scripts/evaluate_tuning.py --out cache/validation/tuning

Each of player 3's 12 takes (direct input) is pitch-shifted by d semitones to simulate a guitar
tuned off A440, for every d in ``--detune``; then Basic Pitch (``CHOSEN_PARAMS``) transcribes it
as it is ("off") and after ``retune(estimate_offset(...))`` ("on"), and the default decoder
strings the notes. Writes ``<out>/d<d>_<off|on>.json`` in the per-track form
``scripts/compare_validation.py`` compares, and prints E2 and each take's estimated offset.
Scored as ``scripts/tune_transcriber.py`` scores: cleaned labels, each take's label delay taken
off (ADR 0055). Validation data only; the hypotheses and the adoption rule are the plan's.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from tabsampler.audio.tuning import estimate_offset, retune
from tabsampler.config import load_eval_config, load_phase1_config
from tabsampler.data.guitartechs import aligned, load_takes, usable
from tabsampler.decode.robust import decode_best_effort
from tabsampler.eval.metrics import exact_tab_f1, note_f1, tab_notes_to_placed
from tabsampler.eval.playability import playability_rate
from tabsampler.eval.recovery import RecoveryReport, add_track
from tabsampler.fingering.candidates import group_notes
from tabsampler.fingering.costs import HandSetScorer
from tabsampler.transcribe.basic_pitch_cli import CHOSEN_PARAMS, BasicPitchCLITranscriber

DETUNE = (0.0, 0.25, 0.45, -0.45)
VALIDATION_PLAYER = 3  # Guitar-TECHS (ADR 0051)
TOLERANCE = 0.05


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("data/guitar-techs"))
    parser.add_argument("--config", type=Path, default=Path("configs/m1_full_eval.yaml"))
    parser.add_argument("--decoder-config", type=Path, default=Path("configs/decoder_clean.yaml"))
    parser.add_argument("--detune", type=float, nargs="+", default=list(DETUNE))
    args = parser.parse_args()
    cfg = load_eval_config(args.config)
    dec = load_phase1_config(args.decoder_config)
    scorer = HandSetScorer(weights=dec.weights)
    transcriber = BasicPitchCLITranscriber(
        exe=cfg.transcriber.exe, params=CHOSEN_PARAMS, cache_dir=cfg.transcriber.cache_dir
    )
    takes = [t for t in load_takes(args.root)[0] if t.player == VALIDATION_PLAYER and usable(t)]
    references = {take.name: aligned(take) for take in takes}
    audio_dir = args.out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(takes)} takes, {sum(len(r) for r in references.values())} reference notes")

    for d in args.detune:
        reports = {"off": RecoveryReport(), "on": RecoveryReport()}
        e1 = {"off": [0, 0], "on": [0, 0]}
        estimates: list[float] = []
        for take in takes:
            audio, sr = sf.read(take.direct_input, dtype="float32", always_2d=True)
            mono = np.ascontiguousarray(audio.mean(axis=1), dtype=np.float32)
            detuned = (
                mono
                if d == 0
                else np.asarray(librosa.effects.pitch_shift(mono, sr=sr, n_steps=d), np.float32)
            )
            offset = estimate_offset(detuned, sr)
            estimates.append(offset)
            corrected = retune(detuned, sr, offset)
            for arm, signal in (("off", detuned), ("on", corrected)):
                path = audio_dir / f"{take.name}_d{d:+g}_{arm}.wav"
                sf.write(path, signal, sr)
                estimated = transcriber.transcribe_file(path)
                reference = list(references[take.name])
                heard = note_f1([n for n, _ in reference], estimated, TOLERANCE)
                e1[arm][0] += 2 * heard.n_match
                e1[arm][1] += heard.n_ref + heard.n_est
                groups = group_notes(estimated, window_s=dec.group_window_s)
                tab = decode_best_effort(groups, scorer, dec.context)[0] if groups else []
                e2 = exact_tab_f1(reference, tab_notes_to_placed(tab), TOLERANCE)
                e3 = playability_rate(tab, dec.rules, window_s=dec.group_window_s)
                add_track(reports[arm], f"P3 {take.name}", "e2e", e2, e3)
        for arm, report in reports.items():
            payload = {
                "split": "validation",
                "detune_semitones": d,
                "correction": arm,
                "estimated_offsets": estimates,
                **report.to_dict(),
            }
            (args.out / f"d{d:+g}_{arm}.json").write_text(json.dumps(payload))
            right, total = report.counts("e2e")
            print(
                f"  d {d:+.2f}  correction {arm:3s}  E1 {e1[arm][0] / e1[arm][1]:.4f}  "
                f"E2 {right / total:.4f}",
                flush=True,
            )
        print(f"  d {d:+.2f}  estimated offsets: {' '.join(f'{o:+.2f}' for o in estimates)}")


if __name__ == "__main__":
    main()

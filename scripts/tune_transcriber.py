"""Basic Pitch's note thresholds, scored end to end on Guitar-TECHS's player 3 (Phase 5, Task 3).

    caffeinate -i uv run python -u scripts/tune_transcriber.py --out cache/validation/p5-grid

For every setting in the grid -- onset threshold, frame threshold, minimum note length, by default
Task 3's 30, or as ``--onsets``, ``--frames`` and ``--lengths-ms`` give them -- Basic
Pitch transcribes player 3's 12 takes (direct input) and the default decoder strings its notes;
each take's end-to-end E2 and chord-shape counts are written as ``<out>/<setting>.json``, in the
per-track form ``scripts/compare_validation.py`` compares. Scored against each take's cleaned
labels with its label delay taken off (ADR 0055). Validation data only: no transcriber has
trained on Guitar-TECHS. The rule that picks a setting is the plan's, fixed before the run.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import replace
from pathlib import Path

from tabsampler.config import load_eval_config, load_phase1_config
from tabsampler.data.guitartechs import aligned, load_takes, usable
from tabsampler.decode.robust import decode_best_effort
from tabsampler.eval.metrics import exact_tab_f1, note_f1, tab_notes_to_placed
from tabsampler.eval.playability import playability_rate
from tabsampler.eval.recovery import RecoveryReport, add_track
from tabsampler.fingering.candidates import group_notes
from tabsampler.fingering.costs import HandSetScorer
from tabsampler.transcribe.basic_pitch_cli import BasicPitchCLITranscriber

ONSETS = (0.3, 0.4, 0.5, 0.6, 0.7)
FRAMES = (0.3, 0.4, 0.5)
LENGTHS_MS = (58.0, 127.70)
VALIDATION_PLAYER = 3  # Guitar-TECHS (ADR 0051)
TOLERANCE = 0.05


def name(onset: float, frame: float, length_ms: float) -> str:
    return f"onset{onset:g}_frame{frame:g}_min{length_ms:g}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("data/guitar-techs"))
    parser.add_argument("--config", type=Path, default=Path("configs/m1_full_eval.yaml"))
    parser.add_argument("--decoder-config", type=Path, default=Path("configs/decoder_clean.yaml"))
    parser.add_argument("--onsets", type=float, nargs="+", default=list(ONSETS))
    parser.add_argument("--frames", type=float, nargs="+", default=list(FRAMES))
    parser.add_argument("--lengths-ms", type=float, nargs="+", default=list(LENGTHS_MS))
    parser.add_argument("--model-path", type=Path, help="A fine-tuned model (ADR 0056).")
    args = parser.parse_args()
    cfg = load_eval_config(args.config)
    dec = load_phase1_config(args.decoder_config)
    scorer = HandSetScorer(weights=dec.weights)
    takes = [t for t in load_takes(args.root)[0] if t.player == VALIDATION_PLAYER and usable(t)]
    references = {take.name: aligned(take) for take in takes}
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(takes)} takes, {sum(len(r) for r in references.values())} reference notes")
    for onset, frame, length in itertools.product(args.onsets, args.frames, args.lengths_ms):
        params = replace(
            cfg.transcriber.params,
            onset_threshold=onset,
            frame_threshold=frame,
            minimum_note_length_ms=length,
        )
        transcriber = BasicPitchCLITranscriber(
            exe=cfg.transcriber.exe,
            params=params,
            cache_dir=cfg.transcriber.cache_dir,
            model=args.model_path,
        )
        report = RecoveryReport()
        e1 = [0, 0]
        for take in takes:
            reference = list(references[take.name])
            estimated = transcriber.transcribe_file(take.direct_input)
            heard = note_f1([n for n, _ in reference], estimated, TOLERANCE)
            e1 = [e1[0] + 2 * heard.n_match, e1[1] + heard.n_ref + heard.n_est]
            groups = group_notes(estimated, window_s=dec.group_window_s)
            tab = decode_best_effort(groups, scorer, dec.context)[0] if groups else []
            e2 = exact_tab_f1(reference, tab_notes_to_placed(tab), TOLERANCE)
            e3 = playability_rate(tab, dec.rules, window_s=dec.group_window_s)
            add_track(report, f"P3 {take.name}", "e2e", e2, e3)
        setting = name(onset, frame, length)
        payload = {
            "split": "validation",
            "model": str(args.model_path) if args.model_path else "released",
            "transcriber": {"onset": onset, "frame": frame, "min_ms": length},
            **report.to_dict(),
        }
        (args.out / f"{setting}.json").write_text(json.dumps(payload))  # one model per --out
        right, total = report.counts("e2e")
        print(f"  {setting:28s} E1 {e1[0] / e1[1]:.4f}  E2 {right / total:.4f}", flush=True)


if __name__ == "__main__":
    main()

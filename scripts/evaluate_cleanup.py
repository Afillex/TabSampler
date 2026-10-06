"""Filters on Basic Pitch's notes, scored end to end on Guitar-TECHS's player 3 (plan
2026-10-06-optimise-current, Task 3).

    uv run python -u scripts/evaluate_cleanup.py --out cache/validation/cleanup
    uv run python -u scripts/evaluate_cleanup.py --out cache/validation/cleanup --floors 0.3 \\
        --ghosts

Basic Pitch at ``CHOSEN_PARAMS`` transcribes player 3's 12 takes (direct input); each setting
filters the notes (``transcribe/cleanup.py``) and the default decoder strings them. Writes
``<out>/<setting>.json`` in the per-track form ``scripts/compare_validation.py`` compares, and
prints E1 and E2. Settings: no filter; a loudness floor for each of ``--floors``; with
``--ghosts``, octave ghosts dropped after each floor (and after none). Scored against cleaned
labels with each take's label delay taken off (ADR 0055). Validation only; the adoption rule
is the plan's, fixed before the run.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

from tabsampler.config import load_phase1_config
from tabsampler.data.guitartechs import aligned, load_takes, usable
from tabsampler.decode.robust import decode_best_effort
from tabsampler.eval.metrics import exact_tab_f1, note_f1, tab_notes_to_placed
from tabsampler.eval.playability import playability_rate
from tabsampler.eval.recovery import RecoveryReport, add_track
from tabsampler.fingering.candidates import group_notes
from tabsampler.fingering.costs import HandSetScorer
from tabsampler.transcribe.basic_pitch_cli import CHOSEN_PARAMS, BasicPitchCLITranscriber
from tabsampler.transcribe.cleanup import drop_octave_ghosts, drop_quiet
from tabsampler.types import NoteEvent

FLOORS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
TOLERANCE = 0.05

Filter = Callable[[list[NoteEvent]], list[NoteEvent]]


def settings(floors: list[float], ghosts: bool) -> dict[str, Filter]:
    out: dict[str, Filter] = {"none": lambda notes: notes}
    for floor in floors:
        out[f"quiet{floor:g}"] = lambda notes, f=floor: drop_quiet(notes, f)
    if ghosts:
        for name, base in list(out.items()):
            out[f"{name}+ghosts"] = lambda notes, b=base: drop_octave_ghosts(b(notes))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("data/guitar-techs"))
    parser.add_argument("--decoder-config", type=Path, default=Path("configs/decoder_clean.yaml"))
    parser.add_argument("--floors", type=float, nargs="*", default=list(FLOORS))
    parser.add_argument("--ghosts", action="store_true")
    args = parser.parse_args()
    dec = load_phase1_config(args.decoder_config)
    scorer = HandSetScorer(weights=dec.weights)
    transcriber = BasicPitchCLITranscriber(params=CHOSEN_PARAMS)
    takes = [t for t in load_takes(args.root)[0] if t.player == 3 and usable(t)]
    transcribed = {take.name: transcriber.transcribe_file(take.direct_input) for take in takes}
    args.out.mkdir(parents=True, exist_ok=True)

    for name, keep in settings(args.floors, args.ghosts).items():
        report = RecoveryReport()
        e1 = [0, 0]
        kept = 0
        for take in takes:
            reference = list(aligned(take))
            notes = keep(transcribed[take.name])
            kept += len(notes)
            heard = note_f1([n for n, _ in reference], notes, TOLERANCE)
            e1 = [e1[0] + 2 * heard.n_match, e1[1] + heard.n_ref + heard.n_est]
            groups = group_notes(notes, window_s=dec.group_window_s)
            tab = decode_best_effort(groups, scorer, dec.context)[0] if groups else []
            e2 = exact_tab_f1(reference, tab_notes_to_placed(tab), TOLERANCE)
            add_track(
                report,
                f"P3 {take.name}",
                "e2e",
                e2,
                playability_rate(tab, dec.rules, window_s=dec.group_window_s),
            )
        payload = {"split": "validation", "setting": name, **report.to_dict()}
        (args.out / f"{name}.json").write_text(json.dumps(payload))
        right, total = report.counts("e2e")
        print(
            f"  {name:16s} notes kept {kept:5d}  E1 {e1[0] / e1[1]:.4f}  E2 {right / total:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()

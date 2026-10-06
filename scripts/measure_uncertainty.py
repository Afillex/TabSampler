"""Do the uncertainty marks point at wrong notes? (plan 2026-10-06-optimise-current, Task 2)

    uv run python -u scripts/measure_uncertainty.py --out cache/validation/uncertainty

End to end -- Basic Pitch at ``CHOSEN_PARAMS``, the default decoder -- on Guitar-TECHS's player 3
(12 takes, direct input; Basic Pitch never trained on it) and GuitarSet's player 00 (60 tracks,
microphone; labelled, ADR 0055). Every tab note is scored right or wrong by E2's matching. For
each threshold t, a note is *marked* when its posterior is below t; printed per player: the share
of notes marked, the error rate among marked and unmarked notes, and their ratio (the lift).
Validation only; the rule that picks t is the plan's, fixed before the run.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from tabsampler.config import load_phase1_config
from tabsampler.data.guitarset import load_dataset, reference_tab
from tabsampler.data.guitartechs import aligned, load_takes, usable
from tabsampler.data.splits import guitarset_validation_ids
from tabsampler.decode.robust import decode_best_effort
from tabsampler.eval.metrics import exact_tab_f1_with_matches, tab_notes_to_placed
from tabsampler.fingering.candidates import group_notes
from tabsampler.fingering.costs import HandSetScorer
from tabsampler.transcribe.basic_pitch_cli import CHOSEN_PARAMS, BasicPitchCLITranscriber
from tabsampler.types import NoteEvent, Position

THRESHOLDS = [round(0.30 + 0.05 * k, 2) for k in range(13)]  # 0.30 ... 0.90
TOLERANCE = 0.05

Piece = tuple[str, list[tuple[NoteEvent, Position]], Path]


def player3(root: Path) -> Iterator[Piece]:
    for take in load_takes(root)[0]:
        if take.player == 3 and usable(take):
            yield f"P3 {take.name}", list(aligned(take)), take.direct_input


def player00(tuning: Any) -> Iterator[Piece]:
    dataset: Any = load_dataset(Path("data/guitarset"))
    for track_id in guitarset_validation_ids():
        track = dataset.track(track_id)
        yield track_id, list(reference_tab(track, tuning)), Path(track.audio_mic_path)


def table(scored: list[tuple[float, bool]]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for t in THRESHOLDS:
        marked = [ok for p, ok in scored if p < t]
        plain = [ok for p, ok in scored if p >= t]
        wrong_m = sum(not ok for ok in marked) / len(marked) if marked else float("nan")
        wrong_u = sum(not ok for ok in plain) / len(plain) if plain else float("nan")
        rows.append(
            {
                "t": t,
                "marked_share": len(marked) / len(scored),
                "wrong_marked": wrong_m,
                "wrong_unmarked": wrong_u,
                "lift": wrong_m / wrong_u if plain and wrong_u > 0 else float("nan"),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--decoder-config", type=Path, default=Path("configs/decoder_clean.yaml"))
    args = parser.parse_args()
    dec = load_phase1_config(args.decoder_config)
    scorer = HandSetScorer(weights=dec.weights)
    transcriber = BasicPitchCLITranscriber(params=CHOSEN_PARAMS)
    args.out.mkdir(parents=True, exist_ok=True)

    for name, pieces in (
        ("player 3", player3(Path("data/guitar-techs"))),
        ("player 00", player00(dec.tuning)),
    ):
        scored: list[tuple[float, bool]] = []
        for _, reference, audio in pieces:
            notes = transcriber.transcribe_file(audio)
            groups = group_notes(notes, window_s=dec.group_window_s)
            tab = decode_best_effort(groups, scorer, dec.context)[0] if groups else []
            _, flags = exact_tab_f1_with_matches(reference, tab_notes_to_placed(tab), TOLERANCE)
            scored += [(t.posterior, ok) for t, ok in zip(tab, flags, strict=True)]
        rows = table(scored)
        wrong = sum(not ok for _, ok in scored) / len(scored)
        print(f"{name}: {len(scored)} tab notes, {wrong:.3f} wrong overall")
        for r in rows:
            print(
                f"  t {r['t']:.2f}  marked {r['marked_share']:.3f}  wrong if marked "
                f"{r['wrong_marked']:.3f}  if not {r['wrong_unmarked']:.3f}  lift {r['lift']:.2f}"
            )
        slug = name.replace(" ", "")
        (args.out / f"{slug}.json").write_text(
            json.dumps({"split": "validation", "n_notes": len(scored), "rows": rows})
        )


if __name__ == "__main__":
    main()

"""Evaluate the learned model on GuitarSet's validation player, oracle mode (C4, ADR 0043).

The spec's Phase 2 comparison, on player 00's 60 tracks, from the reference notes:

    (a) the default decoder, exactly as the CLI runs it (``decode_best_effort``);
    (b) the learned term alone: per group, the node with the lowest learned cost;
    (c) the learned model: its full energies, decoded by Viterbi.

(b) and (c) decode exactly the groups (a) does -- spans widened and unfingerable notes dropped
by ``prepare_groups`` -- in float64. Writes per-track counts for
``scripts/compare_validation.py`` and prints E2 and E3's chord-shape rate for each decoding.
Reads player 00 only (ADR 0037): no test-set look is made or logged.

    uv run python scripts/evaluate_model.py --run cache/model/c4 --out cache/validation/c4
    uv run python scripts/evaluate_model.py --untrained --out cache/validation/c4-untrained

``--untrained`` scores the model as it starts, with no learned term: (c) must then place the
notes as (a) does, which checks the plumbing on real data.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from tabsampler.config import Phase1Config, load_phase1_config
from tabsampler.data.guitarset import load_dataset, reference_notes, reference_tab
from tabsampler.data.splits import guitarset_validation_ids
from tabsampler.decode.robust import decode_best_effort, prepare_groups
from tabsampler.eval.metrics import exact_tab_f1, tab_notes_to_placed
from tabsampler.eval.playability import playability_rate
from tabsampler.eval.recovery import RecoveryReport, add_track
from tabsampler.fingering.candidates import group_notes
from tabsampler.fingering.costs import HandSetScorer
from tabsampler.model.batch import make_batch
from tabsampler.model.crf import viterbi
from tabsampler.model.lattice import lattice_arrays
from tabsampler.model.net import LearnedModel
from tabsampler.types import NoteEvent, NoteGroup, Position, TabNote

ONSET_TOLERANCE = 0.05  # E2's, as in configs/m2_validation_*.yaml
DECODINGS = ("a", "b", "c")


def learned_tabs(
    model: LearnedModel, groups: Sequence[NoteGroup], spans: Sequence[int], dec: Phase1Config
) -> dict[str, list[TabNote]]:
    """Decodings (b) and (c) of ``groups``, as tab."""
    arrays = lattice_arrays(groups, dec.context, spans)
    batch = make_batch([arrays], dtype=torch.float64)
    with torch.no_grad():
        energies, transitions = model.energies(batch)
        full, _ = viterbi(energies, transitions, batch.lengths)
        alone = model.learned_term(batch).argmin(dim=2)
    return {
        "b": _tab(groups, arrays.states, alone[0].tolist()),
        "c": _tab(groups, arrays.states, full[0].tolist()),
    }


def _tab(
    groups: Sequence[NoteGroup], states: Sequence[Sequence[Any]], path: Sequence[int]
) -> list[TabNote]:
    tab: list[TabNote] = []
    for group, level, node in zip(groups, states, path, strict=True):
        for index, note in enumerate(group.notes):
            position: Position = level[node].positions[index]
            tab.append(TabNote(note=note, position=position, posterior=1.0))
    return tab


def evaluate(
    model: LearnedModel,
    reference: Sequence[tuple[NoteEvent, Position]],
    notes: Sequence[NoteEvent],
    dec: Phase1Config,
) -> dict[str, tuple[Any, Any]]:
    """E2 and E3 of each decoding on one track."""
    groups = group_notes(notes, window_s=dec.group_window_s)
    if not groups:
        raise ValueError("a track with no notes cannot be scored")
    default, _ = decode_best_effort(groups, HandSetScorer(weights=dec.weights), dec.context)
    prepared, spans, _ = prepare_groups(groups, dec.context)
    tabs = {"a": default, **learned_tabs(model, prepared, spans, dec)}
    return {
        name: (
            exact_tab_f1(reference, tab_notes_to_placed(tab), ONSET_TOLERANCE),
            playability_rate(tab, dec.rules, window_s=dec.group_window_s),
        )
        for name, tab in tabs.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="The training run whose best.pt to score.")
    parser.add_argument("--untrained", action="store_true", help="Score the starting model.")
    parser.add_argument("--decoder-config", type=Path, default=Path("configs/decoder_clean.yaml"))
    parser.add_argument("--out", type=Path, required=True, help="Directory for per-track counts.")
    args = parser.parse_args()
    if args.untrained == (args.run is not None):
        parser.error("give exactly one of --run and --untrained")

    dec = load_phase1_config(args.decoder_config)
    model = LearnedModel(dec.weights, dtype=torch.float64)
    if args.run is not None:
        model.load_state_dict(torch.load(args.run / "best.pt", weights_only=True))
    model.eval()

    dataset: Any = load_dataset(Path("data/guitarset"))
    reports = {name: RecoveryReport() for name in DECODINGS}
    for track_id in guitarset_validation_ids():
        track = dataset.track(track_id)
        scores = evaluate(model, reference_tab(track, dec.tuning), reference_notes(track), dec)
        for name, (e2, e3) in scores.items():
            add_track(reports[name], track_id, "oracle", e2, e3)

    args.out.mkdir(parents=True, exist_ok=True)
    source = "untrained" if args.untrained else str(args.run)
    for name, report in reports.items():
        payload = {"split": "validation", "decoding": name, "model": source, **report.to_dict()}
        (args.out / f"{name}.json").write_text(json.dumps(payload))
        right, total = report.counts("oracle")
        passed, shapes = report.shapes["oracle"]
        print(
            f"({name}) E2 {right / total:.4f}   chord shapes {passed}/{shapes}   "
            f"-> {args.out / f'{name}.json'}"
        )


if __name__ == "__main__":
    main()

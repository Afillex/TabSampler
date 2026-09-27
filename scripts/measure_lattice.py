"""Measure the node lattice against the bare state count (ADR 0018).

Complexity instrumentation, not a metric: it computes no E-number and no threshold is
chosen from it. Viterbi is ``O(T * S^2)``, so the question the ADR had to answer is how
much the hand-position augmentation grows S. Synthetic inputs need nothing external.

``--guitarset`` additionally measures it on real reference notes. That reads the test set,
so it logs the access with a written reason (ADR 0003); it still computes no metric.

Usage:
    uv run python scripts/measure_lattice.py
    uv run python scripts/measure_lattice.py --guitarset 60
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tabsampler.decode.viterbi import build_lattice
from tabsampler.fingering.states import enumerate_states
from tabsampler.types import Context, NoteEvent, NoteGroup, Tuning

STANDARD = Tuning.STANDARD

#: Open-string pitches in standard tuning. A line of nothing but these is the worst case
#: for the carry: they are the only pitches with an all-open state, and only an all-open
#: state makes a level fan out.
OPEN_PITCHES = list(STANDARD.open_pitches)

CASES: tuple[tuple[str, list[int]], ...] = (
    ("E-minor pentatonic run", [40, 43, 45, 47, 50, 52, 55, 57, 59, 62, 64, 62, 59, 57, 55, 52]),
    ("chromatic cycle from A2", [45 + (index % 12) for index in range(120)]),
    ("open-string pitches only (worst case)", OPEN_PITCHES * 20),
)


def notes_to_groups(pitches: Sequence[int]) -> list[NoteGroup]:
    return [
        NoteGroup.of([NoteEvent(onset=i * 0.5, offset=i * 0.5 + 0.4, pitch=p, confidence=1.0)])
        for i, p in enumerate(pitches)
    ]


def report(label: str, groups: Sequence[NoteGroup], ctx: Context) -> None:
    states = sum(len(enumerate_states(g, ctx.tuning, ctx.max_span)) for g in groups)
    lattice = build_lattice(groups, ctx)
    nodes = sum(len(level) for level in lattice)
    carried = {node.carried_hand for level in lattice for node in level}
    print(
        f"  {label:40s} groups {len(groups):5d}  states/grp {states / len(groups):6.3f}  "
        f"nodes/grp {nodes / len(groups):6.3f}  ratio {nodes / states:5.3f}x  "
        f"distinct carried hands {len(carried)}"
    )


def measure_guitarset(n_tracks: int, ctx: Context, window_s: float) -> None:
    from tabsampler.data.guitarset import load_dataset, reference_notes
    from tabsampler.data.splits import guitarset_test_ids, record_test_set_access
    from tabsampler.fingering.candidates import group_notes

    record_test_set_access(
        f"measured node-lattice size (ADR 0018) on {n_tracks} GuitarSet tracks via "
        f"scripts/measure_lattice.py: decoder complexity instrumentation, no metric "
        f"computed and no threshold chosen from it"
    )
    dataset = load_dataset(Path("data/guitarset"))
    tracks: dict[str, Any] = dataset.load_tracks()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    groups: list[NoteGroup] = []
    for track_id in list(guitarset_test_ids())[:n_tracks]:
        for group in group_notes(reference_notes(tracks[track_id]), window_s=window_s):
            if enumerate_states(group, ctx.tuning, ctx.max_span):
                groups.append(group)
    report(f"GuitarSet reference notes, {n_tracks} tracks", groups, ctx)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-span", type=int, default=5)
    parser.add_argument("--window-s", type=float, default=0.03)
    parser.add_argument(
        "--guitarset", type=int, default=0, help="Also measure on N GuitarSet tracks."
    )
    args = parser.parse_args()
    ctx = Context(tuning=STANDARD, max_span=args.max_span)

    print(f"node lattice vs state count, max_span {args.max_span}")
    for label, pitches in CASES:
        report(label, notes_to_groups(pitches), ctx)
    if args.guitarset:
        measure_guitarset(args.guitarset, ctx, args.window_s)


if __name__ == "__main__":
    main()

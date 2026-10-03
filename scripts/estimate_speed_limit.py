"""Set E3's speed limit from human tab and check it on unseen artists (ADR 0031).

Pre-registered in ADR 0031 before this ran: on every cleared song of the artist-split
training side, find the smallest limit under which 0.9986 of transitions pass -- the rate at
which ADR 0011's chord rules hold on human tab -- rounded up to a whole number of frets per
second; then check that at least 0.9976 of validation transitions pass at it. The hand and
the clock follow E3 exactly (ADR 0025, ADR 0029, ADR 0030): this script only collects
``hand_moves`` and fits nothing else.

    uv run python scripts/estimate_speed_limit.py data/dadagp/DadaGP-v1.1.zip \\
        data/dadagp/track_meta.json
"""

from __future__ import annotations

import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from tabsampler.data.dadagp import load_tracks
from tabsampler.data.splits import Split
from tabsampler.eval.playability import PlayabilityRules, hand_moves
from tabsampler.eval.speed import speed_limit_for

TARGET = 0.9986  # ADR 0022: the chord rules' pass rate on human tab
CHECK = 0.9976  # ADR 0031: the validation side may fall short of it by 0.1 point


class Side:
    """Every human hand move on one side of the split."""

    def __init__(self) -> None:
        self.transitions = 0
        self.free = 0  # no move: passes at any limit
        self.instant = 0  # a move with no time between groups: fails at any limit
        self.speeds: list[float] = []
        self.distances: list[int] = []

    def add(self, moves: list[tuple[int, float]]) -> None:
        for distance, seconds in moves:
            self.transitions += 1
            if distance == 0:
                self.free += 1
            elif seconds <= 0.0:
                self.instant += 1
            else:
                self.speeds.append(distance / seconds)
                self.distances.append(distance)

    def moves_passing(self, limit: float) -> int:
        """Timed moves at or under ``limit`` -- moves where the hand actually shifts."""
        return int(np.count_nonzero(np.asarray(self.speeds) <= limit))

    def pass_rate(self, limit: float) -> float:
        return (self.free + self.moves_passing(limit)) / self.transitions


def collect(archive: Path, meta: Path, split: Split) -> Side:
    side = Side()
    for track in load_tracks(archive, split, meta, scheme="artist"):
        side.add(hand_moves([(g.onset, s.positions) for g, s in track.steps]))
    return side


def main() -> None:
    archive, meta = Path(sys.argv[1]), Path(sys.argv[2])
    started = time.perf_counter()
    train = collect(archive, meta, Split.TRAIN)
    val = collect(archive, meta, Split.VALIDATION)
    print(f"collected in {time.perf_counter() - started:.0f}s", flush=True)
    for name, side in (("training", train), ("validation", val)):
        print(
            f"{name:10s}: {side.transitions} transitions, {side.free} with no move, "
            f"{len(side.speeds)} timed moves, {side.instant} moves with no time"
        )

    raw = speed_limit_for(TARGET, train.speeds, train.free, train.transitions)
    limit = float(math.ceil(raw))
    current = PlayabilityRules().max_frets_per_second
    print()
    print(f"training limit for {TARGET}: {raw:.2f} frets/s, rounded up to {limit:.0f}")
    print(f"training pass rate   at {current:.0f} frets/s: {train.pass_rate(current):.4f}")
    print(f"training pass rate   at {limit:.0f} frets/s: {train.pass_rate(limit):.4f}")
    print(f"validation pass rate at {current:.0f} frets/s: {val.pass_rate(current):.4f}")
    rate = val.pass_rate(limit)
    verdict = "HOLDS" if rate >= CHECK else "FAILS"
    print(f"validation pass rate at {limit:.0f} frets/s: {rate:.4f}  (check >= {CHECK}: {verdict})")

    # 0.9986 is a share of transitions, most of which move nothing; count the moves too.
    for name, side in (("training", train), ("validation", val)):
        for at in (current, limit):
            moved = len(side.speeds) + side.instant
            print(
                f"{name:10s} at {at:.0f} frets/s: {side.free + side.moves_passing(at)}/"
                f"{side.transitions} transitions pass; {side.moves_passing(at)}/{moved} "
                f"of the moves where the hand shifts ({side.moves_passing(at) / moved:.4%})"
            )
    speeds, distances = np.asarray(val.speeds), np.asarray(val.distances)
    failing = Counter(int(d) for d in distances[speeds > limit])
    print(
        f"validation moves still failing at {limit:.0f} frets/s, by frets moved: "
        f"{dict(sorted(failing.items()))}; plus {val.instant} with no time"
    )


if __name__ == "__main__":
    main()

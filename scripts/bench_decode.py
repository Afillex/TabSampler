"""Benchmark the decoder directly, on synthetic input. Needs no dataset.

Why this exists: the E7 column of the ``eval-m1`` rows in ``experiments/results.csv`` is
decode time with a warm transcriber cache, and on one machine in one afternoon it measured
0.0370, 0.17, 0.20 and 0.04 seconds per audio minute for equivalent work. Its noise floor
is wider than any decode-time change our code has made, so it cannot answer "did that
change make decoding slower". This can.

Used to measure ADR 0018's node lattice against the bare-state lattice it replaced, by
running it once per commit. To compare two commits, check the older one out into a
worktree and point PYTHONPATH at each tree in turn so each run loads its own source::

    git worktree add /tmp/before <older-commit>
    for tree in /tmp/before .; do
        PYTHONPATH=$tree/src .venv/bin/python scripts/bench_decode.py
    done

Usage:
    uv run python scripts/bench_decode.py [--groups 500] [--repeats 7]
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable

from tabsampler.decode.forward_backward import decode
from tabsampler.decode.viterbi import viterbi
from tabsampler.fingering.costs import HandSetScorer
from tabsampler.types import Context, NoteEvent, NoteGroup, Tuning

#: A chromatic cycle from A2 upward. Deliberately includes the open-string pitches 45, 50
#: and 55, because those are the groups that make the ADR 0018 lattice carry a hand
#: position and so are what the measurement is about.
BASE_PITCH = 45
CYCLE = 12


def build_groups(n_groups: int) -> list[NoteGroup]:
    return [
        NoteGroup.of(
            [
                NoteEvent(
                    onset=index * 0.2,
                    offset=index * 0.2 + 0.4,
                    pitch=BASE_PITCH + (index % CYCLE),
                    confidence=1.0,
                )
            ]
        )
        for index in range(n_groups)
    ]


def time_median_ms(run: Callable[[], object], repeats: int) -> tuple[float, float]:
    """(median, min) wall time in milliseconds, after one warm-up call."""
    run()
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        run()
        samples.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(samples), min(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--max-span", type=int, default=5)
    args = parser.parse_args()

    groups = build_groups(args.groups)
    ctx = Context(tuning=Tuning.STANDARD, max_span=args.max_span)
    scorer = HandSetScorer()

    print(
        f"{args.groups} single-note groups, max_span {args.max_span}, "
        f"median of {args.repeats} warmed runs"
    )
    for label, run in (
        ("viterbi", lambda: viterbi(groups, scorer, ctx)),
        ("decode (viterbi + forward-backward)", lambda: decode(groups, scorer, ctx)),
    ):
        median, fastest = time_median_ms(run, args.repeats)
        print(f"  {label:38s} median {median:7.1f} ms   min {fastest:7.1f} ms")


if __name__ == "__main__":
    main()

"""Score one decoder config on DadaGP validation, per song and part (ADR 0023, ADR 0028).

    uv run python scripts/score_validation.py data/dadagp/DadaGP-v1.1.zip \\
        data/dadagp/track_meta.json --decoder-config configs/phase1_baseline.yaml \\
        --split artist --out cache/validation/window.json

The JSON names DadaGP songs, so it goes under cache/, which is never committed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tabsampler.config import load_phase1_config
from tabsampler.data.dadagp import load_tracks
from tabsampler.data.splits import Split
from tabsampler.decode.viterbi import build_lattice
from tabsampler.eval.recovery import PartSequence, recover
from tabsampler.fingering.costs import HandSetScorer
from tabsampler.fingering.fit import human_sequences
from tabsampler.fingering.states import enumerate_states
from tabsampler.types import Context, Tuning

DADAGP_TUNING = Tuning(n_frets=24)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("meta", type=Path)
    parser.add_argument("--decoder-config", type=Path, required=True)
    parser.add_argument("--split", choices=("shipped", "artist"), required=True)
    parser.add_argument("--val-songs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    weights = load_phase1_config(args.decoder_config).weights
    ctx = Context(tuning=DADAGP_TUNING, max_span=5)
    started = time.perf_counter()
    items = [
        PartSequence(
            track.song, "clean" if track.instrument.startswith("clean") else "distorted", s
        )
        for track in load_tracks(
            args.archive,
            Split.VALIDATION,
            args.meta,
            scheme=args.split,
            tuning=DADAGP_TUNING,
            sample=args.val_songs,
            seed=args.seed,
        )
        for s in human_sequences(track.steps, ctx)
    ]
    print(
        f"split scheme: {args.split}; {len(items)} sequences loaded in "
        f"{time.perf_counter() - started:.0f}s",
        flush=True,
    )

    nodes = states = 0
    for item in items:
        lattice = build_lattice(item.sequence.groups, ctx, item.sequence.spans)
        nodes += sum(len(level) for level in lattice)
        states += sum(
            len(enumerate_states(g, ctx.tuning, span))
            for g, span in zip(item.sequence.groups, item.sequence.spans, strict=True)
        )
    report = recover(items, HandSetScorer(weights=weights), ctx)
    for part in (None, "clean", "distorted"):
        hits, notes = report.counts(part)
        print(
            f"{part or 'all':9s}: recovery {report.share(part):.4f} ({hits} of {notes} notes)"
            f"   decoded chord shapes {report.chord_shape_rate(part):.4f}"
        )
    print(f"lattice nodes per state: {nodes / states:.3f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"decoder": str(args.decoder_config), **report.to_dict()}))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

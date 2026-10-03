"""Paired, song-level comparison of two validation reports (ADR 0028).

    uv run python scripts/compare_validation.py cache/validation/a.json cache/validation/b.json

Each file is what scripts/score_validation.py (or fit_cost_weights.py --per-song-out)
writes. The delta is the second minus the first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tabsampler.eval.bootstrap import paired_bootstrap
from tabsampler.eval.recovery import RecoveryReport


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired comparison of two validation reports.")
    parser.add_argument("base", type=Path, help="the decoder compared against")
    parser.add_argument("new", type=Path, help="the decoder being judged")
    args = parser.parse_args()
    a, b = (RecoveryReport.from_dict(json.loads(p.read_text())) for p in (args.base, args.new))
    for part in (None, "clean", "distorted"):
        diff = paired_bootstrap(a.song_counts(part), b.song_counts(part))
        print(
            f"{part or 'all':9s}: {a.share(part):.4f} -> {b.share(part):.4f}   "
            f"delta {diff.delta:+.4f}  95% interval [{diff.low:+.4f}, {diff.high:+.4f}]   "
            f"chord shapes {a.chord_shape_rate(part):.4f} -> {b.chord_shape_rate(part):.4f}   "
            f"({diff.n_songs} songs, {diff.n_notes} notes)"
        )


if __name__ == "__main__":
    main()

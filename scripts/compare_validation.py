"""Paired, song-level comparison of two validation reports (ADR 0028).

    uv run python scripts/compare_validation.py cache/validation/a.json cache/validation/b.json

Each file is what scripts/score_validation.py (or fit_cost_weights.py --per-song-out, or
tabsampler eval-m1 --split validation --per-track-out) writes. The delta is the second
minus the first. A file holding any of GuitarSet's test tracks is refused (ADR 0037).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tabsampler.data.splits import assert_no_test_tracks
from tabsampler.errors import TestSetMisuseError
from tabsampler.eval.bootstrap import paired_bootstrap
from tabsampler.eval.recovery import RecoveryReport


def shape_counts(report: RecoveryReport, part: str | None) -> tuple[int, int]:
    """(decoded shapes passing E3's chord rules, decoded shapes), for one part or all."""
    chosen = [c for name, c in report.shapes.items() if part is None or name == part]
    return sum(c[0] for c in chosen), sum(c[1] for c in chosen)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired comparison of two validation reports.")
    parser.add_argument("base", type=Path, help="the decoder compared against")
    parser.add_argument("new", type=Path, help="the decoder being judged")
    args = parser.parse_args()
    a, b = (RecoveryReport.from_dict(json.loads(p.read_text())) for p in (args.base, args.new))
    for path, report in ((args.base, a), (args.new, b)):
        try:
            assert_no_test_tracks(report.per_song)
        except TestSetMisuseError as exc:
            raise SystemExit(f"{path}: {exc}") from exc
    parts = sorted(set(a.shapes) | set(b.shapes))
    # DadaGP reports split by style and pool meaningfully; GuitarSet ones split by mode
    # (oracle, end to end), and pooling the two modes would mean nothing.
    order: list[str | None] = [None, *parts] if set(parts) <= {"clean", "distorted"} else [*parts]
    for part in order:
        songs = a.song_counts(part)
        if not songs:
            continue
        diff = paired_bootstrap(songs, b.song_counts(part))
        (pa, na), (pb, nb) = shape_counts(a, part), shape_counts(b, part)
        print(
            f"{part or 'all':9s}: {a.share(part):.4f} -> {b.share(part):.4f}   "
            f"delta {diff.delta:+.4f}  95% interval [{diff.low:+.4f}, {diff.high:+.4f}]   "
            f"({diff.n_songs} songs, {diff.n_notes} notes)"
        )
        # Exact, because pre-registered rules put thresholds on this (ADR 0032).
        print(f"           chord shapes {pa}/{na} -> {pb}/{nb}, drop {pa / na - pb / nb:+.6f}")


if __name__ == "__main__":
    main()

"""Where the default decoder loses human fingerings on GuitarSet's validation player.

Phase 2 task C3, ``docs/plans/2026-10-04-c3-guitarset-errors.md`` Task 1, whose questions
and predictions were fixed before the run. A diagnostic, not a metric: it writes no
``results.csv`` row, and nothing is chosen from it except which feature groups are worth
arguing for. Oracle mode: the decoder is given the player's own notes, so every error is a
fingering error.

Reads player 00 only (ADR 0037): validation data, so no test-set look is logged.

    uv run python scripts/analyse_errors.py
    uv run python scripts/analyse_errors.py --decoder-config configs/decoder_clean.yaml
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from tabsampler.config import load_phase1_config
from tabsampler.data.splits import guitarset_validation_ids
from tabsampler.decode.robust import decode_best_effort
from tabsampler.eval.metrics import exact_tab_f1, tab_notes_to_placed
from tabsampler.fingering.candidates import group_notes
from tabsampler.fingering.costs import HandSetScorer
from tabsampler.types import NoteEvent, Position, TabNote

#: A GuitarSet track id: player, style and number, tempo, key, and comping or soloing.
TRACK_ID = re.compile(r"^(\d\d)_([A-Za-z]+)\d-\d+-[A-G][#b]?_(comp|solo)$")
STYLES = ("BN", "Funk", "Jazz", "Rock", "SS")

#: E2's onset tolerance, so a note counts as right here exactly when E2 could match it.
ONSET_TOLERANCE = 0.05

#: Question 6: a run of at least this many consecutive misplaced notes is a passage placed
#: in another position rather than a one-note slip.
LONG_RUN = 4


class Outcome(NamedTuple):
    """One reference note: where the player put it, where the decoder did, and its context."""

    track: str
    style: str
    mode: str
    onset: float
    in_chord: bool
    human: Position
    #: ``None`` when the decoder dropped the note.
    decoded: Position | None

    @property
    def right(self) -> bool:
        return self.decoded is not None and self.decoded.string == self.human.string


def track_style_and_mode(track_id: str) -> tuple[str, str]:
    """``("BN", "comp")`` for ``00_BN1-129-Eb_comp``."""
    match = TRACK_ID.match(track_id)
    if match is None or match.group(2) not in STYLES:
        raise ValueError(f"not a GuitarSet track id: {track_id!r}")
    return match.group(2), match.group(3)


def pair(
    reference: Sequence[tuple[NoteEvent, Position]], decoded: Sequence[TabNote]
) -> list[Position | None]:
    """For each reference note, the decoder's position for the same note, or ``None``.

    The same note means the same pitch within E2's onset tolerance. Among several, one on
    the player's string is preferred, then the nearest onset, so that a note is right here
    whenever E2 could match it.
    """
    unused = list(range(len(decoded)))
    out: list[Position | None] = []
    for note, human in reference:
        candidates = [
            j
            for j in unused
            if decoded[j].note.pitch == note.pitch
            and round(abs(decoded[j].note.onset - note.onset), 4) <= ONSET_TOLERANCE
        ]
        if not candidates:
            out.append(None)
            continue
        best = min(
            candidates,
            key=lambda j: (
                decoded[j].position.string != human.string,
                abs(decoded[j].note.onset - note.onset),
            ),
        )
        unused.remove(best)
        out.append(decoded[best].position)
    return out


def chord_flags(reference: Sequence[tuple[NoteEvent, Position]], window_s: float) -> list[bool]:
    """Whether each reference note sounds in a group of two or more, as the decoder groups."""
    size: dict[tuple[float, int], int] = {}
    for group in group_notes([note for note, _ in reference], window_s=window_s):
        for note in group.notes:
            size[(note.onset, note.pitch)] = len(group.notes)
    return [size[(note.onset, note.pitch)] > 1 for note, _ in reference]


def outcomes_for(
    track_id: str,
    reference: Sequence[tuple[NoteEvent, Position]],
    decoded: Sequence[TabNote],
    window_s: float,
) -> list[Outcome]:
    style, mode = track_style_and_mode(track_id)
    placed = pair(reference, decoded)
    in_chord = chord_flags(reference, window_s)
    return [
        Outcome(track_id, style, mode, note.onset, chord, human, where)
        for (note, human), where, chord in zip(reference, placed, in_chord, strict=True)
    ]


def region(fret: int) -> str:
    """The fret regions of ADR 0034, with the open string on its own."""
    if fret == 0:
        return "open"
    if fret <= 4:
        return "1-4"
    if fret <= 11:
        return "5-11"
    return "12+"


def errors_in_long_runs(wrong: Sequence[bool], long_run: int = LONG_RUN) -> int:
    """How many of the misplaced notes sit in runs of at least ``long_run`` in a row."""
    total = run = 0
    for flag in [*wrong, False]:
        if flag:
            run += 1
            continue
        if run >= long_run:
            total += run
        run = 0
    return total


def share(part: int, whole: int) -> str:
    return f"{part:6d} / {whole:6d} = {part / whole:.4f}" if whole else f"{part:6d} / {whole:6d}"


def accuracy_by(
    outcomes: Iterable[Outcome], key: Callable[[Outcome], str]
) -> dict[str, tuple[int, int]]:
    """key -> (right, notes)."""
    right: Counter[str] = Counter()
    total: Counter[str] = Counter()
    for o in outcomes:
        total[key(o)] += 1
        right[key(o)] += o.right
    return {k: (right[k], total[k]) for k in sorted(total)}


def report(outcomes: Sequence[Outcome]) -> list[str]:
    """The six questions' answers, as lines."""
    wrong = [o for o in outcomes if not o.right]
    moved = [o for o in wrong if o.decoded is not None]
    lines = [
        f"notes {len(outcomes)}, right {len(outcomes) - len(wrong)}, misplaced {len(moved)}, "
        f"dropped {len(wrong) - len(moved)}"
    ]

    def table(title: str, key: Callable[[Outcome], str]) -> None:
        lines.append(f"\n{title}  (right / notes; share of all errors)")
        for k, (r, n) in accuracy_by(outcomes, key).items():
            lines.append(f"  {k:10s} {share(r, n)}   errors {share(n - r, len(wrong))}")

    table("1. single notes and notes in chords", lambda o: "chord" if o.in_chord else "single")
    table("2a. comping and soloing", lambda o: o.mode)
    table("2b. styles", lambda o: o.style)

    lines.append("\n3. misplaced notes: strings away, and where on the neck the decoder went")
    distance = Counter(min(abs(o.decoded.string - o.human.string), 3) for o in moved if o.decoded)
    for d in (1, 2, 3):
        label = f"{d} string" + ("s" if d > 1 else "") + (" or more" if d == 3 else "")
        lines.append(f"  {label:20s} {share(distance[d], len(moved))}")
    higher = sum(1 for o in moved if o.decoded and o.decoded.fret > o.human.fret)
    lines.append(f"  {'higher on the neck':20s} {share(higher, len(moved))}")

    table("4. the player's fret region", lambda o: region(o.human.fret))
    changed = sum(1 for o in moved if o.decoded and region(o.decoded.fret) != region(o.human.fret))
    lines.append(f"  misplaced and in another region: {share(changed, len(moved))}")

    lines.append("\n5. open strings among the misplaced notes")
    player_open = sum(1 for o in moved if o.decoded and o.human.fret == 0 and o.decoded.fret > 0)
    decoder_open = sum(1 for o in moved if o.decoded and o.decoded.fret == 0 and o.human.fret > 0)
    lines.append(f"  player open, decoder fretted: {share(player_open, len(moved))}")
    lines.append(f"  decoder open, player fretted: {share(decoder_open, len(moved))}")

    in_runs = 0
    for track in sorted({o.track for o in outcomes}):
        ordered = sorted(
            (o for o in outcomes if o.track == track), key=lambda o: (o.onset, o.human.string)
        )
        in_runs += errors_in_long_runs([not o.right for o in ordered])
    lines.append(f"\n6. errors in runs of {LONG_RUN} or more: {share(in_runs, len(wrong))}")
    return lines


def main() -> None:
    from tabsampler.data.guitarset import load_dataset, reference_notes, reference_tab

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder-config", type=Path, default=Path("configs/decoder_clean.yaml"))
    args = parser.parse_args()
    dec = load_phase1_config(args.decoder_config)
    scorer = HandSetScorer(weights=dec.weights)
    dataset: Any = load_dataset(Path("data/guitarset"))

    outcomes: list[Outcome] = []
    n_match = n_ref = n_est = 0
    for track_id in guitarset_validation_ids():
        track = dataset.track(track_id)
        reference = reference_tab(track, dec.tuning)
        groups = group_notes(reference_notes(track), window_s=dec.group_window_s)
        decoded, _ = decode_best_effort(groups, scorer, dec.context) if groups else ([], None)
        e2 = exact_tab_f1(reference, tab_notes_to_placed(decoded), ONSET_TOLERANCE)
        n_match, n_ref, n_est = n_match + e2.n_match, n_ref + e2.n_ref, n_est + e2.n_est
        outcomes.extend(outcomes_for(track_id, reference, decoded, dec.group_window_s))

    right = sum(o.right for o in outcomes)
    print(f"player 00, oracle mode, decoder {args.decoder_config}")
    print(
        f"E2 {2 * n_match / (n_ref + n_est):.4f} (matches {n_match}); "
        f"right here {right} -- the two must agree"
    )
    print("\n".join(report(outcomes)))


if __name__ == "__main__":
    main()

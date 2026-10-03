# E3 Timing, Hand Stretch, Style Decoders and First C3 Features — Implementation Plan

> Tasks run in order; steps use checkbox (`- [ ]`) syntax. Every run that produces a
> reported number is pre-registered in a commit made before it runs, with a tolerance and a
> noise estimate stated in advance (ADR 0028's lesson).

**Status:** executed 2026-10-03; results, misses and open questions in
`docs/devlog/2026-10-03.md`.

**Goal:** close the open questions left by the hand-window work and start Phase 2 task C3:
fix how E3 times moves across open strings, settle E3's speed limit, let a wide chord
stretch the hand, give clean and distorted guitar their own decoders and temperatures, and
try the first two richer feature groups — then measure the default once on GuitarSet.

**Architecture:** E3's transition rule gains one pure function that walks a piece and
returns each hand move as (frets, seconds), timed from the last fretted group; the speed
limit is then estimated from human tab on training artists and checked on validation
artists. The hand state becomes a covered fret range, `Hand = (lowest, highest)`, so a
chord wider than the window can stretch it. Validation scoring moves into a pure
`eval/recovery.py` that keeps per-song counts, and `eval/bootstrap.py` compares two
decoders on the same songs. The CRF fitter's feature vector grows to 11 columns, with
groups switched on by name, so each feature group is one experiment.

**Tech Stack:** Python 3.13, `uv`, numpy, scipy, pytest + hypothesis.

**Spec:** `docs/spec.md`; this plan implements Phase 2 tasks C3 (first part) and C5 of
`docs/plans/2026-10-01-phase-2.md`, and the four open items Ege approved on 2026-10-03.

## Decisions already taken by Ege (2026-10-03)

- Fix the `eval-m1` label bug and correct the mislabelled 2026-10-01 rows.
- Pre-register an experiment on E3's 12 frets/s speed limit, with the open-string timing
  fixed first.
- A chord wider than the 4-fret window stretches the hand instead of anchoring it.
- Clean and distorted guitar may each get their own temperature (and, by Phase 2 task C3's
  text, their own fitted weights).
- Go on with Phase 2 task C3.

## Global Constraints

- `requires-python = ">=3.13,<3.14"` (ADR 0001). `uv run pyright` strict passes on `src/`.
- Pure functions in `decode/`, `fingering/`, `eval/`. All I/O in `cli.py`, `config.py`,
  `results.py`, `data/` loaders and `scripts/`.
- GuitarSet is test-only (ADR 0003). It is read once, in Task 10, pre-registered and logged.
- `experiments/results.csv` rows only from runs that executed; a metric not computed is blank.
- Never edit an accepted ADR; supersede it. After the branch is reviewed, no in-place
  corrections either (2026-09-27 precedent). New ADRs: **0029** E3 timing, **0030** hand
  stretch, **0031** speed limit, **0032** style decoders, **0033** per-style temperature,
  **0034** C3 feature groups, **0035** E3 transition guardrail baseline.
- The repository is public: never commit `data/` or `cache/` (the datasets are
  research-use-only), nor local editor or tooling files. Per-song validation counts name
  DadaGP songs, so they live under `cache/validation/`, never in the repository.
- Every validation comparison states its tolerance before it runs and reports a song-level
  paired bootstrap 95% interval (Task 2).
- One variable per experiment. Long DadaGP runs go under `caffeinate -i`.
- `make check` before every commit, checking its exit code. Metric-code changes (Tasks 3, 4,
  5) are flagged loudly in the session report.

## Review Focus

1. **Open shapes at the start of a piece, or several in a row** — the hand has no position
   yet, or is carried across a run of open shapes; the timing must neither crash nor charge
   a move. → Task 3 `test_a_piece_that_starts_with_open_strings_has_no_moves_to_time` and
   the hypothesis property test.
2. **A wide chord alternating with its own top or bottom note**, also across an all-open
   shape — after the first placement, no movement at all. → Task 4
   `test_alternating_a_wide_chord_with_its_own_notes_never_moves_the_hand` and
   `test_the_stretch_is_carried_across_an_all_open_shape`.
3. **Comparing decoders on different song sets, or a song with different note counts** —
   the bootstrap must refuse, not compare apples with pears. → Task 2
   `test_the_bootstrap_refuses_different_songs` and `..._different_note_counts`.
4. **An existing decoder config without the new weight keys** must load and decode exactly
   as before. → Task 8 `test_an_old_config_loads_with_the_new_weights_at_zero`.
5. **Fitting one feature group** must leave every other new weight at exactly its starting
   value. → Task 8 `test_fitting_a_subset_leaves_the_other_weights_untouched`.

---

### Task 1: The `eval-m1` label (done, commit 3022bc0)

- [x] `results.describe_weights` and `results.m1_row_notes` state the decoder's actual
  weights and temperature; the banner, the notes column and the table title use them; the
  two 2026-10-01 rows carry a visible correction. Tests:
  `test_m1_row_notes_state_the_decoders_actual_weights`,
  `test_described_weights_include_the_temperature`.

---

### Task 2: Per-song validation counts and a paired bootstrap

**Files:**
- Create: `src/tabsampler/eval/recovery.py`, `src/tabsampler/eval/bootstrap.py`
- Create: `scripts/score_validation.py`, `scripts/compare_validation.py`
- Modify: `scripts/fit_cost_weights.py` (use `recover`; add `--per-song-out DIR`)
- Test: `tests/eval/test_recovery.py`, `tests/eval/test_bootstrap.py`

**Interfaces:**
- Produces: `PartSequence(song: str, part: str, sequence: HumanSequence)`;
  `recover(items: Sequence[PartSequence], scorer: FingeringScorer, ctx: Context,
  rules: PlayabilityRules | None = None) -> RecoveryReport`;
  `RecoveryReport.counts(part: str | None = None) -> tuple[int, int]`,
  `.share(part) -> float`, `.chord_shape_rate(part) -> float`,
  `.song_counts(part) -> dict[str, tuple[int, int]]`, `.to_dict() -> dict[str, Any]`,
  `RecoveryReport.from_dict(data) -> RecoveryReport`;
  `paired_bootstrap(a: Mapping[str, tuple[int, int]], b: Mapping[str, tuple[int, int]],
  n_resamples: int = 2000, seed: int = 0) -> PairedDifference(delta, low, high, n_songs,
  n_notes)`.

- [x] **Step 1: Write the failing tests** (`tests/eval/test_bootstrap.py`)

```python
from __future__ import annotations

import pytest

from tabsampler.eval.bootstrap import paired_bootstrap


def test_identical_decoders_differ_by_exactly_zero() -> None:
    counts = {"a": (5, 10), "b": (7, 10), "c": (1, 4)}
    result = paired_bootstrap(counts, counts)
    assert (result.delta, result.low, result.high) == (0.0, 0.0, 0.0)


def test_a_uniform_gain_has_a_degenerate_interval() -> None:
    a = {"a": (5, 10), "b": (3, 10)}
    b = {"a": (6, 10), "b": (4, 10)}
    result = paired_bootstrap(a, b)
    assert result.delta == pytest.approx(0.1)
    assert result.low == pytest.approx(0.1) and result.high == pytest.approx(0.1)


def test_the_interval_contains_the_estimate_and_is_reproducible() -> None:
    a = {f"s{i}": (i % 7, 10) for i in range(40)}
    b = {f"s{i}": ((i * 3) % 9, 10) for i in range(40)}
    first, second = paired_bootstrap(a, b, seed=1), paired_bootstrap(a, b, seed=1)
    assert first == second
    assert first.low <= first.delta <= first.high


def test_the_bootstrap_refuses_different_songs() -> None:
    with pytest.raises(ValueError, match="same songs"):
        paired_bootstrap({"a": (1, 2)}, {"b": (1, 2)})


def test_the_bootstrap_refuses_different_note_counts() -> None:
    with pytest.raises(ValueError, match="note counts"):
        paired_bootstrap({"a": (1, 2)}, {"a": (1, 3)})
```

And `tests/eval/test_recovery.py`:

```python
from __future__ import annotations

from tabsampler.eval.recovery import PartSequence, RecoveryReport, recover
from tabsampler.fingering.costs import HandSetScorer
from tabsampler.fingering.fit import HumanSequence
from tabsampler.types import ChordState, Context, NoteEvent, NoteGroup, Position, Tuning

CTX = Context(tuning=Tuning.STANDARD, max_span=5)


def one_note(pitch: int, human: Position) -> HumanSequence:
    group = NoteGroup.of([NoteEvent(onset=0.0, offset=0.4, pitch=pitch, confidence=1.0)])
    return HumanSequence((group,), (ChordState(positions=(human,)),), (5,))


def test_recovery_counts_hits_per_song_and_part() -> None:
    # Hand-set weights reward open strings, so the decoder plays E4 (64) on the open e.
    items = [
        PartSequence("song1", "clean", one_note(64, Position(5, 0))),  # recovered
        PartSequence("song1", "distorted", one_note(64, Position(4, 5))),  # not
        PartSequence("song2", "clean", one_note(64, Position(4, 5))),  # not
    ]
    report = recover(items, HandSetScorer(), CTX)
    assert report.counts() == (1, 3)
    assert report.counts("clean") == (1, 2)
    assert report.song_counts("clean") == {"song1": (1, 1), "song2": (0, 1)}
    assert report.song_counts("distorted") == {"song1": (0, 1)}
    assert report.chord_shape_rate() == 1.0


def test_a_report_survives_a_round_trip_through_plain_data() -> None:
    items = [PartSequence("s", "clean", one_note(64, Position(5, 0)))]
    report = recover(items, HandSetScorer(), CTX)
    assert RecoveryReport.from_dict(report.to_dict()) == report
```

- [x] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/eval/test_bootstrap.py tests/eval/test_recovery.py -q`
Expected: collection errors — `tabsampler.eval.bootstrap` and `tabsampler.eval.recovery` do
not exist.

- [x] **Step 3: Write `src/tabsampler/eval/bootstrap.py`**

```python
"""Song-level paired bootstrap: is one decoder better than another on the same songs?

Pure: no I/O, no global state. Notes inside one song are not independent -- a decoder that
mis-places a riff mis-places every repeat of it -- so the song is the unit resampled, and
both decoders are always scored on the same resampled songs (ADR 0028).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PairedDifference:
    """``b``'s recovery minus ``a``'s, pooled over notes, with a 95% bootstrap interval."""

    delta: float
    low: float
    high: float
    n_songs: int
    n_notes: int


def paired_bootstrap(
    a: Mapping[str, tuple[int, int]],
    b: Mapping[str, tuple[int, int]],
    n_resamples: int = 2000,
    seed: int = 0,
) -> PairedDifference:
    """Compare two decoders scored on the same songs: song -> (notes recovered, notes).

    Raises:
        ValueError: if the two cover different songs, or a song has different note counts --
            then they were not scored on the same notes and a paired comparison is void.
    """
    if set(a) != set(b):
        raise ValueError("a paired comparison needs the same songs on both sides")
    songs = sorted(a)
    if any(a[s][1] != b[s][1] for s in songs):
        raise ValueError("a song has different note counts on the two sides")
    hits_a = np.array([a[s][0] for s in songs], dtype=float)
    hits_b = np.array([b[s][0] for s in songs], dtype=float)
    notes = np.array([a[s][1] for s in songs], dtype=float)
    delta = float((hits_b.sum() - hits_a.sum()) / notes.sum())
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(songs), size=(n_resamples, len(songs)))
    deltas = (hits_b[picks].sum(axis=1) - hits_a[picks].sum(axis=1)) / notes[picks].sum(axis=1)
    low, high = (float(v) for v in np.percentile(deltas, [2.5, 97.5]))
    return PairedDifference(delta, low, high, len(songs), int(notes.sum()))
```

- [x] **Step 4: Write `src/tabsampler/eval/recovery.py`**

```python
"""How much of the human fingering a decoder recovers on held-out human tab (ADR 0023).

Pure: no I/O, no global state. Counts are kept per song and per part, so two decoders
scored on the same songs can be compared with :func:`tabsampler.eval.bootstrap.paired_bootstrap`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from tabsampler.decode.viterbi import viterbi
from tabsampler.eval.playability import PlayabilityRules, group_is_playable
from tabsampler.fingering.candidates import candidates
from tabsampler.fingering.fit import HumanSequence
from tabsampler.types import Context, FingeringScorer


@dataclass(frozen=True, slots=True)
class PartSequence:
    """One human sequence, its song, and whether its part is ``clean`` or ``distorted``."""

    song: str
    part: str
    sequence: HumanSequence


@dataclass(slots=True)
class RecoveryReport:
    """Per song and part, [notes placed where the human placed them, notes]; per part,
    [decoded shapes passing E3's chord rules, decoded shapes]."""

    per_song: dict[str, dict[str, list[int]]] = field(default_factory=dict)
    shapes: dict[str, list[int]] = field(default_factory=dict)
    single_candidate: int = 0

    def counts(self, part: str | None = None) -> tuple[int, int]:
        pairs = self.song_counts(part).values()
        return sum(h for h, _ in pairs), sum(n for _, n in pairs)

    def share(self, part: str | None = None) -> float:
        hits, notes = self.counts(part)
        return hits / notes if notes else math.nan

    def song_counts(self, part: str | None = None) -> dict[str, tuple[int, int]]:
        """song -> (recovered, notes), for one part or all parts pooled. A song with no
        notes in that part is left out rather than counted as zero of zero."""
        out: dict[str, tuple[int, int]] = {}
        for song, parts in self.per_song.items():
            chosen = [c for name, c in parts.items() if part is None or name == part]
            if chosen:
                out[song] = (sum(c[0] for c in chosen), sum(c[1] for c in chosen))
        return out

    def chord_shape_rate(self, part: str | None = None) -> float:
        chosen = [c for name, c in self.shapes.items() if part is None or name == part]
        total = sum(c[1] for c in chosen)
        return sum(c[0] for c in chosen) / total if total else math.nan

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_song": self.per_song,
            "shapes": self.shapes,
            "single_candidate": self.single_candidate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecoveryReport:
        return cls(
            per_song={
                song: {part: [int(c[0]), int(c[1])] for part, c in parts.items()}
                for song, parts in data["per_song"].items()
            },
            shapes={part: [int(c[0]), int(c[1])] for part, c in data["shapes"].items()},
            single_candidate=int(data["single_candidate"]),
        )


def recover(
    items: Sequence[PartSequence],
    scorer: FingeringScorer,
    ctx: Context,
    rules: PlayabilityRules | None = None,
) -> RecoveryReport:
    """Decode every sequence with Viterbi and count the notes placed as the human did."""
    active = rules or PlayabilityRules()
    report = RecoveryReport()
    for item in items:
        seq = item.sequence
        path, _ = viterbi(seq.groups, scorer, ctx, seq.spans)
        notes = report.per_song.setdefault(item.song, {}).setdefault(item.part, [0, 0])
        shapes = report.shapes.setdefault(item.part, [0, 0])
        for group, truth, guess in zip(seq.groups, seq.states, path, strict=True):
            shapes[0] += group_is_playable(guess.positions, active) is None
            shapes[1] += 1
            for note, t, g in zip(group.notes, truth.positions, guess.positions, strict=True):
                notes[0] += t == g
                notes[1] += 1
                report.single_candidate += len(candidates(note.pitch, ctx.tuning)) == 1
    return report
```

- [x] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/eval/test_bootstrap.py tests/eval/test_recovery.py -q`
Expected: 7 passed.

- [x] **Step 6: Write the two scripts and switch the fit script to `recover`**

`scripts/score_validation.py` — decodes the artist validation sample with one decoder
config, prints recovery by part, the decoded chord-shape rate and lattice nodes per state,
and writes the report as JSON:

```python
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
        PartSequence(track.song, "clean" if track.instrument.startswith("clean") else "distorted", s)
        for track in load_tracks(
            args.archive, Split.VALIDATION, args.meta, scheme=args.split,
            tuning=DADAGP_TUNING, sample=args.val_songs, seed=args.seed,
        )
        for s in human_sequences(track.steps, ctx)
    ]
    print(f"split scheme: {args.split}; {len(items)} sequences loaded in "
          f"{time.perf_counter() - started:.0f}s", flush=True)

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
        print(f"{part or 'all':9s}: recovery {report.share(part):.4f} ({hits} of {notes} notes)"
              f"   decoded chord shapes {report.chord_shape_rate(part):.4f}")
    print(f"lattice nodes per state: {nodes / states:.3f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"decoder": str(args.decoder_config), **report.to_dict()}))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
```

`scripts/compare_validation.py` — the paired comparison, by part:

```python
"""Paired, song-level comparison of two score_validation.py outputs (ADR 0028).

    uv run python scripts/compare_validation.py cache/validation/a.json cache/validation/b.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tabsampler.eval.bootstrap import paired_bootstrap
from tabsampler.eval.recovery import RecoveryReport


def main() -> None:
    a, b = (RecoveryReport.from_dict(json.loads(Path(p).read_text())) for p in sys.argv[1:3])
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
```

In `scripts/fit_cost_weights.py`, replace the local `recovery()` with `recover` over
`PartSequence`s and add `--per-song-out DIR`, which writes `DIR/hand-set.json` and
`DIR/fitted.json` in `score_validation.py`'s format. The printed lines keep their wording.

- [x] **Step 7: Check and commit**

Run: `make check` — Expected: exit 0.

```bash
git add src/tabsampler/eval/recovery.py src/tabsampler/eval/bootstrap.py \
    scripts/score_validation.py scripts/compare_validation.py scripts/fit_cost_weights.py \
    tests/eval/test_recovery.py tests/eval/test_bootstrap.py
git commit -m "Keep validation counts per song and compare decoders with a paired bootstrap"
```

---

### Task 3: E3 times a carried hand's move from the last fretted group (ADR 0029)

⚠️ Metric code.

**Files:**
- Create: `docs/adr/0029-e3-move-timing.md` (+ index row)
- Modify: `src/tabsampler/eval/playability.py`, `scripts/validate_playability.py`
- Test: `tests/eval/test_playability.py`

**Interfaces:**
- Produces: `hand_moves(shapes: Sequence[Shape]) -> list[tuple[int, float]]` (frets moved,
  seconds available, one per transition); `move_verdict(distance: int, seconds: float,
  rules: PlayabilityRules) -> str | None`; `judge_transitions(shapes: Sequence[Shape],
  rules: PlayabilityRules) -> list[str | None]`. Tasks 4 and 5 consume `hand_moves`.

- [x] **Step 1: Write ADR 0029 as proposed, with the rerun's prediction, and commit it**

Decision: a hand move is timed from the onset of the last group with a fretted note, the
last moment the hand was in place; an all-open group in between leaves the hand free and
does not restart the clock. Pre-registered rerun, same data as ADR 0025's gate (all 19,995
cleared songs of DadaGP's shipped training list, 16,732,524 transitions): **the pass rate
rises from 0.9795 but stays below 0.99**, because the fix only touches moves that cross an
all-open shape. Commit: `Pre-register the E3 move-timing fix (ADR 0029)`.

- [x] **Step 2: Write the failing tests**

```python
from hypothesis import given
from hypothesis import strategies as st

from tabsampler.eval.playability import judge_transitions


def test_a_move_across_an_open_string_is_timed_from_the_last_fretted_note() -> None:
    # fret 2 -> open -> fret 10 at 0.25 s steps: the hand had 0.5 s, not 0.25 s.
    tab = [tabnote(0.0, 0, 2), tabnote(0.25, 1, 0), tabnote(0.5, 0, 10)]
    report = playability_rate(tab)
    assert report.n_transitions_pass == 2


def test_a_piece_that_starts_with_open_strings_has_no_moves_to_time() -> None:
    shapes = [(0.0, pos((1, 0))), (0.1, pos((2, 0))), (0.2, pos((0, 5)))]
    assert judge_transitions(shapes, RULES) == [None, None]


@given(
    a=st.integers(1, 15),
    b=st.integers(1, 20),
    gap=st.floats(0.02, 1.0),
    cuts=st.lists(st.floats(0.01, 0.99), min_size=1, max_size=3),
)
def test_open_strings_between_two_fretted_shapes_never_change_the_verdict(
    a: int, b: int, gap: float, cuts: list[float]
) -> None:
    first, last = (0.0, pos((0, a))), (gap, pos((1, b)))
    opens = [(gap * c, pos((2, 0))) for c in sorted(cuts)]
    plain = judge_transitions([first, last], RULES)[-1]
    assert judge_transitions([first, *opens, last], RULES)[-1] == plain
```

- [x] **Step 3: Run them to verify they fail**

Run: `uv run pytest tests/eval/test_playability.py -q`
Expected: the first fails (1 of 2 transitions pass); the other two fail on the missing
`judge_transitions` import.

- [x] **Step 4: Implement in `playability.py`**

```python
def hand_moves(shapes: Sequence[Shape]) -> list[tuple[int, float]]:
    """For each transition, (frets the hand moves, seconds it has to move in).

    The hand is carried across all-open shapes (ADR 0018) and so is the clock: a move is
    timed from the last shape with a fretted note -- the last moment the hand was in place
    -- because an all-open shape in between leaves the hand free (ADR 0029).
    """
    moves: list[tuple[int, float]] = []
    hand: int | None = None
    placed_at: float | None = None
    for index, (onset, positions) in enumerate(shapes):
        fretted = fretted_frets(positions)
        hand, distance = shift_window(hand, fretted, HAND_WINDOW)
        if index > 0:
            since = shapes[index - 1][0] if placed_at is None else placed_at
            moves.append((distance, onset - since))
        if fretted:
            placed_at = onset
    return moves


def move_verdict(distance: int, seconds: float, rules: PlayabilityRules) -> str | None:
    """None if the hand can move ``distance`` frets in ``seconds``, else the reason."""
    if distance == 0:
        return None
    if seconds <= 0.0:
        return f"{distance}-fret jump with no time between groups"
    speed = distance / seconds
    if speed > rules.max_frets_per_second:
        return f"{speed:.1f} frets/s exceeds {rules.max_frets_per_second}"
    return None


def judge_transitions(shapes: Sequence[Shape], rules: PlayabilityRules) -> list[str | None]:
    """E3's verdict on every transition of a piece: None, or why it is not playable."""
    return [move_verdict(d, s, rules) for d, s in hand_moves(shapes)]
```

`hand_move_is_playable` keeps its signature and becomes `shift_window` plus `move_verdict`.
`playability_rate` replaces its transition loop with `judge_transitions(shapes, active)`,
pairing each verdict with `shapes[1:]` for the failure text. In
`scripts/validate_playability.py`, the per-track transition loop becomes
`judge_transitions([(g.onset, s.positions) for g, s in track.steps], rules)`, and the
printed label reads `E3 transitions, hand window timed from the last fretted group (ADR 0029)`.

- [x] **Step 5: Run the tests, then the whole suite**

Run: `uv run pytest tests/eval/test_playability.py -q` — Expected: all pass.
Run: `make check` — Expected: exit 0. Commit:
`Time E3 hand moves from the last fretted group (ADR 0029)`.

- [x] **Step 6: Rerun the human-tab check, record it, accept the ADR**

Run (background): `caffeinate -i uv run python -u scripts/validate_playability.py
data/dadagp/DadaGP-v1.1.zip data/dadagp/track_meta.json`
Expected: chord-shape rate unchanged at 0.9986; the transition rate is the result. Add a
results row (commit = the code commit; E3 = chord-shape rate; transition rate in notes),
accept ADR 0029 with a Result section, commit: `Record the E3 timing fix's effect on human tab`.

---

### Task 4: A chord wider than the window stretches the hand (ADR 0030)

⚠️ Metric code (E3 uses the same hand) and a contract change (`transition_cost_from`).

**Files:**
- Create: `docs/adr/0030-hand-stretch.md` (+ index row; ADR 0025's status line points to it)
- Modify: `src/tabsampler/types.py` (`Hand`, the Protocol), `fingering/states.py`,
  `fingering/costs.py`, `decode/viterbi.py`, `fingering/fit.py`, `eval/playability.py`,
  `scripts/validate_playability.py`, `tests/decode/brute_force.py`
- Test: `tests/fingering/test_states.py`, and the existing tests whose hand values change

**Interfaces:**
- Produces: `Hand = tuple[int, int]` in `types.py` — (lowest, highest) fret covered;
  `shift_window(previous: Hand | None, fretted: Sequence[int], window: int = HAND_WINDOW)
  -> tuple[Hand | None, int]`; `carry_hand(previous: Hand | None, state) -> Hand | None`;
  `LatticeNode.carried_hand: Hand | None`; `transition_cost_from(previous_hand: Hand | None,
  curr)`.

- [x] **Step 1: Score the current decoder per song** (the paired baseline)

Run: `caffeinate -i uv run python -u scripts/score_validation.py data/dadagp/DadaGP-v1.1.zip
data/dadagp/track_meta.json --decoder-config configs/phase1_baseline.yaml --split artist
--out cache/validation/window.json`
Expected: recovery 0.6379 overall, clean 0.8257, distorted 0.5705 (as Task 6 of the previous
plan measured).

- [x] **Step 2: Write ADR 0030 as proposed and commit it**

Decision, written so the oracle can be built from it: the hand covers a fret range, at rest
`h` to `h + 4`. A fretted shape inside the covered range costs nothing and leaves the index
where it was. Otherwise the index moves to the start, among those whose rest window covers
the shape, closest to where it was; a shape wider than the rest window can only be covered
stretched, from its own lowest fret. After every fretted shape the hand covers from its
index to the higher of index + 4 and the shape's highest fret — stretched only while a
shape needs it. An all-open shape leaves it as it was; the first fretted shape places it
for free. Movement is the distance the index moves. Pre-registered check, tolerance stated
in advance: **hand-set recovery changes by less than 0.001 on clean and on distorted
artist-validation parts, each 95% paired interval inside ±0.002, and human tab's E3
transition pass rate does not fall.** Keep rule: the stretch stays unless a part's recovery
falls by more than 0.001 with its interval wholly below zero. Commit:
`Pre-register the hand stretch (ADR 0030)`.

- [x] **Step 3: Write the failing tests** (`tests/fingering/test_states.py`)

```python
def test_a_wide_chord_stretches_the_hand_over_its_own_span() -> None:
    assert shift_window((5, 9), [12, 17]) == ((12, 17), 7)


def test_alternating_a_wide_chord_with_its_own_notes_never_moves_the_hand() -> None:
    hand, moved = shift_window(None, [12, 17])
    total = 0
    for fretted in ([17], [12, 17], [12], [12, 17], [17]):
        hand, moved = shift_window(hand, fretted)
        total += moved
    assert total == 0


def test_the_stretch_relaxes_once_no_shape_needs_it() -> None:
    hand, _ = shift_window((12, 17), [13])
    assert hand == (12, 16)
    assert shift_window(hand, [17]) == ((13, 17), 1)


def test_the_stretch_is_carried_across_an_all_open_shape() -> None:
    hand, moved = shift_window((12, 17), [])
    assert (hand, moved) == ((12, 17), 0)
    assert shift_window(hand, [17]) == ((12, 17), 0)
```

- [x] **Step 4: Run them to verify they fail**

Run: `uv run pytest tests/fingering/test_states.py -q`
Expected: the four new tests fail (the hand is still an int).

- [x] **Step 5: Rewrite the oracle's movement from ADR 0030's text, before the code**

In `tests/decode/brute_force.py`, `path_cost` keeps `hand` as `(start, end)` and, for a
shape not inside it, searches the candidate starts literally:

```python
    hand: tuple[int, int] | None = None  # (lowest, highest) fret the hand covers
    for state in path:
        fretted = sorted(p.fret for p in state.positions if p.fret > 0)
        if not fretted:
            continue  # open strings need no hand; it stays as it was, stretch and all
        low, high = fretted[0], fretted[-1]
        if hand is None:
            hand = (low, max(low + 4, high))  # the first fretted shape places it, free
            continue
        start, end = hand
        if start <= low and high <= end:
            new_start = start  # inside what the hand covers: a finger reaches
        elif high - low > 4:
            new_start = low  # only a stretch covers it, from its own lowest fret
        else:
            covering = range(high - 4, low + 1)  # every rest window that holds the shape
            new_start = min(covering, key=lambda s: abs(s - start))
        total += scorer.weights.move * abs(new_start - start)  # type: ignore[attr-defined]
        hand = (new_start, max(new_start + 4, high))
```

- [x] **Step 6: Implement**

`types.py`: `Hand = tuple[int, int]`, documented as the fret range a hand covers; the
Protocol's `transition_cost_from(self, previous_hand: Hand | None, curr: ChordState)`.

`states.py`:

```python
def shift_window(
    previous: Hand | None, fretted: Sequence[int], window: int = HAND_WINDOW
) -> tuple[Hand | None, int]:
    """The frets the hand covers after a shape, and how far its index moved (ADR 0030)."""
    if not fretted:
        return previous, 0
    low, high = fretted[0], fretted[-1]
    if previous is None:
        return (low, max(low + window, high)), 0
    start, end = previous
    if start <= low and high <= end:
        new_start = start
    elif low < start or high - low > window:
        new_start = low
    else:
        new_start = high - window
    return (new_start, max(new_start + window, high)), abs(new_start - start)
```

`carry_hand` returns `Hand | None`. `costs.transition_cost` becomes
`self.transition_cost_from(carry_hand(None, prev), curr)`. `LatticeNode.carried_hand`,
`build_lattice`'s `carried`, `fit.path_features`'s `hand`, and `playability.hand_moves` /
`hand_move_is_playable` take `Hand | None`. Update the existing tests whose expected hands
were ints (`test_states`, `test_costs`, `test_viterbi`, `test_forward_backward`,
`test_playability`, `test_types`) to the `(start, end)` pairs, each value derived by hand.

- [x] **Step 7: Run the oracle and the suite**

Run: `make oracle` — Expected: all oracle tests pass, including the span-6 ones.
Run: `make check` — Expected: exit 0. Commit: `Let a wide chord stretch the hand (ADR 0030)`.

- [x] **Step 8: Measure, compare, record**

Run `score_validation.py` again into `cache/validation/stretch.json`, then
`scripts/compare_validation.py cache/validation/window.json cache/validation/stretch.json`,
then `scripts/validate_playability.py` (background). Record results rows, the lattice
nodes per state, and `scripts/bench_decode.py`; accept ADR 0030 with the result. Commit:
`Record the hand stretch's validation check`.

---

### Task 5: E3's speed limit, set from human tab and checked on unseen artists (ADR 0031)

⚠️ Metric code if adopted.

**Files:**
- Create: `src/tabsampler/eval/speed.py`, `scripts/estimate_speed_limit.py`,
  `docs/adr/0031-e3-speed-limit.md` (+ index row; ADR 0011's and ADR 0022's status lines)
- Test: `tests/eval/test_speed.py`

**Interfaces:**
- Consumes: `hand_moves` (Task 3).
- Produces: `speed_limit_for(pass_rate: float, speeds: Sequence[float], free: int,
  transitions: int) -> float`.

- [x] **Step 1: Write ADR 0031 as proposed and commit it**

Hypothesis: with the window, the timing fix and the stretch, human hand moves have an upper
speed that generalises across artists. Method, fixed now: on every cleared song of the
artist-split **training** side, find the smallest limit under which **0.9986** of
transitions pass — the rate at which ADR 0011's chord rules hold on human tab, so both
halves of E3 are equally strict — and round it **up** to a whole number of frets per second.
Check, on every cleared song of the **validation** side: at least **0.9976** of transitions
pass (0.1 point of tolerance). If it holds, E3's limit becomes that number and its
transition rate may be quoted as "no faster than 99.86% of human moves", lifting ADR 0022's
caveat; if not, 12 frets/s and the caveat stay. Supersedes ADR 0011's 12 frets/s and ADR
0022's rejection of raising the limit: that rejection rested on a point model of the hand,
which is gone. Commit: `Pre-register the E3 speed-limit experiment (ADR 0031)`.

- [x] **Step 2: Write the failing tests** (`tests/eval/test_speed.py`)

```python
import pytest

from tabsampler.eval.speed import speed_limit_for


def test_the_limit_is_the_slowest_speed_that_reaches_the_pass_rate() -> None:
    # 10 transitions, 5 free, moves at 1..5 frets/s: 90% needs 4 of the moves to pass.
    assert speed_limit_for(0.9, [5.0, 1.0, 4.0, 2.0, 3.0], free=5, transitions=10) == 4.0


def test_free_transitions_alone_can_reach_the_rate() -> None:
    assert speed_limit_for(0.5, [9.0], free=5, transitions=10) == 0.0


def test_a_rate_no_finite_limit_reaches_is_refused() -> None:
    # One zero-time move fails at any limit, so 100% is out of reach.
    with pytest.raises(ValueError, match="no finite"):
        speed_limit_for(1.0, [1.0, 2.0, 3.0, 4.0], free=5, transitions=10)
```

- [x] **Step 3: Run them to verify they fail** — Expected: `tabsampler.eval.speed` missing.

- [x] **Step 4: Implement `src/tabsampler/eval/speed.py`**

```python
"""The speed limit under which a given share of human hand moves pass (ADR 0031).

Pure: no I/O, no global state.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def speed_limit_for(pass_rate: float, speeds: Sequence[float], free: int, transitions: int) -> float:
    """The smallest limit, in frets per second, under which ``pass_rate`` of ``transitions`` pass.

    ``speeds`` are the moves that take time; ``free`` counts transitions with no move, which
    pass at any limit. A move with no time between groups is in neither and fails at any limit.

    Raises:
        ValueError: if no finite limit reaches ``pass_rate``.
    """
    needed = math.ceil(pass_rate * transitions) - free
    if needed <= 0:
        return 0.0
    if needed > len(speeds):
        raise ValueError(f"no finite limit passes {pass_rate} of {transitions} transitions")
    return sorted(speeds)[needed - 1]
```

- [x] **Step 5: Run the tests** — Expected: 3 passed. `make check` — exit 0.

- [x] **Step 6: Write `scripts/estimate_speed_limit.py` and commit**

For each side of the artist split, every cleared song: per track,
`hand_moves([(g.onset, s.positions) for g, s in track.steps])`; count transitions, free
transitions (`distance == 0`), zero-time moves; collect `distance / seconds` for the rest.
Print: the training limit (raw and rounded up), the training pass rate at 12 frets/s and
at the rounded limit, the validation pass rate at the rounded limit, and the distances of
the moves that still fail. Commit: `Add the speed-limit estimate (ADR 0031)`.

- [x] **Step 7: Run it, decide by the rule, record**

Run (background): `caffeinate -i uv run python -u scripts/estimate_speed_limit.py
data/dadagp/DadaGP-v1.1.zip data/dadagp/track_meta.json`.
If the check holds: set `PlayabilityRules.max_frets_per_second` to the rounded limit, with a
test pinning the value and a comment citing ADR 0031. Results rows either way; accept ADR
0031 with its result. Commit: `Record the speed-limit experiment (ADR 0031)`.

---

### Task 6: Clean and distorted guitar get their own fitted weights (ADR 0032)

**Files:**
- Create: `docs/adr/0032-style-decoders.md` (+ index row), `configs/decoder_clean.yaml`,
  `configs/decoder_distorted.yaml`
- Modify: `scripts/fit_cost_weights.py` (`--part {all,clean,distorted}`),
  `src/tabsampler/cli.py` (default decoder config)

- [x] **Step 1: Write ADR 0032 as proposed and commit it**

Hypothesis: fitted on one style's training parts alone, the four weights beat the hand-set
ones on that style's validation parts — the fits so far lost on clean parts because three
quarters of their training notes were distorted. Fair test per style, fixed now: the
style's fitted weights replace the hand-set weights *for that style* iff, on that style's
artist-validation parts, (a) recovery is higher with its 95% paired interval above zero
and (b) the decoded chord-shape rate is not lower by more than 0.0005. Product decision,
fixed now: each style gets a decoder config, `configs/decoder_clean.yaml` and
`configs/decoder_distorted.yaml`, holding whichever weights won; the CLI's default becomes
the clean one, because clean or acoustic guitar is the typical recording (and GuitarSet's);
`configs/phase1_baseline.yaml` stays as the record of the hand-set baseline. Commit:
`Pre-register the style decoders (ADR 0032)`.

- [x] **Step 2: Add `--part`** — training sequences are filtered to that part; validation
  is still scored on all parts. Commit: `Fit the cost weights on one style's parts`.

- [x] **Step 3: Run the two fits** (background, one at a time):

`caffeinate -i uv run python -u scripts/fit_cost_weights.py data/dadagp/DadaGP-v1.1.zip
data/dadagp/track_meta.json --split artist --part clean --skip-halves
--per-song-out cache/validation/fit-clean` and the same with `--part distorted`.

- [x] **Step 4: Apply the fair test** with `compare_validation.py` on the style's part
  (hand-set.json against fitted.json). Write both decoder configs (temperature 2.9974 until
  Task 7), point the CLI's default `--decoder-config`/`--config` at
  `configs/decoder_clean.yaml`, results rows, accept ADR 0032. Commit:
  `Record the style fits and add a decoder per style (ADR 0032)`.

---

### Task 7: A temperature per style (ADR 0033)

**Files:**
- Create: `docs/adr/0033-per-style-temperature.md` (+ index row)
- Modify: `scripts/calibrate_temperature.py` (`--part`), both decoder configs

- [x] **Step 1: Write ADR 0033 as proposed and commit it** — Hypothesis: the styles need
  different temperatures, and clean parts, where the decoder recovers far more of the human
  fingering, need a **lower** one than distorted parts. Calibrating each style's decoder on
  its own validation parts lowers that style's calibration error against T = 2.9974.

- [x] **Step 2: Add `--part`** (validation sequences filtered to that part). Commit.

- [x] **Step 3: Run both calibrations** (background), write each T into its config, results
  rows, accept ADR 0033. Commit: `Calibrate a temperature per style (ADR 0033)`.

---

### Task 8: C3 feature group 1 — a per-string preference (ADR 0034)

Contract change: `CostWeights` gains `string_bias` (six floats, low E first, default zeros)
and, for Task 9, `low_region` and `high_region` (default 0).

**Files:**
- Create: `docs/adr/0034-c3-feature-groups.md` (+ index row)
- Modify: `src/tabsampler/types.py`, `src/tabsampler/config.py`,
  `src/tabsampler/fingering/costs.py`, `src/tabsampler/fingering/fit.py`,
  `src/tabsampler/results.py` (`describe_weights` lists non-zero new weights),
  `scripts/fit_cost_weights.py` (`--features`)
- Test: `tests/fingering/test_fit.py`, `tests/fingering/test_costs.py`, `tests/test_config.py`

**Interfaces:**
- Produces: `WEIGHT_NAMES` with 11 entries — `move, span, high, open_reward, string_1 ..
  string_5, low_region, high_region` (string 0, the low E, is the fixed reference);
  `FEATURE_GROUPS = {"base": ..., "string": ..., "region": ...}`;
  `fit_weights(sequences, initial, active: Sequence[str] = FEATURE_GROUPS["base"])`.

- [x] **Step 1: Write ADR 0034 as proposed and commit it** — the two feature groups, why
  each could help, their identifiability (string 0 and the middle region 5–11 are the
  references), and the rule, fixed now: a group is kept for a style iff its refit beats the
  same style's refit without it on that style's validation parts, with the 95% paired
  interval above zero and the chord-shape rate not lower by more than 0.0005. A kept model
  then replaces the style's decoder only by ADR 0032's fair test against it.

- [x] **Step 2: Write the failing tests**

```python
def test_an_old_config_loads_with_the_new_weights_at_zero() -> None:
    weights = load_phase1_config(CONFIGS / "phase1_baseline.yaml").weights
    assert weights.string_bias == (0.0,) * 6
    assert (weights.low_region, weights.high_region) == (0.0, 0.0)


def test_a_string_bias_costs_each_note_on_that_string() -> None:
    state = ChordState(positions=(Position(1, 2), Position(2, 2)))
    group = NoteGroup.of([n(47), n(52)])
    base = HandSetScorer().emission_cost(group, state, CTX)
    biased = HandSetScorer(weights=CostWeights(string_bias=(0, 1.5, 0.25, 0, 0, 0)))
    assert biased.emission_cost(group, state, CTX) == pytest.approx(base + 1.75)


def test_fitting_a_subset_leaves_the_other_weights_untouched() -> None:
    result = fit_weights(FEATS, CostWeights(), active=FEATURE_GROUPS["base"])
    assert result.weights.string_bias == (0.0,) * 6
    assert (result.weights.low_region, result.weights.high_region) == (0.0, 0.0)
```

plus a hypothesis test that `weights_to_vector(w) @ path_features(path)` equals the oracle's
`path_cost` for random weights with every group non-zero, and the finite-difference
gradient test extended to all 11 columns.

- [x] **Step 3: Run them to verify they fail** — Expected: `CostWeights` has no
  `string_bias`; `fit_weights` has no `active`.

- [x] **Step 4: Implement** — `CostWeights` fields with a length check on `string_bias`;
  the loader accepts `string_bias` (a list of six numbers), `low_region`, `high_region`;
  `HandSetScorer.emission_cost` adds `sum(w.string_bias[p.string] for p in positions)`,
  `w.low_region * (fretted notes at frets 1-4)` and `w.high_region * (fretted notes at fret
  12 or above)`; `fit._shape_features` returns the 11 columns; `nll_and_gradient` sizes by
  `len(WEIGHT_NAMES)`; `fit_weights` optimises only the `active` columns:

```python
def fit_weights(
    sequences: Sequence[SequenceFeatures],
    initial: CostWeights,
    active: Sequence[str] = FEATURE_GROUPS["base"],
) -> FitResult:
    mask = np.array([name in active for name in WEIGHT_NAMES])
    start = weights_to_vector(initial)

    def objective(x: Vector) -> tuple[float, Vector]:
        w = start.copy()
        w[mask] = x
        nll, gradient = nll_and_gradient(w, sequences)
        return nll, gradient[mask]

    x, iterations, converged = _minimise(objective, start[mask])
    w = start.copy()
    w[mask] = x
    nll, gradient = nll_and_gradient(w, sequences)
    return FitResult(
        weights=weights_from_vector(w, temperature=initial.temperature),
        nll=nll,
        n_groups=sum(s.n_groups for s in sequences),
        iterations=iterations,
        converged=converged,
        gradient_norm=float(np.linalg.norm(gradient[mask])),
    )
```

- [x] **Step 5: Run the tests, the oracle and the suite** — all pass; `make check` exit 0.
  Commit: `Add per-string and fret-region weights to the cost model (ADR 0034)`.

- [x] **Step 6: Run the string-group fits** — `--features base,string` with `--part clean`
  and `--part distorted`, per-song outputs to `cache/validation/fit-<part>-string`; compare
  each against Task 6's `fitted.json` for the same part. Record rows; note the result in
  the plan. Commit: `Record the per-string feature experiment`.

---

### Task 9: C3 feature group 2 — fret regions

- [x] **Step 1: Run the region-group fits** — `--features base,region` (plus `string` for a
  style that kept it in Task 8), both parts, compared against that style's best model so
  far by ADR 0034's rule. Record rows. Commit: `Record the fret-region feature experiment`.

- [x] **Step 2: Consolidate** — for each style, the best kept model faces ADR 0032's fair test
  against the style's decoder; a winner is written to the style's config and its temperature
  recalibrated (Task 7's script). Accept ADR 0034 with all results. Commit:
  `Settle the style decoders after the first C3 features`.

---

### Task 10: One GuitarSet measurement of the new default

⚠️ Reads the test set — once, logged, nothing chosen from it.

- [x] **Step 1: Pre-register** `configs/m2_style_eval.yaml` — hypothesis, single variable
  (the default decoder as this plan leaves it), predictions from validation only, and every
  guardrail of ADR 0016. Write ADR 0035 as proposed: under ADR 0029–0031's rule, ADR 0016's
  transition guardrail has no baseline, and this run sets it. Commit both.

- [x] **Step 2: Run** `uv run tabsampler eval-m1 --config configs/m2_style_eval.yaml
  --decoder-config configs/decoder_clean.yaml`.

- [x] **Step 3: Refresh the README results in the same commit** — previous default kept for
  comparison, every guardrail stated including any breach; accept ADR 0035 with the numbers.

---

### Task 11: Close out

- [x] Devlog `docs/devlog/2026-10-03.md`; `HANDOFF.md`; Phase 2 plan boxes (C3 progress, C5);
  check that no local tooling files are tracked.
- [x] `make check`, `make oracle`; independent review of the whole branch; one fix pass;
  then merge into `main` and push. *(Merged locally; the push waits until the default is
  re-decided on a held-out GuitarSet player, Ege's decision of 2026-10-03.)*

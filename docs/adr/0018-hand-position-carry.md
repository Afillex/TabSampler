# ADR 0018: Carry the hand position across all-open shapes

Status: accepted (2026-09-27); its definition of hand position (the lowest fretted fret) is
superseded by [ADR 0025](0025-hand-window.md)'s 4-fret window. The carry and the lattice design stand.

Closes the first known defect from M1 (`docs/devlog/2026-09-27-m1-retrospective.md`).
Extends the `FingeringScorer` contract from ADR 0007 / spec 2.1.

## Context

Hand position is the lowest fretted fret (spec 2.2). An **all-open** shape has none, so
the stateless `transition_cost(prev, curr)` the spec names returns 0.0 whenever either
side is all open. M1 shipped with the consequence: a passage going fret 2 → open chord →
fret 10 was charged **no** hand movement, though the hand really travelled 8 frets.
`fingering/costs.py` documented it; the README listed it as a v1 limitation.

Leaving it implicit is not neutral. It is not a small under-charge in a rare case: with
movement free across open strings, the *emission* term decides everything around an open
shape, and the emission term rewards open strings and low frets. M1's plausible-looking
open-position fingerings were partly an artefact of the defect rather than of the model.

The obvious fix — putting a hand position into every lattice state — multiplies the state
space by `max_fret + 1 = 23`. Viterbi is `O(T · S²)` and the measured mean is 4.92 states
per group, so that takes transition work from about 24 to about 12 800 per step: a 530×
regression to fix a rare case.

## Decision

**Carry the hand position, and augment the lattice only where it is needed.**

1. A decoder state is a `LatticeNode`: a `ChordState` plus the `carried_hand` in force
   while it is played. A shape with any fretted note *determines* its own hand position
   and so has exactly **one** node. An all-open shape gets one node per distinct hand
   position reachable at the previous level. Nothing precedes group 0, so an all-open
   opening shape carries `None` and the first group's cost is unchanged from M1.
2. A transition between nodes is legal only when the successor's `carried_hand` is
   exactly what the predecessor's hand becomes after playing that shape
   (`fingering.states.carry_hand`). Anything else costs `inf`, which is a `-inf`
   log-potential and zero weight in forward-backward.
3. The `FingeringScorer` Protocol gains
   `transition_cost_from(previous_hand: int | None, curr: ChordState) -> float`. This is
   what the decoder calls. `transition_cost(prev, curr)` stays, defined as
   `transition_cost_from(prev.hand_position, curr)`, so the two cannot drift apart.
4. **The carry is not the scorer's to choose.** Where the hand ends up is spec 2.2
   geometry that every scorer must agree on, and the lattice's correctness depends on it,
   so `carry_hand` is a free function in `fingering/states.py` and is deliberately *not*
   on the Protocol. `HandSetScorer.carry` is a documented delegate for callers holding a
   scorer.
5. `tests/decode/brute_force.py` folds the carry through a path **spelled out inline**,
   not imported from `carry_hand`, so the oracle stays an independent check on it.

Passing a hand *position* rather than a previous *shape* is the substantive part of the
contract change: it states in the signature that only the hand position is carried. A
Phase 2 learned scorer whose transition cost needs more than that has to say so, rather
than silently receiving a shape that does not exist.

## Measured cost

Reproduce with `uv run python scripts/measure_lattice.py [--guitarset 60]`. Nodes per group
against states per group, at `max_span = 5`:

| input | states/group | nodes/group | ratio |
|---|---|---|---|
| GuitarSet reference notes, 60 tracks, 6607 groups | 4.716 | 5.054 | **1.072×** |
| GuitarSet transcriber notes, 60 tracks, 6493 groups | 4.904 | 5.413 | **1.104×** |
| E-minor pentatonic run (synthetic) | 3.500 | 4.562 | 1.30× |
| chromatic cycle from A2, 120 groups | 2.750 | 3.225 | 1.17× |
| **open-string pitches only, 120 groups (worst case)** | 3.333 | 10.092 | **3.03×** |

A level fans out only where a group *can* be played all-open, so the worst input is a line
of nothing but open-string pitches — the only pitches with an all-open state. That is not
music, and the ratio there is bounded rather than growing with the piece: a level can carry
only hand positions earlier groups actually put the hand in, so the carried set **saturates**
at the distinct fretted hand positions those pitches reach. In standard tuning that is
`{4, 5, 9, 10, 14, 15, 19}` plus `None` — eight values, measured, and the same for a
20-group line as a 200-group one. A test pins the saturation.

Against the 23× of full augmentation, 1.07× on real playing is free and even the
pathological 3.03× is cheap. Both GuitarSet rows are complexity instrumentation: no metric
was computed from them and no threshold was chosen from them (ADR 0003), and the access is
logged in `experiments/test_set_access.log`.

> **Correction.** An earlier draft of this ADR gave the worst case as 1.74×, measured on a
> line that mixed open-string pitches with ordinary ones and so diluted the effect. The
> row above is the real bound. Both figures came from runs that executed; the first was
> the wrong input for the claim it supported.

## Alternatives considered

- **Augment every node with a hand position.** Rejected on the 530× arithmetic above.
- **Carry the last fretted *shape* instead of an integer, and keep calling
  `transition_cost(prev, curr)`.** This needs no contract change at all, and was
  rejected for that reason: it works only because `HandSetScorer.transition_cost`
  happens to depend on nothing but `prev.hand_position`. Deduplicating nodes by hand
  position, and picking an arbitrary representative shape, both silently assume it. The
  assumption is the same either way; `transition_cost_from` is the version that states
  it where a future scorer will see it.
- **Reset the hand to fret 0 after an all-open shape.** Rejected: it is simply false.
  A guitarist playing an open chord in the middle of a phrase at fret 10 does not move.
  ADR 0011 already says the hand carries forward for the E3 metric.
- **Leave it, and note it in the README.** That is what M1 did. Rejected now because the
  defect was not inert: see below.

## Consequences

**Easier.** Movement is charged for what the hand does rather than for what the shapes
look like. Headline E2 on GuitarSet rose from 0.6366 to **0.6599** in oracle mode and
from 0.4231 to **0.4318** end to end — the decoder agrees more with human fingerings.

**Harder.** Two nodes can hold the same shape, so anything reading per-shape quantities
out of `forward_backward` must sum over nodes. `note_posteriors` does, which
marginalises the carried hand out again.

Decode is slower, benchmarked directly rather than read off E7, with
`scripts/bench_decode.py` run once per commit against its own source (the script's
docstring gives the worktree invocation). 500 single-note groups, warmed, median of seven:

| | 057deb8 | this commit | |
|---|---|---|---|
| `viterbi` | 5.5 ms | 7.2 ms | +31% |
| `decode` (Viterbi + forward-backward) | 53.8 ms | 57.9 ms | +8% |

Against spec 4's 60 s budget for a 3-minute file this is nothing. **E7 in
`experiments/results.csv` cannot be used for this comparison**: `eval-m1` measured 0.0370,
0.17, 0.20 and 0.04 s per audio minute for equivalent decode work on this machine within
one afternoon, so its noise floor is wider than the effect. An earlier draft of this ADR
quoted "roughly 20%" from two of those readings; the benchmark above replaces it, and the
script exists so the next person does not have to trust either number.

**What we accept as a cost, and it is the important part.** The fix made the *weights*
visibly wrong. `move` is 1.0 per fret while `high` is 0.1 per **octave** — a ratio of
120:1 — so now that movement is charged honestly it dominates every other term, and the
decoder pre-positions the hand in the middle of the range the phrase will visit rather
than starting in open position. On the golden clip the phrase 52-55-57-59-60-64-67-72
moves from open position to frets 8–12: a genuine minimiser at 3.667 against 7.383 for
the open-position fingering a guitarist would actually use.

This is **not** repaired by reweighting. Phase 1 has no legal validation data (ADR 0012),
and adjusting `move` or `high` to make one hand-picked clip look better would be
selection on an example. E2 rose, so the corpus does not agree that the new output is
worse. The honest reading is that both the old and the new fingerings are artefacts of
untuned weights, and that this is now visible instead of hidden.

**Revisit** when DadaGP/ProgGP arrive and cost-weight tuning is legal. The `move : high`
ratio is the first thing to fit; the prediction to test is that `high` rises sharply.

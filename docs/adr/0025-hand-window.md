# ADR 0025: The hand is a 4-fret window, not a point

Status: accepted (2026-10-02) — **its acceptance gate was missed, and Ege accepted it anyway,
knowingly; both are recorded below**

Implements the fix ADR 0022 proposed. Supersedes **ADR 0018's definition of hand position**
(the lowest fretted fret) while keeping its lattice design — a node is still a shape plus the
hand carried into it — and supersedes **ADR 0011's transition rule** for E3.

## Context

ADR 0022 found that defining the hand as a point — the lowest fretted fret — counts a finger
reaching inside one position as the hand moving. Human tab failed E3's speed rule on 11.7% of
moves for that reason, and the cost model's movement term charged the same phantom movement,
so a melody climbing frets 5, 7, 9 paid for two hand shifts that never happened.

## Decision

**The hand covers frets `h` to `h + 4`** — four frets in span units, the same reach ADR 0011's
chord rule allows below fret 12, fixed by decision rather than fitted. One pure function,
`fingering.states.shift_window`, defines it, and everything that reasons about the hand uses
it: E3's transition rule, the cost model's movement term, the decoder's lattice and the CRF
fitter's features.

- The first fretted shape places the window at its lowest fret, for free.
- A shape whose fretted notes all lie inside the window costs no movement.
- Otherwise the window moves by the least distance that brings the shape inside it — down to
  the lowest note, or up until the highest fits. A shape wider than the window anchors at its
  lowest fret. Movement costs `move` per fret the window moves.
- An all-open shape leaves the window where it was (ADR 0018's carry, unchanged).

Because the window a shape is played in now depends on where the hand came from for *every*
shape, not only all-open ones, each lattice level gets one node per distinct reachable window.

The brute-force oracle (`tests/decode/brute_force.py`) was rewritten from this text first,
not from the code, and does not import `shift_window`. Viterbi, forward-backward and the
fitter all agree with it.

One consequence worth knowing: moves are not symmetric. From a hand at fret 3 (frets 3–7),
reaching fret 9 moves the window up 2; from fret 9 (frets 9–13), reaching fret 3 moves it
down 6, because a hand is placed at its lowest note.

## The acceptance gate, and what happened to it

Fixed in advance, from ADR 0022: **human tab must pass the new transition rule at ≥ 0.99.**

| on 16,732,524 human moves (all cleared DadaGP training songs) | pass rate |
|---|---|
| point model, as implemented (ADR 0022) | 0.8835 |
| **hand window (this ADR)** | **0.9795** |

**The gate was missed.** The window removed 82% of the failures, but 2.05% remain against a
bar of 1%.

An exploratory breakdown on 2,000 songs says what is left: 47.7% of the remaining failures
are window shifts of only one or two frets, 55.2% are between single notes, and 64.8% are at
24 frets per second or less — at most twice the 12 frets-per-second limit. That is mostly
small, quick shifts that the **speed limit** calls impossible, not finger reach the window
misreads. The gate tested the window and the speed limit together; the decoder uses only
the window, and the speed limit appears nowhere in the cost model.

**Ege's decision (2026-10-02): keep the window, fix the speed limit later.** The window
stays in the decoder and in E3; the miss stays on record here and in `results.csv`; E3's
transition rate keeps ADR 0022's caveat — not a playability measure — until a separate,
pre-registered experiment settles the speed limit. Neither the window nor the limit was
tuned to make the gate pass.

## Measured cost

| | point model | hand window |
|---|---|---|
| lattice nodes per state, GuitarSet reference notes (60 tracks) | 1.072× | **2.569×** |
| E-minor pentatonic run / open-string line (synthetic) | 1.30× / 3.03× | 3.61× / 4.60× |
| `viterbi`, 500 groups (median of 7) | 7.2 ms | 32.6 ms |
| `decode`, 500 groups (median of 7) | 58.3 ms | 113.7 ms |
| lattice building for fitting, per group | 42 µs | 150 µs |
| NLL and gradient, per group | 57 µs | 64 µs |

The carried set still saturates — ten distinct windows for a 48-group line and for a
96-group one. A 3-minute file decoded in 0.11 s under the point model; twice that is still
nothing against spec §4's 60-second budget. Reproduce with `scripts/measure_lattice.py` and
`scripts/bench_decode.py`.

## What it does to a fingering

The golden clip — 52-55-57-59-60-64-67-72 — moves from frets 12, 0, 12, 0, 10, 0, 8, 8 (the
point model shifting 12 → 10 → 8) to **7, 0, 7, 0, 10, 0, 8, 8: all in 7th position, with no
hand shift at all.** Every fretted note sits inside the 7–11 window. That is a fingering a
player could use. It is still not the open-position one a beginner would reach for: with the
hand-set weights, `move` still outweighs neck height, which is a question for the weights
(ADR 0027), not for the window. The rendered clip also marks more notes uncertain: several
windows now give paths of equal cost, and the posterior spreads across them.

## Alternatives considered

- **Raise the speed limit until human tab passes.** Rejected in ADR 0022 and again here:
  it would fit a threshold to make a test pass.
- **Let the window width be fitted.** Rejected by Ege's decision on 2026-10-02: fixed at 4,
  simple and checkable.
- **Keep the point model because the gate was missed.** Offered to Ege and declined, on the
  evidence above that what remains is the speed limit's problem rather than the window's.

## Consequences

**Easier.** The cost model stops paying for movement that does not happen, which is what
ADR 0023's fitted weights were partly compensating for.

**Harder.** Decoding is about twice as slow, and the lattice about 2.6 times larger on real
playing. Every M1-era number moves, so GuitarSet is re-measured once, at the end of this
plan, with the README refreshed in the same commit.

**Revisit** when the speed-limit experiment lands: rerun `scripts/validate_playability.py`,
and only then may E3's transition rate be quoted as playability.

## Addendum (2026-10-03): the validation check, and a second decision by Ege

Added before this branch was reviewed or merged, under the precedent narrowed on 2026-10-01.
The plan's second pre-registered check: *the hand window does not lower the hand-set weights'
recovery of human fingerings on clean or distorted artist-disjoint validation parts.*

| hand-set weights, 300 validation songs | clean | distorted | NLL / group |
|---|---|---|---|
| point model | 0.8217 | **0.5728** | 0.7489 |
| hand window | 0.8257 | **0.5705** | **0.6937** |

**The check failed, by 0.0023 on distorted parts**, while clean rose by 0.0040 and the
window made the model fit human fingerings noticeably better. The rule set no allowance for
noise — a flaw in the rule, not a finding — and a 0.2-point move on 300 songs is within what
the sample can resolve. **Ege decided to keep the window.** Both checks this ADR faced were
missed, both by the margins given above, and both decisions to keep it were Ege's, made
with these numbers in hand.

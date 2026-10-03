# ADR 0030: A chord wider than the window stretches the hand

Status: proposed (2026-10-03) — pre-registration; accepted with its result

Supersedes **one sentence of [ADR 0025](0025-hand-window.md)**: "a shape wider than the
window anchors at its lowest fret". The 4-fret window, its rest width and everything else
in ADR 0025 stand. Changes the scorer contract (ADR 0007): the hand passed to
`transition_cost_from` becomes a fret range instead of a single fret. Approved in principle
by Ege on 2026-10-03.

## Context

Under ADR 0025 the hand is a window of frets `h` to `h + 4`, stored as `h` alone. A chord
wider than that — span 5 at fret 12 and above is legal under ADR 0011 — was placed with the
window at its lowest fret, so its own top note lay outside the window. Alternating such a
chord with its top note was charged a fret of movement each way: `(12, 17) → 17 → (12, 17)`
cost two frets though the hand never moved. It is the same phantom movement the window was
built to remove, left over for wide chords. Found in the review of the hand-window branch.

## Decision

**The hand covers a range of frets.** At rest it is the window, `h` to `h + 4`. Written so
that the oracle can be built from this text:

- The first fretted shape places the hand, for free.
- A fretted shape whose notes all lie inside the range the hand covers costs nothing; the
  hand's lowest fret (the index finger) stays where it was.
- Otherwise the index moves: to the start, among those whose rest window holds the whole
  shape, closest to where it was. A shape wider than the rest window can only be held
  stretched, with the index at the shape's own lowest fret.
- After every fretted shape the hand covers from its index to the higher of index + 4 and
  the shape's highest fret: **it stays stretched only while a shape needs it.**
- An all-open shape leaves the hand as it was, stretch included.
- Movement is the distance the index moves, charged at `move` per fret.

For every shape that fits the rest window this is exactly ADR 0025. Only wide chords and the
shapes played right after them change. In code the range is `Hand = (lowest, highest)` in
`types.py`; `fingering.states.shift_window` remains the single definition, used by E3, the
cost model, the lattice and the CRF fitter. The brute-force oracle is rewritten from this
text first, as a literal search over window starts, and does not import it.

## Check, pre-registered by committing this ADR before the run

Tolerance and noise estimate fixed in advance (ADR 0028's lesson). Wide shapes are under
0.2% of human shapes, so the expected effect is small.

**Hypothesis: the hand-set decoder's recovery of human fingerings changes by less than 0.001
on clean and on distorted artist-validation parts, with each 95% song-level paired interval
inside ±0.002; and human tab's E3 transition pass rate does not fall.** Single variable: the
hand model for wide shapes. Metrics: per-part recovery on the 300 artist-validation songs
(`scripts/score_validation.py`, compared with `scripts/compare_validation.py`), human tab's
E3 transition rate (`scripts/validate_playability.py`, same 16,732,524 moves).

**Keep rule:** the stretch stays unless a part's recovery falls by more than 0.001 with its
95% interval wholly below zero. Misses of the hypothesis that do not trip the keep rule are
reported, not acted on.

## Alternatives considered

- **A wider window at fret 12 and above**, matching ADR 0011's span-5 rule. Rejected: it
  changes the window's width, fixed at 4 by Ege's decision of 2026-10-02, for single notes
  as well as chords.
- **Anchor a wide chord at its highest fret instead.** Rejected: it moves the problem to
  the chord's bottom note.
- **Keep the stretch until the hand next moves.** Rejected: it would let one wide chord
  widen the hand for the rest of a passage.

## Consequences

**Easier.** No movement is charged for a chord's own notes after the chord.

**Harder.** The lattice carries a range rather than a fret, so a shape can have more nodes
after a wide chord; the growth is measured on DadaGP validation, not GuitarSet. E3's
transition rate moves slightly, and GuitarSet's figures are refreshed only at the next
GuitarSet run.

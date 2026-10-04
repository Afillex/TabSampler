# ADR 0039: Open strings played up the neck cost extra

Status: proposed (2026-10-04) — pre-registration; accepted with its result

Continues Phase 2 task C3 (`docs/plans/2026-10-01-phase-2.md`) on GuitarSet's validation
player (ADR 0037), from the error analysis in `docs/plans/2026-10-04-c3-guitarset-errors.md`.
Changes the `CostWeights` contract (ADR 0007).

## Context

On player 00, a third of the default's misplaced notes — 860 of 2,559 — are an open string
where the player fretted the same pitch; the reverse happens 12 times. The cost model
rewards every open string by `open_reward`, wherever the hand is. In 430 of the 860 the
decoder's hand was up the neck, its index at fret 5 or above, and in 326 of those the
player's hand was too: the decoder stayed in the player's position and took the open string
where the player fretted the note.

The Phase 2 plan lists "open strings conditioned on the neighbouring shapes" as a C3
candidate, and ADR 0034 deferred it as needing edge features in the fitter. It does not: a
lattice node already carries the hand its shape is played with (ADRs 0018, 0030), so a cost
that depends on a shape and that hand is a cost of the node.

## Decision

**Contract.** `CostWeights` gains `open_up_neck`, default zero, so every existing config
decodes exactly as before. A shape costs `open_up_neck` for each of its open strings when
the hand it is played with — the frets the hand covers after it (ADR 0030) — has its index
finger above open position, at fret 5 or higher; frets 1–4 are open position, as in ADR
0034. An all-open shape is played with the hand it inherits (ADR 0018). Before anything has
been fretted there is no hand, and nothing is charged.

**Where it is charged.** With the move into the shape, in the scorer's
`transition_cost_from`, which is where the hand is known. So the first shape of a passage is
never charged, just as it is never charged movement. The decoder, the fitter and the
brute-force oracle all charge it there.

**Experiment, fixed before it runs.** One weight, fitted by maximum likelihood on the clean
parts of the same 600 artist-split DadaGP training songs as ADR 0032, every other weight
held at its hand-set value. The challenger is the default with this one weight added, at the
default's temperature, which changes neither E2 nor E3.

**Hypothesis:** the fitted weight is positive, and on player 00 the challenger's oracle E2 is
higher than the default's. The 430 notes above bound the gain at +0.033; predicted +0.005
to +0.02.

**Rule.** The challenger replaces the default iff, on player 00's 60 tracks in oracle mode,
its E2 is higher than the default's with the 95% track-level paired interval wholly above
zero, **and** its chord-shape rate is not lower than the default's by more than the
allowance Ege sets before the run. ADR 0038 used 0.0005, which on player 00 is three chord
shapes of 6,607. Otherwise the default is unchanged and the weight stays at zero. If the
challenger is adopted, its temperature is recalibrated by ADR 0033's method, as ADR 0038
did. End-to-end E2 on player 00 and recovery on DadaGP's clean validation parts are
recorded, not ruled on.

## Alternatives considered

- **Refit `open_reward` alone.** It would reach all 860 errors, but also the 811 open
  strings the default places right; a cost that ignores where the hand is cannot tell them
  apart.
- **A hand-set value**, such as cancelling the open reward up the neck. Defensible from
  guitar knowledge, but fitting is C3's method; the hand-set route stays open if this fit
  comes out at zero or below.
- **A node cost on the scorer contract** (`emission_cost` given the hand). The more literal
  home, but a Protocol change through the decoder, forward-backward and the oracle; charging
  on the move gives the same cost to every shape but a passage's first.
- **Another threshold for "up the neck".** Fret 5 is ADR 0034's boundary, fixed before this
  fit and used by the error analysis; no other was tried.

## Consequences

**Easier.** One weight that can reach a sixth of the default's errors on player 00.

**Harder.** The scorer's transition cost is no longer movement alone: whoever reads
`transition_cost_from` must know it carries this term too.

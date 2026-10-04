# ADR 0039: Open strings played up the neck cost extra

Status: accepted (2026-10-04) — **the challenger clearly won on player 00: the default gains
`open_up_neck` 0.7615**

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
zero, **and** its chord-shape rate shows no clear drop: it fails this condition only if
the whole 95% track-level paired interval of its chord-shape rate minus the default's lies
below zero. That replaces the fixed 0.0005 of ADRs 0032, 0034 and 0038 — three chord
shapes of player 00's 6,607, below what 60 tracks can tell from noise — by Ege's decision
of 2026-10-04, for this rule and later ones on player 00; ADR 0016's guardrail still binds
at the M2 evaluation on the test players. Otherwise the default is unchanged and the
weight stays at zero. If the
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

## Result (2026-10-04)

**The fit:** `open_up_neck` = 0.7615 on the clean parts of the 600 training songs, the base
weights held, converged in 6 iterations. **Positive, as predicted.**

**Player 00**, 60 tracks, 13,223 reference notes; track-level paired bootstrap, exact counts,
both decoders at T = 1.5728 and committed before the runs (420c088):

| | default | challenger | challenger − default, 95% interval |
|---|---|---|---|
| **E2 oracle** | 0.8065 (10,664 notes) | 0.8273 (10,940) | **+0.0209 [+0.0066, +0.0363]** |
| **E3 chord shapes, oracle** | 6543 / 6607 | 6541 / 6607 | −0.0003 [−0.0011, +0.0003]: **no clear drop** |
| E2 end to end | 0.4861 | 0.4900 | +0.0039 [−0.0065, +0.0146] |
| E3 transitions, oracle / e2e | 0.9997 / 0.9952 | 0.9997 / 0.9952 | |

**Both conditions met: the challenger replaces the default.** The gain, +0.0209, is just above
the +0.005 to +0.02 predicted and inside the +0.033 bound. The fixed allowance this rule
replaced would have given the same verdict: the drop, 0.0003, is under 0.0005.

`configs/decoder_clean.yaml` now carries `open_up_neck: 0.7615`, at a temperature recalibrated
for it by ADR 0033's method: **T = 1.2934**, calibration error 0.1056 → 0.0707 on DadaGP's
clean validation parts, the songs it was fitted on. On player 00 the new default's calibration
error is 0.0830 in oracle mode (0.1178 before) and 0.2387 end to end (0.2100 before): a sharper
decoder is surer of notes the transcriber got wrong, as under ADR 0033. ADR 0016's end-to-end
guardrail, below 0.3851, holds. `configs/decoder_clean_open.yaml` keeps the challenger as it
was run.

**Recorded, not ruled on.** On DadaGP's clean validation parts recovery rose 0.8257 → 0.8460
(+0.0203 [+0.0085, +0.0348]); on distorted parts 0.5705 → 0.5768 (+0.0062 [−0.0010, +0.0154]).
Re-run on player 00, the error analysis
(`scripts/analyse_errors.py --decoder-config configs/decoder_clean_open.yaml`) shows open
strings where the player fretted falling from 860 to 663: with the decoder's hand up the neck
from 430 to 97, while those in open position rose from 430 to 566 — the decoder now sometimes
moves down the neck to take an open string. Every style, comping and soloing improved; notes at
fret 12 and above lost ten.

**Not yet measured on the test players.** The default's figures on players 01–05 are still the
previous default's; the new default's wait for the next pre-registered test evaluation, Phase
2 task C6.

# ADR 0027: The fair test the fitted weights must pass to become the default

Status: proposed (2026-10-02) — pre-registration; accepted with its result

## Context

ADR 0023 measured weights fitted on DadaGP and declined to make them the default: they lost
11 points of human-fingering recovery on clean guitar parts while gaining on distorted ones.
It named two legitimate routes to adoption, and Ege chose the second: the fitted weights
must win a test fixed in advance, on validation, and GuitarSet may never be that test.

## Decision (pre-registered before the run)

Weights refitted by maximum likelihood on the artist-disjoint training side (ADR 0024),
under the hand-window model (ADR 0025), replace the hand-set weights as the default **if
and only if**, on the artist-disjoint validation side, with identical lattices, **all
three** hold:

1. they recover more of the human fingering than the hand-set weights on **clean** parts;
2. they recover more on **distorted** parts;
3. the decoded output's **E3 chord-shape rate** — the validated playability measure
   (ADR 0022) — is not below the hand-set weights'.

All three, or no change. A tie on 1 or 2 is not a win. The temperature is calibrated
separately (ADR 0026's method) for whichever weights end up as the default.

**Single variable:** the weights. **Split:** artist validation. **Measured by:**
`scripts/fit_cost_weights.py --split artist`, which prints recovery by part and the decoded
E3 chord-shape rate for both weight sets.

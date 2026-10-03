# ADR 0027: The fair test the fitted weights must pass to become the default

Status: accepted (2026-10-03) — **the fitted weights failed the test; the hand-set weights stay the default**

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

## Result (2026-10-03)

Refitted under the hand window on 600 artist-split training songs: move 1.0143, span 0.6149,
high 0.2511, open_reward −0.8487. On 300 artist-disjoint validation songs (505,847 notes):

| | clean recovery | distorted recovery | decoded E3 chord shapes |
|---|---|---|---|
| hand-set | **0.8257** | 0.5705 | **0.9998** |
| fitted | 0.6919 | **0.6149** | 0.9987 |
| condition | 1: **fails** | 2: passes | 3: **fails** |

**The fitted weights fail two of three conditions, so nothing changes.** They still trade a
large loss on clean guitar for a gain on distorted guitar, as they did under the point model
(ADR 0023). This settles ADR 0023's open question for these weights: they are not adopted.

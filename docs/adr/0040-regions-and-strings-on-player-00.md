# ADR 0040: Fret regions and per-string preferences, re-judged on player 00

Status: proposed (2026-10-04) — pre-registration; accepted with its results

Continues Phase 2 task C3 (`docs/plans/2026-10-04-c3-guitarset-errors.md`, Task 3). Re-tries
ADR 0034's two feature groups the way ADR 0039 tried its one: on top of today's default, and
judged on GuitarSet's validation player.

## Context

ADR 0034 fitted fret regions (2 weights) and a per-string preference (5) together with the four
base weights, and judged them on DadaGP; neither was adopted for clean guitar. Two things have
changed. Player 00 is now the validation data that resembles the test set (ADR 0037), and ADR
0039 showed a group can be fitted with every other weight held at the default's values, so the
experiment changes one thing — and the per-string biases cannot trade against `high`, which
HANDOFF warned they do when both are fitted.

Re-run on today's default, the error analysis of player 00 says where the errors now are: 69%
are notes the player fretted at frets 5–11, 88% of misplaced notes land in another fret region,
and the decoder put 64% of them lower on the neck than the player did.

## Decision

**Two experiments, in this order, each against the default of the moment.** Each fits one
group on the clean parts of the same 600 artist-split DadaGP training songs as ADRs 0032 and
0039, every other weight held at `configs/decoder_clean.yaml`'s values
(`scripts/fit_cost_weights.py --base-config configs/decoder_clean.yaml --hold-base`):

1. **Fret regions** (`low_region`, `high_region`). Hypothesis: `low_region` comes out
   positive, since the decoder plays lower than the player, and on player 00 the challenger's
   oracle E2 is higher than the default's. Prediction: a gain under +0.01, which may not
   clear the interval — on DadaGP's clean parts the group gained +0.0022.
2. **Per-string preference** (`string_bias`, five weights, the low E the reference), on top of
   whatever experiment 1 leaves as the default. Hypothesis: on player 00 the challenger's
   oracle E2 is higher. Prediction: no clear gain — on DadaGP's clean parts the group lost
   0.0104 when fitted with the base weights.

**Rule, for each:** ADR 0039's. The challenger replaces the default iff, on player 00's 60
tracks in oracle mode, its E2 is higher with the 95% track-level paired interval wholly above
zero, and its chord-shape rate shows no clear drop (it fails only if that interval lies wholly
below zero). If it is adopted, its temperature is recalibrated by ADR 0033's method. End-to-end
E2 on player 00 and recovery on DadaGP's clean validation parts are recorded, not ruled on.

## Alternatives considered

- **Both groups at once.** Seven weights against the default would not say which helped.
- **Regularisation first**, as the C3 plan had it. Needed when correlated groups are fitted
  together; with every other weight held, the fit has no partner to trade against.
- **New position features first.** ADR 0034's two groups are the plan's candidates, already in
  the contract and the fitter, and never yet judged on GuitarSet-like playing.

## Consequences

**Easier.** Two cheap experiments that settle what ADR 0034 left open on DadaGP.

**Harder.** If both win, the default carries twelve cost weights, eight of them fitted in
three separate experiments on top of one another.

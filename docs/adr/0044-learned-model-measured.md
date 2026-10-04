# ADR 0044: The learned model, measured on player 00 — no clear gain over the decoder

Status: accepted (2026-10-04)

Reports the pre-registered run of ADR 0043's model (`docs/plans/2026-10-04-c4-learned-model.md`,
Task 6, committed in 227ca87 before training).

## Context

ADR 0043's model is today's decoder plus a learned cost from a bidirectional GRU over the note
groups, the CRF kept as the output layer. It was trained by the fitter's objective on the clean
parts of 1,500 artist-split DadaGP training songs — 336,315 groups — and stopped on DadaGP's
clean validation parts, then judged once on GuitarSet's player 00 by ADR 0039's rule.

## Result

**On DadaGP**, hypothesis 1 held: the negative log-likelihood per group on the clean
validation parts fell from 0.4327 (the default) to 0.3136, the best of 16 epochs (the 13th),
training stopping on patience.

**On player 00**, oracle mode, 60 tracks, track-level paired bootstrap, every decoding on the
same groups:

| decoding | oracle E2 | against (a), 95% interval | chord shapes |
|---|---|---|---|
| (a) today's decoder | 0.8273 | | 6541 / 6607 |
| (b) the learned term alone | 0.4652 | −0.3622 [−0.4162, −0.3054] | 6224 / 6607, a clear drop |
| (c) the learned model, Viterbi | 0.7970 | −0.0303 [−0.0861, +0.0199] | 6536 / 6607, no clear drop |

**Hypothesis 2 failed:** (c)'s interval is not wholly above zero — its point estimate is below
(a)'s — so the learned model is not preferred, and the prediction (−0.01 to +0.03) failed low.
Before training, the untrained model's (c) equalled (a) on every track, so the loss is the
training's, not the plumbing's.

## Decision

- **The default decoder stays as it is.** ADR 0042 would have kept a DadaGP-trained model out
  of the CLI in any case; it is now also not preferred on the evidence.
- **The comparison the spec's Phase 2 gate asks for exists**, on validation: the Viterbi
  decoder, a learned model on its own, and the learned model decoded by Viterbi. Its version on
  the test players waits for the one pre-registered evaluation of task C6.

## What it says

More capacity fitted DadaGP better and GuitarSet's player worse: the same direction as ADRs
0023, 0032 and 0040, now with a model that can learn context. The limit is the proxy, not the
model — crowd-sourced, mostly distorted rock tab is not how GuitarSet's players finger
acoustic comping and soloing.

**Looked at after the result, and not used to choose anything:** by style, (c) gained on bossa
nova (+0.074) and jazz (+0.046) and on soloing (+0.019), and lost on rock (−0.079),
singer-songwriter (−0.054), funk (−0.039) and comping (−0.045); it was better on 24 tracks,
worse on 32, level on 4. It plays differently rather than uniformly worse, which a later,
pre-registered experiment could test; this run cannot.

## Consequences

**Easier.** Phase 2's comparison is in hand, with a clear negative result on validation.

**Harder.** Closing the gap to M2 now needs training data that resembles the test set, or a
way to judge a learned model on GuitarSet-like playing without exhausting a 60-track
validation player.

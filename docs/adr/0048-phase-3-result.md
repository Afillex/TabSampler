# ADR 0048: Phase 3's result — the audio evidence costs the decoder on GuitarSet

Status: accepted (2026-10-06) — **Phase 3 closed on this record by Ege's sign-off**

Reports Phase 3 (`docs/plans/2026-10-04-phase-3-audio.md`). The spec's "done when" is a measured
change in oracle-mode Exact Tab F1 against Phase 2, with an ablation that removes the audio input;
M3 reads "I showed, with an ablation, whether audio evidence improves string assignment".

## Result

The evidence is ADR 0046's: a string probability per note from a small CNN trained on SynthTab's
development set — 8.9 hours of guitar rendered from DadaGP songs, 0.93 hours of it acoustic — that
enters the decoder through ADR 0047's acoustic term. On its own it picks the right string for 47%
of player 00's notes that more than one string can sound, against 28% by chance and 82% for the
decoder.

Oracle E2 of the default decoder, Phase 2's, without the evidence and with it:

| | without | with | change [95% track-level interval] |
|---|---|---|---|
| player 00, weight 1.0, uncalibrated (Task 5) | 0.8273 | 0.7001 | −0.1273 [−0.1834, −0.0729] |
| player 00, calibrated (Task 6) | 0.8273 | 0.7865 | −0.0408 [−0.0724, −0.0142] |
| **players 01–05, calibrated (Task 7, one logged look)** | **0.6819** | **0.6073** | **−0.0746**, no interval on test |

*Calibrated* is temperature 1.8985 and acoustic weight 0.25, both chosen on SynthTab's 33 held-out
tracks without reading GuitarSet. On those tracks the evidence raised the share of notes the
decoder placed on their labelled string from 0.7419 to 0.7724; they are the tracks the weight was
chosen on, so that gain is optimistic. Chord shapes showed no clear drop once calibrated (player
00: 6541 → 6541 of 6607; test: 30619 → 30604 of 30685) and a clear one before (6541 → 6479).

**The audio evidence costs the decoder on GuitarSet's recordings and helps it on SynthTab's.**
Calibration cut the loss on player 00 by two thirds without removing it; on the test players the
loss is larger. Every pre-registered prediction about the evidence on GuitarSet failed: that it
would help (Task 5), that calibrated it would no longer be a clear loss (Task 6), and the size of
the loss on the test players, predicted at −0.06 to −0.02 (Task 7). The hypotheses about the
classifier on its own and about its calibration held, though one predicted range missed by 0.005.

## Decision

- **The default decoder keeps the acoustic weight at zero.** The term stays in the code (ADR
  0047), off, and the classifier's weights stay unpublished (ADR 0046).
- **The spec's Phase 3 deliverable exists, as a negative result**: the change with the audio
  ablated, measured on validation and once on test. Ege signed off on closing the phase on it.

## What it does not settle

SynthTab is rendered and mostly electric; GuitarSet is an acoustic guitar recorded through a
microphone. The evidence helps on one and hurts on the other, so the cost lies in the distance
between them — but whether that distance is rendered against real or electric against acoustic,
this phase did not measure: the two are confounded. It is the pattern of Phase 2's learned model,
which fitted DadaGP better and player 00 worse (ADR 0044). Training on real recordings is the
spec's Phase 4; which guitar sound it serves first is D2, still open, and this result bears on it.

## Consequences

**Easier.** Phase 4 starts with the pipeline in place — note windows, classifier, calibration, the
acoustic term and the ablation script — rather than from nothing.

**Harder.** The test players have now been read with this evidence. A retrained classifier needs
its own pre-registered look, and the result here is for this classifier at these settings, not
for audio evidence in general.

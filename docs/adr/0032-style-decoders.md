# ADR 0032: Clean and distorted guitar get a decoder each

Status: accepted (2026-10-03) — **clean passed its fair test; distorted failed it on the
chord-shape condition**

Carries out the style item of Phase 2 task C3 (`docs/plans/2026-10-01-phase-2.md`): "fit on
clean and distorted parts separately and measure each on its own validation parts", and
makes the decision that item said needs its own ADR. Approved in principle by Ege on
2026-10-03.

## Context

Every fit so far pooled DadaGP's guitar parts, about three quarters of whose notes are
distorted. The fitted weights then gained on distorted parts and lost heavily on clean
ones — under the hand window, 0.6919 against the hand-set weights' 0.8257 (ADR 0027) — and
clean or acoustic guitar is what a typical user records, and what GuitarSet is. One set of
weights may simply not serve both styles.

## Decision, fixed before the fits

**Experiment.** Fit the four weights twice by maximum likelihood (`fingering/fit.py`): once
on the clean parts and once on the distorted parts of the same 600 artist-split training
songs (`scripts/fit_cost_weights.py --part`). Score each on the 300 artist-validation songs.
Single variable: which style's parts the weights are fitted on.

**Hypothesis:** each style's fitted weights beat the hand-set weights on that style's own
validation parts. Prediction: distorted holds easily, as pooled fits already won there;
clean is the open question.

**Fair test per style**, with its noise estimate fixed in advance (ADR 0028): a style's
fitted weights replace the hand-set weights *for that style* iff, on that style's
artist-validation parts, (a) recovery of the human fingering is higher with the 95%
song-level paired interval wholly above zero, and (b) the decoded chord-shape rate is not
lower by more than 0.0005.

**Product decision, fixed now whatever the result:** each style gets a decoder config,
`configs/decoder_clean.yaml` and `configs/decoder_distorted.yaml`, holding whichever weights
won its test. **The CLI's default becomes `configs/decoder_clean.yaml`**, because clean or
acoustic guitar is the typical recording and GuitarSet's material; a user with distorted
guitar passes the other config. `configs/phase1_baseline.yaml` stays as the record of the
hand-set baseline. Each config starts with T = 2.9974 until ADR 0033 calibrates one per
style.

## Alternatives considered

- **Detect the style from the audio.** A classifier is Phase 3 work at the earliest, and a
  wrong guess would be worse than an explicit choice.
- **Marginalise over both styles.** It would hand clean users the distorted model's habits
  in proportion to a prior nobody has measured.
- **One decoder, keep pooling.** Rejected on ADR 0027's evidence.

## Consequences

**Easier.** Each style is judged on its own playing; a model that is good for metal no
longer has to be good for bossa nova too.

**Harder.** Two decoders to calibrate and maintain, and a choice the user must make — the
default covers the common case.

## Result (2026-10-03)

| 300 artist-validation songs | hand-set | fitted on that style | delta, 95% paired interval | chord shapes, that style's parts |
|---|---|---|---|---|
| **clean parts**, clean-only fit | 0.8257 | **0.8616** | +0.0359 [+0.0134, +0.0622] | 0.99974 → 0.99964 (−0.00010) |
| **distorted parts**, distorted-only fit | 0.5705 | **0.6480** | +0.0774 [+0.0358, +0.1177] | 0.99978 → 0.99913 (**−0.00065**) |

Fitted weights: clean — move 1.2316, span 0.7279, high 0.8867, open_reward −0.1702 (131,345
training groups); distorted — move 0.9466, span 0.7102, high 0.1610, open_reward −1.3742
(390,284 groups).

**Clean passes: for the first time, fitted weights beat the hand-set ones on clean guitar**,
by 3.6 points of recovery, once they are fitted on clean playing alone. They are
`configs/decoder_clean.yaml`, the CLI's default; the rounded weights reproduce the fit
exactly.

**Distorted fails, on condition (b).** Its recovery gain is large and clearly real, but its
decoded output has more chord shapes that E3 calls unplayable: 0.00065 more, against the
0.0005 allowed. By the rule fixed before the fit, `configs/decoder_distorted.yaml` keeps
the hand-set weights. Whether that allowance is right for a style whose model recovers 7.7
points more of the human fingering is a question for Ege, not a reason to bend the rule.

The hypothesis held for recovery on both styles; the fair test is what decides adoption,
and it split.

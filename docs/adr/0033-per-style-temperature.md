# ADR 0033: Each style's decoder gets its own temperature

Status: accepted (2026-10-03) — hypothesis held

Follows [ADR 0026](0026-default-temperature.md) (one temperature, T = 2.9974, calibrated on
all artist-validation parts) and [ADR 0032](0032-style-decoders.md) (a decoder per style).
Carries out Phase 2 task C5 for the decoders that actually ship. Approved in principle by
Ege on 2026-10-03.

## Context

ADR 0026's temperature was calibrated on validation parts that are three-quarters distorted.
On GuitarSet, which is clean, the default's oracle calibration error then did not move
(0.1652 → 0.1651) while the end-to-end error fell by two thirds. One temperature for two
styles that the decoder gets right at very different rates (clean 0.83, distorted 0.57 of
notes) may be the reason; that question belongs to DadaGP validation, not to GuitarSet.

## Decision, fixed before the runs

For each style, take that style's decoder config (ADR 0032) and calibrate its temperature
by per-note negative log-likelihood on **that style's parts** of the 300 artist-validation
songs, exactly as ADR 0026 did on all of them (`scripts/calibrate_temperature.py --part`).
Report calibration error, computed as E5 computes it, at T = 2.9974 and at the new T. Write
each new T into its config. Single variable: the temperature, per style.

**Hypothesis: the styles need different temperatures, and clean parts — where the decoder
recovers far more of the human fingering — need a lower one than distorted parts. Each
style's own temperature lowers that style's calibration error against T = 2.9974.**

As in ADR 0026, the error is measured on the songs the temperature was fitted on: one
number fitted to tens of thousands of notes, so the optimism is small, and it is said
wherever the figure is quoted.

## Alternatives considered

- **Keep one temperature.** It is what the hypothesis tests against; it stays for any style
  where the new one does not lower the error.
- **Calibrate on half the validation songs and measure on the other half.** Cleaner, but it
  halves the data for a one-parameter fit whose optimism is already small; not worth a
  second variable.

## Consequences

**Easier.** Each decoder's posteriors are calibrated on the playing it is meant for.

**Harder.** Two numbers to keep in step with their weights: whenever a style's weights
change, its temperature is recalibrated (Phase 2 task C5).

## Result (2026-10-03)

| each style's decoder, on its own validation parts | T | calibration error | per-note NLL |
|---|---|---|---|
| clean (clean-fitted weights), at the pooled T | 2.9974 | 0.2478 | 0.6463 |
| **clean, own T** | **1.1975** | **0.0757** | **0.4842** |
| distorted (hand-set weights), at the pooled T | 2.9974 | 0.0785 | 0.7576 |
| **distorted, own T** | **4.2982** | **0.0601** | **0.7493** |

**Held on both counts**: clean needs a much lower temperature than distorted (1.1975 against
4.2982), and each style's own temperature lowers its calibration error. The pooled 2.9974
was close to right for distorted playing, which dominated the pool, and badly wrong for the
clean decoder, which it made far too unsure of itself.

One thing the table does not show: for the clean decoder, T = 1 gives a *lower* calibration
error (0.0559) than the likelihood-optimal 1.1975, though a worse likelihood (0.5056). The
method fixed above chooses by likelihood, the proper scoring rule, and is kept; the gap is
recorded, not acted on. All figures are in-sample, on the songs the temperatures were
fitted on.

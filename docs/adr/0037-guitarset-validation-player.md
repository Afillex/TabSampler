# ADR 0037: GuitarSet's player 00 becomes validation data

Status: accepted (2026-10-03) — Ege's decision

Amends [ADR 0003](0003-evaluation-protocol-guitarset-held-out.md), the project's most
consequential decision: GuitarSet's players 01–05 stay test-only exactly as ADR 0003 says,
and player 00 becomes validation data.

## Context

Model choices are made on validation data, and the only validation data so far was DadaGP,
crowd-sourced tab that is mostly distorted rock. Its clean parts have now been a poor guide
to GuitarSet three times: the pooled fitted weights (−11 points on clean validation, +3.9 on
GuitarSet, ADR 0023), the hand window (+0.4 and +2.8) and the clean decoder (+3.6 and −1.9,
ADR 0032). The last one moved the default the wrong way. Without validation data that
resembles the test set, every further choice is a guess about acoustic playing.

GuitarSet has six players, each recording the same 30 lead sheets in five styles, comping
and soloing: 60 recordings per player, 360 in all.

## Decision

- **Player 00's 60 recordings are validation data**: fitting, selection and calibration
  may use them. **Players 01–05's 300 recordings are the test set**, test-only as before:
  read only by pre-registered, logged runs, and never used to choose anything.
- The player is chosen **by rule — the lowest ID** — fixed before any per-player figure
  was looked at; none had been.
- The committed 360-track snapshot stays the corpus. `data/splits.py` derives both lists
  from it and refuses a snapshot in which any player has other than 60 tracks.
- **ADR 0016's target and guardrail values are kept as numbers** (oracle E2 ≥ 0.760; chord
  shapes ≥ 0.9970 / 0.9896; transitions ≥ ADR 0035's 0.9996 / 0.9969; end-to-end E5
  < 0.3851) **and judged on the 300 test tracks from now on.** Each new test figure is shown
  next to the same decoder's 360-track figure where one exists, so the change of test set
  is visible rather than silent.

## What it costs

- **Contamination.** The 300 test tracks were part of every earlier 360-track evaluation.
  A decision made now about a decoder those runs measured is made knowing how it did on
  them, in aggregate, so its later test figure on the 300 is not an independent
  confirmation. Independence returns for questions those runs did not answer. Every test
  figure quoted while this applies says so.
- **The same songs.** The validation player plays the same 30 lead sheets as the test
  players, so validation can reward a fit to those compositions. With a handful of weights
  the risk is small; with a learned model it would not be, and would need revisiting.
- **Comparability.** Figures on 300 tracks are not figures on 360; the README keeps the 360
  history, labelled.
- **Power.** 60 tracks, about ten thousand notes, give wider intervals than 300 DadaGP
  songs; the track-level paired bootstrap reports them.

## Alternatives considered

- **Other recorded datasets with string labels**, keeping GuitarSet whole. Slower, and which
  ones have usable labels and licences is unknown. Offered to Ege; not chosen.
- **DadaGP's acoustic-guitar parts only.** The cheapest, but still tab writers' fingerings
  rather than players'. Offered; not chosen.
- **Two players as validation.** More power for a smaller test set; one was chosen.
- **Rotating the held-out player.** Every player would then have been used for selection,
  leaving no test set.

## Consequences

**Easier.** A validation result now speaks about the kind of playing the test measures.

**Harder.** GuitarSet is both the measure and, for one player, the proxy — the line ADR
0003 drew is now between players, and it has to be held as strictly as before.

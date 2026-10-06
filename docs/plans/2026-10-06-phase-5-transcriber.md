# Phase 5 — A Better Transcriber

> Tasks run in order; steps use checkbox (`- [ ]`) syntax. Every run that produces a
> reported number is pre-registered in a commit made before it runs.

**Status:** started 2026-10-06, after Ege closed Phase 4 (ADR 0054) and said to go on; the spec's
order puts Phase 5 next.

**Goal (spec §6):** fine-tune Basic Pitch on guitar data, then try heavier models. **Done when:**
end-to-end Tab F1 improves. Note F1 improving alone is not enough.

## Where this starts

- **The transcriber:** Basic Pitch 0.4.0 behind a subprocess (ADR 0001), its CLI defaults —
  onset threshold 0.5, frame threshold 0.3, minimum note length 127.70 ms. Its released model
  trained on 90% of GuitarSet (ADR 0055); its training code ships in the package
  (`basic_pitch/train.py`, a Beam/TFRecord data pipeline).
- **What transcription costs:** EGDB (clean for it) E1 0.7230, end-to-end E2 0.4251 against oracle
  0.6762; GuitarSet's test players, labelled, E1 0.7493, E2 0.4418 against 0.6819.
- **Validation:** Guitar-TECHS's player 3 — 12 takes, about 8.5 minutes, direct input, labels
  cleaned (ADR 0051), 86% of its pitches confirmed by the audio; each take's label delay is
  subtracted from the reference onsets when a transcriber is scored (ADR 0055). Small and noisy:
  intervals over 12 takes are wide. **Test:** EGDB, one look at the end; GuitarSet beside it,
  labelled.
- **Candidates checked so far:** the GAPS benchmark model (zero-shot on GuitarSet) is unreleased,
  per its paper, and its dataset is CC BY-NC-SA.

## Global constraints

- One variable per experiment; the hypothesis committed before the run.
- Transcribers compared and tuned only on audio no candidate trained on (ADR 0055).
- The transcriber stays behind the `Transcriber` protocol and its own environment (ADR 0001).
- Weights fine-tuned on Guitar-TECHS alone could be publishable (CC BY 4.0); that is D15, Ege's.

## Task 1: The training overlap (ADR)

- [x] ADR 0055; the README's end-to-end figures labelled.

## Task 2: Where the end-to-end loss goes (`scripts/analyse_e2e.py`)

On player 3 (and on GuitarSet's player 00, labelled — it chooses nothing): every reference note is
either missed by the transcriber or heard (E1's matching, pitch within a quarter tone and onset
within 50 ms). Missed notes split by duration and by what the transcriber put near them (an
octave off, a semitone off, nothing); extra notes likewise. Heard notes split by whether the
default decoder strings them right end to end and in oracle mode. From those counts, the E2 gap
splits into **note errors** (oracle E2 down to E2 with the transcriber's notes strung as oracle mode
strings them) and **context** (down from there to the measured end-to-end E2). Descriptive: no
results row, nothing chosen.

**Predictions, fixed before the run:** note errors carry most of the gap, context a quarter or
less; the missed notes are mostly short — under Basic Pitch's 127.7 ms minimum note length — on
player 3's musical excerpts.

- [ ] Write it, with tests; run on player 3 and player 00; report.

## Task 3: Basic Pitch's note thresholds

A grid over onset threshold, frame threshold and minimum note length, scored by end-to-end E2 on
player 3; the rule — and whether the grid is worth running at all — fixed after Task 2 says where
the loss is.

- [ ] Pre-register; run; record.

## Task 4: Fine-tune Basic Pitch on Guitar-TECHS

Its training code is in the package; players 1–2 train, player 3 validates (ADR 0051). Designed
after Tasks 2 and 3.

- [ ] Design (ADR); pre-register; train; measure on player 3; record.

## Task 5: Heavier models

Only candidates with released weights, a licence that allows use, and stated training data that
excludes EGDB.

- [ ] Survey; one candidate at most per pre-registration.

## Task 6: One look at the test sets

- [ ] Pre-register the chosen transcriber against Basic Pitch's defaults on EGDB, GuitarSet beside
      it labelled; run once; write Phase 5's result (ADR); README, HANDOFF, devlog; Ege's sign-off.

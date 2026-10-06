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

- [x] Write it, with tests; run on player 3 and player 00; report. *(Player 3: note errors 0.2137
      of the 0.2304 gap, context 0.0167; 217 of 353 missed notes short — both predictions held.
      Basic Pitch writes 2,456 notes against 1,629: 1,180 extras, precision about half. Player 00,
      labelled: note errors 0.2408, context 0.0966 — 29%, so the context prediction fails there;
      its misses are mostly long notes, 2,620 of 3,872, most with nothing near.)*

## Task 3: Basic Pitch's note thresholds

A grid over onset threshold, frame threshold and minimum note length, scored by end-to-end E2 on
player 3. Task 2 says it is worth running: on player 3 nearly all the loss is the transcriber's
notes, and most of that is notes it adds.

**Pre-registered:** `scripts/tune_transcriber.py --out cache/validation/p5-grid`: onset threshold
0.3, 0.4, 0.5, 0.6, 0.7 × frame threshold 0.3, 0.4, 0.5 × minimum note length 58 and 127.70 ms —
30 settings, Basic Pitch's defaults (0.5, 0.3, 127.70) among them. **The rule:** the candidate is
the setting with the highest pooled end-to-end E2 on player 3, the default on a tie; it is
adopted for Phase 5 only if `scripts/compare_validation.py` (default → candidate) puts its 95%
take-level interval wholly above zero with no clear chord-shape drop — ADR 0039's rule, Ege's
chord-shape condition. Otherwise the defaults stay. Picking the best of 30 on 12 takes flatters
the winner, so the test look decides nothing either way. **Hypothesis 1:** a stricter setting
wins, as precision is player 3's problem. **Predictions:** the candidate's onset threshold is
0.6 or above; it gains 0.01 to 0.05 of E2 over the default; its interval clears zero.

- [x] Run; record. *(Default 0.3491 → onset 0.7 / frame 0.4 / min 58 ms 0.4185, +0.0694
      [+0.0081, +0.1303], no clear chord-shape drop: **adopted**. E1 0.6247 → 0.7737. H1 held; the
      size prediction failed high. The winner sits at the grid's edge. `compare_validation.py`
      needed a ratio bootstrap end to end, where the two sides count different notes.)*

### Task 3b: past the grid's edge

**Pre-registered:** the adopted setting sits at the grid's edge, onset 0.7. `scripts/tune_transcriber.py
--out cache/validation/p5-grid --onsets 0.75 0.8 0.85 0.9 --frames 0.3 0.4 0.5 --lengths-ms 58` —
12 settings beyond it. **The rule:** the best of them by end-to-end E2 on player 3 replaces onset
0.7 / frame 0.4 / min 58 ms only if its take-level interval against that setting lies wholly above
zero with no clear chord-shape drop. **Prediction:** E2 peaks between 0.7 and 0.8, and nothing
beyond clears the interval: onset 0.7 / frame 0.4 / min 58 ms stays.

- [x] Run; record. *(Best past the edge 0.4218, +0.0033 [−0.0459, +0.0791]: not adopted;
      prediction held. Onset 0.7 / frame 0.4 / min 58 ms stays.)*

## Task 4: Fine-tune Basic Pitch on Guitar-TECHS

Its training code is in the package; players 1–2 train, player 3 validates (ADR 0051). Designed
after Tasks 2 and 3.

- [x] Design (ADR). *(ADR 0056, Ege's route: a separate TensorFlow environment, the result run by
      the unchanged CLI through `--model-path`. A check first claimed here — the released weights,
      exported, give identical notes on all 12 of player 3's takes — **proved nothing**: the
      adapter passed `--model-serialization`, which makes basic-pitch ignore `--model-path`.
      Fixed; redone with a negative control: the released export identical on 12 of 12, the
      fine-tuned model different on 12 of 12.)*

**Pre-registered:** `finetune_basic_pitch.py train --data cache/transcriber/data --run
cache/transcriber/ft` (86 training takes, 4.48 hours; 12 validation takes) as ADR 0056 fixes it,
then `export --run cache/transcriber/ft`, then `tune_transcriber.py --out cache/validation/p5-ft
--model-path cache/transcriber/ft/model.mlpackage --onsets 0.7 --frames 0.4 --lengths-ms 58`,
compared by `compare_validation.py` against the released model at the same thresholds
(`cache/validation/p5-grid/onset0.7_frame0.4_min58.json`). **One variable: the weights.** **The
rule:** the fine-tuned model is adopted if its take-level interval clears zero with no clear
chord-shape drop. Player 3 both stops the training and judges it, so the figure is slightly
flattered. **Hypothesis 2:** fine-tuning on real electric direct input raises end-to-end E2 on
player 3. **Predictions:** player 3's loss falls at least 5% below epoch 0's; E2 changes by 0.00 to
+0.04 — drills (scales, chords, single notes) teach less about music than their hours suggest — and
the interval does not clear zero.

- [x] Train; export; measure on player 3; record. *(Loss 1.2491 → 1.0149, −18.8%: held. At the
      released model's thresholds, E2 0.4185 → 0.0946, −0.3239 [−0.4294, −0.2057]: it writes 9,858
      notes where the released model writes 1,951 — fine-tuning moved the posteriors' scale. H2
      and its prediction failed.)*

## Task 5: Heavier models

Only candidates with released weights, a licence that allows use, and stated training data that
excludes EGDB.

- [ ] Survey; one candidate at most per pre-registration.

## Task 6: One look at the test sets

- [ ] Pre-register the chosen transcriber against Basic Pitch's defaults on EGDB, GuitarSet beside
      it labelled; run once; write Phase 5's result (ADR); README, HANDOFF, devlog; Ege's sign-off.

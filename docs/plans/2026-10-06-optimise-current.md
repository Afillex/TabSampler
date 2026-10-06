# Tighten what exists before Phase 6

> Tasks run in order; steps use checkbox (`- [ ]`) syntax. Every rule that picks or adopts
> something is written here and committed before the run it judges.

**Status:** started 2026-10-06, after the app track closed. Ege chose these three, in this order,
over starting Phase 6.

**Goal:** a faster app, uncertainty marks that point at likely errors, and fewer of Basic Pitch's
extra notes on electric guitar — each measured on validation data, with one test look at the end.

## Where this starts

- A 3-minute file (validation takes, concatenated): transcription 3.1 s cold, the tuning check
  added on 2026-10-06 **5.4 s**, decoding 0.1 s.
- The uncertainty mark (posterior < 0.6, never tuned): 125 of 217 notes on player 3's take 04,
  39 of 118 on Ege's downloaded file.
- End to end on EGDB (test): E2 0.4547; given the true notes, 0.6762. On player 3 at Basic Pitch's
  **default** thresholds, 1,180 of 2,456 transcribed notes were extra, 227 of them an octave from a
  played note (`scripts/analyse_e2e.py`); at today's thresholds (ADR 0057) the split is unmeasured.

## Global constraints

- Validation only until Task 3's last step: Guitar-TECHS player 3 (12 takes, direct input; Basic
  Pitch never trained on it), and GuitarSet player 00 reported beside it, labelled (ADR 0055).
- Player 3 also chose Phase 5's thresholds; anything chosen on it again is checked on EGDB once.
- One variable per measurement; no new dependency.

## Task 1: The tuning check, fast

**Candidates:** (a) today: harmonic part (HPSS), then `librosa.estimate_tuning`, whole file;
(b) as (a) on at most the first 60 s after leading silence; (c) no HPSS, whole file; (d) no HPSS,
60 s.

**Rule (fixed now):** take the fastest candidate whose estimate is within 0.05 semitone of (a) —
the difference taken around the circle, so +0.49 and −0.49 are 0.02 apart — on every check file:
player 3's 12 takes, the same takes shifted +0.45 (Task 6 of the app plan), the 3-minute file and
Ege's two files; and that takes at most 1.0 s on the 3-minute file. If none qualifies, keep (a)
and report it.

- [x] Measure the four candidates; record times and the largest difference; apply the rule.
      *Result (26 files — Ege's first recording was no longer on disk): (a) 4.98 s; (b) 1.63 s,
      worst difference 0.070; (c) 0.14 s, 0.160; (d) 0.05 s, 0.160. **None qualifies: (a) stays.***
- [x] *Instead, outside the rule because it changes no estimate:* the check runs in a thread while
      Basic Pitch transcribes in its own process. 3-minute file, cold cache: 9.05 s → 5.17 s, same
      notes and offset. Test: the transcriber waits for the check to have started.

## Task 2: Uncertainty marks that point at errors

**Measurement:** end to end with the default decoder and Basic Pitch at `CHOSEN_PARAMS`, every
tab note scored right or wrong by E2's matching (`exact_tab_f1_with_matches`: onset within 50 ms,
pitch and string). For each threshold t in 0.30, 0.35, …, 0.90: the share of notes marked
(posterior < t), the error rate among marked and unmarked notes, and their ratio (the *lift*).
Player 3 decides; player 00 is reported beside it.

**Rule (fixed now):** the largest t that marks at most a quarter of player 3's notes; adopt it
as `uncertainty_threshold` if its lift is at least 1.5. If no t that marks at most a quarter
reaches 1.5, the posterior does not single out wrong notes well enough: report that to Ege and
change nothing.

**Prediction:** at today's 0.6 more than 40% of player 3's notes are marked, with a lift of at
least 1.5; the chosen t falls between 0.40 and 0.50. (Many end-to-end errors are extra notes,
which the posterior does not see, so the lift may be lower than in oracle mode.)

- [ ] `scripts/measure_uncertainty.py`; run; record; apply the rule.
- [ ] If adopted: the config value, the page's wording, `docs/using-the-app.md`; `make check`;
      commit.

## Task 3: Fewer extra notes from Basic Pitch

Two filters on the transcribed notes, before grouping, each its own measurement on player 3
end to end (E2, paired ratio bootstrap from `compare_validation.py`). Pure functions in
`transcribe/cleanup.py`.

- **3a — a loudness floor.** Drop notes whose `confidence` (Basic Pitch's amplitude) is below c,
  for c in 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, against no floor.
- **3b — octave ghosts.** On top of 3a's outcome: drop a note when another note exactly 12
  semitones lower starts within 50 ms of it and is at least as loud. No parameter.

**Rule (fixed now), for each:** adopt the best setting only if it beats the current pipeline by at
least +0.01 E2 on player 3. First, the split of extra notes at today's thresholds is measured with
`scripts/analyse_e2e.py`, and reported.

**Prediction:** 3a gains +0.01 to +0.03 with c between 0.15 and 0.30; 3b gains less than +0.01
and is not adopted.

**The test look:** only if something was adopted — the adopted pipeline against today's, end to
end, on EGDB (and GuitarSet's test players, labelled), once, pre-registered in its own commit with
its own predictions, logged to `experiments/test_set_access.log`.

- [ ] Extra-note split at today's thresholds.
- [ ] 3a: measure, record, apply the rule.
- [ ] 3b: measure, record, apply the rule.
- [ ] If adopted: wire into the pipeline; pre-register and run the test look; ADR.

## Not in this plan (one line each)

- The electric audio-evidence path (weights, fresh validation data, PyTorch at run time).
- Tuning and capo on the page; playback of the tab; opening the exports in MuseScore.

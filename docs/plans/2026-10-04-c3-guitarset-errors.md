# C3, Continued — Features Argued From GuitarSet's Own Playing

> Tasks run in order; steps use checkbox (`- [ ]`) syntax. Every run that produces a
> reported number is pre-registered in a commit made before it runs.

**Status:** in progress (2026-10-04).

**Goal:** continue Phase 2 task C3 (`docs/plans/2026-10-01-phase-2.md`) now that GuitarSet's
player 00 is validation data (ADR 0037): find where the default decoder loses human
fingerings on GuitarSet-like playing, and try only the features that evidence argues for,
each judged on player 00 by a rule fixed before it runs.

**Spec:** `docs/spec.md` §2.2 and §6, Phase 2. **Data:** player 00's 60 tracks (validation)
and DadaGP v1.1's artist split (ADR 0024) for fitting. Players 01–05 are not read here.

## Where this starts

- **The default** (hand-set weights, the hand window and its stretch, T = 1.5728; ADR 0038):
  oracle E2 0.8065 on player 00 and 0.6559 on the test players. M2 needs 0.760 on the test
  players.
- **C3 so far** (ADR 0034): a per-string preference and fret regions, fitted and judged on
  DadaGP; neither adopted. Not yet tried: open strings conditioned on the neighbouring
  shapes. The per-string biases trade off against `high` in a fit, so regularisation comes
  before more feature groups.
- **Every C3 choice so far was made on DadaGP**, a proxy that misled three times. Player 00
  is the first validation data that is GuitarSet's own kind of playing.

## Global constraints

- Players 01–05 are not read. Everything here is player 00 (validation) or DadaGP.
- One variable per experiment; hypothesis, variable, metric and split stated before a run.
- No feature group is tried that Task 1 does not argue for.

## Task 1: Where the default loses fingerings on player 00

**Files:** `scripts/analyse_errors.py` (new), `tests/test_scripts.py`.

A diagnostic, not a metric: no `results.csv` row, and nothing is chosen from it except which
feature groups are worth arguing for. Oracle mode, the default decoder, player 00's 60
tracks. A note is right when the decoder put it on the string the player used, as in E2.

**Questions, fixed before the run:**

1. Accuracy of single notes and of notes in chords, and each one's share of the errors.
2. Accuracy when comping and when soloing, and in each of the five styles.
3. For a misplaced note: how many strings away the decoder put it, and whether it played
   the note higher or lower on the neck than the player.
4. Accuracy by the player's fret region (open string, 1–4, 5–11, 12 and above), and how
   often a misplaced note also changed region.
5. Open strings: how often the player used an open string where the decoder fretted the
   note, and the reverse.
6. Whether errors come in runs (a passage placed in another position) or alone: the share of
   errors that sit in runs of four or more consecutive misplaced notes.

**Question 7, added after the first run** as a follow-up to question 5, which found a third
of all errors to be the decoder playing an open string where the player fretted the note:

7. For those notes, where each hand was — nothing fretted yet, open position (index fret
   1–4), or up the neck (5 and above) — as the hand window follows each path.

Prediction: mostly the decoder's hand at 1–4 and the player's at 5 or above, i.e. the
decoder chose a lower position, not an open string inside the player's position. What it
argues: with the decoder's hand mostly up the neck, a cost on open strings played there
(the Phase 2 plan's "open strings conditioned on the neighbouring shapes") targets these
errors; with it mostly in open position, that feature would not reach them, and the
position features of Task 3 are argued instead.

**Predictions:** notes in chords are placed better than single notes; soloing worse than
comping; most misplaced notes are one string away; and most errors sit in runs of four or
more, which would mean the decoder chooses the wrong *position* more often than the wrong
string for one note.

- [x] Write the script and its tests (offline, synthetic notes). Commit before the run.
- [x] Run it; check that its overall accuracy agrees with ADR 0038's E2 for the default
  (0.8065). *(The first run was one note short, 10,663 against 10,664: a unison on two
  strings paired in order. Fixed with a test; the rerun agrees exactly.)*
- [x] Question 7: add it to the script, commit, run; write all seven answers into the devlog.

## Task 2: Open strings played up the neck (ADR 0039)

**Argued by** questions 5 and 7: a third of the default's errors are an open string where the
player fretted the note, and in 430 of those 860 the decoder's hand was up the neck.

**Files:** `docs/adr/0039-open-strings-up-the-neck.md`, `types.py` (`CostWeights.open_up_neck`),
`fingering/costs.py`, `fingering/fit.py`, `config.py`, `results.py`,
`scripts/fit_cost_weights.py` (`--hold-base`, `--features open`), `tests/decode/brute_force.py`
and their tests.

- [x] ADR 0039, proposed: the contract, where the cost is charged, the experiment, its rule.
- [x] Implement it in the scorer, the fitter and, from the ADR's text, the oracle. Oracle
  tests with the term switched on; mutations of its threshold and of where it is charged
  are caught.
- [x] Ege sets the rule's chord-shape condition: **no clear drop**, judged by a track-level
  interval like the E2 gain (2026-10-04). Per-track chord-shape counts added to the
  per-track files, and their interval to `compare_validation.py`; the rule and
  `configs/m2_validation_open.yaml` committed before any run.
- [x] Fit the one weight on DadaGP's clean training parts with the base weights held
  (`--hold-base --features open --part clean`); write the challenger's config with the
  fitted value; commit it before the runs. *(0.7615; `configs/decoder_clean_open.yaml`.)*
- [x] `eval-m1 --split validation --per-track-out` for the default and the challenger;
  `scripts/compare_validation.py`; apply the rule; results rows; accept ADR 0039.
  *(Adopted: oracle E2 0.8065 → 0.8273, interval [+0.0066, +0.0363], no clear chord-shape
  drop; recalibrated to T = 1.2934.)*

## Task 3: Regularisation and position features, if Task 2 leaves the gap

Argued by question 4 (69% of errors with the player at frets 5–11, 87% of misplaced notes
in another region): an L2 penalty in the fitter first, so that correlated features such as
the per-string biases and `high` can be fitted together, then position features judged on
player 00 like Task 2's. Detailed after Task 2.

## Then

C5 for whatever ships (its temperature, on player 00), and the judgement the Phase 2 plan
asks for: has C3 plateaued short of M2, so that C4 (a learned sequence model) is due?

## Review focus

1. **Nothing reads players 01–05.** The script must take its tracks from
   `guitarset_validation_ids()` only.
2. **The analysis agrees with E2.** Its overall accuracy on player 00 must match the
   default's oracle E2 there, or the pairing of notes is wrong.

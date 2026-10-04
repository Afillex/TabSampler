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

**Predictions:** notes in chords are placed better than single notes; soloing worse than
comping; most misplaced notes are one string away; and most errors sit in runs of four or
more, which would mean the decoder chooses the wrong *position* more often than the wrong
string for one note.

- [ ] Write the script and its tests (offline, synthetic notes). Commit before the run.
- [ ] Run it; check that its overall accuracy agrees with ADR 0038's E2 for the default
  (0.8065); write the answers into the devlog.

## Task 2: Regularisation in the fitter

To be detailed after Task 1. An L2 penalty in `fingering/fit.py`, with its gradient checked
against the brute-force oracle, so that correlated features (the per-string biases and
`high`) can be fitted without trading one against the other.

## Task 3: Feature groups argued by Task 1, judged on player 00

To be detailed after Task 1. Each group gets its own ADR with a rule fixed before its run,
in the form of ADR 0038's: a challenger replaces the default only with the 95% track-level
paired interval of its oracle E2 gain wholly above zero, and a chord-shape rate not lower
by more than an allowance.

## Then

C5 for whatever ships (its temperature, on player 00), and the judgement the Phase 2 plan
asks for: has C3 plateaued short of M2, so that C4 (a learned sequence model) is due?

## Review focus

1. **Nothing reads players 01–05.** The script must take its tracks from
   `guitarset_validation_ids()` only.
2. **The analysis agrees with E2.** Its overall accuracy on player 00 must match the
   default's oracle E2 there, or the pairing of notes is wrong.

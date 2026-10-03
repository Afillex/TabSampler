# A Held-Out GuitarSet Player as Validation — Implementation Plan

> Tasks run in order; steps use checkbox (`- [ ]`) syntax. Every run that produces a
> reported number is pre-registered in a commit made before it runs, with its tolerance and
> noise estimate stated in advance.

**Goal:** give the project validation data that resembles its test set — GuitarSet's player
00 — re-decide the default decoder on it, and re-base the test set on the other five
players.

**Architecture:** `data/splits.py` keeps the 360-track snapshot as the corpus and derives
two disjoint lists from it by player: validation (player 00, 60 tracks) and test (players
01–05, 300 tracks). `eval-m1` gains `--split validation`, which evaluates the validation
player without touching the test-set log, and `--per-track-out`, which writes per-track
counts that `scripts/compare_validation.py` already knows how to compare.

**Tech Stack:** Python 3.13, `uv`, numpy, scipy, pytest + hypothesis.

**Spec:** `docs/spec.md` §3.3 (splits); amends ADR 0003. Decided by Ege on 2026-10-03.

## Decisions already taken by Ege (2026-10-03)

- Hold out one GuitarSet player (60 recordings) as validation; test on the other five
  (300); re-measure the latest figures on the 300; re-decide the default on the player.

## Global Constraints

- Everything in `docs/plans/2026-10-03-timing-stretch-style-features.md`'s Global Constraints.
- Players 01–05 remain test-only: tuned on never, read only by pre-registered, logged runs.
- **Contamination, said every time a test figure is quoted:** the 300 test tracks were part
  of every earlier 360-track run, so decisions made now are made knowing those runs.
- New ADRs: **0037** the split, **0038** the default re-decided on player 00.

## Review Focus

1. **A snapshot in which some player has other than 60 tracks** must be refused, not split
   unevenly. → Task 1 `test_a_player_with_the_wrong_track_count_is_refused`.
2. **Validation and test must never share a track**, and together must be the corpus.
   → Task 1 `test_validation_and_test_partition_the_corpus`.
3. **`eval-m1 --split validation` must not write to the test-set access log.**
   → Task 2 `test_a_validation_run_does_not_touch_the_test_set_log`.
4. **The per-track counts must reproduce the pooled E2** of the run they came from.
   → Task 2 `test_per_track_counts_pool_to_the_reported_f1`.
5. **Tuning against the test split is still refused.** → existing
   `test_tuning_against_the_test_split_is_refused`.

---

### Task 1: The split (ADR 0037)

**Files:** `docs/adr/0037-guitarset-validation-player.md` (+ index; ADR 0003's status line),
`src/tabsampler/data/splits.py`, `src/tabsampler/data/guitarset_test_ids.txt` (header only),
`tests/data/test_splits.py`.

**Interfaces — produces:** `VALIDATION_PLAYER = "00"`;
`guitarset_track_ids(snapshot=None, expected_count=360) -> tuple[str, ...]` (the reader,
renamed from today's `guitarset_test_ids`); `split_by_player(ids, player) -> tuple[tuple[str,
...], tuple[str, ...]]` (validation, test), refusing a player with other than 60 tracks;
`guitarset_validation_ids(snapshot=None)` (60) and `guitarset_test_ids(snapshot=None)` (300).

- [ ] Write ADR 0037 (accepted: Ege's decision) — the player chosen by rule as the lowest
  ID, before any per-player figure is seen; ADR 0016's values kept and judged on the 300;
  every new test figure shown beside the same decoder's 360-track figure where one exists;
  the contamination caveat. Point ADR 0003's status line at it.
- [ ] Failing tests: the real snapshot gives 60 validation tracks, all `00_`, and 300 test
  tracks, none `00_`; the two partition the corpus; a fixture where one player has 59 tracks
  is refused; the validation player is pinned at `"00"`.
- [ ] Implement; rename the reader; fix the module comment's arithmetic (360 = 6 players × 5
  styles × 3 progressions × 2 tempi × comp/solo, not "6 × 2 × 5 × 3"); `check-split` prints
  both counts; `eval-notes` reads `guitarset_test_ids()`; `scripts/measure_lattice.py` reads
  the validation player and no longer logs a test-set access.
- [ ] `make check`; commit `Hold out GuitarSet's player 00 as validation (ADR 0037)`.

### Task 2: `eval-m1 --split validation` and per-track counts

**Files:** `src/tabsampler/cli.py`, `src/tabsampler/eval/recovery.py`,
`scripts/compare_validation.py`, `tests/eval/test_recovery.py`, `tests/test_cli.py`.

**Interfaces — produces:** `add_track(report: RecoveryReport, track_id: str, mode: str,
e2: PRF, e3: PlayabilityReport) -> None` — per track and mode, `[2 × matches, estimated +
reference notes]`, whose pooled ratio is the micro-averaged E2 F1, and chord-shape counts.

- [ ] Failing tests: `add_track` pools to the reported F1; a validation run writes no
  test-set log line (the access logger is replaced in the test); `compare_validation`
  compares by mode when the parts are `oracle`/`e2e`.
- [ ] Implement `--split {test,validation}` (default test; rows record the split) and
  `--per-track-out PATH`.
- [ ] `make check`; commit `Evaluate on the validation player and write per-track counts`.

### Task 3: Re-decide the default on player 00 (ADR 0038)

- [ ] Write ADR 0038 as proposed: candidates — the clean-fitted weights (today's default)
  and the hand-set weights, both with the window and the stretch. **Rule:** the clean-fitted
  weights stay the default iff, on player 00's 60 tracks in oracle mode, E2 is higher than
  the hand-set weights' with the 95% track-level paired interval wholly above zero, and the
  E3 chord-shape rate is not lower by more than 0.0005; otherwise `configs/decoder_clean.yaml`
  takes the hand-set weights and a temperature calibrated for them on DadaGP clean parts by
  ADR 0033's method. Prediction, with the contamination caveat: the clean fit loses, as it
  did on all 360 tracks. Commit before the runs.
- [ ] Run `eval-m1 --split validation` for both decoders with `--per-track-out`; compare;
  apply the rule; if reverted, calibrate and write the config; results rows; accept ADR 0038.

### Task 4: The latest figures on the 300 test tracks

⚠️ Reads the test set — logged, nothing chosen from it.

- [ ] Pre-register `configs/m2_heldout_eval.yaml`: the default as Task 3 leaves it, and the
  other candidate, on players 01–05; prediction: each within 0.01 of its 360-track figure.
- [ ] Run both; refresh the README in the same commit — the 300-track table, the 360-track
  figures kept as history, ADR 0016's values judged on the 300, the contamination caveat.

### Task 5: Close out

- [ ] Devlog, HANDOFF, plan boxes; check that no local tooling files are tracked.
- [ ] `make check`, `make oracle`; independent review; one fix pass; merge into `main` and
  push both this branch and the one held back on 2026-10-03.

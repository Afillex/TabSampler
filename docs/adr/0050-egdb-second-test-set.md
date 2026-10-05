# ADR 0050: EGDB is the second test set — all of it, clean direct input

Status: accepted (2026-10-06)

Carries out Phase 4's Task 4 (`docs/plans/2026-10-06-phase-4-electric.md`) and the rest-of-project
plan's Chunk E: "add EGDB to `data/splits.py` with the same snapshot-and-guard mechanism GuitarSet
has — do not weaken the guard to accommodate a second test set". Spec §3.3 names EGDB the second
test set, to catch overfitting to GuitarSet's acoustic guitar; ADR 0049 made clean electric
Phase 4's target.

## Decision

- **Every EGDB clip is test data**: 240 clips, numbered 1–240 in the dataset's Google Drive
  folder, held in a committed snapshot (`src/tabsampler/data/egdb_test_ids.txt`, ids
  `egdb_001`…`egdb_240`) whose count is checked, as GuitarSet's is. There is no validation part:
  choices are made on Guitar-TECHS's validation player and GuitarSet's player 00.
- **The guard covers it**: `assert_tuning_allowed(Split.TEST)` names both test sets;
  `assert_no_test_tracks` refuses any EGDB id; every look is logged by
  `record_test_set_access`, with a written reason, as for GuitarSet.
- **The input is the direct-input recording** (`audio_DI`), the clean electric sound ADR 0049
  targets. The five amplifier renderings are not downloaded; reading them would be a later
  decision, for distorted guitar.
- **Labels**: one MIDI track per string, `1` the high e to `6` the low E, standard tuning
  (`data/egdb.py`), checked on the first clip: every pitch lies on its string, and the notes end
  inside the audio. The dataset's own clip-wise split (clips 216–241 for testing, per its README)
  is not used: all of it is test here.
- **Terms**: the project page states no licence. Nothing derived from EGDB is published but
  metrics, and its audio stays out of the repository (`data/` is ignored).

## Alternatives considered

- **Hold some EGDB clips for validation.** One player plays every clip, so a validation part
  would be the test set's own playing — exactly what ADR 0037 avoided for GuitarSet.
- **The amplifier renderings as the test input.** Distortion is not the target yet (ADR 0049).

## Consequences

**Easier.** Phase 4 can report electric and acoustic guitar side by side, each on audio nothing
was chosen with.

**Harder.** Two test sets mean two logged looks per pre-registered evaluation, and the
README must report both without letting either stand for the other.

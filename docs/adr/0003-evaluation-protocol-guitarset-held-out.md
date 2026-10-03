# ADR 0003: GuitarSet is held out entirely as a test set

Status: accepted (2026-09-27); amended by [ADR 0037](0037-guitarset-validation-player.md):
GuitarSet's player 00 is validation data, players 01–05 stay test-only.

Decides spec D8. **This is the most consequential decision in the project.**

## Context

GuitarSet is the only dataset that gives us string-level annotations for real acoustic
guitar audio, which makes it simultaneously the only honest way to measure E2 (Exact Tab
F1) and the most tempting thing to tune against. Spec 3.3 states
the rule; this ADR records the mechanism, because a rule that depends on remembering it at
1 a.m. is not a mechanism.

If GuitarSet is used for tuning, model selection, or threshold sweeps, then every number
reported afterwards is a training-set number wearing a test-set label. The M1 claim
("a rigorous evaluation harness") collapses, and there is no way to recover the result
short of finding another annotated dataset.

## Decision

**GuitarSet is test-only.** No tuning, no hyperparameter search, no threshold sweep, no
model selection, no early stopping, and no architecture choice may consult a GuitarSet
number. There is no GuitarSet train or validation split — not "we don't use one", but
**no such thing exists in this project**.

Enforced mechanically in `src/tabsampler/data/splits.py`:

- `guitarset_test_ids()` reads a committed snapshot file, never a glob over whatever
  happens to be downloaded, so the split cannot drift.
- `Split` has `TEST`, `VALIDATION` and `TRAIN`; `assert_tuning_allowed(split)` raises
  `TestSetMisuseError` for `TEST`. Every tuning entry point calls it first.
- `record_test_set_access(reason)` appends to `experiments/test_set_access.log`, so
  looking at test numbers is a deliberate, auditable act.

Test-set evaluation happens **at milestones only**. Validation data comes from training
sources: DadaGP for symbolic fingering work, and GOAT / GAPS / Guitar-TECHS from Phase 4.
EGDB joins as a second test set at Phase 4 to catch overfitting to acoustic guitar.

`audio_hex` and `audio_hex_cln` must never be fed to a transcriber under evaluation: the
hexaphonic pickup has one channel per string, which is the ground truth for E2. See
ADR 0005.

## Alternatives considered

- **Six-fold cross-validation by player, as in TabCNN.** Rejected as the primary protocol
  because it requires training on GuitarSet, which forfeits the only clean test set we
  have. It becomes the right protocol only if we ever deliberately train on GuitarSet, and
  that would need a new ADR superseding this one.
- **Both protocols.** Rejected for now: it doubles the reporting burden and invites quoting
  whichever number looks better.
- **Carve a small validation slice out of GuitarSet "just for the cost weights".** This is
  the failure mode this ADR exists to prevent. Cost-weight tuning needs only *symbolic*
  tabs, so DadaGP serves that purpose with no audio at all. See ADR 0011.

## Consequences

**Easier.** Every reported number means what it says. Oracle mode keeps the fingering work
measurable even when the transcriber is poor (spec 3.2).

**Harder.** Phase 1 has no legal tuning data at all until DadaGP access arrives, because
GOAT and friends are Phase 4. M1 therefore ships with hand-set cost weights — a real
result, not a placeholder. Recorded in ADR 0011.

**Revisit:** only via a superseding ADR. Never edit this one.

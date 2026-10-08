# ADR 0067: Phase 6's result — separating the guitar first raises full-song E2 on both test sets

Status: proposed — Phase 6's gate, for Ege's sign-off

Reports Phase 6 (`docs/plans/2026-10-07-phase-6-full-songs.md`). Every rule was committed before
its run; the test look was pre-registered in 59cccb8 and run on the CPU by two pre-registered
checks (15e014d, 8129105). Mixes: ADR 0066. Separation: ADR 0065.

## Result

**The test look** (2026-10-07/08; access-log lines 37–42): EGDB's 240 clips and GuitarSet's 300 test
tracks (labelled, ADR 0055), each over a BabySlakh song from 11–20, at two guitar-to-backing levels.
End-to-end E2, the default decoder, Basic Pitch at `CHOSEN_PARAMS`; intervals are 95% bootstraps over
takes, reported, nothing chosen from them.

| E2, test | isolated take | isolated, separated | mix 0 dB | **stem 0 dB** | mix −6 dB | **stem −6 dB** |
|---|---|---|---|---|---|---|
| EGDB | 0.4547 | 0.4269 | 0.2578 | **0.3598** (+0.1019 [+0.0801, +0.1232]) | 0.1690 | **0.3322** (+0.1632 [+0.1379, +0.1878]) |
| GuitarSet 01–05 (labelled) | 0.4553 | 0.4521 | 0.2281 | **0.3769** (+0.1488 [+0.1322, +0.1649]) | 0.1476 | **0.3298** (+0.1823 [+0.1624, +0.2012]) |
| both, pooled | 0.4550 | 0.4413 | 0.2410 | **0.3695** (+0.1285 [+0.1147, +0.1414]) | 0.1567 | **0.3309** (+0.1741 [+0.1587, +0.1890]) |

The electric option on EGDB's stems (reported, not judged): 0.3843 at 0 dB, 0.3528 at −6 dB.
The isolated figures reproduce the recorded ones exactly (EGDB 0.4547, ADR 0057; GuitarSet 0.4553),
a check that the look read the same takes the same way.

**Predictions** (from validation):

- EGDB, stem against mix between −0.04 and +0.04 at each level — **failed**: +0.10 and +0.16. On
  validation, IDMT's direct-input licks did not gain from separation; EGDB's direct-input clips did.
- Separation alone costs EGDB's isolated take at least 0.05 — **failed**: it cost 0.0278.
- GuitarSet, stem beats mix by +0.10 to +0.20 at each level — **held**: +0.15 and +0.18.

Both failures went in the stem's favour: IDMT's eleven licks (ADR 0062) were a poor stand-in for EGDB.

**Validation** (Tasks 2–3): stem against mix +0.13 pooled at both levels; adding a share of the mix
back to the stem lowered E2 at every share, so the stem alone is what the app uses.

## Decision

1. **Full-song mode separates the guitar with `htdemucs_6s` and transcribes the stem alone**
   (α = 0, Task 3) — the page's "full song" option and `--full-song` (Task 5, e7425c6).
2. **Full-song figures are their own table**, never merged with isolated ones (ADR 0002), always
   with ADR 0066's limits beside them.
3. **Phase 6's done-when** (proposed in the plan) is met if Ege signs off: on both test sets the mix,
   the stem and the chosen stem-plus-mix (the stem alone) are reported beside the isolated figure,
   and the app accepts a full song.

## What these figures cannot show

- **Built mixes, not songs**: guitar and backing share no key or tempo; the backing is synthesized
  at 16 kHz; there is one guitar (ADR 0066). Real songs may separate worse. MoisesDB is the check.
- **Stems are reproducible only from their cache**: demucs applies one unseeded random time shift
  (`shifts=1`); a rerun on the CPU moved a validation figure by up to 0.0176 (8129105). The figures
  are for the stems in `cache/stems/` as separated.
- **GuitarSet is labelled**: Basic Pitch trained on most of it (ADR 0055). EGDB is the clean test.
- The scoring ran across four restarts from its caches — after a power loss, the move back to the
  CPU, a system crash, and an overnight pause that was lost — each a logged line (39–42); no figure
  was printed or read before the end.

## Consequences

Separating the guitar first keeps about four-fifths of the isolated score at 0 dB (0.3695 against
0.4550 pooled, where the mix alone keeps a half) and nearly three-quarters at −6 dB (0.3309, where
the mix alone keeps a third). EGDB and GuitarSet have now been read once more each, for this
phase.

**Revisit** with real multitrack songs (MoisesDB), a better separator, or a transcriber trained on
separated stems.

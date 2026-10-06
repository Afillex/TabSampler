# ADR 0062: Electric string data — EGFxSet trains; EGSet12 and IDMT-SMT-Guitar validate

Status: accepted (2026-10-06) — point 3 decided by Ege: player 3 moves to training

Plan: `docs/plans/2026-10-06-electric-audio-evidence.md`, Task 1. Ege chose open data only.

## Context

Phase 4's string classifier helped the decoder on EGDB (oracle E2 0.6762 → 0.7200, ADR 0054), but
its weights start from SynthTab's, it was trained on 14,788 notes of drills from two players, and
the only real electric music to judge it on was Guitar-TECHS's player 3, which has since chosen
three things (ADRs 0054, 0057, 0061). The search (plan, Task 1) found three open or evaluation-only
datasets with string labels. Each was downloaded and checked against its own audio
(`scripts/check_electric.py`, 2026-10-06), against GuitarSet player 00's control (lag +11.6 ms,
0.949 of pitches confirmed):

| dataset | licence | read | labels against audio |
|---|---|---|---|
| EGFxSet, clean | CC BY 4.0 | 690 single notes, one Stratocaster, 5 pickup settings, direct input | one clear onset per file, 70–139 ms in |
| EGSet12 | CC BY 4.0 | 12 solo pieces, 1,567 notes, 367 s; amplifier and microphone | 0.918 confirmed; lag +23 ms (+46 on one take) |
| IDMT-SMT-Guitar, `dataset2` licks | CC BY-NC-ND 4.0, "for evaluation purpose" | 135 takes of normal notes, 2,811 notes, about 23 min, 3 guitars, 11 distinct licks | 0.980 confirmed; lag +35 to +58 ms |

String numbering differs and was checked on every note: EGFxSet's string 1 is the high e (measured
from its open strings' pitch), IDMT's string 1 the low E, EGSet12's `data_source` 0 the low E.

## Decision

1. **Training adds EGFxSet's clean notes** to Guitar-TECHS players 1–2. All CC BY 4.0, so the
   weights may be published with attribution (D15, still Ege's).
2. **Validation is EGSet12 and IDMT's licks**, each reported on its own and pooled. IDMT is used
   for evaluation only, as its licence says; nothing is trained on it. Its 135 takes are 11 licks
   played several ways, so they are far fewer independent pieces than takes.
3. **Player 3 moves to training** (Ege, 2026-10-06; amends ADR 0051's split for the classifier). It is Guitar-TECHS's only music,
   and as a judge it is spent: three choices made on it, one of which did not carry to EGDB.
   Training stops on a hashed 15% of the training takes instead.
4. **Every take's label delay is measured from its own audio** (`onset_lag`, as ADR 0052) and taken
   off its onsets wherever windows are cut or notes scored: IDMT's labels run about 35 ms ahead of
   its sound, close to E2's 50 ms tolerance.
5. EGDB and GuitarSet stay the test sets; GOAT is not requested.

## Alternatives considered

- **GOAT** (5.9 h, CC BY-NC 4.0, by request): the best training data found, but weights trained on
  it could not be released for commercial use. Ege chose open data only.
- **Keeping player 3 as validation**: its three uses make it a weak judge (ADR 0061).
- **Training on IDMT**: its no-derivatives licence rules it out.

## Consequences

The classifier sees four guitars instead of two and every fret of one of them. Validation becomes
real music from two sources other players recorded. EGSet12 is microphone-on-amp, unlike the
direct input the app is built for, so its figures and IDMT's are reported apart.

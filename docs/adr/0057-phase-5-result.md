# ADR 0057: Phase 5's result — stricter Basic Pitch thresholds improve end to end; fine-tuning did not

Status: proposed — Phase 5 closes on this record if Ege signs off

Reports Phase 5 (`docs/plans/2026-10-06-phase-5-transcriber.md`). The spec's "done when":
end-to-end Tab F1 improves — note F1 alone is not enough. Basic Pitch trained on most of GuitarSet
(ADR 0055), so **EGDB is the clean test** and GuitarSet's figures are labelled.

## Result

**The transcriber:** the released Basic Pitch with its note thresholds chosen on Guitar-TECHS's
player 3 by a pre-registered rule — onset 0.7, frame 0.4, minimum note length 58 ms, against its
defaults 0.5, 0.3 and 127.7 ms. The default decoder end to end:

| end-to-end E2 (E1) | Basic Pitch's defaults | onset 0.7 / frame 0.4 / min 58 ms | change |
|---|---|---|---|
| Guitar-TECHS player 3 (validation; chosen on it) | 0.3491 (0.6247) | 0.4185 (0.7737) | +0.0694 [+0.0081, +0.1303] |
| **EGDB, 240 clips (test, clean)** | **0.4251 (0.7230)** | **0.4547 (0.7416)** | **+0.0296** |
| GuitarSet players 01–05 (test, labelled) | 0.4418 (0.7493) | 0.4553 (0.7812) | +0.0135 |

No interval is computed on the test sets and nothing was chosen from them. Oracle mode is unchanged
(EGDB 0.6762, GuitarSet 0.6819). On GuitarSet every ADR 0016 value still holds; end-to-end
calibration error falls from 0.2439 to 0.1841, notes out of range from 105 to 46, notes dropped
from 100 to 23. **Phase 5's done-when is met on EGDB.** The transcriber still costs 0.22 of E2 on
both sounds.

**Where the loss was** (Task 2, `scripts/analyse_e2e.py`): on electric direct input Basic Pitch
writes half as many notes again as were played — precision about a half — and the stricter
thresholds remove most of them; on acoustic recordings it misses long notes, and its errors cost
the decoder more context.

**Fine-tuning** Basic Pitch on Guitar-TECHS's players 1–2 (ADR 0056) failed twice, each a clear loss
on player 3 at its own best thresholds: with the paper's class-weighted onset loss the onset
posteriors inflated (18.7% of onset frames above 0.5, against the released model's 0.35%) — E2
0.3079; with the unweighted loss of Basic Pitch's `train.py` the onset head collapsed (never above
0.405) — E2 0.3052. By Ege's rule fine-tuning closes as a negative result.

**Predictions:** of Phase 5's pre-registered predictions, those that failed are stated in
`results.csv` — among them the size of Task 3's gain (+0.069, above 0.01–0.05), every fine-tuning
hypothesis, and GuitarSet's test players, which gained where no gain was predicted.

## Decision

- **The `transcribe` command uses onset 0.7 / frame 0.4 / min 58 ms** once Phase 5 closes; the
  eval configs that recorded earlier figures keep their settings, so those figures stay
  reproducible.
- **The spec's Phase 5 deliverable exists**; closing the phase on it is Ege's decision.

## What it does not settle

The thresholds were chosen on one player's 12 takes. Heavier transcribers (Task 5) were not
surveyed. Fine-tuning with an onset weight between the two tried, or with the onset head frozen,
was not tried.

## Consequences

**Easier.** A user's tab improves with no new model, and the fine-tuning pipeline — data, training,
CoreML export, the `--model-path` adapter — is in place and verified for another attempt.

**Harder.** Both test sets have been read with this transcriber; another one needs its own look.

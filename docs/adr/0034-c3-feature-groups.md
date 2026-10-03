# ADR 0034: Two richer feature groups for the cost model — strings and fret regions

Status: accepted (2026-10-03) — **one group kept, for distorted only, and not adopted:
both decoders are unchanged**

Starts Phase 2 task C3 (`docs/plans/2026-10-01-phase-2.md`): richer features in the same
CRF, one feature group per experiment, each kept only if it improves validation recovery.
Changes the `CostWeights` contract (ADR 0007). Approved in principle by Ege on 2026-10-03.

## Context

The cost model has four weights: hand movement, span, neck height (linear in the mean
fret) and a reward per open string. It cannot express two things guitarists plainly do:
prefer some strings to others for the same pitch, and treat open position (frets 1–4) and
the upper neck (fret 12 and above) differently from what a straight line in fret height
allows. `fingering/fit.py` fits any feature that is linear in its weight, with exact
gradients and oracle checks, so adding features is cheap and keeps the model readable.

## Decision

**Contract.** `CostWeights` gains `string_bias`, six numbers indexed by string, low E first,
and `low_region` and `high_region`. All default to zero, so every existing config decodes
exactly as before. A shape's emission cost adds:

- `string_bias[s]` for each of its notes on string `s`, open or fretted;
- `low_region` for each fretted note at frets 1–4, and `high_region` for each at fret 12
  or above; frets 5–11 are the reference.

**Identifiability.** A group's notes always number the same whichever shape plays them, so
the six string counts sum to a constant and only five can be fitted: the fitter pins the
low E string's bias at zero. Likewise the three fret regions plus the open strings sum to
the group size, which is why the middle region has no weight.

**Experiments, one feature group each, fixed before they run.** For each style, refit on
that style's parts of the same 600 artist-split training songs as ADR 0032, with the group
added, and compare against the same style's fit without it on that style's parts of the
300 artist-validation songs.

1. **Strings** (5 weights): hypothesis — a per-string preference raises recovery on both
   styles.
2. **Fret regions** (2 weights), added to whatever group 1 left: hypothesis — they raise
   recovery on clean parts, where open-position chords are common.

**Keep rule, per style:** a group is kept iff recovery is higher with the 95% song-level
paired interval wholly above zero, and the decoded chord-shape rate is not lower by more
than 0.0005. **Adoption:** a style's best kept model replaces that style's decoder only by
ADR 0032's fair test against the decoder it would replace, and its temperature is then
recalibrated by ADR 0033's method.

## Alternatives considered

- **Free weights for every string and fret.** Over a hundred weights; the lattice would
  fit habits of the training artists rather than of guitar playing. Five and two first.
- **A per-string preference that depends on the style of the passage.** That is what fitting
  per style already does.
- **Open strings conditioned on the neighbouring shapes**, listed in the Phase 2 plan: a
  transition feature, which needs edge features beyond movement in the fitter. Later.

## Consequences

**Easier.** Each experiment is a flag on `scripts/fit_cost_weights.py` (`--features`), and a
group that does not help costs nothing: its weights stay at zero.

**Harder.** Eleven weights instead of four to read and explain; and a contract field that
may stay at zero in the shipped configs if no group earns its place.

## Results (2026-10-03)

Each against the same style's four-weight fit, on that style's parts of the 300
artist-validation songs (delta, 95% song-level paired interval):

| feature group | clean parts | distorted parts |
|---|---|---|
| strings | 0.8616 → 0.8512, −0.0104 [−0.0244, +0.0026]: **not kept** | 0.6480 → 0.6669, +0.0190 [−0.0052, +0.0420]: **not kept** |
| fret regions | 0.8616 → 0.8638, +0.0022 [−0.0006, +0.0056]: **not kept** | 0.6480 → 0.6735, +0.0256 [+0.0027, +0.0478]: **kept** |

Neither the per-string hypothesis nor the clean-region one held. The per-string group did
raise the likelihood of human fingerings on validation, all parts together (negative
log-likelihood per group: clean-style fits 0.6492 → 0.6311, distorted-style 0.5690 →
0.5516), without a recovery gain the bootstrap could trust: likelihood is what the fit
maximises, recovery is what the keep rule judges, and here they parted. The region group
barely moved the likelihood (0.6492 → 0.6473; 0.5690 → 0.5698). *(This paragraph was
corrected before review: it first said both groups raised the likelihood, which the
distorted region fit did not.)*

**Fret regions are kept for distorted guitar but not adopted.** Against the decoder they
would replace — the hand-set weights, which the distorted four-weight fit already lost to
in ADR 0032 — they recover far more of the human fingering, 0.5705 → 0.6735 (+0.1030,
[+0.0494, +0.1552]), but lower the decoded chord-shape rate by 0.00062, beyond the 0.0005
ADR 0032 allows. It is the same verdict, for the same reason, as the four-weight fit.

So `configs/decoder_clean.yaml` and `configs/decoder_distorted.yaml` are unchanged, and so
are their temperatures. The new weights stay in the contract at zero, ready for the next
feature groups. The 11-weight code reproduced the four-weight clean fit song for song.

# ADR 0034: Two richer feature groups for the cost model — strings and fret regions

Status: proposed (2026-10-03) — pre-registration; accepted with its results

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

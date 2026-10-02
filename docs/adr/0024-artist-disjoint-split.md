# ADR 0024: An artist-disjoint DadaGP split for every fit from now on

Status: accepted (2026-10-02)

Supersedes ADR 0021's choice of split for fitting. ADR 0021's filter, note handling and
"GuitarSet is evaluated once, never consulted" rule all stand; the shipped split stays
loadable (`scheme="shipped"`) so the 2026-10-01 numbers remain reproducible.

## Context

DadaGP's shipped split is by song: 1,164 of 4,869 artists appear on both sides. ADR 0021
accepted that for fitting four scalar weights and made fixing it a precondition for any model
with real capacity, because such a model can learn an artist's habits and then be "validated"
on the same artist. The Phase 2 plan is about to give the cost model more to learn — a hand
window now, richer features after — so the split moves first.

## Decision

**Whole artists go to one side.** The artist is the folder under the initial letter,
case-folded, because folders differing only by case are one artist that a case-insensitive
filesystem has already merged (ADR 0021). Each artist's SHA-256 bucket modulo 10 decides its
side: bucket 0 goes to validation, the other nine to training. The assignment depends on
nothing but the artist's name, so it does not change with order, sampling or the filter.

**Frozen by hash**, as the shipped split is: the SHA-256 of the sorted, newline-joined
validation keys is `538be675…b087`, committed as `ARTIST_VALIDATION_SHA256` in
`data/dadagp.py` and checked on every load. A different assignment is a different corpus and
fails loudly.

| | training | validation |
|---|---|---|
| artists | 4,347 | 519 |
| songs listed | 23,574 | 2,607 (10.0%) |
| songs cleared by ADR 0021's filter | 19,794 | 2,240 |

Both sides come from the union of DadaGP's own two lists, each still checked against its
ADR 0021 hash first.

**Every fit, calibration and validation from now on uses `scheme="artist"`** (scripts take
`--split artist`). Numbers measured on the shipped split stay in `results.csv` labelled as
such and are not compared with artist-split numbers as if they were the same measurement.

## Re-baseline, pre-registered by committing this ADR before the run

Hypothesis: on the artist-disjoint validation side, the hand-set and fitted weights show the
pattern the shipped split showed — fitted better overall and on distorted parts, worse on
clean parts — and the artist overlap inflated the shipped split's validation scores for four
scalar weights by little, if at all. Single variable: the split. Metric: per-note recovery of
the human fingering, overall and by part; split: artist validation. These numbers are the
baseline Task 6 of `docs/plans/2026-10-02-hand-window-and-calibration.md` compares against.

## Alternatives considered

- **Split by notes rather than artists, to hit exactly 10% of notes.** Rejected: any rule that
  lets an artist's songs fall on both sides reintroduces the leak this ADR removes.
- **Stratify by genre.** Rejected for now: DadaGP's genre tokens are sparse and mostly
  `unknown_genre`; stratifying on them would add machinery without a reliable signal.
- **Keep the shipped split.** Rejected on ADR 0021's own terms, now that models with more
  capacity are next.

## Consequences

**Easier.** A validation score from here on means "artists the model has never seen", which
is the question a GuitarSet player — also never seen — asks of it.

**Harder.** The validation side is 519 artists rather than a random tenth of songs, so it is
lumpier: one prolific artist's style can move it. Reports give the part breakdown and the
note counts so that is visible.

**Revisit** if ProgGP or another corpus is added: their artists must be assigned by the same
rule, and the frozen hash changes, which needs a superseding ADR.

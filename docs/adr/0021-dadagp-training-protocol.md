# ADR 0021: DadaGP v1.1 is the training corpus — its own split, frozen, clean guitar only

Status: accepted (2026-10-01)

Implements ADR 0003's "tune on validation drawn from training sources" now that one exists.
GuitarSet remains the only test set.

## Context

DadaGP v1.1 arrived on 2026-10-01 (archive SHA-256 `81d999da…f25062`); ProgGP did not. It is
26,181 GuitarPro songs plus a token file per song, and it unblocks everything that needed
legal non-test data: fitting the cost weights, calibrating the temperature, validating
ADR 0011's playability rules, and Phase 2.

Before a single number is measured on it, the protocol has to be fixed — otherwise the
choice of split or filter could drift towards whatever produces a good result. Three facts
about the data, all verified rather than assumed, decide the protocol:

1. **It ships its own split.** `_DadaGP_training.json` lists 23,719 songs and
   `_DadaGP_validation.json` 2,462; they are disjoint and together cover every song.
2. **That split is by song, not by artist.** 1,164 of 4,869 artists appear on both sides.
3. **The tokens are E-standardised, and lose physical fingering silently.** Read from the
   dadaGP encoder's source: a fret token is `GP value + capo − drop shift`, with `s1` the
   highest string. Pitch survives, but a capo'd track's open strings become fretted notes
   at the capo, and a drop-tuned track's low-string frets are shifted by two — and nothing
   in the tokens marks either kind of track. Measured on 400 random training songs against
   their original GP files, a token-only filter ("no negative fret, no seventh string")
   wrongly keeps capo'd songs at **2.9%** (10 of 340), though it catches every drop-tuned
   one it was given.

## Decision

**Split.** Use the shipped split exactly, frozen by the SHA-256 of both split files and
checked on every load (`data/dadagp.py`). Cost weights are **fitted on training**;
temperature and any other selection happen on **validation**; **GuitarSet is evaluated once,
afterwards, and never consulted to choose anything.** The loader refuses `Split.TEST`.

**The artist overlap is accepted for now, and only for now.** Four or five scalar weights
cannot memorise an artist's style, so overlap between training and validation artists
cannot inflate a validation score for them. A model with real capacity can, so **any learned
fingering model must move to an artist-disjoint split first** — that is a requirement on the
Phase 2 plan, not an option.

**Song filter, from the original GP files, not the tokens.** `scripts/dadagp_track_meta.py`
reads every song's GuitarPro file once and records its guitar tunings and capos. A song is
used only if **every** clean or distorted guitar track is a **standard-tuned 6-string with no
capo**. A uniform downtune is allowed: it moves every string together, changes no fingering
decision, and leaves the candidate sets identical, so pitches are taken in standard tuning.
PyGuitarPro is needed only for that one-off pass and is not a project dependency.

Over all 26,181 songs the pass found:

| verdict | songs |
|---|---|
| clean — used | **22,034** (84.2%) |
| drop or other tuning | 2,311 |
| no guitar track | 1,099 |
| capo | 532 |
| 7-string | 173 |
| listed but absent from the archive | 32 |

That leaves **19,995 training** and **2,039 validation** songs. The 32 absent files are
folders whose names differ only by case, collapsed into one on a case-insensitive
filesystem when the archive was built; they are skipped.

**Notes.** Only the clean and distorted guitar instruments are read. A tied note, a dead
note and a harmonic are not fretted onsets and are skipped; a grace note is an ornament on
its main note; repeats are taken in written order. A group with a negative fret, a
seventh-string note or two notes on one string is dropped and counted — after the filter
these should be vanishingly rare, and the counts say whether they are.

## Alternatives considered

- **A fresh random split.** Rejected: the shipped split is the one anyone else using
  DadaGP will use, and there is no evidence-free reason to prefer a different one.
- **An artist-disjoint split now.** Rejected for the scalar fits, where it buys nothing and
  costs data. Required before Phase 2, above.
- **The token-only filter.** Rejected: it measurably leaks capo'd songs, and the exact pass
  costs about eight minutes once.
- **Keep drop-tuned songs by modelling their tuning** (ADR 0008 supports other tunings).
  Rejected for now: the dropped string's physical fret is not in the tokens, so it would
  mean extracting notes from the GP files directly. Worth it only if 8.8% more data turns
  out to matter.
- **Ship the split's song list in the repository.** Rejected: the hash freezes it without
  redistributing any part of a research-use dataset.

## Consequences

**Easier.** Every DadaGP number has one definition of which songs it covers, and changing
that definition means changing a hash in code, which a reviewer will see.

**Harder.** The corpus is 84% of DadaGP rather than all of it, and it is skewed the way
DadaGP is: heavily rock and metal, mostly distorted guitar, against GuitarSet's acoustic
jazz, bossa nova, funk, rock and singer-songwriter. Weights fitted here are measured on
GuitarSet as a **transfer** result, and should be read as one.

**Revisit** before Phase 2 trains anything with capacity (artist-disjoint split), and if
ProgGP arrives (its tunings are mostly drop and 7-string, which this filter excludes).

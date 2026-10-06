# ADR 0066: Full-song mixes are built: a labelled take over BabySlakh backing without guitars

Status: accepted (2026-10-07)

Plan: `docs/plans/2026-10-07-phase-6-full-songs.md`, Task 1. Backing chosen by Ege: BabySlakh now,
MoisesDB later.

## Context

Phase 6 needs full songs whose guitar notes carry string labels. A search on 2026-10-07 found none
public: every dataset with string labels is guitar alone. So mixes are built from labelled takes and
a multitrack corpus. BabySlakh (Zenodo 4603870, CC BY 4.0): Slakh2100's first 20 songs, synthesized
from MIDI at 16 kHz, 161–348 s each, every one with 1–5 guitar stems and 3–14 others.

## Decision

1. **Backing** is a song's rendered stems except the "Guitar" and "Ethnic" classes (the latter holds
   banjo, sitar and other plucked strings), summed to mono (`data/babyslakh.py`). The labelled take is
   then the only guitar, so every guitar note in the mix has a label.
2. **Split**: songs 1–10 back the validation takes (EGSet12, IDMT's licks, GuitarSet's player 00),
   songs 11–20 the test takes (EGDB, GuitarSet's players 01–05). No song backs both.
3. **Pairing** by a hash of the take's name (`data/mixes.py`); the backing starts 30 s into its song,
   is resampled to 44.1 kHz, looped or cut to the take's length.
4. **Two levels**, guitar against backing by RMS over the take: **0 dB** and **−6 dB**, both reported.
   One gain scales the whole mix so it never clips; the guitar is not shifted in time, so a take's
   labels — with its delay taken off as before (ADRs 0052, 0062) — are the mix's labels.
5. Mixes are written at 44.1 kHz mono under `cache/mixes/<split>/` (`scripts/build_mixes.py`); the
   test split is built only at its pre-registered look.

## What the mixes cannot show

- **No shared key or tempo**: the guitar and its backing come from different music, so notes clash
  more and line up less than in a real song.
- **Synthesized, 16 kHz backing**: no content above 8 kHz and a cleaner sound than recordings;
  separation may look easier than on real songs. MoisesDB, if Ege registers, is the check on this.
- **One guitar**: real songs often have several; separating guitars from each other is not attempted.

## Consequences

Full-song figures from these mixes are reported apart from isolated ones (ADR 0002) and always with
these limits beside them.

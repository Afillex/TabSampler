# ADR 0052: Guitar-TECHS's label delay is measured per take

Status: accepted (2026-10-06) — supersedes ADR 0051's label-delay bullet

## Context

ADR 0051 corrected Guitar-TECHS's late labels with one delay per player (23, 16, 15 ms), measured
before player 2's chords had downloaded. Those chords, checked with `scripts/check_guitartechs.py
--hop 64` on 2026-10-06, split into two groups: 14 of the 28 takes lag like the rest of player 2,
and 14 — with player 2's pinch harmonics — read 55–76 ms the other way, their labels **early**.
Their pitches are 93–100% confirmed once each take's own lag is applied, so the labels are good
and only their timing differs, take by take. Player 1's takes spread too, from 6 to 52 ms. A
65 ms error would move a note's window, which starts 35 ms before its onset, off its attack.

## Decision

- **Each take's delay is measured from its own audio** when its windows are cut:
  `MEASURE_LAG − onset_lag(signal, onsets)` (`audio/windows.py`), where `onset_lag` finds the
  shift, within ±0.3 s at 2.9 ms frames, at which the onset strength lines up best with the
  take's labelled onsets, and `MEASURE_LAG`, 8.7 ms, is what it reads on GuitarSet's player 00,
  whose onsets are trusted. Training logs every take's delay.
- **Labels are never shifted**; only where windows are cut, as before.
- The rest of ADR 0051 stands.

## Alternatives considered

- **Per player, or per session.** Player 2's takes do not say which session they come from,
  and per take costs nothing: every take has from tens to hundreds of onsets to align.

## Consequences

**Easier.** A take recorded with another offset is aligned without anyone noticing it.

**Harder.** The correction now depends on the onset measure; a take with few or soft onsets
could be misaligned. Every take's delay is printed, so an outlier shows.

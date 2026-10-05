# ADR 0053: EGDB's string is the MIDI channel, not the track name

Status: accepted (2026-10-06) — supersedes ADR 0050's "Labels" bullet

## Context

ADR 0050 read EGDB's labels as one MIDI track per string, named `1` (high e) to `6` (low E),
from the first clip alone. Phase 4's look at EGDB (logged, 489be2d) stopped before computing
anything: clip 26 names its tracks `6i`. A format check over all 240 label files — track names,
channels, and whether each note's string can sound it; no score — found that **853 of the 1,167
tracks holding notes have no name at all**; 1,166 use a single MIDI channel; where a track is
named `1`–`6`, its channel is the name less one in 300 of 304 cases.

## Decision

- **A note's string is its MIDI channel**: 0 the high e to 5 the low E (`data/egdb.py`,
  read with `mido` under the file's tempo map). Track names are ignored.
- Read so, 85 of EGDB's 35,704 notes (0.24%) lie below their string's open pitch; they are
  dropped and counted, as Guitar-TECHS's unplayable notes are. The four tracks whose name and
  channel disagree follow the channel.
- The rest of ADR 0050 stands. The look that stopped is logged; the rerun is a second logged line
  with its reason.

## Consequences

**Easier.** Every clip reads, whatever its tracks are called.

**Harder.** Nothing in the files states the channel convention; it rests on the 300 named tracks
that agree with it and on 99.76% of notes being playable on the string it gives.

# ADR 0051: Guitar-TECHS — split by player, labels cleaned before training

Status: accepted (2026-10-06)

Carries out Phase 4's Tasks 2 and 3 (`docs/plans/2026-10-06-phase-4-electric.md`): how
Guitar-TECHS's takes become training and validation examples for the string classifier (ADR 0046)
on clean electric guitar (ADR 0049).

## Context

Guitar-TECHS's labels come from a multi-track MIDI pickup that tracks pitch on each string.
`scripts/check_guitartechs.py` compared them with the direct-input audio on 2026-10-06, with
GuitarSet's player 00 as the control (0.949 of its pitches confirmed by the audio; its onset
measure reads +8.7 ms at 2.9 ms frames):

- **The labels arrive late**: the median take reads 23 ms (player 1), 16 ms (player 2) and
  15 ms (player 3) behind the control.
- **Overlaps are an artefact**: notes that start while their string still sounds overlap by
  3–4 ms and change pitch — the note-off arriving after the next note-on.
- **Glitches are short**: on player 1's scales, 250 of the 354 notes under 60 ms were not
  confirmed by the audio, against 12 of 3,272 longer ones; on player 3's music, 248 of 313
  against 226 of 1,628.
- **Bends and harmonics** sound a pitch other than the fretted note's.

## Decision

- **Players 1 and 2 train; player 3 is validation**, used to stop training, to calibrate and to
  choose — player-disjoint, as ADR 0037 keeps GuitarSet's players apart. Player 3 recorded only
  the musical excerpts (12 takes, 8.5 minutes), the closest material to what a user plays.
- **The input is the direct-input signal**; the amp microphone and the room microphones are kept
  for later.
- **Cleaning, the same for every player** (`data/guitartechs.py`): notes the pickup reports as
  shorter than 60 ms are dropped, judged on the pickup's own durations; then each note ends no
  later than the next note on its string. Takes of bends, harmonics and pinch harmonics are left
  out of training. Every drop is counted where it is made.
- **The label delay is corrected where windows are cut** (`LABEL_DELAY`, per player), never in
  the labels, as SynthTab's rendering latency is (ADR 0046).

## Alternatives considered

- **A validation take from each player.** Takes from players 1 and 2 share their players'
  hands with the training side; player 3 does not.
- **Keeping glitches, or a learned label filter.** The duration floor removes most of the
  unconfirmed notes for 3–4% of the confirmed ones, and is simple to state.

## Consequences

**Easier.** Every number on player 3 is player-disjoint and on audio of the target sound.

**Harder.** Player 3's labels are noisier than GuitarSet's: after cleaning, 0.861 of its
pitches are confirmed by the audio, against 0.996 on player 1's scales. A wrong label counts
against every decoder alike, so comparisons hold, but absolute figures on player 3 are lower
than its playing deserves. Validation is 12 short takes, so intervals are wide.

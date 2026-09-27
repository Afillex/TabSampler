# ADR 0006: Round GuitarSet's float MIDI reference pitches to the nearest integer

Status: accepted (2026-09-27)

Not a spec decision — a gap found while reading mirdata's loader. It affects every E2 and
E4 number, so it is recorded rather than buried in a comment.

## Context

mirdata builds GuitarSet note annotations as
`NoteData(np.array(intervals), "s", np.array(values), "midi")`, and **`values` are MIDI
floats, not integers** — GuitarSet annotates continuous pitch, so a note may be annotated
as 45.48 rather than 45. Meanwhile `NoteEvent.pitch` in spec 2.1 is `int`, and a
(string, fret) pair can only produce an integer pitch.

So a conversion is needed, and the choice is observable in the metrics:

- **E2** requires exact pitch equality between reference and prediction, so the reference
  must be integers.
- **E4** checks `open_pitch[string] + capo + fret == pitch`, which is integer arithmetic.
- **E1** goes through `mir_eval`, which compares pitches in Hz with a 50-cent tolerance and
  so tolerates the fractional part on its own.

## Decision

- For **E2 and E4** (the reference *tab*): round to the nearest integer,
  `int(np.rint(value))`. A note annotated 45.48 becomes MIDI 45.
- For **E1** (the reference *notes*): keep the float and convert to Hz with
  `librosa.midi_to_hz`, letting `mir_eval`'s 50-cent tolerance absorb the fraction.

`reference_tab` additionally asserts that each derived fret lies in `[0, n_frets]` and
raises `ReferenceInconsistencyError`, naming the track, if it does not. A note we cannot
place on the fretboard is a data or convention problem to look at, never something to drop
silently — dropping it would quietly inflate precision.

## Alternatives considered

- **Truncate (`int(value)`).** Rejected: it biases every pitch downward, and 45.98 would
  become 45 rather than 46. Objectively wrong; noted here because `int()` is the default
  thing to reach for.
- **Round for E1 too.** Rejected: it throws away information `mir_eval` can use, and would
  make a heavily-bent note count as a pitch error rather than a near miss.
- **Keep floats throughout and compare with a tolerance in E2.** Rejected: it would make E2
  disagree with E4 about what pitch a fret produces, and a fret cannot sound 45.48 anyway.

## Consequences

**Easier.** One conversion rule, applied at one boundary (`data/guitarset.py`), tested
directly (`test_float_midi_is_rounded_not_truncated`). E4 can be exactly 1.0 and mean it.

**Harder.** Notes bent more than a quarter tone are rounded to a neighbouring semitone in
the reference tab, which slightly misstates heavily-bent passages. Acceptable for v1, since
bends are explicitly out of scope until Phase 7; revisit then, when `NoteEvent.bend`
becomes real.

**Revisit at:** Phase 7.

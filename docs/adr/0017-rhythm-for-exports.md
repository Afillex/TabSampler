# ADR 0017: Exports carry a fixed grid and say that rhythm is not transcribed

Status: accepted (2026-09-27)

Decides the question ADR 0013 left open and **unblocks plan Task B1**. Does **not**
supersede ADR 0009: v1 still transcribes no rhythm, and this ADR is about what the
exporters do with a file format that refuses to be written without durations.

## Context

ADR 0013 put MusicXML and Guitar Pro export on the app track. Both formats are built on
note *values*: MusicXML has `<duration>` against a `<divisions>` unit, Guitar Pro has
beats inside measures. Neither can be written at all without them. ADR 0009 says v1 has
none — notes are spaced proportionally to time and rhythm is left implicit.

So the exporters cannot simply inherit the renderer's model, and the tempting default is
the dangerous one. An exporter that gives every note an eighth note produces a file that
opens in MuseScore looking like a transcription, with bar lines, a time signature and a
rhythm nobody measured. It is the plan's Review Focus 1 and it is worse than no export:
ASCII output is visibly approximate, whereas engraved notation asserts precision.

## Decision

**Export on a fixed high-resolution grid, declare the tempo as a constant, and put the
disclaimer inside the file.** Concretely:

1. **A declared tempo, not an estimated one.** 120 BPM, written as a constant. It is a
   unit of time, not a measurement: one quarter note is 0.5 s by definition. No tempo is
   estimated and none is claimed.
2. **A high-resolution grid.** MusicXML `divisions` of 480 per quarter, so a note lands
   within about 1 ms of its true onset at 120 BPM and the encoding is lossless at the
   precision our onsets have. Durations come from the note's own onset and offset, not
   from a beat grid snapped to.
3. **The disclaimer is in the file, not only in our docs.** Every export carries, in a
   field the host application displays:
   > Rhythm is NOT transcribed. Note positions encode measured onset times on a fixed
   > 120 BPM grid; the note values are an artefact of the file format. Bar lines are
   > arbitrary. Produced by Tab Sampler.
   In MusicXML this is a `<credit>` plus a `<words>` direction at the start of the part;
   in Guitar Pro, the score notice field and a text marker on the first measure. A test
   asserts its presence in the bytes of the written file — not in a docstring.
4. **String, fret and capo survive the round trip.** String and fret go in
   `<notations><technical><string>` and `<fret>`. A capo is exported *as a capo*, never
   folded into the fret numbers, or the tab would be unplayable by anyone reading it.
5. **The exporters never invent a time signature that implies phrasing.** 4/4 at 120 BPM,
   stated as the arbitrary container it is.

## Alternatives considered

- **(a) Refuse to export until beat tracking exists.** Honest, and rejected anyway. Spec
  gates beat tracking after M3, and the refusal buys nothing: a guitarist who wants the
  tab in Guitar Pro to fix the rhythm by hand is well served by a file with exact onsets
  and a warning, and badly served by no file at all. It also leaves ADR 0013's promise
  unmet for the whole app track.
- **(c) Quantise now with a named beat tracker** (`beat_this`, `madmom`). Rejected for
  exactly ADR 0009's reason: it puts an unmeasured estimator between the decoder and the
  output, so a wrong-looking export could be the decoder's fault or the beat tracker's and
  we would not know which. It also pulls Phase-6-era work forward into the app track, and
  it would need its own evaluation before any number it produced could be quoted.
- **Export MIDI instead**, which needs no note values. Rejected as an answer to *this*
  question: MIDI carries no string or fret, which is the entire output of this project.
  Worth adding later as a separate format; it does not decide the rhythm question.
- **Eighth notes everywhere.** Named only to record that it was rejected. It is the
  failure mode this ADR exists to prevent.

## Consequences

**Easier.** Task B1 is unblocked and its acceptance tests are decidable: the disclaimer is
in the file, the capo is a capo, string and fret survive, and the file opens in MuseScore
and TuxGuitar.

**Harder.** The export will look strange to a musician — a stream of oddly-tied notes
inside arbitrary bars. That is the correct appearance for a file with no rhythm in it, and
the disclaimer explains it. Anyone wanting engraved rhythm must wait for beat tracking.

**What we accept.** A reader who ignores the disclaimer may still mistake the note values
for transcribed rhythm. We accept that risk because the alternative — withholding the
export entirely — is a bigger cost to the user, and because the disclaimer is inside the
file rather than in documentation they will never see.

**Revisit** when beat tracking lands after M3. At that point the fixed grid becomes an
option (`--no-quantise`) rather than the only behaviour, and the disclaimer changes from a
statement of fact to a statement about which mode produced the file.

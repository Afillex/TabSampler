# ADR 0010: Decoder states are chords, not single notes

Status: accepted (2026-09-27)

Decides spec D6.

## Context

Chords are the whole difficulty of guitar fingering. Two constraints only exist at the
chord level:

- **One note per string.** A string sounds one pitch at a time. With note-level states
  there is no place to express this, so a decoder would happily assign two simultaneous
  notes to the same string and produce an unplayable tab that still scores well on
  pitch.
- **Span.** How far the hand must stretch is a property of the whole shape, not of any
  single note in it.

## Decision

A state is a `ChordState`: an assignment of every note in a `NoteGroup` to a **distinct**
string, with the fret determined by pitch and string. Notes whose onsets fall inside one
window (default 30 ms) form a group.

Hard constraints prune states during enumeration, not after:

- one note per string;
- the span of *fretted* notes is at most `max_span` (open strings need no finger and are
  excluded from the span).

`ChordState` rejects duplicate strings at construction, so an invalid state cannot exist.

**Known v1 limitation, from spec 2.2:** a note that is still ringing does not reserve its
string against the next group. A fast passage can therefore be assigned a fingering that
requires re-using a string still sounding. Documented rather than fixed; revisit if the
metrics show it matters.

## Alternatives considered

- **Note-level states.** Simpler and cheaper, and the transition structure would be a
  plain chain. Rejected because it cannot express one-note-per-string within a chord,
  which is the constraint that makes the output playable.
- **Chord states without pruning.** Rejected: the state count explodes. Six notes with
  six candidate positions each is 6! orderings before span filtering, and spec 7 names
  this as a risk. Pruning inside the search keeps it bounded, and
  `state_count_stats` instruments it so the claim is measured rather than hoped.

## Consequences

**Easier.** Playability is structural: an unplayable chord shape is not in the search
space at all, rather than being penalised and possibly winning anyway.

**Harder.** The state space is larger, so the decoder is slower and needs the pruning to
be correct. Enumeration is the most complexity-sensitive code in Phase 1, and a group
with no legal state becomes a case every consumer must handle
(`UnfingerableGroupError`) rather than an empty list to ignore.

**Also decided here (contract addition):** `NoteGroup`, `ChordState` and `Context` are
defined in `types.py`. Spec 2.1's Protocols reference all three without defining them.
See ADR 0007.

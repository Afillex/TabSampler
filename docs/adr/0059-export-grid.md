# ADR 0059: Exports use a 1/128-note grid and one voice (supersedes ADR 0017's point 2)

Status: accepted (2026-10-06)

Supersedes **point 2 only** of ADR 0017 (the grid's resolution). Everything else in ADR 0017 stands:
120 BPM declared, not estimated; 4/4 as an arbitrary container; the disclaimer in the file's
bytes; string, fret and capo exported as such.

## Context

ADR 0017 asked for MusicXML `divisions` of 480 per quarter so that "a note lands within about
1 ms of its true onset", with durations taken from each note's onset and offset. Writing the
exporters (plan `docs/plans/2026-10-06-app-track.md`, Task 4) showed that this cannot be had
together with a file that hosts display properly:

- A MusicXML `<duration>` can be any number of divisions, but MuseScore and TuxGuitar draw a note
  from its note value (`<type>`, dots, tuplets). A 37-division note has no note value; hosts
  either reject it or invent tuplets — notation that asserts a rhythm, the failure ADR 0017
  exists to prevent.
- Guitar Pro has only standard note values (whole to 1/128, dots, a fixed set of tuplets); there
  is no free duration at all.
- Both formats as we write them have one voice per bar. A note still ringing when the next one
  starts cannot keep its full length without a second voice.

## Decision

1. **One grid for both formats: the 1/128 note**, 15.625 ms at the declared 120 BPM (MusicXML
   `divisions` 32 per quarter). Onsets and ends round to it, so a note moves by at most 7.8 ms —
   well inside the ±50 ms within which E2 already counts an onset as right.
2. **Durations are sums of plain note values, tied** (largest first, split at bar lines). No
   dots, no tuplets: those would read as phrasing.
3. **One voice.** Notes within the decoder's grouping window (30 ms, anchored at the first, as
   `group_notes` does) form one chord at the first note's onset. A chord lasts until its longest
   note ends or the next chord starts, whichever is first; a gap becomes a rest. **A note held
   under the next one is cut where the next one starts** — the exports lose sustain; the JSON
   keeps it.
4. **The disclaimer says so.** The text of ADR 0017 point 3 stays, with one sentence added:
   "Positions are rounded to a 1/128 note (about 16 ms), and a note held under the next is cut
   where the next starts."

## Alternatives considered

- **Exact divisions without note values** (ADR 0017 as written). Unreadable or misread in the
  hosts the export exists for.
- **A coarser grid (1/32, 62.5 ms)**, fewer ties and tidier notation. It moves onsets by up to
  31 ms, more than half of E2's tolerance, and merges fast notes; the tidiness would be bought
  with accuracy.
- **Two voices**, keeping sustain. More code in both writers for a property most tab readers do
  not rely on; revisit if a user asks for it.

## Consequences

The exports look busy — many tied short notes — which ADR 0017 already accepted as the right
appearance for a file without transcribed rhythm. Onsets lose up to 7.8 ms and sustains are cut;
both are stated inside the file. Revisit with ADR 0017's own trigger, beat tracking.

# ADR 0049: The primary guitar sound — clean electric first

Status: accepted (2026-10-06)

Decides D2 (spec §7), due before Phase 4 because it drives which data Phase 4 trains on.

## Context

Phase 3's audio evidence, trained on SynthTab's rendered and mostly electric guitar, helped on
SynthTab and cost the decoder on GuitarSet's acoustic recordings (ADR 0048); rendered against real
and electric against acoustic were confounded. Phase 4's corpora, by the spec, are GOAT, GAPS and
Guitar-TECHS. Of these only GOAT (electric DI, Guitar Pro tabs) and Guitar-TECHS (electric, one
MIDI track per string from a multi-track pickup) label the string a note is played on; GAPS is
classical guitar aligned to scores, with no strings. No real steel-string acoustic recording with
string labels exists among them apart from GuitarSet, which is the test set (ADR 0037). The spec's
cue was acoustic and clean electric, measured separately.

## Decision

- **Clean electric guitar is Phase 4's primary target** (Ege, 2026-10-06). Distorted guitar
  stays later, as the spec's cue has it.
- **Acoustic stays measured, separately**: GuitarSet's players remain the test set, and Phase 4
  reports the gap from electric to acoustic, as its "done when" asks; EGDB becomes the second,
  electric, test set.
- **Guitar-TECHS first** (Ege): open, CC BY 4.0, real electric audio from three players with
  string labels. GOAT's access is unclear (by request, per its repository) and is Ege's call.

## Alternatives considered

- **Acoustic first.** Matches GuitarSet and the cue, but nothing in Phase 4's corpora could train
  the string evidence on real acoustic audio without GuitarSet itself.
- **All sounds at once.** Leaves Phase 3's confound in place and spreads three players' data thin.

## Consequences

**Easier.** Phase 4 can train on real, labelled audio of the sound it targets, and separates the
halves of Phase 3's confound: rendered against real (electric to electric) from electric against
acoustic (electric to GuitarSet).

**Harder.** The headline test set is acoustic while the target is electric, so the README must
report both and not let one stand for the other. Guitar-TECHS's three players are few; its
validation split must keep players apart, as ADR 0037 does for GuitarSet.

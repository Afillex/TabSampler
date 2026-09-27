# ADR 0005: Evaluate on GuitarSet's `audio_mic` channel

Status: accepted (2026-09-27)

Not a spec decision — a gap. The spec names GuitarSet but not which of its audio channels
we evaluate on, and E1 is not reproducible until that is pinned.

## Context

mirdata's GuitarSet loader exposes four audio properties:

| Property | Content |
|---|---|
| `audio_mic` | mono recording from a reference microphone |
| `audio_mix` | mono downmix of the hexaphonic pickup |
| `audio_hex` | 6 channels, **one per string** |
| `audio_hex_cln` | 6 channels per string, after bleed removal |

These are different timbres and would give different note F1 values for the same system.

## Decision

Evaluate on **`audio_mic`**. It is the natural acoustic recording — the closest of the four
to what a user would actually feed the app — and it is the channel most commonly reported
for GuitarSet.

`audio_hex` and `audio_hex_cln` are **forbidden as transcriber input under evaluation**.
One channel per string *is* the string ground truth for E2; feeding them in would leak the
label into the input and produce a meaningless score. They may be used later only as
training signal for an audio-conditioned model (Phase 3), never as evaluation input.

The channel is set in config, not hard-coded, and recorded in each `results.csv` row so a
number can never be silently compared against one measured on a different channel.

## Alternatives considered

- **`audio_mix`.** Rejected as primary: a pickup downmix is a narrower, more processed
  timbre than a microphone recording, and less representative of real input. Worth running
  once as a robustness check, reported separately.
- **Report both.** Rejected for now: it doubles every table for a difference we have no
  hypothesis about yet. Revisit if the acoustic/electric gap at Phase 4 makes it interesting.

## Consequences

**Easier.** One channel, reproducible numbers, a documented reason. Skipping the hex
archives also saves several GB of download (`partial_download=["annotations", "audio_mic"]`).

**Harder.** Our numbers are not directly comparable to any paper that used `audio_mix`
without our rerunning it — which spec 3.4 already forbids anyway.

**Revisit at:** Phase 4, alongside the acoustic-versus-electric comparison.

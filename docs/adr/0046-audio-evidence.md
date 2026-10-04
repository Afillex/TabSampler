# ADR 0046: The audio evidence — string probabilities per note (D13)

Status: accepted (2026-10-04)

Decides spec D13 for Phase 3 (`docs/plans/2026-10-04-phase-3-audio.md`).

## Context

The decoder's cost has had an acoustic term from the start — `CostWeights.acoustic`, multiplying
−log P(string | audio), identically zero until now (spec §2.2). D13 asks what form the evidence
takes: (a) a per-note CNN embedding, as in TART; (b) a TabCNN-style head predicting per-frame
string probabilities; (c) cross-attention from the tab model to audio frames. The spec's cue is
(b) first: it plugs straight into the acoustic term and is easy to inspect.

The decoder scores notes, not frames: each note in a group gets a string, and its cost is a sum
over notes. Phase 2 also showed where fingering fails — whole passages placed in another position
— which is exactly where the strings' different timbres should be heard.

## Decision

- **The evidence is a probability per string, per note**: P(string | audio, pitch), over the
  strings that can sound the note's pitch, from a small CNN over a constant-Q window around the
  note's onset, given the pitch. It is (b)'s output — string probabilities for the acoustic term
  — computed where the decoder needs it, per note rather than per frame.
- **It enters the decoder only through the acoustic term**: a note placed on string `s` costs
  `acoustic × −log P(s | audio)`. With `acoustic` zero, the decoder is exactly Phase 2's.
- **The model hears what a user's recording gives**: `audio_mic` on GuitarSet, never `audio_hex`
  (ADR 0005), and the transcriber's notes end to end.
- **It is pretrained on SynthTab** (spec §6), starting with its development set; its weights stay
  unpublished, as ADR 0042's do, since SynthTab is rendered from DadaGP.

## Alternatives considered

- **Per-frame string and fret activations (TabCNN as published).** The decoder would then have to
  turn frames into notes and resolve overlaps; per-note probabilities skip that step and are what
  the cost sums.
- **(a), a per-note embedding.** It needs a learned model to read the embedding; a probability
  can be read directly and fed to today's decoder. It stays the spec's second step.
- **(c), cross-attention.** The heaviest, and it couples the audio model to a tab model Phase 2
  did not adopt.

## Consequences

**Easier.** The evidence is inspectable on its own — per-note string accuracy — before it touches
the decoder, and the ablation is one weight set to zero.

**Harder.** Probabilities must be calibrated well enough that their logarithm is a cost; a
confident wrong string would cost the decoder more than an unsure one.

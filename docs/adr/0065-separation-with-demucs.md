# ADR 0065: Full songs are separated with Demucs's `htdemucs_6s`, in a dependency group of its own

Status: accepted (2026-10-07)

Plan: `docs/plans/2026-10-07-phase-6-full-songs.md`, Task 2. Spec §6 names the model.

## Context

Phase 6 puts a source separator in front of the pipeline (ADR 0002): the guitar stem of a full
song goes to the transcriber instead of the mix. The spec names `htdemucs_6s`, the 6-source Hybrid
Transformer Demucs (drums, bass, other, vocals, guitar, piano). Checked 2026-10-07: demucs 4.1.0
resolves on Python 3.13 with the project's PyTorch 2.14.1 and no other change; in a throwaway
environment it separated 20 s of audio in 5 s on this machine's CPU.

From its repository (github.com/facebookresearch/demucs): released under the MIT licence; archived
by its owner on 2025-01-01 and "not maintained anymore", with the author's fork taking important
fixes only; its README notes the 6-source model's piano stem "is not working great". It does not
state a separate licence for the pretrained weights, which demucs downloads at first use; we never
redistribute them.

## Decision

- **demucs 4.1.0, pinned, in a `separate` dependency group**, as `model` holds PyTorch (ADR 0041):
  the CLI, the evaluation harness and the isolated-guitar app install without it.
- **`audio/separate.py`** runs `htdemucs_6s` on CPU and writes the guitar stem as a 44.1 kHz WAV,
  cached under `cache/stems/` by a hash of the audio's bytes and the model's name. Demucs is imported
  only when a separation runs; tests use a stand-in.
- The weights are fetched by demucs into its own cache, outside the repository.

## Alternatives considered

- **Running it as a subprocess in a separate environment**, as Basic Pitch is (ADR 0001): not
  needed — it installs beside the project's own packages.
- **Other separators** (open-unmix and similar MUSDB-trained four-stem models): they have no guitar
  stem; guitar would sit inside "other" with keys and the rest.

## Consequences

**The project now declares its platforms** (`[tool.uv] environments`): Apple-silicon Macs and Linux.
On Intel Macs demucs 4.1.0 pins numpy below 2 while the project needs 2.5 or later, and PyTorch
2.14.1 already shipped no Intel-Mac wheel (checked in `uv.lock`), so this states what was true. An archived dependency: if a future PyTorch breaks it, the pin and the author's fork are the ways
out. Its weights' terms are not stated explicitly; if the app ever ships them, that is a question
for D15 first.

# ADR 0056: Fine-tuning Basic Pitch on Guitar-TECHS

Status: accepted (2026-10-06)

Carries out Phase 5's Task 4 (`docs/plans/2026-10-06-phase-5-transcriber.md`): the spec's "fine-tune
Basic Pitch on guitar data". Ege chose TensorFlow beside Basic Pitch over a PyTorch conversion.

## Decision

- **A training environment of its own** (`scripts/setup_transcriber_training.sh`): Python 3.11,
  basic-pitch 0.4.0 with its TensorFlow extra, outside the repository — beside, not inside, the
  inference environment of ADR 0001, because installing TensorFlow there would re-resolve librosa
  and coremltools under the cached transcriptions, a change the cache key cannot see.
- **The fine-tuned model runs where the released one does**: converted to CoreML with the
  released model's input and output names and passed to the unchanged CLI with `--model-path`.
  Checked first (2026-10-06): the released weights, copied into a freshly built model, reproduce the
  shipped TensorFlow model to 1.2e-7, and converted to CoreML they match the shipped CoreML model as
  closely as it matches TensorFlow (about 1e-5).
- **Targets built as Basic Pitch builds them for notes-only data**: mirdata's
  `NoteData.to_sparse_index` on its 1/86 s grid and its frequency grids — notes and contours from
  each note's whole duration, onsets from its first frame — as its MAESTRO pipeline does. The grids
  are read from the installed package, not retyped.
- **Data**: Guitar-TECHS's usable takes (ADR 0051), cleaned and aligned (ADR 0055), direct input
  at 22,050 Hz; players 1–2 train, player 3 stops the training — player-disjoint, as for the
  string classifier.
- **Training**: from the released weights; Basic Pitch's loss — label smoothing 0.2, the onset loss
  class-weighted with its paper's 0.95 on positives; Adam at 1e-4, a tenth of its rate from scratch;
  batches of 16 random two-second windows, as it samples; no augmentation; seed 0; an epoch is a
  fixed number of batches, and training stops when player 3's loss has not fallen for three epochs.
- **Weights**: Basic Pitch is Apache 2.0 and Guitar-TECHS CC BY 4.0, so fine-tuned weights could be
  published; that is D15, Ege's. Unpublished until decided.

## Alternatives considered

- **Basic Pitch's own `train.py`.** It trains from scratch and reads TFRecords made by Apache Beam
  pipelines for its five datasets; fine-tuning needs the released weights loaded and our data fed,
  so a short script replaces it, reusing its model, constants and loss.
- **PyTorch through ONNX.** Two new dependencies and a training loop of our own (Ege's choice).

## Consequences

**Easier.** The fine-tuned transcriber is a model path in a config; the cache, the CLI and the
decoder are untouched.

**Harder.** Two Python 3.11 environments to keep, and a conversion step that coremltools has not
been tested against for TensorFlow 2.15 — which is why the round trip is checked before trusting it.

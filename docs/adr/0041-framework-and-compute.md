# ADR 0041: The learned model's framework and compute (D11)

Status: accepted (2026-10-04) — Ege chose to start task C4 the same day

Decides spec D11 for Phase 2 task C4 (`docs/plans/2026-10-01-phase-2.md`): a learned sequence
model, tried because every hand-designed C3 candidate has now been judged on GuitarSet's
validation player (ADRs 0039, 0040). The spec's Phase 2 gate is itself a comparison of the
decoder against a learned model.

## Context

Measured on this project's machine on 2026-10-04: an Apple M4 with 16 GB of memory, macOS
27.0, Python 3.13.14. PyTorch 2.14.1 resolves for this Python and installs; its Metal
backend (MPS) is built and available. On a 2048 × 2048 matrix product repeated 20 times the
CPU took 0.21 s and MPS 0.42 s, warm-up included: for small models the CPU is not the slow
choice. There is no other compute, and none is wanted.

CI runs on Linux and installs every dependency group (`uv sync --all-groups`). PyTorch's
default Linux wheel brings several gigabytes of CUDA libraries a CPU-only CI cannot use.

## Decision

- **PyTorch 2.14.1**, in a dependency group of its own, `model`. Decoding and evaluating the
  cost model need nothing new; only the learned model's code imports `torch`.
- **Linux takes PyTorch's CPU-only build** from its own index, through `[tool.uv.sources]`
  with a `sys_platform == 'linux'` marker; macOS takes PyPI's, which includes MPS. CI keeps
  `uv sync --all-groups` and tests the model code on CPU.
- **Compute is the M4, CPU first.** MPS is used for a run only where a measurement on that
  model shows it faster, and the run says which device it used.
- **Training runs survive interruption.** Background jobs here are killed at 30 minutes and by
  the machine sleeping, so a training run checkpoints under `cache/` (gitignored), resumes
  from its last checkpoint, and runs under `caffeinate -i`.
- **Reproducibility:** every run fixes its seeds and records its device, its data split
  (DadaGP's artist split, ADR 0024; player 00 for selection, ADR 0037) and its commit.
- **The model code lives in `src/tabsampler/model/`**, under the same rules as the rest of
  `src/`: strict pyright, pure functions apart from the training loop's I/O at the edge, and
  offline tests small enough to run on CPU in seconds.

## Alternatives considered

- **JAX or MLX.** MLX is native to Apple silicon, but CI runs on Linux; JAX's Metal support is
  experimental. PyTorch runs on both, and the plan already named it.
- **Cloud GPUs.** Not needed for a model of this size, and a cost the project does not carry.
- **PyTorch in the core dependencies.** Every install — and the app — would carry it before
  anything shows a learned model is worth shipping.

## Consequences

**Easier.** A learned model can be built and tested without touching how the cost model is
installed or run.

**Harder.** Two platforms resolve PyTorch from different indexes, and `uv.lock` records both.

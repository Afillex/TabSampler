# C4 — A Learned Model on Top of the Decoder

> Tasks run in order; steps use checkbox (`- [ ]`) syntax. Every run that produces a
> reported number is pre-registered in a commit made before it runs.

**Status:** in progress (2026-10-04).

**Goal:** Phase 2 task C4 (`docs/plans/2026-10-01-phase-2.md`) and the spec's Phase 2 gate: build
ADR 0043's model — today's decoder plus a learned cost in context — train it on DadaGP's clean
parts, and produce the comparison table in oracle mode on player 00.

**Decisions:** ADR 0041 (PyTorch, `model` group, the M4's CPU first), ADR 0042 (weights stay
under `cache/`, unpublished), ADR 0043 (the design).

## Global constraints

- Players 01–05 are not read before task C6, and then once, pre-registered.
- Nothing is tuned on GuitarSet: training stops on DadaGP's clean validation parts; player 00
  judges the finished model once, by ADR 0039's rule.
- No checkpoint or weight file is committed (ADR 0042).
- **With the learned term off, the model is today's default** — tested, not assumed.

## Review focus

1. The torch log-likelihood with the learned term off equals `fingering/fit.py`'s, which the
   brute-force oracle already checks.
2. Torch Viterbi with the learned term off returns a path the oracle-verified decoder agrees with.
3. No path reads the test players; no weights land in git.

## Task 1: The lattice as arrays (`src/tabsampler/model/lattice.py`)

Per sequence: each level's node features Φ (as `fit.node_features`), node descriptors ψ, group
inputs, the movement and allowed-transition matrices, and the human path's node at each level.
Pure numpy.

- [x] Tests: Φ equals `fit.sequence_features`'s; the human node at each level is the human
  shape with the hand the human path carries; ψ and the group inputs have the documented
  shapes and ranges.
- [x] Implement; `make check`; commit.

## Task 2: The CRF in torch (`src/tabsampler/model/crf.py`)

Padded batches of lattices: path energy, log-partition by the forward algorithm in log space,
and Viterbi.

- [x] Tests: with the learned term off and the default's weights, the negative
  log-likelihood equals `fit.nll_and_gradient`'s, and Viterbi's cost equals the decoder's on
  random short inputs; padding changes nothing.
- [x] Implement; `make check`; commit. *(With `model/batch.py`; a length-masking mutation
  is caught.)*

## Task 3: The network (`src/tabsampler/model/net.py`)

The group encoder (two bidirectional GRU layers, 64 units), the node scorer `g`, and a switch
that turns `g` off.

- [x] Tests: output shapes; `g` off gives Task 2's energies; a fixed seed gives the same
  output twice.
- [x] Implement; `make check`; commit. *(A new model starts exactly at the default; a
  sequence scores the same alone and padded, and dropping the packing is caught.)*

## Task 4: Training (`scripts/train_model.py`)

DadaGP's clean artist-split parts in chunks of 128 groups; Adam 1e-3, batch 32, seed 0; early
stopping on DadaGP clean validation NLL (patience 3); checkpoints under `cache/model/`,
resumable; the device and commit recorded.

- [x] Offline smoke test on synthetic sequences (one epoch, resume from a checkpoint), and
  a test that a resumed run ends bit-identical to an uninterrupted one.
- [x] Implement; `make check`; commit.

## Task 5: Evaluation on player 00 (`scripts/evaluate_model.py`)

Player 00 only, oracle mode: decodings (a), (b) and (c) of ADR 0043, E2 and E3, per-track
counts for `scripts/compare_validation.py`.

- [x] Offline test on a synthetic track; a test that it reads only `guitarset_validation_ids()`.
- [x] Implement; `make check`; commit. *(On player 00 the untrained model's (c) equals (a)
  exactly: E2 0.8273 both, chord shapes 6541/6607 both, every track identical.)*

## Task 6: The run — pre-registered

**Data.** The clean parts of 1,500 artist-split DadaGP training songs (seed 0), and of the 300
validation songs the cost-model fits use. Every training song would not fit in memory: arrays
take about 5,800 bytes per group, about 310 groups per song's clean parts (measured on 50
songs), so 1,500 songs is about 470,000 groups and 2.7 GB — 2.5 times the cost-model fits' 600.
Every other setting is ADR 0043's: chunks of 128, batch 32, Adam 1e-3, patience 3, at most 30
epochs, seed 0, the CPU, starting at `configs/decoder_clean.yaml`.

    caffeinate -i uv run python -u scripts/train_model.py data/dadagp/DadaGP-v1.1.zip \
        data/dadagp/track_meta.json --run cache/model/c4 --train-songs 1500
    uv run python scripts/evaluate_model.py --run cache/model/c4 --out cache/validation/c4

**Hypotheses.** (1) Training lowers the NLL on DadaGP's clean validation parts below the
default's. (2) On player 00, (c) places more notes on the player's string than (a): oracle E2
higher with the 95% track-level paired interval wholly above zero, and no clear chord-shape drop
— ADR 0039's rule.

**Prediction.** (1) holds by a wide margin. (2) is uncertain: DadaGP's clean parts have misled
choices judged on player 00 before (ADR 0040). (c) − (a) between −0.01 and +0.03 in oracle E2;
(b) below (a).

- [ ] Commit this; train; evaluate on player 00; apply the rule to (c) against (a); results
  rows; ADR 0043's result; devlog.

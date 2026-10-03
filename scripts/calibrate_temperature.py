"""Calibrate the decoder's temperature on DadaGP validation (stage 4, ADR 0023).

Pre-registered before the run:

- Hypothesis: with the cost weights held fixed, temperature scaling on DadaGP validation
  lowers the validation calibration error of the per-note posteriors. Prediction: the fitted
  temperature is close to 1, because maximum-likelihood weights already carry the right
  overall scale -- so most of any calibration gain comes from fitting the weights, not from
  the temperature.
- Single variable: the temperature. Weights, songs, lattices and span bounds are fixed.
- Objective: the negative log-probability the per-note posteriors give the human position
  (temperature scaling's standard proper scoring rule). The reported metric is calibration
  error computed exactly as E5 computes it -- the posterior of the decoded position, binned
  against whether that position was the human one.
- Split: DadaGP validation only. GuitarSet plays no part (ADR 0003).

    uv run python scripts/calibrate_temperature.py data/dadagp/DadaGP-v1.1.zip \\
        data/dadagp/track_meta.json --decoder-config configs/fitted_dadagp.yaml
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar  # pyright: ignore[reportUnknownVariableType]

from tabsampler.config import load_phase1_config
from tabsampler.data.dadagp import load_tracks
from tabsampler.data.splits import Split
from tabsampler.decode.forward_backward import (  # the decoder's own log-space passes
    _cost_arrays,  # pyright: ignore[reportPrivateUsage]
    _log_alpha_beta,  # pyright: ignore[reportPrivateUsage]
    _lse_scalar,  # pyright: ignore[reportPrivateUsage]
    decode,
)
from tabsampler.decode.viterbi import build_lattice, viterbi
from tabsampler.eval.calibration import expected_calibration_error
from tabsampler.fingering.costs import HandSetScorer
from tabsampler.fingering.fit import HumanSequence, human_sequences
from tabsampler.types import Context, CostWeights, Tuning

DADAGP_TUNING = Tuning(n_frets=24)
#: Floor on a probability before taking its log, so one confidently wrong note is a large
#: finite penalty rather than an infinite one.
EPSILON = 1e-12


def score(
    sequences: Sequence[HumanSequence], weights: CostWeights, ctx: Context, temperature: float
):
    """(mean per-note NLL of the human position, calibration error as E5 computes it)."""
    scorer = HandSetScorer(weights=weights)
    nll = 0.0
    confidences: list[float] = []
    correct: list[bool] = []
    for seq in sequences:
        tab = decode(seq.groups, scorer, ctx, temperature=temperature, spans=seq.spans)
        truth = [p for state in seq.states for p in state.positions]
        for note, human in zip(tab, truth, strict=True):
            if note.position == human:
                p = note.posterior
            else:
                p = dict(note.alternatives).get(human, 0.0)
            nll -= math.log(max(p, EPSILON))
            confidences.append(note.posterior)
            correct.append(note.position == human)
    return nll / len(confidences), expected_calibration_error(confidences, correct)


def prepare(sequences: Sequence[HumanSequence], weights: CostWeights, ctx: Context):
    """Everything the temperature cannot change, computed once: per sequence, the lattice's
    cost arrays and, per level and note, which nodes place that note at the human position
    and which at the decoded (Viterbi) one. The Viterbi path does not depend on T."""
    scorer = HandSetScorer(weights=weights)
    prepared = []
    for seq in sequences:
        lattice = build_lattice(seq.groups, ctx, seq.spans)
        emissions, transitions = _cost_arrays(seq.groups, lattice, scorer, ctx)
        path, _ = viterbi(seq.groups, scorer, ctx, seq.spans)
        masks = []
        for nodes, truth, chosen in zip(lattice, seq.states, path, strict=True):
            per_note = []
            for i, (human, decoded) in enumerate(
                zip(truth.positions, chosen.positions, strict=True)
            ):
                at_human = np.array([n.state.positions[i] == human for n in nodes])
                at_chosen = np.array([n.state.positions[i] == decoded for n in nodes])
                per_note.append((at_human, at_chosen, human == decoded))
            masks.append(per_note)
        prepared.append((emissions, transitions, masks))
    return prepared


def fast_score(prepared, temperature: float):
    """``score`` on prepared sequences: only forward-backward is redone for each T."""
    nll = 0.0
    confidences: list[float] = []
    correct: list[bool] = []
    for emissions, transitions, masks in prepared:
        log_alpha, log_beta = _log_alpha_beta(emissions, transitions, temperature)
        log_z = _lse_scalar(log_alpha[-1])
        for a, b, per_note in zip(log_alpha, log_beta, masks, strict=True):
            log_gamma = a + b - log_z
            gamma = np.exp(log_gamma - _lse_scalar(log_gamma))  # as decode() renormalises
            for at_human, at_chosen, hit in per_note:
                nll -= math.log(max(float(gamma[at_human].sum()), EPSILON))
                confidences.append(min(max(float(gamma[at_chosen].sum()), 0.0), 1.0))
                correct.append(hit)
    return nll / len(confidences), expected_calibration_error(confidences, correct)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("meta", type=Path)
    parser.add_argument("--decoder-config", type=Path, required=True)
    parser.add_argument("--val-songs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=("shipped", "artist"), default="shipped")
    args = parser.parse_args()

    weights = load_phase1_config(args.decoder_config).weights
    ctx = Context(tuning=DADAGP_TUNING, max_span=5)
    val = [
        s
        for track in load_tracks(
            args.archive,
            Split.VALIDATION,
            args.meta,
            tuning=DADAGP_TUNING,
            sample=args.val_songs,
            seed=args.seed,
            scheme=args.split,
        )
        for s in human_sequences(track.steps, ctx)
    ]

    prepared = prepare(val, weights, ctx)
    before_nll, before_ece = fast_score(prepared, 1.0)
    print(
        f"T = 1.0000: per-note NLL {before_nll:.4f}  calibration error {before_ece:.4f}", flush=True
    )

    result = minimize_scalar(  # pyright: ignore[reportUnknownVariableType]
        lambda log_t: fast_score(prepared, math.exp(log_t))[0],
        bounds=(math.log(0.1), math.log(10.0)),
        method="bounded",
        options={"xatol": 1e-3},
    )
    best = math.exp(float(result.x))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    after_nll, after_ece = fast_score(prepared, best)
    print(f"T = {best:.4f}: per-note NLL {after_nll:.4f}  calibration error {after_ece:.4f}")


if __name__ == "__main__":
    main()

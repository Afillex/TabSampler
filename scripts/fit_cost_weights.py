"""Fit the cost weights by maximum likelihood on DadaGP, and check them on validation.

Pre-registered before the run (ADR 0023):

- Hypothesis: maximum-likelihood weights fitted on DadaGP training beat the hand-set
  weights on DadaGP validation, on both per-group NLL and per-note recovery of the human
  fingering, decoded on identical lattices. Prediction on record since Phase 1.5: ``high``
  rises sharply relative to ``move``, so the hand-set 120:1 ratio falls by at least an order
  of magnitude.
- Single variable: the weights. Lattices, span bounds, songs and tuning are identical for
  both weight sets.
- Adoption rule, fixed now: the fitted weights replace the hand-set ones as the default if
  and only if they improve validation per-note recovery. GuitarSet plays no part in that
  decision; it is evaluated once, afterwards (ADR 0003).
- Stability: two disjoint training samples are fitted independently. If their weights
  disagree badly, the sample is too small and the fit is not reported as a result.

Fitting is at temperature 1 (ADR 0023). The guitar is modelled with 24 frets, because DadaGP
guitars often have them and a lattice that cannot reach fret 23 cannot express the human's
choice there; the cost model itself does not depend on the number of frets.

    uv run python scripts/fit_cost_weights.py data/dadagp/DadaGP-v1.1.zip \\
        data/dadagp/track_meta.json --train-songs 600 --val-songs 300 --split artist
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from tabsampler.config import load_phase1_config
from tabsampler.data.dadagp import load_tracks
from tabsampler.data.splits import Split
from tabsampler.eval.recovery import PartSequence, recover
from tabsampler.fingering.costs import HandSetScorer
from tabsampler.fingering.fit import (
    FEATURE_GROUPS,
    HumanSequence,
    fit_weights,
    human_sequences,
    nll_and_gradient,
    sequence_features,
    weights_to_vector,
)
from tabsampler.types import Context, Tuning

DADAGP_TUNING = Tuning(n_frets=24)


def sequences_for(
    archive: Path, meta: Path, split: Split, songs: int, seed: int, ctx: Context, scheme: str
):
    """(song, part, sequence) for each human sequence; part is "clean" or "distorted"."""
    out: list[tuple[str, str, HumanSequence]] = []
    for track in load_tracks(
        archive, split, meta, tuning=DADAGP_TUNING, sample=songs, seed=seed, scheme=scheme
    ):
        part = "clean" if track.instrument.startswith("clean") else "distorted"
        out += [(track.song, part, s) for s in human_sequences(track.steps, ctx)]
    return out


def half(song: str) -> int:
    """Assign a song to one of two disjoint training halves, independent of its name's order."""
    return hashlib.sha1(song.encode()).digest()[0] % 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("meta", type=Path)
    parser.add_argument("--train-songs", type=int, default=600)
    parser.add_argument("--val-songs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split",
        choices=("shipped", "artist"),
        required=True,
        help="artist for every new fit (ADR 0024); shipped only to reproduce older rows.",
    )
    parser.add_argument(
        "--skip-halves",
        action="store_true",
        help="Skip the two half-sample stability fits (they inform, no rule uses them).",
    )
    parser.add_argument(
        "--part",
        choices=("all", "clean", "distorted"),
        default="all",
        help="Fit on one style's training parts only (ADR 0032); validation is always "
        "scored on every part, reported by part.",
    )
    parser.add_argument(
        "--features",
        default="",
        help="Feature groups to fit beyond the four base weights, comma-separated: "
        "string, region (ADR 0034).",
    )
    parser.add_argument(
        "--weights-out",
        type=Path,
        default=None,
        help="Write the fitted weights, at full precision, as JSON.",
    )
    parser.add_argument(
        "--per-song-out",
        type=Path,
        default=None,
        help="Directory for per-song validation counts (hand-set.json, fitted.json); "
        "they name DadaGP songs, so keep them under cache/.",
    )
    args = parser.parse_args()

    hand_set = load_phase1_config("configs/phase1_baseline.yaml").weights
    ctx = Context(tuning=DADAGP_TUNING, max_span=5)

    started = time.perf_counter()
    feature_groups = ["base", *(g for g in args.features.split(",") if g)]
    unknown = sorted(set(feature_groups) - set(FEATURE_GROUPS))
    if unknown:
        parser.error(f"unknown feature groups {unknown}; known: {sorted(FEATURE_GROUPS)}")
    active = tuple(name for group in feature_groups for name in FEATURE_GROUPS[group])
    print(
        f"split scheme: {args.split}; training parts: {args.part}; "
        f"feature groups: {', '.join(feature_groups)}"
    )
    train = [
        (song, s)
        for song, part, s in sequences_for(
            args.archive, args.meta, Split.TRAIN, args.train_songs, args.seed, ctx, args.split
        )
        if args.part in ("all", part)
    ]
    val_parts = [
        PartSequence(song, part, s)
        for song, part, s in sequences_for(
            args.archive, args.meta, Split.VALIDATION, args.val_songs, args.seed, ctx, args.split
        )
    ]
    val = [item.sequence for item in val_parts]
    print(f"loaded in {time.perf_counter() - started:.0f}s")

    every = [s for _, s in train]
    halves = {h: [s for song, s in train if half(song) == h] for h in (0, 1)}
    for label, seqs in (
        ("all training", every),
        ("half A", halves[0]),
        ("half B", halves[1]),
        ("validation", val),
    ):
        widened = sum(sp > ctx.max_span for s in seqs for sp in s.spans)
        groups = sum(len(s.groups) for s in seqs)
        print(f"{label:13s}: {len(seqs)} sequences, {groups} groups, {widened} widened bounds")

    started = time.perf_counter()
    train_feats = [(half(song), sequence_features(s, ctx)) for song, s in train]
    feats = {
        "all": [f for _, f in train_feats],
        "A": [f for h, f in train_feats if h == 0],
        "B": [f for h, f in train_feats if h == 1],
        "val": [sequence_features(s, ctx) for s in val],
    }
    print(f"lattices built in {time.perf_counter() - started:.0f}s")

    fits = {}
    for name in ("all",) if args.skip_halves else ("A", "B", "all"):
        started = time.perf_counter()
        fits[name] = fit_weights(feats[name], hand_set, active=active)
        r = fits[name]
        print(
            f"fit {name:3s}: move {r.weights.move:8.4f}  span {r.weights.span:8.4f}  "
            f"high {r.weights.high:8.4f}  open_reward {r.weights.open_reward:8.4f}  "
            f"NLL/group {r.nll_per_group:.4f}  iters {r.iterations}  converged {r.converged}  "
            f"|grad| {r.gradient_norm:.2e}  ({time.perf_counter() - started:.0f}s)"
        )

    fitted = fits["all"].weights
    if len(feature_groups) > 1:
        print(
            f"    string_bias {tuple(round(b, 4) for b in fitted.string_bias)}  "
            f"low_region {fitted.low_region:.4f}  high_region {fitted.high_region:.4f}"
        )
    if args.weights_out is not None:
        args.weights_out.parent.mkdir(parents=True, exist_ok=True)
        values = {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(fitted).items()}
        args.weights_out.write_text(json.dumps(values, indent=2))
        print(f"    wrote {args.weights_out}")
    for label, weights in (("hand-set", hand_set), ("fitted", fitted)):
        nll, _ = nll_and_gradient(weights_to_vector(weights), feats["val"])
        n_groups = sum(f.n_groups for f in feats["val"])
        report = recover(val_parts, HandSetScorer(weights=weights), ctx)
        ratio = weights.move / (weights.high / 12.0) if weights.high else float("inf")
        n_notes = report.counts()[1]
        print(
            f"validation {label:8s}: NLL/group {nll / n_groups:.4f}   recovery "
            f"{report.share():.4f} of {n_notes} notes "
            f"({report.single_candidate / n_notes:.4f} single-candidate)   "
            f"move:high per fret {ratio:.1f}:1"
        )
        print(
            f"    by part: clean {report.share('clean'):.4f} ({report.counts('clean')[1]} notes)"
            f"   distorted {report.share('distorted'):.4f} "
            f"({report.counts('distorted')[1]} notes)   "
            f"decoded E3 chord shapes {report.chord_shape_rate():.4f}"
        )
        if args.per_song_out is not None:
            args.per_song_out.mkdir(parents=True, exist_ok=True)
            out = args.per_song_out / f"{label}.json"
            out.write_text(json.dumps({"decoder": label, **report.to_dict()}))
            print(f"    wrote {out}")
    if args.skip_halves:
        return
    a, b = (weights_to_vector(fits[h].weights) for h in ("A", "B"))
    worst = float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-9)))
    print(f"half A vs half B, largest relative difference: {worst:.3f}")


if __name__ == "__main__":
    main()

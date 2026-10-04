"""Train ADR 0043's learned model on DadaGP's clean parts (Phase 2 task C4).

    caffeinate -i uv run python -u scripts/train_model.py data/dadagp/DadaGP-v1.1.zip \\
        data/dadagp/track_meta.json --run cache/model/c4

The model starts at the default decoder (``configs/decoder_clean.yaml``) and trains by the
fitter's objective -- the negative log-likelihood of the human path through the lattice -- on
the clean parts of DadaGP's artist-split training songs (ADR 0024), in chunks of 128 groups.
Training stops when the NLL on DadaGP's clean validation parts has not fallen for three epochs;
the best epoch is kept. Nothing here reads GuitarSet.

A run checkpoints after every epoch under ``--run`` and resumes from there when started again
(ADR 0041). Its weights stay under ``cache/``, unpublished (ADR 0042).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from tabsampler.config import load_phase1_config
from tabsampler.data.dadagp import load_tracks
from tabsampler.data.splits import Split
from tabsampler.fingering.fit import HumanSequence, human_sequences
from tabsampler.model.batch import make_batch
from tabsampler.model.crf import log_partition, path_energy
from tabsampler.model.lattice import LatticeArrays, lattice_arrays
from tabsampler.model.net import LearnedModel
from tabsampler.types import Context, Tuning

DADAGP_TUNING = Tuning(n_frets=24)
CTX = Context(tuning=DADAGP_TUNING, max_span=5)


def clean_sequences(
    archive: Path, meta: Path, split: Split, songs: int | None, seed: int
) -> list[HumanSequence]:
    """The human sequences of the clean parts of ``songs`` artist-split songs (all if None), as
    ``scripts/fit_cost_weights.py`` reads them."""
    out: list[HumanSequence] = []
    for track in load_tracks(
        archive, split, meta, tuning=DADAGP_TUNING, sample=songs, seed=seed, scheme="artist"
    ):
        if track.instrument.startswith("clean"):
            out += human_sequences(track.steps, CTX)
    return out


def chunked(sequence: HumanSequence, size: int) -> list[HumanSequence]:
    """``sequence`` in pieces of at most ``size`` groups, each its own lattice."""
    return [
        HumanSequence(
            sequence.groups[i : i + size],
            sequence.states[i : i + size],
            sequence.spans[i : i + size],
        )
        for i in range(0, len(sequence.groups), size)
    ]


def arrays_for(sequences: Sequence[HumanSequence], size: int) -> list[LatticeArrays]:
    return [
        lattice_arrays(piece.groups, CTX, piece.spans, piece.states)
        for sequence in sequences
        for piece in chunked(sequence, size)
    ]


def nll(model: LearnedModel, items: Sequence[LatticeArrays]) -> tuple[torch.Tensor, int]:
    """Summed NLL of ``items`` and their number of groups."""
    batch = make_batch(items)
    energies, transitions = model.energies(batch)
    total = path_energy(energies, transitions, batch.lengths, batch.human) + log_partition(
        energies, transitions, batch.lengths
    )
    return total.sum(), int(batch.lengths.sum())


def mean_nll(model: LearnedModel, items: Sequence[LatticeArrays], batch_size: int) -> float:
    model.eval()
    total, groups = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            value, count = nll(model, items[start : start + batch_size])
            total, groups = total + float(value), groups + count
    return total / groups


def commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
        )
    except OSError:
        return "unknown"
    return out.stdout.strip() or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("meta", type=Path)
    parser.add_argument("--run", type=Path, required=True, help="Where checkpoints go.")
    parser.add_argument("--train-songs", type=int, default=None, help="Default: every song.")
    parser.add_argument("--val-songs", type=int, default=300)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--start-config", type=Path, default=Path("configs/decoder_clean.yaml"))
    args = parser.parse_args()
    if args.device != "cpu":
        parser.error("only cpu is wired up; ADR 0041 adds mps where a measurement favours it")

    started = time.perf_counter()
    train = arrays_for(
        clean_sequences(args.archive, args.meta, Split.TRAIN, args.train_songs, args.seed),
        args.chunk,
    )
    val = arrays_for(
        clean_sequences(args.archive, args.meta, Split.VALIDATION, args.val_songs, args.seed),
        args.chunk,
    )
    print(
        f"{len(train)} training chunks ({sum(len(a.states) for a in train)} groups), "
        f"{len(val)} validation chunks ({sum(len(a.states) for a in val)} groups), "
        f"built in {time.perf_counter() - started:.0f}s",
        flush=True,
    )

    torch.manual_seed(args.seed)
    model = LearnedModel(load_phase1_config(args.start_config).weights)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    args.run.mkdir(parents=True, exist_ok=True)
    checkpoint, history_path = args.run / "checkpoint.pt", args.run / "history.jsonl"
    epoch, best, bad = 0, float("inf"), 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, weights_only=True)  # tensors and numbers only
        model.load_state_dict(saved["model"])
        optimiser.load_state_dict(saved["optimiser"])
        epoch, best, bad = saved["epoch"], saved["best"], saved["bad"]
        print(f"resumed after epoch {epoch} (best validation NLL/group {best:.4f})", flush=True)
    else:
        baseline = mean_nll(model, val, args.batch_size)
        print(f"before training: validation NLL/group {baseline:.4f} (the default)", flush=True)

    while epoch < args.max_epochs and bad < args.patience:
        epoch += 1
        clock = time.perf_counter()
        model.train()
        # Each epoch's order depends only on the seed and the epoch, so a resumed run trains
        # exactly as an uninterrupted one would have.
        order = np.random.default_rng([args.seed, epoch]).permutation(len(train))
        total, groups = 0.0, 0
        for start in range(0, len(order), args.batch_size):
            items = [train[i] for i in order[start : start + args.batch_size]]
            value, count = nll(model, items)
            optimiser.zero_grad()
            (value / count).backward()
            optimiser.step()
            total, groups = total + float(value.detach()), groups + count
        validation = mean_nll(model, val, args.batch_size)
        improved = validation < best
        best, bad = (validation, 0) if improved else (best, bad + 1)
        if improved:
            torch.save(model.state_dict(), args.run / "best.pt")
        entry = {
            "epoch": epoch,
            "train_nll_per_group": total / groups,
            "validation_nll_per_group": validation,
            "best": improved,
            "seconds": round(time.perf_counter() - clock, 1),
            "device": args.device,
            "commit": commit(),
        }
        with history_path.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")
        torch.save(
            {
                "model": model.state_dict(),
                "optimiser": optimiser.state_dict(),
                "epoch": epoch,
                "best": best,
                "bad": bad,
            },
            checkpoint,
        )
        print(
            f"epoch {epoch}: train NLL/group {entry['train_nll_per_group']:.4f}  validation "
            f"{validation:.4f}{'  best' if improved else ''}  ({entry['seconds']}s)",
            flush=True,
        )
    reason = "patience" if bad >= args.patience else "max epochs"
    print(f"stopped after epoch {epoch} ({reason}); best validation NLL/group {best:.4f}")


if __name__ == "__main__":
    main()

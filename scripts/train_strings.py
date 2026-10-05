"""Train the acoustic model's string classifier on SynthTab or Guitar-TECHS (ADRs 0046, 0051).

    caffeinate -i uv run python -u scripts/train_strings.py data/synthtab/SynthTab_Dev \\
        --run cache/acoustic/dev
    caffeinate -i uv run python -u scripts/train_strings.py data/guitar-techs \\
        --corpus guitartechs --run cache/acoustic/gt

Every note that more than one string can sound gives one example: its pitch-centred window, cut
with SynthTab's rendering latency corrected (``RENDER_LATENCY``), its pitch, the strings that can
sound it in its track's tuning (24 frets), and the string it was written on. Tracks are split by
a hash of their name: about 15% to stop on, the rest to train. ``--corpus guitartechs`` reads
Guitar-TECHS's direct-input takes instead, as ADR 0051 has them: players 1 and 2 train and
player 3 is held out; bent and harmonic takes are left out, the pickup's glitches dropped, and
each take's label delay, measured from its own audio (ADR 0052), corrected where windows are cut.
``--init`` starts from another run's weights. Adam 1e-3, batch 256, stopping when the held-out
NLL has not fallen for three epochs, at most 30; seed 0. A run checkpoints after
every epoch and resumes when started again; weights trained on SynthTab, or started from them,
stay unpublished (ADR 0046).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from numpy.typing import NDArray

from tabsampler.audio.windows import (
    MEASURE_LAG,
    RATE,
    note_window,
    onset_lag,
    possible_strings,
    track_cqt,
)
from tabsampler.data.guitartechs import STANDARD, clean, load_takes, usable
from tabsampler.data.synthtab import RENDER_LATENCY, is_held_out, load_tracks
from tabsampler.model.strings import StringClassifier

MAX_FRET = 24

#: Guitar-TECHS's player held out to stop on (ADR 0051); the others train.
HELD_OUT_PLAYER = 3

Examples = dict[str, NDArray[np.generic]]


def examples_for(root: Path, corpus: str = "synthtab") -> tuple[Examples, Examples]:
    """(training, held-out) examples: windows (float16), pitches, possible strings, strings."""
    if corpus == "guitartechs":
        return guitartechs_examples(root)
    tracks, skipped = load_tracks(root)
    print(f"{len(tracks)} tracks ({len(skipped)} skipped)", flush=True)
    sides: dict[bool, dict[str, list[NDArray[np.generic]]]] = {
        side: {"windows": [], "pitches": [], "possible": [], "strings": []}
        for side in (False, True)
    }
    for track in tracks:
        signal, _ = sf.read(track.audio, dtype="float32")
        cqt = track_cqt(signal)
        side = sides[is_held_out(track.name)]
        for note, position in track.notes:
            possible = possible_strings(note.pitch, track.open_pitches, MAX_FRET)
            if sum(possible) < 2:
                continue
            onset = note.onset + RENDER_LATENCY[track.family]
            side["windows"].append(note_window(cqt, onset, note.pitch).astype(np.float16))
            side["pitches"].append(np.int16(note.pitch))
            side["possible"].append(np.array(possible))
            side["strings"].append(np.int8(position.string))
    return tuple(  # type: ignore[return-value]
        {key: np.stack(values) for key, values in sides[side].items()} for side in (False, True)
    )


def guitartechs_examples(root: Path) -> tuple[Examples, Examples]:
    """Guitar-TECHS's examples, as :func:`examples_for` gives SynthTab's (ADR 0051)."""
    takes, skipped = load_takes(root)
    takes = [take for take in takes if usable(take)]
    print(f"{len(takes)} usable takes ({len(skipped)} skipped)", flush=True)
    sides: dict[bool, dict[str, list[NDArray[np.generic]]]] = {
        side: {"windows": [], "pitches": [], "possible": [], "strings": []}
        for side in (False, True)
    }
    glitches = 0
    for take in takes:
        signal, _ = librosa.load(take.direct_input, sr=RATE, mono=True)
        signal = np.asarray(signal, dtype=np.float32)
        cqt = track_cqt(signal)
        notes, dropped = clean(take.notes)
        glitches += dropped
        if not notes:
            continue
        delay = MEASURE_LAG - onset_lag(signal, sorted({n.onset for n, _ in notes}))
        print(f"  P{take.player} {take.name}: labels {1000 * delay:+.1f} ms late", flush=True)
        side = sides[take.player == HELD_OUT_PLAYER]
        for note, position in notes:
            possible = possible_strings(note.pitch, STANDARD, MAX_FRET)
            if sum(possible) < 2:
                continue
            onset = note.onset - delay
            side["windows"].append(note_window(cqt, onset, note.pitch).astype(np.float16))
            side["pitches"].append(np.int16(note.pitch))
            side["possible"].append(np.array(possible))
            side["strings"].append(np.int8(position.string))
    print(f"{glitches} pickup glitches dropped", flush=True)
    return tuple(  # type: ignore[return-value]
        {key: np.stack(values) for key, values in sides[side].items()} for side in (False, True)
    )


def batches(examples: Examples, size: int, order: NDArray[np.int64]):  # type: ignore[no-untyped-def]
    for start in range(0, len(order), size):
        pick = order[start : start + size]
        yield (
            torch.from_numpy(examples["windows"][pick].astype(np.float32)),
            torch.from_numpy(examples["pitches"][pick].astype(np.int64)),
            torch.from_numpy(examples["possible"][pick]),
            torch.from_numpy(examples["strings"][pick].astype(np.int64)),
        )


def evaluate(model: StringClassifier, examples: Examples, size: int) -> tuple[float, float, float]:
    """(mean NLL, accuracy, chance) over ``examples``."""
    model.eval()
    nll = right = 0.0
    with torch.no_grad():
        for windows, pitches, possible, strings in batches(
            examples, size, np.arange(len(examples["strings"]))
        ):
            log_probs = model(windows, pitches, possible)
            nll -= float(log_probs.gather(1, strings.unsqueeze(1)).sum())
            right += float((log_probs.argmax(dim=1) == strings).sum())
    count = len(examples["strings"])
    chance = float(np.mean(1.0 / examples["possible"].sum(axis=1)))
    return nll / count, right / count, chance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--corpus", choices=("synthtab", "guitartechs"), default="synthtab")
    parser.add_argument("--init", type=Path, help="A run's best.pt to start from.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.run.mkdir(parents=True, exist_ok=True)
    cache = args.run / "examples.npz"
    if cache.exists():
        saved = np.load(cache)
        train = {k[6:]: saved[k] for k in saved.files if k.startswith("train_")}
        held = {k[5:]: saved[k] for k in saved.files if k.startswith("held_")}
    else:
        started = time.perf_counter()
        train, held = examples_for(args.root, args.corpus)
        np.savez(
            cache,
            **{f"train_{k}": v for k, v in train.items()},
            **{f"held_{k}": v for k, v in held.items()},
        )
        print(f"examples built in {time.perf_counter() - started:.0f}s", flush=True)
    print(f"{len(train['strings'])} training notes, {len(held['strings'])} held out", flush=True)

    torch.manual_seed(args.seed)
    model = StringClassifier()
    if args.init is not None:
        model.load_state_dict(torch.load(args.init, weights_only=True))
    optimiser = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    checkpoint, history = args.run / "checkpoint.pt", args.run / "history.jsonl"
    epoch, best, bad = 0, float("inf"), 0
    if checkpoint.exists():
        saved_state = torch.load(checkpoint, weights_only=True)
        model.load_state_dict(saved_state["model"])
        optimiser.load_state_dict(saved_state["optimiser"])
        epoch, best, bad = saved_state["epoch"], saved_state["best"], saved_state["bad"]
        print(f"resumed after epoch {epoch}", flush=True)
    while epoch < args.max_epochs and bad < args.patience:
        epoch += 1
        clock = time.perf_counter()
        model.train()
        order = np.random.default_rng([args.seed, epoch]).permutation(len(train["strings"]))
        for windows, pitches, possible, strings in batches(train, args.batch_size, order):
            loss = -model(windows, pitches, possible).gather(1, strings.unsqueeze(1)).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        nll, accuracy, chance = evaluate(model, held, args.batch_size)
        improved = nll < best
        best, bad = (nll, 0) if improved else (best, bad + 1)
        if improved:
            torch.save(model.state_dict(), args.run / "best.pt")
        entry = {
            "epoch": epoch,
            "held_out_nll": nll,
            "held_out_accuracy": accuracy,
            "chance": chance,
            "best": improved,
            "seconds": round(time.perf_counter() - clock, 1),
        }
        with history.open("a") as handle:
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
            f"epoch {epoch}: held-out NLL {nll:.4f}  accuracy {accuracy:.4f} (chance {chance:.4f})"
            f"{'  best' if improved else ''}  ({entry['seconds']}s)",
            flush=True,
        )
    print(f"stopped after epoch {epoch}; best held-out NLL {best:.4f}")


if __name__ == "__main__":
    main()

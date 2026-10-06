"""Fine-tune Basic Pitch on Guitar-TECHS (Phase 5, Task 4; ADR 0056).

Runs in the training environment (``scripts/setup_transcriber_training.sh``), not the project's:

    PY=~/.local/share/tabsampler/bp-train/bin/python
    $PY scripts/finetune_basic_pitch.py grids --out cache/transcriber/grids.npz
    $PY scripts/finetune_basic_pitch.py train --data cache/transcriber/data \\
        --run cache/transcriber/ft
    $PY scripts/finetune_basic_pitch.py export --run cache/transcriber/ft
    $PY scripts/finetune_basic_pitch.py export --released --run cache/transcriber/released

``grids`` writes Basic Pitch's own time and frequency grids for ``prepare_transcriber_data.py``.
``train`` starts from the released weights and trains on ``<data>/train`` -- batches of 16 random
two-second windows, cut as Basic Pitch cuts them -- with its loss (label smoothing 0.2, the onset
loss class-weighted at 0.95, or unweighted with ``--unweighted-onsets``) and Adam at 1e-4; after
every epoch of ``--steps`` batches it scores ``<data>/validation`` in fixed consecutive windows,
keeps the best weights and stops after three epochs without improvement. Epoch 0 is the released
model's score. ``export`` converts the best weights -- or, with ``--released``, the released ones,
the round-trip check -- to CoreML with the released model's input and output names, checks it
against TensorFlow on a probe, and writes ``<run>/model.mlpackage`` for the CLI's ``--model-path``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf
from basic_pitch import ICASSP_2022_MODEL_PATH, models
from basic_pitch.constants import (
    ANNOT_N_FRAMES,
    ANNOTATION_HOP,
    ANNOTATIONS_FPS,
    AUDIO_N_SAMPLES,
    AUDIO_SAMPLE_RATE,
    AUDIO_WINDOW_LENGTH,
    FREQ_BINS_CONTOURS,
    FREQ_BINS_NOTES,
)

BATCH = 16
LEARNING_RATE = 1e-4
LABEL_SMOOTHING = 0.2
ONSET_POSITIVE_WEIGHT = 0.95
PATIENCE = 3


def grids(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        annotation_hop=ANNOTATION_HOP,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        freq_bins_notes=FREQ_BINS_NOTES,
        freq_bins_contours=FREQ_BINS_CONTOURS,
    )
    print(
        f"grids: hop {ANNOTATION_HOP:.6f} s, {len(FREQ_BINS_NOTES)} note bins, "
        f"{len(FREQ_BINS_CONTOURS)} contour bins -> {out}"
    )


def released_model() -> tf.keras.Model:
    """A freshly built model holding the released weights (checked equal to the shipped one)."""
    saved = tf.saved_model.load(str(ICASSP_2022_MODEL_PATH))
    values = {v.name: v for v in saved.variables}
    model = models.model()
    for variable in model.variables:
        variable.assign(values[variable.name])
    return model


def load_side(folder: Path) -> list[dict[str, np.ndarray]]:
    takes = []
    for path in sorted(folder.glob("*.npz")):
        saved = np.load(path)
        takes.append({key: saved[key] for key in ("audio", "onsets", "notes", "contours")})
    return takes


def window(take: dict[str, np.ndarray], t_start: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """One two-second window, cut as Basic Pitch's ``extract_window`` cuts it."""
    begin = round(AUDIO_SAMPLE_RATE * t_start)
    audio = take["audio"][begin : begin + AUDIO_N_SAMPLES]
    audio = np.pad(audio, (0, AUDIO_N_SAMPLES - len(audio)))[:, None]
    frame = round(ANNOTATIONS_FPS * t_start)
    length = math.ceil(ANNOTATIONS_FPS * AUDIO_WINDOW_LENGTH)
    out = {}
    for key, name in (("onsets", "onset"), ("notes", "note"), ("contours", "contour")):
        part = take[key][frame : frame + length].astype(np.float32)
        out[name] = np.pad(part, ((0, ANNOT_N_FRAMES - len(part)), (0, 0)))
    return audio, out


def batches(takes: list[dict[str, np.ndarray]], rng: np.random.Generator):  # type: ignore[no-untyped-def]
    """Endless batches of random windows, each take drawn in proportion to its length."""
    lengths = np.array([len(t["audio"]) / AUDIO_SAMPLE_RATE for t in takes])
    spans = np.maximum(lengths - AUDIO_N_SAMPLES / AUDIO_SAMPLE_RATE, 0.0)
    weights = spans / spans.sum()
    while True:
        picks = rng.choice(len(takes), size=BATCH, p=weights)
        cut = [window(takes[i], float(rng.uniform(0.0, spans[i]))) for i in picks]
        yield (
            np.stack([a for a, _ in cut]),
            {name: np.stack([t[name] for _, t in cut]) for name in ("onset", "note", "contour")},
        )


def fixed_windows(takes: list[dict[str, np.ndarray]]):  # type: ignore[no-untyped-def]
    """Consecutive two-second windows over every validation take, in order."""
    for take in takes:
        seconds = len(take["audio"]) / AUDIO_SAMPLE_RATE
        for k in range(int(seconds // AUDIO_WINDOW_LENGTH)):
            yield window(take, float(k * AUDIO_WINDOW_LENGTH))


def train(
    data: Path, run: Path, steps: int, max_epochs: int, seed: int, weighted_onsets: bool = True
) -> None:
    run.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(seed)
    training, validation = load_side(data / "train"), load_side(data / "validation")
    held = list(fixed_windows(validation))
    val_x = np.stack([a for a, _ in held])
    val_y = {name: np.stack([t[name] for _, t in held]) for name in ("onset", "note", "contour")}
    print(
        f"{len(training)} training takes; {len(validation)} validation takes, {len(held)} windows",
        flush=True,
    )
    model = released_model()
    model.compile(
        loss=models.loss(
            label_smoothing=LABEL_SMOOTHING,
            weighted=weighted_onsets,
            positive_weight=ONSET_POSITIVE_WEIGHT,
        ),
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
    )
    stream = batches(training, rng)
    history = run / "history.jsonl"
    history.unlink(missing_ok=True)
    best, bad, epoch = math.inf, 0, 0
    while True:
        clock = time.perf_counter()
        if epoch > 0:
            for _ in range(steps):
                x, y = next(stream)
                model.train_on_batch(x, y)
        scores = model.evaluate(val_x, val_y, batch_size=BATCH, verbose=0, return_dict=True)
        loss = float(scores["loss"])
        improved = loss < best
        best, bad = (loss, 0) if improved else (best, bad + 1)
        if improved:
            model.save_weights(str(run / "best.weights.h5"))
        entry = {
            "epoch": epoch,
            "validation": {k: float(v) for k, v in scores.items()},
            "best": improved,
            "seconds": round(time.perf_counter() - clock, 1),
        }
        with history.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")
        print(
            f"epoch {epoch}: validation loss {loss:.5f}{'  best' if improved else ''}"
            f"  ({entry['seconds']}s)",
            flush=True,
        )
        epoch += 1
        if bad >= PATIENCE or epoch > max_epochs:
            break
    print(f"stopped after epoch {epoch - 1}; best validation loss {best:.5f}")


def export(run: Path, released: bool) -> None:
    import coremltools as ct

    model = released_model()
    if not released:
        model.load_weights(str(run / "best.weights.h5"))
    mlmodel = ct.convert(
        model,
        source="tensorflow",
        inputs=[ct.TensorType(name="input_1", shape=(1, AUDIO_N_SAMPLES, 1))],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
    )
    spec = mlmodel.get_spec()
    ct.utils.rename_feature(spec, "input_1", "input_2")
    mlmodel = ct.models.MLModel(spec, weights_dir=mlmodel.weights_dir)
    probe = np.random.default_rng(0).standard_normal((1, AUDIO_N_SAMPLES, 1)).astype(np.float32)
    expected = {k: v.numpy() for k, v in model(probe, training=False).items()}
    got = mlmodel.predict({"input_2": probe})
    names = {"contour": "Identity", "note": "Identity_1", "onset": "Identity_2"}
    worst = max(float(np.abs(got[names[k]] - v).max()) for k, v in expected.items())
    if worst > 1e-3:
        raise SystemExit(f"CoreML departs from TensorFlow by {worst:.2e}: not exported")
    run.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(run / "model.mlpackage"))
    print(f"exported {run / 'model.mlpackage'}; largest departure from TensorFlow {worst:.2e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    g = commands.add_parser("grids")
    g.add_argument("--out", type=Path, required=True)
    t = commands.add_parser("train")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--run", type=Path, required=True)
    t.add_argument("--steps", type=int, default=100, help="Batches per epoch.")
    t.add_argument("--max-epochs", type=int, default=50)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument(
        "--unweighted-onsets",
        action="store_true",
        help="The onset loss unweighted, as Basic Pitch's train.py defaults (Task 4c).",
    )
    e = commands.add_parser("export")
    e.add_argument("--run", type=Path, required=True)
    e.add_argument("--released", action="store_true", help="Export the released weights.")
    args = parser.parse_args()
    if args.command == "grids":
        grids(args.out)
    elif args.command == "train":
        train(
            args.data, args.run, args.steps, args.max_epochs, args.seed, not args.unweighted_onsets
        )
    else:
        export(args.run, args.released)


if __name__ == "__main__":
    main()

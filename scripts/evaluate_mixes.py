"""Full-song mixes end to end on validation: the mix, the separated guitar, and more (Phase 6).

    caffeinate -i uv run python -u scripts/evaluate_mixes.py --add-mix 0
    caffeinate -i uv run python -u scripts/evaluate_mixes.py --add-mix 0 0.1 0.25 0.5

Reads ``cache/mixes/validation/index.json`` (``scripts/build_mixes.py``). For every take, Basic
Pitch at ``CHOSEN_PARAMS`` hears in turn the isolated take (the ceiling), the isolated take
through the separator (what separation alone costs), and per level the mix itself and its
``htdemucs_6s`` guitar stem with each ``--add-mix`` share of the mix added back; the default
decoder strings the notes (``transcribe_path``). Scored by E2 against the take's labels (EGSet12
and IDMT with their delay taken off, ADR 0062; GuitarSet's own). Prints E2 per set, level and
input, and each input's change against the mix with a bootstrap interval over takes. Validation
only.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tabsampler.audio.separate import GuitarSeparator
from tabsampler.config import load_phase1_config
from tabsampler.data.electric import aligned, load_egset12, load_idmt_licks
from tabsampler.eval.metrics import exact_tab_f1, tab_notes_to_placed
from tabsampler.pipeline import transcribe_path
from tabsampler.transcribe.basic_pitch_cli import CHOSEN_PARAMS, BasicPitchCLITranscriber
from tabsampler.types import NoteEvent, Position

TOLERANCE = 0.05
Counts = tuple[int, int]  # 2 x matches, reference + estimated notes


def test_labels() -> dict[str, list[tuple[NoteEvent, Position]]]:
    """EGDB's labels as they are (ADR 0050) and GuitarSet's players 01-05."""
    from tabsampler.data.egdb import load_clips
    from tabsampler.data.guitarset import load_dataset, reference_tab
    from tabsampler.data.splits import guitarset_test_ids

    out = {clip.clip_id: list(clip.notes) for clip in load_clips(Path("data/egdb"))[0]}
    dataset: Any = load_dataset(Path("data/guitarset"))
    for track_id in guitarset_test_ids():
        out[track_id] = list(reference_tab(dataset.track(track_id)))
    return out


def labels() -> dict[str, list[tuple[NoteEvent, Position]]]:
    out: dict[str, list[tuple[NoteEvent, Position]]] = {}
    for take in [
        *load_egset12(Path("data/egset12")),
        *load_idmt_licks(Path("data/idmt-smt-guitar")),
    ]:
        out[take.name] = list(aligned(take).notes)
    from tabsampler.data.guitarset import load_dataset, reference_tab
    from tabsampler.data.splits import guitarset_validation_ids

    dataset: Any = load_dataset(Path("data/guitarset"))
    for track_id in guitarset_validation_ids():
        out[track_id] = list(reference_tab(dataset.track(track_id)))
    return out


def pooled(counts: dict[str, Counts]) -> float:
    """E2 over all the takes' notes together."""
    return sum(v[0] for v in counts.values()) / sum(v[1] for v in counts.values())


def interval(pairs: list[tuple[Counts, Counts]]) -> tuple[float, float, float]:
    """Change in pooled E2 (second minus first) and a 95% bootstrap interval over takes."""
    n = np.array([[a[0], a[1], b[0], b[1]] for a, b in pairs], dtype=float)

    def delta(m: np.ndarray) -> float:
        return float(m[:, 2].sum() / m[:, 3].sum() - m[:, 0].sum() / m[:, 1].sum())

    rng = np.random.default_rng(0)
    draws = [delta(n[rng.integers(0, len(n), len(n))]) for _ in range(5000)]
    low, high = np.percentile(draws, [2.5, 97.5])
    return delta(n), float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--add-mix", type=float, nargs="+", default=[0.0])
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--note", default="", help="The pre-registration, for the access log.")
    parser.add_argument(
        "--electric",
        action="store_true",
        help="Also EGDB's stems with configs/decoder_electric.yaml (reported, not judged).",
    )
    args = parser.parse_args()
    index = Path(f"cache/mixes/{args.split}/index.json")
    cfg = load_phase1_config("configs/decoder_clean.yaml")
    transcriber = BasicPitchCLITranscriber(params=CHOSEN_PARAMS)
    separator = GuitarSeparator()
    if args.split == "test":
        from tabsampler.data.splits import record_test_set_access

        record_test_set_access(
            "evaluate_mixes.py --split test: full-song mixes of EGDB and GuitarSet's players "
            "01-05 (ADR 0066), mix and htdemucs_6s stem end to end; a pre-registered look. "
            f"{args.note}".strip()
        )
    reference = test_labels() if args.split == "test" else labels()
    electric_cfg = load_phase1_config("configs/decoder_electric.yaml") if args.electric else None
    rows: list[dict[str, Any]] = json.loads(index.read_text())

    def score(path: Path, take: str) -> Counts:
        result = transcribe_path(path, cfg, transcriber, estimate=lambda _: None)
        prf = exact_tab_f1(reference[take], tab_notes_to_placed(result.tab), TOLERANCE)
        return 2 * prf.n_match, prf.n_ref + prf.n_est

    # (set, level, input) -> take -> counts
    table: dict[tuple[str, str, str], dict[str, Counts]] = defaultdict(dict)
    isolated: dict[str, tuple[Counts, Counts]] = {}
    for i, row in enumerate(rows, start=1):
        take, group, level = row["take"], row["set"], f"{row['level_db']:+g} dB"
        if take not in isolated:
            source = Path(row["source"])
            isolated[take] = (score(source, take), score(separator.stem(source), take))
        table[(group, "isolated", "isolated")][take] = isolated[take][0]
        table[(group, "isolated", "isolated, separated")][take] = isolated[take][1]
        mix = Path(row["mix"])
        table[(group, level, "mix")][take] = score(mix, take)
        for share in args.add_mix:
            name = "stem" if share == 0 else f"stem + {share:g} mix"
            table[(group, level, name)][take] = score(separator.stem(mix, add_mix=share), take)
        if electric_cfg is not None and group == "egdb":
            stem = separator.stem(mix)
            result = transcribe_path(stem, electric_cfg, transcriber, estimate=lambda _: None)
            prf = exact_tab_f1(reference[take], tab_notes_to_placed(result.tab), TOLERANCE)
            table[(group, level, "stem, electric option")][take] = (
                2 * prf.n_match,
                prf.n_ref + prf.n_est,
            )
        if i % 50 == 0:
            print(f"  {i} of {len(rows)} mixes", flush=True)

    groups = sorted({k[0] for k in table})
    levels = sorted({k[1] for k in table if k[1] != "isolated"}, reverse=True)
    for level in levels:
        for group in [*groups, "all"]:

            def pick(name: str, g: str = group, lv: str = level) -> dict[str, Counts]:
                merged: dict[str, Counts] = {}
                for gg in groups if g == "all" else [g]:
                    merged.update(table.get((gg, lv, name), {}))
                return merged

            mix_counts = pick("mix")
            line = [f"{level} {group:12s}"]
            for name in ["isolated", "isolated, separated"]:
                c = pick(name, lv="isolated")
                line.append(f"{name} {pooled(c):.4f}")
            line.append(f"mix {pooled(mix_counts):.4f}")
            for share in args.add_mix:
                name = "stem" if share == 0 else f"stem + {share:g} mix"
                c = pick(name)
                d, lo, hi = interval([(mix_counts[t], c[t]) for t in c])
                e2 = pooled(c)
                line.append(f"{name} {e2:.4f} ({d:+.4f} [{lo:+.4f}, {hi:+.4f}] vs mix)")
            electric = pick("stem, electric option")
            if electric:
                line.append(f"stem, electric option {pooled(electric):.4f}")
            print("  ".join(line), flush=True)


if __name__ == "__main__":
    main()

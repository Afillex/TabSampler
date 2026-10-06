"""The electric decoder end to end on the electric validation sets (electric plan, Task 3).

    caffeinate -i uv run python -u scripts/evaluate_electric.py

Basic Pitch at ``CHOSEN_PARAMS`` transcribes each take of EGSet12 and IDMT-SMT-Guitar's licks
(ADR 0062); ``transcribe_path`` decodes the notes with ``configs/decoder_clean.yaml`` and with
``configs/decoder_electric.yaml``, which adds what the string classifier hears at each transcribed
note. Scored by E2 against each take's labels, its label delay taken off (``aligned``). Prints
E2 per set for both, and the change with a bootstrap interval over takes. EGSet12 judges; IDMT
chose the evidence's weight and temperature, so its figure is optimistic. Validation only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tabsampler.config import load_phase1_config
from tabsampler.data.electric import aligned, load_egset12, load_idmt_licks
from tabsampler.eval.metrics import exact_tab_f1, tab_notes_to_placed
from tabsampler.pipeline import classifier_hearing, transcribe_path
from tabsampler.transcribe.basic_pitch_cli import CHOSEN_PARAMS, BasicPitchCLITranscriber

TOLERANCE = 0.05


def main() -> None:
    plain = load_phase1_config("configs/decoder_clean.yaml")
    electric = load_phase1_config("configs/decoder_electric.yaml")
    hear = classifier_hearing(electric)
    transcriber = BasicPitchCLITranscriber(params=CHOSEN_PARAMS)
    for label, takes in (
        ("EGSet12 (judges)", load_egset12(Path("data/egset12"))),
        ("IDMT licks (chose the calibration)", load_idmt_licks(Path("data/idmt-smt-guitar"))),
    ):
        rows: list[tuple[int, int, int, int]] = []  # 2*matches and ref+est, plain then electric
        for take in takes:
            reference = list(aligned(take).notes)
            row: list[int] = []
            for cfg in (plain, electric):
                result = transcribe_path(
                    take.audio, cfg, transcriber, estimate=lambda _: None, hear=hear
                )
                prf = exact_tab_f1(reference, tab_notes_to_placed(result.tab), TOLERANCE)
                row += [2 * prf.n_match, prf.n_ref + prf.n_est]
            rows.append((row[0], row[1], row[2], row[3]))
        n = np.array(rows, dtype=float)
        before, after = n[:, 0].sum() / n[:, 1].sum(), n[:, 2].sum() / n[:, 3].sum()
        rng = np.random.default_rng(0)
        deltas = []
        for _ in range(10000):
            p = n[rng.integers(0, len(n), len(n))]
            deltas.append(p[:, 2].sum() / p[:, 3].sum() - p[:, 0].sum() / p[:, 1].sum())
        low, high = np.percentile(deltas, [2.5, 97.5])
        print(
            f"{label}: {len(n)} takes, end-to-end E2 {before:.4f} -> {after:.4f}, "
            f"change {after - before:+.4f} [{low:+.4f}, {high:+.4f}]",
            flush=True,
        )


if __name__ == "__main__":
    main()

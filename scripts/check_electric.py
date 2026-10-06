"""Check EGSet12's and IDMT-SMT-Guitar's labels against their audio (electric audio-evidence plan).

Per take, the two measures ``scripts/check_guitartechs.py`` uses on Guitar-TECHS, with its control
(GuitarSet's player 00: +11.6 ms, 0.949 of pitches confirmed, 2026-10-06): the onset lag, within
+-0.3 s, at which the audio's onset strength is highest on average at the labelled onsets, and the
share of labelled pitches the audio confirms 30-150 ms after the onset, lag corrected. Also the
notes that start before the previous note on their string has ended. EGFxSet has no onsets in its
labels; its files are checked for one clear note: the first detected onset's time, per file.
A data check, not a metric: no results row.

    uv run python scripts/check_electric.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from tabsampler.audio.windows import HOP, RATE, track_cqt
from tabsampler.data.electric import egfxset_notes, load_egset12, load_idmt_licks

_spec = importlib.util.spec_from_file_location(
    "check_guitartechs", Path(__file__).with_name("check_guitartechs.py")
)
assert _spec is not None and _spec.loader is not None
gt: Any = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gt)


def main() -> None:
    frame_ms = HOP / RATE * 1000
    for label, takes in (
        ("EGSet12", load_egset12(Path("data/egset12"))),
        ("IDMT licks", load_idmt_licks(Path("data/idmt-smt-guitar"))),
    ):
        lags: list[float] = []
        hits = checked = overlapping = 0
        worst: list[tuple[float, str]] = []
        for take in takes:
            notes = [n for n, _ in take.notes]
            signal = gt.load_audio(take.audio)
            lag = gt.best_lag(signal, sorted({n.onset for n in notes}))
            h, c = gt.confirmed(track_cqt(signal), notes, lag)
            hits, checked = hits + h, checked + c
            overlapping += gt.overlaps(take.notes)
            lags.append(lag * frame_ms)
            worst.append((h / max(c, 1), take.name))
        q = np.percentile(lags, [0, 25, 50, 75, 100])
        print(
            f"{label}: {len(takes)} takes; best lag ms min/q1/median/q3/max "
            f"{' '.join(f'{x:+.0f}' for x in q)}; pitches confirmed {hits}/{checked} "
            f"({hits / checked:.3f}); notes overlapping their string's last {overlapping}"
        )
        print("  lowest-confirmed takes:", ", ".join(f"{n} {r:.2f}" for r, n in sorted(worst)[:5]))

    firsts: list[float] = []
    for wav, _, _ in egfxset_notes(Path("data/egfxset")):
        y, _ = librosa.load(wav, sr=RATE, mono=True)
        onsets = librosa.onset.onset_detect(y=y, sr=RATE, units="time")
        firsts.append(float(onsets[0]) if len(onsets) else float("nan"))
    f = np.array(firsts)
    print(
        f"EGFxSet clean: {len(f)} files; first onset s min/median/max "
        f"{np.nanmin(f):.3f}/{np.nanmedian(f):.3f}/{np.nanmax(f):.3f}; "
        f"none found {np.isnan(f).sum()}"
    )


if __name__ == "__main__":
    main()

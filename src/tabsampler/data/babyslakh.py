"""BabySlakh as backing for full-song mixes (Phase 6; Zenodo 4603870, CC BY 4.0).

The first 20 songs of Slakh2100, synthesized from MIDI, at 16 kHz: per song a ``metadata.yaml``
naming each stem's instrument class and a ``stems/`` folder of WAVs. The backing is every rendered
stem except the guitars and the "Ethnic" class (banjo, sitar and other plucked strings), so the
labelled take is the only guitar in the mix. Stems listed but never rendered have no file and are
not in the song's own mix either. Songs 1-10 back the validation takes, 11-20 the test takes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from numpy.typing import NDArray

#: Instrument classes left out of the backing.
LEFT_OUT = frozenset({"Guitar", "Ethnic"})

SPLITS = {"validation": range(1, 11), "test": range(11, 21)}


def songs(root: Path, split: str) -> list[Path]:
    """The songs backing ``split``'s takes, by their number."""
    numbers = SPLITS[split]
    return sorted(
        p for p in root.iterdir() if p.name.startswith("Track") and int(p.name[5:]) in numbers
    )


def backing(song: Path) -> tuple[NDArray[np.float32], int]:
    """The song without its guitars, summed to mono, and its sample rate.

    Raises:
        ValueError: if no stem is left once the guitars are taken out.
    """
    stems = yaml.safe_load((song / "metadata.yaml").read_text())["stems"]
    total: NDArray[np.float32] | None = None
    rate = 0
    for key, info in sorted(stems.items()):
        path = song / "stems" / f"{key}.wav"
        if info["inst_class"] in LEFT_OUT or not path.is_file():
            continue
        data, rate = sf.read(path, dtype="float32", always_2d=True)
        mono = np.asarray(data, dtype=np.float32).mean(axis=1)
        if total is None:
            total = mono
        else:
            n = max(len(total), len(mono))
            total = np.pad(total, (0, n - len(total))) + np.pad(mono, (0, n - len(mono)))
    if total is None:
        raise ValueError(f"{song.name} has nothing but guitars")
    return total, int(rate)

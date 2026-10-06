"""Electric guitar with string labels besides Guitar-TECHS (plan: electric audio evidence).

- **EGFxSet** (CC BY 4.0): every note of a Stratocaster, direct input, one five-second file each,
  named ``<string>-<fret>.wav`` under ``Clean/<pickup>/``. Its string 1 is the **high** e: measured
  from the open strings' audio (2026-10-06), string 1 sounds MIDI 64 and string 6 MIDI 40.
  Training data.
- **EGSet12** (CC BY 4.0): twelve solo pieces through an amplifier and a microphone, one JAMS file
  each with a ``note_midi`` annotation per string, ``data_source`` 0 the low E, as GuitarSet's.
  Validation data.
- **IDMT-SMT-Guitar** (CC BY-NC-ND 4.0, "for evaluation purpose"): ``dataset2``'s licks, one XML
  file each with ``stringNumber`` 1 the **low** E. Only takes of normal notes are read -- no
  harmonics, bends, slides, vibrato or dead notes -- and not its drills. Validation data only.

Every note's fret must sound its labelled pitch in standard tuning, or the file is refused.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from tabsampler.types import NoteEvent, Position

#: Standard tuning, low string first.
STANDARD = (40, 45, 50, 55, 59, 64)
MAX_FRET = 24


@dataclass(frozen=True, slots=True)
class LabelledTake:
    """A recording and every note's string and fret."""

    name: str
    audio: Path
    notes: tuple[tuple[NoteEvent, Position], ...]  # by onset, then string


def _placed(source: str, onset: float, offset: float, pitch: int, string: int, fret: int):
    if not 0 <= string < len(STANDARD) or not 0 <= fret <= MAX_FRET:
        raise ValueError(f"{source}: string {string} fret {fret} is not on the neck")
    if STANDARD[string] + fret != pitch:
        raise ValueError(f"{source}: string {string} fret {fret} does not sound pitch {pitch}")
    note = NoteEvent(onset=onset, offset=max(offset, onset + 1e-3), pitch=pitch, confidence=1.0)
    return note, Position(string=string, fret=fret)


def _sorted(notes: list[tuple[NoteEvent, Position]]) -> tuple[tuple[NoteEvent, Position], ...]:
    return tuple(sorted(notes, key=lambda placed: (placed[0].onset, placed[1].string)))


def load_idmt_licks(root: Path) -> list[LabelledTake]:
    """IDMT-SMT-Guitar's licks of normal notes, under ``root`` as its zip unpacks."""
    base = root / "IDMT-SMT-GUITAR_V2" / "dataset2"
    takes: list[LabelledTake] = []
    for xml in sorted((base / "annotation").glob("*.xml")):
        if "Lick" not in xml.stem:
            continue
        events = list(ET.parse(xml).getroot().iter("event"))
        if not events or any(e.findtext("expressionStyle") != "NO" for e in events):
            continue
        notes = [
            _placed(
                xml.stem,
                float(e.findtext("onsetSec") or "nan"),
                float(e.findtext("offsetSec") or "nan"),
                round(float(e.findtext("pitch") or "nan")),
                int(e.findtext("stringNumber") or "0") - 1,
                int(e.findtext("fretNumber") or "-1"),
            )
            for e in events
        ]
        takes.append(
            LabelledTake(f"idmt {xml.stem}", base / "audio" / f"{xml.stem}.wav", _sorted(notes))
        )
    return takes


def load_egset12(root: Path) -> list[LabelledTake]:
    """EGSet12's twelve pieces, as its files sit in ``root``."""
    takes: list[LabelledTake] = []
    for jams in sorted(root.glob("*.jams")):
        doc = json.loads(jams.read_text())
        notes: list[tuple[NoteEvent, Position]] = []
        for annotation in doc["annotations"]:
            if annotation["namespace"] != "note_midi":
                continue
            string = int(annotation["annotation_metadata"]["data_source"])
            for n in annotation["data"]:
                pitch = round(float(n["value"]))
                onset = float(n["time"])
                notes.append(
                    _placed(
                        jams.stem,
                        onset,
                        onset + float(n["duration"]),
                        pitch,
                        string,
                        pitch - STANDARD[string],
                    )
                )
        takes.append(LabelledTake(f"egset12 {jams.stem}", jams.with_suffix(".wav"), _sorted(notes)))
    return takes


_EGFX_NAME = re.compile(r"^([1-6])-(\d+)\.wav$")


def egfxset_notes(root: Path) -> Iterator[tuple[Path, int, Position]]:
    """Every clean EGFxSet file with its note's pitch and position; onsets come from the audio."""
    for wav in sorted((root / "Clean").glob("*/*.wav")):
        match = _EGFX_NAME.match(wav.name)
        if match is None:
            continue
        string = 6 - int(match.group(1))  # its string 1 is the high e
        fret = int(match.group(2))
        yield wav, STANDARD[string] + fret, Position(string=string, fret=fret)

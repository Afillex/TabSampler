"""MusicXML export (ADRs 0017, 0059): a tab staff on the shared 1/128-note grid.

Written with the standard library's ``xml.etree`` (ADR 0058). The disclaimer that rhythm is
not transcribed goes in a ``<credit>`` and a ``<words>`` direction, so it is in the file a host
opens, not only in our docs. String and fret go in ``<technical>``; MusicXML counts strings
from the highest, so our string 0 (low E) is its string 6. A capo is a ``<capo>``, and frets
stay relative to it as ours are.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Sequence

from tabsampler.render.grid import (
    BEATS_PER_BAR,
    DISCLAIMER,
    TEMPO_BPM,
    UNITS_PER_BAR,
    UNITS_PER_QUARTER,
    GridEvent,
    to_grid,
)
from tabsampler.types import TabNote, Tuning

#: Note value names by length in 1/128 notes.
TYPES = {
    128: "whole",
    64: "half",
    32: "quarter",
    16: "eighth",
    8: "16th",
    4: "32nd",
    2: "64th",
    1: "128th",
}

#: Sharps only: spelling is not something the transcription knows.
_SPELLING = (
    ("C", 0),
    ("C", 1),
    ("D", 0),
    ("D", 1),
    ("E", 0),
    ("F", 0),
    ("F", 1),
    ("G", 0),
    ("G", 1),
    ("A", 0),
    ("A", 1),
    ("B", 0),
)


def _sub(parent: ET.Element, tag: str, text: str | None = None, **attrs: str) -> ET.Element:
    node = ET.SubElement(parent, tag, attrs)
    if text is not None:
        node.text = text
    return node


def _spell(midi: int) -> tuple[str, int, int]:
    step, alter = _SPELLING[midi % 12]
    return step, alter, midi // 12 - 1


def _attributes(measure: ET.Element, tuning: Tuning) -> None:
    attrs = _sub(measure, "attributes")
    _sub(attrs, "divisions", str(UNITS_PER_QUARTER))
    _sub(_sub(attrs, "key"), "fifths", "0")
    time = _sub(attrs, "time")
    _sub(time, "beats", str(BEATS_PER_BAR))
    _sub(time, "beat-type", "4")
    clef = _sub(attrs, "clef")
    _sub(clef, "sign", "TAB")
    _sub(clef, "line", "5")
    details = _sub(attrs, "staff-details")
    _sub(details, "staff-lines", str(tuning.n_strings))
    for line, pitch in enumerate(tuning.open_pitches, start=1):  # line 1 is the lowest string
        step, alter, octave = _spell(pitch)
        staff_tuning = _sub(details, "staff-tuning", line=str(line))
        _sub(staff_tuning, "tuning-step", step)
        if alter:
            _sub(staff_tuning, "tuning-alter", str(alter))
        _sub(staff_tuning, "tuning-octave", str(octave))
    if tuning.capo:
        _sub(details, "capo", str(tuning.capo))


def _directions(measure: ET.Element) -> None:
    direction = _sub(measure, "direction", placement="above")
    kind = _sub(direction, "direction-type")
    metronome = _sub(kind, "metronome")
    _sub(metronome, "beat-unit", "quarter")
    _sub(metronome, "per-minute", str(TEMPO_BPM))
    _sub(_sub(direction, "direction-type"), "words", DISCLAIMER)
    _sub(direction, "sound", tempo=str(TEMPO_BPM))


def _notes(measure: ET.Element, event: GridEvent, tuning: Tuning) -> None:
    if not event.positions:
        note = _sub(measure, "note")
        _sub(note, "rest")
        _sub(note, "duration", str(event.length))
        _sub(note, "voice", "1")
        _sub(note, "type", TYPES[event.length])
        return
    for k, position in enumerate(event.positions):
        note = _sub(measure, "note")
        if k:
            _sub(note, "chord")
        step, alter, octave = _spell(tuning.pitch_at(position.string, position.fret))
        pitch = _sub(note, "pitch")
        _sub(pitch, "step", step)
        if alter:
            _sub(pitch, "alter", str(alter))
        _sub(pitch, "octave", str(octave))
        _sub(note, "duration", str(event.length))
        if event.tie_stop:
            _sub(note, "tie", type="stop")
        if event.tie_start:
            _sub(note, "tie", type="start")
        _sub(note, "voice", "1")
        _sub(note, "type", TYPES[event.length])
        notations = _sub(note, "notations")
        if event.tie_stop:
            _sub(notations, "tied", type="stop")
        if event.tie_start:
            _sub(notations, "tied", type="start")
        technical = _sub(notations, "technical")
        _sub(technical, "string", str(tuning.n_strings - position.string))
        _sub(technical, "fret", str(position.fret))


def render_musicxml(tab: Sequence[TabNote], tuning: Tuning, window_s: float = 0.03) -> bytes:
    """The tab as a MusicXML 4.0 partwise score, UTF-8."""
    root = ET.Element("score-partwise", version="4.0")
    work = _sub(root, "work")
    _sub(work, "work-title", "Tab Sampler transcription")
    credit = _sub(root, "credit", page="1")
    _sub(credit, "credit-words", DISCLAIMER)
    part_list = _sub(root, "part-list")
    score_part = _sub(part_list, "score-part", id="P1")
    _sub(score_part, "part-name", "Guitar")
    part = _sub(root, "part", id="P1")

    events = to_grid(tab, window_s)
    n_bars = (events[-1].start + events[-1].length) // UNITS_PER_BAR
    measures = [_sub(part, "measure", number=str(i + 1)) for i in range(n_bars)]
    _attributes(measures[0], tuning)
    _directions(measures[0])
    for event in events:
        _notes(measures[event.start // UNITS_PER_BAR], event, tuning)

    ET.indent(root)
    body = ET.tostring(root, encoding="unicode")
    doctype = (
        '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
        '"http://www.musicxml.org/dtds/partwise.dtd">'
    )
    return f'<?xml version="1.0" encoding="UTF-8"?>\n{doctype}\n{body}\n'.encode()

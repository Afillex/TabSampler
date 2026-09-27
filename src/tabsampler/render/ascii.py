"""ASCII tab rendering (ADR 0013).

Six lines, high e at the top and low E at the bottom, the way tab is conventionally
written -- which is upside down relative to ``Position.string``, where 0 is the low E.

Spacing is **proportional to time** (ADR 0009): there are no bar lines, beats or note
values, because that would need beat tracking, which is deliberately out of scope for v1.
Horizontal distance is the only rhythmic information here.

Uncertain notes -- posterior below a threshold -- are wrapped in parentheses. Column
widths are computed from the widest token in each column, so a two-digit fret or an
``(11)`` marker cannot shift the columns after it out of alignment.
"""

from __future__ import annotations

from collections.abc import Sequence

from tabsampler.types import TabNote, Tuning

#: Note names for the standard-tuning strings, low to high, for the line labels.
STANDARD_LABELS = ("E", "A", "D", "G", "B", "e")

#: How much wall-clock time one filler character represents (ADR 0009).
DEFAULT_SECONDS_PER_CHAR = 0.08

#: Posteriors below this are marked uncertain.
DEFAULT_UNCERTAINTY_THRESHOLD = 0.6

#: Filler between columns, so a line reads as a string.
FILL = "-"


def string_labels(tuning: Tuning) -> tuple[str, ...]:
    """One label per string, low to high."""
    if tuning.n_strings == len(STANDARD_LABELS):
        return STANDARD_LABELS
    return tuple(str(index) for index in range(tuning.n_strings))


def render_ascii(
    tab: Sequence[TabNote],
    tuning: Tuning = Tuning.STANDARD,
    uncertainty_threshold: float = DEFAULT_UNCERTAINTY_THRESHOLD,
    seconds_per_char: float = DEFAULT_SECONDS_PER_CHAR,
    window_s: float = 0.03,
) -> str:
    """Render ``tab`` as ASCII guitar tablature.

    An empty tab renders as six empty strings rather than raising, so a silent clip
    produces something a reader can recognise.
    """
    if seconds_per_char <= 0:
        raise ValueError(f"seconds_per_char must be positive, got {seconds_per_char}")

    labels = string_labels(tuning)
    n_strings = tuning.n_strings
    lines: list[list[str]] = [[] for _ in range(n_strings)]

    columns = _columns(tab, window_s=window_s)
    previous_onset: float | None = None

    for onset, notes in columns:
        gap = (
            0
            if previous_onset is None
            else max(1, round((onset - previous_onset) / seconds_per_char))
        )
        for line in lines:
            line.append(FILL * gap)

        tokens: dict[int, str] = {}
        for note in notes:
            if not 0 <= note.position.string < n_strings:
                continue
            tokens[note.position.string] = _token(note, uncertainty_threshold)

        width = max((len(t) for t in tokens.values()), default=1)
        for string in range(n_strings):
            token = tokens.get(string, "")
            lines[string].append(token.ljust(width, FILL) if token else FILL * width)

        previous_onset = onset

    # Trailing filler so the tab does not end flush against the last note.
    for line in lines:
        line.append(FILL * 2)

    # Low E is string 0 but belongs at the bottom, so render the lines reversed.
    rendered = [
        f"{labels[string]}|{''.join(lines[string])}|" for string in reversed(range(n_strings))
    ]
    return "\n".join(rendered)


def _token(note: TabNote, threshold: float) -> str:
    fret = str(note.position.fret)
    return f"({fret})" if note.posterior < threshold else fret


def _columns(tab: Sequence[TabNote], window_s: float) -> list[tuple[float, list[TabNote]]]:
    """Group notes into one column per sounding moment, in time order."""
    if not tab:
        return []
    ordered = sorted(tab, key=lambda t: (t.note.onset, t.position.string))
    out: list[tuple[float, list[TabNote]]] = []
    index = 0
    while index < len(ordered):
        anchor = ordered[index].note.onset
        end = index
        while end < len(ordered) and ordered[end].note.onset - anchor <= window_s:
            end += 1
        out.append((anchor, list(ordered[index:end])))
        index = end
    return out


def render_ascii_with_legend(
    tab: Sequence[TabNote],
    tuning: Tuning = Tuning.STANDARD,
    uncertainty_threshold: float = DEFAULT_UNCERTAINTY_THRESHOLD,
    seconds_per_char: float = DEFAULT_SECONDS_PER_CHAR,
) -> str:
    """ASCII tab plus a short legend explaining the v1 limitations honestly."""
    body = render_ascii(
        tab,
        tuning=tuning,
        uncertainty_threshold=uncertainty_threshold,
        seconds_per_char=seconds_per_char,
    )
    uncertain = sum(1 for t in tab if t.posterior < uncertainty_threshold)
    legend = [
        "",
        f"Spacing is proportional to time ({seconds_per_char:.3g} s per character).",
        "There are no bar lines or note values: v1 does not do rhythmic notation.",
        f"(n) marks a note whose position the decoder is unsure of "
        f"(posterior < {uncertainty_threshold:g}): {uncertain} of {len(tab)}.",
    ]
    return body + "\n" + "\n".join(legend)

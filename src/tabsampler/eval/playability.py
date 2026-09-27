"""E3: is the output physically playable? (spec 3.1, ADR 0011)

Pure: no I/O, no global state.

**E3 needs no reference tab.** That is what makes it the only metric runnable on
arbitrary audio, including my own playing, and the one that catches the failure E2
cannot see: output that is pitch-correct and physically impossible.

Rules, from ADR 0011, passed in as data so they can change without editing this code:

- span of fretted notes at most 4 below fret 12, at most 5 at fret 12 and above
  (frets narrow as you go up);
- open strings are free and excluded from the span;
- at most 4 **fingers** (ADR 0019, superseding ADR 0011's count of fretted *notes*);
- one note per string;
- hand movement at most 12 frets per second between consecutive groups.

ADR 0011 counted fretted notes and so called every full barre chord unplayable. ADR 0019
counts fingers instead: one finger covers every string at the **lowest** fretted fret, and
each note above it costs a finger of its own. The discount is confined to the lowest fret
because the other fingers are already committed and cannot also lie flat across strings --
which is what stops the relaxation from accepting a barre plus four fingers over nine
frets. ``allow_barre=False`` restores ADR 0011's behaviour.

Group and transition rates are reported **separately**: a single conflated number hides
which rule is failing.

These functions work on plain tuples of :class:`Position` rather than on
:class:`ChordState`. ``ChordState`` refuses to hold two notes on one string, which is the
right invariant for the decoder -- but E3 must be able to *report* a tab that has one,
including a tab from somewhere other than our decoder.

**Caveat that belongs with any quoted number.** ADR 0011 owes a validation: the share of
real, human-made tab that passes these rules should be near 100%, and that check needs
DadaGP. Until it runs, read E3 as "passes our current rules", not "playable".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from tabsampler.types import Position, TabNote

#: (onset in seconds, the positions sounding at that onset)
Shape = tuple[float, tuple[Position, ...]]


@dataclass(frozen=True, slots=True)
class PlayabilityRules:
    """ADR 0011's numbers, with ADR 0019's finger counting."""

    max_span_low: int = 4
    max_span_high: int = 5
    high_neck_fret: int = 12
    #: Fingers on the fretting hand. Renamed from ADR 0011's ``max_fretted_notes``,
    #: which counted notes and so failed every barre chord (ADR 0019).
    max_fingers: int = 4
    #: Whether one finger may cover several strings at the lowest fretted fret.
    allow_barre: bool = True
    max_frets_per_second: float = 12.0

    def max_span_at(self, lowest_fretted: int) -> int:
        """The span allowed for a shape whose lowest fretted note is here."""
        return self.max_span_high if lowest_fretted >= self.high_neck_fret else self.max_span_low


@dataclass(frozen=True, slots=True)
class PlayabilityReport:
    """E3, split by what is being judged."""

    n_groups: int
    n_groups_pass: int
    n_transitions: int
    n_transitions_pass: int
    failures: tuple[str, ...] = field(default=())

    @property
    def group_rate(self) -> float:
        """Share of chord shapes that are playable. 1.0 when there are none."""
        return self.n_groups_pass / self.n_groups if self.n_groups else 1.0

    @property
    def transition_rate(self) -> float:
        """Share of hand moves that are possible. 1.0 when there are none."""
        return self.n_transitions_pass / self.n_transitions if self.n_transitions else 1.0

    @property
    def overall_rate(self) -> float:
        """Groups and transitions pooled. Reported alongside the two, never instead."""
        total = self.n_groups + self.n_transitions
        if not total:
            return 1.0
        return (self.n_groups_pass + self.n_transitions_pass) / total


def fretted_frets(positions: Sequence[Position]) -> tuple[int, ...]:
    """Frets needing a finger, ascending. Open strings need none."""
    return tuple(sorted(p.fret for p in positions if not p.is_open))


def _barre_is_blocked(positions: Sequence[Position], barred: Sequence[Position]) -> bool:
    """Whether an open string sits inside the run of strings the barre would cover.

    The barring finger lies flat across a contiguous run, so it presses every string
    between the outermost notes it is holding. A string that must ring **open** inside
    that run is therefore a conflict. A string fretted *higher* inside the run is not:
    the barre presses it too, but it sounds at its own higher fret, which is exactly how
    an E-shape barre chord works.
    """
    low = min(p.string for p in barred)
    high = max(p.string for p in barred)
    return any(p.is_open and low < p.string < high for p in positions)


def fingers_needed(positions: Sequence[Position], rules: PlayabilityRules) -> int:
    """How many fretting fingers the shape needs (ADR 0019).

    With ``allow_barre``, one finger barres every string at the lowest fretted fret and
    each note above it costs a finger -- unless an open string sits inside the barre's
    run of strings, which rules the barre out and charges those notes individually.
    Without ``allow_barre``, every fretted note costs a finger, which is ADR 0011's
    original rule. Open strings cost nothing either way.
    """
    fretted = [p for p in positions if not p.is_open]
    if not fretted:
        return 0
    if not rules.allow_barre:
        return len(fretted)

    lowest = min(p.fret for p in fretted)
    barred = [p for p in fretted if p.fret == lowest]
    above = len(fretted) - len(barred)
    if len(barred) > 1 and _barre_is_blocked(positions, barred):
        # No discount: the barre is impossible, so fall back to one finger per note,
        # which is exactly the count allow_barre=False would give.
        return len(fretted)
    return 1 + above


def hand_position(positions: Sequence[Position]) -> int | None:
    """Lowest fretted fret, or None for an all-open shape (spec 2.2)."""
    fretted = fretted_frets(positions)
    return fretted[0] if fretted else None


def group_is_playable(positions: Sequence[Position], rules: PlayabilityRules) -> str | None:
    """None if the shape is playable, else a short reason."""
    strings = [p.string for p in positions]
    if len(set(strings)) != len(strings):
        return f"two notes on one string: {sorted(strings)}"

    fingers = fingers_needed(positions, rules)
    if fingers > rules.max_fingers:
        return f"{fingers} fingers exceeds {rules.max_fingers}"
    fretted = fretted_frets(positions)
    if fretted:
        allowed = rules.max_span_at(fretted[0])
        span = fretted[-1] - fretted[0]
        if span > allowed:
            return f"span {span} exceeds {allowed} at fret {fretted[0]}"
    return None


def transition_is_playable(
    previous: Sequence[Position],
    current: Sequence[Position],
    seconds: float,
    rules: PlayabilityRules,
) -> str | None:
    """None if the hand can make the move in ``seconds``, else a short reason.

    An all-open shape has no hand position, so nothing has to move.
    """
    a, b = hand_position(previous), hand_position(current)
    if a is None or b is None:
        return None
    distance = abs(b - a)
    if distance == 0:
        return None
    if seconds <= 0.0:
        return f"{distance}-fret jump with no time between groups"
    speed = distance / seconds
    if speed > rules.max_frets_per_second:
        return f"{speed:.1f} frets/s exceeds {rules.max_frets_per_second}"
    return None


def group_tab_by_onset(tab: Sequence[TabNote], window_s: float = 0.03) -> list[Shape]:
    """Reconstruct (onset, positions) shapes from a flat tab.

    Uses the same windowing rule as
    :func:`tabsampler.fingering.candidates.group_notes` -- anchored at each group's
    first onset rather than chained -- so this grouping matches the decoder's.
    """
    if not tab:
        return []
    ordered = sorted(tab, key=lambda t: (t.note.onset, t.note.pitch))
    out: list[Shape] = []
    index = 0
    while index < len(ordered):
        anchor = ordered[index].note.onset
        end = index
        while end < len(ordered) and ordered[end].note.onset - anchor <= window_s:
            end += 1
        out.append((anchor, tuple(t.position for t in ordered[index:end])))
        index = end
    return out


def playability_rate(
    tab: Sequence[TabNote],
    rules: PlayabilityRules | None = None,
    window_s: float = 0.03,
) -> PlayabilityReport:
    """E3 over a whole tab. No reference needed."""
    active = rules or PlayabilityRules()
    shapes = group_tab_by_onset(tab, window_s=window_s)

    failures: list[str] = []
    groups_pass = 0
    for onset, positions in shapes:
        reason = group_is_playable(positions, active)
        if reason is None:
            groups_pass += 1
        else:
            failures.append(f"group @{onset:.3f}s: {reason}")

    transitions_pass = 0
    for index in range(1, len(shapes)):
        prev_onset, prev_positions = shapes[index - 1]
        onset, positions = shapes[index]
        reason = transition_is_playable(prev_positions, positions, onset - prev_onset, active)
        if reason is None:
            transitions_pass += 1
        else:
            failures.append(f"transition @{onset:.3f}s: {reason}")

    return PlayabilityReport(
        n_groups=len(shapes),
        n_groups_pass=groups_pass,
        n_transitions=max(len(shapes) - 1, 0),
        n_transitions_pass=transitions_pass,
        failures=tuple(failures),
    )

# ADR 0007: Contract collections are tuples, and the three missing types are defined here

Status: accepted (2026-09-27)

A deviation from the literal text of spec 2.1, plus a gap in it. The project requires an ADR
before changing a data contract, so both are recorded here rather than in a code comment.

## Context

Spec 2.1 sketches the contracts as:

```python
@dataclass(frozen=True)
class NoteEvent:
    ...
    bend: list[float] | None = None

@dataclass(frozen=True)
class TabNote:
    ...
    alternatives: list[tuple[Position, float]]
```

A `list` inside a `frozen=True` dataclass does not do what the word "frozen" suggests.
The dataclass refuses *attribute assignment*, so `note.bend = [...]` raises — but
`note.bend.append(3.0)` succeeds, and the instance is unhashable, so it cannot go in a set
or serve as a dict key. `eval/metrics.py` wants exactly that: `Position` keys for per-note
posteriors, and set operations while matching reference against prediction.

Separately, spec 2.1's Protocols reference `NoteGroup`, `ChordState` and `Context`, and the
spec never defines them. They cannot be left undefined, since the `FingeringScorer`
signature is written in terms of all three.

## Decision

1. **Collections in the contracts are tuples.** `NoteEvent.bend` is
   `tuple[float, ...] | None`; `TabNote.alternatives` is
   `tuple[tuple[Position, float], ...]`. Every contract is genuinely immutable and hashable.
2. **Runtime guards, not just annotations.** Each `__post_init__` raises `TypeError` if a
   list arrives where a tuple is declared. These objects are constructed from parsed
   basic-pitch CSV, mirdata annotations and YAML config, so the declared type is a promise
   rather than a fact. The guards carry a narrow
   `# pyright: ignore[reportUnnecessaryIsInstance]`, because pyright correctly observes
   that they are unnecessary *statically*.
3. **`NoteGroup`, `ChordState`, `Context` and `CostWeights` are defined in `types.py`.**
   `NoteGroup` requires its notes pre-sorted by `(pitch, onset)` and offers `NoteGroup.of()`
   to sort them, so a hand-built group cannot silently differ from one the grouping stage
   produced. `ChordState` rejects two notes on one string at construction, and exposes
   `fretted_frets`, `span` and `hand_position` — the last returning `None` for an all-open
   shape, which the transition cost must handle rather than treat as fret 0.
4. **`Tuning.pitch_at` is the single source of pitch arithmetic**, with fret numbers
   relative to the capo. Candidate generation and the E4 metric both call it, so they
   cannot disagree about what a fret sounds.

## Alternatives considered

- **Keep lists, exactly as the spec text.** Rejected: it makes `frozen` misleading and
  blocks hashing, which the metrics need.
- **Keep lists but copy defensively at every boundary.** Rejected: more code, more places
  to forget, and it still leaves the instances unhashable.
- **Validate at the parsing boundaries only, not in the contracts.** A reasonable position,
  and the usual advice for typed Python. Rejected because the cost of the guard is one line
  and the failure it prevents is silent.

## Consequences

**Easier.** Contracts are hashable and truly immutable, so they work in sets and as dict
keys, and can be compared with `==` freely. `hand_position` returning `None` makes the
all-open edge case impossible to overlook.

**Harder.** Callers must write `(x,)` rather than `[x]`, and code that builds bends
incrementally has to build a list and convert once at the end. Five one-line pyright
suppressions exist that a reader must not mistake for dead code.

**Revisit at:** Phase 7, when `NoteEvent.bend` stops being a carried-through unknown and
becomes a real contour with verified units.

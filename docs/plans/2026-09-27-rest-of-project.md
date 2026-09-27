# Tab Sampler — Plan for the Rest of the Project

> Tasks are executed in order; steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take Tab Sampler from M1 (a working, measured, hand-set baseline) to a system a
guitarist can use and a set of answered research questions: does a learned fingering model
beat the DP baseline, and does audio evidence improve string assignment.

**Architecture:** Unchanged — the spec §2 pipeline behind the §2.1 contracts. Everything
ahead either replaces one stage behind its existing interface (Phases 2, 3, 5, 6) or adds
an output surface (the app track). No stage boundary moves.

**Tech Stack:** Python 3.13, `uv`, numpy 2.5, scipy 1.18, librosa 1.0, `mirdata` 1.0,
`mir_eval` 0.8.2, `typer`. Verified available for work ahead: `pyguitarpro` 0.11,
`music21` 10.5.0, `fastapi` 0.141.1 + `uvicorn` 0.54.0, `torch` 2.14.0.

**Spec:** `docs/spec.md`.

---

## Scope: this is decomposed on purpose

The remaining spec covers six largely independent subsystems. Writing one plan for all of
them would produce a document nobody can execute and that goes stale on contact with the
first dataset. So:

| # | Chunk | Status | Plan |
|---|---|---|---|
| **A** | **Phase 1.5 — close the M1 gaps** | **unblocked, start here** | **in full below** |
| **B** | **App v1 — a guitarist can use it** | **unblocked** | **in full below** |
| C | Phase 2 — learned fingering, symbolic | **blocked on symbolic tab data** | outline below; own plan when unblocked |
| D | Phase 3 — audio conditioning | blocked on C + SynthTab | outline below |
| E | Phase 4 — real-data fine-tuning | blocked on D + GOAT/GAPS/Guitar-TECHS | outline below |
| F | Phase 5–7 — transcriber, full songs, techniques | blocked on E | outline below |

Each of C–F gets its own plan, written when its gate opens, because none of them can be
specified honestly before its data is in hand. Chunks A and B are specified fully here
because they need nothing that does not already exist in this repository.

## The one blocking action, which is Ege's

**Send one email requesting two datasets.** Both are access-by-request for research use,
and Pedro Sarmento is a contact on both, so this is one message:

- **DadaGP** — 26,181 GuitarPro songs, 739 genres, plus a token format. Contact Dadabots
  or Pedro Sarmento (`@dadabots` / `@umpedronosapato`). Source:
  <https://github.com/dada-bots/dadaGP>
- **ProgGP** — 173 progressive-metal GuitarPro songs in DadaGP's token format. Contact
  Jack Loth or Pedro Sarmento (`@jackjamesloth` / `@umpedronosapato`). Source:
  <https://github.com/otnemrasordep/ProgGP>

**This single action gates three separate things**, which is why it is worth doing before
anything else in this plan:

1. Phase 2 entirely (chunk C).
2. Cost-weight tuning, still outstanding from M1 (ADR 0012).
3. ADR 0011's owed validation — the check that real human tab passes our playability rules,
   which the evidence already suggests it does not.

**Verified gotcha for whoever uses it:** `dadagp.py` works with **PyGuitarPro 0.6 only**,
not the 0.11 that resolves on Python 3.13. It needs its own pinned environment, exactly the
pattern ADR 0001 already established for basic-pitch. Do not try to make one env serve both.

The dadaGP encoder/decoder itself is public and MIT-licensed, so the token format can be
studied now; only the corpora need the request.

---

## Global Constraints

Every task's requirements implicitly include this section.

- **Python** `requires-python = ">=3.13,<3.14"` (ADR 0001). A tool needing another version
  gets its own isolated environment and a subprocess boundary, never a downgrade of the core.
- **`uv run pyright` strict must pass on `src/`.** Confine untyped-library calls to one
  small function with an explicit return type and a narrow `# pyright: ignore[...]`.
- **Purity:** no I/O and no global state in `decode/`, `fingering/`, `eval/`.
- **Log space** for all decoder probability arithmetic, via `scipy.special.logsumexp`.
- **GuitarSet and EGDB are test-only** (ADR 0003). `assert_tuning_allowed(Split.TEST)`
  raises; every look is logged with a written reason.
- **Never invent results.** A metric not computed is blank in `results.csv`, never 0.0.
- **No cross-paper comparison** (spec §3.4), including TART's figures in the spec.
- **One variable per experiment**, hypothesis written before the run.
- **Contract changes need an ADR** (ADR 0007). Accepted ADRs are never edited.
- **CI stays offline and under 5 minutes.** No test may need `data/` or the transcriber.
- **Commits:** small, imperative, one feature per branch.
- `make check` must pass before every commit. Piping it to `tail` hides its exit code.

## Review Focus

Failure modes the spec implies that no task's headline test would otherwise exercise. Each
has a test pinned to the task that owns it.

1. **A rendered export that silently invents rhythm.** MusicXML and Guitar Pro both require
   note durations to mean something, and ADR 0009 says v1 has none. An export that writes
   every note as an eighth note looks authoritative and is a lie. → Task B1 refuses to
   export until a rhythm decision exists; Task A5 records the decision.
2. **The web UI showing a confident-looking tab built from 0.2-posterior notes.** M1
   measured ECE at 0.37 end to end: the numbers are not yet honest, so a UI that renders
   them as certainty misleads worse than ASCII does. → Task B3.
3. **An uploaded file that is not guitar, not audio, or enormous.** A local UI still gets a
   40-minute podcast, a 2 GB WAV, or a `.txt` renamed to `.wav`. → Task B2.
4. **Hand-position carry-forward changing decoder output silently.** Task A1 changes the
   cost model, so every M1 number moves. If the golden clip and the results table are not
   both refreshed in the same commit, the README describes a system that no longer exists.
   → Task A1 Steps 6–8.
5. **A barre rule that accepts physically impossible shapes.** Task A2 relaxes E3 by
   treating same-fret notes as one finger; relaxed too far it would accept a barre plus
   four fingers spanning nine frets. → Task A2.

---

# CHUNK A — Phase 1.5: close the M1 gaps

**Why first:** M1 shipped with three known defects and one undone decision, all recorded in
`docs/devlog/2026-09-27-m1-retrospective.md`. Each is small, each needs no new data, and two
of them change reported numbers — so they belong before any new capability is layered on top.

**Gate:** `make eval-m1` rerun, README results table refreshed, and every number in the repo
traceable to a `results.csv` row from a run actually executed.

---

### Task A1: Charge hand movement across all-open chords

**Files:**
- Modify: `src/tabsampler/fingering/costs.py`
- Modify: `src/tabsampler/decode/viterbi.py`, `src/tabsampler/decode/forward_backward.py`
- Modify: `tests/decode/brute_force.py`
- Test: `tests/fingering/test_costs.py`, `tests/decode/test_viterbi.py`

**Interfaces:**
- Consumes: `ChordState.hand_position -> int | None`, `HandSetScorer.transition_cost`,
  `build_lattice(groups, ctx, spans)`.
- Produces: `HandSetScorer.transition_cost_from(previous_hand: int | None, curr: ChordState) -> float`,
  and lattice nodes carrying a `carried_hand: int | None`.

**The defect.** `transition_cost` is stateless by contract, so it returns 0.0 whenever
either shape is all-open. A passage that goes fret 2 → open chord → fret 10 is therefore
charged **no** hand movement, though the hand really travelled 8 frets. `costs.py` documents
this; M1 shipped with it.

**Why the obvious fix is wrong.** Augmenting every lattice node with a hand position
multiplies the state space by up to `max_fret + 1 = 23`. Viterbi is `O(T·S²)`, and measured
mean states per group is 4.92, so that would take transition work from ~24 to ~12 800 per
step — a 530× regression to fix a rare case.

**The fix that is cheap.** A shape that has any fretted note *determines* its own hand
position, so it needs no augmentation. Only **all-open** states need to carry one, and the
value they can carry is limited to the distinct hand positions present in the previous
level — measured at ≤5. Augment the lattice only at all-open states.

- [ ] **Step 1: Write the failing cost test**

```python
def test_movement_is_charged_across_an_intervening_open_chord() -> None:
    s = scorer()
    # fret 2 -> all-open -> fret 10. The hand really moves 8 frets.
    assert s.transition_cost_from(2, state((1, 0))) == 0.0          # nothing to move to
    assert s.transition_cost_from(2, state((0, 10))) == pytest.approx(8.0 * s.weights.move)
    # An all-open shape carries the previous position forward rather than resetting to 0.
    assert s.carry(previous_hand=2, curr=state((1, 0))) == 2
    assert s.carry(previous_hand=2, curr=state((0, 10))) == 10
    assert s.carry(previous_hand=None, curr=state((1, 0))) is None
```

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run pytest tests/fingering/test_costs.py -k open_chord -v`
Expected: FAIL, `AttributeError: 'HandSetScorer' object has no attribute 'transition_cost_from'`

- [ ] **Step 3: Add the two methods to `HandSetScorer`**

```python
def carry(self, previous_hand: int | None, curr: ChordState) -> int | None:
    """The hand position after playing ``curr``. An all-open shape inherits."""
    here = curr.hand_position
    return previous_hand if here is None else here

def transition_cost_from(self, previous_hand: int | None, curr: ChordState) -> float:
    """Movement cost given where the hand was, not which shape it was in."""
    here = curr.hand_position
    if previous_hand is None or here is None:
        return 0.0
    return self.weights.move * abs(here - previous_hand)
```

Keep `transition_cost(prev, curr)` as `transition_cost_from(prev.hand_position, curr)` so
the `FingeringScorer` Protocol is unchanged and needs no ADR.

- [ ] **Step 4: Augment the lattice only where needed**

In `build_lattice`, return nodes of `(ChordState, carried_hand)`. For a state with a fretted
note, `carried_hand` is that state's own hand position and there is exactly one node. For an
all-open state, emit one node per distinct `carried_hand` reachable from the previous level.
Assert in a test that mean nodes per group stays within 1.5× of mean states per group.

- [ ] **Step 5: Teach the brute force to carry too**

`path_cost` must fold the carry through the path, or the oracle will disagree with the fix
and the oracle is the specification:

```python
def path_cost(groups, path, scorer, ctx) -> float:
    total = sum(scorer.emission_cost(g, s, ctx) for g, s in zip(groups, path, strict=True))
    hand = None
    for state in path:
        total += scorer.transition_cost_from(hand, state)
        hand = scorer.carry(hand, state)
    return total
```

Note the first group now contributes a transition cost of 0.0 because `hand` starts `None`,
which keeps the old behaviour for the first group.

- [ ] **Step 6: Run the oracle suite**

Run: `make oracle`
Expected: PASS. If Viterbi now disagrees with brute force, the augmentation is wrong, not
the oracle.

- [ ] **Step 7: Regenerate the golden clip and say why it changed**

The cost model changed, so `tests/fixtures/golden_clip.txt` must change. Regenerate it, and
**diff it by eye**: the new fingering should sit in one region more consistently than the old
one. If it does not, the fix made the model worse and that is a finding, not a regression to
paper over.

- [ ] **Step 8: Rerun `make eval-m1` and update the README table in the same commit**

Every M1 number moves. A commit that changes the cost model without refreshing the results
table leaves the README describing a system that no longer exists.

- [ ] **Step 9: Commit**

```bash
git add -A && git commit -m "Charge hand movement across all-open chords"
```

---

### Task A2: Model barre chords in E3

**Files:**
- Modify: `src/tabsampler/eval/playability.py`
- Test: `tests/eval/test_playability.py`

**Interfaces:**
- Consumes: `PlayabilityRules`, `fretted_frets(positions)`.
- Produces: `fingers_needed(positions: Sequence[Position], rules: PlayabilityRules) -> int`;
  `PlayabilityRules.max_fingers: int = 4` **renamed from** `max_fretted_notes` (it counted
  notes, it now counts fingers); `PlayabilityRules.allow_barre: bool = True` added.
- **The rename is config-visible.** `config.py`'s `_require_known_keys("rules", ...)` names
  `max_fretted_notes`; change it to `max_fingers` and add `allow_barre`, or an existing config
  starts failing with "unknown key" — which is the loader working correctly, not a bug.

**The defect.** `max_fretted_notes = 4` counts *notes*, so an E-shape barre chord — six
notes, five of them fretted — is called unplayable. Every full barre chord in GuitarSet is
currently scored as a failure, which makes E3's group rate pessimistic by an unknown amount.

**The guitar rule.** One finger can cover several strings **at the same fret**, and that
barring finger must be at the **lowest** fret in the shape. So fingers needed = number of
distinct fretted frets, and a barre is free at the lowest fret only.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_full_barre_chord_is_playable() -> None:
    # E-shape barre at fret 5: notes at 5,7,7,6,5,5 -> distinct frets {5,6,7} = 3 fingers.
    shape = pos((0, 5), (1, 7), (2, 7), (3, 6), (4, 5), (5, 5))
    assert group_is_playable(shape, RULES) is None

def test_five_distinct_fretted_frets_is_still_impossible() -> None:
    # Five different frets needs five fingers; a hand has four.
    shape = pos((0, 5), (1, 6), (2, 7), (3, 8), (4, 9))
    reason = group_is_playable(shape, RULES)
    assert reason is not None and "fingers" in reason

def test_a_barre_above_the_lowest_fret_does_not_get_the_discount() -> None:
    # Review Focus 5: two notes at fret 9 cannot be barred while fret 5 is held below them.
    shape = pos((0, 5), (1, 9), (2, 9), (3, 7), (4, 6))
    assert group_is_playable(shape, RULES) is not None

def test_barre_modelling_can_be_switched_off() -> None:
    strict = PlayabilityRules(allow_barre=False)
    shape = pos((0, 5), (1, 7), (2, 7), (3, 6), (4, 5), (5, 5))
    assert group_is_playable(shape, strict) is not None
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/eval/test_playability.py -k barre -v`
Expected: FAIL on the first — the current rule rejects 5 fretted notes.

- [ ] **Step 3: Implement `fingers_needed` and use it**

Distinct fretted frets when `allow_barre`; count of fretted notes otherwise. The barre
discount applies only to the lowest fretted fret: notes above it at a shared fret still need
one finger each.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/eval/test_playability.py -v`
Expected: PASS, and the existing span tests must still pass unchanged.

- [ ] **Step 5: Rerun `make eval-m1` and record how much E3 moved**

Write the before/after E3 group rate into the devlog. This is the measurement that says how
pessimistic M1's E3 was; the M1 README number was 0.9835 oracle, 0.9751 end to end.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "Model barre chords in the E3 playability rules"
```

---

### Task A3: Report E1 at both measurement points

**Files:**
- Modify: `src/tabsampler/eval/harness.py`, `src/tabsampler/cli.py`
- Test: `tests/eval/test_harness.py`

**Interfaces:**
- Consumes: `evaluate_full(...)`, `FullReport`.
- Produces: `FullReport.e1_incoming: PRF` alongside the existing `FullReport.e1_onset`.

**The question this settles.** M1's end-to-end E1 was 0.7452 against Phase 0's 0.7437 for
the same transcriber on the same audio. The difference is that `decode_best_effort` drops
notes the guitar cannot sound, which raises precision. So "note F1" was silently measuring
two different things in two different tables.

**The answer:** report both, labelled. E1 on the transcriber's raw output is the
**transcriber's** score. E1 on the notes that survived placement is the **pipeline's**.
Neither is wrong; conflating them is.

- [ ] **Step 1: Write the failing test**

```python
def test_e1_is_reported_before_and_after_the_fingering_stage() -> None:
    # Two reference notes. The estimator returns three, one of which is unplayable
    # (MIDI 39, below the open low E) and is dropped during placement.
    report = evaluate_full(..., notes_in=lambda _: est_notes([40, 45]) + [n(2.0, 39)], ...)
    assert report.e1_incoming.n_est == 3   # what the transcriber said
    assert report.e1_onset.n_est == 2      # what survived placement
    assert report.e1_onset.f1 >= report.e1_incoming.f1
```

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run pytest tests/eval/test_harness.py -k both_ -v`
Expected: FAIL, `AttributeError: 'FullReport' object has no attribute 'e1_incoming'`

- [ ] **Step 3: Compute E1 on `incoming` before `place` is called**

`evaluate_full` already has `incoming` in hand. Score it with the same
`note_f1_arrays(..., match_offsets=False)` call and store it as `e1_incoming`.

- [ ] **Step 4: Add both rows to the printed table and the results notes**

Label them "E1 transcriber (raw)" and "E1 pipeline (placed)". In oracle mode they are equal
by construction, which is itself a useful check.

- [ ] **Step 5: Run the tests, then `make eval-m1`, then fix the README's E1 row**

Run: `make check` then `make eval-m1`
Expected: two E1 rows; the README gains the distinction and the Phase 0 comparison stops
being confusing.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "Report E1 before and after the fingering stage"
```

---

### Task A4: A synthetic round-trip diagnostic (explicitly not for tuning)

**Files:**
- Create: `src/tabsampler/eval/synthetic.py`, `tests/eval/test_synthetic.py`
- Modify: `src/tabsampler/cli.py` (a `diagnose` command)

**Interfaces:**
- Produces: `sample_playable_path(rng: np.random.Generator, n_groups: int, tuning: Tuning,
  rules: PlayabilityRules) -> list[tuple[NoteGroup, ChordState]]`, and
  `round_trip_accuracy(paths, scorer, ctx) -> RoundTripReport` where

```python
@dataclass(frozen=True, slots=True)
class RoundTripReport:
    n_notes: int             # total notes across all sampled paths
    n_recovered: int         # notes whose recovered Position equals the generating one
    n_paths: int
    n_single_candidate: int  # notes with one legal position -- free wins, reported apart

    @property
    def accuracy(self) -> float:   # micro-averaged over notes, never over paths
        return self.n_recovered / self.n_notes if self.n_notes else 1.0
```

**What this is for.** Sample a random *playable* fingering, read off the pitches it would
sound, feed only those pitches to the decoder, and ask whether it recovers the fingering. It
answers a question the oracle tests cannot: not "does Viterbi minimise the cost correctly"
but "does minimising this cost recover a fingering a player would use".

**⚠️ What this is NOT for, and this matters.** The generated fingerings are sampled from
*our own* notion of plausible. **Fitting cost weights or the temperature on them would be
circular** — it would tune the model to agree with its own prior and report the agreement as
accuracy. This is a diagnostic and a regression test. The real tuning signal is human tab,
which is what the DadaGP/ProgGP request is for. State this in the module docstring, in the
`diagnose` command's output, and in any devlog entry that quotes a number from it.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_sampled_path_is_playable_by_construction() -> None:
    rng = np.random.default_rng(0)
    for group, state in sample_playable_path(rng, n_groups=20, tuning=STANDARD, rules=RULES):
        assert group_is_playable(state.positions, RULES) is None

def test_sampling_is_deterministic_given_a_seed() -> None:
    a = sample_playable_path(np.random.default_rng(7), 10, STANDARD, RULES)
    b = sample_playable_path(np.random.default_rng(7), 10, STANDARD, RULES)
    assert [s.positions for _, s in a] == [s.positions for _, s in b]

def test_sampled_pitches_are_reproducible_from_the_sampled_positions() -> None:
    for group, state in sample_playable_path(np.random.default_rng(1), 15, STANDARD, RULES):
        for note, p in zip(group.notes, state.positions, strict=True):
            assert STANDARD.pitch_at(p.string, p.fret) == note.pitch

def test_round_trip_accuracy_is_one_when_every_note_has_a_single_candidate() -> None:
    # MIDI 86 is reachable only at high e fret 22, so any correct decoder recovers it
    # whatever the weights. Isolates decoder plumbing from cost-model quality.
    paths = [(NoteGroup.of([n(86)]), ChordState(positions=(Position(5, 22),)))]
    report = round_trip_accuracy(paths, SCORER, CTX)
    assert report.accuracy == 1.0
    assert report.n_single_candidate == 1

def test_round_trip_reports_per_note_accuracy_not_per_path() -> None:
    # One 10-note path recovered plus one 1-note path missed is 10/11, not 0.5 -- the
    # same micro-averaging rule the corpus metrics use.
    report = round_trip_accuracy(ten_recovered + one_missed, SCORER, CTX)
    assert report.n_notes == 11
    assert report.accuracy == pytest.approx(10 / 11)
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/eval/test_synthetic.py -v`
Expected: FAIL, `ModuleNotFoundError: tabsampler.eval.synthetic`

- [ ] **Step 3: Implement sampling**

Walk a hand position with a bounded random step, pick 1–4 strings, pick frets inside the
allowed span, reject shapes `group_is_playable` refuses. Pure: the `Generator` is passed in,
never created inside.

- [ ] **Step 4: Implement `round_trip_accuracy`**

Decode the sampled pitches, compare recovered positions against the generating ones, and
report per-note accuracy with the count — not a mean of per-path rates, for the same reason
the corpus metrics micro-average.

- [ ] **Step 5: Run, then wire `tabsampler diagnose --paths 200 --seed 0`**

The command prints the accuracy **with the circularity warning attached to the number**.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "Add a synthetic round-trip diagnostic for the fingering model"
```

---

### Task A5: Two decisions M1 left open

**Files:**
- Create: `docs/adr/0016-headline-target.md`, `docs/adr/0017-rhythm-for-exports.md`
- Modify: `docs/adr/README.md`

- [ ] **Step 1: ADR 0016 — set the D10 target**

ADR 0004 deferred the target until a baseline existed. It does now: end-to-end E2 = 0.4231,
oracle 0.6366. Propose the target as "baseline + X" with X argued from where the headroom
actually is — the transcriber costs 0.2134 of E2, so end-to-end gains are bounded by note F1
until Phase 5, and Phase 2/3 gains will show up in **oracle** mode first. A target that only
names an end-to-end number would be unreachable for reasons unrelated to the fingering work.
**Ege signs off on X; propose it, do not assume it.**

- [ ] **Step 2: ADR 0017 — what rhythm, if any, exports get**

Blocks Task B1. MusicXML and Guitar Pro both need durations to mean something, and ADR 0009
gave v1 none. Three options, with the honest cost of each: (a) refuse to export until beat
tracking exists (spec gates that after M3); (b) export at a fixed high-resolution grid with
a tempo marking and a prominent "rhythm not transcribed" annotation in the file itself;
(c) crude quantisation now with a named beat tracker, which pulls Phase-6-era work forward.
Recommend (b) — it is honest, cheap, and does not pretend.

- [ ] **Step 3: Update the ADR index and commit**

```bash
git add -A && git commit -m "Add ADRs 0016 and 0017: headline target and export rhythm"
```

---

# CHUNK B — App v1: a guitarist can use it

**Why now:** spec §6 puts the application track after M1, and M1 is met. It needs no data
that is not already here, and D14 ("app surface") is the only open decision that is already
due. It is also the first thing that makes the E5 = 0.37 calibration problem *visible* rather
than a number in a table.

**Sequencing insight that shapes this chunk.** ADR 0013 put MusicXML and Guitar Pro export
on the app track, but both formats need rhythm and ADR 0009 says v1 has none. The web UI
needs no rhythm at all — it can show time-positioned tab exactly as the ASCII renderer does,
and it is where uncertainty and alternatives actually belong. **So the UI comes first and
the exports are gated on ADR 0017.**

**Gate:** Ege drops one of his own recordings into a local web page and gets back playable
tab with uncertain notes visibly marked and alternatives on hover.

---

### Task B1: MusicXML and Guitar Pro export — gated on ADR 0017

**Files:**
- Create: `src/tabsampler/render/musicxml.py`, `src/tabsampler/render/gp.py`
- Test: `tests/render/test_musicxml.py`, `tests/render/test_gp.py`

**Do not start this task until ADR 0017 exists.** Writing an exporter that assigns every note
an eighth note produces a file that looks authoritative and misstates the music — Review
Focus 1. `music21` 10.5.0 and `pyguitarpro` 0.11 are both verified to resolve on Python 3.13;
neither is yet a dependency, and adding them needs the ADR first.

When unblocked, the tests that matter are: string and fret survive the round trip
(`<technical><string>`/`<fret>` in MusicXML), the file opens in MuseScore and TuxGuitar, the
"rhythm not transcribed" annotation is present in the file and not only in our docs, and a
capo is represented rather than silently folded into fret numbers.

---

### Task B2: Serve the pipeline locally over HTTP

**Files:**
- Create: `src/tabsampler/web/app.py`, `src/tabsampler/web/models.py`
- Create: `tests/web/test_app.py`
- Modify: `pyproject.toml` (a `web` dependency group: `fastapi`, `uvicorn`, `python-multipart`)
- Modify: `src/tabsampler/cli.py` (a `serve` command)

**Interfaces:**
- Consumes: `BasicPitchCLITranscriber.transcribe_file`, `group_notes`, `decode_best_effort`,
  `tab_to_dict`, `playability_rate`, `pitch_validity_rate`, `load_phase1_config`.
- Produces: `POST /api/transcribe` (multipart audio) → the `tab_to_dict` document plus
  `degradation` and `metrics` objects; `GET /api/health`.

**Bind to localhost only.** This is a local tool; there is no authentication and it shells
out to a transcriber. Binding `0.0.0.0` would expose file upload and subprocess execution to
the network. Default `127.0.0.1`, and make the host an explicit flag with a warning.

- [ ] **Step 1: Write the failing tests**

```python
def test_health_reports_whether_the_transcriber_is_available(client) -> None:
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert isinstance(body["transcriber_available"], bool)

def test_transcribe_returns_the_json_tab_document(client, wav) -> None:
    body = client.post("/api/transcribe", files={"audio": ("clip.wav", wav, "audio/wav")}).json()
    assert body["schema_version"] == 1
    assert "notes" in body and "degradation" in body and "metrics" in body

def test_degradation_is_surfaced_to_the_client_not_hidden(client, wav_with_subsonic) -> None:
    # A dropped note lowers recall; the UI must be able to say so.
    body = client.post("/api/transcribe", files={"audio": ("c.wav", wav_with_subsonic, "audio/wav")}).json()
    assert body["degradation"]["n_notes_out_of_range"] >= 1

def test_a_file_that_is_not_audio_gets_400_not_500(client) -> None:   # Review Focus 3
    r = client.post("/api/transcribe", files={"audio": ("x.wav", b"not audio", "audio/wav")})
    assert r.status_code == 400
    assert "could not be read as audio" in r.json()["detail"]

def test_a_file_over_the_size_limit_is_refused_with_413(client) -> None:  # Review Focus 3
    oversize = b"RIFF" + bytes(MAX_UPLOAD_BYTES + 1)
    r = client.post("/api/transcribe", files={"audio": ("big.wav", oversize, "audio/wav")})
    assert r.status_code == 413

def test_audio_longer_than_the_duration_limit_is_refused(client, long_wav) -> None:  # Review Focus 3
    # A 40-minute podcast is not isolated guitar (ADR 0002) and would block the request.
    r = client.post("/api/transcribe", files={"audio": ("long.wav", long_wav, "audio/wav")})
    assert r.status_code == 413
    assert "longer than" in r.json()["detail"]

def test_a_missing_transcriber_returns_503_with_setup_instructions(monkeypatch, client, wav) -> None:
    # TranscriberUnavailableError must not become an opaque 500.
    monkeypatch.setattr(app_module, "TRANSCRIBER_EXE", "definitely-not-installed")
    r = client.post("/api/transcribe", files={"audio": ("c.wav", wav, "audio/wav")})
    assert r.status_code == 503
    assert "setup_transcriber.sh" in r.json()["detail"]

def test_the_server_binds_to_localhost_by_default() -> None:
    # No auth, and it shells out to a subprocess. 0.0.0.0 would expose both.
    assert DEFAULT_HOST == "127.0.0.1"
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/web -v`
Expected: FAIL, `ModuleNotFoundError: tabsampler.web.app`

- [ ] **Step 3: Add the `web` dependency group and sync**

Run: `uv add --group web fastapi uvicorn python-multipart && uv sync --all-groups`
Keep it a **group**, not a core dependency: the CLI and the evaluation harness must stay
installable without a web stack.

- [ ] **Step 4: Implement the endpoints**

Map errors deliberately: unreadable audio → 400, too large → 413, too long → 413 with a
duration message, `TranscriberUnavailableError` → 503 naming
`scripts/setup_transcriber.sh`, `TranscriberFailedError` → 502. Never a bare 500.

- [ ] **Step 5: Run the tests and `make check`**
- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "Serve the transcription pipeline over a local HTTP API"
```

---

### Task B3: The tab view, with uncertainty made honest

**Files:**
- Create: `src/tabsampler/web/static/index.html`, `app.js`, `style.css`
- Test: `tests/web/test_static.py`

**Interfaces:**
- Consumes: `POST /api/transcribe` from Task B2.
- Produces: a single page served at `/`.

**The design constraint that matters.** M1 measured ECE at 0.37 end to end: a posterior of
0.8 does **not** mean 80% correct yet. So the UI must not render posteriors as if they were
calibrated probabilities — Review Focus 2. Show **relative** confidence (this note is less
certain than that one) and say plainly that the numbers are uncalibrated, with a link to the
E5 row. When Phase 1.5 or Phase 2 calibrates them, the wording changes; until then it is a
ranking, not a probability.

- [ ] **Step 1: Write the failing tests**

```python
def test_the_index_page_is_served_at_root(client) -> None:
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]

def test_the_page_states_that_confidences_are_uncalibrated(client) -> None:
    # Review Focus 2: an honest UI cannot present ECE 0.37 posteriors as probabilities.
    assert "uncalibrated" in client.get("/").text.lower()

def test_the_page_states_that_rhythm_is_not_transcribed(client) -> None:
    # ADR 0009: spacing is proportional to time; there are no bars or note values.
    assert "rhythm" in client.get("/").text.lower()

def test_static_assets_are_served(client) -> None:
    for path, kind in (("/static/app.js", "javascript"), ("/static/style.css", "css")):
        r = client.get(path)
        assert r.status_code == 200 and kind in r.headers["content-type"]
```

- [ ] **Step 2: Run them and watch them fail**
- [ ] **Step 3: Build the page**

Six string lines, high e on top, notes positioned by onset — the same layout as
`render_ascii`, which is already tested, so keep the two consistent. Uncertain notes visually
distinct (not red/green alone — Review Focus in reverse: colour-blind readers). Hover shows
the alternatives the JSON already carries. A visible banner for the degradation counts, so a
dropped note is never invisible.

- [ ] **Step 4: Run the tests, then drive it for real**

Use the `agent-browser` skill: start the server, upload a GuitarSet clip, screenshot the
result, and check the page against what `tabsampler transcribe` prints for the same file.
They must agree note for note.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "Add the local tab view with honest uncertainty display"
```

---

### Task B4: Make it runnable by someone who has not read this repo

**Files:**
- Modify: `README.md`, `Makefile`
- Create: `docs/using-the-app.md`

- [ ] **Step 1: `make serve`** wrapping `uvicorn`, with the two prerequisites checked and
      named (`scripts/setup_transcriber.sh`, `uv sync --all-groups`).
- [ ] **Step 2: Write `docs/using-the-app.md`** — what it does, what it cannot do (ADR 0002
      isolated guitar only, ADR 0009 no rhythm, uncalibrated confidences), and the E2 = 0.4231
      end-to-end number so expectations are set by measurement rather than by hope.
- [ ] **Step 3: Record a demo GIF** and put it in the README.
- [ ] **Step 4: Commit and tag `v0.2-app1`.**

---

# CHUNKS C–F — outlines only, each gets its own plan

These are outlines on purpose. None can be specified honestly before its data is in hand,
and a fake-detailed plan for them would be worse than none. **Write the real plan when the
gate opens.**

### Chunk C — Phase 2: learned fingering, symbolic only → M2

**Gate:** symbolic tab data in hand (the email above).

Spec §6 and D12. Shape of the work: an isolated PyGuitarPro 0.6 environment for the dadaGP
tokeniser (ADR 0001's pattern); `data/tokenize.py` designing the token format; splits from
DadaGP/ProgGP with `assert_tuning_allowed`; a small encoder-decoder trained from scratch
(D12's cue) on the 4070 with ONNX export for Mac inference (D11); **constrained decoding so
only pitch-valid tokens can be generated**; and the controlled comparison in **oracle mode**
— Viterbi vs transformer vs transformer-probabilities-decoded-by-Viterbi.

Two things to get right, both already visible from M1:
- **The learned model's probabilities must feed the existing decoder**, not replace it. The
  `FingeringScorer` Protocol exists for exactly this, and keeping it means E3 playability and
  E5 calibration keep working unchanged.
- **Do the outstanding cost-weight tuning first** (ADR 0012), on the same new data. Otherwise
  the comparison is a tuned transformer against an untuned baseline, which is not a
  comparison. This is the single most likely way to get M2 wrong.

**Done when:** the comparison table exists. A result where the transformer does *not* beat
Viterbi is a valid result and gets written up as one.

### Chunk D — Phase 3: audio conditioning → M3

**Gate:** Chunk C done, SynthTab obtained.

D13's cue is (b) first: a TabCNN-style head predicting per-frame string probabilities, which
plugs straight into the decoder's existing `weights.acoustic` term — already in `CostWeights`
and identically zero, so the interface does not change. Then (a), a per-note CNN embedding.
**The ablation that removes the audio input is the deliverable**, not the improvement.

### Chunk E — Phase 4: real-data fine-tuning

**Gate:** Chunk D done; GOAT, GAPS, Guitar-TECHS obtained.

Fine-tune on GOAT/GAPS/Guitar-TECHS; evaluate on GuitarSet **and EGDB**, both held out.
Decides D2 (primary guitar sound). **Done when** both test sets are reported, including the
acoustic-versus-electric gap. Add EGDB to `data/splits.py` with the same snapshot-and-guard
mechanism GuitarSet has — do not weaken the guard to accommodate a second test set.

### D15 — licensing and publication, due before any public release

Not a phase, but a spec decision with no owner until now. The repository is **private** (created so
deliberately: D15 is undecided and there is no LICENSE file). The trigger for this ADR is
**making the repository public, publishing weights, or publishing a demo** — whichever
comes first.

What it has to settle, and why each one is not obvious:

- **Code licence.** Spec D15's cue is MIT or Apache-2.0. Cheap to decide, needed before
  anyone can fork it.
- **Weights, per dataset.** This is the part that bites. DadaGP is research-use-by-request,
  ProgGP likewise, and SynthTab and GOAT have their own terms. A model trained on
  research-only data generally cannot be redistributed, so **weight release is decided per
  training corpus, not once.** Check each dataset's terms *before* Chunk C trains anything
  whose weights might be published.
- **What the README may claim.** Every number in it must still trace to a `results.csv`
  row from a run that executed, and the no-cross-paper-comparison rule (spec §3.4) does not
  relax because the audience got bigger.

### Chunk F — Phases 5–7

- **Phase 5, better transcriber.** The measured prize is large: the transcriber costs 0.2134
  of E2, which is more than any plausible fingering gain. **Done when end-to-end E2 improves
  — note F1 improving alone is explicitly not enough.** Note that ADR 0001's isolation makes
  swapping the transcriber a one-adapter change.
- **Phase 6, full songs.** `htdemucs_6s` guitar stem, with the stem-plus-mix experiment.
  Report separately from isolated mode (ADR 0002). Watch decode cost here: ADR 0015 closed the
  native-port question on the basis that decode is 3% of runtime, and full songs are the one
  scenario that could change that. `state_count_stats` is the measurement to re-run.
- **Phase 7, techniques.** Bends first — but **verify `NoteEvent.bend`'s units before
  building on it**. Open issue (c) from the M1 plan is still open: basic-pitch's `pitch_bend`
  is in internal contour bins and the bins-per-semitone factor has never been checked.

---

## Verification

**Every task:** `make check` (lint, `pyright --strict`, 294+ tests, offline) before commit.
Check the exit code; piping to `tail` hides it.

**The correctness core:** `make oracle`. Task A1 changes the cost model, so the brute force
changes with it — if Viterbi and the oracle disagree after A1, the implementation is wrong,
not the oracle.

**Chunk A gate:**
```bash
make check && make oracle && make eval-m1     # then refresh the README table
uv run tabsampler diagnose --paths 200 --seed 0
```

**Chunk B gate:**
```bash
bash scripts/setup_transcriber.sh
make serve                                     # then open http://127.0.0.1:8000
uv run tabsampler transcribe my_riff.wav -o riff.txt   # must agree with the page
```

**The test that is not automated:** play the output on a guitar. It is the fastest
playability check there is and the only one that cannot be scripted. An E3 of 0.98 on tab
that feels wrong under the fingers means ADR 0011's rules are still wrong — which Task A2
already suspects.

## Notes for whoever executes this

- **Order:** A1 → A2 → A3 → A5 → A4, then B2 → B3 → B4, with B1 whenever ADR 0017 lands.
  A1 and A2 both move reported numbers, so do them before anything that quotes one.
- **A1 and A2 each change M1's results.** Rerun `make eval-m1` and refresh the README in the
  same commit as the change. Never leave the table describing an older system.
- **Task A4's numbers must never be used for tuning.** If you find yourself adjusting a
  weight to improve round-trip accuracy, stop: that is the circularity the task warns about.
- **The email is the critical path for everything past chunk B.** If it has not been sent,
  say so at the top of the session rather than starting chunk C work that cannot finish.

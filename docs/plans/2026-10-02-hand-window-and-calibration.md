# Hand Window, Fair Split and Calibrated Default — Implementation Plan

> Tasks run in order; steps use checkbox (`- [ ]`) syntax. Every run that produces a
> reported number is pre-registered in a commit made before it runs.

**Goal:** fix the phantom-movement flaw ADR 0022 found, give DadaGP an artist-disjoint split,
calibrate the default decoder's confidence, and give the fitted weights the fair test ADR 0023
asked for — then measure the default once on GuitarSet.

**Architecture:** the hand becomes a 4-fret *window* that moves only when a note falls outside
it. One pure function, `shift_window`, defines that rule and is shared by E3's transition
rule, the decoder's lattice, the cost model, and the CRF fitter; the brute-force oracle
re-derives it independently from ADR 0025's text. The lattice keeps ADR 0018's design — a node
is a shape plus the hand carried into it — but now every shape, not only all-open ones, can
carry more than one hand position.

**Tech Stack:** Python 3.13, `uv`, numpy, scipy, pytest + hypothesis, PyGuitarPro only via the
existing one-off script.

**Spec:** `docs/spec.md`; this plan implements tasks C1, C2 and C5 of
`docs/plans/2026-10-01-phase-2.md`.

## Decisions already taken by Ege (2026-10-02)

- The hand-set weights stay the default until fitted weights win **a fair test fixed in
  advance** (Task 7 states it). GuitarSet is never that test.
- The hand window is **fixed at 4 frets** (span units, as ADR 0011's limit below fret 12).
- The default decoder gets its own calibrated temperature.
- This plan stops after one GuitarSet measurement of the default; richer models come after review.

## Global Constraints

- `requires-python = ">=3.13,<3.14"` (ADR 0001). `uv run pyright` strict passes on `src/`.
- Pure functions in `decode/`, `fingering/`, `eval/`. All I/O in `cli.py`, `config.py`,
  `results.py`, `data/` loaders and `scripts/`.
- GuitarSet is test-only (ADR 0003): nothing in this plan chooses anything from it. Each
  GuitarSet run is pre-registered, logged, and reported whatever it shows.
- `experiments/results.csv` rows only from runs that executed; a metric not computed is blank.
- Never edit an accepted ADR; supersede. New ADRs this plan writes: **0024** artist split,
  **0025** hand window, **0026** default temperature, **0027** fitted-weights fair test.
- The repository is public: never commit `data/` or `cache/` (the datasets are
  research-use-only), nor local editor or tooling files.
- `make check` before every commit, checking its exit code, not piped output.
- Metric-code changes (Task 4) are flagged loudly in the session report.

## Review Focus

1. **Artist names that differ only by case** (`Mago de Oz` / `Mago de oz`) must land on the
   same side of the split. → Task 2 test `test_case_variants_of_one_artist_share_a_side`.
2. **A chord wider than the window** (span 5 at fret 12, allowed by ADR 0011) must not make
   the window jump back and forth or produce a negative distance. → Task 3 test
   `test_a_shape_wider_than_the_window_anchors_at_its_lowest_fret`.
3. **An all-open shape between two fretted ones** must carry the hand, in E3 *and* in the
   decoder. → Task 3 `test_an_all_open_shape_leaves_the_hand_where_it_was`, Task 4
   `test_a_jump_across_an_open_string_is_still_a_jump`.
4. **Two notes on one string** reach E3 from foreign tabs; the window rule must not crash on
   them. → Task 4 `test_a_shape_with_two_notes_on_one_string_still_moves_the_hand`.
5. **Lattice growth** — every fretted shape can now carry up to five hand positions; a silent
   25× decode slowdown would be a regression nobody notices. → Task 5 node-count guard and
   benchmark acceptance.

---

### Task 1: Merge and publish the DadaGP work

**Files:** none changed.

- [ ] **Step 1: Verify the branch is green**

Run: `make check; echo EXIT=$?` and `make oracle; echo EXIT=$?`
Expected: both `EXIT=0`.

- [ ] **Step 2: Confirm nothing from the dataset is tracked**

Run: `git ls-files | grep -Ei "dadagp.*\.zip|track_meta|^data/" ; echo "none expected"`
Expected: only `none expected` printed.

- [ ] **Step 3: Merge and push**

```bash
git checkout main
git merge --ff-only dadagp-fit
git push origin main
git branch -d dadagp-fit
git checkout -b hand-window
```

- [ ] **Step 4: Confirm CI**

Run: `gh run list --repo Afillex/TabSampler --limit 1`
Expected: `completed success` for the merge commit.

---

### Task 2: Artist-disjoint split (ADR 0024)

**Files:**
- Modify: `src/tabsampler/data/dadagp.py`
- Modify: `scripts/fit_cost_weights.py`, `scripts/calibrate_temperature.py` (a `--split` flag)
- Test: `tests/data/test_dadagp.py`
- Create: `docs/adr/0024-artist-disjoint-split.md`

**Interfaces:**
- Produces: `artist_of(key: str) -> str`;
  `artist_split(keys: Sequence[str], buckets: int = 10, validation_buckets: int = 1) -> dict[str, Split]`;
  `load_tracks(..., scheme: Literal["shipped", "artist"] = "shipped")`;
  constant `ARTIST_VALIDATION_SHA256: str`.

- [ ] **Step 1: Write the failing tests** (append to `tests/data/test_dadagp.py`)

```python
from tabsampler.data.dadagp import artist_of, artist_split


def test_an_artist_is_the_folder_under_the_letter() -> None:
    assert artist_of("M/Mago de Oz/Mago de Oz - Alma.gp4.tokens.txt") == "mago de oz"


def test_case_variants_of_one_artist_share_a_side() -> None:
    a = artist_split(["M/Mago de Oz/x.tokens.txt", "M/Mago de oz/y.tokens.txt"])
    assert len(set(a.values())) == 1


def test_no_artist_appears_on_both_sides() -> None:
    keys = [f"A/Artist{i % 37}/Song{i}.tokens.txt" for i in range(500)]
    assignment = artist_split(keys)
    sides: dict[str, set[Split]] = {}
    for key, side in assignment.items():
        sides.setdefault(artist_of(key), set()).add(side)
    assert all(len(s) == 1 for s in sides.values())
    assert set(assignment.values()) == {Split.TRAIN, Split.VALIDATION}


def test_the_artist_split_is_deterministic() -> None:
    keys = [f"A/Artist{i}/Song.tokens.txt" for i in range(100)]
    assert artist_split(keys) == artist_split(list(reversed(keys)))


def test_the_loader_serves_the_artist_scheme(tmp_path: Path) -> None:
    archive, meta = make_archive(tmp_path)
    expected = hashes_of(archive)
    train = {t.song for t in load_tracks(archive, Split.TRAIN, meta, expected_sha256=expected,
                                         scheme="artist", artist_sha256=None)}
    val = {t.song for t in load_tracks(archive, Split.VALIDATION, meta, expected_sha256=expected,
                                       scheme="artist", artist_sha256=None)}
    assert train | val == {"A/Artist/Song.gp4.tokens.txt", "B/Band/Tune.gp4.tokens.txt"}
    assert not train & val
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/data/test_dadagp.py -q`
Expected: collection error, `ImportError: cannot import name 'artist_of'`.

- [ ] **Step 3: Implement** (in `src/tabsampler/data/dadagp.py`)

```python
def artist_of(key: str) -> str:
    """The artist folder of a token-file key, case-folded: folders differing only by case
    are the same artist on a case-insensitive filesystem (ADR 0021)."""
    return key.split("/")[1].casefold()


def artist_split(
    keys: Sequence[str], buckets: int = 10, validation_buckets: int = 1
) -> dict[str, Split]:
    """Whole artists to one side: about ``validation_buckets / buckets`` of artists to validation."""
    out: dict[str, Split] = {}
    for key in keys:
        bucket = int(hashlib.sha256(artist_of(key).encode()).hexdigest(), 16) % buckets
        out[key] = Split.VALIDATION if bucket < validation_buckets else Split.TRAIN
    return out
```

In `load_tracks`, add parameters `scheme: Literal["shipped", "artist"] = "shipped"` and
`artist_sha256: str | None = ARTIST_VALIDATION_SHA256`. For `"artist"`: read and hash-check
**both** shipped split files, take the union of their keys, keep those `artist_split` assigns
to `split`. If `artist_sha256` is not `None`, compute
`hashlib.sha256("\n".join(sorted(validation_keys)).encode()).hexdigest()` and raise
`ValueError` mentioning "sha256" on mismatch. Start with `ARTIST_VALIDATION_SHA256 = ""` and
skip the check when it is empty, until Step 5 fills it.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/data/test_dadagp.py -q`
Expected: all pass.

- [ ] **Step 5: Freeze the real split**

Run:
```bash
uv run python -c "
import hashlib, json, zipfile
from tabsampler.data.dadagp import artist_split
from tabsampler.data.splits import Split
z = zipfile.ZipFile('data/dadagp/DadaGP-v1.1.zip')
keys = [e['tokens.txt'] for s in ('training', 'validation') for e in json.loads(z.read(f'DadaGP-v1.1/_DadaGP_{s}.json'))]
val = sorted(k for k, s in artist_split(keys).items() if s is Split.VALIDATION)
print(len(val), hashlib.sha256('\n'.join(val).encode()).hexdigest())"
```
Paste the printed digest into `ARTIST_VALIDATION_SHA256`, remove the empty-string skip, and
add `assert ARTIST_VALIDATION_SHA256.startswith("<first 8 hex chars printed>")` to
`test_the_committed_hashes_are_the_v1_1_release`.

- [ ] **Step 6: Add `--split {shipped,artist}` to both scripts**, passed through as `scheme=`.
The fit script also prints recovery by part (`clean` vs `distorted`, from
`track.instrument`), as the 2026-10-01 exploratory check did.

- [ ] **Step 7: Write ADR 0024** — whole artists to one side by SHA-256 bucket of the
case-folded artist folder, 1 bucket in 10 to validation, frozen by the digest from Step 5;
used for every fit from now on. Report the song and note counts on each side. Add it to
`docs/adr/README.md`.

- [ ] **Step 8: Re-baseline on the new validation side** (pre-registered: commit before running)

Commit first, then run:
`uv run python scripts/fit_cost_weights.py data/dadagp/DadaGP-v1.1.zip data/dadagp/track_meta.json --split artist --train-songs 600 --val-songs 300`
Record hand-set and fitted recovery (all / clean / distorted) as results rows,
`split=validation`, dataset `dadagp-v1.1 artist-disjoint`. These are the baselines Task 6
compares against.

- [ ] **Step 9: Commit**

```bash
git add src/tabsampler/data/dadagp.py tests/data/test_dadagp.py scripts/ docs/adr/ experiments/results.csv
git commit -m "Add an artist-disjoint DadaGP split as ADR 0024"
```

---

### Task 3: The hand-window rule

**Files:**
- Modify: `src/tabsampler/fingering/states.py`
- Test: `tests/fingering/test_states.py`

**Interfaces:**
- Produces: `HAND_WINDOW = 4`;
  `shift_window(previous_hand: int | None, fretted: Sequence[int], window: int = HAND_WINDOW) -> tuple[int | None, int]`
  returning (new hand, frets moved); `fretted` is ascending.
  `carry_hand(previous_hand, state)` keeps its signature and becomes `shift_window(previous_hand, state.fretted_frets)[0]`.

- [ ] **Step 1: Write the failing tests** (append to `tests/fingering/test_states.py`)

```python
from tabsampler.fingering.states import HAND_WINDOW, shift_window


def test_the_first_fretted_shape_places_the_hand_at_its_lowest_fret() -> None:
    assert shift_window(None, (5, 7)) == (5, 0)


def test_notes_inside_the_window_do_not_move_the_hand() -> None:
    # Fret 5 -> 7 -> 9 is a finger reaching inside one position, not the hand moving.
    assert shift_window(5, (7,)) == (5, 0)
    assert shift_window(5, (9,)) == (5, 0)


def test_a_note_above_the_window_moves_it_up_just_enough() -> None:
    assert HAND_WINDOW == 4
    assert shift_window(5, (11,)) == (7, 2)


def test_a_note_below_the_window_moves_it_down_to_that_note() -> None:
    assert shift_window(5, (3,)) == (3, 2)


def test_an_all_open_shape_leaves_the_hand_where_it_was() -> None:
    assert shift_window(5, ()) == (5, 0)
    assert shift_window(None, ()) == (None, 0)


def test_a_shape_wider_than_the_window_anchors_at_its_lowest_fret() -> None:
    # A 5-fret span at fret 12 is legal (ADR 0011) but wider than 4: anchor low, never negative.
    assert shift_window(2, (12, 17)) == (12, 10)
    assert shift_window(14, (12, 17)) == (12, 2)


def test_the_distance_is_never_negative() -> None:
    for prev in (None, 0, 3, 9, 20):
        for fretted in ((), (1,), (4, 8), (10, 15), (19,)):
            _, moved = shift_window(prev, fretted)
            assert moved >= 0
```

- [ ] **Step 2: Run and watch them fail**

Run: `uv run pytest tests/fingering/test_states.py -q`
Expected: `ImportError: cannot import name 'HAND_WINDOW'`.

- [ ] **Step 3: Implement** (in `states.py`, above `carry_hand`)

```python
#: Frets one hand position covers, in span units: fret h to fret h + 4, as ADR 0011's span
#: limit below fret 12. Fixed by decision, not fitted (ADR 0025).
HAND_WINDOW = 4


def shift_window(
    previous_hand: int | None, fretted: Sequence[int], window: int = HAND_WINDOW
) -> tuple[int | None, int]:
    """Where the hand's window starts after a shape, and how far it had to move (ADR 0025).

    The hand covers frets ``hand`` to ``hand + window`` and moves only when a fretted note
    falls outside that range, by the least distance that brings the shape inside it. A
    shape wider than the window anchors at its lowest fret. Open strings need no hand.
    """
    if not fretted:
        return previous_hand, 0
    low, high = fretted[0], fretted[-1]
    if previous_hand is None:
        return low, 0
    if low >= previous_hand and high <= previous_hand + window:
        return previous_hand, 0
    new = low if low < previous_hand else min(low, high - window)
    return new, abs(new - previous_hand)
```

and replace `carry_hand`'s body with `return shift_window(previous_hand, state.fretted_frets)[0]`,
updating its docstring to cite ADR 0025 instead of the lowest-fret rule.

- [ ] **Step 4: Run** `uv run pytest tests/fingering/test_states.py -q` — Expected: PASS.
Other suites will now fail where they pinned the old rule; Tasks 4 and 5 own them. Do not
commit until Task 5 is green.

---

### Task 4: E3's transition rule on the window — with an acceptance gate

⚠️ **Metric code.** This changes E3's transition rate; flag it at the top of the session report.

**Files:**
- Modify: `src/tabsampler/eval/playability.py`, `scripts/validate_playability.py`
- Test: `tests/eval/test_playability.py`

**Interfaces:**
- Consumes: `shift_window`, `HAND_WINDOW` (Task 3).
- Produces: `hand_move_is_playable(hand: int | None, positions: Sequence[Position], seconds: float, rules: PlayabilityRules) -> tuple[str | None, int | None]`
  (reason or None, new hand). **Removes** `transition_is_playable`.

- [ ] **Step 1: Write the failing tests** — replace the six transition tests in
`tests/eval/test_playability.py` with:

```python
from tabsampler.eval.playability import hand_move_is_playable


def test_a_reach_inside_one_position_is_not_a_hand_move() -> None:
    # 5 -> 7 in a sixteenth note used to read as 16 frets/s (ADR 0022). Now: no move.
    reason, hand = hand_move_is_playable(5, pos((0, 7)), 0.125, RULES)
    assert reason is None and hand == 5


def test_a_slow_long_jump_passes() -> None:
    reason, hand = hand_move_is_playable(2, pos((1, 12)), 2.0, RULES)
    assert reason is None and hand == 8


def test_the_same_jump_played_fast_fails() -> None:
    reason, _ = hand_move_is_playable(2, pos((1, 12)), 0.1, RULES)
    assert reason is not None and "frets/s" in reason


def test_a_jump_with_no_time_between_groups_fails() -> None:
    reason, _ = hand_move_is_playable(2, pos((1, 12)), 0.0, RULES)
    assert reason is not None


def test_a_jump_across_an_open_string_is_still_a_jump() -> None:
    # fret 2 -> open -> fret 20 in 0.1 s: the open string must not reset the hand.
    tab = [tabnote(0.00, 0, 2), tabnote(0.05, 1, 0), tabnote(0.10, 0, 20)]
    assert playability_rate(tab, RULES).transition_rate < 1.0


def test_a_shape_with_two_notes_on_one_string_still_moves_the_hand() -> None:
    reason, hand = hand_move_is_playable(5, pos((0, 3), (0, 12)), 5.0, RULES)
    assert hand == 3 and reason is None
```

- [ ] **Step 2: Run** `uv run pytest tests/eval/test_playability.py -q` — Expected: ImportError.

- [ ] **Step 3: Implement** in `playability.py`:

```python
def hand_move_is_playable(
    hand: int | None, positions: Sequence[Position], seconds: float, rules: PlayabilityRules
) -> tuple[str | None, int | None]:
    """Whether the hand can make the move this shape needs in ``seconds``, and where it ends.

    The hand is a window (ADR 0025): a finger reaching inside it is not a move, and an
    all-open shape leaves it where it was.
    """
    new_hand, distance = shift_window(hand, fretted_frets(positions), HAND_WINDOW)
    if distance == 0:
        return None, new_hand
    if seconds <= 0.0:
        return f"{distance}-fret jump with no time between groups", new_hand
    speed = distance / seconds
    if speed > rules.max_frets_per_second:
        return f"{speed:.1f} frets/s exceeds {rules.max_frets_per_second}", new_hand
    return None, new_hand
```

and in `playability_rate` replace the transition loop with one that carries `hand` through
the whole tab: `hand = shift_window(None, fretted_frets(shapes[0][1]))[0]` before the loop,
then `reason, hand = hand_move_is_playable(hand, positions, onset - prev_onset, active)`.
Delete `transition_is_playable` and the old `hand_position` helper if nothing else uses it
(`grep -rn hand_position src tests scripts`).

- [ ] **Step 4: Update `scripts/validate_playability.py`** to use `hand_move_is_playable`
with a carried hand, and delete its separate "carried" variant (the rule now carries).

- [ ] **Step 5: Run the acceptance check** (pre-registered: commit the script change first)

Run: `uv run python scripts/validate_playability.py data/dadagp/DadaGP-v1.1.zip data/dadagp/track_meta.json`
**Acceptance, fixed in ADR 0022: human tab passes the transition rule at ≥ 0.99.**
If it does not: stop, record the number as a results row, and report to Ege — do not tune
the window or the speed limit to pass.

---

### Task 5: Decoder, cost model, fitter and oracle on the window (ADR 0025)

**Files:**
- Modify: `src/tabsampler/fingering/costs.py`, `src/tabsampler/decode/viterbi.py`,
  `src/tabsampler/fingering/fit.py`, `tests/decode/brute_force.py`
- Test: `tests/fingering/test_costs.py`, `tests/decode/test_viterbi.py`,
  `tests/fingering/test_fit.py`, `tests/fixtures/golden_clip.txt`
- Create: `docs/adr/0025-hand-window.md`

**Interfaces:**
- Consumes: `shift_window`, `carry_hand` (Task 3).
- Produces: `HandSetScorer.transition_cost_from(previous_hand, curr)` = `move * shift_window(previous_hand, curr.fretted_frets)[1]`.

- [ ] **Step 1: Rewrite the oracle first, from the ADR text, not the code.** In
`tests/decode/brute_force.py::path_cost`, replace the hand loop with:

```python
    hand: int | None = None
    for state in path:
        fretted = sorted(p.fret for p in state.positions if p.fret > 0)
        if not fretted:
            continue  # open strings need no hand; it stays where it was
        low, high = fretted[0], fretted[-1]
        if hand is None:
            hand = low  # the first fretted shape places the hand, at no cost
            continue
        if hand <= low and high <= hand + 4:
            continue  # inside the 4-fret window: a finger reaches, the hand does not move
        new = low if low < hand else min(low, high - 4)
        total += scorer.weights.move * abs(new - hand)  # pyright: ignore[reportAttributeAccessIssue]
        hand = new
```

Keep `transition_cost_from` out of the oracle on purpose: it must check the scorer, not
reuse it. (The oracle's scorer is always `HandSetScorer`.)

- [ ] **Step 2: Update the cost tests** in `tests/fingering/test_costs.py`:
`test_movement_is_charged_across_an_intervening_open_chord` now expects
`s.transition_cost_from(2, state((0, 10))) == pytest.approx(4.0 * s.weights.move)` and
`s.carry(previous_hand=2, curr=state((0, 10))) == 6`, with a comment that the window starting
at 2 covers frets 2–6, so reaching fret 10 moves it 4. Add:

```python
def test_a_reach_inside_the_window_costs_nothing() -> None:
    s = scorer()
    assert s.transition_cost_from(5, state((0, 8))) == 0.0
```

- [ ] **Step 3: Run** `make oracle` — Expected: FAIL (decoder still on the old rule).

- [ ] **Step 4: Implement**
- `costs.py`: `transition_cost_from` returns
  `self.weights.move * shift_window(previous_hand, curr.fretted_frets)[1]`.
- `viterbi.py::build_lattice`: every state now takes one node per distinct carried hand:
  ```python
  for state in states:
      hands = dict.fromkeys(carry_hand(hand, state) for hand in carried)
      level.extend(LatticeNode(state, hand) for hand in hands)
  ```
  (`node_transition_cost` is unchanged: legality is still `carry_hand(prior, state) == node.carried_hand`.)
- `fit.py`: in `path_features`, `hand, moved = shift_window(hand, state.fretted_frets); total[0] += moved`;
  in `sequence_features`, `new, moved = shift_window(prior.carried_hand, node.state.fretted_frets)`,
  forbidden if `new != node.carried_hand`, else `move[i, j] = moved`.

- [ ] **Step 5: Run** `make oracle` and `uv run pytest tests/decode tests/fingering -q`
Expected: PASS. If the oracle disagrees, the decoder is wrong (ADR 0014).

- [ ] **Step 6: Replace the node-count guards** in `tests/decode/test_viterbi.py`
(`test_the_lattice_only_augments_all_open_states`, the 1.5× and 3.5× guards) after measuring:
run `uv run python scripts/measure_lattice.py --guitarset 60` (logged access, complexity only),
record the ratios in ADR 0025, and guard the pentatonic passage at `n_nodes <= 6 * n_states`
with the measured value in the comment.

- [ ] **Step 7: Benchmark** with `scripts/bench_decode.py` against `main`, as its docstring
shows. **Acceptance:** spec §4 still met — a 3-minute file decodes far inside 60 s. Record
`viterbi` and `decode` times in ADR 0025.

- [ ] **Step 8: Regenerate the golden clip** and diff it by eye: the phrase should no longer
pre-position the hand high (ADR 0018's worked example). Record what changed in ADR 0025.

- [ ] **Step 9: Write ADR 0025** (hand window, supersedes ADR 0018's lowest-fret hand
position and ADR 0011's transition rule), with Task 4's acceptance result, Step 6–8
measurements. Add it to the index. `make check`, then commit:

```bash
git add -A
git commit -m "Model the hand as a 4-fret window in E3, the cost model and the decoder"
```

---

### Task 6: Does the window help the default? (pre-registered)

**Files:** `experiments/results.csv`, `docs/adr/0025-hand-window.md` (results appendix
written before ADR 0025 is accepted).

- [ ] **Step 1: Pre-register** in the commit message of an empty commit:
`git commit --allow-empty -m "Pre-register: the hand window does not lower hand-set recovery on clean or distorted artist-disjoint validation parts"`.
**Rule:** the window stays only if hand-set recovery does not fall on *either* part versus
Task 2's baseline. If it falls, report and stop for Ege's call.

- [ ] **Step 2: Run** `uv run python scripts/fit_cost_weights.py ... --split artist` and
record hand-set recovery (all / clean / distorted) as results rows.

---

### Task 7: Calibrate the default's temperature (ADR 0026) and the fitted weights' fair test (ADR 0027)

**Files:** `configs/phase1_baseline.yaml`, `configs/fitted_dadagp.yaml`,
`docs/adr/0026-default-temperature.md`, `docs/adr/0027-fitted-weights-fair-test.md`,
`experiments/results.csv`.

- [ ] **Step 1: Pre-register both** by committing ADRs 0026 and 0027 as `proposed`:
- 0026 hypothesis: temperature scaling on artist-disjoint validation lowers the hand-set
  decoder's per-note calibration error; single variable T.
- 0027 **fair test, fixed now**: refit weights (window model, artist-disjoint training)
  replace the default iff, on artist-disjoint validation, they (a) recover more human
  fingerings than hand-set on clean parts, (b) recover more on distorted parts, and (c) the
  decoded output's E3 group rate is not below hand-set's. All three, or no change.

- [ ] **Step 2: Calibrate the default**:
`uv run python scripts/calibrate_temperature.py data/dadagp/DadaGP-v1.1.zip data/dadagp/track_meta.json --decoder-config configs/phase1_baseline.yaml --split artist`
Write the fitted T into `configs/phase1_baseline.yaml` with a comment citing ADR 0026; add a
results row.

- [ ] **Step 3: Refit** with `scripts/fit_cost_weights.py --split artist`, calibrate its T,
write both into `configs/fitted_dadagp.yaml`, and add `--e3` reporting of the decoded
validation output's group rate to the fit script (one line: `group_is_playable` over each
Viterbi path). Apply ADR 0027's rule exactly; accept both ADRs with their results.

---

### Task 8: One GuitarSet measurement of the default

⚠️ Reads the test set — once, logged, nothing chosen from it.

- [ ] **Step 1: Pre-register** `configs/m2_window_eval.yaml` (copy `configs/m2_fitted_eval.yaml`,
new hypothesis: the window model with calibrated temperature changes E2 and lowers E5 for the
default decoder; prediction written in the file). Commit.

- [ ] **Step 2: Run** `uv run tabsampler eval-m1 --config configs/m2_window_eval.yaml --decoder-config configs/phase1_baseline.yaml`
(or `configs/fitted_dadagp.yaml` if ADR 0027 adopted it — the default is whichever ADR 0027 left).

- [ ] **Step 3: Refresh the README results table in the same commit**, with the old row kept
for comparison and every guardrail of ADR 0016 stated, including any breach.

---

### Task 9: Close out

- [ ] Write `docs/devlog/2026-10-02.md` (done, measured, rulings, open questions); update
`HANDOFF.md` and the Phase 2 plan's C1/C2/C5 checkboxes; check that no local tooling files
are tracked.
- [ ] `make check`, `make oracle`, then `git checkout main && git merge --ff-only hand-window && git push origin main`.

# Tab Sampler: Specification & Implementation Plan

> Working name: **Tab Sampler**. An application that turns a guitar recording into guitar tablature.
> Status: pre-implementation. Architecture is settled. What's left is a set of experiments.

---

## 1. Goal

Given an audio file, produce a tab that is:

1. **Correct in pitch and timing.** The right notes at the right times.
2. **Correct in fingering where it can be known.** The string and fret the player actually used.
3. **Always playable.** Even when the fingering differs from the original, a guitarist can play it.
4. **Honest about uncertainty.** Notes and positions the system is unsure of are marked, not guessed silently.

"High accuracy" has no meaning until it is stated as a number on a named benchmark (see Decision D10).

### Non-goals for v1

- Real-time transcription (the app converts files offline).
- Full mixed songs (v1 handles isolated guitar only; full-song mode comes in Phase 6).
- Expressive techniques: bends, slides, hammer-ons and so on (Phase 7).
- Rhythmic notation with measures and note values (see D5).
- Bass, 7-string or 12-string guitars, or instruments other than guitar.

---

## 2. Architecture

The pipeline is modular. Each stage sits behind an interface, so you can swap one model without touching the others.

```text
audio file
   │
   ▼
[0] Preprocess ──────── load, resample, mono; (Phase 6: source separation)
   │
   ▼
[1] Transcriber ─────── audio → NoteEvents          (Basic Pitch → fine-tuned → MT3-style)
   │                     + AudioEvidence per note    (Phase 3+)
   ▼
[2] Candidate generator  NoteEvent → every legal (string, fret) for the chosen tuning
   │
   ▼
[3] Fingering scorer ── scores candidates           (hand-set costs → learned model)
   │
   ▼
[4] Decoder ─────────── global best sequence + per-note posteriors
   │                     (Viterbi + forward-backward → constrained beam search)
   ▼
[5] Renderer ────────── ASCII / JSON / MusicXML / Guitar Pro
```

### 2.1 Data contracts

Fix these contracts early. The main job of the architecture is keeping them stable.

```python
@dataclass(frozen=True)
class NoteEvent:
    onset: float          # seconds
    offset: float         # seconds
    pitch: int            # MIDI note number
    confidence: float     # 0..1, from the transcriber
    bend: list[float] | None = None   # pitch contour in semitones (Basic Pitch provides one)

@dataclass(frozen=True)
class Tuning:
    open_pitches: tuple[int, ...]     # standard: (40, 45, 50, 55, 59, 64)  low E → high E
    n_frets: int = 22
    capo: int = 0

@dataclass(frozen=True)
class Position:
    string: int           # 0 = low E
    fret: int

@dataclass(frozen=True)
class TabNote:
    note: NoteEvent
    position: Position
    posterior: float                          # confidence from forward-backward
    alternatives: list[tuple[Position, float]]  # other candidates with their probabilities

class Transcriber(Protocol):
    def transcribe(self, audio: np.ndarray, sr: int) -> list[NoteEvent]: ...

class FingeringScorer(Protocol):
    def emission_cost(self, group: NoteGroup, state: ChordState, ctx: Context) -> float: ...
    def transition_cost(self, prev: ChordState, curr: ChordState) -> float: ...

class Decoder(Protocol):
    def decode(self, groups: list[NoteGroup], scorer: FingeringScorer) -> list[TabNote]: ...
```

### 2.2 Decoder formulation (Phase 1)

- **Grouping.** Notes whose onsets fall within a window (e.g. 30 ms, a tunable setting) form one `NoteGroup`, i.e. a chord or a single note.
- **States.** For a group of *k* notes, a `ChordState` assigns each note to a *different* string. The fret must equal pitch minus the string's open pitch, and must lie between 0 and `n_frets`.
- **Hard constraints** (states that break them are pruned):
  - one note per string;
  - the span of the fretted notes is at most `max_span` (e.g. 4–5 frets; see D9).
- **Emission cost** (how comfortable a single state is): the span, how high up the neck it sits, and a reward or penalty for open strings. From Phase 3 on, it also includes the acoustic evidence term −log P(string | audio).
- **Transition cost** (how hard it is to move between consecutive states): how far the hand position moves, where hand position is e.g. the lowest fretted note. Optionally add a term for string crossings.
- **Total cost:**
  `C = Σ emission + λ_move·Σ movement + λ_span·Σ span + λ_high·Σ height + λ_ac·Σ acoustic`
- **Viterbi** returns the single lowest-cost sequence.
- **Forward-backward** runs on the potentials exp(−C/T) and gives each note's probability for each candidate position. Those probabilities become `posterior` and `alternatives`. T is a temperature to tune.
- **Known limitation in v1:** a note that is still ringing does not stop the next group from using its string. Document this; fix it later if the metrics show it matters.

### 2.3 Later stages (designed now, built later)

| Stage | Phase | Design sketch |
|---|---|---|
| Learned fingering model | 2 | Small encoder-decoder over note tokens that outputs string/fret tokens. Pretrained on DadaGP and SynthTab labels. Its probabilities replace the hand-set emission cost, so it still runs through the decoder. |
| Audio conditioning | 3 | A per-note audio embedding (a small CNN over the audio around each note), like TART's approach. Alternative: a TabCNN-style head that predicts string probabilities (D13). |
| Better transcriber | 5 | Fine-tune Basic Pitch, then try a heavier model. |
| Source separation | 6 | `htdemucs_6s` guitar stem. Experiment: feed the model both the stem and the original mix. |
| Techniques | 7 | A classifier head per note, or extra tokens in the tab model. Uses GOAT and Guitar-TECHS annotations. |

---

## 3. Evaluation specification

The evaluation harness is built **before** any model work (Phase 0). A number is only reported if the harness produced it.

### 3.1 Metrics

| ID | Metric | Question it answers | How |
|---|---|---|---|
| E1 | **Note F1** | Did we hear the right notes? | `mir_eval.transcription`, 50 ms onset tolerance, onset-only and onset+offset variants |
| E2 | **Exact Tab F1** | Did we recover the reference string and fret? | A hit needs onset (±50 ms), pitch *and* string to match |
| E3 | **Playability rate** | Is our output physically playable? | Share of groups and transitions passing the D9 rules, computed with **no** reference tab |
| E4 | **Pitch-validity rate** | Does every string/fret produce the pitch we claimed? | Should be 100%. Anything lower is a bug. |
| E5 | **Calibration** | When we say 0.8, are we right about 80% of the time? | Expected calibration error (ECE) of posteriors against exact-position correctness |
| E6 | Technique F1 | Did we detect bends, slides and so on? | Phase 7 only |
| E7 | Runtime | Is it usable? | Seconds of processing per minute of audio, on the M4 Mac |

### 3.2 Two evaluation modes (always report both)

- **Oracle mode:** feed the reference notes into stage 2. This measures fingering quality on its own.
- **End-to-end mode:** feed the transcriber's notes. This is what users actually get.

The gap between the two shows how much the transcriber costs you.

### 3.3 Datasets

| Dataset | Content | Role (proposed; D8) | Notes |
|---|---|---|---|
| GuitarSet | about 3 h acoustic guitar, 6 players, string-level annotations | **Held-out test set** | Load via `mirdata`. Also the reference benchmark in TART. |
| GOAT | 5.9 h DI electric guitar + Guitar Pro tabs + techniques, and 29.5 h amp-augmented | Fine-tuning (Phases 4, 7) | Electric, so closest to many real use cases |
| GAPS | about 14 h classical guitar, 200+ performers | Transcriber fine-tuning | Classical timbre, not representative of electric |
| Guitar-TECHS | 5+ h electric, varied players/gear/techniques | Fine-tuning; robustness; techniques | |
| EGDB | electric guitar with tab annotations | Second test set | Catches overfitting to acoustic guitar |
| SynthTab | audio synthesized from tabs | Audio-conditioning pretraining | Large scale, but the audio is synthetic |
| DadaGP | 26,181 Guitar Pro songs, symbolic only | Fingering-model pretraining | Access is **by request**, for research purposes |

**Rule:** the test split is never used for tuning. Cost weights and hyperparameters are tuned on a validation split taken from the *training* sources.

### 3.4 Reference points (not targets)

TART (arXiv 2609.11904), on GuitarSet, under its own protocol: **69.2%** Exact Tab F1 given reference notes, **56.0%** end to end.
Numbers from other papers are only comparable if you rerun them under your protocol.

---

## 4. Non-functional requirements

- **Reproducibility.** Every result is produced by `config file + git commit + seed`. Dependencies are pinned (`uv` lockfile).
- **Determinism.** Given the same input and config, Phases 0–1 produce byte-identical output.
- **Hardware.** Development and inference on the M4 MacBook Air (16 GB). Training on the RTX 4070 laptop (8 GB VRAM). Models must fit that budget, or you use cloud GPUs (D11).
- **Performance target (v1).** Transcribe a 3-minute isolated-guitar file in under 60 s on the M4. Measure it rather than assume it.
- **Licensing.** Check every dataset's and model's license before publishing weights or shipping (D15).

---

## 5. Repository layout (proposed)

```text
tab-sampler/
├── pyproject.toml / uv.lock
├── README.md                  # results table, demo, architecture diagram
├── configs/                   # one YAML file per experiment
├── docs/
│   ├── spec.md                # this file
│   ├── adr/                   # one Architecture Decision Record per decision (D1…)
│   └── devlog/                # one entry per work session or phase
├── src/tabsampler/
│   ├── types.py               # data contracts (§2.1)
│   ├── audio/                 # loading, resampling, separation
│   ├── transcribe/            # Transcriber implementations
│   ├── fingering/             # candidates, cost models, learned scorers
│   ├── decode/                # viterbi.py, forward_backward.py, beam.py
│   ├── render/                # ascii.py, json.py, musicxml.py, gp.py
│   ├── eval/                  # metrics.py, playability.py, calibration.py, harness.py
│   ├── data/                  # dataset loaders, splits, tokenization
│   └── cli.py
├── tests/                     # unit + property-based + brute-force oracle tests
├── experiments/results.csv    # one row per run: commit, config, metrics
└── scripts/                   # training / download helpers
```

---

## 6. Phased plan

Durations assume about 8–10 focused hours a week alongside university. Treat them as rough guesses and adjust after Phase 1.

### Phase 0: Foundations (1–2 weeks)
- Set up the repo, tooling (`uv`, `ruff`, `pyright`, `pytest`, pre-commit) and CI (GitHub Actions).
- Write `types.py` with the data contracts.
- Load GuitarSet through `mirdata`; fix the splits (D8).
- Evaluation harness for E1, E2 and E4. Unit tests using hand-made reference/prediction pairs.
- **Sanity check:** run Basic Pitch on GuitarSet and get a note F1 in a plausible range.
- **Done when:** `make eval-notes` prints note F1 for Basic Pitch on GuitarSet, and CI is green.

### Phase 1: Ground-zero baseline (2–3 weeks) → **Milestone M1**
- Candidate generation, note grouping, chord-state enumeration with pruning.
- Hand-set cost model; Viterbi; forward-backward.
- ASCII and JSON renderers with markers on uncertain notes; a CLI: `tabsampler transcribe in.wav -o out.txt`.
- Playability metric (E3) and calibration metric (E5).
- **Tests:** for short inputs, Viterbi must match brute-force search over all paths. Property test: output pitch always equals input pitch.
- Tune the cost weights on a validation split (grid or random search).
- **Done when:** results table v1 exists (oracle + end-to-end, E1–E5, E7). This is your **ground-zero number**.

### Phase 2: Learned fingering, symbolic only (4–6 weeks) → **Milestone M2**
- Request DadaGP access **early** (at the start of Phase 1).
- Design tokenization (D12); build the data pipeline; train a small encoder-decoder.
- Constrained decoding: only pitch-valid tokens can be generated.
- **Controlled comparison in oracle mode:** Viterbi vs. transformer vs. transformer probabilities decoded by Viterbi/beam search.
- **Done when:** the comparison table exists. A result where the transformer does *not* beat Viterbi is still a valid result; write it up.

### Phase 3: Audio conditioning (4–6 weeks) → **Milestone M3**
- Pick the form of audio evidence (D13); pretrain on SynthTab.
- Add the acoustic term to the scorer/decoder.
- **Done when:** you have a measured change in oracle-mode Exact Tab F1 against Phase 2, with an ablation that removes the audio input.

### Phase 4: Real-data fine-tuning (2–4 weeks)
- Fine-tune on GOAT, GAPS and Guitar-TECHS. Evaluate on GuitarSet and EGDB, which stay held out.
- **Done when:** results are reported on both test sets, including the gap between electric and acoustic guitar.

### Phase 5: Better transcriber (open-ended)
- Fine-tune Basic Pitch on guitar data, then try heavier models.
- **Done when:** end-to-end Tab F1 improves. Note F1 improving alone is not enough.

### Phase 6: Full-song mode
- `htdemucs_6s` guitar stem, with an experiment feeding stem + mix. Report results separately from isolated mode.

### Phase 7: Techniques
- Bends (the pitch-contour data is already in `NoteEvent.bend`), slides, hammer-ons/pull-offs, palm mutes, harmonics.

### Parallel track: Application (starting after M1)
- CLI → local web UI that shows the tab with uncertain notes highlighted and alternatives shown on hover (D14).
- Export to MusicXML and Guitar Pro (`PyGuitarPro` writes gp3–gp5).

### Milestones summary

| Milestone | What you can say afterwards |
|---|---|
| **M1** | "I built an end-to-end audio-to-tab system with a probabilistic decoder and a rigorous evaluation harness." |
| **M2** | "I trained a sequence model for fingering and ran a controlled comparison against a DP baseline." |
| **M3** | "I showed, with an ablation, whether audio evidence improves string assignment." |
| **App v1** | "Anyone can drop in a guitar recording and get playable tab with confidence marks." |

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| DadaGP access is delayed or refused | Request it early; SynthTab's labels come from the same kind of source; fall back to a smaller public Guitar Pro corpus |
| Chord-state explosion in the decoder | Span pruning, beam limits; measure the state counts |
| Overfitting to GuitarSet (acoustic, 6 players) | EGDB as a second test set; never tune on the test split |
| Transcriber errors dominate end-to-end results | Oracle mode keeps the fingering work measurable no matter what |
| Scope creep (UI, techniques, full songs) | The phase gates in §6: no phase starts until the previous "done when" is met |
| The 8 GB GPU limits model size | Small models first; cloud GPU only for specific experiments (D11) |

---

## 8. Open decisions: what you need to decide

Each decision lists the question, why it matters, the options, a **cue** (my default if you have no strong preference) and **when** you need to decide. Record each final choice as an ADR in `docs/adr/`.

### Product scope

**D1. What input does v1 accept?**
- Why it matters: it sets the difficulty, the datasets and how honestly you can evaluate.
- Options: (a) isolated guitar only; (b) isolated guitar + a best-effort full-song mode; (c) full songs from day one.
- Cue: **(a)**. Full songs are Phase 6.
- When: before Phase 0.

**D2. Which guitar sound is the primary target?**
- Why it matters: GuitarSet is acoustic, GOAT is electric DI, GAPS is classical. Distortion changes everything.
- Options: acoustic / clean electric / distorted electric / all.
- Cue: **acoustic + clean electric** for v1, measured separately; distorted later.
- Ask yourself: what will *you* actually feed it? Use your own recordings as a private test set.
- When: before Phase 4 (it drives which data you fine-tune on).

**D3. Which tunings?**
- Options: standard only / a user-chosen tuning and capo / automatic tuning detection.
- Cue: **user-chosen tuning via the `Tuning` type from day one** (it's cheap). Leave automatic detection out of scope.
- When: before Phase 1 (it's baked into candidate generation).

**D4. Output formats**
- Options: ASCII, JSON, MIDI, MusicXML, Guitar Pro.
- Cue: **ASCII + JSON in Phase 1**, MusicXML/Guitar Pro in the app track.
- When: Phase 1.

**D5. Timing-only tab, or rhythmic notation?** *(the biggest hidden scope item)*
- Why it matters: real tabs have measures, beats and note values. That needs beat tracking and quantization, which is a whole separate problem.
- Options: (a) time-positioned tab (spacing proportional to time); (b) quantize to a beat grid from a beat tracker such as `beat_this`; (c) full notation with durations.
- Cue: **(a) for v1**; (b) as its own mini-project after M3.
- When: before Phase 1 renderer work.

### Algorithm design

**D6. What is a decoder state: a single note or a chord?**
- Options: (a) note-level states, simple but can't enforce one-note-per-string within a chord; (b) chord-level states with pruning.
- Cue: **(b)**. Chords are the whole difficulty of guitar.
- When: Phase 1.

**D7. How are the cost weights set?**
- Options: hand-set / grid search on validation / learned (e.g. maximizing likelihood of reference fingerings, which makes this a CRF).
- Cue: **hand-set → grid search in Phase 1**. Learning the weights is a nice Phase 2 bridge experiment.
- When: Phase 1.

**D8. Evaluation protocol**
- Options: (a) GuitarSet held out entirely as a test set, like TART; (b) 6-fold cross-validation by player, like TabCNN; (c) both.
- Cue: **(a) as the primary protocol**, fine-tuning only on other datasets; **(b)** only if you ever train on GuitarSet.
- When: **Phase 0. Lock it before the first number exists.**

**D9. What counts as "playable"?**
- Define the concrete rules: maximum span of fretted notes (4? 5? depends on the neck position), maximum hand jump between groups within Δt seconds, one note per string, and any open-string rules.
- Cue: start with **span ≤ 4 frets below fret 12 and ≤ 5 at fret 12 and above; jumps limited by a speed rule (frets per second)**. Check the rules on real tabs from DadaGP: the share of real tab that passes your metric should be close to 100%.
- When: Phase 1.

**D10. What is the headline metric, and what counts as "high accuracy"?**
- Options: Exact Tab F1 (end-to-end) / note F1 + playability / a weighted mix.
- Cue: headline = **end-to-end Exact Tab F1 on GuitarSet under your protocol**, always reported alongside playability and note F1. Set the target *after* M1, as "baseline + X points".
- When: define the metric in Phase 0, the target after M1.

### Machine learning

**D11. Framework and compute**
- Options: PyTorch (CUDA on the 4070, MPS on the Mac) / JAX / MLX (Mac only).
- Cue: **PyTorch** (the ecosystem, Basic Pitch ports and TART-style code all use it). Train on the 4070; export to ONNX for Mac inference. Use cloud GPUs only for runs that don't fit.
- When: before Phase 2.

**D12. Fingering model design**
- Options: a small transformer trained from scratch / fine-tuned T5-small / BiLSTM-CRF.
- Tokenization options: per note `(pitch, time-shift) → (string, fret)` / per chord group.
- Cue: **small encoder-decoder from scratch first** (you'll learn more, and it's small enough for 8 GB). T5-small as a comparison.
- When: start of Phase 2.

**D13. What form does the audio evidence take?**
- Options: (a) a per-note CNN embedding, as in TART; (b) a TabCNN-style head predicting per-frame string probabilities; (c) cross-attention from the tab model to audio frames.
- Cue: **(b) first**. It plugs straight into the Viterbi acoustic term and is easy to inspect. Then (a).
- When: start of Phase 3.

### Engineering & product

**D14. App surface**
- Options: CLI only / local web UI (FastAPI + a small front end) / desktop app / DAW plugin.
- Cue: **CLI → local web UI**. Consider a DAW plugin only if you later port the inference core to C++.
- When: after M1.

**D15. Licensing and publication**
- Question: will you publish weights, a demo or a paper? DadaGP is research-only by request; check the terms of GOAT, SynthTab, GAPS and so on before shipping anything trained on them.
- Cue: publish the code under MIT/Apache; decide weight release per dataset.
- When: before any public release.

**D16. Should any part be written in C/C++?**
- Why it matters: performance is not a reason (offline Python is fast enough). *Learning* could be.
- Cue: an optional stretch goal after M1: port the decoder to C++ with pybind11 and benchmark it against Python and Numba.
- When: after M1, optional.

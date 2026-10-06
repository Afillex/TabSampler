# Using Tab Sampler

Tab Sampler turns a recording of one guitar into tablature: which string and fret each note
was most likely played on, with the notes it is least sure of marked.

## Install and run

```bash
uv sync --all-groups                 # the package and the web page (Python 3.13)
bash scripts/setup_transcriber.sh    # the transcriber, Basic Pitch, in its own Python 3.11
make serve                           # then open http://127.0.0.1:8000
```

The page and the command line give the same tab for the same file:

```bash
uv run tabsampler transcribe take.wav                 # ASCII tab in the terminal
uv run tabsampler transcribe take.wav -o tab.json     # every note, its time and its alternatives
uv run tabsampler transcribe take.wav -o tab.musicxml # for MuseScore
uv run tabsampler transcribe take.wav -o tab.gp5      # for Guitar Pro or TuxGuitar
```

The server listens on `127.0.0.1` only. It has no login and runs the transcriber on every
upload, so `--host` with any other address is a deliberate choice, and it warns.

Transcriptions are cached under `cache/note_events/`, keyed by the audio's bytes and the
transcriber's settings, so the same file a second time comes back in well under a second.

## Clean electric guitar

Choose **Clean electric** on the page, or pass `--electric` to `tabsampler transcribe`, and the
decoder also listens to each note to judge which string it was played on (ADR 0063). On EGDB's
clean electric recordings this raised the end-to-end score from 0.4547 to 0.4992. It needs the
`model` dependency group (PyTorch) and the trained classifier at `cache/acoustic/electric/best.pt`;
without them the option is greyed out on the page and `tabsampler serve` says why. It was trained
and tested on electric guitar only, so leave it off for acoustic recordings.

## What the page shows

- **Six string lines**, high e on top, as tab is written. Notes are placed by when they start,
  in proportion to time.
- **Uncertain notes** are in parentheses with a dashed outline: the decoder's posterior for them
  is below 0.6 (`uncertainty_threshold` in `configs/decoder_clean.yaml`). Hover over a note, or
  move to it with Tab, for its pitch, its time, its certainty rank in the take, and the other
  ways to play it, likeliest first.
- **A banner** when the transcriber heard notes the tab does not contain — outside the guitar's
  range, or dropped to make an over-full chord playable.
- **Downloads**: MusicXML and Guitar Pro 5.

## What it cannot do

- **Isolated guitar only** (ADR 0002). One guitar, no band, no vocals; up to 5 minutes and
  100 MB per file.
- **No rhythm** (ADR 0009). There are no bars or note values in the tab. The exports need them,
  so they place notes on a fixed 120 BPM grid of 1/128 notes and say inside the file that the
  rhythm is not transcribed (ADRs 0017, 0059). A note held under the next one is cut where the
  next starts. Bar lines are arbitrary.
- **The confidences are uncalibrated.** End to end on GuitarSet's test players the calibration
  error is 0.1841 (ADR 0057), so the page shows a ranking, never a percentage.
- **One tuning per run.** The tuning and capo come from the decoder config's `tuning` block
  (`--config`, default `configs/decoder_clean.yaml`: standard, no capo); the page uses the config
  `tabsampler serve` was started with. Every figure below is for standard tuning.
- **Standard pitch (A440).** The transcriber assumes it. A guitar tuned between semitones loses
  notes or has them written a semitone off; when a recording is at least a quarter of a semitone
  off, the page and the CLI say so. Automatic correction was measured and did more harm than good
  (ADR 0060), so tune to A440 and record again.
- **No techniques**: bends, slides, hammer-ons, palm mutes and harmonics are not transcribed.

## What to expect

Measured, end to end, on EGDB's 240 clips of clean electric guitar, which neither the
transcriber nor the decoder was tuned on (ADR 0057): **E2 0.4547** (0.4992 with the electric
option, ADR 0063), the F1 of notes whose
start (within 50 ms), pitch and string all match the player's. Given the true notes instead of
transcribed ones, the decoder alone scores 0.6762 (ADR 0054). So transcription errors — missed
and extra notes — cost 0.22, and strings chosen differently from the player's cost 0.32.

On GuitarSet's acoustic test players the end-to-end figure is 0.4553, labelled because Basic
Pitch was trained on most of GuitarSet (ADR 0055).

Read the tab as a first draft: check the marked notes by ear, and expect missed and extra
notes, which come from the transcriber.

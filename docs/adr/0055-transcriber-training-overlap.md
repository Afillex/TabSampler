# ADR 0055: Basic Pitch was trained on GuitarSet — how transcribers are compared from now on

Status: accepted (2026-10-06)

## Context

Every end-to-end figure in this project feeds Basic Pitch's notes to the decoder (ADR 0001). Its
paper (Bittner et al., ICASSP 2022, arXiv 2203.09893, Table 1) lists **GuitarSet: 648 audio files
for training, 72 for testing** — 90% — with "a random 5% of tracks from the training set ... used
for validation". The installed package says the same in its own words: `basic_pitch/data/README.md`
calls its dataset scripts "a selection of the datasets used to train the original model", and
`data/datasets/guitarset.py` splits GuitarSet at random by track (80/10/10 by default, unseeded).
Which tracks trained the released model is not published. Checked 2026-10-06 against the paper
and the installed basic-pitch 0.4.0.

So **Basic Pitch has most likely heard most of GuitarSet's recordings, our test players' and player
00's included**. Oracle mode is untouched — it never calls the transcriber. But every E1 and
end-to-end figure on GuitarSet, from Phase 0's 0.7437 to ADR 0045's 0.4418, is partly in-sample
for the transcriber, and the cost of transcription on acoustic recordings it has not heard is
likely larger than the 0.2402 measured. EGDB (2022) and Guitar-TECHS (2025) postdate it.

## Decision

- **GuitarSet's end-to-end figures stand as measured and are labelled** wherever quoted: the
  transcriber trained on most of those recordings.
- **Transcribers are compared, and transcriber settings chosen, only on audio no candidate trained
  on**: Guitar-TECHS's player 3 for validation (ADR 0051), EGDB for the test (ADR 0050). GuitarSet
  is reported beside, labelled, and decides nothing between transcribers. Player 00 chooses
  nothing about a transcriber.
- **A candidate states its training data.** One trained on GuitarSet or EGDB is not judged on it.
- **Against Guitar-TECHS's labels, each take's measured label delay (ADR 0052) is subtracted from
  the reference onsets when a transcriber is scored**, since a transcriber's onsets are the
  audio's; the labels themselves are not edited.

## Consequences

**Easier.** Phase 5's comparisons cannot be won by having memorised the test set.

**Harder.** The validation player is small (12 takes) and its labels noisy (ADR 0051): wrong labels
count against every transcriber alike, so comparisons hold, but intervals are wide. The README's
headline end-to-end figure is now the labelled one; EGDB's is the clean one.

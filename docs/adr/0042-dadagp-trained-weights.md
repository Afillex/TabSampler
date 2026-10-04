# ADR 0042: Weights trained on DadaGP stay unpublished (D15)

Status: accepted (2026-10-04) — Ege's decision

Completes ADR 0020 for DadaGP, the corpus the learned model of Phase 2 task C4 trains on
(ADR 0041).

## Context

ADR 0020 decided weights per training corpus, published nothing by default, and said that if
a corpus's terms are unclear the weights stay unpublished. DadaGP is research-use, by request
(spec §3.3, ADR 0021); its archive carries no licence file. A neural model trained on it is a
derivative of the corpus in a way a measured number is not: it can encode the training tabs
themselves.

The project has also published a few numbers fitted on DadaGP since ADR 0023: four cost
weights (`configs/fitted_dadagp.yaml`, `configs/fitted_clean_dadagp.yaml`), temperatures, and
now one weight in the default decoder (ADR 0039).

## Decision

- **A learned model's weights, when trained on DadaGP, are not published.** They live under
  `cache/` (gitignored), are never committed and never attached to a release. The code that
  trains them is MIT and public, so anyone with DadaGP access can reproduce them.
- **Its measured results are reported**, like any other (ADR 0020).
- **The few cost weights fitted on DadaGP stay public**, as since ADR 0023: a handful of
  scalars, each stated in an ADR and a config, saying how much a hand movement or an open
  string costs. A few numbers cannot carry the corpus's tabs.
- **So the public app ships what can be published**: the cost-model decoder. A learned model
  trained on DadaGP stays a research result unless a corpus whose terms allow it trains one.

## Alternatives considered

- **Publish the weights as research use.** Optimistic about terms nobody has read in that
  light; ADR 0020 rules it out.
- **Ask DadaGP's authors.** Possible later, if a learned model earns a place in the app; not
  needed to train and measure one.
- **Withdraw the fitted cost weights too.** They are results in all but name, and already
  part of every published figure.

## Consequences

**Easier.** Task C4 can train and report freely.

**Harder.** If the learned model wins, the app cannot ship it as trained.

# ADR 0064: The electric string classifier's weights are published as a release, under CC BY 4.0

Status: accepted (2026-10-06) — D15 for these weights, decided by Ege

Settles D15 (ADR 0020 left weights per training corpus) for one model: the classifier of ADR
0063 (`cache/acoustic/electric/best.pt`, 47,334 parameters, 193 KB, a PyTorch state dict only).

## Context

The app's electric option (ADR 0063) needs these weights, which live under `cache/`, so nobody who
clones the repository can use it without retraining. Every corpus they were trained on allows
redistribution of derived work with attribution: Guitar-TECHS (CC BY 4.0, ADR 0051) and EGFxSet
(CC BY 4.0, ADR 0062). IDMT-SMT-Guitar (CC BY-NC-ND) only chose the temperature and weight, which
are two numbers in a config, and EGSet12 (CC BY 4.0) only judged. Neither SynthTab nor DadaGP is
involved (ADRs 0042, 0046).

## Decision

1. **Published as an asset of a GitHub release**, tag `electric-model-v1`, named
   `tabsampler-electric-strings-v1.pt`, with a notice crediting Guitar-TECHS and EGFxSet.
2. **Licensed CC BY 4.0**, as the data it learnt from, separately from the MIT code.
3. **Pinned by SHA-256** in the repository (`scripts/get_electric_model.sh`):
   `0d46fc2986969a0cd342d54a90cf48091f3878845d9779a087cf37f6bb69f759`. `make electric-model`
   downloads it to `cache/acoustic/electric/best.pt` and refuses a file that does not match.
4. Not in git: a binary in history cannot be taken back, and a release can.

## Consequences

Anyone can use the electric option with `make electric-model`. A retrained model gets a new
release and a new hash; the old one stays reproducible. Weights trained on GOAT (CC BY-NC) would
need their own decision.

.PHONY: install test oracle lint format typecheck check eval-notes eval-m1 check-split

install:
	uv sync --all-groups

test:
	uv run pytest

oracle:
	uv run pytest -m oracle -v

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run pyright

check: lint typecheck test

# Phase 0 gate: Basic Pitch note F1 on GuitarSet.
# Needs the dataset (scripts/download_guitarset.py) and the transcriber
# (scripts/setup_transcriber.sh).
eval-notes:
	uv run tabsampler eval-notes --config configs/phase0_eval_notes.yaml

check-split:
	uv run tabsampler check-split

# M1 gate: E1-E5 and E7 on GuitarSet in oracle and end-to-end mode.
eval-m1:
	uv run tabsampler eval-m1 --config configs/m1_full_eval.yaml --decoder-config configs/phase1_baseline.yaml

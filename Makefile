.PHONY: install test oracle lint format typecheck check eval-notes eval-m1 check-split serve

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

# M1 gate: E1-E5 and E7 on GuitarSet in oracle and end-to-end mode -- since ADR 0037
# on the test players 01-05, so not on the 360 tracks M1 itself was measured on.
eval-m1:
	uv run tabsampler eval-m1 --split test --config configs/m1_full_eval.yaml --decoder-config configs/phase1_baseline.yaml

# The local web page (ADR 0058) at http://127.0.0.1:8000. Needs the `web` group and the
# transcriber (scripts/setup_transcriber.sh); `tabsampler serve` says so if it is missing.
serve:
	@uv run python -c "import fastapi, uvicorn" 2>/dev/null || { \
		echo "the web page needs the web dependency group: run uv sync --all-groups"; exit 1; }
	uv run tabsampler serve

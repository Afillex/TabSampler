#!/usr/bin/env bash
# The environment that fine-tunes Basic Pitch (Phase 5, Task 4; ADR 0056).
#
#     bash scripts/setup_transcriber_training.sh
#
# A Python 3.11 virtual environment of its own, beside -- not inside -- the one
# scripts/setup_transcriber.sh makes for inference: installing TensorFlow there would re-resolve
# librosa and coremltools under the cached transcriptions, which the cache key cannot see. This
# one holds basic-pitch 0.4.0 with its TensorFlow extra, trains, and converts the result to
# CoreML, which the untouched inference environment runs through --model-path.
set -euo pipefail

ENV_DIR="${TABSAMPLER_BP_TRAIN:-$HOME/.local/share/tabsampler/bp-train}"
BASIC_PITCH_VERSION="0.4.0"

echo "Creating ${ENV_DIR} (Python 3.11)..."
uv venv --python 3.11 "${ENV_DIR}"
VIRTUAL_ENV="${ENV_DIR}" uv pip install \
    "basic-pitch[tf]==${BASIC_PITCH_VERSION}" \
    "setuptools<81"

echo
echo "Verifying TensorFlow and Basic Pitch's model import..."
"${ENV_DIR}/bin/python" -c "
import tensorflow as tf, coremltools as ct
from basic_pitch import models
print('tensorflow', tf.__version__, '| coremltools', ct.__version__,
      '| parameters', models.model().count_params())
"
echo "OK. Training environment ready: ${ENV_DIR}/bin/python"

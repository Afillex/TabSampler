#!/usr/bin/env bash
# Install Basic Pitch into its own Python 3.11 environment (ADR 0001).
#
# WHY A SEPARATE ENVIRONMENT
#   basic-pitch 0.4.0 declares
#       tensorflow-macos<2.15.1; platform_system == "Darwin" and python_version > "3.11"
#   and tensorflow-macos has no wheel past cp311. On macOS arm64 it therefore installs
#   only on Python <= 3.11, where it needs no TensorFlow at all and uses CoreML. Rather
#   than drag the whole project back to 3.11 and numpy 1.x, the transcriber is
#   quarantined here and driven as a subprocess.
#
# WHY setuptools<81
#   basic-pitch pins resampy<0.4.3, and resampy 0.4.2 imports pkg_resources. setuptools
#   removed pkg_resources in v81 (gone entirely by 84), so without this pin the CLI dies
#   at import with ModuleNotFoundError: No module named 'pkg_resources'. Verified on
#   2026-09-27: setuptools 84.0.0 fails, 80.10.2 works.
#
# This environment is deliberately NOT in uv.lock. It is a tool, not a dependency.

set -euo pipefail

BASIC_PITCH_VERSION="0.4.0"
SETUPTOOLS_CONSTRAINT="setuptools<81"

echo "Installing basic-pitch ${BASIC_PITCH_VERSION} on Python 3.11..."
uv tool install "basic-pitch==${BASIC_PITCH_VERSION}" \
    --python 3.11 \
    --with "${SETUPTOOLS_CONSTRAINT}" \
    --force

EXE="$(command -v basic-pitch || echo "$HOME/.local/bin/basic-pitch")"

if [ ! -x "$EXE" ]; then
    echo "ERROR: basic-pitch was installed but no executable found at $EXE" >&2
    exit 1
fi

echo
echo "Verifying the CLI actually starts (this is where the pkg_resources failure shows)..."
if ! "$EXE" --help >/dev/null 2>&1; then
    echo "ERROR: $EXE exists but fails to run. Full output:" >&2
    "$EXE" --help || true
    exit 1
fi

echo
echo "OK. Transcriber ready."
echo "  executable : $EXE"
echo
echo "Note: 'uv tool install' puts this in ~/.local/bin, which is NOT on PATH by"
echo "default. Either add it to PATH, or point the transcriber at it explicitly:"
echo "    transcriber.exe = \"$EXE\""
echo "in configs/*.yaml (key: transcriber.exe)."

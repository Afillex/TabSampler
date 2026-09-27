"""Smoke test: gives CI something real to run from the very first commit."""

import sys


def test_python_version_is_pinned() -> None:
    # ADR 0001: the core is pinned to 3.13 because basic-pitch cannot be installed
    # alongside it on macOS arm64; the transcriber is isolated on 3.11 instead.
    assert sys.version_info[:2] == (3, 13)

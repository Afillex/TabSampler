"""Typed exceptions shared across stages.

Each exists so that a failure surfaces as itself rather than as a plausible-looking
number. The transcriber ones matter most: an empty note list scores as note F1 = 0.0,
which is indistinguishable from a genuine result.
"""


class TabSamplerError(Exception):
    """Base class for every error this package raises deliberately."""


class ReferenceInconsistencyError(TabSamplerError):
    """A reference annotation cannot be represented on the fretboard.

    Raised rather than dropping the note, because silently dropping unplaceable
    reference notes inflates precision.
    """


class TestSetMisuseError(TabSamplerError):
    """Something tried to tune, select or sweep against the test set (ADR 0003)."""

    # The name earns its "Test" prefix, but pytest would otherwise try to collect this
    # as a test class in every module that imports it.
    __test__ = False


class TranscriberUnavailableError(TabSamplerError):
    """The transcriber executable could not be found or run at all."""


class TranscriberFailedError(TabSamplerError):
    """The transcriber ran but did not produce usable output."""


class UnfingerableGroupError(TabSamplerError):
    """A note group has no legal chord state under the current tuning and constraints."""

"""Typed exceptions shared across the navigation core.

Contract violations must not be expressed as bare ``assert`` statements:
asserts are stripped under ``python -O``, and inside the orchestrator's
broad plugin sandboxes an ``AssertionError`` would be swallowed as an
ordinary technique failure.  Raising :class:`NavContractError` instead
keeps the check active in optimized runs and lets the orchestrator treat
the violation distinctly (error-level log plus a failed ``NavResult``
with ``NavStatusReason.CONTRACT_VIOLATION``).

:class:`NavPointingError` serves the mirror-image purpose: it names the
failures the corrected-attitude computation can legitimately hit, so the
one caller that must survive them absorbs exactly those and everything
else -- a programming defect -- keeps propagating.
"""

__all__ = ['NavContractError', 'NavPointingError']


class NavContractError(Exception):
    """An internal navigation invariant (contract) was violated.

    Raised when an upstream component hands core code a value outside its
    documented bounds (for example a 3-DoF technique result whose rotation
    exceeds the ensemble's small-angle bound).  A ``NavContractError``
    always indicates a programming error, never bad image data, so it is
    never swallowed by the orchestrator's plugin sandboxes: the sandboxes
    log it at error level and re-raise, and ``NavOrchestrator.navigate``
    converts it into a failed ``NavResult`` with
    ``NavStatusReason.CONTRACT_VIOLATION``.
    """


class NavPointingError(Exception):
    """A navigated image's corrected attitude could not be computed or applied.

    Raised for the failures the attitude computation and the attitude reader
    expect: an attitude the furnished kernels cannot supply, a frame or
    spacecraft clock that does not resolve, a spacecraft clock that is not
    the mission's, the guards refusing a matrix that is not a proper
    rotation, and a recorded attitude that fails the reader's gates.  It is
    the only exception ``NavOrchestrator.with_pointing`` and the metadata
    readers absorb, so a pointing the environment cannot supply or apply is
    reported and degraded while a programming defect propagates and fails
    the run on its first image.

    Parameters:
        message: The human-readable account of what failed.
        reason: A short machine-readable classification of the failure, or
            None when the caller has no per-reason accounting to feed.  The
            reader's gates set it so a run can tally degradations per
            reason.
    """

    def __init__(self, message: str, *, reason: str | None = None) -> None:
        """Build the exception, carrying the optional per-reason classification."""
        super().__init__(message)
        self.reason = reason

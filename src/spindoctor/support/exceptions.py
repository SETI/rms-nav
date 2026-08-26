"""Typed exceptions shared across the navigation core.

Contract violations must not be expressed as bare ``assert`` statements:
asserts are stripped under ``python -O``.  Raising :class:`NavContractError`
instead keeps the check active in optimized runs and lets the orchestrator
treat the violation distinctly (error-level log plus a failed ``NavResult``
with ``NavStatusReason.CONTRACT_VIOLATION``).

:class:`NavInternalError` is what every other exception out of a NavModel or
a NavTechnique becomes.  The orchestrator wraps each plugin call, and the
wrapper's job is to name the component and fail the image rather than to
absorb the failure: an exception nothing planned for means the image was not
navigated as designed, and an offset computed from whatever else survived is
an answer with no way to tell it apart from a whole one.

:class:`NavPointingError` serves the mirror-image purpose: it names the
failures the corrected-attitude computation can legitimately hit, so the
one caller that must survive them absorbs exactly those and everything
else -- a programming defect -- keeps propagating.
"""

__all__ = ['NavContractError', 'NavInternalError', 'NavPointingError']


class NavContractError(Exception):
    """An internal navigation invariant (contract) was violated.

    Raised when an upstream component hands core code a value outside its
    documented bounds (for example a 3-DoF technique result whose rotation
    exceeds the ensemble's small-angle bound).  A ``NavContractError``
    always indicates a programming error, never bad image data, so the
    orchestrator's plugin wrappers log it at error level and re-raise it
    unwrapped, and ``NavOrchestrator.navigate`` converts it into a failed
    ``NavResult`` with ``NavStatusReason.CONTRACT_VIOLATION``.  Keeping it
    distinct from :class:`NavInternalError` is what separates "upstream code
    handed us a value outside its documented bounds" from "a plugin raised
    something nobody anticipated".
    """


class NavInternalError(Exception):
    """A NavModel or NavTechnique raised an exception nothing anticipated.

    The orchestrator wraps every plugin call -- ``create_model``,
    ``to_features``, ``to_annotations`` and ``navigate`` -- and converts any
    exception other than :class:`NavContractError` into one of these.
    ``NavOrchestrator.navigate`` then converts it into a failed ``NavResult``
    with ``NavStatusReason.INTERNAL_ERROR``, carrying ``component`` and
    ``exception_type`` into the metadata document so the failure is visible
    to every consumer that reads the document rather than only to a human
    reading the log.

    Failing the image is not the same as raising through to the caller: the
    batch drivers see an ordinary failed frame and continue to the next one,
    exactly as they do for any other failure.  What no longer happens is an
    image reporting ``success`` on an offset computed from whatever evidence
    survived the exception.

    ``prepare`` is the exception to that, and deliberately: it has no
    ``NavResult`` to fail, so this propagates to its caller as
    :class:`NavContractError` already does.

    Parameters:
        component: What raised, as ``ClassName.method`` -- for example
            ``NavModelRings.create_model``.  Named rather than derived so
            the wrapper reports the plugin's registered name and not
            whatever the traceback's innermost frame happens to be.
        cause: The exception the plugin raised.  Only its class name is
            kept; the traceback reaches the log through ``raise ... from``.
    """

    def __init__(self, component: str, cause: BaseException) -> None:
        """Build the exception from the component that raised and what it raised."""
        self.component = component
        self.exception_type = type(cause).__name__
        super().__init__(f'{component} raised {self.exception_type}: {cause}')


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

"""Typed exceptions shared across the navigation core.

Contract violations must not be expressed as bare ``assert`` statements:
asserts are stripped under ``python -O``, and inside the orchestrator's
broad plugin sandboxes an ``AssertionError`` would be swallowed as an
ordinary technique failure.  Raising :class:`NavContractError` instead
keeps the check active in optimized runs and lets the orchestrator treat
the violation distinctly (error-level log plus a failed ``NavResult``
with ``NavStatusReason.CONTRACT_VIOLATION``).
"""

__all__ = ['NavContractError']


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

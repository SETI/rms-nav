"""NavFeasibilityReport — what a NavTechnique reports about its applicability.

Each technique's ``is_feasible(features)`` returns one of these reports.  The
orchestrator consults the report before invoking ``navigate``; infeasible
techniques are skipped silently with their reason recorded for diagnostics.
"""

from dataclasses import dataclass

__all__ = ['NavFeasibilityReport']


@dataclass(frozen=True)
class NavFeasibilityReport:
    """Outcome of a technique's feasibility check.

    Parameters:
        feasible: True if the technique can run on the supplied feature set.
        reason: Human-readable text; required when ``feasible`` is False
            and ignored when True.  Stable wording so the orchestrator can
            correlate similar refusals across images.
        consumed_feature_count: Number of features the technique *would*
            consume if invoked (after its own type filter).  Used by
            diagnostics; safe to set to 0 when feasible is False.

    Raises:
        TypeError: if any field has the wrong type.
        ValueError: if ``consumed_feature_count`` is negative or
            ``feasible=False`` is supplied with an empty ``reason``.
    """

    feasible: bool
    reason: str
    consumed_feature_count: int = 0

    def __post_init__(self) -> None:
        """Validate types and invariants on every field."""
        if not isinstance(self.feasible, bool):
            raise TypeError(
                f'NavFeasibilityReport.feasible must be bool; got {type(self.feasible).__name__}'
            )
        if not isinstance(self.reason, str):
            raise TypeError(
                f'NavFeasibilityReport.reason must be str; got {type(self.reason).__name__}'
            )
        if not isinstance(self.consumed_feature_count, int) or isinstance(
            self.consumed_feature_count, bool
        ):
            raise TypeError(
                'NavFeasibilityReport.consumed_feature_count must be int; got '
                f'{type(self.consumed_feature_count).__name__}'
            )
        if self.consumed_feature_count < 0:
            raise ValueError(
                'NavFeasibilityReport.consumed_feature_count must be >= 0; '
                f'got {self.consumed_feature_count!r}'
            )
        if not self.feasible and not self.reason:
            raise ValueError('NavFeasibilityReport.reason must be non-empty when feasible is False')

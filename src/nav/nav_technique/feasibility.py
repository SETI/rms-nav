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
    """

    feasible: bool
    reason: str
    consumed_feature_count: int = 0

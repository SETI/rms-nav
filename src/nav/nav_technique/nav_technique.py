"""NavTechnique base class for the autonomous-navigation pipeline.

Concrete subclasses self-register via ``__init_subclass__`` and the
orchestrator iterates the registry.  Every concrete technique is safe to
instantiate per-image without depending on prior runs;
``__init_subclass__`` only records a class reference — no instances are
constructed at import time.
"""

from __future__ import annotations

import fnmatch
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from nav.feature.feature import NavFeature
from nav.feature.feature_type import NavFeatureType
from nav.nav_technique.feasibility import NavFeasibilityReport
from nav.nav_technique.technique_result import NavTechniqueResult
from nav.support.nav_base import NavBase

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from nav.nav_orchestrator.nav_context import NavContext

__all__ = [
    'NavTechnique',
    'filter_technique_names',
]


class NavTechnique(NavBase, ABC):
    """Abstract base for autonomous-navigation techniques.

    Concrete subclasses self-register via ``__init_subclass__``.  The
    orchestrator iterates ``NavTechnique._registry`` to discover the
    techniques it should run.  Subclasses must override the class
    attributes ``name``, ``accepts_feature_types``, and (when relevant)
    ``requires_prior``.
    """

    _registry: ClassVar[list[type[NavTechnique]]] = []
    _abstract: ClassVar[bool] = True

    #: Human-readable technique name; used as the registry key.
    name: ClassVar[str] = ''
    #: Frozen set of feature types this technique consumes.  The
    #: orchestrator skips invocation when no input feature has a
    #: matching type.
    accepts_feature_types: ClassVar[frozenset[NavFeatureType]] = frozenset()
    #: If ``True``, this technique requires a prior offset on
    #: ``NavContext`` and is run only on pass 2.
    requires_prior: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-register concrete subclasses."""
        super().__init_subclass__(**kwargs)
        if not cls.__dict__.get('_abstract', False):
            NavTechnique._registry.append(cls)

    @abstractmethod
    def is_feasible(self, features: list[NavFeature]) -> NavFeasibilityReport:
        """Return whether this technique can run on the supplied features.

        ``is_feasible`` reads feature metadata only — never pixels.  It is
        called *before* ``navigate`` and short-circuits work that has no
        chance of producing a useful result.

        Parameters:
            features: Full feature set after the reliability gate.  The
                technique filters to its accepted types internally.

        Returns:
            A ``NavFeasibilityReport`` whose ``feasible`` field tells the
            orchestrator whether to invoke ``navigate``.
        """

    @abstractmethod
    def navigate(self, features: list[NavFeature], context: NavContext) -> NavTechniqueResult:
        """Compute and return a single per-technique offset estimate.

        Parameters:
            features: Features filtered to this technique's accepted types
                (the orchestrator pre-filters; the technique need not
                re-filter unless an internal sub-check requires it).
            context: Per-image NavContext with image, masks, and shared
                derivatives.

        Returns:
            A ``NavTechniqueResult`` with offset, covariance, confidence,
            and per-technique diagnostics.
        """


def filter_technique_names(names: list[str], patterns: str | list[str]) -> list[str]:
    """Return ``names`` filtered by gitignore-style glob patterns.

    A leading ``!`` marks an exclusion pattern; otherwise the pattern is
    inclusion.  An exclusion-only pattern list implies an inclusion default
    of ``'*'``.

    Parameters:
        names: List of technique names (or any candidate strings).
        patterns: Single pattern or list of patterns.  Patterns matching
            shell-glob syntax (``*``, ``?``, ``[abc]``).  Leading ``!``
            means exclusion.

    Returns:
        Names that match at least one inclusion pattern and no exclusion
        patterns.

    Raises:
        ValueError: if an empty pattern list is supplied.
    """
    if isinstance(patterns, str):
        patterns = [patterns]
    if not patterns:
        raise ValueError('patterns must contain at least one entry')
    includes: list[str] = []
    excludes: list[str] = []
    for pat in patterns:
        if pat.startswith('!'):
            excludes.append(pat[1:])
        else:
            includes.append(pat)
    if not includes:
        # Pure-exclusion pattern list: implicit '*' include.
        includes = ['*']
    out: list[str] = []
    for name in names:
        if not any(fnmatch.fnmatch(name, p) for p in includes):
            continue
        if any(fnmatch.fnmatch(name, p) for p in excludes):
            continue
        out.append(name)
    return out

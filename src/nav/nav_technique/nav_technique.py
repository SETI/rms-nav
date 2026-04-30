"""NavTechnique base class for the autonomous-navigation pipeline.

Concrete subclasses self-register via ``__init_subclass__`` and the
orchestrator iterates the registry.  Every concrete technique is safe to
instantiate per-image without depending on prior runs;
``__init_subclass__`` only records a class reference — no instances are
constructed at import time.
"""

from __future__ import annotations

import fnmatch
import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np

from nav.feature.feature import NavFeature
from nav.feature.feature_type import NavFeatureType
from nav.nav_technique.confidence import ConfidenceBreakdown, ConfidenceSpec
from nav.nav_technique.feasibility import NavFeasibilityReport
from nav.nav_technique.technique_result import NavTechniqueResult
from nav.support.nav_base import NavBase
from nav.support.types import NDArrayFloatType

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from nav.nav_orchestrator.nav_context import NavContext

__all__ = [
    'ROTATION_UNOBSERVABLE_VARIANCE',
    'NavTechnique',
    'embed_rotation_unobservable',
    'filter_technique_names',
    'log_confidence_breakdown',
    'rotation_pivot_distance_px',
    'rotation_unobservable_sigma_rad',
    'search_window_for_obs',
    'validate_registered_confidence_specs',
]


ROTATION_UNOBSERVABLE_VARIANCE: float = 1.0e15
"""Finite sentinel used as the rotation-axis variance when a technique
provides no rotation evidence (centroid-based fits, single-star matches,
template NCC without an outer rotation search).

The pseudoinverse used by the ensemble combine maps this huge variance to
a near-zero information contribution in the rotation direction, so the
technique contributes no rotation constraint to the precision-weighted
merge.  A literal ``+inf`` would break covariance validation
(``np.linalg.eigvalsh`` returns NaN for non-finite matrices), so this
finite sentinel is preferred.
"""


def embed_rotation_unobservable(cov_2x2: NDArrayFloatType) -> NDArrayFloatType:
    """Promote a 2x2 translation covariance to a 3x3 with rotation unobservable.

    The third diagonal carries :data:`ROTATION_UNOBSERVABLE_VARIANCE`; the
    cross-covariance entries are zero.  The ensemble's pseudo-inverse
    combine treats the rotation eigenvalue as null on input from this
    technique, so the rotation direction is ignored when fusing with
    rotation-aware techniques.

    Parameters:
        cov_2x2: 2x2 translation covariance.

    Returns:
        ``(3, 3)`` covariance matrix with the original 2x2 in the
        top-left block and the rotation-unobservable sentinel on the
        bottom-right diagonal.
    """
    out = np.zeros((3, 3), dtype=np.float64)
    out[:2, :2] = cov_2x2
    out[2, 2] = ROTATION_UNOBSERVABLE_VARIANCE
    return cast(NDArrayFloatType, out)


def rotation_unobservable_sigma_rad() -> float:
    """Return the rotation sigma corresponding to the unobservable sentinel.

    Convenience for techniques that report ``sigma_rotation_rad`` on
    their :class:`NavTechniqueResult` when rotation is unconstrained.
    """
    return float(math.sqrt(ROTATION_UNOBSERVABLE_VARIANCE))


def search_window_for_obs(context: NavContext) -> tuple[int, int]:
    """Return the ``(margin_v, margin_u)`` extfov margin from the obs.

    Every translation-fit technique reads the per-instrument extfov
    margin from ``context.obs.extfov_margin_vu`` to bound its at-edge
    detection.  Centralising the read in one helper keeps the convention
    uniform: a test obs stand-in that omits the attribute surfaces an
    ``AttributeError`` rather than silently falling through to a
    fabricated default.
    """
    margin = context.obs.extfov_margin_vu  # type: ignore[attr-defined]
    return (int(margin[0]), int(margin[1]))


def rotation_pivot_distance_px(
    pivot_vu: tuple[float, float], image_shape_vu: tuple[int, int]
) -> float:
    """Return the rotation-pivot's distance from the image centre.

    Used by 3-DoF DT-fit techniques to convert the LM rotation step into a
    pixel-equivalent value for the convergence test.  Floored at 1.0 so a
    pivot that lands exactly on the image centre still yields a usable
    convergence threshold.
    """
    centre_v = float(image_shape_vu[0]) / 2.0
    centre_u = float(image_shape_vu[1]) / 2.0
    dist = math.hypot(float(pivot_vu[0]) - centre_v, float(pivot_vu[1]) - centre_u)
    return max(dist, 1.0)


def log_confidence_breakdown(
    logger: Any, breakdown: ConfidenceBreakdown, *, low_threshold: float = 0.1
) -> None:
    """Emit a human-readable per-term explanation of a confidence value.

    Always logs the breakdown at DEBUG.  When the confidence falls below
    ``low_threshold`` the breakdown is *also* emitted at INFO so an
    operator running at the default INFO level sees why a fit reported
    near-zero confidence (typically: a single un-divided term has driven
    the sigmoid argument to a large negative value).

    Parameters:
        logger: A pdslogger compatible with ``info(...)`` / ``debug(...)``.
        breakdown: The :class:`ConfidenceBreakdown` returned by
            ``evaluate_sigmoid_combination(..., return_breakdown=True)``.
        low_threshold: Confidence at or below this value triggers the
            promotion from DEBUG to INFO.
    """
    summary_fmt = 'Confidence breakdown: alpha0=%.3f, sigmoid_arg=%.3f -> confidence=%.4f%s'
    summary_args = (
        breakdown.alpha0,
        breakdown.sigmoid_arg,
        breakdown.confidence,
        ' (hard_cap applied)' if breakdown.hard_cap_applied else '',
    )
    term_fmt = '  term %r: raw=%.4g, normalized=%.4g, alpha=%+.3f -> contribution=%+.4g'
    if breakdown.hard_zero is not None:
        logger.debug(summary_fmt, *summary_args)
        for term in breakdown.terms:
            logger.debug(
                term_fmt,
                term.feature,
                term.raw,
                term.normalized,
                term.alpha,
                term.contribution,
            )
        logger.info('Confidence forced to 0 by hard_zero_if[%r]=True', breakdown.hard_zero)
        return
    logger.debug(summary_fmt, *summary_args)
    for term in breakdown.terms:
        logger.debug(
            term_fmt,
            term.feature,
            term.raw,
            term.normalized,
            term.alpha,
            term.contribution,
        )
    if breakdown.confidence <= low_threshold:
        logger.info(summary_fmt, *summary_args)
        for term in breakdown.terms:
            logger.info(
                term_fmt,
                term.feature,
                term.raw,
                term.normalized,
                term.alpha,
                term.contribution,
            )


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
    #: Confidence-formula spec consumed by ``evaluate_sigmoid_combination``.
    #: Loaded from ``config_510_techniques.yaml`` and assigned at
    #: ``Config.read_config`` time.  ``None`` for techniques that opt out
    #: of the autonomous registry (e.g. ``NavTechniqueManual``).
    confidence_spec: ClassVar[ConfidenceSpec | None] = None
    #: Per-technique runtime tuning loaded from
    #: ``config_510_techniques.yaml.techniques.<name>.tuning``.  Each
    #: technique pulls the values it needs by name from this dict and
    #: falls back to the module-level default constant when a key is
    #: missing.  Empty for techniques that opt out of the autonomous
    #: registry or that have no tunable parameters.
    tuning: ClassVar[dict[str, float | int]] = {}
    #: Names of every attribute the technique's confidence spec may read
    #: (diagnostics fields plus side-channel flags such as ``at_edge``).
    #: ``validate_registered_confidence_specs`` ensures every term in
    #: ``confidence_spec`` references a member of this set.
    confidence_attributes: ClassVar[frozenset[str]] = frozenset()

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


def validate_registered_confidence_specs() -> None:
    """Validate every registered ``NavTechnique``'s confidence spec.

    Each technique whose spec was loaded from
    ``config_510_techniques.yaml`` (assigned by
    :meth:`nav.config.config.Config._validate_registered_techniques`)
    must declare the full set of valid attribute names in
    ``confidence_attributes``.  Every term's ``feature`` and every
    ``hard_zero_if`` key must appear in that set; otherwise the
    technique would raise at navigate time.  Validation runs at
    config-load time so the failure surfaces during process startup
    rather than mid-image.  Techniques whose spec is ``None`` (test-
    only registry entries opted out of YAML lookup) are skipped.

    Raises:
        ValueError: if any term references an unknown attribute.  The
            message names both the technique class and the bad
            attribute.
    """
    for cls in NavTechnique._registry:
        spec = cls.confidence_spec
        if spec is None:
            continue
        valid = cls.confidence_attributes
        for term in spec.terms:
            if term.feature not in valid:
                raise ValueError(
                    f'NavTechnique {cls.name!r}: confidence term references unknown '
                    f'attribute {term.feature!r}; declared attributes are '
                    f'{sorted(valid)!r}'
                )
        for key in spec.hard_zero_if:
            if key not in valid:
                raise ValueError(
                    f'NavTechnique {cls.name!r}: hard_zero_if references unknown '
                    f'attribute {key!r}; declared attributes are {sorted(valid)!r}'
                )


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

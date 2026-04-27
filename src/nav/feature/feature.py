"""NavFeature — the smallest independently-navigable scene element.

Every scene that the orchestrator considers is decomposed into NavFeatures by
the registered feature extractors.  Each feature carries everything a
technique needs to know to use it or ignore it: identity, geometry,
uncertainty, a preferred filter, a reliability score, and which technique
types are allowed to consume it.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from nav.feature.feature_type import NavFeatureType
from nav.feature.flags import NavFeatureFlags
from nav.feature.geometry import NavFeatureGeometry
from nav.support.filters import NavFilterSpec
from nav.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'NavFeature',
    'NavReliabilityBreakdown',
]


@dataclass(frozen=True)
class NavReliabilityBreakdown:
    """Per-component reliability contributions for a NavFeature.

    All fields are optional ``[0, 1]`` floats (or booleans where noted).  A
    missing field means "not applicable for this feature type", not zero.

    Parameters:
        predicted_snr: STAR — predicted detection SNR contribution.
        visible_arc_fraction: LIMB_ARC, TERMINATOR_ARC, RING_EDGE — fraction
            of total predicted arc length in extfov and not occluded.
        incidence_factor: LIMB_ARC — limb-incidence-angle softening
            factor.
        albedo_penalty: TERMINATOR_ARC — albedo-variation penalty term.
        shadow_occluded_fraction: RING_EDGE — fraction of polyline vertices
            dropped to planet shadow.
        visible_lit_fraction: BODY_DISC — fraction of the predicted disc
            that is both lit (cos(incidence) >= 0) and inside the sensor.
        overflow_fraction: BODY_DISC — fraction of the predicted disc area
            outside the sensor.
        blob_snr: BODY_BLOB — image SNR within the predicted body bbox.
        blob_extent_px: BODY_BLOB — predicted body extent (longer axis) in
            pixels.
        in_body_silhouette: STAR — True if the predicted star is inside a
            predicted body silhouette.
        in_saturation_or_cosmic: STAR — True if the predicted star pixel
            falls inside a saturation/cosmic-ray mask pixel.
        smear_length_ok: STAR — True if smear length is below the
            ``stars.max_smear`` reject gate.
    """

    predicted_snr: float | None = None
    visible_arc_fraction: float | None = None
    incidence_factor: float | None = None
    albedo_penalty: float | None = None
    shadow_occluded_fraction: float | None = None
    visible_lit_fraction: float | None = None
    overflow_fraction: float | None = None
    blob_snr: float | None = None
    blob_extent_px: float | None = None
    in_body_silhouette: bool | None = None
    in_saturation_or_cosmic: bool | None = None
    smear_length_ok: bool | None = None


@dataclass(frozen=True, eq=False)
class NavFeature:
    """A single navigable scene element with everything techniques need.

    NavFeatures are produced per-image by stateless extractors; no cross-image
    state is involved.  They carry their own uncertainty in image-plane pixel
    units, a preferred filter for the matching metric, and a self-assessed
    reliability score that the orchestrator's gate consults before invoking
    techniques.

    Parameters:
        feature_id: Unique identifier within a single NavResult.  Format is
            ``<type_lc>:<scope>``; e.g. ``star:UCAC4:144787700``,
            ``limb_arc:MIMAS``, ``ring_edge:SATURN:A_outer``.  Two features
            with the same ID is an extractor bug.
        feature_type: One of the ``NavFeatureType`` values.
        source_model: Name of the NavModel that produced this feature
            (``'stars'``, ``'body'``, ``'rings'``, etc.).
        geometry: One of the ``NavFeatureGeometry`` variants matching the
            feature type.
        subject_range_km: Distance from observer to the feature subject in
            kilometers.  Used to depth-sort body features when building
            combined templates.
        template_img: Optional pre-rendered model template (small postage
            stamp).  Set for BODY_DISC, RING_ANNULUS, and CARTOGRAPHIC_MODEL;
            ``None`` for feature types whose geometry alone is sufficient.
        template_mask: Optional boolean mask matching ``template_img``;
            ``True`` where the template carries signal.
        position_cov_px: 2x2 image-plane covariance in pixels for scalar
            features (STAR, BODY_BLOB).  ``None`` for polyline-payload and
            template-payload features whose covariance is per-vertex or
            derived from the matching peak.
        intensity_sigma_rel: Relative brightness uncertainty (scalar in
            ``[0, 1]``); ``0`` is "perfect brightness prediction".
        preferred_filter: NavFilterSpec the feature requests for both the
            template and the surrounding image patch.
        reliability: Self-assessed reliability score in ``[0, 1]``.
        reliability_reasons: Per-component breakdown of the reliability
            score; lets the curator surface why a feature was scored
            confidently or rejected.
        usable_types: Frozen set of feature types this feature is allowed
            to be consumed as.  Always contains its own ``feature_type`` and
            may include compatible cohabitants (a body's LIMB_ARC and
            BODY_DISC are separate features with the same body_name; they
            differ only in payload shape).
        flags: Typed feature-type-specific flags dataclass.
    """

    feature_id: str
    feature_type: NavFeatureType
    source_model: str
    geometry: NavFeatureGeometry
    subject_range_km: float
    position_cov_px: NDArrayFloatType | None
    intensity_sigma_rel: float
    preferred_filter: NavFilterSpec
    reliability: float
    reliability_reasons: NavReliabilityBreakdown
    usable_types: frozenset[NavFeatureType]
    flags: NavFeatureFlags
    template_img: NDArrayFloatType | None = None
    template_mask: NDArrayBoolType | None = None

    def __post_init__(self) -> None:
        """Validate invariants and freeze numpy arrays.

        Raises ValueError on out-of-range reliability, empty feature_id, or
        covariance matrices that are not 2x2 symmetric positive-semidefinite.
        """
        if not isinstance(self.feature_id, str) or not self.feature_id:
            raise ValueError(f'feature_id must be a non-empty string, got {self.feature_id!r}')
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError(f'reliability must lie in [0, 1]; got {self.reliability!r}')
        if self.feature_type not in self.usable_types:
            raise ValueError(
                f'usable_types must contain feature_type={self.feature_type.name}; '
                f'got {self.usable_types!r}'
            )
        if self.position_cov_px is not None:
            cov = np.asarray(self.position_cov_px, np.float64)
            if cov.shape != (2, 2):
                raise ValueError(f'position_cov_px must be 2x2; got shape {cov.shape}')
            if not np.isfinite(cov).all():
                raise ValueError('position_cov_px must be finite (no NaN or inf entries)')
            # Symmetric (within numerical tolerance) and positive-semidefinite.
            if not np.allclose(cov, cov.T, atol=1e-9):
                raise ValueError('position_cov_px must be symmetric')
            eigvals = np.linalg.eigvalsh(cov)
            if eigvals.min() < -1e-9:
                raise ValueError(
                    f'position_cov_px must be positive-semidefinite; got eigenvalues {eigvals!r}'
                )
            cov.setflags(write=False)
            # Replace the user-supplied array with the canonical float64
            # read-only copy so techniques see uniform dtype + immutability.
            # object.__setattr__ is the standard escape hatch for normalising
            # inputs in a frozen dataclass's __post_init__.
            object.__setattr__(self, 'position_cov_px', cov)
        if (self.template_img is None) != (self.template_mask is None):
            raise ValueError(
                'template_img and template_mask must be provided together '
                '(both None or both non-None)'
            )
        if self.template_img is not None:
            self.template_img.setflags(write=False)
        if self.template_mask is not None:
            self.template_mask.setflags(write=False)
        if (
            self.template_img is not None
            and self.template_mask is not None
            and self.template_img.shape != self.template_mask.shape
        ):
            raise ValueError(
                f'template_img shape {self.template_img.shape} does not match '
                f'template_mask shape {self.template_mask.shape}'
            )

    # Equality and hashing operate on feature_id only because the dataclass
    # carries unhashable numpy arrays (template_img, template_mask,
    # position_cov_px); the auto-generated dataclass __eq__/__hash__ would
    # break.  feature_id is required to be unique within a NavResult, so this
    # is well-defined.
    def __hash__(self) -> int:
        return hash(self.feature_id)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, NavFeature):
            return NotImplemented
        return self.feature_id == other.feature_id

"""NavTechniqueResult — the per-technique output consumed by the ensemble."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from nav.nav_technique.diagnostics import NavTechniqueDiagnostics
from nav.support.types import NDArrayFloatType

__all__ = ['NavTechniqueResult']


@dataclass(frozen=True, eq=False)
class NavTechniqueResult:
    """One per-technique navigation result.

    Parameters:
        technique_name: Class name of the producing technique.
        feature_ids: Tuple of NavFeature.feature_id values actually
            consumed.  Stored as an immutable ``tuple[str, ...]`` so the
            hash is stable across the lifetime of the instance; passing
            a ``list`` is accepted and converted in ``__post_init__``.
        offset_px: ``(dv, du)`` translational offset.  Convention: predicted
            position ``(v, u)`` means actual position is ``(v + dv, u + du)``.
        covariance_px2: 2x2 (or 3x3 with rotation) covariance matrix in
            pixel^2 (or pixel^2 / radian^2) units.
        confidence: Self-assessed [0, 1] calibrated confidence score.
        spurious: Hard-reject flag; the ensemble drops spurious results
            unconditionally.
        at_edge: True if the solution touched the search-window boundary.
        diagnostics: Per-technique typed diagnostics dataclass.
        rotation_rad: Optional fitted camera rotation (radians); ``None``
            on instruments where rotation fitting is disabled.
        sigma_rotation_rad: Optional 1-sigma uncertainty on the fitted
            rotation.
    """

    technique_name: str
    feature_ids: tuple[str, ...]
    offset_px: tuple[float, float]
    covariance_px2: NDArrayFloatType
    confidence: float
    spurious: bool
    at_edge: bool
    diagnostics: NavTechniqueDiagnostics
    rotation_rad: float | None = None
    sigma_rotation_rad: float | None = None

    def __post_init__(self) -> None:
        """Validate covariance shape and freeze the array."""
        cov = np.asarray(self.covariance_px2, np.float64)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError(f'covariance_px2 must be square 2-D; got shape {cov.shape}')
        if cov.shape[0] not in (2, 3):
            raise ValueError(f'covariance_px2 must be 2x2 or 3x3; got shape {cov.shape}')
        if not np.allclose(cov, cov.T, atol=1e-9):
            raise ValueError('covariance_px2 must be symmetric')
        eigvals = np.linalg.eigvalsh(cov)
        if eigvals.min() < -1e-9:
            raise ValueError(
                f'covariance_px2 must be positive-semidefinite; got eigenvalues {eigvals!r}'
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f'confidence must lie in [0, 1]; got {self.confidence!r}')
        cov.setflags(write=False)
        # Replace with the canonical float64 read-only copy.
        object.__setattr__(self, 'covariance_px2', cov)
        # Coerce a list-of-str input into the canonical tuple-of-str form
        # so the hash is stable across instance lifetimes.
        if not isinstance(self.feature_ids, tuple):
            object.__setattr__(self, 'feature_ids', tuple(self.feature_ids))

    # Equality and hashing operate on (technique_name, feature_ids) because
    # numpy-array fields prevent the default dataclass equality.
    def __hash__(self) -> int:
        return hash((self.technique_name, self.feature_ids))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, NavTechniqueResult):
            return NotImplemented
        return self.technique_name == other.technique_name and self.feature_ids == other.feature_ids

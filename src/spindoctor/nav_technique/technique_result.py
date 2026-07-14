"""NavTechniqueResult — the per-technique output consumed by the ensemble."""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from spindoctor.nav_technique.diagnostics import NavTechniqueDiagnostics
from spindoctor.support.types import NDArrayFloatType

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
        source_bodies: SPICE body names this result was computed against
            (from each consumed feature's structured ``body_name``).  Empty
            for ring / star techniques.  The ensemble's fallback-supersession
            filter reads this instead of parsing ``feature_ids`` strings.
        prior_source_techniques: Technique names whose pass-1 results
            seeded the prior this (pass-2) result refined; empty for
            prior-free results.  Stamped by the orchestrator after the
            pass-2 run.  The ensemble treats such a result as
            conditionally dependent on these techniques: it may refine
            their offset but not corroborate it.
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
    source_bodies: frozenset[str] = frozenset()
    prior_source_techniques: frozenset[str] = frozenset()

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
        # offset_px must be a length-2 finite pair: a NaN/inf offset would flow
        # straight into the ensemble combine and poison the result.
        if len(self.offset_px) != 2 or not all(math.isfinite(float(c)) for c in self.offset_px):
            raise ValueError(f'offset_px must be a length-2 finite tuple; got {self.offset_px!r}')
        # Rotation fields, when present, must be finite for the same reason.
        for field_name in ('rotation_rad', 'sigma_rotation_rad'):
            val = getattr(self, field_name)
            if val is not None and not math.isfinite(float(val)):
                raise ValueError(f'{field_name} must be finite when set; got {val!r}')
        cov.setflags(write=False)
        # Replace with the canonical float64 read-only copy.
        object.__setattr__(self, 'covariance_px2', cov)
        # Coerce a list-of-str input into the canonical tuple-of-str form
        # so the hash is stable across instance lifetimes.
        if not isinstance(self.feature_ids, tuple):
            object.__setattr__(self, 'feature_ids', tuple(self.feature_ids))
        if not all(isinstance(fid, str) for fid in self.feature_ids):
            raise ValueError(f'feature_ids must all be strings; got {self.feature_ids!r}')

    # Equality and hashing operate on (technique_name, feature_ids) because
    # numpy-array fields prevent the default dataclass equality.
    def __hash__(self) -> int:
        return hash((self.technique_name, self.feature_ids))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, NavTechniqueResult):
            return NotImplemented
        return self.technique_name == other.technique_name and self.feature_ids == other.feature_ids

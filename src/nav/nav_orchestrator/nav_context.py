"""NavContext — per-image global state shared across extractors and techniques.

Created once per navigation by the orchestrator.  Every member is computed
without knowing where any feature lives in the image: global statistics,
sensor-vs-extfov masks, shared image-side derivatives, and provenance.

The context is frozen.  Pass-2 techniques receive a copy with the pass-1
ensemble's prior offset and covariance attached via ``with_prior``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from nav.nav_orchestrator.image_classifier_result import NavImageClassifierResult
from nav.nav_orchestrator.provenance import Provenance
from nav.support.filters import NavFilterSpec
from nav.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = ['NavContext']


@dataclass(frozen=True, eq=False)
class NavContext:
    """Per-image global state shared across feature extraction and techniques.

    The context is frozen; ``with_prior`` returns a new instance via
    ``dataclasses.replace`` rather than mutating in place.

    Parameters:
        obs: The observation snapshot under navigation.  Typed loosely as
            ``object`` to avoid an import cycle; concrete value is an
            ``ObsSnapshotInst`` subclass.
        image_ext: The extended-FOV image array (post source-image filter).
        sensor_mask_ext: ``True`` where the pixel is real sensor data,
            ``False`` for extfov padding.
        image_noise_sigma: Robust MAD-based noise sigma (DN units), computed
            over the entire sensor area.
        saturation_mask_ext: ``True`` where pixels at or above the
            instrument's full-well DN.
        cosmic_ray_mask_ext: ``True`` where single-pixel cosmic-ray spikes
            were detected.
        image_classifier: The image-quality classifier's verdict.
        image_gradient_ext: Optional shared Sobel-of-Gaussian magnitude
            (computed once, reused by every DT-based technique).
        image_edge_dt_ext: Optional shared signed distance transform of the
            thresholded gradient image.
        prior_offset_px: Prior offset from pass 1, ``None`` on pass 1.
        prior_covariance_px2: Prior offset covariance from pass 1.
        pre_filter_applied: NavFilterSpec applied to the source image (for
            diagnostic provenance), ``None`` if none.
        provenance: Provenance metadata; populated at context creation.
    """

    obs: object
    image_ext: NDArrayFloatType
    sensor_mask_ext: NDArrayBoolType
    image_noise_sigma: float
    saturation_mask_ext: NDArrayBoolType
    cosmic_ray_mask_ext: NDArrayBoolType
    image_classifier: NavImageClassifierResult
    provenance: Provenance
    image_gradient_ext: NDArrayFloatType | None = None
    image_edge_dt_ext: NDArrayFloatType | None = None
    prior_offset_px: tuple[float, float] | None = None
    prior_covariance_px2: NDArrayFloatType | None = None
    pre_filter_applied: NavFilterSpec | None = None

    def with_prior(
        self,
        *,
        offset_px: tuple[float, float],
        covariance_px2: NDArrayFloatType,
    ) -> NavContext:
        """Return a new NavContext with pass-1 prior attached.

        ``with_prior`` is non-mutating; the existing instance is unchanged.

        Parameters:
            offset_px: ``(dv, du)`` offset to install as the pass-2 prior.
            covariance_px2: 2x2 covariance of that offset.

        Returns:
            New ``NavContext`` with ``prior_offset_px`` and
            ``prior_covariance_px2`` populated.
        """
        return dataclasses.replace(
            self,
            prior_offset_px=offset_px,
            prior_covariance_px2=covariance_px2,
        )

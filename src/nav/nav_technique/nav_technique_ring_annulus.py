"""``RingAnnulusNav`` — multi-ring composite NCC translation fit.

Consumes every ``RING_ANNULUS`` feature in the input set, fuses the per-
planet templates into a single composite by Z-buffer paint (closer ring
system's pixels overwrite farther ones), runs the existing pyramid
kpeaks NCC against the composite, and returns one combined translation.
``use_gradient`` defaults to ``'auto'`` so the NCC self-selects raw vs
gradient mode per image — raw wins on broad-brightness-gradient ring
geometries (low-resolution Saturn rings where the C-ring is uniformly
dim), gradient wins when sharp ringlet edges dominate.

Multi-planet annulus composites improve disambiguation in the same way
as multi-body disc composites: each annulus contributes its own
translational constraint to the joint NCC peak, the geometric alignment
between ring systems removes the "swap planet assignments" ambiguity,
and the SNR of the combined peak grows roughly as ``sqrt(N)`` if
backgrounds are independent.  Multi-planet scenes are rare but real
(the Cassini approach phase imaged Jupiter and Saturn together; New
Horizons imaged Jupiter from Pluto distance), so ``is_feasible`` must
handle ``len(features) > 1``.
"""

from __future__ import annotations

import numbers
from typing import TYPE_CHECKING

import numpy as np

from nav.config import Config
from nav.feature.composition import compose_template_features
from nav.feature.feature import NavFeature
from nav.feature.feature_type import NavFeatureType
from nav.feature.geometry import RingAnnulusGeometry
from nav.nav_technique.confidence import evaluate_sigmoid_combination
from nav.nav_technique.diagnostics import RingAnnulusDiagnostics
from nav.nav_technique.feasibility import NavFeasibilityReport
from nav.nav_technique.nav_technique import (
    NavTechnique,
    embed_rotation_unobservable,
    log_confidence_breakdown,
    rotation_unobservable_sigma_rad,
    search_window_for_obs,
)
from nav.nav_technique.technique_result import NavTechniqueResult
from nav.support.correlate import navigate_with_pyramid_kpeaks
from nav.support.types import NDArrayFloatType

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from nav.nav_orchestrator.nav_context import NavContext

__all__ = ['RingAnnulusNav']


_DEFAULT_UPSAMPLE_FACTOR: int = 128
"""Default FFT upsample factor when ``config.offset`` is missing."""


_MAX_UPSAMPLE_FACTOR: int = 1_000_000
"""Upper bound on the FFT upsample factor.

A misconfigured value above this would push the upsampled correlation
peak grid into multi-gigabyte allocations and effectively hang the
process; raise ``ValueError`` instead.  The bound is generous — the
shipped default is 128 — so legitimate per-instrument tuning is never
clipped.
"""


def _filter_annulus_features(features: list[NavFeature]) -> list[NavFeature]:
    """Return the subset that carries a ``RING_ANNULUS`` template payload."""
    return [
        f
        for f in features
        if f.feature_type is NavFeatureType.RING_ANNULUS
        and isinstance(f.geometry, RingAnnulusGeometry)
        and f.template_img is not None
        and f.template_mask is not None
    ]


def _peak_to_runner_up_ratio(top_k_peaks: list[tuple[float, float, float]]) -> float:
    """Return the ratio of the winning peak's quality to the runner-up's.

    ``top_k_peaks`` is ``[(quality, dv, du), ...]`` sorted by quality
    descending (the convention :func:`navigate_with_pyramid_kpeaks`
    uses).  Returns ``1.0`` when only one peak survives non-maximum
    suppression — which is what an unambiguous correlation looks
    like, so a value at or above 1.0 is the "good" tail.  Returns
    ``0.0`` when no peaks are present.  Negative-quality runners-up
    (rare; happens with the prior penalty) are floored at a small
    positive value so the ratio stays well-defined.
    """
    if not top_k_peaks:
        return 0.0
    if len(top_k_peaks) == 1:
        return 1.0
    winner_q = top_k_peaks[0][0]
    runner_q = top_k_peaks[1][0]
    if runner_q <= 1e-9:
        return float(max(winner_q, 0.0)) / 1e-9
    return float(winner_q) / float(runner_q)


class RingAnnulusNav(NavTechnique):
    """Ring-annulus full-template NCC translation fit (multi-planet, Z-buffer paint).

    Class attributes:
        accepts_feature_types: ``frozenset({RING_ANNULUS})``.
        requires_prior: ``False`` — the technique runs in pass 1.
    """

    name = 'RingAnnulusNav'
    accepts_feature_types = frozenset({NavFeatureType.RING_ANNULUS})
    requires_prior = False
    confidence_attributes = frozenset(
        {
            'at_edge',
            'spurious',
            'ncc_peak',
            'peak_to_runner_up_ratio',
            'used_gradient',
            'annulus_count',
        }
    )

    def __init__(self, *, config: Config | None = None) -> None:
        super().__init__(config=config)

    def is_feasible(self, features: list[NavFeature]) -> NavFeasibilityReport:
        """Return whether the input set carries any usable RING_ANNULUS feature.

        Reads only feature metadata — never any pixels — so the report is
        cheap to obtain even on large feature sets.  The feature list may
        carry one ``RING_ANNULUS`` per detectable ring system in
        multi-planet scenes; the technique handles ``len(features) > 1``
        by Z-buffer painting all annuli into one combined template.

        Parameters:
            features: Feature list filtered to this technique's accepted
                types.

        Returns:
            ``NavFeasibilityReport`` with ``feasible=True`` iff at least
            one ``RING_ANNULUS`` feature carries a template payload.
        """
        eligible = _filter_annulus_features(features)
        if not eligible:
            return NavFeasibilityReport(
                feasible=False,
                reason='no_ring_annulus_features_with_template',
            )
        return NavFeasibilityReport(
            feasible=True,
            reason='ok',
            consumed_feature_count=len(eligible),
        )

    def navigate(self, features: list[NavFeature], context: NavContext) -> NavTechniqueResult:
        """Compute the joint-translation offset from the input RING_ANNULUS templates.

        Parameters:
            features: Feature list filtered to this technique's accepted
                types.  Features without a template payload are dropped
                before fitting.
            context: Per-image NavContext.  Reads ``image_ext``,
                ``sensor_mask_ext``, ``obs.extfov_margin_vu``, and
                ``fit_camera_rotation``.

        Returns:
            A :class:`NavTechniqueResult` with the recovered offset,
            calibrated confidence, and a populated
            :class:`RingAnnulusDiagnostics`.  The covariance shape and
            the rotation fields depend on ``context.fit_camera_rotation``:

            - ``False`` (the default Cassini / NHLORRI posture):
              ``covariance_px2`` is ``(2, 2)`` and ``rotation_rad`` /
              ``sigma_rotation_rad`` are ``None``.
            - ``True``: the translation NCC pyramid carries no rotation
              evidence, so the result reports the rank-deficient
              ``(3, 3)`` form returned by
              :func:`~nav.nav_technique.nav_technique.embed_rotation_unobservable`
              with ``rotation_rad = 0.0`` and ``sigma_rotation_rad``
              equal to the rotation-unobservable sentinel.  Multi-
              planet ring scenes that warrant a 3-D NCC pyramid are
              tracked for Phase 12+; the rank-deficient encoding
              flows through the ensemble combine without contaminating
              other techniques' rotation slots.
        """
        with self.logger.open(f'TECHNIQUE: {self.name}'):
            eligible = _filter_annulus_features(features)
            self.logger.info(
                'Consuming %d RING_ANNULUS features (out of %d offered)',
                len(eligible),
                len(features),
            )
            if not eligible:
                raise ValueError(
                    'No usable RING_ANNULUS templates available for navigation '
                    '(every input feature was missing a template payload).  '
                    'Call is_feasible() before navigate() to gate this case.'
                )
            extfov_shape = context.image_ext.shape
            template_img, template_mask = compose_template_features(eligible, extfov_shape)
            margin_v, margin_u = search_window_for_obs(context)
            up_factor = self._upsample_factor()
            self.logger.debug(
                'Composite annulus template: %d painted pixels; '
                'search window (v, u) = (%d, %d) px; upsample factor = %d',
                int(template_mask.sum()),
                margin_v,
                margin_u,
                up_factor,
            )
            ncc_result = navigate_with_pyramid_kpeaks(
                image=context.image_ext,
                model=template_img,
                mask=template_mask,
                upsample_factor=up_factor,
                max_offset_vu=(margin_v, margin_u),
                data_mask=context.sensor_mask_ext,
                use_gradient='auto',
                logger=self.logger,
            )
            dv = float(ncc_result['offset'][0])
            du = float(ncc_result['offset'][1])
            covariance_2x2 = np.asarray(ncc_result['cov'], np.float64)
            if covariance_2x2.shape != (2, 2):
                covariance_2x2 = covariance_2x2[:2, :2]
            fit_rotation = bool(context.fit_camera_rotation)
            covariance: NDArrayFloatType = (
                embed_rotation_unobservable(covariance_2x2) if fit_rotation else covariance_2x2
            )
            spurious = bool(ncc_result['spurious'])
            at_edge = bool(ncc_result['at_edge'])
            quality = float(ncc_result['quality'])
            used_gradient = bool(ncc_result.get('used_gradient', False))
            top_k_peaks = ncc_result.get('top_k_peaks', [])
            diagnostics = RingAnnulusDiagnostics(
                ncc_peak=quality,
                peak_to_runner_up_ratio=_peak_to_runner_up_ratio(top_k_peaks),
                annulus_count=len(eligible),
                used_gradient=used_gradient,
            )
            assert self.confidence_spec is not None  # set as class attribute
            confidence, breakdown = evaluate_sigmoid_combination(
                self.confidence_spec,
                _AnnulusConfidenceContext(
                    at_edge=at_edge, spurious=spurious, diagnostics=diagnostics
                ),
                technique_name=self.name,
                return_breakdown=True,
            )
            log_confidence_breakdown(self.logger, breakdown)
            self.logger.info(
                'Converged at offset (%.4f, %.4f) px, quality %.3f, '
                'mode=%s, annuli=%d, confidence %.4f',
                dv,
                du,
                quality,
                'gradient' if used_gradient else 'raw',
                len(eligible),
                float(confidence),
            )
            if spurious or at_edge:
                self.logger.info('Diagnostic flags: spurious=%s, at_edge=%s', spurious, at_edge)
            return NavTechniqueResult(
                technique_name=self.name,
                feature_ids=tuple(f.feature_id for f in eligible),
                offset_px=(dv, du),
                covariance_px2=covariance,
                confidence=float(confidence),
                spurious=spurious,
                at_edge=at_edge,
                diagnostics=diagnostics,
                rotation_rad=0.0 if fit_rotation else None,
                sigma_rotation_rad=(rotation_unobservable_sigma_rad() if fit_rotation else None),
            )

    def _upsample_factor(self) -> int:
        """Return the FFT upsample factor configured under ``config.offset``.

        Validates the raw value: must be a real (non-bool) number, must
        coerce to ``int >= 1``, and must lie below
        :data:`_MAX_UPSAMPLE_FACTOR` so a malformed config cannot push
        the FFT into a multi-gigabyte allocation.
        """
        offset_block = getattr(self.config, 'offset', None)
        if offset_block is None:
            return _DEFAULT_UPSAMPLE_FACTOR
        raw = getattr(offset_block, 'correlation_fft_upsample_factor', None)
        if raw is None:
            return _DEFAULT_UPSAMPLE_FACTOR
        if isinstance(raw, bool) or not isinstance(raw, numbers.Real):
            raise ValueError(
                f'config.offset.correlation_fft_upsample_factor must be a '
                f'real (non-bool) number; got {type(raw).__name__} {raw!r}'
            )
        coerced = int(raw)
        if coerced < 1 or coerced > _MAX_UPSAMPLE_FACTOR:
            raise ValueError(
                f'config.offset.correlation_fft_upsample_factor must lie in '
                f'[1, {_MAX_UPSAMPLE_FACTOR}]; got {raw!r}'
            )
        return coerced


class _AnnulusConfidenceContext:
    """Adapter binding ``RingAnnulusDiagnostics`` plus ``at_edge`` / ``spurious``.

    The shared :func:`evaluate_sigmoid_combination` helper accepts any
    object whose attributes match the spec's term names.  ``at_edge`` and
    ``spurious`` are not part of ``RingAnnulusDiagnostics`` (they live
    on ``NavTechniqueResult``) so this small adapter exposes both
    alongside the diagnostic fields the spec consumes.
    """

    def __init__(
        self, *, at_edge: bool, spurious: bool, diagnostics: RingAnnulusDiagnostics
    ) -> None:
        self.at_edge = at_edge
        self.spurious = spurious
        self.ncc_peak = diagnostics.ncc_peak
        self.peak_to_runner_up_ratio = diagnostics.peak_to_runner_up_ratio
        self.used_gradient = diagnostics.used_gradient
        self.annulus_count = float(diagnostics.annulus_count)

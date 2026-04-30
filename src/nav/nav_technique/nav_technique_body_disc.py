"""``BodyDiscCorrelateNav`` — full-disc NCC translation fit.

Consumes every ``BODY_DISC`` feature in the input set, fuses the per-body
templates into a single composite by Z-buffer paint (closer body's pixels
overwrite farther body's), runs the existing pyramid kpeaks NCC against
the composite, and returns one combined translation.  ``use_gradient``
defaults to ``'auto'`` so the NCC self-selects raw vs gradient mode per
image — raw wins on smooth Lambert-shaded discs that fill the FOV;
gradient wins when only the limb carries unique-alignment signal.

Multi-body composites improve disambiguation: with ``N`` bodies the
correlation peak's SNR grows roughly as ``sqrt(N)`` if backgrounds are
independent, and the joint geometric constraint removes the
"swap moon assignments" mode-failure that plagues per-body solo
correlation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nav.config import Config
from nav.feature.composition import compose_template_features
from nav.feature.feature import NavFeature
from nav.feature.feature_type import NavFeatureType
from nav.feature.geometry import BodyDiscGeometry
from nav.nav_technique.confidence import evaluate_sigmoid_combination
from nav.nav_technique.diagnostics import BodyDiscDiagnostics
from nav.nav_technique.feasibility import NavFeasibilityReport
from nav.nav_technique.nav_technique import (
    NavTechnique,
    log_confidence_breakdown,
    search_window_for_obs,
)
from nav.nav_technique.technique_result import NavTechniqueResult
from nav.support.correlate import navigate_with_pyramid_kpeaks

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from nav.nav_orchestrator.nav_context import NavContext

__all__ = ['BodyDiscCorrelateNav']


def _filter_disc_features(features: list[NavFeature]) -> list[NavFeature]:
    """Return the subset that carries a ``BODY_DISC`` template payload."""
    return [
        f
        for f in features
        if f.feature_type is NavFeatureType.BODY_DISC
        and isinstance(f.geometry, BodyDiscGeometry)
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


class BodyDiscCorrelateNav(NavTechnique):
    """Body-disc full-disc NCC translation fit (multi-body, Z-buffer paint).

    Class attributes:
        accepts_feature_types: ``frozenset({BODY_DISC})``.
        requires_prior: ``False`` — the technique runs in pass 1.
    """

    name = 'BodyDiscCorrelateNav'
    accepts_feature_types = frozenset({NavFeatureType.BODY_DISC})
    requires_prior = False
    confidence_attributes = frozenset(
        {
            'at_edge',
            'spurious',
            'ncc_peak',
            'peak_to_runner_up_ratio',
            'consistency_px',
            'used_gradient',
            'body_count',
        }
    )

    def __init__(self, *, config: Config | None = None) -> None:
        super().__init__(config=config)

    def is_feasible(self, features: list[NavFeature]) -> NavFeasibilityReport:
        """Return whether the input set carries any usable BODY_DISC feature.

        Reads only feature metadata — never any pixels — so the report is
        cheap to obtain even on large feature sets.

        Parameters:
            features: Feature list filtered to this technique's accepted
                types.

        Returns:
            ``NavFeasibilityReport`` with ``feasible=True`` iff at least
            one ``BODY_DISC`` feature carries a template payload.
        """
        eligible = _filter_disc_features(features)
        if not eligible:
            return NavFeasibilityReport(
                feasible=False,
                reason='no_body_disc_features_with_template',
            )
        return NavFeasibilityReport(
            feasible=True,
            reason='ok',
            consumed_feature_count=len(eligible),
        )

    def navigate(self, features: list[NavFeature], context: NavContext) -> NavTechniqueResult:
        """Compute the joint-translation offset from the input BODY_DISC templates.

        Parameters:
            features: Feature list filtered to this technique's accepted
                types.  Features without a template payload are dropped
                before fitting.
            context: Per-image NavContext.  Reads ``image_ext``,
                ``sensor_mask_ext``, and ``obs.extfov_margin_vu``.

        Returns:
            A ``NavTechniqueResult`` with the recovered offset, 2x2
            covariance, calibrated confidence, and a populated
            :class:`BodyDiscDiagnostics`.
        """
        with self.logger.open(f'TECHNIQUE: {self.name}'):
            eligible = _filter_disc_features(features)
            self.logger.info(
                'Consuming %d BODY_DISC features (out of %d offered)',
                len(eligible),
                len(features),
            )
            extfov_shape = context.image_ext.shape
            template_img, template_mask = compose_template_features(eligible, extfov_shape)
            margin_v, margin_u = search_window_for_obs(context)
            up_factor = self._upsample_factor()
            self.logger.debug(
                'Composite template: %d painted pixels; search window (v, u) = (%d, %d) px; '
                'upsample factor = %d',
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
            covariance = np.asarray(ncc_result['cov'], np.float64)
            if covariance.shape != (2, 2):
                covariance = covariance[:2, :2]
            spurious = bool(ncc_result['spurious'])
            at_edge = bool(ncc_result['at_edge'])
            quality = float(ncc_result['quality'])
            consistency = float(ncc_result['consistency'])
            used_gradient = bool(ncc_result.get('used_gradient', False))
            top_k_peaks = ncc_result.get('top_k_peaks', [])
            diagnostics = BodyDiscDiagnostics(
                ncc_peak=quality,
                peak_to_runner_up_ratio=_peak_to_runner_up_ratio(top_k_peaks),
                consistency_px=consistency,
                used_gradient=used_gradient,
                body_count=len(eligible),
            )
            assert self.confidence_spec is not None  # set as class attribute
            confidence, breakdown = evaluate_sigmoid_combination(
                self.confidence_spec,
                _DiscConfidenceContext(at_edge=at_edge, spurious=spurious, diagnostics=diagnostics),
                technique_name=self.name,
                return_breakdown=True,
            )
            log_confidence_breakdown(self.logger, breakdown)
            self.logger.info(
                'Converged at offset (%.4f, %.4f) px, quality %.3f, consistency %.3f, '
                'mode=%s, bodies=%d, confidence %.4f',
                dv,
                du,
                quality,
                consistency,
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
            )

    def _upsample_factor(self) -> int:
        """Return the FFT upsample factor configured under ``config.offset``."""
        offset_block = getattr(self.config, 'offset', None)
        if offset_block is None:
            return 128
        return int(getattr(offset_block, 'correlation_fft_upsample_factor', 128))


class _DiscConfidenceContext:
    """Adapter binding ``BodyDiscDiagnostics`` plus ``at_edge`` / ``spurious``.

    The shared :func:`evaluate_sigmoid_combination` helper accepts any
    object whose attributes match the spec's term names.  ``at_edge`` and
    ``spurious`` are not part of ``BodyDiscDiagnostics`` (they live on
    ``NavTechniqueResult``) so this small adapter exposes both alongside
    the diagnostic fields the spec consumes.
    """

    def __init__(self, *, at_edge: bool, spurious: bool, diagnostics: BodyDiscDiagnostics) -> None:
        self.at_edge = at_edge
        self.spurious = spurious
        self.ncc_peak = diagnostics.ncc_peak
        self.peak_to_runner_up_ratio = diagnostics.peak_to_runner_up_ratio
        self.consistency_px = diagnostics.consistency_px
        self.used_gradient = diagnostics.used_gradient
        self.body_count = float(diagnostics.body_count)

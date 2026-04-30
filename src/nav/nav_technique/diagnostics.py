"""Per-technique diagnostics dataclasses.

Each NavTechnique returns a typed diagnostics object on its
``NavTechniqueResult.diagnostics`` field.  The curator walks
``CURATOR_FIELDS`` on each diagnostics class to decide which fields land in
the JSON metadata; an unmapped field is a CI failure.
"""

from dataclasses import dataclass
from typing import ClassVar

__all__ = [
    'BodyBlobDiagnostics',
    'BodyDiscDiagnostics',
    'BodyLimbDiagnostics',
    'BodyTerminatorDiagnostics',
    'NavTechniqueDiagnostics',
    'RingAnnulusDiagnostics',
    'RingEdgeDiagnostics',
    'StarFieldDiagnostics',
    'StarRefineDiagnostics',
    'StarUniqueMatchDiagnostics',
]


@dataclass(frozen=True)
class BodyDiscDiagnostics:
    """Diagnostics emitted by ``BodyDiscCorrelateNav``.

    Parameters:
        ncc_peak: Peak normalized cross-correlation value.
        peak_to_runner_up_ratio: Ratio of NCC peak to second-highest peak
            outside the exclusion radius around the peak.
        consistency_px: Mean per-axis disagreement between coarse-pyramid
            and full-resolution sub-pixel locations.
        used_gradient: True if gradient mode was selected by ``auto``.
        body_count: Number of BODY_DISC features fused into the combined
            template.
    """

    ncc_peak: float = 0.0
    peak_to_runner_up_ratio: float = 0.0
    consistency_px: float = 0.0
    used_gradient: bool = False
    body_count: int = 0
    CURATOR_FIELDS: ClassVar[dict[str, str | None]] = {
        'ncc_peak': 'ncc_peak',
        'peak_to_runner_up_ratio': 'peak_to_runner_up_ratio',
        'consistency_px': 'consistency_px',
        'used_gradient': 'used_gradient',
        'body_count': 'body_count',
    }


@dataclass(frozen=True)
class BodyLimbDiagnostics:
    """Diagnostics emitted by ``BodyLimbNav``.

    Parameters:
        visible_limb_arc_fraction: Fused visible-arc fraction across input
            LIMB_ARC features.
        visible_arc_px: Total surviving polyline arc length in pixels.
        dt_fit_rms_px: Final root-mean-square DT residual.
        lm_iterations: Levenberg-Marquardt iteration count.
        tukey_inlier_count: Number of polyline vertices accepted by the
            Tukey biweight robust estimator.
    """

    visible_limb_arc_fraction: float = 0.0
    visible_arc_px: float = 0.0
    dt_fit_rms_px: float = 0.0
    lm_iterations: int = 0
    tukey_inlier_count: int = 0
    CURATOR_FIELDS: ClassVar[dict[str, str | None]] = {
        'visible_limb_arc_fraction': 'visible_limb_arc_fraction',
        'visible_arc_px': 'visible_arc_px',
        'dt_fit_rms_px': 'dt_fit_rms_px',
        'lm_iterations': 'lm_iterations',
        'tukey_inlier_count': 'tukey_inlier_count',
    }


@dataclass(frozen=True)
class BodyTerminatorDiagnostics:
    """Diagnostics emitted by ``BodyTerminatorNav``.

    Parameters: same shape as ``BodyLimbDiagnostics`` with
    ``visible_terminator_arc_fraction`` substituted.
    """

    visible_terminator_arc_fraction: float = 0.0
    visible_arc_px: float = 0.0
    dt_fit_rms_px: float = 0.0
    lm_iterations: int = 0
    tukey_inlier_count: int = 0
    CURATOR_FIELDS: ClassVar[dict[str, str | None]] = {
        'visible_terminator_arc_fraction': 'visible_terminator_arc_fraction',
        'visible_arc_px': 'visible_arc_px',
        'dt_fit_rms_px': 'dt_fit_rms_px',
        'lm_iterations': 'lm_iterations',
        'tukey_inlier_count': 'tukey_inlier_count',
    }


@dataclass(frozen=True)
class BodyBlobDiagnostics:
    """Diagnostics emitted by ``BodyBlobNav``.

    Parameters:
        body_snr_inside_predicted_bbox: SNR within the predicted bbox.
        body_extent_px: Predicted body's longer-axis extent in pixels.
        blob_count: Number of BODY_BLOB features fused.
        residual_px: Centroid-fit RMS residual.
    """

    body_snr_inside_predicted_bbox: float = 0.0
    body_extent_px: float = 0.0
    blob_count: int = 0
    residual_px: float = 0.0
    CURATOR_FIELDS: ClassVar[dict[str, str | None]] = {
        'body_snr_inside_predicted_bbox': 'body_snr_inside_predicted_bbox',
        'body_extent_px': 'body_extent_px',
        'blob_count': 'blob_count',
        'residual_px': 'residual_px',
    }


@dataclass(frozen=True)
class RingEdgeDiagnostics:
    """Diagnostics emitted by ``RingEdgeNav``.

    Parameters:
        total_edge_length_px: Cumulative pixel length of all surviving
            ring-edge polylines.
        per_edge_dt_rms_summed: Sum of per-edge final DT RMS values.
        edge_count: Number of RING_EDGE features fused.
        is_rank_1: True if every ring-edge feature was straight-line and the
            combined covariance is rank-1.
    """

    total_edge_length_px: float = 0.0
    per_edge_dt_rms_summed: float = 0.0
    edge_count: int = 0
    is_rank_1: bool = False
    CURATOR_FIELDS: ClassVar[dict[str, str | None]] = {
        'total_edge_length_px': 'total_edge_length_px',
        'per_edge_dt_rms_summed': 'per_edge_dt_rms_summed',
        'edge_count': 'edge_count',
        'is_rank_1': 'is_rank_1',
    }


@dataclass(frozen=True)
class RingAnnulusDiagnostics:
    """Diagnostics emitted by ``RingAnnulusNav``.

    Parameters:
        ncc_peak: Peak NCC value.
        peak_to_runner_up_ratio: NCC peak ratio.
        annulus_count: Number of RING_ANNULUS features (one per planet).
        used_gradient: True if gradient mode was selected.
    """

    ncc_peak: float = 0.0
    peak_to_runner_up_ratio: float = 0.0
    annulus_count: int = 0
    used_gradient: bool = False
    CURATOR_FIELDS: ClassVar[dict[str, str | None]] = {
        'ncc_peak': 'ncc_peak',
        'peak_to_runner_up_ratio': 'peak_to_runner_up_ratio',
        'annulus_count': 'annulus_count',
        'used_gradient': 'used_gradient',
    }


@dataclass(frozen=True)
class StarFieldDiagnostics:
    """Diagnostics emitted by ``StarFieldFromCatalogNav``.

    Parameters:
        n_inliers: Number of detection-to-catalog inliers after RANSAC.
        median_residual_px: Median position residual on inliers.
        n_detected_sources: Number of bright sources detected in the image.
        n_catalog_predicted: Number of catalog stars in the extfov.
        n_triplets_evaluated: Number of triplet candidates considered by
            RANSAC.
    """

    n_inliers: int = 0
    median_residual_px: float = 0.0
    n_detected_sources: int = 0
    n_catalog_predicted: int = 0
    n_triplets_evaluated: int = 0
    CURATOR_FIELDS: ClassVar[dict[str, str | None]] = {
        'n_inliers': 'n_inliers',
        'median_residual_px': 'median_residual_px',
        'n_detected_sources': 'n_detected_sources',
        'n_catalog_predicted': 'n_catalog_predicted',
        'n_triplets_evaluated': 'n_triplets_evaluated',
    }


@dataclass(frozen=True)
class StarUniqueMatchDiagnostics:
    """Diagnostics emitted by ``StarUniqueMatchNav``.

    Parameters:
        mode: ``'one_star'`` or ``'two_star'``.
        predicted_snr: Predicted SNR of the brightest catalog star.
        brightness_margin_mag: Mag difference to the next-brightest *unmatched*
            catalog source predictable in extfov; ``+inf`` when no unmatched
            star exists (a 1-star scene with no other predictable star, or a
            2-star scene with no third predictable star to compare against).
        residual_px: Detection-vs-prediction residual.
    """

    mode: str = ''
    predicted_snr: float = 0.0
    brightness_margin_mag: float = 0.0
    residual_px: float = 0.0
    CURATOR_FIELDS: ClassVar[dict[str, str | None]] = {
        'mode': 'mode',
        'predicted_snr': 'predicted_snr',
        'brightness_margin_mag': 'brightness_margin_mag',
        'residual_px': 'residual_px',
    }


@dataclass(frozen=True)
class StarRefineDiagnostics:
    """Diagnostics emitted by ``StarRefineNav``.

    Parameters:
        n_stars_used: Number of stars that survived per-star quality gates.
        median_pos_err_px: Median refinement positional error.
        residual_scatter_px: Per-axis RMS scatter of the per-star residuals.
    """

    n_stars_used: int = 0
    median_pos_err_px: float = 0.0
    residual_scatter_px: float = 0.0
    CURATOR_FIELDS: ClassVar[dict[str, str | None]] = {
        'n_stars_used': 'n_stars_used',
        'median_pos_err_px': 'median_pos_err_px',
        'residual_scatter_px': 'residual_scatter_px',
    }


NavTechniqueDiagnostics = (
    BodyDiscDiagnostics
    | BodyLimbDiagnostics
    | BodyTerminatorDiagnostics
    | BodyBlobDiagnostics
    | RingEdgeDiagnostics
    | RingAnnulusDiagnostics
    | StarFieldDiagnostics
    | StarUniqueMatchDiagnostics
    | StarRefineDiagnostics
)
"""Sum type spanning every per-technique diagnostics dataclass.

The orchestrator's curator and the technique-result type both consume
this union; adding a new technique means adding both its diagnostics
dataclass above and a new entry into this union.
"""

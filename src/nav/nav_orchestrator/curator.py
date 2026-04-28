"""Curator — turn a NavResult into a JSON-friendly metadata dict.

The curator picks JSON-friendly fields from a NavResult, rounds floats to
documented precision, replaces ``inf`` with the ``JSON_INF_SENTINEL`` finite
sentinel, and emits the ``navigation_result`` block consumed by downstream
readers.

Every per-technique diagnostic field that ships in the JSON appears in the
technique's ``CURATOR_FIELDS`` allow-list; ``assert_diagnostic_fields_present``
fails CI if a new diagnostic field is added without updating the allow-list.
"""

import dataclasses
import logging
import math
from typing import Any

from nav.feature.constants import JSON_INF_SENTINEL
from nav.nav_orchestrator.feature_summary import NavFeatureSummary
from nav.nav_orchestrator.image_classifier_result import NavImageClassifierResult
from nav.nav_orchestrator.nav_result import NavResult
from nav.nav_orchestrator.provenance import Provenance
from nav.nav_technique.technique_result import NavTechniqueResult

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    'assert_diagnostic_fields_present',
    'build_metadata_dict',
]


# Float-rounding policy: 4 decimals for pixel quantities, 3 for confidence
# scores, 6 for ET timestamps.  Tighter precision than the per-image
# tolerance budget; chosen to give stable byte-identical JSON outputs.
PIXEL_DECIMALS = 4
CONFIDENCE_DECIMALS = 3
ET_DECIMALS = 6


def _round_float(value: float, decimals: int) -> float:
    """Round a float to ``decimals`` digits, preserving ``inf``/``nan``."""
    if math.isinf(value):
        return JSON_INF_SENTINEL if value > 0 else -JSON_INF_SENTINEL
    if math.isnan(value):
        return 0.0
    return round(value, decimals)


def _round_pair(value: tuple[float, float] | None) -> list[float] | None:
    if value is None:
        return None
    return [_round_float(value[0], PIXEL_DECIMALS), _round_float(value[1], PIXEL_DECIMALS)]


def _round_2x2(matrix: Any) -> list[list[float]] | None:
    """Round a 2x2 covariance into a JSON-friendly nested list."""
    if matrix is None:
        return None
    rows = []
    for r in range(matrix.shape[0]):
        rows.append(
            [_round_float(float(matrix[r, c]), PIXEL_DECIMALS) for c in range(matrix.shape[1])]
        )
    return rows


def assert_diagnostic_fields_present(result: NavResult) -> None:
    """Verify every per-technique diagnostic field has a CURATOR_FIELDS entry.

    An unmapped field is a programmer error and raises ``AssertionError``
    so CI fails the build before a new diagnostic silently disappears from
    the JSON output.

    Parameters:
        result: NavResult whose per_technique entries are inspected.

    Raises:
        AssertionError: if any field on a diagnostic dataclass is missing
            from its ``CURATOR_FIELDS`` allow-list.
    """
    for tech_result in result.per_technique:
        diag = tech_result.diagnostics
        if not hasattr(diag, 'CURATOR_FIELDS'):
            raise AssertionError(f'{type(diag).__name__} is missing CURATOR_FIELDS class attribute')
        curator_fields = diag.CURATOR_FIELDS
        declared = set(curator_fields)
        actual = {f.name for f in dataclasses.fields(diag)}
        missing = actual - declared
        if missing:
            raise AssertionError(
                f'{type(diag).__name__} has unmapped fields {sorted(missing)!r}; '
                f'add them to CURATOR_FIELDS or set value to None to skip'
            )


def _curate_diagnostics(diag: Any) -> dict[str, Any]:
    """Return a JSON-friendly subset of a diagnostic dataclass."""
    out: dict[str, Any] = {}
    curator_fields = diag.CURATOR_FIELDS
    for attr_name, json_key in curator_fields.items():
        if json_key is None:
            continue
        value = getattr(diag, attr_name)
        if isinstance(value, float):
            out[json_key] = _round_float(value, CONFIDENCE_DECIMALS)
        else:
            out[json_key] = value
    return out


def _curate_technique_result(res: NavTechniqueResult) -> dict[str, Any]:
    """Return a JSON-friendly subset of a NavTechniqueResult."""
    out: dict[str, Any] = {
        'technique_name': res.technique_name,
        'feature_ids': list(res.feature_ids),
        'offset_px': _round_pair(res.offset_px),
        'covariance_px2': _round_2x2(res.covariance_px2),
        'confidence': _round_float(res.confidence, CONFIDENCE_DECIMALS),
        'spurious': res.spurious,
        'at_edge': res.at_edge,
        'diagnostics': _curate_diagnostics(res.diagnostics),
    }
    if res.rotation_rad is not None:
        out['rotation_deg'] = _round_float(math.degrees(res.rotation_rad), CONFIDENCE_DECIMALS)
    if res.sigma_rotation_rad is not None:
        out['sigma_rotation_deg'] = _round_float(
            math.degrees(res.sigma_rotation_rad), CONFIDENCE_DECIMALS
        )
    return out


def _curate_feature_summary(summary: NavFeatureSummary) -> dict[str, Any]:
    """Return a JSON-friendly entry for one NavFeatureSummary."""
    return {
        'feature_id': summary.feature_id,
        'feature_type': summary.feature_type.value,
        'source_model': summary.source_model,
        'reliability': _round_float(summary.reliability, CONFIDENCE_DECIMALS),
        'gated': summary.gated,
        'gate_reason': summary.gate_reason,
        'bbox_extfov_vu': list(summary.bbox_extfov_vu),
    }


def _curate_image_classifier(classifier: NavImageClassifierResult) -> dict[str, Any]:
    """Return a JSON-friendly entry for the image-quality classifier result."""
    return {
        'class': classifier.image_class,
        'saturation_frac': _round_float(classifier.saturation_frac, CONFIDENCE_DECIMALS),
        'missing_frac': _round_float(classifier.missing_frac, CONFIDENCE_DECIMALS),
        'noise_sigma': _round_float(classifier.noise_sigma, CONFIDENCE_DECIMALS),
        'max_dn': _round_float(classifier.max_dn, CONFIDENCE_DECIMALS),
        'flags': list(classifier.flags),
    }


def _curate_provenance(provenance: Provenance) -> dict[str, Any]:
    """Return a JSON-friendly entry for the provenance envelope."""
    return {
        'rms_nav_version': provenance.rms_nav_version,
        'rms_nav_git_sha': provenance.rms_nav_git_sha,
        'spice_kernels': list(provenance.spice_kernels),
        'spice_kernel_count': provenance.spice_kernel_count,
        'static_data_hashes': dict(provenance.static_data_hashes),
        'technique_names': list(provenance.technique_names),
        'extractor_names': list(provenance.extractor_names),
        'image_et': _round_float(provenance.image_et, ET_DECIMALS),
        'pipeline_run_iso8601': provenance.pipeline_run_iso8601,
    }


def build_metadata_dict(result: NavResult) -> dict[str, Any]:
    """Build the additive ``navigation_result`` metadata block.

    Output is a JSON-serializable dict; the orchestrator merges it into the
    existing per-image ``_metadata.json`` schema as the additive
    ``navigation_result`` key.

    Parameters:
        result: The full in-memory NavResult.

    Returns:
        Dict ready for ``json.dump``.

    Raises:
        AssertionError: if any diagnostic field is missing a
            ``CURATOR_FIELDS`` entry.
    """
    assert_diagnostic_fields_present(result)
    sigma_along_unobservable_px: float | None
    if result.sigma_along_unobservable_px is None:
        sigma_along_unobservable_px = None
    else:
        sigma_along_unobservable_px = _round_float(
            result.sigma_along_unobservable_px, PIXEL_DECIMALS
        )
    feature_count_by_type: dict[str, int] = {}
    for entry in result.feature_inventory:
        if entry.gated:
            continue
        key = entry.feature_type.value
        feature_count_by_type[key] = feature_count_by_type.get(key, 0) + 1
    out: dict[str, Any] = {
        'status': result.status,
        'status_reason': result.status_reason.value,
        'offset_px': _round_pair(result.offset_px),
        'sigma_px': _round_pair(result.sigma_px),
        'sigma_along_unobservable_px': sigma_along_unobservable_px,
        'confidence': _round_float(result.confidence, CONFIDENCE_DECIMALS),
        'confidence_rank': result.confidence_rank,
        'covariance_px2': _round_2x2(result.covariance_px2),
        'techniques_used': sorted({r.technique_name for r in result.per_technique}),
        'feature_count_by_type': feature_count_by_type,
        'per_technique': [_curate_technique_result(r) for r in result.per_technique],
        'feature_inventory': [_curate_feature_summary(s) for s in result.feature_inventory],
        'image_classifier': _curate_image_classifier(result.image_classifier),
        'provenance': _curate_provenance(result.provenance),
    }
    if result.rotation_rad is not None:
        out['rotation_deg'] = _round_float(math.degrees(result.rotation_rad), CONFIDENCE_DECIMALS)
    if result.sigma_rotation_rad is not None:
        out['sigma_rotation_deg'] = _round_float(
            math.degrees(result.sigma_rotation_rad), CONFIDENCE_DECIMALS
        )
    return out

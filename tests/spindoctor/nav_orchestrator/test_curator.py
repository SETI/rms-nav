"""Tests for ``spindoctor.nav_orchestrator.curator.build_metadata_dict``."""

import json
import math

import numpy as np
import pytest

from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.nav_orchestrator.curator import (
    assert_diagnostic_fields_present,
    build_metadata_dict,
)
from spindoctor.nav_orchestrator.feature_summary import NavFeatureSummary
from spindoctor.nav_orchestrator.image_classifier_result import NavImageClassifierResult
from spindoctor.nav_orchestrator.nav_result import NavResult
from spindoctor.nav_orchestrator.provenance import Provenance
from spindoctor.nav_technique.diagnostics import BodyLimbDiagnostics
from spindoctor.nav_technique.technique_result import NavTechniqueResult
from spindoctor.support.status_reason import NavStatusReason


def _classifier() -> NavImageClassifierResult:
    return NavImageClassifierResult(
        image_class='clean',
        saturation_frac=0.0,
        missing_frac=0.0,
        noise_sigma=1.234567,
        max_dn=10.0,
    )


def _provenance() -> Provenance:
    return Provenance(
        spindoctor_version='0.5.2',
        image_et=414504000.123456789,
        pipeline_run_iso8601='2026-04-26T12:00:00Z',
    )


def _ok_result_with_one_technique() -> NavResult:
    cov = np.diag([0.04, 0.16])
    diag = BodyLimbDiagnostics(visible_arc_px=120.0, dt_fit_rms_px=0.456789)
    tech = NavTechniqueResult(
        technique_name='BodyLimbNav',
        feature_ids=('limb_arc:MIMAS',),
        offset_px=(1.234567, 2.345678),
        covariance_px2=cov,
        confidence=0.876543,
        spurious=False,
        at_edge=False,
        diagnostics=diag,
    )
    inv = [
        NavFeatureSummary(
            feature_id='limb_arc:MIMAS',
            feature_type=NavFeatureType.LIMB_ARC,
            source_model='body:MIMAS',
            reliability=0.9,
            gated=False,
            gate_reason=None,
            bbox_extfov_vu=(100, 200, 300, 400),
        ),
    ]
    return NavResult.success(
        offset_px=(1.234567, 2.345678),
        covariance_px2=cov,
        confidence=0.876543,
        confidence_rank='high',
        status_reason=NavStatusReason.OK,
        per_technique=[tech],
        feature_inventory=inv,
        image_classifier=_classifier(),
        provenance=_provenance(),
    )


def test_metadata_dict_contains_top_level_keys() -> None:
    """The metadata dict has the exact required top-level key set."""
    md = build_metadata_dict(_ok_result_with_one_technique())
    expected = {
        'status',
        'status_reason',
        'offset_px',
        'sigma_px',
        'sigma_along_unobservable_px',
        'confidence',
        'confidence_provisional',
        'confidence_rank',
        'covariance_px2',
        'techniques_used',
        'feature_count_by_type',
        'per_technique',
        'feature_inventory',
        'image_classifier',
        'provenance',
    }
    assert set(md) == expected


def test_metadata_dict_rounds_offset_to_4_decimals() -> None:
    """offset_px is rounded to 4 decimals."""
    md = build_metadata_dict(_ok_result_with_one_technique())
    assert md['offset_px'] == [1.2346, 2.3457]


def test_metadata_dict_rounds_confidence_to_3_decimals() -> None:
    """confidence is rounded to 3 decimals."""
    md = build_metadata_dict(_ok_result_with_one_technique())
    assert md['confidence'] == 0.877


def test_metadata_dict_marks_confidence_provisional() -> None:
    """confidence_provisional is present and literally true.

    The marker flags that confidence values and tiers are calibrated
    against simulated planted-truth recovery only and must not be read
    as probabilities of real-image accuracy; it stays true until a
    calibration against real-image error measurements lands.
    """
    md = build_metadata_dict(_ok_result_with_one_technique())
    assert md['confidence_provisional'] is True


def test_metadata_dict_provenance_carries_config_and_catalog_fields() -> None:
    """The provenance block serializes the config hash, overrides, and catalogs."""
    prov = Provenance(
        spindoctor_version='0.5.2',
        image_et=414504000.123456789,
        pipeline_run_iso8601='2026-04-26T12:00:00Z',
        config_hash='cd' * 32,
        config_overrides=('/etc/spindoctor/site.yaml',),
        star_catalogs={'ucac4': 'gs://bucket/UCAC4'},
    )
    result = NavResult.success(
        offset_px=(0.0, 0.0),
        covariance_px2=np.eye(2, dtype=np.float64),
        confidence=0.5,
        confidence_rank='low',
        status_reason=NavStatusReason.OK,
        per_technique=[],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=prov,
    )
    md = build_metadata_dict(result)
    assert md['provenance']['config_hash'] == 'cd' * 32
    assert md['provenance']['config_overrides'] == ['/etc/spindoctor/site.yaml']
    assert md['provenance']['star_catalogs'] == {'ucac4': 'gs://bucket/UCAC4'}


def test_metadata_dict_provenance_config_fields_default_to_empty() -> None:
    """A minimal provenance still emits the new keys (null hash, empty lists)."""
    md = build_metadata_dict(_ok_result_with_one_technique())
    assert md['provenance']['config_hash'] is None
    assert md['provenance']['config_overrides'] == []
    assert md['provenance']['star_catalogs'] == {}


def test_metadata_dict_rounds_image_et_to_6_decimals() -> None:
    """image_et is rounded to 6 decimals."""
    md = build_metadata_dict(_ok_result_with_one_technique())
    assert md['provenance']['image_et'] == 414504000.123457


def test_metadata_dict_is_json_serializable() -> None:
    """The output dict can be JSON-serialized round-trip."""
    md = build_metadata_dict(_ok_result_with_one_technique())
    s = json.dumps(md)
    again = json.loads(s)
    assert again['confidence'] == 0.877


def test_metadata_dict_replaces_inf_with_sentinel() -> None:
    """A rank-1 result's sigma_along_unobservable_px is finite-clamped."""
    cov = np.diag([0.04, 1e10])
    result = NavResult.success(
        offset_px=(0.0, 0.0),
        covariance_px2=cov,
        confidence=0.5,
        confidence_rank='low',
        status_reason=NavStatusReason.RANK_1_ONLY,
        per_technique=[],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
        sigma_along_unobservable_px=math.inf,
    )
    md = build_metadata_dict(result)
    # ``math.inf`` is clamped to the curator's JSON_INF_SENTINEL sentinel.
    from spindoctor.feature.constants import JSON_INF_SENTINEL

    assert md['sigma_along_unobservable_px'] == JSON_INF_SENTINEL


def test_assert_diagnostic_fields_present_passes() -> None:
    """A well-formed NavResult passes the curator allow-list assertion."""
    result = _ok_result_with_one_technique()
    assert_diagnostic_fields_present(result)


def test_assert_diagnostic_fields_present_detects_missing_curator_field() -> None:
    """A diagnostic without CURATOR_FIELDS triggers AssertionError."""
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class BadDiag:
        x: float = 0.0
        # No CURATOR_FIELDS attribute.

    cov = np.eye(2, dtype=np.float64)
    tech = NavTechniqueResult(
        technique_name='BadTechnique',
        feature_ids=(),
        offset_px=(0.0, 0.0),
        covariance_px2=cov,
        confidence=0.5,
        spurious=False,
        at_edge=False,
        diagnostics=BadDiag(),  # type: ignore[arg-type]
    )
    result = NavResult.success(
        offset_px=(0.0, 0.0),
        covariance_px2=cov,
        confidence=0.5,
        confidence_rank='low',
        status_reason=NavStatusReason.OK,
        per_technique=[tech],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    with pytest.raises(AssertionError, match='CURATOR_FIELDS'):
        assert_diagnostic_fields_present(result)


def test_metadata_dict_techniques_used_sorted() -> None:
    """``techniques_used`` matches the per_technique technique names sorted."""
    md = build_metadata_dict(_ok_result_with_one_technique())
    assert md['techniques_used'] == ['BodyLimbNav']

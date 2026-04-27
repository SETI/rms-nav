"""Tests for ``nav.feature.composition.compose_template_features``."""

import numpy as np

from nav.feature.composition import compose_template_features
from nav.feature.feature import NavFeature, NavReliabilityBreakdown
from nav.feature.feature_type import NavFeatureType
from nav.feature.flags import BodyDiscFlags
from nav.feature.geometry import BodyDiscGeometry
from nav.support.filters import NavFilterKind, NavFilterSpec


def _make_body(
    *,
    feature_id: str,
    bbox: tuple[int, int, int, int],
    template_value: float,
    subject_range_km: float,
) -> NavFeature:
    """Build a BODY_DISC feature with a uniform-value template."""
    h = bbox[2] - bbox[0]
    w = bbox[3] - bbox[1]
    template_img = np.full((h, w), template_value, dtype=np.float64)
    template_mask = np.ones((h, w), dtype=bool)
    return NavFeature(
        feature_id=feature_id,
        feature_type=NavFeatureType.BODY_DISC,
        source_model='body',
        geometry=BodyDiscGeometry(
            bbox_extfov_vu=bbox,
            predicted_center_vu=((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
            overflow_fraction=0.0,
        ),
        subject_range_km=subject_range_km,
        position_cov_px=None,
        intensity_sigma_rel=0.0,
        preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
        reliability=1.0,
        reliability_reasons=NavReliabilityBreakdown(visible_lit_fraction=1.0),
        usable_types=frozenset({NavFeatureType.BODY_DISC}),
        flags=BodyDiscFlags(body_name='X', overflow_fov_fraction=0.0),
        template_img=template_img,
        template_mask=template_mask,
    )


def test_compose_empty_features_yields_zero_image() -> None:
    """No template features → zero image and false mask."""
    image, mask = compose_template_features([], (10, 10))
    assert image.shape == (10, 10)
    assert image.sum() == 0.0
    assert not mask.any()


def test_compose_single_feature_paints_template() -> None:
    """A single feature paints its template at the bbox location."""
    feat = _make_body(
        feature_id='body_disc:A',
        bbox=(2, 3, 4, 6),
        template_value=5.0,
        subject_range_km=100.0,
    )
    image, mask = compose_template_features([feat], (10, 10))
    assert image[2, 3] == 5.0
    assert image[3, 5] == 5.0
    assert image[5, 5] == 0.0
    assert mask[2:4, 3:6].all()
    assert not mask[0, 0]


def test_compose_nearer_feature_overwrites_farther() -> None:
    """Z-buffer paint: nearer subject_range_km overwrites farther on overlap."""
    far = _make_body(
        feature_id='body_disc:far',
        bbox=(0, 0, 4, 4),
        template_value=1.0,
        subject_range_km=1000.0,
    )
    near = _make_body(
        feature_id='body_disc:near',
        bbox=(2, 2, 6, 6),
        template_value=9.0,
        subject_range_km=10.0,
    )
    image, _ = compose_template_features([far, near], (10, 10))
    # Far-only region (0:2, 0:2) keeps far's value 1.0.
    assert image[0, 0] == 1.0
    # Overlap region (2:4, 2:4) takes near's value 9.0.
    assert image[2, 2] == 9.0
    assert image[3, 3] == 9.0
    # Near-only region (4:6, 4:6) is near's value.
    assert image[5, 5] == 9.0


def test_compose_input_order_does_not_matter() -> None:
    """The composite is order-independent (sort by range internally)."""
    far = _make_body(
        feature_id='body_disc:far',
        bbox=(0, 0, 4, 4),
        template_value=1.0,
        subject_range_km=1000.0,
    )
    near = _make_body(
        feature_id='body_disc:near',
        bbox=(2, 2, 6, 6),
        template_value=9.0,
        subject_range_km=10.0,
    )
    img1, _ = compose_template_features([far, near], (10, 10))
    img2, _ = compose_template_features([near, far], (10, 10))
    assert np.array_equal(img1, img2)


def test_compose_skips_features_without_templates() -> None:
    """Features without template_img+template_mask are silently skipped."""
    # Polyline-style feature: no template.
    from nav.feature.flags import LimbArcFlags
    from nav.feature.geometry import LimbPolyline

    polyline = NavFeature(
        feature_id='limb_arc:X',
        feature_type=NavFeatureType.LIMB_ARC,
        source_model='body',
        geometry=LimbPolyline(
            vertices_vu=np.array([[1.0, 1.0], [2.0, 2.0]]),
            normals_vu=np.array([[1.0, 0.0], [0.0, 1.0]]),
            sigma_normal_per_vertex_px=np.array([0.5, 0.5]),
            sigma_tangent_per_vertex_px=np.array([0.5, 0.5]),
            bbox_extfov_vu=(0, 0, 5, 5),
        ),
        subject_range_km=100.0,
        position_cov_px=None,
        intensity_sigma_rel=0.0,
        preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
        reliability=0.9,
        reliability_reasons=NavReliabilityBreakdown(visible_arc_fraction=1.0),
        usable_types=frozenset({NavFeatureType.LIMB_ARC}),
        flags=LimbArcFlags(body_name='X', visible_arc_fraction=1.0),
    )
    body = _make_body(
        feature_id='body_disc:X',
        bbox=(0, 0, 4, 4),
        template_value=3.0,
        subject_range_km=100.0,
    )
    image, mask = compose_template_features([polyline, body], (10, 10))
    assert image[0, 0] == 3.0
    assert mask[0, 0]


def test_compose_clamps_bbox_to_extfov() -> None:
    """Templates with bbox extending past the ext-FOV are clamped."""
    feat = _make_body(
        feature_id='body_disc:X',
        bbox=(8, 8, 12, 12),  # extends past 10x10 array
        template_value=4.0,
        subject_range_km=100.0,
    )
    image, mask = compose_template_features([feat], (10, 10))
    # The 2x2 in-bounds region is painted.
    assert image[8, 8] == 4.0
    assert image[9, 9] == 4.0
    assert mask[8, 8]
    assert mask[9, 9]
    # No paint leaks outside the (8:10, 8:10) clipped region.
    assert image[:8, :8].sum() == 0.0
    assert not mask[:8, :8].any()

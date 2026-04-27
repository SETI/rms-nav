"""Tests for ``nav.feature.geometry`` payload variants."""

import numpy as np

from nav.feature.geometry import (
    BodyBlobGeometry,
    BodyDiscGeometry,
    CartographicModelGeometry,
    LimbPolyline,
    RingAnnulusGeometry,
    RingEdgePolyline,
    StarGeometry,
    TerminatorPolyline,
)


def test_star_geometry_fields() -> None:
    """StarGeometry stores predicted, catalog, and bbox tuples."""
    geom = StarGeometry(
        predicted_vu=(100.0, 200.0),
        catalog_vu=(100.5, 200.5),
        bbox_extfov_vu=(95, 195, 105, 205),
    )
    assert geom.predicted_vu == (100.0, 200.0)
    assert geom.bbox_extfov_vu == (95, 195, 105, 205)


def test_limb_polyline_fields_compatible_with_numpy() -> None:
    """LimbPolyline accepts numpy arrays for vertices/normals/sigmas."""
    vertices = np.array([[10.0, 20.0], [11.0, 21.0]])
    normals = np.array([[1.0, 0.0], [0.7, 0.7]])
    sigma_n = np.array([0.5, 0.5])
    sigma_t = np.array([0.5, 0.5])
    poly = LimbPolyline(
        vertices_vu=vertices,
        normals_vu=normals,
        sigma_normal_per_vertex_px=sigma_n,
        sigma_tangent_per_vertex_px=sigma_t,
        bbox_extfov_vu=(10, 20, 12, 22),
    )
    assert poly.vertices_vu.shape == (2, 2)
    assert poly.normals_vu.shape == (2, 2)


def test_terminator_polyline_field_set() -> None:
    """TerminatorPolyline mirrors LimbPolyline fields."""
    poly = TerminatorPolyline(
        vertices_vu=np.zeros((1, 2)),
        normals_vu=np.zeros((1, 2)),
        sigma_normal_per_vertex_px=np.zeros(1),
        sigma_tangent_per_vertex_px=np.zeros(1),
        bbox_extfov_vu=(0, 0, 1, 1),
    )
    assert poly.vertices_vu.shape == (1, 2)


def test_ring_edge_polyline_carries_straight_line_flag() -> None:
    """RingEdgePolyline stores the curvature flag."""
    poly = RingEdgePolyline(
        vertices_vu=np.zeros((1, 2)),
        normals_vu=np.zeros((1, 2)),
        sigma_radial_per_vertex_px=np.zeros(1),
        sigma_along_edge_per_vertex_px=np.zeros(1),
        is_straight_line=True,
        bbox_extfov_vu=(0, 0, 1, 1),
    )
    assert poly.is_straight_line is True


def test_body_disc_geometry_overflow_fraction() -> None:
    """BodyDiscGeometry stores predicted center and overflow_fraction."""
    geom = BodyDiscGeometry(
        bbox_extfov_vu=(0, 0, 100, 100),
        predicted_center_vu=(50.0, 50.0),
        overflow_fraction=0.25,
    )
    assert geom.overflow_fraction == 0.25


def test_body_blob_geometry_predicted_diameter() -> None:
    """BodyBlobGeometry stores the predicted diameter in pixels."""
    geom = BodyBlobGeometry(
        predicted_center_vu=(10.0, 20.0),
        bbox_extfov_vu=(5, 15, 15, 25),
        predicted_diameter_px=8.5,
    )
    assert geom.predicted_diameter_px == 8.5


def test_ring_annulus_geometry_minimal() -> None:
    """RingAnnulusGeometry stores bbox and predicted center."""
    geom = RingAnnulusGeometry(
        bbox_extfov_vu=(0, 0, 100, 100),
        predicted_center_vu=(50.0, 50.0),
    )
    assert geom.predicted_center_vu == (50.0, 50.0)


def test_cartographic_model_geometry_overflow_fraction() -> None:
    """CartographicModelGeometry mirrors BodyDiscGeometry shape."""
    geom = CartographicModelGeometry(
        bbox_extfov_vu=(0, 0, 64, 64),
        predicted_center_vu=(32.0, 32.0),
        overflow_fraction=0.0,
    )
    assert geom.overflow_fraction == 0.0

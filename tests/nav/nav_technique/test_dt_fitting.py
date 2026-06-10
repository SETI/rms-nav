"""Unit tests for the shared distance-transform fitting helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

from nav.nav_orchestrator.image_derivatives import (
    DEFAULT_IMAGE_GRADIENT_SIGMA_PX,
    compute_image_gradient_vu,
)
from nav.nav_technique.dt_fitting import (
    DEFAULT_TUKEY_C,
    LMRefineResult,
    coarse_ncc_search,
    information_matrix_to_covariance,
    lm_subpixel_refine,
    polarity_filter,
    tukey_biweight_weights,
)


def _build_circle_polyline(
    center_vu: tuple[float, float], radius_px: float, n_vertices: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(N, 2)`` vertices and ``(N, 2)`` outward normals on a circle."""
    angles = np.linspace(0.0, 2.0 * np.pi, n_vertices, endpoint=False)
    cv, cu = center_vu
    vs = cv + radius_px * np.sin(angles)
    us = cu + radius_px * np.cos(angles)
    nv = np.sin(angles)
    nu = np.cos(angles)
    vertices = np.stack([vs, us], axis=-1).astype(np.float64)
    normals = np.stack([nv, nu], axis=-1).astype(np.float64)
    return vertices, normals


def _render_circle_mask(
    shape_vu: tuple[int, int],
    center_vu: tuple[float, float],
    radius_px: float,
    thickness_px: float = 1.0,
) -> np.ndarray:
    """Return a boolean mask of an annulus around the given circle."""
    vs, us = np.meshgrid(
        np.arange(shape_vu[0]),
        np.arange(shape_vu[1]),
        indexing='ij',
    )
    cv, cu = center_vu
    rr = np.hypot(vs - cv, us - cu)
    out: np.ndarray = np.abs(rr - radius_px) <= thickness_px
    return out


def _render_image_with_circle(
    shape_vu: tuple[int, int],
    center_vu: tuple[float, float],
    radius_px: float,
    *,
    inside_dn: float = 100.0,
    outside_dn: float = 0.0,
) -> np.ndarray:
    """Return a step-edge disc image (inside_dn inside, outside_dn outside)."""
    vs, us = np.meshgrid(
        np.arange(shape_vu[0]),
        np.arange(shape_vu[1]),
        indexing='ij',
    )
    cv, cu = center_vu
    rr = np.hypot(vs - cv, us - cu)
    image = np.where(rr <= radius_px, inside_dn, outside_dn)
    return image.astype(np.float64)


# ---------------------------------------------------------------------------
# tukey_biweight_weights
# ---------------------------------------------------------------------------


def test_tukey_biweight_weights_returns_one_at_zero_residual() -> None:
    weights = tukey_biweight_weights(np.array([0.0, 0.0, 0.0]))
    assert np.allclose(weights, [1.0, 1.0, 1.0])


def test_tukey_biweight_weights_is_zero_outside_cutoff() -> None:
    c = DEFAULT_TUKEY_C
    weights = tukey_biweight_weights(np.array([c + 0.01, -c - 0.5, 100.0]))
    assert np.array_equal(weights, [0.0, 0.0, 0.0])


def test_tukey_biweight_weights_matches_holland_welsch_at_half_cutoff() -> None:
    c = DEFAULT_TUKEY_C
    r = c / 2.0
    weights = tukey_biweight_weights(np.array([r]))
    expected = (1.0 - 0.25) ** 2
    assert weights[0] == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    ('invalid_c', 'message'),
    [
        (-1.0, 'c must be a positive finite'),
        (0.0, 'c must be a positive finite'),
        (float('inf'), 'c must be a positive finite'),
        (float('nan'), 'c must be a positive finite'),
    ],
)
def test_tukey_biweight_weights_rejects_invalid_c(invalid_c: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        tukey_biweight_weights(np.array([1.0]), c=invalid_c)


@pytest.mark.parametrize('shape', [(2, 2), (3, 3, 3), (1, 4)])
def test_tukey_biweight_weights_rejects_non_1d_input(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match='residuals must be 1-D'):
        tukey_biweight_weights(np.zeros(shape))


# ---------------------------------------------------------------------------
# information_matrix_to_covariance
# ---------------------------------------------------------------------------


def test_information_matrix_to_covariance_recovers_identity() -> None:
    jacobian = np.eye(2, dtype=np.float64)
    weights = np.array([1.0, 1.0])
    cov = information_matrix_to_covariance(jacobian, weights)
    assert np.allclose(cov, np.eye(2), atol=1e-12)


def test_information_matrix_to_covariance_handles_rank_one_input() -> None:
    # All Jacobian rows along axis 0: only "v" axis is observable.
    jacobian = np.zeros((10, 2), dtype=np.float64)
    jacobian[:, 0] = 1.0
    weights = np.ones(10)
    cov = information_matrix_to_covariance(jacobian, weights)
    eigvals = np.linalg.eigvalsh(cov)
    null_eigval = float(eigvals.min())
    observed_eigval = float(eigvals.max())
    assert abs(null_eigval) < 1.0e-12
    assert observed_eigval == pytest.approx(0.1, rel=1e-9)


@pytest.mark.parametrize(
    ('jacobian', 'weights', 'message'),
    [
        # Negative weight: rejected by the non-negativity guard.
        (np.eye(2), np.array([1.0, -0.5]), 'weights must be non-negative'),
        # Wrong-rank weight vector: caught by the 1-D shape guard.
        (np.eye(3), np.ones(2), 'must be a 1-D vector'),
        # 1-D Jacobian: caught by the 2-D shape guard.
        (np.zeros(3), np.ones(3), 'jacobian must be 2-D'),
    ],
)
def test_information_matrix_to_covariance_rejects_invalid_inputs(
    jacobian: np.ndarray, weights: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        information_matrix_to_covariance(jacobian, weights)


# ---------------------------------------------------------------------------
# coarse_ncc_search
# ---------------------------------------------------------------------------


def test_coarse_ncc_search_recovers_planted_integer_offset() -> None:
    shape = (64, 64)
    polyline_mask = _render_circle_mask(shape, (32.0, 32.0), 12.0)
    edge_mask = np.roll(polyline_mask, shift=(3, -2), axis=(0, 1))
    dv, du = coarse_ncc_search(edge_mask, polyline_mask, (10, 10))
    assert (dv, du) == (3, -2)


def test_coarse_ncc_search_returns_zero_for_aligned_inputs() -> None:
    shape = (32, 32)
    polyline_mask = _render_circle_mask(shape, (16.0, 16.0), 8.0)
    dv, du = coarse_ncc_search(polyline_mask, polyline_mask, (4, 4))
    assert (dv, du) == (0, 0)


def test_coarse_ncc_search_returns_zero_with_empty_polyline() -> None:
    edge_mask = np.zeros((16, 16), dtype=bool)
    edge_mask[8, 8] = True
    polyline_mask = np.zeros_like(edge_mask)
    dv, du = coarse_ncc_search(edge_mask, polyline_mask, (3, 3))
    assert (dv, du) == (0, 0)


@pytest.mark.parametrize(
    ('edge_mask', 'polyline_mask', 'window', 'message'),
    [
        # Shape mismatch between the two masks.
        (
            np.zeros((4, 4), bool),
            np.zeros((5, 5), bool),
            (1, 1),
            'shape mismatch',
        ),
        # 1-D edge mask: caught by the 2-D shape guard.
        (np.zeros(4, bool), np.zeros((4, 4), bool), (1, 1), 'must be 2-D'),
        # Negative window margin: caught by the non-negative guard.
        (
            np.zeros((4, 4), bool),
            np.zeros((4, 4), bool),
            (-1, 1),
            'must be non-negative',
        ),
        (
            np.zeros((4, 4), bool),
            np.zeros((4, 4), bool),
            (1, -1),
            'must be non-negative',
        ),
        # Wrong-length window: caught by the length-2 guard.
        (
            np.zeros((4, 4), bool),
            np.zeros((4, 4), bool),
            (1, 2, 3),
            'length-2 sequence of ints',
        ),
        (
            np.zeros((4, 4), bool),
            np.zeros((4, 4), bool),
            (1,),
            'length-2 sequence of ints',
        ),
    ],
)
def test_coarse_ncc_search_rejects_invalid_inputs(
    edge_mask: np.ndarray,
    polyline_mask: np.ndarray,
    window: tuple[int, ...],
    message: str,
) -> None:
    """Invalid mask shapes or window tuples are rejected with a named message."""
    with pytest.raises(ValueError, match=message):
        coarse_ncc_search(edge_mask, polyline_mask, window)  # type: ignore[arg-type]


def test_coarse_ncc_search_rejects_float_window_entry() -> None:
    """A float window entry is rejected with TypeError instead of being truncated."""
    edge_mask = np.zeros((4, 4), bool)
    polyline_mask = np.zeros((4, 4), bool)
    with pytest.raises(TypeError, match='search_window_vu\\[0\\] must be int'):
        coarse_ncc_search(edge_mask, polyline_mask, (1.5, 1))  # type: ignore[arg-type]


def test_coarse_ncc_search_rejects_non_sequence_window() -> None:
    """A non-tuple/list window is rejected by the length-2 sequence guard."""
    edge_mask = np.zeros((4, 4), bool)
    polyline_mask = np.zeros((4, 4), bool)
    with pytest.raises(ValueError, match='length-2 sequence of ints'):
        coarse_ncc_search(edge_mask, polyline_mask, np.array([1, 1]))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# polarity_filter
# ---------------------------------------------------------------------------


def test_polarity_filter_accepts_normals_aligned_with_image_gradient() -> None:
    shape = (32, 32)
    image = _render_image_with_circle(shape, (16.0, 16.0), 8.0)
    grad = compute_image_gradient_vu(image, sigma_px=DEFAULT_IMAGE_GRADIENT_SIGMA_PX)
    vertices, outward_normals = _build_circle_polyline((16.0, 16.0), 8.0, 16)
    # Bright-disc on dark sky: the image gradient at the limb points INTO the
    # body (low DN to high DN).  ``polarity_filter`` tests strict
    # ``model_dir . image_gradient > 0``, so the model must supply the inward
    # direction (the negation of the geometric outward normal) to be accepted.
    inward_normals = -outward_normals
    keep = polarity_filter(vertices, inward_normals, grad)
    assert keep.sum() == 16


def test_polarity_filter_rejects_normals_opposing_image_gradient() -> None:
    shape = (32, 32)
    image = _render_image_with_circle(shape, (16.0, 16.0), 8.0)
    grad = compute_image_gradient_vu(image, sigma_px=DEFAULT_IMAGE_GRADIENT_SIGMA_PX)
    vertices, outward_normals = _build_circle_polyline((16.0, 16.0), 8.0, 16)
    keep = polarity_filter(vertices, outward_normals, grad)
    assert keep.sum() == 0


def test_polarity_filter_per_vertex_decision() -> None:
    shape = (32, 32)
    image = _render_image_with_circle(shape, (16.0, 16.0), 8.0)
    grad = compute_image_gradient_vu(image, sigma_px=DEFAULT_IMAGE_GRADIENT_SIGMA_PX)
    vertices, outward_normals = _build_circle_polyline((16.0, 16.0), 8.0, 16)
    inward_normals = -outward_normals
    # Half the vertices kept aligned (inward), half flipped to outward.
    mixed_normals = inward_normals.copy()
    mixed_normals[::2] = outward_normals[::2]
    keep = polarity_filter(vertices, mixed_normals, grad)
    assert keep.sum() == 8


def test_polarity_filter_rejects_misshaped_inputs() -> None:
    grad = np.zeros((4, 4, 2))
    with pytest.raises(ValueError, match='vertices_vu must have shape'):
        polarity_filter(np.zeros(2), np.zeros(2), grad)


def test_polarity_filter_rejects_2d_gradient() -> None:
    with pytest.raises(ValueError, match='image_gradient_vu must have shape'):
        polarity_filter(np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((4, 4)))


# ---------------------------------------------------------------------------
# lm_subpixel_refine — translation-only
# ---------------------------------------------------------------------------


def _build_dt_for_circle(shape: tuple[int, int], radius: float) -> np.ndarray:
    cv = shape[0] / 2.0
    cu = shape[1] / 2.0
    edge_mask = _render_circle_mask(shape, (cv, cu), radius, thickness_px=0.5)
    raw = distance_transform_edt(~edge_mask)
    dt: np.ndarray = np.asarray(raw, dtype=np.float64)
    return dt


def test_lm_subpixel_refine_recovers_subpixel_translation() -> None:
    shape = (96, 96)
    radius = 18.0
    dt = _build_dt_for_circle(shape, radius)
    cv = shape[0] / 2.0 + 1.5
    cu = shape[1] / 2.0 + 2.5
    vertices, outward_normals = _build_circle_polyline((cv, cu), radius, 64)
    inward_normals = -outward_normals
    sigmas = np.full(vertices.shape[0], 0.5, dtype=np.float64)
    image = _render_image_with_circle(shape, (shape[0] / 2.0, shape[1] / 2.0), radius)
    grad = compute_image_gradient_vu(image, sigma_px=DEFAULT_IMAGE_GRADIENT_SIGMA_PX)
    result = lm_subpixel_refine(
        vertices_vu=vertices,
        normals_vu=inward_normals,
        sigma_normal_per_vertex_px=sigmas,
        image_edge_dt=dt,
        image_gradient_vu=grad,
        initial_offset_vu=(-1.0, -2.0),
        use_polarity=True,
    )
    assert result.offset_vu[0] == pytest.approx(-1.5, abs=0.05)
    assert result.offset_vu[1] == pytest.approx(-2.5, abs=0.05)
    assert result.iterations >= 1
    assert result.inlier_count == 64


def test_lm_subpixel_refine_trust_region_caps_offset_displacement() -> None:
    """The trust-region kwarg physically prevents the LM from leaving the seed.

    Plant a circle at (-1.5, -2.5) but seed the LM at (5, 5) — well
    outside the planted basin.  Without a trust region, the LM either
    walks back toward the truth or, on noisy data, walks to an
    unrelated DT minimum.  With a 1.0-px trust region the converged
    offset is constrained to ``hypot(dv-5, du-5) <= 1.0``: the LM can
    refine inside the trust radius but cannot escape it.
    """
    shape = (96, 96)
    radius = 18.0
    dt = _build_dt_for_circle(shape, radius)
    cv = shape[0] / 2.0 + 1.5
    cu = shape[1] / 2.0 + 2.5
    vertices, outward_normals = _build_circle_polyline((cv, cu), radius, 64)
    inward_normals = -outward_normals
    sigmas = np.full(vertices.shape[0], 0.5, dtype=np.float64)
    image = _render_image_with_circle(shape, (shape[0] / 2.0, shape[1] / 2.0), radius)
    grad = compute_image_gradient_vu(image, sigma_px=DEFAULT_IMAGE_GRADIENT_SIGMA_PX)
    initial = (5.0, 5.0)
    trust_region = 1.0
    result = lm_subpixel_refine(
        vertices_vu=vertices,
        normals_vu=inward_normals,
        sigma_normal_per_vertex_px=sigmas,
        image_edge_dt=dt,
        image_gradient_vu=grad,
        initial_offset_vu=initial,
        use_polarity=True,
        trust_region_px=trust_region,
    )
    displacement = float(
        np.hypot(result.offset_vu[0] - initial[0], result.offset_vu[1] - initial[1])
    )
    # The displacement is bounded by the trust radius (with a small
    # tolerance for the final commit; the LM may accept a step
    # exactly at the boundary).
    assert displacement <= trust_region + 1.0e-6


def test_lm_subpixel_refine_rejects_outliers_via_tukey() -> None:
    shape = (96, 96)
    radius = 18.0
    dt = _build_dt_for_circle(shape, radius)
    cv = shape[0] / 2.0
    cu = shape[1] / 2.0
    vertices, normals = _build_circle_polyline((cv, cu), radius, 100)
    # Move 10 % of the vertices far away from the actual circle: they should
    # be Tukey-rejected so the recovered offset is dominated by the inliers.
    bad = np.arange(0, 100, 10)
    vertices[bad, 0] += 30.0
    sigmas = np.full(vertices.shape[0], 0.5, dtype=np.float64)
    image = _render_image_with_circle(shape, (cv, cu), radius)
    grad = compute_image_gradient_vu(image, sigma_px=DEFAULT_IMAGE_GRADIENT_SIGMA_PX)
    result = lm_subpixel_refine(
        vertices_vu=vertices,
        normals_vu=normals,
        sigma_normal_per_vertex_px=sigmas,
        image_edge_dt=dt,
        image_gradient_vu=grad,
        initial_offset_vu=(0.0, 0.0),
        use_polarity=False,
    )
    assert result.offset_vu[0] == pytest.approx(0.0, abs=0.1)
    assert result.offset_vu[1] == pytest.approx(0.0, abs=0.1)
    # Outlier vertices should have zero (or near-zero) Tukey weight.
    assert float(result.weights[bad].max()) < 1.0e-6
    # In-lier vertices should retain weight of 1 / sigma**2 = 4 (with sigma=0.5)
    inlier_idx = np.setdiff1d(np.arange(100), bad)
    assert float(result.weights[inlier_idx].min()) > 1.0


def test_lm_subpixel_refine_degenerate_when_all_vertices_rejected() -> None:
    """A fit with no surviving inliers reports +inf RMS and inf covariance.

    With a zero gradient image every polarity dot product is zero (not
    strictly positive), so the polarity filter rejects every vertex.
    Each rejected vertex gets the infinity penalty, the Tukey biweight
    zeroes its weight, and no evidence remains to constrain the fit.  The
    result must advertise this honestly: ``rms_px`` is ``+inf`` (not the
    misleading ``0.0`` that downstream spurious gates would read as a
    perfect fit), ``degenerate`` is True, and the covariance is all-inf.
    """
    shape = (96, 96)
    radius = 18.0
    dt = _build_dt_for_circle(shape, radius)
    cv = shape[0] / 2.0
    cu = shape[1] / 2.0
    vertices, outward_normals = _build_circle_polyline((cv, cu), radius, 64)
    sigmas = np.full(vertices.shape[0], 0.5, dtype=np.float64)
    zero_grad = np.zeros((*shape, 2), dtype=np.float64)
    result = lm_subpixel_refine(
        vertices_vu=vertices,
        normals_vu=outward_normals,
        sigma_normal_per_vertex_px=sigmas,
        image_edge_dt=dt,
        image_gradient_vu=zero_grad,
        initial_offset_vu=(0.0, 0.0),
        use_polarity=True,
    )
    assert result.inlier_count == 0
    assert result.degenerate is True
    assert result.rms_px == float('inf')
    assert np.isinf(result.covariance).all()


# ---------------------------------------------------------------------------
# lm_subpixel_refine — translation + rotation
# ---------------------------------------------------------------------------


def test_lm_subpixel_refine_recovers_planted_rotation_and_translation() -> None:
    shape = (200, 200)
    cv = shape[0] / 2.0
    cu = shape[1] / 2.0
    # Use a four-arm cross template: well-constrained for translation in both
    # axes and for in-plane rotation about the cross centre.
    arm_length_px = 60.0
    arm_density = 60
    arm_offsets_along = np.linspace(8.0, arm_length_px, arm_density)
    east_v = np.full_like(arm_offsets_along, cv)
    east_u = cu + arm_offsets_along
    west_v = np.full_like(arm_offsets_along, cv)
    west_u = cu - arm_offsets_along
    north_v = cv - arm_offsets_along
    north_u = np.full_like(arm_offsets_along, cu)
    south_v = cv + arm_offsets_along
    south_u = np.full_like(arm_offsets_along, cu)
    vs = np.concatenate([east_v, west_v, north_v, south_v])
    us = np.concatenate([east_u, west_u, north_u, south_u])
    vertices = np.stack([vs, us], axis=-1)
    # The normals are placeholders here (``use_polarity=False`` below).
    normals = np.zeros_like(vertices)
    edge_mask = np.zeros(shape, dtype=bool)
    iv = np.rint(vs).astype(int)
    iu = np.rint(us).astype(int)
    edge_mask[iv, iu] = True
    dt = distance_transform_edt(~edge_mask).astype(np.float64)
    theta_true = 0.04
    dv_true = -1.2
    du_true = 0.7
    pivot = (cv, cu)
    cos_t = math.cos(theta_true)
    sin_t = math.sin(theta_true)
    rot_v = pivot[0] + cos_t * (vertices[:, 0] - pivot[0]) - sin_t * (vertices[:, 1] - pivot[1])
    rot_u = pivot[1] + sin_t * (vertices[:, 0] - pivot[0]) + cos_t * (vertices[:, 1] - pivot[1])
    misaligned_vertices = np.stack([rot_v + dv_true, rot_u + du_true], axis=-1)
    sigmas = np.full(misaligned_vertices.shape[0], 0.5, dtype=np.float64)
    result = lm_subpixel_refine(
        vertices_vu=misaligned_vertices,
        normals_vu=normals,
        sigma_normal_per_vertex_px=sigmas,
        image_edge_dt=dt,
        initial_offset_vu=(0.0, 0.0),
        initial_rotation_rad=0.0,
        fit_rotation=True,
        pivot_vu=pivot,
        pivot_distance_px=arm_length_px,
        use_polarity=False,
    )
    # The undoing parameters: dtheta_lm = -theta_true; the LM translation is
    # ``-R(dtheta_lm) (dv_true, du_true)``.
    expected_dtheta = -theta_true
    expected_cos = math.cos(expected_dtheta)
    expected_sin = math.sin(expected_dtheta)
    expected_dv_lm = -(expected_cos * dv_true - expected_sin * du_true)
    expected_du_lm = -(expected_sin * dv_true + expected_cos * du_true)
    assert result.offset_vu[0] == pytest.approx(expected_dv_lm, abs=0.05)
    assert result.offset_vu[1] == pytest.approx(expected_du_lm, abs=0.05)
    assert result.rotation_rad == pytest.approx(expected_dtheta, abs=2.0e-3)


def test_lm_subpixel_refine_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match='must have shape'):
        lm_subpixel_refine(
            vertices_vu=np.zeros((4,)),
            normals_vu=np.zeros((4, 2)),
            sigma_normal_per_vertex_px=np.ones(4),
            image_edge_dt=np.zeros((8, 8)),
        )


def test_lm_subpixel_refine_rejects_non_positive_sigma() -> None:
    with pytest.raises(ValueError, match='must be finite and > 0'):
        lm_subpixel_refine(
            vertices_vu=np.zeros((1, 2)),
            normals_vu=np.zeros((1, 2)),
            sigma_normal_per_vertex_px=np.array([0.0]),
            image_edge_dt=np.zeros((4, 4)),
        )


def test_lm_subpixel_refine_requires_pivot_distance_for_rotation() -> None:
    with pytest.raises(ValueError, match='pivot_distance_px > 0'):
        lm_subpixel_refine(
            vertices_vu=np.zeros((1, 2)),
            normals_vu=np.zeros((1, 2)),
            sigma_normal_per_vertex_px=np.array([1.0]),
            image_edge_dt=np.zeros((4, 4)),
            fit_rotation=True,
        )


def test_lm_refine_result_freezes_arrays() -> None:
    shape = (32, 32)
    dt = np.full(shape, 5.0, dtype=np.float64)
    vertices = np.array([[16.0, 16.0]], dtype=np.float64)
    normals = np.array([[1.0, 0.0]], dtype=np.float64)
    sigmas = np.array([1.0], dtype=np.float64)
    result = lm_subpixel_refine(
        vertices_vu=vertices,
        normals_vu=normals,
        sigma_normal_per_vertex_px=sigmas,
        image_edge_dt=dt,
        use_polarity=False,
    )
    assert isinstance(result, LMRefineResult)
    assert not result.covariance.flags.writeable
    assert not result.weights.flags.writeable
    assert not result.residuals_px.flags.writeable


# ---------------------------------------------------------------------------
# Coverage-completion tests for dt_fitting helpers
# ---------------------------------------------------------------------------


def test_tukey_biweight_weights_zero_at_exact_cutoff() -> None:
    """Holland-Welsch is half-open: ``|r| == c`` evaluates to weight zero.

    The implementation uses ``np.abs(scaled) <= 1.0`` to gate the
    polynomial, so the boundary itself is *kept* but the polynomial
    ``(1 - 1**2) ** 2`` is exactly zero.  Verify both halves of the
    statement: weight is zero, no off-by-one excludes the boundary.
    """
    c = DEFAULT_TUKEY_C
    weights = tukey_biweight_weights(np.array([c, -c]), c=c)
    assert float(weights[0]) == 0.0
    assert float(weights[1]) == 0.0


def test_coarse_ncc_search_zero_window_returns_origin() -> None:
    """A search window of ``(0, 0)`` means only the origin shift is scanned.

    Off-by-one safety: the implementation iterates
    ``range(-margin_v, margin_v + 1)`` which yields a single ``0`` step.
    The function must return ``(0, 0)`` for any inputs without raising.
    """
    edge_mask = np.zeros((8, 8), dtype=bool)
    edge_mask[3, 3] = True
    polyline_mask = np.zeros((8, 8), dtype=bool)
    polyline_mask[3, 3] = True
    dv, du = coarse_ncc_search(edge_mask, polyline_mask, (0, 0))
    assert (dv, du) == (0, 0)


def test_coarse_ncc_search_unit_window_visits_each_axis_step() -> None:
    """A window of ``(1, 1)`` must cover -1, 0, +1 in each axis.

    Plant a single edge pixel at the boundary of the (1, 1) window and
    verify the function reports that exact integer offset (which means
    the iteration scanned all nine cells, not just the eight non-origin
    ones).
    """
    edge_mask = np.zeros((8, 8), dtype=bool)
    edge_mask[4, 4] = True
    polyline_mask = np.zeros((8, 8), dtype=bool)
    polyline_mask[3, 3] = True
    dv, du = coarse_ncc_search(edge_mask, polyline_mask, (1, 1))
    assert (dv, du) == (1, 1)


def test_lm_subpixel_refine_requires_image_gradient_when_polarity_enabled() -> None:
    """Polarity-enabled LM must reject a ``None`` gradient input.

    The validation path returns a clear ``ValueError`` so a caller that
    forgot to populate ``NavContext.image_gradient_vu_ext`` learns about
    the omission immediately rather than via a silent NoneType crash
    inside the polarity filter.
    """
    with pytest.raises(ValueError, match='use_polarity=True requires image_gradient_vu'):
        lm_subpixel_refine(
            vertices_vu=np.zeros((1, 2)),
            normals_vu=np.zeros((1, 2)),
            sigma_normal_per_vertex_px=np.array([1.0]),
            image_edge_dt=np.zeros((4, 4)),
            use_polarity=True,
            image_gradient_vu=None,
        )


def test_lm_subpixel_refine_bails_out_when_damping_saturates() -> None:
    """The LM loop has an early exit when ``lambda`` saturates at ``1.0e6``.

    With a flat (constant-valued) DT the cost function has zero gradient
    and zero Hessian; every trial step fails to reduce cost; the damping
    factor ramps up multiplicatively until it crosses ``1.0e6`` and the
    loop terminates without convergence.  This exercise covers the
    ``if lambda_ >= 1.0e6: break`` branch in the LM iteration body.
    """
    shape = (16, 16)
    flat_dt = np.full(shape, 5.0, dtype=np.float64)
    vertices = np.array([[8.0, 8.0]], dtype=np.float64)
    normals = np.array([[1.0, 0.0]], dtype=np.float64)
    sigmas = np.array([1.0], dtype=np.float64)
    result = lm_subpixel_refine(
        vertices_vu=vertices,
        normals_vu=normals,
        sigma_normal_per_vertex_px=sigmas,
        image_edge_dt=flat_dt,
        use_polarity=False,
        max_iterations=50,
    )
    # The loop terminates without converging on a flat DT.
    assert result.converged is False
    # The offset stays at the initial value (no successful step reduced cost).
    assert result.offset_vu == (0.0, 0.0)

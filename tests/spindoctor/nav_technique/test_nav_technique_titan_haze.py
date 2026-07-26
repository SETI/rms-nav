"""Tests for ``spindoctor.nav_technique.nav_technique_titan_haze.TitanHazeNav``.

The technique is a thin wrapper around the pure fitting library, so these
tests exercise the wrapping: feasibility, the config-driven tuning load, the
assembled offset and its anisotropic covariance, the gate-to-spurious
mapping, and the result-level at-edge rule that per-pass flags cannot
express.  Recovery bounds match the fitting library's noisy case (0.15 px
cross-track, 1.5 px along-track) because the rendered scene is the same one.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from tests.spindoctor.nav_technique.conftest import (
    HazeDiscImageFactory,
    NavContextFactory,
    NavFeatureFactory,
)

from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.nav_technique.diagnostics import TitanHazeDiagnostics
from spindoctor.nav_technique.nav_technique import NavTechnique
from spindoctor.nav_technique.nav_technique_titan_haze import TitanHazeNav
from spindoctor.nav_technique.technique_result import NavTechniqueResult
from spindoctor.nav_technique.titan_fitting import axis_vectors

SHAPE_VU = (170, 170)
CENTER_VU = (85.0, 85.0)
R_LIMB_PX = 44.0
R_SOLID_PX = 40.0
R_ENV_PX = 46.0
WINDOW_PX = 10
CASE2_CROSS_PX = 0.15
CASE2_ALONG_PX = 1.50


def _displaced(offset_vu: tuple[float, float]) -> tuple[float, float]:
    """Return the true disc center for a planted offset from the prediction."""
    return (CENTER_VU[0] + offset_vu[0], CENTER_VU[1] + offset_vu[1])


def _run_technique(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
    *,
    offset_vu: tuple[float, float],
    theta_rad: float = 0.0,
    window_px: int = WINDOW_PX,
) -> NavTechniqueResult:
    """Render a displaced haze disc and navigate it through the full interface."""
    image = haze_disc_image(SHAPE_VU, _displaced(offset_vu), R_LIMB_PX, theta_rad)
    context = make_nav_context(image, extfov_margin_vu=(window_px, window_px))
    feature = make_titan_feature(
        predicted_center_vu=CENTER_VU,
        r_solid_px=R_SOLID_PX,
        r_env_px=R_ENV_PX,
        sun_angle_rad=theta_rad,
    )
    return TitanHazeNav().navigate([feature], context)


# ---------------------------------------------------------------------------
# Registration and configuration
# ---------------------------------------------------------------------------


def test_technique_is_registered() -> None:
    """The technique self-registers so the orchestrator can discover it."""
    assert TitanHazeNav in NavTechnique._registry


def test_technique_accepts_only_titan_limb() -> None:
    """The technique consumes TITAN_LIMB features and nothing else."""
    assert TitanHazeNav.accepts_feature_types == frozenset({NavFeatureType.TITAN_LIMB})


def test_technique_is_primary_tier() -> None:
    """A hazy body has no second estimator, so the technique is primary."""
    assert TitanHazeNav.tier == 'primary'


def test_technique_runs_without_a_prior() -> None:
    """The technique runs in pass 1."""
    assert TitanHazeNav.requires_prior is False


def test_confidence_spec_loads_from_config() -> None:
    """The confidence spec comes from config_510, not from hardcoded terms."""
    technique = TitanHazeNav()
    assert technique.confidence_spec is not None


def test_confidence_terms_are_declared_attributes() -> None:
    """Every confidence term names an attribute the technique declares."""
    technique = TitanHazeNav()
    assert technique.confidence_spec is not None
    referenced = {term.feature for term in technique.confidence_spec.terms}
    assert referenced <= TitanHazeNav.confidence_attributes


def test_model_error_floor_is_loaded() -> None:
    """The covariance model-error floor is a configured tunable."""
    assert TitanHazeNav.tuning['model_error_floor_px'] > 0.0


# ---------------------------------------------------------------------------
# Feasibility
# ---------------------------------------------------------------------------


def test_is_infeasible_without_titan_features(make_star_feature: NavFeatureFactory) -> None:
    """A frame with no haze feature is infeasible."""
    technique = TitanHazeNav()
    star = make_star_feature('star:test:1', predicted_vu=(10.0, 10.0), predicted_snr=20.0)
    assert technique.is_feasible([star]).feasible is False


def test_infeasibility_reason_names_the_missing_type(
    make_star_feature: NavFeatureFactory,
) -> None:
    """The infeasibility reason names the missing feature type."""
    technique = TitanHazeNav()
    star = make_star_feature('star:test:1', predicted_vu=(10.0, 10.0), predicted_snr=20.0)
    assert technique.is_feasible([star]).reason == 'no TITAN_LIMB features'


def test_is_feasible_with_one_titan_feature(make_titan_feature: NavFeatureFactory) -> None:
    """One haze feature makes the technique feasible."""
    technique = TitanHazeNav()
    feature = make_titan_feature(
        predicted_center_vu=CENTER_VU, r_solid_px=R_SOLID_PX, r_env_px=R_ENV_PX
    )
    assert technique.is_feasible([feature]).feasible is True


def test_feasibility_reports_the_consumed_count(make_titan_feature: NavFeatureFactory) -> None:
    """The feasibility report counts the features it would consume."""
    technique = TitanHazeNav()
    feature = make_titan_feature(
        predicted_center_vu=CENTER_VU, r_solid_px=R_SOLID_PX, r_env_px=R_ENV_PX
    )
    assert technique.is_feasible([feature]).consumed_feature_count == 1


# ---------------------------------------------------------------------------
# End-to-end recovery through the NavTechnique interface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('offset_vu', 'theta_rad'),
    [
        ((0.3, -0.4), 0.0),
        ((-1.2, 0.8), 0.7),
        ((2.5, 1.5), 1.9),
    ],
)
def test_planted_offset_recovered_cross_track(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
    offset_vu: tuple[float, float],
    theta_rad: float,
) -> None:
    """The cross-track component is recovered within the fitting library's bound."""
    result = _run_technique(
        haze_disc_image,
        make_nav_context,
        make_titan_feature,
        offset_vu=offset_vu,
        theta_rad=theta_rad,
    )
    c_hat, _ = axis_vectors(theta_rad)
    error = np.asarray(result.offset_px) - np.asarray(offset_vu)
    assert abs(float(error @ c_hat)) <= CASE2_CROSS_PX


@pytest.mark.parametrize(
    ('offset_vu', 'theta_rad'),
    [
        ((0.3, -0.4), 0.0),
        ((-1.2, 0.8), 0.7),
        ((2.5, 1.5), 1.9),
    ],
)
def test_planted_offset_recovered_along_track(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
    offset_vu: tuple[float, float],
    theta_rad: float,
) -> None:
    """The along-track component is recovered within the fitting library's bound."""
    result = _run_technique(
        haze_disc_image,
        make_nav_context,
        make_titan_feature,
        offset_vu=offset_vu,
        theta_rad=theta_rad,
    )
    _, a_hat = axis_vectors(theta_rad)
    error = np.asarray(result.offset_px) - np.asarray(offset_vu)
    assert abs(float(error @ a_hat)) <= CASE2_ALONG_PX


def test_clean_scene_is_not_spurious(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
) -> None:
    """A clean rendered haze disc navigates rather than reporting spurious."""
    result = _run_technique(
        haze_disc_image, make_nav_context, make_titan_feature, offset_vu=(0.3, -0.4)
    )
    assert result.spurious is False


def test_clean_scene_reports_no_failed_gate(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
) -> None:
    """No gate fires on a clean scene."""
    result = _run_technique(
        haze_disc_image, make_nav_context, make_titan_feature, offset_vu=(0.3, -0.4)
    )
    assert isinstance(result.diagnostics, TitanHazeDiagnostics)
    assert result.diagnostics.gate_failed is None


def test_result_names_the_source_body(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
) -> None:
    """The result attributes itself to the hazy body it consumed."""
    result = _run_technique(
        haze_disc_image, make_nav_context, make_titan_feature, offset_vu=(0.3, -0.4)
    )
    assert result.source_bodies == frozenset({'TITAN'})


def test_result_reports_no_rotation(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
) -> None:
    """Rotation is unobservable from one quasi-circular feature."""
    result = _run_technique(
        haze_disc_image, make_nav_context, make_titan_feature, offset_vu=(0.3, -0.4)
    )
    assert result.rotation_rad is None


def test_rotation_fitting_instrument_gets_a_rank_deficient_covariance(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
) -> None:
    """On a rotation-fitting instrument the covariance is the 3x3 unobservable form.

    Every result in an ensemble group must share the same degrees of
    freedom, so a technique that carries no rotation evidence declares that
    through the sentinel rather than by shipping a smaller matrix.
    """
    image = haze_disc_image(SHAPE_VU, _displaced((0.3, -0.4)), R_LIMB_PX, 0.0)
    context = make_nav_context(
        image, extfov_margin_vu=(WINDOW_PX, WINDOW_PX), fit_camera_rotation=True
    )
    feature = make_titan_feature(
        predicted_center_vu=CENTER_VU, r_solid_px=R_SOLID_PX, r_env_px=R_ENV_PX
    )
    result = TitanHazeNav().navigate([feature], context)
    assert result.covariance_px2.shape == (3, 3)


# ---------------------------------------------------------------------------
# Covariance orientation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('theta_rad', [0.0, 0.7, 1.9, -2.4])
def test_covariance_major_axis_follows_the_symmetry_axis(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
    theta_rad: float,
) -> None:
    """The largest-variance direction lies along the symmetry axis.

    The mirror-symmetry scan localizes across the axis far better than the
    limb-arc fit localizes along it, so the error ellipse must be elongated
    along ``a_hat`` -- within 5 degrees of it.
    """
    result = _run_technique(
        haze_disc_image,
        make_nav_context,
        make_titan_feature,
        offset_vu=(0.3, -0.4),
        theta_rad=theta_rad,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(result.covariance_px2)[:2, :2])
    major = eigenvectors[:, int(np.argmax(eigenvalues))]
    _, a_hat = axis_vectors(theta_rad)
    cosine = abs(float(major @ a_hat))
    assert math.degrees(math.acos(min(cosine, 1.0))) <= 5.0


def test_covariance_is_anisotropic(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
) -> None:
    """Along-track uncertainty genuinely exceeds cross-track uncertainty."""
    result = _run_technique(
        haze_disc_image, make_nav_context, make_titan_feature, offset_vu=(0.3, -0.4)
    )
    eigenvalues = np.linalg.eigvalsh(np.asarray(result.covariance_px2)[:2, :2])
    assert float(eigenvalues[1]) > float(eigenvalues[0])


# ---------------------------------------------------------------------------
# Gates and edge flags
# ---------------------------------------------------------------------------


def test_noise_only_scene_is_spurious(
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
) -> None:
    """A frame with no hazy body at all is rejected, not fitted."""
    rng = np.random.default_rng(seed=7)
    image = rng.standard_normal(SHAPE_VU) * 10.0 + 100.0
    context = make_nav_context(image, extfov_margin_vu=(WINDOW_PX, WINDOW_PX))
    feature = make_titan_feature(
        predicted_center_vu=CENTER_VU, r_solid_px=R_SOLID_PX, r_env_px=R_ENV_PX
    )
    result = TitanHazeNav().navigate([feature], context)
    assert result.spurious is True


def test_noise_only_scene_names_a_gate(
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
) -> None:
    """The rejection is attributed to a specific named fit gate."""
    rng = np.random.default_rng(seed=7)
    image = rng.standard_normal(SHAPE_VU) * 10.0 + 100.0
    context = make_nav_context(image, extfov_margin_vu=(WINDOW_PX, WINDOW_PX))
    feature = make_titan_feature(
        predicted_center_vu=CENTER_VU, r_solid_px=R_SOLID_PX, r_env_px=R_ENV_PX
    )
    result = TitanHazeNav().navigate([feature], context)
    assert isinstance(result.diagnostics, TitanHazeDiagnostics)
    assert result.diagnostics.gate_failed in {
        'valid_fraction',
        'peak_score',
        'second_peak',
        'ray_yield',
        'arc_inliers',
        'arc_radius',
        'arc_residual',
    }


def test_spurious_result_carries_zero_confidence(
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
) -> None:
    """A rejected fit contributes no confidence to the ensemble."""
    rng = np.random.default_rng(seed=7)
    image = rng.standard_normal(SHAPE_VU) * 10.0 + 100.0
    context = make_nav_context(image, extfov_margin_vu=(WINDOW_PX, WINDOW_PX))
    feature = make_titan_feature(
        predicted_center_vu=CENTER_VU, r_solid_px=R_SOLID_PX, r_env_px=R_ENV_PX
    )
    result = TitanHazeNav().navigate([feature], context)
    assert result.confidence == 0.0


_AT_EDGE_WINDOW_PX = 8
"""Search half-window for the at-edge cases."""

_AT_EDGE_OVERSHOOT = 1.3
"""Multiple of the search window the at-edge cases plant the body at.

Comfortably past the bound rather than exactly on it, so the assertion
cannot be satisfied by a fraction of a pixel of fit overshoot.
"""


def _at_edge_result(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
) -> NavTechniqueResult:
    """Navigate a body planted well beyond the search window along the axis."""
    _, a_hat = axis_vectors(0.0)
    reach = _AT_EDGE_OVERSHOOT * _AT_EDGE_WINDOW_PX
    offset_vu = (float(reach * a_hat[0]), float(reach * a_hat[1]))
    return _run_technique(
        haze_disc_image,
        make_nav_context,
        make_titan_feature,
        offset_vu=offset_vu,
        window_px=_AT_EDGE_WINDOW_PX,
    )


def test_offset_beyond_the_window_sets_at_edge(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
) -> None:
    """An assembled component past the search window flags the result.

    Each pass gates its own component, so a total beyond the declared
    search bound has to be flagged at the result level for the ensemble's
    conservative at-edge handling to apply.
    """
    assert _at_edge_result(haze_disc_image, make_nav_context, make_titan_feature).at_edge is True


def test_at_edge_result_carries_zero_confidence(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
) -> None:
    """An at-edge fit contributes no confidence, matching every other technique.

    The offset may be reporting the search bound rather than the body, so
    the confidence spec's hard-zero gate refuses to price it at all.
    """
    result = _at_edge_result(haze_disc_image, make_nav_context, make_titan_feature)
    assert result.confidence == 0.0


def test_at_edge_result_is_still_committed(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
) -> None:
    """An at-edge fit is flagged, not discarded: the ensemble decides its fate."""
    assert _at_edge_result(haze_disc_image, make_nav_context, make_titan_feature).spurious is False


# ---------------------------------------------------------------------------
# Diagnostics payload
# ---------------------------------------------------------------------------


def test_diagnostics_record_the_fitted_haze_radius(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
) -> None:
    """The fitted radius is reported in kilometers for haze-table accumulation.

    The feature's scale is 20 km/px and the rendered limb sits at 44 px, so
    the fitted radius should land near 880 km.
    """
    result = _run_technique(
        haze_disc_image, make_nav_context, make_titan_feature, offset_vu=(0.3, -0.4)
    )
    assert isinstance(result.diagnostics, TitanHazeDiagnostics)
    assert result.diagnostics.fitted_haze_radius_km == pytest.approx(880.0, abs=40.0)


def test_diagnostics_record_the_envelope_diameter(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
) -> None:
    """The envelope diameter travels from the feature into the diagnostics."""
    result = _run_technique(
        haze_disc_image, make_nav_context, make_titan_feature, offset_vu=(0.3, -0.4)
    )
    assert isinstance(result.diagnostics, TitanHazeDiagnostics)
    assert result.diagnostics.envelope_diameter_px == pytest.approx(2.0 * R_ENV_PX)


def test_diagnostics_record_the_filters(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
) -> None:
    """Filter names ride into the diagnostics for filter-dependent analysis."""
    result = _run_technique(
        haze_disc_image, make_nav_context, make_titan_feature, offset_vu=(0.3, -0.4)
    )
    assert isinstance(result.diagnostics, TitanHazeDiagnostics)
    assert result.diagnostics.filters == ('CL1', 'CL2')


def test_diagnostics_split_the_offset_onto_the_axis(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
) -> None:
    """The reported cross / along components reconstruct the offset."""
    theta = 0.7
    result = _run_technique(
        haze_disc_image,
        make_nav_context,
        make_titan_feature,
        offset_vu=(-1.2, 0.8),
        theta_rad=theta,
    )
    assert isinstance(result.diagnostics, TitanHazeDiagnostics)
    c_hat, a_hat = axis_vectors(math.radians(result.diagnostics.sun_angle_deg))
    rebuilt = result.diagnostics.cross_track_px * c_hat + result.diagnostics.along_track_px * a_hat
    assert float(np.max(np.abs(rebuilt - np.asarray(result.offset_px)))) < 1.0e-9


def test_degenerate_axis_suppresses_angle_refinement(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
) -> None:
    """A degenerate axis leaves the reported angle exactly where the model put it.

    Any axis is equally valid on a rotationally symmetric disc, so refining
    it would only chase noise.
    """
    image = haze_disc_image(SHAPE_VU, _displaced((0.3, -0.4)), R_LIMB_PX, 0.0)
    context = make_nav_context(image, extfov_margin_vu=(WINDOW_PX, WINDOW_PX))
    feature = make_titan_feature(
        predicted_center_vu=CENTER_VU,
        r_solid_px=R_SOLID_PX,
        r_env_px=R_ENV_PX,
        sun_angle_rad=0.0,
        axis_degenerate=True,
    )
    result = TitanHazeNav().navigate([feature], context)
    assert isinstance(result.diagnostics, TitanHazeDiagnostics)
    assert result.diagnostics.theta_refined_deg == pytest.approx(0.0)

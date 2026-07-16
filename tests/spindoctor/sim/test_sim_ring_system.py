"""The optical-depth ring system: photometry closed forms and compositing.

Pins the single-scattering photometry (lit and unlit closed forms, the
mu0 -> mu limit, the Henyey-Greenstein defaults), the flat-ring regression
identity against the legacy annulus renderer at |B| = 90, the per-pixel
depth compositing (near arm in front of a body at the ring center's range,
far arm behind), the transmission-screen behavior (a gap reveals the
background; a star behind a tau = 2 ring at B = 30 attenuates by exp(-4)),
and opaque-body star extinction.
"""

import math
from typing import Any

import numpy as np
import pytest

from spindoctor.sim.forward.ring import render_ring
from spindoctor.sim.forward.ring_system import (
    RING_ALBEDO_DEFAULT,
    RING_PHASE_G_DEFAULT,
    henyey_greenstein_phase,
    render_ring_system,
    ring_reflection_factor,
)
from spindoctor.sim.forward.scene_radiance import compose_scene_radiance
from spindoctor.sim.forward.stages import new_sim_frame
from spindoctor.sim.render import render_combined_model
from spindoctor.sim.scene import validate_sim_params
from spindoctor.sim.scene_schema import SimSceneValidationError
from spindoctor.support.types import NDArrayFloatType


def _tau_array(*values: float) -> NDArrayFloatType:
    return np.asarray(values, dtype=np.float64)


# ---------------------------------------------------------------------------
# Photometry closed forms
# ---------------------------------------------------------------------------


def test_henyey_greenstein_backscatter_peak() -> None:
    """P(g, alpha) at opposition matches (1 - g^2)/(1 + g)^3 for g < 0."""
    g = -0.3
    assert henyey_greenstein_phase(g, 0.0) == pytest.approx((1 - g * g) / (1 + g) ** 3)


def test_henyey_greenstein_forward_scatter() -> None:
    """A dusty positive g brightens strongly toward alpha = 180."""
    assert henyey_greenstein_phase(0.6, 180.0) > 10.0 * henyey_greenstein_phase(0.6, 0.0)


def test_lit_closed_form() -> None:
    """The lit form as written: mu0/(mu0+mu) * (1 - exp(-tau*(1/mu0 + 1/mu)))."""
    tau, mu, mu0 = 1.0, 0.5, 0.3
    factor = ring_reflection_factor(_tau_array(tau), mu, mu0, lit=True)
    expected = mu0 / (mu0 + mu) * (1.0 - math.exp(-tau * (1.0 / mu0 + 1.0 / mu)))
    assert factor[0] == pytest.approx(expected, rel=1e-12)


def test_unlit_closed_form() -> None:
    """The unlit form as written: mu0/(mu0-mu) * (exp(-tau/mu0) - exp(-tau/mu))."""
    tau, mu, mu0 = 0.5, 0.5, 0.3
    factor = ring_reflection_factor(_tau_array(tau), mu, mu0, lit=False)
    expected = mu0 / (mu0 - mu) * (math.exp(-tau / mu0) - math.exp(-tau / mu))
    assert factor[0] == pytest.approx(expected, rel=1e-12)


def test_unlit_limit_form_at_matched_mu() -> None:
    """|mu0 - mu| < 1e-6 takes the analytic limit (tau/mu) * exp(-tau/mu)."""
    tau, mu = 0.8, 0.5
    factor = ring_reflection_factor(_tau_array(tau), mu, mu, lit=False)
    assert factor[0] == pytest.approx((tau / mu) * math.exp(-tau / mu), rel=1e-12)


def test_unlit_limit_is_continuous_with_the_closed_form() -> None:
    """Just outside the tolerance the closed form approaches the limit."""
    tau, mu = 0.8, 0.5
    limit = ring_reflection_factor(_tau_array(tau), mu, mu, lit=False)
    near = ring_reflection_factor(_tau_array(tau), mu, mu + 2.0e-6, lit=False)
    assert near[0] == pytest.approx(limit[0], rel=1e-4)


def test_unlit_side_inverts_with_tau() -> None:
    """The dark-side inversion: high tau is darker than moderate tau."""
    mu, mu0 = 0.5, 0.3
    moderate = ring_reflection_factor(_tau_array(0.5), mu, mu0, lit=False)
    opaque = ring_reflection_factor(_tau_array(4.0), mu, mu0, lit=False)
    assert opaque[0] < moderate[0]


# ---------------------------------------------------------------------------
# Rendered maps
# ---------------------------------------------------------------------------


def _system(
    features: list[dict[str, Any]],
    *,
    b_obs: float = 30.0,
    b_sun: float = 30.0,
    node: float = 0.0,
    **extra: Any,
) -> dict[str, Any]:
    """A ring_system mapping centred for a 96x96 frame."""
    system: dict[str, Any] = {
        'geometry': {
            'center_v': 48.0,
            'center_u': 48.0,
            'opening_deg_obs': b_obs,
            'opening_deg_sun': b_sun,
            'node_deg': node,
        },
        'features': features,
    }
    system.update(extra)
    return system


def _ringlet(a: float, width: float, tau: float, **extra: Any) -> dict[str, Any]:
    feature: dict[str, Any] = {
        'kind': 'ringlet',
        'tau': tau,
        'width': width,
        'orbit': {'a': a, 'ae': 0.0, 'long_peri': 0.0, 'rate_peri': 0.0},
    }
    feature.update(extra)
    return feature


def test_face_on_system_matches_the_legacy_annulus() -> None:
    """|B| = 90 regression identity: tau map == tau * legacy coverage.

    A circular ringlet rendered face-on must reproduce the legacy sky-plane
    annulus exactly: same edges, same anti-aliasing, so the tau map is the
    feature tau times the legacy coverage image, pixel for pixel.
    """
    tau = 1.7
    maps = render_ring_system(
        (96, 96),
        _system([_ringlet(20.0, 8.0, tau)], b_obs=90.0, b_sun=90.0),
        center_v=48.0,
        center_u=48.0,
        node_deg=0.0,
    )
    legacy = np.zeros((96, 96), dtype=np.float64)
    render_ring(
        legacy,
        {
            'center_v': 48.0,
            'center_u': 48.0,
            'feature_type': 'RINGLET',
            'inner_data': [{'mode': 1, 'a': 20.0}],
            'outer_data': [{'mode': 1, 'a': 28.0}],
        },
        0.0,
        offset_u=0.0,
    )
    tau_map = -np.log(maps.transmission)  # mu = 1 face-on, so tau = -ln(T)
    np.testing.assert_allclose(tau_map, tau * legacy, atol=1e-12)


def test_default_photometry_matches_the_closed_form() -> None:
    """A feature without albedo/phase_g uses A = 0.5 and g = -0.3 exactly."""
    maps = render_ring_system(
        (96, 96),
        _system([_ringlet(20.0, 8.0, 1.0)], b_obs=30.0, b_sun=20.0),
        center_v=48.0,
        center_u=48.0,
        node_deg=0.0,
    )
    mu = math.sin(math.radians(30.0))
    mu0 = math.sin(math.radians(20.0))
    expected = (
        RING_ALBEDO_DEFAULT
        / 4.0
        * henyey_greenstein_phase(RING_PHASE_G_DEFAULT, 0.0)
        * mu0
        / (mu0 + mu)
        * (1.0 - math.exp(-(1.0 / mu0 + 1.0 / mu)))
    )
    # The band interior (full coverage) carries exactly the closed form.
    assert float(maps.intensity.max()) == pytest.approx(expected, rel=1e-9)


def test_inclined_system_foreshortens_the_minor_axis() -> None:
    """At B = 30, node = 0 the projected band compresses by sin(B) along v."""
    maps = render_ring_system(
        (96, 96),
        _system([_ringlet(24.0, 2.0, 1.0)], b_obs=30.0, b_sun=30.0),
        center_v=48.0,
        center_u=48.0,
        node_deg=0.0,
    )
    vs, us = np.where(maps.mask)
    half_v = float(np.max(np.abs(vs + 0.5 - 48.0)))
    half_u = float(np.max(np.abs(us + 0.5 - 48.0)))
    assert half_u == pytest.approx(26.0, abs=1.0)
    assert half_v == pytest.approx(26.0 * math.sin(math.radians(30.0)), abs=1.0)


def test_edge_on_system_renders_nothing() -> None:
    """An opening angle of exactly 0 renders nothing (either side)."""
    for b_obs, b_sun in ((0.0, 30.0), (30.0, 0.0)):
        maps = render_ring_system(
            (64, 64),
            _system([_ringlet(20.0, 8.0, 1.0)], b_obs=b_obs, b_sun=b_sun),
            center_v=32.0,
            center_u=32.0,
            node_deg=0.0,
        )
        assert not maps.mask.any()
        assert float(np.max(maps.intensity)) == 0.0


# ---------------------------------------------------------------------------
# Pipeline compositing
# ---------------------------------------------------------------------------


def _scene(**extra: Any) -> dict[str, Any]:
    """A minimal noiseless 96x96 scene."""
    params: dict[str, Any] = {
        'instrument': 'coiss_nac',
        'size_v': 96,
        'size_u': 96,
        'random_seed': 3,
        'exposure_sec': 1.0,
    }
    params.update(extra)
    return validate_sim_params(params)


def _body_at_center(range_km: float) -> dict[str, Any]:
    return {
        'name': 'DISC',
        'center_v': 48.0,
        'center_u': 48.0,
        'axis1': 30.0,
        'axis2': 30.0,
        'axis3': 30.0,
        'illumination_angle': 0.0,
        'phase_angle': 0.0,
        'range_km': range_km,
    }


def _signal_after_radiance(scene: dict[str, Any]) -> NDArrayFloatType:
    """Run only the radiance stage and return its signal plane."""
    frame = new_sim_frame(int(scene['size_v']), int(scene['size_u']))
    compose_scene_radiance(frame, params=scene, rng=np.random.default_rng(0))
    return frame.signal


def test_near_arm_composites_in_front_far_arm_behind() -> None:
    """The pinned depth configuration composites against a co-ranged body.

    B = 30, node = 0: the near arm (positive dv) crosses in front of a body
    at the ring center's range, the far arm (negative dv) passes behind it.
    """
    ring_system = _system(
        [_ringlet(18.0, 10.0, 1.0)], b_obs=30.0, b_sun=30.0, range_km=1.0e6, km_per_pixel=1000.0
    )
    img = _signal_after_radiance(_scene(bodies=[_body_at_center(1.0e6)], ring_system=ring_system))
    body_img = _signal_after_radiance(_scene(bodies=[_body_at_center(1.0e6)]))
    # At du ~ 0 the band spans ring-plane radii 18-28, which project to
    # |dv| in 9-14 at B = 30; probe one full-coverage pixel inside each arm
    # (pixel row 59 sits at dv = +11.5, ring radius ~23, mid-band).
    near_v, far_v = 59, 37
    mu = math.sin(math.radians(30.0))
    factor = mu / (mu + mu) * (1.0 - math.exp(-1.0 * (2.0 / mu)))
    intensity = (
        RING_ALBEDO_DEFAULT / 4.0 * henyey_greenstein_phase(RING_PHASE_G_DEFAULT, 0.0) * factor
    )
    transmission = math.exp(-1.0 / mu)
    expected_near = intensity + transmission * body_img[near_v, 48]
    assert img[near_v, 48] == pytest.approx(expected_near, rel=1e-9)
    assert img[far_v, 48] == pytest.approx(body_img[far_v, 48], rel=1e-12)
    assert body_img[far_v, 48] > 0.0


def _point_e_after_radiance(scene: dict[str, Any]) -> NDArrayFloatType:
    """Run only the radiance stage and return its point-source plane."""
    frame = new_sim_frame(int(scene['size_v']), int(scene['size_u']))
    compose_scene_radiance(frame, params=scene, rng=np.random.default_rng(0))
    return frame.point_e


def test_star_behind_tau2_ring_attenuates_by_exp_minus_4() -> None:
    """A star behind the B ring (tau = 2) at B = 30 loses > 98% of its flux."""
    star = {'name': 'S', 'v': 59.5, 'u': 48.5, 'vmag': 6.0}
    ring_system = _system([_ringlet(15.0, 14.0, 2.0)], b_obs=30.0, b_sun=30.0)
    with_ring = _point_e_after_radiance(_scene(stars=[star], ring_system=ring_system))
    without = _point_e_after_radiance(_scene(stars=[star]))
    region = np.s_[56:64, 45:53]
    ratio = float(with_ring[region].sum() / without[region].sum())
    assert ratio == pytest.approx(math.exp(-2.0 / math.sin(math.radians(30.0))), rel=1e-9)
    assert ratio < 0.02


def test_gap_reveals_the_background_star() -> None:
    """A gap carved into a sheet lets the background through unattenuated."""
    star = {'name': 'S', 'v': 59.5, 'u': 48.5, 'vmag': 6.0}
    sheet = _ringlet(5.0, 35.0, 2.0)
    gap = dict(_ringlet(18.0, 10.0, 2.0), kind='gap')
    with_gap = _point_e_after_radiance(
        _scene(stars=[star], ring_system=_system([sheet, gap], b_obs=30.0, b_sun=30.0))
    )
    without_ring = _point_e_after_radiance(_scene(stars=[star]))
    region = np.s_[56:64, 45:53]
    # The star deposit sits behind the gap interior (ring radii ~23-25,
    # inside 18-28), where the sheet's tau is fully suppressed.
    assert float(with_gap[region].sum()) == pytest.approx(
        float(without_ring[region].sum()), rel=1e-9
    )


def test_gap_reveals_the_body_behind_the_sheet() -> None:
    """The signal path too: a gap over a body shows the body through it."""
    ring_system = _system(
        [_ringlet(5.0, 35.0, 2.0), dict(_ringlet(18.0, 10.0, 2.0), kind='gap')],
        b_obs=30.0,
        b_sun=30.0,
        range_km=9.0e5,
        km_per_pixel=1000.0,
    )
    img = _signal_after_radiance(_scene(bodies=[_body_at_center(1.0e6)], ring_system=ring_system))
    body_img = _signal_after_radiance(_scene(bodies=[_body_at_center(1.0e6)]))
    # dv = +11.5, du ~ 0 (ring radius ~23) sits mid-gap: full transmission,
    # zero emission, so the body shows through exactly.
    assert img[59, 48] == pytest.approx(body_img[59, 48], rel=1e-12)
    assert body_img[59, 48] > 0.0
    # dv = +6.5 (ring radius ~13) sits on the sheet in front of the body.
    assert img[54, 48] != pytest.approx(body_img[54, 48])


def test_star_behind_an_opaque_body_vanishes() -> None:
    """A lit body silhouette extinguishes the point sources behind it."""
    star = {'name': 'S', 'v': 48.5, 'u': 48.5, 'vmag': 6.0}
    with_body = _point_e_after_radiance(_scene(stars=[star], bodies=[_body_at_center(1.0e6)]))
    without = _point_e_after_radiance(_scene(stars=[star]))
    region = np.s_[45:53, 45:53]
    assert float(without[region].sum()) > 0.0
    assert float(with_body[region].sum()) == 0.0


def test_star_clear_of_the_body_is_untouched() -> None:
    """Extinction is per pixel: a star off the silhouette keeps its flux."""
    star = {'name': 'S', 'v': 10.5, 'u': 10.5, 'vmag': 6.0}
    with_body = _point_e_after_radiance(_scene(stars=[star], bodies=[_body_at_center(1.0e6)]))
    without = _point_e_after_radiance(_scene(stars=[star]))
    region = np.s_[7:15, 7:15]
    assert float(with_body[region].sum()) == pytest.approx(float(without[region].sum()), rel=1e-12)


def test_ring_system_over_body_requires_both_ranges() -> None:
    """Depth ordering against a body needs range_km on both sides."""
    ring_system = _system([_ringlet(20.0, 4.0, 1.0)], b_obs=30.0, b_sun=30.0)
    scene = _scene(bodies=[_body_at_center(1.0e6)], ring_system=ring_system)
    with pytest.raises(SimSceneValidationError, match=r"ring_system and body 'DISC' overlap"):
        render_combined_model(scene)


def test_ring_system_over_legacy_ring_is_an_error() -> None:
    """A legacy painted ring has no per-pixel depth to order against."""
    legacy = {
        'name': 'OLD',
        'feature_type': 'RINGLET',
        'center_v': 48.0,
        'center_u': 48.0,
        'range_km': 2.0e6,
        'inner_data': [{'mode': 1, 'a': 8.0}],
        'outer_data': [{'mode': 1, 'a': 30.0}],
    }
    ring_system = _system(
        [_ringlet(20.0, 4.0, 1.0)], b_obs=30.0, b_sun=30.0, range_km=1.0e6, km_per_pixel=1000.0
    )
    scene = _scene(rings=[legacy], ring_system=ring_system)
    with pytest.raises(SimSceneValidationError, match='overlaps legacy ring'):
        render_combined_model(scene)


def test_planted_offset_translates_the_ring_system() -> None:
    """The planted pointing offset shifts the projected system rigidly."""
    ring_system = _system([_ringlet(15.0, 5.0, 1.0)], b_obs=40.0, b_sun=40.0, node=25.0)
    base, _m0 = render_combined_model(_scene(ring_system=ring_system))
    shifted, _m1 = render_combined_model(
        _scene(ring_system=ring_system, offset_v=5.0, offset_u=-3.0)
    )
    np.testing.assert_allclose(shifted[5:, :-3], base[: 96 - 5, 3:], atol=1e-12)


def test_oversampled_interior_matches_detector_grid_render() -> None:
    """The band interior is grid-independent; only the AA edges differ."""
    ring_system = _system([_ringlet(16.0, 10.0, 1.0)], b_obs=35.0, b_sun=25.0, node=10.0)
    at_1, _m1 = render_combined_model(_scene(ring_system=ring_system))
    at_4, _m4 = render_combined_model(_scene(ring_system=ring_system, oversample=4))
    both_on = (at_1 > 0.0) & (at_4 > 0.0)
    interior = both_on & (np.abs(at_1 - at_4) < 1e-3)
    # The overwhelming majority of lit pixels agree to 1e-3; the disagreeing
    # remainder is the one-detector-pixel anti-aliased edge band.
    assert float(np.count_nonzero(interior)) / float(np.count_nonzero(both_on)) > 0.8


def test_differential_smear_layers_recompose_the_ring_system_exactly() -> None:
    """Zero-vector per-class smear reproduces the composite (exact layers)."""
    ring_system = _system(
        [_ringlet(20.0, 4.0, 1.0)], b_obs=30.0, b_sun=30.0, range_km=9.0e5, km_per_pixel=1000.0
    )
    common: dict[str, Any] = {
        'bodies': [_body_at_center(1.0e6)],
        'stars': [{'name': 'S', 'v': 10.5, 'u': 80.5, 'vmag': 6.0}],
        'ring_system': ring_system,
    }
    plain, _m0 = render_combined_model(_scene(**common))
    layered, _m1 = render_combined_model(
        _scene(**common, optics={'smear': [{'dv_px': 0.0, 'du_px': 0.0, 'object_class': 'rings'}]})
    )
    np.testing.assert_allclose(layered, plain, atol=1e-9)

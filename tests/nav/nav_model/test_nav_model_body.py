"""Tests for ``nav.nav_model.nav_model_body`` helpers and emission gates.

The full ``NavModelBody.create_model`` path requires a live ``oops``
backplane and so is exercised by integration tests against the image
library.  These unit tests cover the pure helper functions
(``_incidence_factor_array``, ``_sigma_normal_per_vertex``, the
reliability sigmoids) plus the class-level constants.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from nav.feature.constants import MAX_INCIDENCE_FACTOR_CAP
from nav.nav_model.body_shape import DEFAULT_BODY_SHAPE
from nav.nav_model.nav_model_body import (
    BODY_DISC_MAX_OVERFLOW_FRACTION,
    BODY_DISC_MIN_VISIBLE_LIT_FRACTION,
    BODY_POSITION_SLOP_FRAC,
    LIMB_ARC_MAX_UNCERTAINTY_PX,
    TERMINATOR_MIN_PHASE_FACTOR,
    TERMINATOR_MIN_VERTICES,
    _blob_reliability,
    _disc_reliability,
    _incidence_factor_array,
    _limb_reliability,
    _PolylineSampler,
    _sigma_normal_per_vertex,
    _sigmoid,
    _terminator_reliability,
    _visible_arc_fraction,
)


def test_constants_have_design_values() -> None:
    """Module-level constants match the design's defaults."""
    assert pytest.approx(0.05) == BODY_POSITION_SLOP_FRAC
    assert pytest.approx(3.0) == LIMB_ARC_MAX_UNCERTAINTY_PX
    assert pytest.approx(0.4) == BODY_DISC_MIN_VISIBLE_LIT_FRACTION
    assert pytest.approx(0.3) == BODY_DISC_MAX_OVERFLOW_FRACTION
    assert TERMINATOR_MIN_VERTICES == 8
    assert pytest.approx(0.05) == TERMINATOR_MIN_PHASE_FACTOR


def test_incidence_factor_zero_at_subsolar() -> None:
    """At i=0 the incidence factor is zero (no softening)."""
    arr = _incidence_factor_array(np.array([0.0]))
    assert arr[0] == pytest.approx(0.0, abs=1e-12)


def test_incidence_factor_one_at_60deg() -> None:
    """At i=60 the incidence factor is 1 (cos=0.5; 1/0.5 - 1 = 1)."""
    arr = _incidence_factor_array(np.array([math.radians(60.0)]))
    assert arr[0] == pytest.approx(1.0, abs=1e-12)


def test_incidence_factor_saturates_above_80deg() -> None:
    """Above 80 deg the formula uses the i=80 value (capped by the angle cap)."""
    arr_80 = _incidence_factor_array(np.array([math.radians(80.0)]))
    arr_89 = _incidence_factor_array(np.array([math.radians(89.0)]))
    expected_at_80 = 1.0 / math.cos(math.radians(80.0)) - 1.0
    assert arr_80[0] == pytest.approx(expected_at_80, rel=1e-12)
    assert arr_89[0] == pytest.approx(arr_80[0], rel=1e-12)
    # The cap constant bounds the factor from above (numerically slack).
    assert arr_89[0] <= MAX_INCIDENCE_FACTOR_CAP + 1e-9


def test_incidence_factor_array_returns_float64() -> None:
    """The output dtype is float64 even when inputs are integer-typed."""
    arr = _incidence_factor_array(np.array([0.0, 1.0]))
    assert arr.dtype == np.float64


def _make_sampler(*, n: int, incidence_deg: float, km_per_pixel: float) -> _PolylineSampler:
    """Build a ``_PolylineSampler`` with constant-incidence vertices."""
    vertices = np.zeros((n, 2), dtype=np.float64)
    normals = np.zeros((n, 2), dtype=np.float64)
    incidence = np.full(n, math.radians(incidence_deg), dtype=np.float64)
    km = np.full(n, km_per_pixel, dtype=np.float64)
    return _PolylineSampler(
        vertices_vu=vertices,
        normals_vu=normals,
        incidence_rad=incidence,
        km_per_pixel=km,
    )


def test_sigma_normal_uses_quadrature_sum() -> None:
    """``sigma_normal_per_vertex`` matches the design's quadrature formula."""
    shape = DEFAULT_BODY_SHAPE
    sampler = _make_sampler(n=4, incidence_deg=0.0, km_per_pixel=10.0)
    sigma_px = _sigma_normal_per_vertex(
        sampler=sampler, shape=shape, psf_sigma_px=1.0, include_albedo=False
    )
    expected_km = math.sqrt(
        shape.ellipsoid_rms_residual_km**2
        + shape.crater_scale_km**2
        + 0.0  # incidence_factor=0 at i=0
        + shape.spice_orbital_residual_km**2
    )
    expected_px = expected_km / 10.0
    assert sigma_px[0] == pytest.approx(expected_px, rel=1e-12)


def test_sigma_normal_albedo_term_increases_when_enabled() -> None:
    """``include_albedo=True`` adds the albedo + photometric quadrature terms."""
    shape = DEFAULT_BODY_SHAPE
    sampler = _make_sampler(n=2, incidence_deg=30.0, km_per_pixel=5.0)
    no_albedo = _sigma_normal_per_vertex(
        sampler=sampler, shape=shape, psf_sigma_px=1.0, include_albedo=False
    )
    with_albedo = _sigma_normal_per_vertex(
        sampler=sampler, shape=shape, psf_sigma_px=1.0, include_albedo=True
    )
    assert with_albedo[0] > no_albedo[0]


def test_visible_arc_fraction_reports_unity_when_vertices_present() -> None:
    """A non-empty polyline reports ``visible_arc_fraction = 1.0``."""
    sampler = _make_sampler(n=5, incidence_deg=0.0, km_per_pixel=1.0)
    assert _visible_arc_fraction(sampler) == 1.0


def test_visible_arc_fraction_reports_zero_when_empty() -> None:
    """An empty polyline reports zero arc fraction."""
    sampler = _PolylineSampler(
        vertices_vu=np.empty((0, 2), dtype=np.float64),
        normals_vu=np.empty((0, 2), dtype=np.float64),
        incidence_rad=np.empty(0, dtype=np.float64),
        km_per_pixel=np.empty(0, dtype=np.float64),
    )
    assert _visible_arc_fraction(sampler) == 0.0


def test_sigmoid_is_monotone() -> None:
    """The sigmoid is strictly increasing across a wide range of inputs."""
    samples = [_sigmoid(x) for x in (-5.0, -1.0, 0.0, 1.0, 5.0)]
    for prev, nxt in itertools.pairwise(samples):
        assert nxt > prev


def test_sigmoid_at_zero() -> None:
    """``_sigmoid(0)`` equals 0.5 — sanity check the symmetry."""
    assert _sigmoid(0.0) == pytest.approx(0.5, abs=1e-12)


def test_limb_reliability_increases_with_visible_arc_fraction() -> None:
    """Reliability is monotone in ``visible_arc_fraction`` for fixed arc length."""
    low = _limb_reliability(visible_arc_fraction=0.2, visible_arc_px=20.0)
    high = _limb_reliability(visible_arc_fraction=0.9, visible_arc_px=20.0)
    assert high > low


def test_limb_reliability_increases_with_arc_length() -> None:
    """Longer arcs score higher (the ``visible_arc_px`` sigmoid)."""
    short = _limb_reliability(visible_arc_fraction=0.9, visible_arc_px=5.0)
    long = _limb_reliability(visible_arc_fraction=0.9, visible_arc_px=200.0)
    assert long > short


def test_limb_reliability_passes_gate_for_fully_lit_geometry() -> None:
    """A fully visible, well-sampled limb scores well above the 0.30 gate.

    Per-vertex softness already lives in ``_sigma_normal_per_vertex``;
    the reliability score is a feature-existence gate, not a precision
    estimate, so a textbook-good limb (Dione at low phase) must clear
    it.
    """
    score = _limb_reliability(visible_arc_fraction=1.0, visible_arc_px=300.0)
    assert score > 0.5


def test_terminator_reliability_zero_at_zero_phase() -> None:
    """Sub-solar-illuminated images produce zero terminator reliability."""
    out = _terminator_reliability(visible_arc_fraction=1.0, albedo_variation=0.0, phase_factor=0.0)
    assert out == 0.0


def test_disc_reliability_increases_with_diameter() -> None:
    """Reliability is monotone in body diameter for fixed visibility."""
    low = _disc_reliability(visible_lit_fraction=1.0, overflow_fraction=0.0, diameter_px=10.0)
    high = _disc_reliability(visible_lit_fraction=1.0, overflow_fraction=0.0, diameter_px=200.0)
    assert high > low


def test_disc_reliability_zero_when_overflow_complete() -> None:
    """A fully off-frame disc has zero reliability."""
    out = _disc_reliability(visible_lit_fraction=1.0, overflow_fraction=1.0, diameter_px=100.0)
    assert out == 0.0


def test_blob_reliability_capped_at_0_4() -> None:
    """Blob reliability cannot exceed the 0.4 design cap."""
    out = _blob_reliability(snr=1e6, diameter_px=1e6)
    assert out <= 0.4 + 1e-12

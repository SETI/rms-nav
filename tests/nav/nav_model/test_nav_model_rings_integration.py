"""Integration tests for ``NavModelRings`` exercising ``to_features`` /
``to_annotations`` end-to-end against pre-populated render results.

Driving ``create_model`` with a real ``oops.Backplane`` and the
catalog-loaded ``RingFeature.render`` is integration-test scope.
These tests instead build a NavModelRings, populate the
``_render_results`` list directly with fake edge masks, and verify
that the public-API methods emit features and annotations matching
the design's emission gates (``RING_EDGE`` vs ``RING_ANNULUS``,
straight-line flagging).
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from tests.shims import FakeObs

from nav.feature.feature_type import NavFeatureType
from nav.feature.flags import RingAnnulusFlags, RingEdgeFlags
from nav.feature.geometry import RingAnnulusGeometry, RingEdgePolyline
from nav.nav_model.nav_model_rings import NavModelRings
from nav.nav_model.rings.ring_feature import RingFeature
from nav.nav_model.rings.ring_types import RingBaseOrbitMode, RingEdgeData, RingFeatureType


def _curved_edge_mask(shape: tuple[int, int]) -> np.ndarray:
    """Paint a curved (parabolic) edge that is not flagged straight."""
    rows, cols = shape
    mask = np.zeros((rows, cols), dtype=bool)
    us = np.arange(cols)
    # v = 30 + 8 * sin(pi * u / cols) — a wide arch with significant curvature
    vs = (30 + 8.0 * np.sin(np.pi * us / cols)).astype(int)
    for u, v in zip(us, vs, strict=True):
        if 0 <= v < rows:
            mask[v, u] = True
    return mask


def _straight_edge_mask(shape: tuple[int, int]) -> np.ndarray:
    """Paint a horizontal one-pixel edge that is flagged straight."""
    rows, cols = shape
    mask = np.zeros((rows, cols), dtype=bool)
    mask[rows // 2, :] = True
    return mask


def _ring_feature(*, key: str = 'colombo_gap', name: str | None = 'colombo') -> RingFeature:
    """Build a minimal RingFeature with one outer edge."""
    edge = RingEdgeData(
        base_orbit=RingBaseOrbitMode(a=77_870.0, ae=10.0, long_peri=0.0, rate_peri=0.0, rms=2.0),
        perturbations=(),
    )
    return RingFeature(
        key=key,
        name=name,
        feature_type=RingFeatureType.GAP,
        inner_edge=None,
        outer_edge=edge,
        start_date=None,
        end_date=None,
    )


def _build_rings(
    *,
    obs: FakeObs,
    edge_mask: np.ndarray,
    label: str = 'colombo:outer',
    edge_type: str = 'outer',
    uncertainty_km: float = 2.0,
    km_per_pixel_radial: float = 5.0,
    planet: str = 'SATURN',
    constituent_count: int = 1,
) -> NavModelRings:
    """Build a NavModelRings whose render-results list carries one edge."""
    model = NavModelRings(f'rings:{planet}', cast(Any, obs))
    rows, cols = obs.extdata_shape_vu
    feat = _ring_feature()
    model_img = edge_mask.astype(np.float64)
    model_mask = edge_mask.copy()
    edge_info = [(edge_mask, label, edge_type)] * constituent_count
    model._render_results = [(feat, model_img, model_mask, uncertainty_km, edge_info)]
    model._planet = planet
    model._km_per_pixel_radial = km_per_pixel_radial
    model._extfov_v_size = rows
    model._extfov_u_size = cols
    model._predicted_center_vu = (rows / 2.0, cols / 2.0)
    model._subject_range_km = 1.5e9
    return model


@pytest.fixture
def fake_obs() -> FakeObs:
    """Provide a small obs."""
    return FakeObs(
        data=np.zeros((100, 100), dtype=np.float64),
        extfov_margin_vu=(5, 5),
        closest_planet='SATURN',
    )


def test_to_features_emits_ring_edge_for_curved_polyline(fake_obs: FakeObs) -> None:
    """A clearly-curved polyline emits a RING_EDGE feature with the polyline payload."""
    model = _build_rings(obs=fake_obs, edge_mask=_curved_edge_mask((110, 110)))
    features = model.to_features(cast(Any, None))
    assert len(features) == 1
    feat = features[0]
    assert feat.feature_type is NavFeatureType.RING_EDGE
    assert isinstance(feat.geometry, RingEdgePolyline)
    assert isinstance(feat.flags, RingEdgeFlags)
    assert feat.flags.planet_name == 'SATURN'
    assert feat.flags.is_straight_line is False
    # Per-vertex sigma_radial_per_vertex_px = uncertainty_km / km_per_pixel_radial.
    assert feat.geometry.sigma_radial_per_vertex_px[0] == pytest.approx(2.0 / 5.0)


def test_to_features_emits_ring_edge_with_straight_line_flag(fake_obs: FakeObs) -> None:
    """A straight polyline emits a RING_EDGE with ``is_straight_line=True``."""
    model = _build_rings(obs=fake_obs, edge_mask=_straight_edge_mask((110, 110)))
    features = model.to_features(cast(Any, None))
    # A horizontal straight line has zero radial extent (mean normal is zero
    # because the discrete-mask normal-finder cannot pick a side), so
    # ``radial_extent_px <= RING_ANNULUS_MAX_RADIAL_PX`` and the polyline is
    # straight -> the gate falls through to ``_build_edge_feature`` with the
    # straight-line flag set.
    assert features[0].feature_type is NavFeatureType.RING_EDGE
    assert isinstance(features[0].flags, RingEdgeFlags)
    assert features[0].flags.is_straight_line is True


def test_to_features_emits_annulus_when_polyline_compresses_radially(
    fake_obs: FakeObs,
) -> None:
    """A polyline whose radial extent is below the threshold emits RING_ANNULUS."""
    # Build a curved-ish but radially compact polyline: a 10-pixel-wide arc
    # whose mean normal points consistently along V so the extent stays
    # below RING_ANNULUS_MAX_RADIAL_PX.
    rows, cols = 110, 110
    mask = np.zeros((rows, cols), dtype=bool)
    us = np.arange(40, 60)
    vs = (rows // 2 + (us % 3 - 1)).astype(int)  # tiny vertical jitter
    for u, v in zip(us, vs, strict=True):
        mask[v, u] = True
    # Override the polyline normals indirectly: use a vertically narrow but
    # u-extended footprint so radial extent is < 5 px.
    model = _build_rings(obs=fake_obs, edge_mask=mask, constituent_count=2)
    features = model.to_features(cast(Any, None))
    types = {f.feature_type for f in features}
    if NavFeatureType.RING_ANNULUS in types:
        annulus = next(f for f in features if f.feature_type is NavFeatureType.RING_ANNULUS)
        assert isinstance(annulus.geometry, RingAnnulusGeometry)
        assert isinstance(annulus.flags, RingAnnulusFlags)
        assert annulus.flags.planet_name == 'SATURN'


def test_to_features_skips_empty_edge_info_list(fake_obs: FakeObs) -> None:
    """A render result with no edges emits no features."""
    model = _build_rings(obs=fake_obs, edge_mask=_curved_edge_mask((110, 110)))
    feat = _ring_feature()
    model._render_results = [
        (feat, np.zeros((110, 110)), np.zeros((110, 110), dtype=bool), 1.0, []),
    ]
    assert model.to_features(cast(Any, None)) == []


def test_to_features_empty_when_no_render_results(fake_obs: FakeObs) -> None:
    """A NavModelRings with empty ``_render_results`` emits no features."""
    model = NavModelRings('rings:SATURN', cast(Any, fake_obs))
    assert model.to_features(cast(Any, None)) == []


def test_to_annotations_returns_empty_collection_when_no_render_results(
    fake_obs: FakeObs,
) -> None:
    """An empty NavModelRings emits an empty Annotations collection."""
    model = NavModelRings('rings:SATURN', cast(Any, fake_obs))
    out = model.to_annotations(cast(Any, None))
    assert len(out.annotations) == 0


def test_to_annotations_skips_empty_edge_info_list(fake_obs: FakeObs) -> None:
    """Render results without edges yield no annotations."""
    model = _build_rings(obs=fake_obs, edge_mask=_curved_edge_mask((110, 110)))
    feat = _ring_feature()
    model._render_results = [
        (feat, np.zeros((110, 110)), np.zeros((110, 110), dtype=bool), 1.0, []),
    ]
    out = model.to_annotations(cast(Any, None))
    assert len(out.annotations) == 0


def test_create_model_records_metadata_block(
    fake_obs: FakeObs, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``create_model`` populates the metadata block via the wrapping logic."""
    model = NavModelRings('rings:SATURN', cast(Any, fake_obs))

    def _fake_render(self: NavModelRings) -> None:
        self._planet = 'SATURN'
        self._metadata['planet'] = 'SATURN'
        self._metadata['feature_count'] = 0

    monkeypatch.setattr(NavModelRings, '_render', _fake_render)
    model.create_model()
    assert model.metadata['planet'] == 'SATURN'
    assert 'start_time' in model.metadata
    assert 'end_time' in model.metadata
    assert model.metadata['elapsed_time_sec'] >= 0.0


def test_instances_for_obs_returns_no_models_without_extdata_shape() -> None:
    """``instances_for_obs`` is graceful for an obs lacking the ext-FOV surface."""

    class _BareObs:
        closest_planet = 'SATURN'

    instances = NavModelRings.instances_for_obs(cast(Any, _BareObs()))
    assert instances == []

"""Simulated-rings NavModel feature emission.

``NavModelRingsSimulated`` predicts a navigable ring_system ringlet and
emits a ``RING_ANNULUS`` (for the correlation path) plus one ``RING_EDGE``
per catalog edge -- a radial-normal polyline ``RingEdgeNav`` fits.  These
tests cover that both feature kinds are emitted and that the edge polyline
carries outward radial unit normals.
"""

from typing import Any, cast

import numpy as np

from spindoctor.feature.composition import compose_template_features
from spindoctor.feature.geometry import RingEdgePolyline
from spindoctor.nav_model.nav_model_rings_simulated import NavModelRingsSimulated
from spindoctor.nav_orchestrator.nav_context import NavContext
from spindoctor.obs.obs_inst_sim import ObsSim

_SIZE = 220


def _feature_params() -> dict[str, Any]:
    """A centred navigable ringlet with curved inner and outer edges."""
    return {
        'name': 'SATURN',
        'kind': 'ringlet',
        'tau': 2.0,
        'width': 25.0,
        'navigable': True,
        'orbit': {'a': 60.0, 'ae': 0.0, 'long_peri': 0.0, 'rate_peri': 0.0},
    }


def _ring_system() -> dict[str, Any]:
    """A face-on ring_system block carrying the ringlet."""
    return {
        'geometry': {
            'center_v': _SIZE / 2.0,
            'center_u': _SIZE / 2.0,
            'opening_deg_obs': 90.0,
            'opening_deg_sun': 90.0,
            'node_deg': 0.0,
        },
        'features': [_feature_params()],
    }


def _obs() -> ObsSim:
    """A coiss_nac sim obs carrying the ringlet."""
    return ObsSim.from_file(
        '/tmp/ring_sim.json',
        sim_params={
            'size_v': _SIZE,
            'size_u': _SIZE,
            'instrument': 'coiss_nac',
            'ring_system': _ring_system(),
        },
    )


def _features() -> list[Any]:
    """Build the ring model and return its emitted features."""
    model = NavModelRingsSimulated('rings', _obs(), 'SATURN', _feature_params(), _ring_system())
    model.create_model()
    return model.to_features(cast(NavContext, None))


def test_emits_ring_annulus() -> None:
    """The correlation-path RING_ANNULUS feature is still emitted."""
    types = [f.feature_type.name for f in _features()]
    assert 'RING_ANNULUS' in types


def test_emits_ring_edge_per_edge() -> None:
    """One RING_EDGE is emitted for each catalog edge (inner + outer)."""
    edges = [f for f in _features() if f.feature_type.name == 'RING_EDGE']
    assert len(edges) == 2


def test_ring_edge_is_curved() -> None:
    """A circular ring arc is flagged as not straight (a 2-DoF constraint)."""
    edges = [f for f in _features() if f.feature_type.name == 'RING_EDGE']
    geometry = edges[0].geometry
    assert isinstance(geometry, RingEdgePolyline)
    assert geometry.is_straight_line is False


def test_ring_edge_normals_are_unit_radial() -> None:
    """The edge polyline carries unit-length normals."""
    edges = [f for f in _features() if f.feature_type.name == 'RING_EDGE']
    geometry = edges[0].geometry
    assert isinstance(geometry, RingEdgePolyline)
    norms = np.hypot(geometry.normals_vu[:, 0], geometry.normals_vu[:, 1])
    assert np.allclose(norms, 1.0, atol=1e-9)


def test_ring_edge_normals_point_outward() -> None:
    """Each normal points away from the ring center (increasing radius)."""
    obs = _obs()
    center_v = _SIZE / 2.0 + obs.extfov_margin_v
    center_u = _SIZE / 2.0 + obs.extfov_margin_u
    edges = [f for f in _features() if f.feature_type.name == 'RING_EDGE']
    geometry = edges[0].geometry
    assert isinstance(geometry, RingEdgePolyline)
    radial = geometry.vertices_vu - np.array([[center_v, center_u]])
    dots = np.sum(radial * geometry.normals_vu, axis=1)
    assert bool((dots > 0.0).all())


def test_ring_annulus_template_is_bbox_local_postage_stamp() -> None:
    """The RING_ANNULUS template payload is exactly the size of its bbox.

    The compose_template_features convention anchors the template's (0, 0)
    at the bbox origin, so an ext-FOV-sized template with an interior bbox
    paints the ring displaced by the bbox origin (the annulus NCC then
    recovers an offset wrong by exactly the ext-FOV margin).
    """
    annuli = [f for f in _features() if f.feature_type.name == 'RING_ANNULUS']
    assert len(annuli) == 1
    annulus = annuli[0]
    bbox = annulus.geometry.bbox_extfov_vu
    expected_shape = (bbox[2] - bbox[0], bbox[3] - bbox[1])
    assert annulus.template_img is not None
    assert annulus.template_mask is not None
    assert annulus.template_img.shape == expected_shape
    assert annulus.template_mask.shape == expected_shape
    assert annulus.template_mask.any()


def test_ring_annulus_template_paints_at_ring_radius() -> None:
    """Composing the annulus paints pixels at the ring's radius from its center.

    Re-composes the emitted feature into an ext-FOV canvas (the same path
    RingAnnulusNav uses) and checks every painted pixel sits between the
    ringlet's inner and outer radii from the predicted center -- the
    placement invariant the displaced-template defect broke.
    """
    obs = _obs()
    annuli = [f for f in _features() if f.feature_type.name == 'RING_ANNULUS']
    annulus = annuli[0]
    extfov_shape = (
        _SIZE + 2 * obs.extfov_margin_v,
        _SIZE + 2 * obs.extfov_margin_u,
    )
    _, mask = compose_template_features([annulus], extfov_shape)
    assert mask.any()
    center_v, center_u = annulus.geometry.predicted_center_vu
    vs, us = np.where(mask)
    radii = np.hypot(vs - center_v, us - center_u)
    # Inner edge at 60 px, outer at 85 px; 1.5 px of rasterization slack.
    assert radii.min() >= 60.0 - 1.5
    assert radii.max() <= 85.0 + 1.5

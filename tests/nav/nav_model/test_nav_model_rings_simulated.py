"""Simulated-rings NavModel feature emission.

``NavModelRingsSimulated`` renders a ringlet and emits a ``RING_ANNULUS`` (for the
correlation path) plus one ``RING_EDGE`` per rendered edge -- a radial-normal
polyline ``RingEdgeNav`` fits.  These tests cover that both feature kinds are
emitted and that the edge polyline carries outward radial unit normals.
"""

from typing import Any, cast

import numpy as np

from nav.feature.geometry import RingEdgePolyline
from nav.nav_model.nav_model_rings_simulated import NavModelRingsSimulated
from nav.nav_orchestrator.nav_context import NavContext
from nav.obs.obs_inst_sim import ObsSim

_SIZE = 220


def _edge(a: float) -> list[dict[str, Any]]:
    """A single mode-1 ring edge at pixel radius ``a``."""
    return [{'mode': 1, 'a': a, 'rms': 1.0, 'ae': 0.0, 'long_peri': 0.0, 'rate_peri': 0.0}]


def _ring_params() -> dict[str, Any]:
    """A centred ringlet with curved inner and outer edges."""
    return {
        'name': 'SATURN',
        'feature_type': 'RINGLET',
        'center_v': _SIZE / 2.0,
        'center_u': _SIZE / 2.0,
        'inner_data': _edge(60.0),
        'outer_data': _edge(85.0),
    }


def _obs() -> ObsSim:
    """A coiss_nac sim obs carrying the ringlet."""
    return ObsSim.from_file(
        '/tmp/ring_sim.json',
        sim_params={
            'size_v': _SIZE,
            'size_u': _SIZE,
            'instrument': 'coiss_nac',
            'rings': [_ring_params()],
        },
    )


def _features() -> list[Any]:
    """Build the ring model and return its emitted features."""
    model = NavModelRingsSimulated('rings', _obs(), 'SATURN', _ring_params())
    model.create_model()
    return model.to_features(cast(NavContext, None))


def test_emits_ring_annulus() -> None:
    """The correlation-path RING_ANNULUS feature is still emitted."""
    types = [f.feature_type.name for f in _features()]
    assert 'RING_ANNULUS' in types


def test_emits_ring_edge_per_edge() -> None:
    """One RING_EDGE is emitted for each rendered edge (inner + outer)."""
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

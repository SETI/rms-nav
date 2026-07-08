"""Navigator-side simulated body model, including mesh prediction (B7).

``NavModelBodySimulated`` renders the *predicted* body silhouette from its own
params, which need not match what was rendered into the image.  These tests
cover the mesh-vs-ellipsoid prediction split, the pose-disagreement fixture, and
that the predicted mesh reproduces the rendered data when the params agree.
"""

from typing import Any

import numpy as np

from spindoctor.nav_model.nav_model_body_simulated import NavModelBodySimulated
from spindoctor.obs.obs_inst_sim import ObsSim
from spindoctor.sim.render import render_combined_model

_SIZE = 96


def _obs() -> ObsSim:
    """A coiss_nac sim obs sized for the body scenes below."""
    return ObsSim.from_file(
        '/tmp/body_sim.json',
        sim_params={'size_v': _SIZE, 'size_u': _SIZE, 'instrument': 'coiss_nac'},
    )


def _body_params(**overrides: Any) -> dict[str, Any]:
    """Centred irregular-body params for prediction and rendering."""
    params = {
        'name': 'HYPERION',
        'center_v': _SIZE / 2.0,
        'center_u': _SIZE / 2.0,
        'axis1': 60.0,
        'axis2': 46.0,
        'axis3': 46.0,
        'illumination_angle': 30.0,
        'phase_angle': 45.0,
    }
    params.update(overrides)
    return params


def _predicted_mask(obs: ObsSim, body_params: dict[str, Any]) -> np.ndarray:
    """Build the model and return its predicted body mask (extfov coords)."""
    model = NavModelBodySimulated('body', obs, body_params['name'], body_params)
    model.create_model()
    assert model._body_mask is not None
    return np.asarray(model._body_mask)


def _data_region(obs: ObsSim, full: np.ndarray) -> np.ndarray:
    """Crop an extfov-coordinate mask to the data region."""
    ev = int(obs.extfov_margin_v)
    eu = int(obs.extfov_margin_u)
    return full[ev : ev + _SIZE, eu : eu + _SIZE]


def _mesh_params(**overrides: Any) -> dict[str, Any]:
    """Irregular-body params using the polyhedral mesh shape."""
    return _body_params(
        shape_model='polyhedral_mesh', mesh_lumpiness=0.45, mesh_seed=2, **overrides
    )


def test_mesh_prediction_differs_from_ellipsoid() -> None:
    """Predicting a mesh vs an ellipsoid of equal axes gives different masks."""
    obs = _obs()
    mesh = _predicted_mask(obs, _mesh_params())
    ellipsoid = _predicted_mask(obs, _body_params())
    assert int((mesh != ellipsoid).sum()) > 50


def test_mesh_prediction_pose_disagreement_changes_mask() -> None:
    """A different assumed pose changes the predicted mesh (chaotic-rotator)."""
    obs = _obs()
    true_pose = _predicted_mask(obs, _mesh_params(pose_euler_deg=[0.0, 0.0, 0.0]))
    wrong_pose = _predicted_mask(obs, _mesh_params(pose_euler_deg=[0.0, 90.0, 30.0]))
    assert int((true_pose != wrong_pose).sum()) > 50


def test_mesh_prediction_matches_rendered_data() -> None:
    """With identical params, the predicted mesh reproduces the rendered shape."""
    obs = _obs()
    body = _mesh_params()
    predicted = _data_region(obs, _predicted_mask(obs, body))
    img, _ = render_combined_model(
        {
            'size_v': _SIZE,
            'size_u': _SIZE,
            'random_seed': 1,
            'instrument': 'coiss_nac',
            'noise': {'poisson': False, 'read_noise_dn': 0.0, 'bias_dn': 0.0},
            'bodies': [body],
        }
    )
    rendered = img > 0
    intersection = int((predicted & rendered).sum())
    union = int((predicted | rendered).sum())
    assert intersection / union > 0.98


def test_mesh_prediction_is_deterministic() -> None:
    """The predicted mesh mask is reproduced exactly across builds."""
    obs = _obs()
    first = _predicted_mask(obs, _mesh_params())
    second = _predicted_mask(obs, _mesh_params())
    assert np.array_equal(first, second)


def test_ellipsoid_prediction_still_renders() -> None:
    """The default ellipsoid prediction path remains non-empty."""
    obs = _obs()
    ellipsoid = _predicted_mask(obs, _body_params())
    assert int(ellipsoid.sum()) > 0


_LIMB_SIZE = 220


def _limb_obs() -> ObsSim:
    """A coiss_nac sim obs sized for a well-resolved limb body."""
    return ObsSim.from_file(
        '/tmp/limb_sim.json',
        sim_params={'size_v': _LIMB_SIZE, 'size_u': _LIMB_SIZE, 'instrument': 'coiss_nac'},
    )


def _body_feature_types(obs: ObsSim, body_params: dict[str, Any]) -> list[str]:
    """Build the model and return the feature-type names it emits."""
    from typing import cast

    from spindoctor.nav_orchestrator.nav_context import NavContext

    model = NavModelBodySimulated('body', obs, body_params['name'], body_params)
    model.create_model()
    features = model.to_features(cast(NavContext, None))
    return [f.feature_type.name for f in features]


def _large_body(**overrides: Any) -> dict[str, Any]:
    """A well-resolved sphere (diameter 130 px) at low phase."""
    params = {
        'name': 'RHEA',
        'center_v': _LIMB_SIZE / 2.0,
        'center_u': _LIMB_SIZE / 2.0,
        'axis1': 130.0,
        'axis2': 130.0,
        'axis3': 130.0,
        'illumination_angle': 25.0,
        'phase_angle': 30.0,
    }
    params.update(overrides)
    return params


def test_large_low_phase_body_emits_limb_arc() -> None:
    """A well-resolved low-phase body emits a LIMB_ARC feature."""
    assert 'LIMB_ARC' in _body_feature_types(_limb_obs(), _large_body())


def test_small_body_emits_no_limb_arc() -> None:
    """A body below the limb-resolution floor emits no LIMB_ARC."""
    assert 'LIMB_ARC' not in _body_feature_types(
        _limb_obs(), _large_body(axis1=40.0, axis2=40.0, axis3=40.0)
    )


def test_high_phase_body_emits_no_limb_arc() -> None:
    """A high-phase body (mostly terminator) emits no LIMB_ARC."""
    assert 'LIMB_ARC' not in _body_feature_types(_limb_obs(), _large_body(phase_angle=120.0))


def test_limb_arc_polyline_has_vertices_and_unit_normals() -> None:
    """The emitted LIMB_ARC carries a vertex polyline with unit outward normals."""
    from typing import cast

    from spindoctor.feature.geometry import LimbPolyline
    from spindoctor.nav_orchestrator.nav_context import NavContext

    model = NavModelBodySimulated('body', _limb_obs(), 'RHEA', _large_body())
    model.create_model()
    limb = next(
        f for f in model.to_features(cast(NavContext, None)) if f.feature_type.name == 'LIMB_ARC'
    )
    geometry = limb.geometry
    assert isinstance(geometry, LimbPolyline)
    assert geometry.vertices_vu.shape[0] >= 30
    norms = np.hypot(geometry.normals_vu[:, 0], geometry.normals_vu[:, 1])
    assert np.allclose(norms, 1.0, atol=1e-9)

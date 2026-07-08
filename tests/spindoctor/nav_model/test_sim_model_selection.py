"""Model selection for simulated observations.

A simulated obs must build the sim-params-driven NavModels (not the SPICE-backed
ones), and the real models must opt out of a simulated obs.
"""

from typing import Any

from spindoctor.nav_model import build_models_for_obs
from spindoctor.nav_model.nav_model_body import NavModelBody
from spindoctor.nav_model.nav_model_body_simulated import NavModelBodySimulated
from spindoctor.obs.obs_inst_sim import ObsSim


class _FakeObs:
    """Minimal stand-in for the is_simulated guard tests."""

    def __init__(self, *, is_simulated: bool) -> None:
        self.is_simulated = is_simulated


def _sim_obs(bodies: list[dict[str, Any]]) -> ObsSim:
    return ObsSim.from_file(
        '/tmp/sel.json',
        sim_params={'size_v': 96, 'size_u': 96, 'instrument': 'coiss_nac', 'bodies': bodies},
    )


def _body(name: str, center: float) -> dict[str, Any]:
    return {
        'name': name,
        'center_v': center,
        'center_u': center,
        'axis1': 50.0,
        'axis2': 50.0,
        'axis3': 50.0,
        'illumination_angle': 20.0,
        'phase_angle': 30.0,
    }


def test_sim_obs_builds_simulated_body_model() -> None:
    """A simulated obs yields a NavModelBodySimulated for its body."""
    models = build_models_for_obs(_sim_obs([_body('RHEA', 48.0)]))
    assert any(isinstance(m, NavModelBodySimulated) for m in models)


def test_sim_obs_builds_one_model_per_body() -> None:
    """Each body in the sim scene gets its own simulated model."""
    models = build_models_for_obs(_sim_obs([_body('A', 30.0), _body('B', 66.0)]))
    body_models = [m for m in models if isinstance(m, NavModelBodySimulated)]
    assert len(body_models) == 2


def test_sim_obs_skips_spice_body_model() -> None:
    """The SPICE-backed body model is not built for a simulated obs."""
    models = build_models_for_obs(_sim_obs([_body('RHEA', 48.0)]))
    assert not any(isinstance(m, NavModelBody) for m in models)


def test_simulated_model_empty_for_real_obs() -> None:
    """NavModelBodySimulated builds nothing for a non-simulated obs."""
    assert NavModelBodySimulated.instances_for_obs(_FakeObs(is_simulated=False)) == []


def test_spice_model_empty_for_simulated_obs() -> None:
    """NavModelBody builds nothing for a simulated obs."""
    assert NavModelBody.instances_for_obs(_FakeObs(is_simulated=True)) == []

"""Tests that model selection honors a per-run config override (issue #146)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np

from spindoctor.config import Config
from spindoctor.nav_model.nav_model_body import NavModelBody
from spindoctor.nav_model.nav_model_rings import NavModelRings


class _FakeJupiterObs:
    """Observation stand-in exposing what ring-model selection inspects."""

    is_simulated = False
    closest_planet = 'jupiter'
    extdata_shape_vu = (10, 10)
    ext_bp = object()


def _config_with_jupiter_rings(tmp_path: Path) -> Config:
    override = tmp_path / 'override.yaml'
    override.write_text(
        'rings:\n  ring_features:\n    jupiter:\n      main:\n        fiducial_features: []\n'
    )
    config = Config()
    config.update_config(override)
    return config


def test_rings_selection_defaults_to_no_jupiter_model() -> None:
    # The bundled defaults carry no Jupiter ring catalog, so no model applies.
    instances = NavModelRings.instances_for_obs(cast(Any, _FakeJupiterObs()))
    assert instances == []


def test_rings_selection_honors_config_override(tmp_path: Path) -> None:
    # A per-run override that adds a Jupiter ring catalog must change which
    # models are instantiated, not just how the instances behave.
    config = _config_with_jupiter_rings(tmp_path)
    instances = NavModelRings.instances_for_obs(cast(Any, _FakeJupiterObs()), config=config)
    assert len(instances) == 1
    assert instances[0].name == 'rings:jupiter'
    assert instances[0]._config is config


class _FakeSaturnObs:
    """Observation stand-in with an inventory for body-model selection."""

    is_simulated = False
    closest_planet = 'saturn'

    def inventory(self, body_list: list[str], return_type: str = 'full') -> dict[str, Any]:
        """Report Saturn present in the FOV; no satellites visible."""
        del return_type
        entry = {'center_uv': np.array([5.0, 5.0]), 'u_pixel_size': 4.0, 'v_pixel_size': 4.0}
        return {name: entry for name in body_list if name.lower() == 'saturn'}

    def inventory_body_in_extfov(self, entry: dict[str, Any]) -> bool:
        """Every reported body is inside the extended FOV."""
        del entry
        return True


def test_body_selection_threads_config_to_instances(tmp_path: Path) -> None:
    # The selection stage and the constructed instances must share the same
    # per-run config object.
    config = _config_with_jupiter_rings(tmp_path)
    instances = NavModelBody.instances_for_obs(cast(Any, _FakeSaturnObs()), config=config)
    assert len(instances) == 1
    assert instances[0]._config is config

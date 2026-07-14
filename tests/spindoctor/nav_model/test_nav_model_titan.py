"""Tests for ``spindoctor.nav_model.nav_model_titan.NavModelTitan``.

The Titan model is built and active whenever Titan is in the FOV, but it
emits no features: it records, per image, that Titan navigation is not
supported.  Titan is a deliberate special case, not one entry of a general
atmospheric-body list.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np

from spindoctor.annotation import Annotations
from spindoctor.config import Config
from spindoctor.nav_model.nav_model_body import NavModelBody
from spindoctor.nav_model.nav_model_titan import NavModelTitan


class _FakeObs:
    """Minimal obs stand-in for the Titan model."""

    midtime: float = 0.0


class _FakeSaturnObs:
    """Observation stand-in exposing the inventory body-model selection reads."""

    is_simulated = False
    closest_planet = 'saturn'

    def inventory(self, body_list: list[str], return_type: str = 'full') -> dict[str, Any]:
        """Report every requested body as present in the FOV."""
        del return_type
        entry = {'center_uv': np.array([5.0, 5.0]), 'u_pixel_size': 4.0, 'v_pixel_size': 4.0}
        return dict.fromkeys(body_list, entry)

    def inventory_body_in_extfov(self, entry: dict[str, Any]) -> bool:
        """Every reported body is inside the extended FOV."""
        del entry
        return True


def _config_with_titan_only_satellites(tmp_path: Path) -> Config:
    override = tmp_path / 'satellites_override.yaml'
    override.write_text('satellites:\n  SATURN:\n    - TITAN\n')
    config = Config()
    config.update_config(override)
    return config


def test_titan_model_built_for_titan_in_fov(tmp_path: Path) -> None:
    """Titan in the FOV builds one active NavModelTitan."""
    config = _config_with_titan_only_satellites(tmp_path)
    instances = NavModelTitan.instances_for_obs(cast(Any, _FakeSaturnObs()), config=config)
    assert [m.name for m in instances] == ['titan:TITAN']


def test_titan_model_reports_titan_in_fov(tmp_path: Path) -> None:
    """The built model reports Titan in the FOV for the orchestrator."""
    config = _config_with_titan_only_satellites(tmp_path)
    instances = NavModelTitan.instances_for_obs(cast(Any, _FakeSaturnObs()), config=config)
    model = cast(NavModelTitan, instances[0])
    assert model.titan_in_fov is True


def test_body_model_excludes_titan(tmp_path: Path) -> None:
    """The shape-based body model builds nothing for Titan."""
    config = _config_with_titan_only_satellites(tmp_path)
    instances = NavModelBody.instances_for_obs(cast(Any, _FakeSaturnObs()), config=config)
    assert all('TITAN' not in m.name.upper() for m in instances)


def test_create_model_records_titan_metadata() -> None:
    """``create_model`` records the body and the non-navigable flag."""
    obs: Any = _FakeObs()
    model = NavModelTitan('titan:TITAN', obs)
    model.create_model()
    assert model.metadata == {'body': 'TITAN', 'navigable': False}


def test_create_model_logs_unsupported_reason(capsys: Any) -> None:
    """``create_model`` logs a clear reason naming Titan."""
    obs: Any = _FakeObs()
    model = NavModelTitan('titan:TITAN', obs)
    model.create_model()
    captured = capsys.readouterr()
    assert 'Titan in FOV: navigation not supported' in captured.out


def test_to_features_returns_empty() -> None:
    """``to_features`` returns an empty list -- Titan navigation is unsupported."""
    obs: Any = _FakeObs()
    model = NavModelTitan('titan:TITAN', obs)
    model.create_model()
    assert model.to_features(context=None) == []  # type: ignore[arg-type]


def test_to_annotations_returns_empty_collection() -> None:
    """``to_annotations`` returns an empty Annotations collection."""
    obs: Any = _FakeObs()
    model = NavModelTitan('titan:TITAN', obs)
    model.create_model()
    annotations = model.to_annotations(context=None)  # type: ignore[arg-type]
    assert isinstance(annotations, Annotations)
    assert len(annotations.annotations) == 0

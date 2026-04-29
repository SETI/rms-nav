"""Smoke tests for ``nav.nav_technique.nav_technique_manual``.

The dialog itself is GUI-only and is exercised by hand; these tests
mock ``ManualNavDialog.run_modal`` so the technique logic
(``run_manual_nav`` orchestration + accept / cancel translation) can be
asserted without firing up Qt.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from nav.annotation import Annotations
from nav.feature.feature import NavFeature, NavReliabilityBreakdown
from nav.feature.feature_type import NavFeatureType
from nav.feature.flags import StarFlags
from nav.feature.geometry import StarGeometry
from nav.nav_model import NavModel
from nav.nav_technique import NavTechniqueManual
from nav.nav_technique.nav_technique_manual import run_manual_nav
from nav.support.filters import NavFilterKind, NavFilterSpec


class _FakeObsForRunManual:
    """Minimal stand-in for ObsSnapshotInst sufficient for run_manual_nav."""

    def __init__(self) -> None:
        rng = np.random.default_rng(seed=7)
        self.data = rng.standard_normal((48, 48)) + 100.0
        self.extdata = self.data
        self._sensor_mask = np.ones(self.data.shape, bool)
        self.midtime = 0.0

    def extfov_data_sensor_mask(self) -> np.ndarray:
        return self._sensor_mask


class _StubStarModel(NavModel):
    """NavModel that emits a single STAR feature with a tiny template."""

    def __init__(self, obs: Any, *, with_template: bool) -> None:
        super().__init__('stars', obs)
        self._with_template = with_template
        self._obs = obs

    def create_model(self) -> None:
        return None

    def to_features(self, context: Any) -> list[NavFeature]:
        template_img: np.ndarray | None
        template_mask: np.ndarray | None
        if self._with_template:
            template_img = np.zeros(self._obs.extdata.shape, dtype=np.float64)
            template_img[10:13, 10:13] = 1.0
            template_mask = template_img > 0
        else:
            template_img = None
            template_mask = None
        return [
            NavFeature(
                feature_id='star:test:0',
                feature_type=NavFeatureType.STAR,
                source_model='stars',
                geometry=StarGeometry(
                    predicted_vu=(11.0, 11.0),
                    catalog_vu=(11.0, 11.0),
                    bbox_extfov_vu=(8, 8, 14, 14),
                ),
                subject_range_km=1e10,
                position_cov_px=np.eye(2, dtype=np.float64) * 0.25,
                intensity_sigma_rel=0.05,
                preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
                reliability=0.9,
                reliability_reasons=NavReliabilityBreakdown(predicted_snr=10.0),
                usable_types=frozenset({NavFeatureType.STAR}),
                flags=StarFlags(),
                template_img=template_img,
                template_mask=template_mask,
            )
        ]

    def to_annotations(self, context: Any) -> Annotations:
        return Annotations()


@pytest.fixture(autouse=True)
def _patch_build_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``build_models_for_obs`` deterministic for these tests.

    ``run_manual_nav`` calls the autoregistered builder; replacing it
    with a single-stub-model factory keeps the test self-contained.
    """
    from nav.nav_technique import nav_technique_manual

    def _stub_builder(obs: Any) -> list[NavModel]:
        return [_StubStarModel(obs, with_template=getattr(obs, 'with_template', True))]

    monkeypatch.setattr(
        nav_technique_manual,
        'build_models_for_obs',
        _stub_builder,
        raising=False,
    )
    # The import inside run_manual_nav happens at call time; patch the
    # source binding too so the local import in the function picks the stub.
    import nav.nav_model as nav_model_pkg

    monkeypatch.setattr(nav_model_pkg, 'build_models_for_obs', _stub_builder, raising=False)


def _make_fake_dialog(
    next_modal_return: tuple[bool, tuple[float, float] | None, float | None],
) -> tuple[type, list[Any]]:
    """Build a fresh per-test ``ManualNavDialog`` stub.

    Each call returns a brand-new class plus the per-test instances list so
    no shared state leaks between pytest-xdist workers.  The previous design
    used a single shared `_FakeDialog` whose ClassVar mutable state raced
    when tests ran in parallel.
    """
    captured_return = next_modal_return
    instances: list[Any] = []

    class _FakeDialog:
        def __init__(self, **_kwargs: Any) -> None:
            instances.append(self)

        def run_modal(self) -> tuple[bool, tuple[float, float] | None, float | None]:
            return captured_return

    return _FakeDialog, instances


def test_run_manual_nav_returns_result_on_accept() -> None:
    """When the dialog returns ``accepted=True`` the technique result is non-spurious."""
    obs = _FakeObsForRunManual()
    obs.with_template = True  # type: ignore[attr-defined]
    fake_dialog, instances = _make_fake_dialog((True, (3.0, -2.0), 0.92))
    with patch('nav.ui.manual_nav_dialog.ManualNavDialog', fake_dialog):
        result = run_manual_nav(obs)  # type: ignore[arg-type]
    assert result is not None
    assert result.spurious is False
    assert result.offset_px == (3.0, -2.0)
    assert result.technique_name == NavTechniqueManual.name
    assert len(instances) == 1


def test_run_manual_nav_returns_spurious_on_cancel() -> None:
    """When the dialog is cancelled the technique result is spurious / zero."""
    obs = _FakeObsForRunManual()
    obs.with_template = True  # type: ignore[attr-defined]
    fake_dialog, _instances = _make_fake_dialog((False, None, None))
    with patch('nav.ui.manual_nav_dialog.ManualNavDialog', fake_dialog):
        result = run_manual_nav(obs)  # type: ignore[arg-type]
    assert result is not None
    assert result.spurious is True
    assert result.confidence == 0.0


def test_run_manual_nav_returns_none_when_no_template_features(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No template-bearing feature → dialog is not opened, ``None`` is returned."""
    obs = _FakeObsForRunManual()
    obs.with_template = False  # type: ignore[attr-defined]
    result = run_manual_nav(obs)  # type: ignore[arg-type]
    assert result is None
    captured = capsys.readouterr()
    assert 'Manual navigation skipped' in captured.out


class _StubLimbModel(NavModel):
    """NavModel that emits a single LIMB_ARC feature (polyline only, no template)."""

    def __init__(self, obs: Any) -> None:
        super().__init__('body', obs)
        self._obs = obs

    def create_model(self) -> None:
        return None

    def to_features(self, context: Any) -> list[NavFeature]:
        from nav.feature.flags import LimbArcFlags
        from nav.feature.geometry import LimbPolyline

        verts = np.array([[10.0, 10.0], [12.0, 12.0], [14.0, 14.0]], dtype=np.float64)
        return [
            NavFeature(
                feature_id='limb_arc:test',
                feature_type=NavFeatureType.LIMB_ARC,
                source_model='body',
                geometry=LimbPolyline(
                    vertices_vu=verts,
                    normals_vu=np.zeros_like(verts),
                    sigma_normal_per_vertex_px=np.full(verts.shape[0], 0.5),
                    sigma_tangent_per_vertex_px=np.full(verts.shape[0], 0.5),
                    bbox_extfov_vu=(8, 8, 16, 16),
                ),
                subject_range_km=1e6,
                position_cov_px=None,
                intensity_sigma_rel=0.0,
                preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
                reliability=0.9,
                reliability_reasons=NavReliabilityBreakdown(visible_arc_fraction=1.0),
                usable_types=frozenset({NavFeatureType.LIMB_ARC}),
                flags=LimbArcFlags(body_name='X', visible_arc_fraction=1.0),
            )
        ]

    def to_annotations(self, context: Any) -> Annotations:
        return Annotations()


def test_run_manual_nav_runs_for_polyline_only_scene(monkeypatch: pytest.MonkeyPatch) -> None:
    """A scene with only LIMB_ARC features (no full-disc templates) is still
    feasible — the dialog gets a polyline-rasterized overlay."""
    obs = _FakeObsForRunManual()
    import nav.nav_model as nav_model_pkg
    from nav.nav_technique import nav_technique_manual

    def _limb_only_builder(obs_arg: Any) -> list[NavModel]:
        return [_StubLimbModel(obs_arg)]

    monkeypatch.setattr(
        nav_technique_manual, 'build_models_for_obs', _limb_only_builder, raising=False
    )
    monkeypatch.setattr(nav_model_pkg, 'build_models_for_obs', _limb_only_builder, raising=False)

    fake_dialog, instances = _make_fake_dialog((True, (1.0, 2.0), 0.5))
    with patch('nav.ui.manual_nav_dialog.ManualNavDialog', fake_dialog):
        result = run_manual_nav(obs)  # type: ignore[arg-type]
    assert result is not None
    assert result.spurious is False
    assert result.offset_px == (1.0, 2.0)
    assert len(instances) == 1

"""Integration tests for ``nav.nav_orchestrator.orchestrator.NavOrchestrator``."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from nav.annotation import Annotations
from nav.feature.feature import (
    NavFeature,
    NavReliabilityBreakdown,
    body_names_from_features,
)
from nav.feature.feature_type import NavFeatureType
from nav.feature.flags import BodyBlobFlags, BodyDiscFlags, StarFlags
from nav.feature.geometry import (
    BodyBlobGeometry,
    BodyDiscGeometry,
    StarGeometry,
)
from nav.nav_model import NavModel
from nav.nav_orchestrator.image_classifier import ImageQualityThresholds
from nav.nav_orchestrator.nav_context import NavContext
from nav.nav_orchestrator.orchestrator import NavOrchestrator
from nav.nav_technique.diagnostics import StarFieldDiagnostics
from nav.nav_technique.feasibility import NavFeasibilityReport
from nav.nav_technique.nav_technique import NavTechnique
from nav.nav_technique.technique_result import NavTechniqueResult
from nav.support.filters import NavFilterKind, NavFilterSpec
from nav.support.status_reason import NavStatusReason


class _FakeObs:
    """Minimal object satisfying the orchestrator's obs.* attribute access."""

    def __init__(
        self,
        *,
        image: np.ndarray | None = None,
        sensor_mask: np.ndarray | None = None,
        midtime: float = 0.0,
        extfov_margin: tuple[int, int] = (0, 0),
        inst_config: dict[str, Any] | None = None,
    ) -> None:
        self.inst_config = inst_config
        if image is None:
            rng = np.random.default_rng(seed=42)
            image = rng.standard_normal(size=(64, 64)) + 100.0
        self.data = image
        # Build an extfov-padded image around ``data`` so the fake matches
        # the real obs API: ``extdata`` is the canonical input the
        # orchestrator reads.
        margin_v, margin_u = extfov_margin
        ext_shape = (image.shape[0] + 2 * margin_v, image.shape[1] + 2 * margin_u)
        ext = np.zeros(ext_shape, dtype=image.dtype)
        ext[margin_v : margin_v + image.shape[0], margin_u : margin_u + image.shape[1]] = image
        self.extdata = ext
        if sensor_mask is None:
            sensor_mask = np.zeros(ext_shape, bool)
            sensor_mask[
                margin_v : margin_v + image.shape[0],
                margin_u : margin_u + image.shape[1],
            ] = True
        self._sensor_mask = sensor_mask
        self.midtime = midtime

    def extfov_data_sensor_mask(self) -> np.ndarray:
        return self._sensor_mask


class _FakeStarModel(NavModel):
    """Fake NavModel that emits one STAR feature per ``feature_count``."""

    def __init__(self, obs: Any, *, feature_count: int = 1) -> None:
        super().__init__('stars', obs)
        self._feature_count = feature_count

    def create_model(self) -> None:
        self._metadata['feature_count'] = self._feature_count

    def to_features(self, context: NavContext) -> list[NavFeature]:
        features: list[NavFeature] = []
        for i in range(self._feature_count):
            features.append(
                NavFeature(
                    feature_id=f'star:test:{i}',
                    feature_type=NavFeatureType.STAR,
                    source_model='stars',
                    geometry=StarGeometry(
                        predicted_vu=(float(10 + i), float(20 + i)),
                        catalog_vu=(float(10 + i), float(20 + i)),
                        bbox_extfov_vu=(0, 0, 16, 16),
                    ),
                    subject_range_km=1.0e10,
                    position_cov_px=np.eye(2, dtype=np.float64) * 0.25,
                    intensity_sigma_rel=0.05,
                    preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
                    reliability=0.8,
                    reliability_reasons=NavReliabilityBreakdown(predicted_snr=10.0),
                    usable_types=frozenset({NavFeatureType.STAR}),
                    flags=StarFlags(),
                )
            )
        return features

    def to_annotations(self, context: NavContext) -> Annotations:
        return Annotations()


class _FakeStarTechnique(NavTechnique):
    """Stand-in technique that always reports a fixed offset."""

    name = '_FakeStarTechnique'
    accepts_feature_types = frozenset({NavFeatureType.STAR})

    def is_feasible(self, features: list[NavFeature]) -> NavFeasibilityReport:
        if len(features) >= 1:
            return NavFeasibilityReport(
                feasible=True, reason='ok', consumed_feature_count=len(features)
            )
        return NavFeasibilityReport(feasible=False, reason='no_stars')

    def navigate(self, features: list[NavFeature], context: NavContext) -> NavTechniqueResult:
        return NavTechniqueResult(
            technique_name=self.name,
            feature_ids=tuple(f.feature_id for f in features),
            offset_px=(1.5, 2.5),
            covariance_px2=np.eye(2, dtype=np.float64) * 0.25,
            confidence=0.85,
            spurious=False,
            at_edge=False,
            diagnostics=StarFieldDiagnostics(n_inliers=len(features)),
        )


@pytest.fixture
def fake_obs() -> _FakeObs:
    """Provide a clean fake observation."""
    return _FakeObs()


def test_orchestrator_runs_pipeline_end_to_end(fake_obs: _FakeObs) -> None:
    """A clean image + 1 star model + 1 technique -> status='success'."""
    obs = fake_obs
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator(
        [model],
        only_techniques=['_FakeStarTechnique'],
    )
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert result.status == 'success'
    assert result.offset_px == (1.5, 2.5)
    assert [t.technique_name for t in result.per_technique] == ['_FakeStarTechnique']
    assert result.confidence_rank == 'high'


def test_orchestrator_handles_nonzero_extfov_margin() -> None:
    """The pipeline accepts obs whose ``extdata`` is larger than ``data``.

    The image-quality classifier consumes ``obs.extdata`` and
    ``obs.extfov_data_sensor_mask()`` together, which must share the
    extended-FOV shape regardless of the per-instrument extfov margin.
    """
    obs = _FakeObs(
        image=np.full((64, 64), 100.0, dtype=np.float64),
        extfov_margin=(8, 12),
    )
    assert obs.extdata.shape == (80, 88)
    assert obs.extfov_data_sensor_mask().shape == (80, 88)
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator([model], only_techniques=['_FakeStarTechnique'])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert result.status == 'success'


def test_orchestrator_blank_image_short_circuits(fake_obs: _FakeObs) -> None:
    """A blank image yields status='failed' before any technique runs."""
    obs = _FakeObs(image=np.zeros((64, 64), np.float64))
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator([model])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert result.status == 'failed'
    assert result.status_reason == NavStatusReason.NO_SIGNAL_IN_IMAGE
    assert result.per_technique == []


def test_orchestrator_no_features_emitted_yields_no_features_extracted(
    fake_obs: _FakeObs,
) -> None:
    """A NavModel emitting zero features yields no_features_extracted."""
    obs = fake_obs
    model = _FakeStarModel(obs, feature_count=0)
    orch = NavOrchestrator([model])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert result.status == 'failed'
    assert result.status_reason == NavStatusReason.NO_FEATURES_EXTRACTED


def test_normalize_model_patterns_expands_bare_prefix() -> None:
    """A bare prefix without colon expands to ``prefix:*`` so ``rings`` matches ``rings:SATURN``."""
    from nav.nav_orchestrator.orchestrator import _normalize_model_patterns

    names = ['stars', 'rings:SATURN', 'body:DIONE', 'body:SATURN']
    out = _normalize_model_patterns('rings', names)
    assert 'rings:*' in out


def test_normalize_model_patterns_uppercases_value_after_colon() -> None:
    """``body:saturn`` is normalized to ``body:SATURN``."""
    from nav.nav_orchestrator.orchestrator import _normalize_model_patterns

    names = ['body:SATURN', 'body:DIONE']
    assert _normalize_model_patterns('body:saturn', names) == ['body:SATURN']


def test_normalize_model_patterns_preserves_bang_exclusion() -> None:
    """A leading ``!`` is preserved through the normalization."""
    from nav.nav_orchestrator.orchestrator import _normalize_model_patterns

    names = ['stars', 'rings:SATURN', 'body:DIONE']
    out = _normalize_model_patterns('!rings', names)
    assert '!rings:*' in out


def test_normalize_model_patterns_preserves_star_token_for_unnamespaced_models() -> None:
    """``stars`` (with no namespace) still matches ``stars`` after normalization."""
    from nav.nav_orchestrator.orchestrator import _normalize_model_patterns

    names = ['stars', 'rings:SATURN']
    # With both styles in the registry the bare token should match both
    # the literal name and any namespaced variant.
    out = _normalize_model_patterns('stars', names)
    assert 'stars' in out  # matches 'stars'
    assert 'stars:*' in out  # matches potential 'stars:FOO'


def test_orchestrator_only_models_filter_drops_models(fake_obs: _FakeObs) -> None:
    """only_models='!stars' drops the stars model entirely."""
    obs = fake_obs
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator([model], only_models='!stars')
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert result.status == 'failed'
    assert result.status_reason == NavStatusReason.NO_FEATURES_EXTRACTED


def test_orchestrator_only_techniques_filter_drops_techniques(
    fake_obs: _FakeObs,
) -> None:
    """An exclusion glob that rejects every STAR technique yields ``no_feasible_techniques``.

    The exclude list covers ``_FakeStarTechnique`` plus every shipped
    STAR-accepting NavTechnique so the only feasible matches against
    the fake STAR features are filtered out, leaving an empty pass-1
    result list and the corresponding status reason.
    """
    obs = fake_obs
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator(
        [model],
        only_techniques=[
            '!_FakeStarTechnique',
            '!_RaisingTechnique',
            '!Star*',
        ],
    )
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert result.status == 'failed'
    assert result.status_reason == NavStatusReason.NO_FEASIBLE_TECHNIQUES


def test_orchestrator_only_models_mixed_include_exclude(fake_obs: _FakeObs) -> None:
    """Mixed include/exclude patterns on ``only_models`` apply both gates.

    ``only_models`` accepts the same glob-with-negation grammar as
    ``only_techniques``: include patterns admit every match; the
    leading-bang exclusion drops any name matching the exclude
    pattern, applied after inclusion.
    """
    obs = fake_obs
    model = _FakeStarModel(obs, feature_count=3)
    # Include everything ('*') but exclude names starting with 'st' (the
    # stars model).  No models survive, so feature extraction yields the
    # NO_FEATURES_EXTRACTED status the single-pattern test already covers.
    orch = NavOrchestrator([model], only_models=['*', '!st*'])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert result.status == 'failed'
    assert result.status_reason == NavStatusReason.NO_FEATURES_EXTRACTED


def test_orchestrator_only_models_mixed_keeps_matching_inclusion(
    fake_obs: _FakeObs,
) -> None:
    """Mixed pattern ``['stars', '!ring*']`` keeps ``stars`` (no exclusion match)."""
    obs = fake_obs
    model = _FakeStarModel(obs, feature_count=3)
    # Include only 'stars'; exclude any 'ring*' (no match — kept).
    orch = NavOrchestrator(
        [model],
        only_models=['stars', '!ring*'],
        only_techniques=['_FakeStarTechnique'],
    )
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert result.status == 'success'


def test_orchestrator_only_techniques_mixed_include_exclude(
    fake_obs: _FakeObs,
) -> None:
    """Mixed include/exclude patterns on ``only_techniques`` apply both gates."""
    obs = fake_obs
    model = _FakeStarModel(obs, feature_count=3)
    # Include everything but exclude techniques whose name contains 'Pass'.
    # ``_FakeStarTechnique`` survives; ``_PassTwoTechnique`` (registered
    # later in this module) does not.
    orch = NavOrchestrator(
        [model],
        only_techniques=['*', '!*Pass*'],
    )
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert result.status == 'success'
    technique_names = {t.technique_name for t in result.per_technique}
    assert '_FakeStarTechnique' in technique_names
    assert '_PassTwoTechnique' not in technique_names


def test_orchestrator_marks_technique_results_in_inventory(
    fake_obs: _FakeObs,
) -> None:
    """Successful runs carry a feature_inventory listing the kept features."""
    obs = fake_obs
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator([model])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert len(result.feature_inventory) == 3
    assert all(not entry.gated for entry in result.feature_inventory)
    assert all(entry.gate_reason is None for entry in result.feature_inventory)


def test_orchestrator_passes_classifier_thresholds_through(fake_obs: _FakeObs) -> None:
    """Custom ImageQualityThresholds change the classification verdict."""
    image = np.full((64, 64), 4095.0, np.float64)
    obs = _FakeObs(image=image)
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator(
        [model],
        image_quality_thresholds=ImageQualityThresholds(saturation_threshold_dn=4095.0),
    )
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert result.status_reason == NavStatusReason.IMAGE_OVEREXPOSED


class _RaisingModel(NavModel):
    """NavModel whose ``to_features`` always raises a RuntimeError."""

    def __init__(self, obs: Any) -> None:
        super().__init__('raising', obs)

    def create_model(self) -> None:
        self._metadata['raised'] = True

    def to_features(self, context: NavContext) -> list[NavFeature]:
        raise RuntimeError('synthetic to_features failure')

    def to_annotations(self, context: NavContext) -> Annotations:
        raise RuntimeError('synthetic to_annotations failure')


class _RaisingTechnique(NavTechnique):
    """NavTechnique whose ``navigate`` always raises a RuntimeError."""

    name = '_RaisingTechnique'
    accepts_feature_types = frozenset({NavFeatureType.STAR})

    def is_feasible(self, features: list[NavFeature]) -> NavFeasibilityReport:
        return NavFeasibilityReport(
            feasible=True, reason='ok', consumed_feature_count=len(features)
        )

    def navigate(self, features: list[NavFeature], context: NavContext) -> NavTechniqueResult:
        raise RuntimeError('synthetic navigate failure')


class _PassTwoTechnique(NavTechnique):
    """Pass-2 (requires_prior=True) technique that captures the prior offset."""

    name = '_PassTwoTechnique'
    accepts_feature_types = frozenset({NavFeatureType.STAR})
    requires_prior = True
    captured_prior: tuple[float, float] | None = None

    def is_feasible(self, features: list[NavFeature]) -> NavFeasibilityReport:
        return NavFeasibilityReport(feasible=True, reason='ok')

    def navigate(self, features: list[NavFeature], context: NavContext) -> NavTechniqueResult:
        type(self).captured_prior = context.prior_offset_px
        return NavTechniqueResult(
            technique_name=self.name,
            feature_ids=tuple(f.feature_id for f in features),
            offset_px=(1.5, 2.5),
            covariance_px2=np.eye(2, dtype=np.float64) * 0.04,
            confidence=0.9,
            spurious=False,
            at_edge=False,
            diagnostics=StarFieldDiagnostics(n_inliers=len(features)),
        )


def test_orchestrator_logs_when_to_features_raises(
    fake_obs: _FakeObs, capsys: pytest.CaptureFixture[str]
) -> None:
    """A misbehaving NavModel.to_features is logged and treated as zero features."""
    obs = fake_obs
    model = _RaisingModel(obs)
    orch = NavOrchestrator([model])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    captured = capsys.readouterr()
    assert result.status_reason == NavStatusReason.NO_FEATURES_EXTRACTED
    assert 'to_features raised' in captured.out
    assert 'synthetic to_features failure' in captured.out


def test_orchestrator_logs_when_technique_raises(
    fake_obs: _FakeObs, capsys: pytest.CaptureFixture[str]
) -> None:
    """A misbehaving NavTechnique.navigate is logged and treated as no result."""
    obs = fake_obs
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator([model], only_techniques='_RaisingTechnique')
    result = orch.navigate(obs)  # type: ignore[arg-type]
    captured = capsys.readouterr()
    assert result.status_reason == NavStatusReason.NO_FEASIBLE_TECHNIQUES
    assert 'navigate raised' in captured.out
    assert 'synthetic navigate failure' in captured.out


def test_orchestrator_pass2_receives_pass1_prior(fake_obs: _FakeObs) -> None:
    """Pass-2 techniques observe the pass-1 ensemble's offset as ``prior_offset_px``."""
    obs = fake_obs
    model = _FakeStarModel(obs, feature_count=3)
    _PassTwoTechnique.captured_prior = None
    orch = NavOrchestrator(
        [model],
        only_techniques=['_FakeStarTechnique', '_PassTwoTechnique'],
    )
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert result.status == 'success'
    # The pass-2 technique captured the pass-1 ensemble's offset (which equals
    # the single _FakeStarTechnique offset of (1.5, 2.5)).
    assert _PassTwoTechnique.captured_prior == (1.5, 2.5)


def test_orchestrator_records_model_metadata(fake_obs: _FakeObs) -> None:
    """The NavResult.model_metadata dict is populated from each model.metadata."""
    obs = fake_obs
    model = _FakeStarModel(obs, feature_count=2)
    orch = NavOrchestrator([model])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert 'stars' in result.model_metadata
    assert result.model_metadata['stars']['feature_count'] == 2


def test_orchestrator_records_annotations(fake_obs: _FakeObs) -> None:
    """Annotations from every NavModel are merged into NavResult.annotations."""
    obs = fake_obs
    model = _FakeStarModel(obs, feature_count=2)
    orch = NavOrchestrator([model])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert isinstance(result.annotations, Annotations)


def test_orchestrator_low_reliability_features_all_gated(fake_obs: _FakeObs) -> None:
    """Every feature below the gate yields ALL_FEATURES_GATED."""
    obs = fake_obs

    class _LowReliabilityStarModel(_FakeStarModel):
        def to_features(self, context: NavContext) -> list[NavFeature]:
            features = super().to_features(context)
            return [
                NavFeature(
                    feature_id=f.feature_id,
                    feature_type=f.feature_type,
                    source_model=f.source_model,
                    geometry=f.geometry,
                    subject_range_km=f.subject_range_km,
                    position_cov_px=f.position_cov_px,
                    intensity_sigma_rel=f.intensity_sigma_rel,
                    preferred_filter=f.preferred_filter,
                    reliability=0.01,
                    reliability_reasons=f.reliability_reasons,
                    usable_types=f.usable_types,
                    flags=f.flags,
                )
                for f in features
            ]

    model = _LowReliabilityStarModel(obs, feature_count=3)
    orch = NavOrchestrator([model])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert result.status_reason == NavStatusReason.ALL_FEATURES_GATED


def test_collect_annotations_skips_failing_model(
    fake_obs: _FakeObs, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failing to_annotations doesn't break the rest of the pipeline."""
    obs = fake_obs

    class _GoodModel(_FakeStarModel):
        pass

    class _BadAnnotationModel(_FakeStarModel):
        def to_annotations(self, context: NavContext) -> Annotations:
            raise RuntimeError('synthetic to_annotations failure')

    bad = _BadAnnotationModel(obs, feature_count=2)
    good = _GoodModel(obs, feature_count=1)
    orch = NavOrchestrator([bad, good], only_techniques=['_FakeStarTechnique'])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    captured = capsys.readouterr()
    assert result.status == 'success'
    assert 'to_annotations raised' in captured.out
    assert 'synthetic to_annotations failure' in captured.out


def test_prepare_returns_context_and_features(fake_obs: _FakeObs) -> None:
    """``prepare`` builds the context and returns the gated feature list."""
    obs = fake_obs
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator([model])
    prep = orch.prepare(obs)  # type: ignore[arg-type]
    assert prep.context.obs is obs
    assert len(prep.features) == 3
    assert {f.feature_type for f in prep.features} == {NavFeatureType.STAR}


def test_prepare_does_not_short_circuit_on_blank_image() -> None:
    """``prepare`` keeps going on hard-failure images so manual nav can inspect them.

    ``navigate`` returns a failed NavResult on a blank image, but
    ``prepare`` is the entry point for the manual-nav dialog — the
    operator may legitimately want to look at a blank or saturated frame.
    """
    obs = _FakeObs(image=np.zeros((64, 64), np.float64))
    model = _FakeStarModel(obs, feature_count=0)
    orch = NavOrchestrator([model])
    prep = orch.prepare(obs)  # type: ignore[arg-type]
    assert prep.context.image_classifier.image_class == 'blank'
    assert prep.features == []


def test_prepare_drops_models_via_only_models_filter() -> None:
    """The ``only_models`` glob filter is honored by ``prepare`` too."""
    obs = _FakeObs()
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator([model], only_models='!stars')
    prep = orch.prepare(obs)  # type: ignore[arg-type]
    assert prep.features == []


def test_prepare_apply_gate_false_returns_gated_features() -> None:
    """``apply_gate=False`` returns every emitted feature, including gated ones."""

    class _LowReliabilityModel(_FakeStarModel):
        def to_features(self, context: NavContext) -> list[NavFeature]:
            features = super().to_features(context)
            return [
                NavFeature(
                    feature_id=f.feature_id,
                    feature_type=f.feature_type,
                    source_model=f.source_model,
                    geometry=f.geometry,
                    subject_range_km=f.subject_range_km,
                    position_cov_px=f.position_cov_px,
                    intensity_sigma_rel=f.intensity_sigma_rel,
                    preferred_filter=f.preferred_filter,
                    reliability=0.01,
                    reliability_reasons=f.reliability_reasons,
                    usable_types=f.usable_types,
                    flags=f.flags,
                )
                for f in features
            ]

    obs = _FakeObs()
    model = _LowReliabilityModel(obs, feature_count=2)
    orch = NavOrchestrator([model])
    gated_prep = orch.prepare(obs, apply_gate=True)  # type: ignore[arg-type]
    full_prep = orch.prepare(obs, apply_gate=False)  # type: ignore[arg-type]
    assert gated_prep.features == []
    assert len(full_prep.features) == 2


# --- Pass-1 fallback-skip regression tests ---


class _FakeBodyModel(NavModel):
    """Fake NavModel that emits one BODY_DISC + one BODY_BLOB for one body."""

    def __init__(self, obs: Any, *, body_name: str = 'TestMoon') -> None:
        super().__init__('body:test', obs)
        self._body_name = body_name

    def create_model(self) -> None:
        self._metadata['feature_count'] = 2

    def to_features(self, context: NavContext) -> list[NavFeature]:
        disc = NavFeature(
            feature_id=f'body_disc:{self._body_name}',
            feature_type=NavFeatureType.BODY_DISC,
            source_model='body:test',
            geometry=BodyDiscGeometry(
                bbox_extfov_vu=(0, 0, 16, 16),
                predicted_center_vu=(8.0, 8.0),
                overflow_fraction=0.0,
            ),
            usable_types=frozenset({NavFeatureType.BODY_DISC}),
            flags=BodyDiscFlags(body_name=self._body_name),
            subject_range_km=1.0e8,
            position_cov_px=np.eye(2, dtype=np.float64) * 0.25,
            intensity_sigma_rel=0.05,
            preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
            reliability=0.8,
            reliability_reasons=NavReliabilityBreakdown(visible_lit_fraction=1.0),
        )
        blob = NavFeature(
            feature_id=f'body_blob:{self._body_name}',
            feature_type=NavFeatureType.BODY_BLOB,
            source_model='body:test',
            geometry=BodyBlobGeometry(
                bbox_extfov_vu=(0, 0, 16, 16),
                predicted_center_vu=(8.0, 8.0),
                predicted_diameter_px=8.0,
            ),
            usable_types=frozenset({NavFeatureType.BODY_BLOB}),
            flags=BodyBlobFlags(body_name=self._body_name, predicted_diameter_px=8.0),
            subject_range_km=1.0e8,
            position_cov_px=np.eye(2, dtype=np.float64) * 0.25,
            intensity_sigma_rel=0.05,
            preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
            reliability=0.8,
            reliability_reasons=NavReliabilityBreakdown(visible_lit_fraction=1.0),
        )
        return [disc, blob]

    def to_annotations(self, context: NavContext) -> Annotations:
        return Annotations()


class _FakeBodyPrimary(NavTechnique):
    """Primary technique consuming BODY_DISC; tracks how many times it ran."""

    name = '_FakeBodyPrimary'
    accepts_feature_types = frozenset({NavFeatureType.BODY_DISC})
    tier = 'primary'
    run_count: int = 0
    spurious_override: bool = False

    def is_feasible(self, features: list[NavFeature]) -> NavFeasibilityReport:
        eligible = [f for f in features if f.feature_type is NavFeatureType.BODY_DISC]
        if not eligible:
            return NavFeasibilityReport(feasible=False, reason='no_body_disc')
        return NavFeasibilityReport(
            feasible=True, reason='ok', consumed_feature_count=len(eligible)
        )

    def navigate(self, features: list[NavFeature], context: NavContext) -> NavTechniqueResult:
        type(self).run_count += 1
        return NavTechniqueResult(
            technique_name=self.name,
            feature_ids=tuple(f.feature_id for f in features),
            offset_px=(1.0, 2.0),
            covariance_px2=np.eye(2, dtype=np.float64) * 0.25,
            confidence=0.85,
            spurious=type(self).spurious_override,
            at_edge=False,
            diagnostics=StarFieldDiagnostics(n_inliers=len(features)),
            source_bodies=body_names_from_features(features),
        )


class _FakeBodyFallback(NavTechnique):
    """Fallback technique consuming BODY_BLOB; tracks how many times it ran."""

    name = '_FakeBodyFallback'
    accepts_feature_types = frozenset({NavFeatureType.BODY_BLOB})
    tier = 'fallback'
    run_count: int = 0

    def is_feasible(self, features: list[NavFeature]) -> NavFeasibilityReport:
        eligible = [f for f in features if f.feature_type is NavFeatureType.BODY_BLOB]
        if not eligible:
            return NavFeasibilityReport(feasible=False, reason='no_body_blob')
        return NavFeasibilityReport(
            feasible=True, reason='ok', consumed_feature_count=len(eligible)
        )

    def navigate(self, features: list[NavFeature], context: NavContext) -> NavTechniqueResult:
        type(self).run_count += 1
        return NavTechniqueResult(
            technique_name=self.name,
            feature_ids=tuple(f.feature_id for f in features),
            offset_px=(0.5, 0.5),
            covariance_px2=np.eye(2, dtype=np.float64) * 0.25,
            confidence=0.6,
            spurious=False,
            at_edge=False,
            diagnostics=StarFieldDiagnostics(n_inliers=len(features)),
            source_bodies=body_names_from_features(features),
        )


def test_orchestrator_skips_fallback_when_primary_covers_body() -> None:
    """A non-spurious primary suppresses the fallback technique on the same body.

    Operators expect the fallback to run only when the primary fails;
    the prior behaviour ran every feasible technique and dropped the
    fallback in the ensemble post-hoc, wasting its compute.  This
    regression test verifies the primary-then-fallback scheduling
    actually skips the fallback's ``navigate`` call.
    """
    _FakeBodyPrimary.run_count = 0
    _FakeBodyPrimary.spurious_override = False
    _FakeBodyFallback.run_count = 0
    obs = _FakeObs()
    model = _FakeBodyModel(obs, body_name='TestMoon')
    orch = NavOrchestrator([model], only_techniques=['_FakeBodyPrimary', '_FakeBodyFallback'])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert result.status == 'success'
    assert _FakeBodyPrimary.run_count == 1
    # Fallback is NOT called because the primary covered TestMoon.
    assert _FakeBodyFallback.run_count == 0
    technique_names = {t.technique_name for t in result.per_technique}
    assert '_FakeBodyPrimary' in technique_names
    assert '_FakeBodyFallback' not in technique_names


def test_orchestrator_runs_fallback_when_primary_is_spurious() -> None:
    """A spurious primary does not cover the body — the fallback runs."""
    _FakeBodyPrimary.run_count = 0
    _FakeBodyPrimary.spurious_override = True
    _FakeBodyFallback.run_count = 0
    obs = _FakeObs()
    model = _FakeBodyModel(obs, body_name='TestMoon')
    orch = NavOrchestrator([model], only_techniques=['_FakeBodyPrimary', '_FakeBodyFallback'])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    # The orchestrator returns whatever status the ensemble derives —
    # the contract under test is that the fallback DID run.
    del result
    assert _FakeBodyPrimary.run_count == 1
    assert _FakeBodyFallback.run_count == 1
    _FakeBodyPrimary.spurious_override = False  # reset for any later test


# --- Calibrated-IF NaN missing-data sentinel regression tests (CODE-ORCH-003) ---


def _ciss_calib_inst_config() -> dict[str, Any]:
    """Return a CISS-CALIB-style inst_config: calibrated_if + NaN marker.

    Mirrors the shape ``instrument_settings_from_obs`` expects for a
    calibrated-IF camera: ``data_units='calibrated_if'``,
    ``noise.marker_value: NaN``, and the I/F-keyed
    ``image_quality_thresholds`` block (no DN-keyed fields, no
    ``saturation_threshold_if``).
    """
    return {
        'data_units': 'calibrated_if',
        'noise': {'marker_value': 'NaN'},
        'image_quality_thresholds': {
            'max_overexposed_frac_clean': 0.80,
            'max_missing_frac_clean': 0.30,
            'partial_dropout_min_frac': 0.05,
            'blank_max_if': 1.0e-4,
            'noisy_threshold_if': 0.01,
        },
    }


def test_orchestrator_calibrated_if_nan_pixels_do_not_raise() -> None:
    """A calibrated-IF image with NaN missing-data markers navigates without raising.

    For ``calibrated_if`` instruments the missing-data sentinel is NaN.
    The orchestrator must sanitise those NaN before the finite-only
    derivative path runs; otherwise ``_smooth_and_compute_gradients``
    raises a ValueError that would propagate out of ``navigate`` and
    violate the never-raise contract.
    """
    image = np.full((64, 64), 0.5, np.float64)
    image[:6, :] = np.nan  # ~9.4% NaN (missing) — below mostly_missing threshold
    obs = _FakeObs(image=image, inst_config=_ciss_calib_inst_config())
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator([model], only_techniques=['_FakeStarTechnique'])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    # The pipeline ran end-to-end and produced a successful offset.
    assert result.status == 'success'
    assert result.offset_px == (1.5, 2.5)


def test_orchestrator_calibrated_if_missing_frac_reflects_nan() -> None:
    """The classifier verdict's missing_frac reflects the NaN fraction.

    The orchestrator threads the true missing fraction (computed from the
    NaN mask before sanitisation) into the classifier rather than relying
    on ``sensor == marker`` (which can never match NaN).
    """
    image = np.full((64, 64), 0.5, np.float64)
    image[:6, :] = np.nan  # ~9.4% NaN
    obs = _FakeObs(image=image, inst_config=_ciss_calib_inst_config())
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator([model], only_techniques=['_FakeStarTechnique'])
    prep = orch.prepare(obs)  # type: ignore[arg-type]
    assert prep.image_classifier.missing_frac == pytest.approx(6.0 / 64.0)


def test_orchestrator_calibrated_if_mostly_nan_short_circuits() -> None:
    """A calibrated-IF image dominated by NaN markers short-circuits as missing.

    With the NaN fraction above ``max_missing_frac_clean`` the classifier
    returns ``mostly_missing_data`` and the orchestrator fails before any
    technique runs — proving the NaN-aware missing detection actually
    drives the hard-failure short-circuit for calibrated images.
    """
    image = np.full((64, 64), 0.5, np.float64)
    image[:48, :] = np.nan  # 75% NaN, above the 0.30 clean threshold
    obs = _FakeObs(image=image, inst_config=_ciss_calib_inst_config())
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator([model], only_techniques=['_FakeStarTechnique'])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert result.status == 'failed'
    assert result.status_reason == NavStatusReason.MISSING_DATA_DOMINANT
    assert result.per_technique == []

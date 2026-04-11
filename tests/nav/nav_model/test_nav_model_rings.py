"""Unit tests for the NavModelRings orchestrator.

Tests cover config loading with the new 'features:' key structure, validation
(missing epoch, invalid params, cross-feature date overlap), ring visibility
check, filter integration, NavModelResult.uncertainty wiring, and the
never_create_model / always_create_model flags.

All oops backplane calls are mocked so these tests run without OOPS_RESOURCES.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nav.nav_model.nav_model_rings import NavModelRings


# ---------------------------------------------------------------------------
# Helpers: build mock configs and observations
# ---------------------------------------------------------------------------

def _make_obs(
    planet: str | None = 'SATURN',
    *,
    shape: tuple[int, int] = (10, 10),
    midtime: float = 252460865.0,  # approx 2008-01-01
) -> MagicMock:
    """Return a mock oops.Observation with necessary attributes."""
    obs = MagicMock()
    obs.closest_planet = planet
    obs.midtime = midtime
    obs.extdata_shape_vu = shape
    obs.extfov_margin_v = 0
    obs.extfov_margin_u = 0
    obs.data_shape_v = shape[0]
    obs.data_shape_u = shape[1]

    arr = np.zeros(shape, dtype=np.float64)
    obs.make_extfov_zeros.side_effect = lambda: arr.copy()
    obs.make_extfov_false.side_effect = lambda: np.zeros(shape, dtype=bool)

    return obs


def _make_bp(obs: MagicMock, *, radii_all_masked: bool = False) -> MagicMock:
    """Wire obs.ext_bp with ring backplane mocks.

    Parameters:
        obs: Mock observation to wire.
        radii_all_masked: If True, ring_radius returns an all-masked backplane.

    Returns:
        The ``MagicMock`` for ``ring_radius`` (``bp_radii``) attached at
        ``obs.ext_bp.ring_radius``.
    """
    shape = obs.extdata_shape_vu

    bp_radii = MagicMock()
    bp_radii.is_all_masked.return_value = radii_all_masked
    if not radii_all_masked:
        bp_radii.min.return_value.vals = 70_000.0
        bp_radii.max.return_value.vals = 140_000.0
    bp_radii.key = 'ring_radius'

    bp_res = MagicMock()
    res_arr = np.full(shape, 1.0, dtype=np.float64)
    bp_res.vals = res_arr

    bp_dist = MagicMock()
    dist_masked = MagicMock()
    dist_masked.filled.return_value = np.full(shape, 1.0e6, dtype=np.float64)
    bp_dist.mvals = dist_masked

    border_mock = MagicMock()
    border_arr = MagicMock()
    border_arr.astype.return_value.filled.return_value = np.zeros(shape, dtype=bool)
    border_mock.mvals = border_arr

    obs.ext_bp.ring_radius.return_value = bp_radii
    obs.ext_bp.ring_radial_resolution.return_value = bp_res
    obs.ext_bp.distance.return_value = bp_dist
    obs.ext_bp.border_atop.return_value = border_mock

    return bp_radii


def _make_edge_data(a: float = 100_000.0) -> list[dict[str, Any]]:
    """Build a one-element inner/outer edge mode list for tests.

    Parameters:
        a: Mode-1 semi-major axis in km (default ``100_000.0``).

    Returns:
        ``list[dict[str, Any]]`` with keys ``mode``, ``a``, ``rms``, ``ae``,
        ``long_peri``, and ``rate_peri``.
    """
    return [{'mode': 1, 'a': a, 'rms': 1.0, 'ae': 0.0, 'long_peri': 0.0, 'rate_peri': 0.0}]


def _make_planet_config(
    epoch: str = '2008-01-01 12:00:00',
    *,
    fade_width_pix: float = 100.0,
    min_allowed: float = 3.0,
    min_feature: float = 2.0,
    features: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a planet_config dict using the new YAML format.

    Parameters:
        epoch: Reference UTC epoch string (default ``2008-01-01 12:00:00``).
        fade_width_pix: Nominal fade width in pixels (default ``100.0``).
        min_allowed: ``min_allowed_fade_width_pix`` (default ``3.0``).
        min_feature: ``min_feature_pixels`` (default ``2.0``).
        features: Optional map of feature key to feature spec dicts; each spec
            includes ``feature_type``, ``name``, ``inner_data``, and ``outer_data``
            when both edges are present.

    Returns:
        Dict with keys ``epoch``, ``fade_width_pix``, ``min_allowed_fade_width_pix``,
        ``min_feature_pixels``, and ``features``.
    """
    if features is None:
        features = {
            'test_ringlet': {
                'feature_type': 'RINGLET',
                'name': 'Test Ringlet',
                'inner_data': _make_edge_data(100_000.0),
                'outer_data': _make_edge_data(101_000.0),
            }
        }
    return {
        'epoch': epoch,
        'fade_width_pix': fade_width_pix,
        'min_allowed_fade_width_pix': min_allowed,
        'min_feature_pixels': min_feature,
        'features': features,
    }


def _make_mock_config(planet_config: dict[str, Any]) -> MagicMock:
    """Return a mock Config whose rings.ring_features contains planet_config."""
    cfg = MagicMock()
    cfg.rings.ring_features = {'SATURN': planet_config}
    return cfg


def _noop_nav_model_rings_init(
    self: NavModelRings, name: str, obs: Any, *, config: Any = None
) -> None:
    """Test helper: skip ``NavModelRings`` / ``NavModel`` ``__init__``.

    Callers assign ``_config``, ``_obs``, ``_models``, ``_metadata``, and
    ``_logger`` to match production layout without running real initialization.
    """


def _make_rings_model(
    obs: MagicMock,
    planet_config: dict[str, Any] | None = None,
) -> NavModelRings:
    """Return a ``NavModelRings`` constructed with ``__init__`` patched to a no-op.

    Sets ``_config``, ``_obs``, ``_models``, ``_metadata``, and ``_logger`` so
    ``_create_model`` tests match the attributes the real constructor would
    populate, while avoiding ``NavModel`` / ``NavBase`` setup.
    """
    if planet_config is None:
        planet_config = _make_planet_config()
    mock_cfg = _make_mock_config(planet_config)
    with patch.object(NavModelRings, '__init__', _noop_nav_model_rings_init):
        model = NavModelRings('test_rings', obs, config=mock_cfg)
    model._config = mock_cfg
    model._obs = obs
    model._models = []
    model._metadata = {}
    model._logger = MagicMock()
    model._logger.open.return_value.__enter__ = lambda self: None
    model._logger.open.return_value.__exit__ = MagicMock(return_value=False)
    return model


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestConfigLoading:
    """Test config loading from the new 'features:' structure."""

    def test_no_closest_planet_returns_early(self) -> None:
        """No planet means no model created."""
        obs = _make_obs(planet=None)
        model = _make_rings_model(obs)
        model._create_model(
            always_create_model=False,
            never_create_model=False,
            create_annotations=False,
        )
        assert model._models == []

    def test_missing_planet_config_returns_early(self) -> None:
        """Planet not in ring_features means no model."""
        obs = _make_obs(planet='URANUS')
        model = _make_rings_model(obs)  # config only has SATURN
        model._create_model(
            always_create_model=False,
            never_create_model=False,
            create_annotations=False,
        )
        assert model._models == []

    def test_missing_epoch_raises(self) -> None:
        """Missing epoch raises ValueError."""
        obs = _make_obs()
        planet_config = _make_planet_config()
        del planet_config['epoch']
        model = _make_rings_model(obs, planet_config)
        with pytest.raises(ValueError, match='epoch'):
            model._create_model(
                always_create_model=False,
                never_create_model=False,
                create_annotations=False,
            )

    def test_invalid_fade_width_pix_raises(self) -> None:
        """Non-positive fade_width_pix raises ValueError."""
        obs = _make_obs()
        model = _make_rings_model(obs, _make_planet_config(fade_width_pix=0.0))
        with pytest.raises(ValueError, match='fade_width_pix'):
            model._create_model(
                always_create_model=False,
                never_create_model=False,
                create_annotations=False,
            )

    def test_invalid_min_allowed_raises(self) -> None:
        """Non-positive min_allowed_fade_width_pix raises ValueError."""
        obs = _make_obs()
        model = _make_rings_model(obs, _make_planet_config(min_allowed=-1.0))
        with pytest.raises(ValueError, match='min_allowed_fade_width_pix'):
            model._create_model(
                always_create_model=False,
                never_create_model=False,
                create_annotations=False,
            )

    def test_malformed_feature_raises(self) -> None:
        """A feature with invalid config raises ValueError during load."""
        obs = _make_obs()
        planet_config = _make_planet_config(features={
            'bad': {'feature_type': 'INVALID_TYPE', 'inner_data': _make_edge_data()}
        })
        model = _make_rings_model(obs, planet_config)
        with pytest.raises(ValueError, match='feature_type'):
            model._create_model(
                always_create_model=False,
                never_create_model=False,
                create_annotations=False,
            )

    def test_non_dict_feature_raises(self) -> None:
        """A feature entry that isn't a dict raises ValueError."""
        obs = _make_obs()
        planet_config = _make_planet_config(features={'bad': 'not_a_dict'})
        model = _make_rings_model(obs, planet_config)
        with pytest.raises(ValueError, match='not a dict'):
            model._create_model(
                always_create_model=False,
                never_create_model=False,
                create_annotations=False,
            )

    def test_no_ring_visibility_returns_empty_model(self) -> None:
        """All-masked ring backplane returns empty model when always_create_model."""
        obs = _make_obs()
        _make_bp(obs, radii_all_masked=True)
        model = _make_rings_model(obs)
        model._create_model(
            always_create_model=True,
            never_create_model=False,
            create_annotations=False,
        )
        assert len(model._models) == 1
        assert np.all(model._models[0].model_img == 0.0)

    def test_no_ring_visibility_no_model_when_not_always(self) -> None:
        """All-masked ring backplane with always_create_model=False returns no model."""
        obs = _make_obs()
        _make_bp(obs, radii_all_masked=True)
        model = _make_rings_model(obs)
        model._create_model(
            always_create_model=False,
            never_create_model=False,
            create_annotations=False,
        )
        assert model._models == []


# ---------------------------------------------------------------------------
# Cross-feature date validation
# ---------------------------------------------------------------------------

class TestCrossFeatureValidation:
    """validate_no_date_overlaps is called during _create_model."""

    def test_overlapping_dated_features_raises(self) -> None:
        """Two dated features with overlapping dates and radii raise ValueError."""
        obs = _make_obs()
        _make_bp(obs)
        features = {
            'a': {
                'feature_type': 'RINGLET',
                'name': 'A',
                'inner_data': _make_edge_data(100_000.0),
                'outer_data': _make_edge_data(101_000.0),
                'start_date': '2007-01-01',
                'end_date': '2009-01-01',
            },
            'b': {
                'feature_type': 'RINGLET',
                'name': 'B',
                'inner_data': _make_edge_data(100_000.0),
                'outer_data': _make_edge_data(101_000.0),
                'start_date': '2008-06-01',
                'end_date': '2010-01-01',
            },
        }
        model = _make_rings_model(obs, _make_planet_config(features=features))
        with pytest.raises(ValueError, match="overlapping date ranges"):
            model._create_model(
                always_create_model=False,
                never_create_model=False,
                create_annotations=False,
            )


# ---------------------------------------------------------------------------
# Filter integration and model generation
# ---------------------------------------------------------------------------

class TestFilterIntegration:
    """Tests that the filter pipeline integrates correctly."""

    def _make_render_result(self, shape: tuple[int, int]) -> MagicMock:
        """Return a mock RingRenderResult."""
        result = MagicMock()
        result.model_img = np.ones(shape, dtype=np.float64) * 0.5
        result.model_mask = np.ones(shape, dtype=bool)
        result.uncertainty = 2.5
        result.edge_info_list = []
        return result

    def test_surviving_feature_creates_model_result(self) -> None:
        """A feature that survives filtering produces a NavModelResult."""
        shape = (10, 10)
        obs = _make_obs(shape=shape)
        _make_bp(obs)
        render_result = self._make_render_result(shape)

        with patch('nav.nav_model.nav_model_rings.RingFeatureFilter') as MockFilter:
            mock_filter_inst = MagicMock()
            MockFilter.return_value = mock_filter_inst

            mock_feature = MagicMock()
            mock_feature.render.return_value = [render_result]
            mock_filter_inst.filter.return_value = [mock_feature]
            mock_feature.all_base_radii.return_value = [(100_000.0, 'IER')]
            mock_feature.name = 'Test'
            mock_feature.feature_type.value = 'RINGLET'

            model = _make_rings_model(obs)
            model._create_model(
                always_create_model=False,
                never_create_model=False,
                create_annotations=False,
            )

        assert len(model._models) == 1
        assert model._models[0].uncertainty == 2.5

    def test_all_features_filtered_out_no_model(self) -> None:
        """If no features survive filtering, no model is created."""
        obs = _make_obs()
        _make_bp(obs)
        with patch('nav.nav_model.nav_model_rings.RingFeatureFilter') as MockFilter:
            mock_filter_inst = MagicMock()
            MockFilter.return_value = mock_filter_inst
            mock_filter_inst.filter.return_value = []

            model = _make_rings_model(obs)
            model._create_model(
                always_create_model=False,
                never_create_model=False,
                create_annotations=False,
            )

        assert model._models == []


# ---------------------------------------------------------------------------
# NavModelResult.uncertainty wiring
# ---------------------------------------------------------------------------

class TestUncertaintyWiring:
    """NavModelResult.uncertainty is taken from render_result.uncertainty."""

    def test_uncertainty_wired_from_render_result(self) -> None:
        """uncertainty in NavModelResult matches the render result's uncertainty."""
        shape = (10, 10)
        obs = _make_obs(shape=shape)
        _make_bp(obs)

        render_result = MagicMock()
        render_result.model_img = np.zeros(shape, dtype=np.float64)
        render_result.model_mask = np.ones(shape, dtype=bool)
        render_result.uncertainty = 7.3
        render_result.edge_info_list = []

        with patch('nav.nav_model.nav_model_rings.RingFeatureFilter') as MockFilter:
            mock_filter_inst = MagicMock()
            MockFilter.return_value = mock_filter_inst
            mock_feature = MagicMock()
            mock_feature.render.return_value = [render_result]
            mock_filter_inst.filter.return_value = [mock_feature]
            mock_feature.all_base_radii.return_value = []
            mock_feature.name = 'X'
            mock_feature.feature_type.value = 'GAP'

            model = _make_rings_model(obs)
            model._create_model(
                always_create_model=False,
                never_create_model=False,
                create_annotations=False,
            )

        assert len(model._models) == 1
        assert model._models[0].uncertainty == 7.3


# ---------------------------------------------------------------------------
# never_create_model flag
# ---------------------------------------------------------------------------

class TestNeverCreateModel:
    """never_create_model=True populates metadata but creates no model images."""

    def test_never_create_model_no_images(self) -> None:
        """never_create_model=True skips rendering."""
        obs = _make_obs()
        _make_bp(obs)

        with patch('nav.nav_model.nav_model_rings.RingFeatureFilter') as MockFilter:
            mock_feature = MagicMock()
            mock_feature.name = 'Test'
            mock_feature.feature_type.value = 'RINGLET'
            mock_feature.all_base_radii.return_value = []
            mock_filter_inst = MagicMock()
            MockFilter.return_value = mock_filter_inst
            mock_filter_inst.filter.return_value = [mock_feature]

            model = _make_rings_model(obs)
            model._metadata = {}
            model._create_model(
                always_create_model=False,
                never_create_model=True,
                create_annotations=False,
            )

        assert model._models == []
        assert model._metadata['planet'] == 'SATURN'
        assert model._metadata['feature_count'] == 1

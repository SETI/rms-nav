"""Unit tests for ``NavModelRingsSimulated``.

Tests cover successful model creation (inner/outer edges), uncertainty wiring
from ``RingFeature`` to ``NavModelResult``, ``sim_params`` edge cases (no outer,
no inner), and ``_sim_params_to_feature_config``.

``render_ring`` and ``compute_border_atop_simulated`` are mocked.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

from nav.nav_model.nav_model_rings_simulated import (
    NavModelRingsSimulated,
    _sim_params_to_feature_config,
)
from nav.nav_model.rings import RingFeature, RingFeatureType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_obs(shape: tuple[int, int] = (10, 10)) -> MagicMock:
    """Return a minimal mock observation."""
    obs = MagicMock()
    obs.extdata_shape_vu = shape
    obs.extfov_margin_v = 0
    obs.extfov_margin_u = 0
    obs.data_shape_v = shape[0]
    obs.data_shape_u = shape[1]
    obs.sim_time = 252460865.0
    obs.sim_epoch = 252460865.0
    obs.make_extfov_zeros.side_effect = lambda: np.zeros(shape, dtype=np.float64)
    obs.make_extfov_false.side_effect = lambda: np.zeros(shape, dtype=bool)
    return obs


def _make_sim_params(
    feature_type: str = 'RINGLET',
    *,
    inner_a: float | None = 100_000.0,
    outer_a: float | None = 101_000.0,
    name: str = 'TestRing',
) -> dict[str, Any]:
    """Return a sim_params dict mimicking GUI-saved ring parameters."""
    params: dict[str, Any] = {
        'feature_type': feature_type,
        'name': name,
        'center_v': 5.0,
        'center_u': 5.0,
        'range': 1.0e6,
    }
    if inner_a is not None:
        params['inner_data'] = [
            {'mode': 1, 'a': inner_a, 'rms': 2.0, 'ae': 0.0, 'long_peri': 0.0, 'rate_peri': 0.0}
        ]
    if outer_a is not None:
        params['outer_data'] = [
            {'mode': 1, 'a': outer_a, 'rms': 3.0, 'ae': 0.0, 'long_peri': 0.0, 'rate_peri': 0.0}
        ]
    return params


def _make_simulated_model(
    obs: MagicMock,
    sim_params: dict[str, Any],
    ring_name: str = 'SATURN',
) -> NavModelRingsSimulated:
    """Return a ``NavModelRingsSimulated`` with mocked config and logger.

    Uses the real constructor so base-class invariants stay aligned; the process
    logger is patched so ``create_model`` / ``_logger.open`` can be asserted
    without touching global ``IMAGE_LOGGER``.
    """
    cfg = MagicMock()
    cfg.rings.label_font = 'Arial'
    cfg.rings.label_font_size = 12
    cfg.rings.label_font_color = 'white'
    cfg.rings.label_limb_color = 'white'
    cfg.rings.label_horiz_gap = 5
    cfg.rings.label_vert_gap = 5
    cfg.rings.label_mask_enlarge = 3

    mock_logger = MagicMock()
    mock_logger.open.return_value.__enter__ = MagicMock(return_value=None)
    mock_logger.open.return_value.__exit__ = MagicMock(return_value=False)

    with patch('nav.support.nav_base.IMAGE_LOGGER', mock_logger):
        return NavModelRingsSimulated(
            'simulated-rings-test',
            obs,
            ring_name,
            sim_params,
            config=cfg,
        )


# ---------------------------------------------------------------------------
# _sim_params_to_feature_config
# ---------------------------------------------------------------------------


class TestSimParamsToFeatureConfig:
    """Test the helper that adapts sim_params to RingFeature.from_config() format."""

    def test_ringlet_with_both_edges(self) -> None:
        """Full ringlet sim_params produces a valid feature config."""
        p = _make_sim_params(feature_type='RINGLET')
        cfg = _sim_params_to_feature_config(p)
        assert cfg['feature_type'] == 'RINGLET'
        assert 'inner_data' in cfg
        assert 'outer_data' in cfg

    def test_gap_type_preserved(self) -> None:
        """GAP feature_type is preserved in the config."""
        p = _make_sim_params(feature_type='GAP')
        cfg = _sim_params_to_feature_config(p)
        assert cfg['feature_type'] == 'GAP'

    def test_missing_outer_data_omitted(self) -> None:
        """Absent outer_data is not included in the config."""
        p = _make_sim_params(outer_a=None)
        cfg = _sim_params_to_feature_config(p)
        assert 'outer_data' not in cfg

    def test_empty_inner_data_treated_as_absent(self) -> None:
        """Empty inner_data list is treated as absent (omitted from config)."""
        p = _make_sim_params(inner_a=100_000.0)
        p['inner_data'] = []
        cfg = _sim_params_to_feature_config(p)
        assert 'inner_data' not in cfg

    def test_name_preserved(self) -> None:
        """Feature name is preserved in config."""
        p = _make_sim_params(name='MyRing')
        cfg = _sim_params_to_feature_config(p)
        assert cfg['name'] == 'MyRing'

    def test_default_feature_type_is_ringlet(self) -> None:
        """Missing feature_type defaults to RINGLET."""
        p = _make_sim_params()
        del p['feature_type']
        cfg = _sim_params_to_feature_config(p)
        assert cfg['feature_type'] == 'RINGLET'


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------


class TestModelCreation:
    """NavModelRingsSimulated creates NavModelResult entries."""

    def test_model_created_with_inner_and_outer_edges(self) -> None:
        """Full ringlet sim_params produces one model result."""
        obs = _make_obs()
        sim_img = np.zeros((10, 10), dtype=np.float64)
        sim_img[3:7, 3:7] = 0.5  # some non-zero pixels

        model = _make_simulated_model(obs, _make_sim_params())
        with (
            patch('nav.nav_model.nav_model_rings_simulated.render_ring') as mock_render,
            patch(
                'nav.nav_model.nav_model_rings_simulated.compute_border_atop_simulated'
            ) as mock_border,
        ):
            mock_render.side_effect = lambda img, *a, **kw: img.__setitem__(
                (slice(3, 7), slice(3, 7)), 0.5
            )
            mock_border.return_value = np.zeros((10, 10), dtype=bool)
            model._create_model(
                always_create_model=False,
                never_create_model=False,
                create_annotations=False,
            )

        assert len(model._models) == 1
        result = model._models[0]
        assert result.model_img is not None
        assert result.model_img.shape == (10, 10)

    def test_model_with_single_inner_edge(self) -> None:
        """Single-edge ringlet (inner only) produces one model result."""
        obs = _make_obs()
        model = _make_simulated_model(obs, _make_sim_params(outer_a=None))
        with (
            patch('nav.nav_model.nav_model_rings_simulated.render_ring'),
            patch(
                'nav.nav_model.nav_model_rings_simulated.compute_border_atop_simulated'
            ) as mock_border,
        ):
            mock_border.return_value = np.zeros((10, 10), dtype=bool)
            model._create_model(
                always_create_model=False,
                never_create_model=False,
                create_annotations=False,
            )
        assert len(model._models) == 1


# ---------------------------------------------------------------------------
# Uncertainty wiring
# ---------------------------------------------------------------------------


class TestUncertaintyWiring:
    """NavModelResult.uncertainty is max RMS from the feature's edges."""

    def test_uncertainty_is_max_rms_of_edges(self) -> None:
        """NavModelResult.uncertainty = max(inner rms, outer rms) = max(2.0, 3.0) = 3.0."""
        # inner_data rms=2.0, outer_data rms=3.0 -> max = 3.0
        obs = _make_obs()
        model = _make_simulated_model(obs, _make_sim_params(inner_a=100_000.0, outer_a=101_000.0))
        with (
            patch('nav.nav_model.nav_model_rings_simulated.render_ring'),
            patch(
                'nav.nav_model.nav_model_rings_simulated.compute_border_atop_simulated'
            ) as mock_border,
        ):
            mock_border.return_value = np.zeros((10, 10), dtype=bool)
            model._create_model(
                always_create_model=False,
                never_create_model=False,
                create_annotations=False,
            )
        assert len(model._models) == 1
        assert model._models[0].uncertainty == 3.0

    def test_uncertainty_single_edge(self) -> None:
        """NavModelResult.uncertainty = rms of the single present edge."""
        obs = _make_obs()
        model = _make_simulated_model(obs, _make_sim_params(outer_a=None))
        with (
            patch('nav.nav_model.nav_model_rings_simulated.render_ring'),
            patch(
                'nav.nav_model.nav_model_rings_simulated.compute_border_atop_simulated'
            ) as mock_border,
        ):
            mock_border.return_value = np.zeros((10, 10), dtype=bool)
            model._create_model(
                always_create_model=False,
                never_create_model=False,
                create_annotations=False,
            )
        assert len(model._models) == 1
        assert model._models[0].uncertainty == 2.0  # inner edge rms


# ---------------------------------------------------------------------------
# Feature type detection
# ---------------------------------------------------------------------------


class TestFeatureTypeDetection:
    """RingFeature is constructed with the correct feature type."""

    def test_gap_feature_type(self) -> None:
        """GAP sim_params produces a GAP feature type (validated by from_config)."""
        p = _make_sim_params(feature_type='GAP')
        cfg = _sim_params_to_feature_config(p)
        feature = RingFeature.from_config('test', cfg)
        assert feature.feature_type is RingFeatureType.GAP

    def test_ringlet_feature_type(self) -> None:
        """RINGLET sim_params produces a RINGLET feature type."""
        p = _make_sim_params(feature_type='RINGLET')
        cfg = _sim_params_to_feature_config(p)
        feature = RingFeature.from_config('test', cfg)
        assert feature.feature_type is RingFeatureType.RINGLET

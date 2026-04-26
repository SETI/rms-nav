"""Unit tests for the ``NavModelRings`` orchestrator.

Tests cover config retrieval (planet block under ``rings.ring_features`` with a
required ``features`` mapping), validation (missing keys, invalid params,
cross-feature date overlap), ring visibility before feature construction, the
``RingFeatureFilter`` pipeline, ``RingsRenderContext`` construction with the
model logger, ``NavModelResult.uncertainty`` wiring, and the
``never_create_model`` / ``always_create_model`` flags.

Notes:
    All oops backplane calls are mocked so these tests run without
    OOPS_RESOURCES.
"""

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
    """Return a planet block dict as used under ``rings.ring_features`` (YAML shape).

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
        ``min_feature_pixels``, and ``features`` (all required by the orchestrator).
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
    # Prevent MagicMock().get() from returning a truthy MagicMock; the shadow-
    # removal path only runs when 'remove_planet_shadow' is explicitly True.
    cfg.rings.get.return_value = False
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
# Config retrieval
# ---------------------------------------------------------------------------


class TestConfigRetrieval:
    """Exercise ``NavModelRings._create_model`` config and visibility gating.

    Covers missing planet, invalid planet block shape, required planet keys,
    ``features`` map validation, scalar validation, per-feature parsing, and
    all-masked ring-radius backplanes with ``always_create_model``.
    """

    def test_no_closest_planet_returns_early(self) -> None:
        """Exit before config lookup when ``obs.closest_planet`` is missing.

        Notes:
            Asserts ``model._models`` stays empty; no ``RingFeatureFilter`` or
            rendering path runs.
        """
        obs = _make_obs(planet=None)
        model = _make_rings_model(obs)
        model._create_model(
            always_create_model=False,
            never_create_model=False,
            create_annotations=False,
        )
        assert model._models == []

    def test_missing_planet_config_returns_early(self) -> None:
        """Exit when the closest planet has no entry under ``rings.ring_features``.

        Notes:
            Uses ``URANUS`` as closest planet while the mock config only defines
            ``SATURN``. Asserts ``model._models`` is empty.
        """
        obs = _make_obs(planet='URANUS')
        model = _make_rings_model(obs)  # config only has SATURN
        model._create_model(
            always_create_model=False,
            never_create_model=False,
            create_annotations=False,
        )
        assert model._models == []

    def test_planet_ring_block_not_dict_raises(self) -> None:
        """Reject a non-mapping planet block under ``rings.ring_features``.

        Notes:
            Planet entry is a list; message must name ``SATURN`` and require a dict.
        """
        obs = _make_obs()
        cfg = MagicMock()
        cfg.rings.ring_features = {'SATURN': ['not', 'a', 'dict']}
        with patch.object(NavModelRings, '__init__', _noop_nav_model_rings_init):
            model = NavModelRings('test_rings', obs, config=cfg)
        model._config = cfg
        model._obs = obs
        model._models = []
        model._metadata = {}
        model._logger = MagicMock()
        model._logger.open.return_value.__enter__ = lambda self: None
        model._logger.open.return_value.__exit__ = MagicMock(return_value=False)
        with pytest.raises(ValueError, match=r"planet 'SATURN'.*must be a dict"):
            model._create_model(
                always_create_model=False,
                never_create_model=False,
                create_annotations=False,
            )

    def test_missing_epoch_raises(self) -> None:
        """Require ``epoch`` on the planet ring block.

        Notes:
            Removes ``epoch`` from an otherwise valid planet config before calling
            ``_create_model``.
        """
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

    def test_missing_fade_width_pix_raises(self) -> None:
        """Require ``fade_width_pix`` on the planet ring block.

        Notes:
            Deletes ``fade_width_pix`` from the planet dict before ``_create_model``.
        """
        obs = _make_obs()
        planet_config = _make_planet_config()
        del planet_config['fade_width_pix']
        model = _make_rings_model(obs, planet_config)
        with pytest.raises(ValueError, match='fade_width_pix'):
            model._create_model(
                always_create_model=False,
                never_create_model=False,
                create_annotations=False,
            )

    def test_missing_min_allowed_fade_width_pix_raises(self) -> None:
        """Require ``min_allowed_fade_width_pix`` on the planet ring block.

        Notes:
            Deletes that key before ``_create_model``.
        """
        obs = _make_obs()
        planet_config = _make_planet_config()
        del planet_config['min_allowed_fade_width_pix']
        model = _make_rings_model(obs, planet_config)
        with pytest.raises(ValueError, match='min_allowed_fade_width_pix'):
            model._create_model(
                always_create_model=False,
                never_create_model=False,
                create_annotations=False,
            )

    def test_missing_min_feature_pixels_raises(self) -> None:
        """Require ``min_feature_pixels`` on the planet ring block.

        Notes:
            Deletes ``min_feature_pixels`` before ``_create_model``.
        """
        obs = _make_obs()
        planet_config = _make_planet_config()
        del planet_config['min_feature_pixels']
        model = _make_rings_model(obs, planet_config)
        with pytest.raises(ValueError, match='min_feature_pixels'):
            model._create_model(
                always_create_model=False,
                never_create_model=False,
                create_annotations=False,
            )

    def test_missing_features_key_raises(self) -> None:
        """Require a ``features`` key on the planet ring block.

        Notes:
            Ring backplanes are mocked so validation runs before visibility-only
            short-circuit paths.
        """
        obs = _make_obs()
        _make_bp(obs)
        planet_config = _make_planet_config()
        del planet_config['features']
        model = _make_rings_model(obs, planet_config)
        with pytest.raises(ValueError, match='features'):
            model._create_model(
                always_create_model=False,
                never_create_model=False,
                create_annotations=False,
            )

    def test_features_not_dict_raises(self) -> None:
        """Require ``features`` to be a mapping, not a sequence or scalar.

        Notes:
            Sets ``planet_config['features']`` to a list before ``_create_model``.
        """
        obs = _make_obs()
        _make_bp(obs)
        planet_config = _make_planet_config()
        planet_config['features'] = ['not', 'a', 'dict']
        model = _make_rings_model(obs, planet_config)
        with pytest.raises(ValueError, match='must be a dict'):
            model._create_model(
                always_create_model=False,
                never_create_model=False,
                create_annotations=False,
            )

    def test_invalid_fade_width_pix_raises(self) -> None:
        """Reject non-positive ``fade_width_pix`` scalars.

        Notes:
            Uses ``fade_width_pix=0.0`` in the planet config.
        """
        obs = _make_obs()
        model = _make_rings_model(obs, _make_planet_config(fade_width_pix=0.0))
        with pytest.raises(ValueError, match='fade_width_pix'):
            model._create_model(
                always_create_model=False,
                never_create_model=False,
                create_annotations=False,
            )

    def test_invalid_min_allowed_raises(self) -> None:
        """Reject non-positive ``min_allowed_fade_width_pix`` scalars.

        Notes:
            Sets ``min_allowed_fade_width_pix`` to ``-1.0``.
        """
        obs = _make_obs()
        model = _make_rings_model(obs, _make_planet_config(min_allowed=-1.0))
        with pytest.raises(ValueError, match='min_allowed_fade_width_pix'):
            model._create_model(
                always_create_model=False,
                never_create_model=False,
                create_annotations=False,
            )

    def test_bool_fade_width_pix_raises(self) -> None:
        """Reject boolean ``fade_width_pix`` (must not pass as numeric).

        Notes:
            Assigns ``True`` to ``fade_width_pix`` in the planet dict.
        """
        obs = _make_obs()
        cfg = _make_planet_config()
        cfg['fade_width_pix'] = True
        model = _make_rings_model(obs, cfg)
        with pytest.raises(ValueError, match=r'fade_width_pix.*bool'):
            model._create_model(
                always_create_model=False,
                never_create_model=False,
                create_annotations=False,
            )

    def test_nan_min_feature_pixels_raises(self) -> None:
        """Reject non-finite ``min_feature_pixels`` (e.g. NaN).

        Notes:
            Sets ``min_feature_pixels`` to ``float('nan')``.
        """
        obs = _make_obs()
        cfg = _make_planet_config()
        cfg['min_feature_pixels'] = float('nan')
        model = _make_rings_model(obs, cfg)
        with pytest.raises(ValueError, match='min_feature_pixels'):
            model._create_model(
                always_create_model=False,
                never_create_model=False,
                create_annotations=False,
            )

    def test_malformed_feature_raises(self) -> None:
        """Fail when ``RingFeature.from_config`` rejects a feature entry.

        Notes:
            Supplies an invalid ``feature_type`` string under a feature key while
            edge data is otherwise valid.
        """
        obs = _make_obs()
        _make_bp(obs)
        planet_config = _make_planet_config(
            features={'bad': {'feature_type': 'INVALID_TYPE', 'inner_data': _make_edge_data()}}
        )
        model = _make_rings_model(obs, planet_config)
        with pytest.raises(ValueError, match='feature_type'):
            model._create_model(
                always_create_model=False,
                never_create_model=False,
                create_annotations=False,
            )

    def test_non_dict_feature_raises(self) -> None:
        """Reject feature map values that are not dicts.

        Notes:
            Sets ``features['bad']`` to a string before ``_create_model``.
        """
        obs = _make_obs()
        _make_bp(obs)
        planet_config = _make_planet_config(features={'bad': 'not_a_dict'})
        model = _make_rings_model(obs, planet_config)
        with pytest.raises(ValueError, match=r"planet 'SATURN'.*'bad'.*must be a dict"):
            model._create_model(
                always_create_model=False,
                never_create_model=False,
                create_annotations=False,
            )

    def test_no_ring_visibility_returns_empty_model(self) -> None:
        """When the ring radius backplane is all-masked, optionally emit an empty model.

        Notes:
            Uses ``radii_all_masked=True`` on the mock backplane and
            ``always_create_model=True``. Asserts one ``NavModelResult`` with a
            zero image array.
        """
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
        """When all-masked and ``always_create_model`` is false, append no models.

        Notes:
            Same backplane mock as the ``always_create_model`` case but expects
            ``model._models`` to remain empty.
        """
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
    """Cross-feature date overlap checks during ``_create_model``.

    Ensures ``validate_no_date_overlaps`` rejects overlapping validity intervals
    when radial extents intersect.
    """

    def test_overlapping_dated_features_raises(self) -> None:
        """Raise when two dated ringlets share a radial band with overlapping dates.

        Notes:
            Builds two ``RINGLET`` features with identical radii and half-open date
            ranges that intersect; ``_create_model`` must fail before filtering.
        """
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
        with pytest.raises(ValueError, match='overlapping date ranges'):
            model._create_model(
                always_create_model=False,
                never_create_model=False,
                create_annotations=False,
            )


# ---------------------------------------------------------------------------
# Filter integration and model generation
# ---------------------------------------------------------------------------


class TestFilterIntegration:
    """Orchestrator wiring for ``RingFeatureFilter`` and ``RingsRenderContext``.

    Mocks the filter and render context while asserting logger threading and
    ``NavModelResult`` creation when features survive filtering.
    """

    def _make_render_result(self, shape: tuple[int, int]) -> MagicMock:
        """Build a stand-in ``RingRenderResult`` for patched ``render`` calls.

        Parameters:
            shape: ``(height, width)`` for ``model_img`` and ``model_mask``.

        Returns:
            ``MagicMock`` with ``model_img``, ``model_mask``, ``uncertainty``, and
            empty ``edge_info_list``.

        Notes:
            ``uncertainty`` is fixed at ``2.5`` for the happy-path integration test.
        """
        result = MagicMock()
        result.model_img = np.ones(shape, dtype=np.float64) * 0.5
        result.model_mask = np.ones(shape, dtype=bool)
        result.uncertainty = 2.5
        result.edge_info_list = []
        return result

    def test_surviving_feature_creates_model_result(self) -> None:
        """Create one ``NavModelResult`` when the filter returns a renderable feature.

        Notes:
            Patches ``RingFeatureFilter`` and ``RingsRenderContext``. Asserts each
            is constructed once with ``logger=model._logger``, and that the result
            ``uncertainty`` matches the mocked ``RingRenderResult.uncertainty``.
        """
        shape = (10, 10)
        obs = _make_obs(shape=shape)
        _make_bp(obs)
        render_result = self._make_render_result(shape)

        with (
            patch('nav.nav_model.nav_model_rings.RingFeatureFilter') as mock_filter_cls,
            patch('nav.nav_model.nav_model_rings.RingsRenderContext') as mock_render_context_cls,
        ):
            mock_filter_inst = MagicMock()
            mock_filter_cls.return_value = mock_filter_inst

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

        assert mock_filter_cls.call_count == 1
        assert mock_filter_cls.call_args.kwargs['logger'] is model._logger
        assert mock_render_context_cls.call_count == 1
        assert mock_render_context_cls.call_args.kwargs['logger'] is model._logger
        assert len(model._models) == 1
        assert model._models[0].uncertainty == 2.5

    def test_all_features_filtered_out_no_model(self) -> None:
        """Append no models when the filter returns an empty feature list.

        Notes:
            Patches ``RingFeatureFilter`` so ``filter`` returns ``[]``. Asserts
            ``model._models`` is empty and the filter still receives
            ``logger=model._logger``.
        """
        obs = _make_obs()
        _make_bp(obs)
        with patch('nav.nav_model.nav_model_rings.RingFeatureFilter') as mock_filter_cls:
            mock_filter_inst = MagicMock()
            mock_filter_cls.return_value = mock_filter_inst
            mock_filter_inst.filter.return_value = []

            model = _make_rings_model(obs)
            model._create_model(
                always_create_model=False,
                never_create_model=False,
                create_annotations=False,
            )

        assert mock_filter_cls.call_count == 1
        assert mock_filter_cls.call_args.kwargs['logger'] is model._logger
        assert model._models == []


# ---------------------------------------------------------------------------
# NavModelResult.uncertainty wiring
# ---------------------------------------------------------------------------


class TestUncertaintyWiring:
    """Tests for how per-feature render uncertainty reaches ``NavModelResult``.

    Covers the path where ``RingFeatureFilter`` yields mock ring-gap features,
    ``render`` returns a ``RenderResult`` with a numeric ``uncertainty``, and
    ``_create_model`` copies that value onto the stored ``NavModelResult``. Also
    asserts filter wiring (``RingFeatureFilter`` constructed with
    ``logger=model._logger``) and single-result list growth.
    """

    def test_uncertainty_wired_from_render_result(self) -> None:
        """Copy ``render_result.uncertainty`` onto the appended ``NavModelResult``.

        Notes:
            Uses a mocked ``GAP`` feature returning a render result with
            ``uncertainty=7.3``. Asserts ``model._models[0].uncertainty`` matches
            and ``RingFeatureFilter`` gets ``logger=model._logger``.
        """
        shape = (10, 10)
        obs = _make_obs(shape=shape)
        _make_bp(obs)

        render_result = MagicMock()
        render_result.model_img = np.zeros(shape, dtype=np.float64)
        render_result.model_mask = np.ones(shape, dtype=bool)
        render_result.uncertainty = 7.3
        render_result.edge_info_list = []

        with patch('nav.nav_model.nav_model_rings.RingFeatureFilter') as mock_filter_cls:
            mock_filter_inst = MagicMock()
            mock_filter_cls.return_value = mock_filter_inst
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

        assert mock_filter_cls.call_count == 1
        assert mock_filter_cls.call_args.kwargs['logger'] is model._logger
        assert len(model._models) == 1
        assert model._models[0].uncertainty == 7.3


# ---------------------------------------------------------------------------
# never_create_model flag
# ---------------------------------------------------------------------------


class TestNeverCreateModel:
    """Tests for ``never_create_model=True`` on ring nav models.

    Exercises the metadata-only path: ``_create_model`` runs the feature filter but
    skips building ``NavModelResult`` image entries, leaving ``model._models`` empty
    while still recording feature counts in ``_metadata``. Verifies logger is passed
    through to ``RingFeatureFilter`` and that no image-generation side effects occur.
    """

    def test_never_create_model_no_images(self) -> None:
        """Populate ``_metadata`` and skip ``NavModelResult`` list growth.

        Notes:
            Sets ``never_create_model=True`` after resetting ``model._metadata``.
            Filter still runs (one surviving mock feature). Asserts
            ``model._models`` is empty, ``feature_count`` metadata is ``1``, and
            ``RingFeatureFilter`` receives ``logger=model._logger``.
        """
        obs = _make_obs()
        _make_bp(obs)

        with patch('nav.nav_model.nav_model_rings.RingFeatureFilter') as mock_filter_cls:
            mock_feature = MagicMock()
            mock_feature.name = 'Test'
            mock_feature.feature_type.value = 'RINGLET'
            mock_feature.all_base_radii.return_value = []
            mock_filter_inst = MagicMock()
            mock_filter_cls.return_value = mock_filter_inst
            mock_filter_inst.filter.return_value = [mock_feature]

            model = _make_rings_model(obs)
            model._metadata = {}
            model._create_model(
                always_create_model=False,
                never_create_model=True,
                create_annotations=False,
            )

        assert mock_filter_cls.call_count == 1
        assert mock_filter_cls.call_args.kwargs['logger'] is model._logger
        assert model._models == []
        assert model._metadata['planet'] == 'SATURN'
        assert model._metadata['feature_count'] == 1

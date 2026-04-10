"""Simulated ring navigation model.

This module provides a navigation model for simulated rings created in the GUI.

The simulated ring model uses a different rendering path from the real ring model
(``NavModelRings``): instead of computing backplane-based ring radii and applying
``RingFeature.render()``, it delegates image generation to
``nav.sim.sim_ring.render_ring()``, which operates entirely in pixel space.

The data model types (``RingFeature``, ``RingEdgeData``) are shared with the real
model for two purposes:

1. **Validation**: ``RingFeature.from_config()`` validates ``sim_params`` structure
   at construction time, catching authoring errors in the same way as the YAML config
   path.
2. **Annotations**: ``_create_edge_annotations`` requires an ``edge_info_list``; the
   feature's ``uncertainty`` field is wired to ``NavModelResult.uncertainty``.

The rendering path itself is NOT shared because simulated rings use pixel-space
geometry rather than backplane geometry. Unifying the two paths would require either
adding pixel-space backplane support to ``RingFeature`` (which would break its clear
single responsibility) or moving simulation physics into the rings subpackage (which
would couple the domain model to the simulator). Keeping them separate is the
correct trade-off.
"""

from typing import Any

import oops

from nav.annotation import Annotations
from nav.config import Config
from nav.sim.sim_ring import compute_border_atop_simulated, render_ring
from nav.support.time import now_dt
from nav.support.types import NDArrayBoolType

from .nav_model_result import NavModelResult
from .nav_model_rings_base import NavModelRingsBase
from .rings import RingFeature


class NavModelRingsSimulated(NavModelRingsBase):
    """Navigation model for simulated rings created in the GUI.

    Uses ``render_ring()`` for image generation (pixel-space) and
    ``RingFeature``/``RingEdgeData`` for annotation data and uncertainty.
    """

    def __init__(
        self,
        name: str,
        obs: oops.Observation,
        ring_name: str,
        sim_params: dict[str, Any],
        *,
        config: Config | None = None,
    ) -> None:
        """Create a navigation model for simulated rings.

        Parameters:
            name: The name of the model.
            obs: The Observation object containing image data.
            ring_name: The name of the ring.
            sim_params: Dictionary of parameters saved by the GUI JSON. Expected keys:
                name, feature_type, center_v, center_u, range, shading_distance,
                inner_data, outer_data. inner_data and outer_data are lists of dicts
                with keys: mode, a, rms, ae, long_peri, rate_peri. Extra keys are
                ignored.
            config: Configuration object to use. If None, uses DEFAULT_CONFIG.
        """
        super().__init__(name, obs, config=config)
        self._ring_name = ring_name.upper()
        self._sim_params = sim_params.copy()

    def create_model(
        self,
        *,
        always_create_model: bool = False,
        never_create_model: bool = False,
        create_annotations: bool = True,
    ) -> None:
        """Create the internal model representation for simulated rings.

        Parameters:
            always_create_model: If True, creates a model even if it won't have
                useful contents.
            never_create_model: If True, only creates metadata without generating
                a model or annotations.
            create_annotations: If True, creates text annotations for the model.
        """
        metadata: dict[str, Any] = {}
        start_time = now_dt()
        metadata['start_time'] = start_time.isoformat()
        metadata['end_time'] = None
        metadata['elapsed_time_sec'] = None

        self._metadata = metadata
        self._models.clear()

        with self._logger.open(f'CREATE SIMULATED RINGS MODEL FOR: {self._ring_name}'):
            self._create_model(
                always_create_model=always_create_model,
                never_create_model=never_create_model,
                create_annotations=create_annotations,
            )

        end_time = now_dt()
        metadata['end_time'] = end_time.isoformat()
        metadata['elapsed_time_sec'] = (end_time - start_time).total_seconds()

    def _create_model(
        self,
        always_create_model: bool,
        never_create_model: bool,
        create_annotations: bool,
    ) -> None:
        """Create the internal model for simulated rings.

        Parameters:
            always_create_model: If True, creates a model even if it won't have
                useful contents.
            never_create_model: If True, only creates metadata without rendering.
            create_annotations: If True, creates text annotations for the model.
        """
        obs = self.obs
        p = self._sim_params

        time = obs.sim_time
        epoch = obs.sim_epoch

        data_size_v = int(obs.data_shape_v)
        data_size_u = int(obs.data_shape_u)

        # Parse sim_params into a RingFeature for validation and annotations.
        # The feature_type defaults to RINGLET if not specified, to match historical
        # behavior when the field was absent.
        feature_config = _sim_params_to_feature_config(p)
        ring_feature = RingFeature.from_config(self._ring_name, feature_config)

        # Render via sim_ring (pixel-space, not backplane-based)
        sim_img = obs.make_extfov_zeros()
        center_v_data = float(p.get('center_v', data_size_v / 2.0))
        center_u_data = float(p.get('center_u', data_size_u / 2.0))
        center_v_extfov = center_v_data + obs.extfov_margin_v
        center_u_extfov = center_u_data + obs.extfov_margin_u
        ring_params_extfov = dict(p)
        ring_params_extfov['center_v'] = center_v_extfov
        ring_params_extfov['center_u'] = center_u_extfov
        render_ring(
            sim_img,
            ring_params_extfov,
            0.0,
            0.0,
            time=time,
            epoch=epoch,
            shade_solid=True,
        )

        ring_mask = sim_img != 0.0
        range_val = p.get('range', 0.0)

        annotations = None
        if create_annotations:
            annotations = self._create_simulated_edge_annotations(
                obs,
                ring_feature,
                ring_mask,
                center_v=float(p.get('center_v', data_size_v / 2.0)),
                center_u=float(p.get('center_u', data_size_u / 2.0)),
                time=time,
                epoch=epoch,
            )

        self._metadata['confidence'] = 1.0

        self._models.append(
            NavModelResult(
                model_img=sim_img,
                model_mask=ring_mask,
                weighted_mask=None,
                range=range_val,
                blur_amount=None,
                uncertainty=ring_feature.uncertainty,
                confidence=1.0,
                stretch_regions=None,
                annotations=annotations,
            )
        )

    def _create_simulated_edge_annotations(
        self,
        obs: oops.Observation,
        ring_feature: RingFeature,
        model_mask: NDArrayBoolType,
        *,
        center_v: float,
        center_u: float,
        time: float,
        epoch: float,
    ) -> Annotations:
        """Create annotations for simulated ring edges.

        Parameters:
            obs: The observation object.
            ring_feature: Parsed ring feature providing edge data and labels.
            model_mask: Model mask array.
            center_v: Ring center v-coordinate in data coordinates.
            center_u: Ring center u-coordinate in data coordinates.
            time: Observation time in TDB seconds.
            epoch: Epoch for mode calculations in TDB seconds.

        Returns:
            Annotations for all present ring edges.
        """
        data_size_v = int(obs.data_shape_v)
        data_size_u = int(obs.data_shape_u)
        edge_info_list: list[tuple[NDArrayBoolType, str, str]] = []
        labels = ring_feature.edge_labels
        feature_name = ring_feature.name or 'UNNAMED'

        for edge_data, edge_type in [
            (ring_feature.inner_edge, 'inner'),
            (ring_feature.outer_edge, 'outer'),
        ]:
            if edge_data is None:
                continue
            label = labels[edge_type]
            label_text = f'{feature_name} {label}'

            base = edge_data.base_orbit
            edge_mask = compute_border_atop_simulated(
                data_size_v,
                data_size_u,
                center_v,
                center_u,
                a=base.a,
                ae=base.ae,
                long_peri=base.long_peri,
                rate_peri=base.rate_peri,
                epoch=epoch,
                time=time,
            )

            edge_mask_extfov: NDArrayBoolType = obs.make_extfov_false()
            edge_mask_extfov[
                obs.extfov_margin_v : obs.extfov_margin_v + data_size_v,
                obs.extfov_margin_u : obs.extfov_margin_u + data_size_u,
            ] = edge_mask

            edge_info_list.append((edge_mask_extfov, label_text, label))

        return self._create_edge_annotations(obs, edge_info_list, model_mask)


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------


def _sim_params_to_feature_config(p: dict[str, Any]) -> dict[str, Any]:
    """Convert GUI sim_params dict to a feature config dict for RingFeature.from_config().

    The GUI stores ring parameters in a flat dict with inner_data / outer_data lists.
    This function adapts that format to the canonical feature config format expected by
    ``from_config()``, which validates all fields.

    Parameters:
        p: GUI sim_params dictionary.

    Returns:
        Feature config dict suitable for ``RingFeature.from_config()``.
    """
    raw_inner = p.get('inner_data') or None
    raw_outer = p.get('outer_data') or None

    # Normalize empty lists to None so from_config sees them as absent
    if isinstance(raw_inner, list) and len(raw_inner) == 0:
        raw_inner = None
    if isinstance(raw_outer, list) and len(raw_outer) == 0:
        raw_outer = None

    feature_type_raw = p.get('feature_type', 'RINGLET')

    config: dict[str, Any] = {
        'feature_type': feature_type_raw,
        'name': p.get('name'),
    }
    if raw_inner is not None:
        config['inner_data'] = raw_inner
    if raw_outer is not None:
        config['outer_data'] = raw_outer

    return config

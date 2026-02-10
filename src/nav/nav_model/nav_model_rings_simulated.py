"""Simulated ring navigation model.

This module provides a navigation model for simulated rings created in the GUI.
"""

from typing import Any, Optional

import oops

from nav.annotation import Annotations
from nav.config import Config
from nav.sim.sim_ring import compute_border_atop_simulated, render_ring
from nav.support.time import now_dt
from nav.support.types import NDArrayBoolType

from .nav_model_rings_base import NavModelRingsBase
from .nav_model_result import NavModelResult


class NavModelRingsSimulated(NavModelRingsBase):
    """Navigation model for simulated rings created in the GUI.

    This class creates navigation models for rings that were defined in the
    simulated image creation GUI, using the ring parameters directly rather
    than computing them from ephemeris data.
    """

    def __init__(
        self,
        name: str,
        obs: oops.Observation,
        ring_name: str,
        sim_params: dict[str, Any],
        *,
        config: Optional[Config] = None,
    ) -> None:
        """Creates a navigation model for simulated rings.

        Parameters:
            name: The name of the model.
            obs: The Observation object containing image data.
            ring_name: The name of the ring.
            sim_params: Dictionary of parameters saved by the GUI JSON. Expected keys:
                name, feature_type, center_v, center_u, range,
                shading_distance,
                inner_data, outer_data.
                inner_data and outer_data are lists of dictionaries with the following keys:
                    mode, a, rms, ae, long_peri, rate_peri.
                Extra keys are ignored.
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
        """Creates the internal model representation for simulated rings.

        Parameters:
            always_create_model: If True, creates a model even if it won't have useful contents.
            never_create_model: If True, only creates metadata without generating a model or
                annotations.
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
        """Creates the internal model representation for simulated rings.

        Parameters:
            always_create_model: If True, creates a model even if it won't have useful contents.
            never_create_model: If True, only creates metadata without generating a model.
            create_annotations: If True, creates text annotations for the model.
        """

        obs = self.obs
        p = self._sim_params

        # Get time and epoch from observation or use defaults
        time = obs.sim_time
        epoch = obs.sim_epoch

        # Get data size for center coordinate calculation
        data_size_v = int(obs.data_shape_v)
        data_size_u = int(obs.data_shape_u)

        # Create a temporary image for this ring in extended FOV
        sim_img = obs.make_extfov_zeros()
        # Get center coordinates in data coordinates
        center_v_data = float(p.get('center_v', data_size_v / 2.0))
        center_u_data = float(p.get('center_u', data_size_u / 2.0))
        # Convert to extended FOV coordinates by adding margins
        center_v_extfov = center_v_data + obs.extfov_margin_v
        center_u_extfov = center_u_data + obs.extfov_margin_u
        # Create modified params with adjusted center for extended FOV coordinates
        ring_params_extfov = dict(p)
        ring_params_extfov['center_v'] = center_v_extfov
        ring_params_extfov['center_u'] = center_u_extfov
        # To fake the normal ring modeling process, we shade solid rings because we
        # don't know what else is in the area between the edges
        render_ring(
            sim_img,
            ring_params_extfov,
            0.0,
            0.0,
            time=time,
            epoch=epoch,
            shade_solid=True,
        )

        # Update model and mask
        ring_mask = sim_img != 0.0

        # Range: set to a constant value for all ring pixels
        range_val = p.get('range', 0.0)

        # Create annotations if requested
        annotations = None
        if create_annotations:
            annotations = self._create_simulated_edge_annotations(obs, ring_mask)

        self._metadata['confidence'] = 1.0

        result = NavModelResult(
            model_img=sim_img,
            model_mask=ring_mask,
            weighted_mask=None,
            range=range_val,
            blur_amount=None,
            uncertainty=0.0,
            confidence=1.0,
            stretch_regions=None,
            annotations=annotations,
        )
        self._models.append(result)

    def _create_simulated_edge_annotations(
        self, obs: oops.Observation, model_mask: NDArrayBoolType
    ) -> Annotations:
        """Create annotations for simulated ring edges using the unified base method.

        Parameters:
            obs: The observation object.
            model_mask: Model mask array.

        Returns:
            Annotations object containing all ring edge annotations.
        """

        # Get time and epoch
        time = obs.sim_time
        epoch = obs.sim_epoch

        data_size_v = int(obs.data_shape_v)
        data_size_u = int(obs.data_shape_u)

        # Build edge_info_list for the base class method
        edge_info_list: list[tuple[NDArrayBoolType, str, str]] = []

        p = self._sim_params
        feature_name = p.get('name', 'UNNAMED')
        feature_type = p.get('feature_type', 'RINGLET')
        center_v = float(p.get('center_v', data_size_v / 2.0))
        center_u = float(p.get('center_u', data_size_u / 2.0))

        # Get inner and outer edge mode 1 data
        inner_data = p.get('inner_data', [])
        outer_data = p.get('outer_data', [])

        inner_mode1 = next((m for m in inner_data if m.get('mode') == 1), None)
        outer_mode1 = next((m for m in outer_data if m.get('mode') == 1), None)

        # Process inner edge
        if inner_mode1 is not None:
            inner_a = float(inner_mode1.get('a', 0.0))
            if inner_a > 0:
                inner_ae = float(inner_mode1.get('ae', 0.0))
                inner_long_peri = float(inner_mode1.get('long_peri', 0.0))
                inner_rate_peri = float(inner_mode1.get('rate_peri', 0.0))

                edge_label = 'IEG' if feature_type == 'GAP' else 'IER'
                label_text = f'{feature_name} {edge_label}'

                # Compute edge mask using simulated border_atop
                edge_mask = compute_border_atop_simulated(
                    data_size_v,
                    data_size_u,
                    center_v,
                    center_u,
                    a=inner_a,
                    ae=inner_ae,
                    long_peri=inner_long_peri,
                    rate_peri=inner_rate_peri,
                    epoch=epoch,
                    time=time,
                )

                # Embed into extended FOV
                edge_mask_extfov = obs.make_extfov_false()
                edge_mask_extfov[
                    obs.extfov_margin_v : obs.extfov_margin_v + data_size_v,
                    obs.extfov_margin_u : obs.extfov_margin_u + data_size_u,
                ] = edge_mask

                edge_info_list.append((edge_mask_extfov, label_text, edge_label))

        # Process outer edge
        if outer_mode1 is not None:
            outer_a = float(outer_mode1.get('a', 0.0))
            if outer_a > 0:
                outer_ae = float(outer_mode1.get('ae', 0.0))
                outer_long_peri = float(outer_mode1.get('long_peri', 0.0))
                outer_rate_peri = float(outer_mode1.get('rate_peri', 0.0))

                edge_label = 'OEG' if feature_type == 'GAP' else 'OER'
                label_text = f'{feature_name} {edge_label}'

                # Compute edge mask using simulated border_atop
                edge_mask = compute_border_atop_simulated(
                    data_size_v,
                    data_size_u,
                    center_v,
                    center_u,
                    a=outer_a,
                    ae=outer_ae,
                    long_peri=outer_long_peri,
                    rate_peri=outer_rate_peri,
                    epoch=epoch,
                    time=time,
                )

                # Embed into extended FOV
                edge_mask_extfov = obs.make_extfov_false()
                edge_mask_extfov[
                    obs.extfov_margin_v : obs.extfov_margin_v + data_size_v,
                    obs.extfov_margin_u : obs.extfov_margin_u + data_size_u,
                ] = edge_mask

                edge_info_list.append((edge_mask_extfov, label_text, edge_label))

        # Use the unified base class method
        return self._create_edge_annotations(obs, edge_info_list, model_mask)

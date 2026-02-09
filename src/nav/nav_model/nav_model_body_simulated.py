from typing import Any, Optional

from oops import Observation
import numpy as np

from nav.config import Config
from nav.sim.sim_body import create_simulated_body
from nav.support.time import now_dt
from nav.support.types import NDArrayBoolType, NDArrayFloatType

from .nav_model_body_base import NavModelBodyBase
from .nav_model_result import NavModelResult


class NavModelBodySimulated(NavModelBodyBase):
    def __init__(self,
                 name: str,
                 obs: Observation,
                 body_name: str,
                 sim_params: dict[str, Any],
                 *,
                 config: Optional[Config] = None) -> None:
        """Navigation model that uses simulated body parameters to build a model/mask.

        Parameters:
            name: Name of this model instance.
            obs: Observation containing image geometry (used for output shapes/margins).
            body_name: Logical body name for metadata/labels.
            sim_params: Dictionary of parameters saved by the GUI JSON. Expected keys:

                - name, center_v, center_u, range,
                - axis1, axis2, axis3,
                - rotation_z (deg), rotation_tilt (deg),
                - illumination_angle (deg), phase_angle (deg),

                Ignored keys:

                - crater_fill, crater_min_radius, crater_max_radius,
                - crater_power_law_exponent, crater_relief_scale,
                - anti_aliasing.

                Extra keys are ignored.
                Note that anti_aliasing is always set to the max when creating this model
                and the anti_aliasing parameter is ignored.
            config: Optional configuration.
        """

        super().__init__(name, obs, config=config)
        self._body_name = body_name.upper()
        self._sim_params = sim_params.copy()

    def create_model(self,
                     *,
                     always_create_model: bool = False,
                     never_create_model: bool = False,
                     create_annotations: bool = True) -> None:
        """Create the simulated model, mask, limb, and annotations."""

        metadata: dict[str, Any] = {}
        start_time = now_dt()
        metadata['start_time'] = start_time.isoformat()
        metadata['end_time'] = None
        metadata['elapsed_time_sec'] = None
        metadata['body_name'] = self._body_name
        self._metadata = metadata
        self._models.clear()

        with self._logger.open(f'CREATE SIMULATED BODY MODEL FOR: {self._body_name}'):
            self._create_model(always_create_model=always_create_model,
                               never_create_model=never_create_model,
                               create_annotations=create_annotations)

        end_time = now_dt()
        metadata['end_time'] = end_time.isoformat()
        metadata['elapsed_time_sec'] = (end_time - start_time).total_seconds()

    def _create_model(self,
                      *,
                      always_create_model: bool,
                      never_create_model: bool,
                      create_annotations: bool) -> None:
        """Generate the model image from the saved GUI parameters and build masks."""

        p = self._sim_params

        # Determine normal (non-extended) image size and extended-FOV margins
        data_size_v = int(self.obs.data_shape_v)
        data_size_u = int(self.obs.data_shape_u)
        ext_margin_v = int(self.obs.extfov_margin_v)
        ext_margin_u = int(self.obs.extfov_margin_u)

        # Convert GUI degrees to radians
        rotation_z_rad = np.radians(p.get('rotation_z', 0.0))
        rotation_tilt_rad = np.radians(p.get('rotation_tilt', 0.0))
        illumination_angle_rad = np.radians(p.get('illumination_angle', 0.0))
        phase_angle_rad = np.radians(p.get('phase_angle', 0.0))

        # Other parameters
        center_v = float(p.get('center_v', data_size_v / 2.0))
        center_u = float(p.get('center_u', data_size_u / 2.0))
        axis1 = float(p.get('axis1', 0.0))
        axis2 = float(p.get('axis2', 0.0))
        axis3 = float(p.get('axis3', min(axis1, axis2)))

        # Build simulated model image (float 0..1) at the normal image size
        sim_img = create_simulated_body(
            size=(data_size_v, data_size_u),
            center=(center_v, center_u),
            axis1=axis1,
            axis2=axis2,
            axis3=axis3,
            rotation_z=rotation_z_rad,
            rotation_tilt=rotation_tilt_rad,
            illumination_angle=illumination_angle_rad,
            phase_angle=phase_angle_rad,
            anti_aliasing=1,
        )

        # Create masks
        body_mask = sim_img > 0.0
        # Limb mask using shared helper
        limb_mask = self._compute_limb_mask_from_body_mask(body_mask)

        # Embed into full extended-FOV arrays at the margins
        model_img_full = self.obs.make_extfov_zeros()
        limb_mask_full = self.obs.make_extfov_false()
        body_mask_full = self.obs.make_extfov_false()

        slice_v = slice(ext_margin_v, ext_margin_v + data_size_v)
        slice_u = slice(ext_margin_u, ext_margin_u + data_size_u)
        model_img_full[slice_v, slice_u] = sim_img
        limb_mask_full[slice_v, slice_u] = limb_mask
        body_mask_full[slice_v, slice_u] = body_mask

        # Range: optionally provided via parameters; set inside body, inf elsewhere
        range_val = self._sim_params.get('range', np.inf)

        annotations = None
        if create_annotations:
            v_center_data = int(round(center_v))
            u_center_data = int(round(center_u))
            annotations = self._create_annotations(u_center_data, v_center_data,
                                                  model_img_full,
                                                  limb_mask_full,
                                                  body_mask_full)

        self._metadata['confidence'] = 1.0

        result = NavModelResult(
            model_img=model_img_full,
            model_mask=body_mask_full,
            weighted_mask=None,
            range=range_val,
            blur_amount=None,
            uncertainty=None,
            confidence=1.0,
            stretch_regions=None,
            annotations=annotations,
        )
        self._models.append(result)

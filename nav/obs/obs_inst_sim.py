import json
from typing import Any, Optional, cast

from filecache import FCPath
import numpy as np
import oops
from oops.observation.snapshot import Snapshot
from psfmodel import GaussianPSF
from starcat import Star

from nav.config import DEFAULT_CONFIG, DEFAULT_LOGGER, Config
from nav.obs.obs_snapshot_inst import ObsSnapshotInst
from nav.support.sim import create_simulated_body
from nav.support.types import MutableStar, PathLike


class ObsSim(ObsSnapshotInst):
    """Observation backed by a description of simulated bodies and stars."""

    def __init__(self,
                 snapshot: Snapshot,
                 **kwargs: Any) -> None:
        super().__init__(snapshot, **kwargs)

    @staticmethod
    def from_file(path: PathLike,
                  *,
                  config: Optional[Config] = None,
                  extfov_margin_vu: Optional[tuple[int, int]] = None,
                  **_kwargs: Any) -> 'ObsSim':
        """Creates an ObsSim from a JSON file.

        Parameters:
            path: Path to the JSON description file.
            config: Navigation configuration. If None, uses defaults.
            extfov_margin_vu: Optional extended FOV margins (v,u) to add around the image.
            **_kwargs: Additional keyword arguments (none for this instrument).
        """

        config = config or DEFAULT_CONFIG
        logger = DEFAULT_LOGGER

        json_path = FCPath(path).absolute()
        logger.debug(f'Reading simulated image JSON {json_path}')

        with json_path.open() as f:
            sim_params = json.load(f)

        # Required fields
        try:
            size_v = int(sim_params['size_v'])
            size_u = int(sim_params['size_u'])
            offset_v = float(sim_params.get('offset_v', 0.0))
            offset_u = float(sim_params.get('offset_u', 0.0))
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError('Invalid of missing size/off field in simulated image '
                             f'JSON file "{json_path}": {e}') from e

        # Create base blank image
        img = np.zeros((size_v, size_u), dtype=np.float64)

        # Build a basic Snapshot with a flat FOV and dummy geometry
        fov = oops.fov.FlatFOV((1.0, 1.0), (size_u, size_v))
        snapshot = oops.observation.snapshot.Snapshot(
            axes=('v', 'u'),
            tstart=0.0,
            texp=1.0,
            fov=fov,
            path='SSB',
            frame='J2000',
        )
        # Store data and the full JSON dictionary for future use
        snapshot.abspath = json_path
        snapshot.insert_subfield('data', img)

        snapshot.sim_params = sim_params
        snapshot.sim_offset_v = offset_v
        snapshot.sim_offset_u = offset_u

        # Add the stars first since they're at infinite range
        ObsSim._add_simulated_stars(snapshot, sim_params)

        # Add the bodies second in front of the stars
        ObsSim._add_simulated_bodies(snapshot, sim_params)

        # Determine extfov margins
        inst_config = config.category('sim')
        if extfov_margin_vu is None:
            extfov_margin_vu_entry = inst_config['extfov_margin_vu']
            if isinstance(extfov_margin_vu_entry, dict):
                extfov_margin_vu = extfov_margin_vu_entry[size_v]
            else:
                extfov_margin_vu = extfov_margin_vu_entry

        new_obs = ObsSim(snapshot, config=config, extfov_margin_vu=extfov_margin_vu,
                         simulated=True)
        new_obs._inst_config = inst_config

        new_obs.spice_kernels = ['fake_kernel1.txt', 'fake_kernel2.txt']
        new_obs._closest_planet = sim_params.get('closest_planet', None)

        return new_obs

    @staticmethod
    def _add_simulated_stars(obs: 'ObsSim',
                             sim_params: dict[str, Any]) -> None:
        """Build the simulated stars for the observation."""

        img = obs.data
        size_v, size_u = img.shape
        offset_v = obs.sim_offset_v
        offset_u = obs.sim_offset_u

        stars_params = sim_params.get('stars', [])
        sim_star_list: list[MutableStar] = []
        for i, star_params in enumerate(stars_params):
            star = cast(MutableStar, Star())
            # Required by NavModelStars
            star.unique_number = i + 1
            star.catalog_name = str(star_params.get('catalog_name', 'SIM'))
            star.pretty_name = str(star_params.get('name', f'SIM-{i+1}'))
            star.name = star.pretty_name
            # Position in pixel coordinates
            star.v = float(star_params.get('v', size_v / 2))
            star.u = float(star_params.get('u', size_u / 2))
            # No movement by default
            star.move_v = float(star_params.get('move_v', 0.0))
            star.move_u = float(star_params.get('move_u', 0.0))
            # VMAG and spectral class
            star.vmag = float(star_params.get('vmag', 8.0))
            star.spectral_class = str(star_params.get('spectral_class', 'G2'))
            # Derive temperature and B/V mags similar to flows in NavModelStars
            star.temperature = Star.temperature_from_sclass(star.spectral_class)
            star.temperature_faked = (star.temperature is None)
            if star.temperature is None:
                star.temperature = 5780.0
            # Johnson magnitudes
            star.johnson_mag_v = star.vmag
            bmv = Star.bmv_from_sclass(star.spectral_class or 'G2') or 0.63
            star.johnson_mag_b = star.johnson_mag_v + bmv
            star.johnson_mag_faked = False
            # Proper motion adjusted RA/DEC placeholders
            star.ra_pm = 0.0
            star.dec_pm = 0.0
            # Conflicts default
            star.conflicts = ''
            # PSF size and DN scale
            star.psf_size = tuple(star_params.get('psf_size', (5, 5)))
            star.dn = 2.512 ** -(star.vmag - 4.0)
            sim_star_list.append(star)

            star_offset_v = star.v + offset_v
            star_offset_u = star.u + offset_u
            v_int = int(star_offset_v)
            u_int = int(star_offset_u)
            v_frac = star_offset_v - v_int
            u_frac = star_offset_u - u_int

            psf_size_half_u = int(star.psf_size[1] + np.round(abs(star.move_u))) // 2
            psf_size_half_v = int(star.psf_size[0] + np.round(abs(star.move_v))) // 2

            max_move_steps = 1  # TODO stars_config.max_movement_steps
            move_gran = max(abs(star.move_u) / max_move_steps,
                            abs(star.move_v) / max_move_steps)
            move_gran = np.clip(move_gran, 0.1, 1.0)

            # psf = self.obs.star_psf()
            sigma = star_params.get('psf_sigma', 3.0)
            psf = GaussianPSF(sigma=sigma)

            if (u_int < psf_size_half_u or
                u_int >= img.shape[1]-psf_size_half_u or
                v_int < psf_size_half_v or
                v_int >= img.shape[0]-psf_size_half_v):
                continue

            star_psf = psf.eval_rect((psf_size_half_v*2+1, psf_size_half_u*2+1),
                                     offset=(v_frac, u_frac),
                                     scale=star.dn,
                                     movement=(star.move_v, star.move_u),
                                     movement_granularity=move_gran)

            img[v_int-psf_size_half_v:v_int+psf_size_half_v+1,
                u_int-psf_size_half_u:u_int+psf_size_half_u+1] += star_psf

        obs.sim_star_list = sim_star_list

    @staticmethod
    def _add_simulated_bodies(obs: 'ObsSim',
                              sim_params: dict[str, Any]) -> None:
        """Add the simulated bodies to the observation."""

        img = obs.data
        size_v, size_u = img.shape
        offset_v = obs.sim_offset_v
        offset_u = obs.sim_offset_u

        body_models: list[dict[str, Any]] = sim_params.get('bodies', [])
        # If there is no "range" field, set it to the body number so the earlier
        # bodies are closer than the later ones.
        for body_number, body_params in enumerate(body_models):
            if 'range' not in body_params:
                body_params['range'] = body_number + 1

        # Sort the body models by range
        sorted_body_models = sorted(body_models, key=lambda x: x['range'], reverse=True)
        inventory = {}
        body_model_dict = {}

        for body_number, params in enumerate(sorted_body_models):
            body_name = params.get('name', f'SIM-BODY-{body_number+1}').upper()

            # Adjust center by top-level offsets
            center_v = float(params.get('center_v', size_v / 2.0)) + offset_v
            center_u = float(params.get('center_u', size_u / 2.0)) + offset_u

            semi_major_axis = float(params.get('semi_major_axis', 0.0))
            semi_minor_axis = float(params.get('semi_minor_axis', 0.0))
            semi_c_axis = float(params.get('semi_c_axis',
                                           min(semi_major_axis, semi_minor_axis)))

            rotation_z = np.radians(params.get('rotation_z', 0.0))
            rotation_tilt = np.radians(params.get('rotation_tilt', 0.0))
            illumination_angle = np.radians(params.get('illumination_angle', 0.0))
            phase_angle = np.radians(params.get('phase_angle', 0.0))

            rough = (float(params.get('rough_mean', 0.0)), float(params.get('rough_std', 0.0)))
            craters = float(params.get('craters', 0.0))
            anti_aliasing = float(params.get('anti_aliasing', 0.0))

            sim_body = create_simulated_body(
                size=(size_v, size_u),
                semi_major_axis=semi_major_axis,
                semi_minor_axis=semi_minor_axis,
                semi_c_axis=semi_c_axis,
                center=(center_v, center_u),
                rotation_z=rotation_z,
                rotation_tilt=rotation_tilt,
                illumination_angle=illumination_angle,
                phase_angle=phase_angle,
                rough=rough,
                craters=craters,
                anti_aliasing=anti_aliasing,
            )

            # Copy the body image into the image, overwriting any bodies there previously
            img[sim_body > 0] = sim_body[sim_body > 0]

            max_dim = max(semi_major_axis, semi_minor_axis, semi_c_axis)
            inventory_item = {
                'v_min_unclipped': center_v - max_dim,
                'v_max_unclipped': center_v + max_dim,
                'u_min_unclipped': center_u - max_dim,
                'u_max_unclipped': center_u + max_dim,
                'v_pixel_size': 2 * max_dim,
                'u_pixel_size': 2 * max_dim,
                'range': params['range'],
            }
            inventory[body_name] = inventory_item
            body_model_dict[body_name] = params

        obs.sim_body_models = body_model_dict
        obs.sim_inventory = inventory

    def star_min_usable_vmag(self) -> float:
        return 0.0

    def star_max_usable_vmag(self) -> float:
        return 100.0

    def get_public_metadata(self) -> dict[str, Any]:
        return {
            'image_path': str(self.abspath),
            'image_name': self.abspath.name,
            'instrument_host_lid': 'sim',
            'instrument_lid': 'sim',
            'image_shape_xy': self.data_shape_uv,
            'description': 'Simulated observation from JSON',
        }

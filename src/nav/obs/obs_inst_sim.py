import json
from pathlib import Path
from typing import Any, cast

import oops
from filecache import FCPath
from oops.observation.snapshot import Snapshot

from nav.config import DEFAULT_CONFIG, DEFAULT_LOGGER, Config
from nav.obs.obs_snapshot_inst import ObsSnapshotInst
from nav.sim.render import render_combined_model
from nav.support.types import PathLike


class ObsSim(ObsSnapshotInst):
    """Observation backed by a description of simulated bodies and stars."""

    def __init__(self, snapshot: Snapshot, **kwargs: Any) -> None:
        super().__init__(snapshot, **kwargs)

    @staticmethod
    def from_file(
        path: PathLike,
        *,
        config: Config | None = None,
        extfov_margin_vu: tuple[int, int] | None = None,
        **kwargs: Any,
    ) -> 'ObsSim':
        """Creates an ObsSim from a JSON file.

        Parameters:
            path: Path to the JSON description file.
            config: Navigation configuration. If None, uses defaults.
            extfov_margin_vu: Optional extended FOV margins (v,u) to add around the image.
            **kwargs: Additional keyword arguments.
                sim_params: Dictionary of parameters saved by the GUI JSON. If present,
                this will override the JSON file.
        """

        config = config or DEFAULT_CONFIG
        logger = DEFAULT_LOGGER

        provided_sim_params = kwargs.get('sim_params')
        json_path = FCPath(path)
        abspath = cast(Path, json_path.get_local_path()).absolute()
        if provided_sim_params is None:
            logger.debug(f'Reading simulated image JSON {json_path}')
            with json_path.open() as f:
                sim_params = json.load(f)
        else:
            sim_params = provided_sim_params
            logger.debug('Using provided sim_params')

        # Required fields
        try:
            size_v = int(sim_params['size_v'])
            size_u = int(sim_params['size_u'])
            offset_v = float(sim_params.get('offset_v', 0.0))
            offset_u = float(sim_params.get('offset_u', 0.0))
        except (KeyError, TypeError, ValueError) as e:
            if provided_sim_params is None:
                raise ValueError(
                    'Invalid or missing size/offset field in simulated image '
                    f'JSON file "{json_path}": {e}'
                ) from e
            else:
                raise ValueError(
                    'Invalid or missing size/offset field in provided '
                    'sim_params for simulated image JSON file '
                    f'"{json_path}": {e}'
                ) from e

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
        snapshot.image_url = str(json_path.absolute())
        snapshot.abspath = abspath

        snapshot.sim_params = sim_params
        snapshot.sim_offset_v = offset_v
        snapshot.sim_offset_u = offset_u
        snapshot.sim_time = float(sim_params.get('time', 0.0))
        snapshot.sim_epoch = float(sim_params.get('epoch', 0.0))

        # Render combined model
        logger.debug('Rendering combined simulated model')
        img_rendered, meta = render_combined_model(sim_params)
        snapshot.insert_subfield('data', img_rendered)
        # Attach metadata similar to previous attributes
        snapshot.sim_star_list = meta.get('stars', [])
        snapshot.sim_body_models = meta.get('bodies', {})
        snapshot.sim_rings = meta.get('rings', [])
        snapshot.sim_inventory = meta.get('inventory', {})
        snapshot.sim_body_order_near_to_far = meta.get('order_near_to_far', [])
        snapshot.sim_body_index_map = meta.get('body_index_map', None)
        snapshot.sim_body_mask_map = meta.get('body_mask_map', {})

        # Determine extfov margins
        inst_config = config.category('sim')
        if extfov_margin_vu is None:
            extfov_margin_vu_entry = inst_config['extfov_margin_vu']
            if isinstance(extfov_margin_vu_entry, dict):
                extfov_margin_vu = extfov_margin_vu_entry[size_v]
            else:
                extfov_margin_vu = extfov_margin_vu_entry

        snapshot._closest_planet = sim_params.get('closest_planet', None)
        new_obs = ObsSim(snapshot, config=config, extfov_margin_vu=extfov_margin_vu, simulated=True)
        new_obs._inst_config = inst_config

        new_obs.spice_kernels = ['fake_kernel1.txt', 'fake_kernel2.txt']

        return new_obs

    def star_min_usable_vmag(self) -> float:
        return 0.0

    def star_max_usable_vmag(self) -> float:
        return 100.0

    def get_public_metadata(self) -> dict[str, Any]:
        return {
            'image_path': self.image_url,
            'image_name': self.abspath.name,
            'instrument_host_lid': 'sim',
            'instrument_lid': 'sim',
            'image_shape_xy': self.data_shape_uv,
            'description': 'Simulated observation from JSON',
        }

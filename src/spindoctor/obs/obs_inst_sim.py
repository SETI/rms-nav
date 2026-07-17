import math
from pathlib import Path
from typing import Any, cast

import oops
from filecache import FCPath
from oops.observation.snapshot import Snapshot

from spindoctor.config import DEFAULT_CONFIG, IMAGE_LOGGER, Config
from spindoctor.obs.obs_snapshot_inst import ObsSnapshotInst
from spindoctor.sim.instruments import resolve_extfov_margin, resolve_sim_inst_config
from spindoctor.sim.render import render_combined_model
from spindoctor.sim.scene import load_sim_scene
from spindoctor.support.types import PathLike


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
        """Creates an ObsSim from a YAML scene file.

        Parameters:
            path: Path to the YAML scene file.
            config: Navigation configuration. If None, uses defaults.
            extfov_margin_vu: Optional extended FOV margins (v,u) to add around the image.
            **kwargs: Additional keyword arguments.
                sim_params: Flat sim-params mapping. If present, this overrides the
                scene file.
        """

        config = config or DEFAULT_CONFIG
        logger = IMAGE_LOGGER

        provided_sim_params = kwargs.get('sim_params')
        scene_path = FCPath(path)
        abspath = cast(Path, scene_path.get_local_path()).absolute()
        if provided_sim_params is None:
            logger.debug(f'Reading simulated image scene {scene_path}')
            sim_params = load_sim_scene(abspath)
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
                    f'scene file "{scene_path}": {e}'
                ) from e
            else:
                raise ValueError(
                    'Invalid or missing size/offset field in provided '
                    'sim_params for simulated image scene file '
                    f'"{scene_path}": {e}'
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
        # Store data and the full sim-params dictionary for future use
        snapshot.image_url = str(scene_path.absolute())
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
        snapshot.sim_body_index_map = meta.get('body_index_map')
        snapshot.sim_body_mask_map = meta.get('body_mask_map', {})

        # Select the per-instrument config block the sim is emulating (or the
        # generic sim block when no instrument is specified), so the
        # orchestrator sees the right units / noise / saturation settings.
        sim_block = config.category('sim')
        inst_config = resolve_sim_inst_config(
            config, sim_params.get('instrument'), sim_params.get('instrument_config')
        )
        if extfov_margin_vu is None:
            extfov_margin_vu = resolve_extfov_margin(inst_config, sim_block, size_v)

        # A scene may toggle camera-rotation fitting independently of the camera
        # it emulates.  In reality ``fit_camera_rotation`` is a per-camera
        # property, but a sim scene is a navigation fixture: it can ask the
        # navigator to solve for a planted roll on top of any instrument's optics
        # (e.g. Cassini NAC's sharp PSF) without pretending to be a camera that
        # happens to fit rotation.  The resolved block is shared live config, so
        # copy before overriding.
        fit_rotation_override = sim_params.get('fit_camera_rotation')
        if fit_rotation_override is not None:
            inst_config = {**inst_config, 'fit_camera_rotation': bool(fit_rotation_override)}

        snapshot._closest_planet = sim_params.get('closest_planet')
        new_obs = ObsSim(snapshot, config=config, extfov_margin_vu=extfov_margin_vu, simulated=True)
        new_obs._inst_config = cast(dict[str, Any], inst_config)

        new_obs.spice_kernels = ['fake_kernel1.txt', 'fake_kernel2.txt']

        return new_obs

    def star_min_usable_vmag(self) -> float:
        return 0.0

    def star_max_usable_vmag(self) -> float:
        """Returns the maximum usable magnitude for stars in this observation.

        Derived from the scene's own detector model rather than anchored to a
        reference exposure the way the real instruments do it: the sim
        renderer draws a star's PSF peak at ``signal_full_scale_dn *
        2.512**-vmag`` DN (see ``sim.render``), so the limiting magnitude is
        where that peak falls to twice the effective per-pixel noise sigma --
        the matched-filter detection boundary measured on single-star sim
        scenes.  Keeping this physical matters beyond the faint-star gate:
        the star NavModel synthesises each STAR feature's predicted SNR (and
        from it the CRLB position covariance and reliability score) from how
        far the star sits above this limit, so an arbitrarily permissive
        placeholder inflates every simulated star's SNR by tens of orders of
        magnitude and collapses its covariance to zero.

        Returns:
            The maximum usable magnitude for stars in this observation.  For
            calibrated-unit sim instruments the renderer applies no detector
            noise, so any rendered star is detectable and the limit is a
            generous constant.
        """
        inst_config = self._inst_config or {}
        inst_noise = inst_config.get('noise') or {}
        if inst_config.get('data_units', 'raw_dn') != 'raw_dn':
            # calibrated_if: the renderer leaves the composed I/F signal
            # noise-free (detector noise there is deferred sim scope).
            return 30.0
        # Mirror the renderer's resolution order: scene noise block first,
        # then the emulated instrument's config, then the sim defaults.
        sim_noise = self.config.category('sim')['noise']
        scene_noise = self.sim_params.get('noise') or {}
        full_scale_frac = float(
            scene_noise.get('signal_full_scale_frac', sim_noise['signal_full_scale_frac'])
        )
        signal_full_scale_dn = float(
            scene_noise.get(
                'signal_full_scale_dn', full_scale_frac * float(inst_noise['full_well_dn'])
            )
        )
        read_noise_dn = float(scene_noise.get('read_noise_dn', inst_noise['read_noise_dn']))
        # Poisson shot noise on the star's own counts keeps the effective
        # sigma above ~1 DN even on a read-noise-free frame.
        sigma_eff = max(read_noise_dn, 1.0)
        return 2.5 * math.log10(signal_full_scale_dn / (2.0 * sigma_eff))

    @property
    def camera(self) -> str:
        """The camera that took this observation.

        Returns:
            Always ``'SIM'``; a simulated scene has one synthetic camera.
        """
        return 'SIM'

    def get_public_metadata(self) -> dict[str, Any]:
        return {
            'image_path': self.image_url,
            'image_name': self.abspath.name,
            'instrument_host_lid': 'sim',
            'instrument_lid': 'sim',
            'image_shape_xy': self.data_shape_uv,
            'camera': self.camera,
            'description': 'Simulated observation from YAML scene',
        }

from typing import Any, Optional, cast

import numpy as np
from psfmodel import GaussianPSF
from starcat import Star

from nav.sim.sim_body import create_simulated_body
from nav.support.types import MutableStar


def render_stars(img: np.ndarray,
                 stars_params: list[dict[str, Any]],
                 offset_v: float,
                 offset_u: float
                 ) -> tuple[np.ndarray, list[MutableStar], list[dict[str, Any]]]:
    """Render stars into img. Returns (img, sim_star_list, star_render_info)."""
    size_v, size_u = img.shape
    sim_star_list: list[MutableStar] = []
    star_info: list[dict[str, Any]] = []

    for i, star_params in enumerate(stars_params):
        star = cast(MutableStar, Star())
        star.unique_number = i + 1
        star.catalog_name = str(star_params.get('catalog_name', 'SIM'))
        star.pretty_name = str(star_params.get('name', f'SIM-{i+1}'))
        star.name = star.pretty_name
        star.v = float(star_params.get('v', size_v / 2))
        star.u = float(star_params.get('u', size_u / 2))
        star.move_v = float(star_params.get('move_v', 0.0))
        star.move_u = float(star_params.get('move_u', 0.0))
        star.vmag = float(star_params.get('vmag', 8.0))
        star.spectral_class = str(star_params.get('spectral_class', 'G2'))
        star.temperature = Star.temperature_from_sclass(star.spectral_class)
        star.temperature_faked = (star.temperature is None)
        if star.temperature is None:
            star.temperature = 5780.0
        star.johnson_mag_v = star.vmag
        bmv = Star.bmv_from_sclass(star.spectral_class or 'G2') or 0.63
        star.johnson_mag_b = star.johnson_mag_v + bmv
        star.johnson_mag_faked = False
        star.ra_pm = 0.0
        star.dec_pm = 0.0
        star.conflicts = ''
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

        max_move_steps = 1  # TODO configurable
        move_gran = max(abs(star.move_u) / max_move_steps,
                        abs(star.move_v) / max_move_steps)
        move_gran = np.clip(move_gran, 0.1, 1.0)

        sigma = star_params.get('psf_sigma', 3.0)
        psf = GaussianPSF(sigma=sigma)

        # Stars where any part of the PSF would be off the edge of the image are ignored.
        # This is because PSF fitting will not work in these cases.
        if (u_int < psf_size_half_u or
            u_int >= img.shape[1]-psf_size_half_u or
            v_int < psf_size_half_v or
            v_int >= img.shape[0]-psf_size_half_v):
            # Still collect info for hit-testing
            star_info.append({
                'name': star.name,
                'center_v': star_offset_v,
                'center_u': star_offset_u,
                'sigma': sigma,
                'psf_half_v': psf_size_half_v,
                'psf_half_u': psf_size_half_u,
            })
            continue

        star_psf = psf.eval_rect((psf_size_half_v*2+1, psf_size_half_u*2+1),
                                 offset=(v_frac, u_frac),
                                 scale=star.dn,
                                 movement=(star.move_v, star.move_u),
                                 movement_granularity=move_gran)

        img[v_int-psf_size_half_v:v_int+psf_size_half_v+1,
            u_int-psf_size_half_u:u_int+psf_size_half_u+1] += star_psf

        star_info.append({
            'name': star.name,
            'center_v': star_offset_v,
            'center_u': star_offset_u,
            'sigma': sigma,
            'psf_half_v': psf_size_half_v,
            'psf_half_u': psf_size_half_u,
        })

    return img, sim_star_list, star_info


def render_bodies(
    img: np.ndarray,
    bodies_params: list[dict[str, Any]],
    offset_v: float,
    offset_u: float,
    *,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """Render bodies over img and return fields by name.

    Returns: a dict with keys:

      - img: np.ndarray the rendered image
      - bodies: dict[str, dict[str, Any]]
      - inventory: dict[str, dict[str, float]]
      - body_masks: list[np.ndarray]
      - order_near_to_far: list[str]
      - body_index_map: np.ndarray (int32), 1-based index into order_near_to_far or 0 if none
    """
    size_v, size_u = img.shape

    # Make a copy before we modify it with range info
    body_models = [dict(x) for x in bodies_params]
    for body_number, body_params in enumerate(body_models):
        if 'range' in body_params:
            body_params['range'] = float(body_params['range'])
        else:
            body_params['range'] = body_number + 1

    # Sort by range: far to near for composition; also prepare near-to-far order for hit-test
    sorted_body_models = sorted(body_models, key=lambda x: x['range'], reverse=True)
    order_near_to_far = [
        bp.get('name', f'SIM-BODY-{i+1}').upper()
        for i, bp in enumerate(sorted(body_models, key=lambda x: x['range']))
    ]

    inventory: dict[str, dict[str, float]] = {}
    body_model_dict: dict[str, dict[str, Any]] = {}
    body_masks: list[np.ndarray] = []
    body_mask_map: dict[str, np.ndarray] = {}
    body_index_map = np.zeros((size_v, size_u), dtype=np.int32)

    for body_number, params in enumerate(sorted_body_models):
        body_name = params.get('name', f'SIM-BODY-{body_number+1}').upper()

        center_v = float(params.get('center_v', size_v / 2.0)) + offset_v
        center_u = float(params.get('center_u', size_u / 2.0)) + offset_u

        axis1 = float(params.get('axis1', 0.0))
        axis2 = float(params.get('axis2', 0.0))
        axis3 = float(params.get('axis3', min(axis1, axis2)))

        rotation_z = np.radians(params.get('rotation_z', 0.0))
        rotation_tilt = np.radians(params.get('rotation_tilt', 0.0))
        illumination_angle = np.radians(params.get('illumination_angle', 0.0))
        phase_angle = np.radians(params.get('phase_angle', 0.0))

        crater_fill = float(params.get('crater_fill', 0.0))
        crater_min_radius = float(params.get('crater_min_radius', 0.05))
        crater_max_radius = float(params.get('crater_max_radius', 0.25))
        crater_power_law_exponent = float(params.get('crater_power_law_exponent', 3.0))
        crater_relief_scale = float(params.get('crater_relief_scale', 0.6))
        anti_aliasing = float(params.get('anti_aliasing', 1.0))
        # Use seed from render_bodies parameter, fall back to body-specific seed if provided
        body_seed = seed if seed is not None else params.get('seed')

        sim_body = create_simulated_body(
            size=(size_v, size_u),
            center=(center_v, center_u),
            axis1=axis1,
            axis2=axis2,
            axis3=axis3,
            rotation_z=rotation_z,
            rotation_tilt=rotation_tilt,
            illumination_angle=illumination_angle,
            phase_angle=phase_angle,
            crater_fill=crater_fill,
            crater_min_radius=crater_min_radius,
            crater_max_radius=crater_max_radius,
            crater_power_law_exponent=crater_power_law_exponent,
            crater_relief_scale=crater_relief_scale,
            anti_aliasing=anti_aliasing,
            seed=body_seed,
        )

        # Composition: overwrite where body contributes
        mask = sim_body > 0
        img[mask] = sim_body[mask]
        body_masks.append(mask)
        body_mask_map[body_name] = mask
        # Index into near-to-far order is 1-based
        near_index = order_near_to_far.index(body_name) + 1
        body_index_map[mask] = near_index

        max_dim = max(axis1, axis2, axis3) / 2.0  # Convert to half-width for dimension calculation
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

    return {
        'img': img,
        'bodies': body_model_dict,
        'inventory': inventory,
        'body_masks': body_masks,
        'body_mask_map': body_mask_map,
        'order_near_to_far': order_near_to_far,
        'body_index_map': body_index_map,
    }


def render_background_noise(img: np.ndarray, noise_level: float, seed: int) -> None:
    """Add Gaussian background noise to the image.

    Parameters:
        img: Image array to modify in-place.
        noise_level: Standard deviation of Gaussian noise (0-1).
        seed: Random seed for reproducibility.
    """
    if noise_level <= 0:
        return
    rng = np.random.RandomState(seed)
    noise = rng.normal(0.0, noise_level, size=img.shape)
    img[:] = np.clip(img + noise, 0.0, 1.0)


def render_background_stars(
        img: np.ndarray, n_stars: int, seed: int, psf_sigma: float = 0.9,
        distribution_exponent: float = 2.5) -> None:
    """Add random background stars to the image.

    Parameters:
        img: Image array to modify in-place (stars are added, not overwritten).
        n_stars: Number of stars to add (0-1000).
        seed: Random seed for reproducibility.
        psf_sigma: PSF sigma value for star rendering (default 0.9).
        distribution_exponent: Power law exponent for intensity distribution (default 2.5).
            Higher values make dimmer stars more common.
    """
    if n_stars <= 0:
        return
    size_v, size_u = img.shape
    rng = np.random.RandomState(seed)

    # Power law for intensity: weight toward dimmer stars
    # intensity = uniform^power where power > 1 makes dimmer stars more common
    uniform_samples = rng.uniform(0.0, 1.0, size=n_stars)
    intensities = uniform_samples ** distribution_exponent

    # PSF size: at least 11x11, but scale with sigma
    # Use at least 3*sigma pixels on each side, minimum 6 for 11x11
    psf_size_half = max(6, int(np.ceil(3.0 * psf_sigma)))

    psf = GaussianPSF(sigma=psf_sigma)

    for i in range(n_stars):
        # Random position
        v = rng.uniform(0.0, float(size_v))
        u = rng.uniform(0.0, float(size_u))

        v_int = int(v)
        u_int = int(u)
        v_frac = v - v_int
        u_frac = u - u_int

        # Skip if too close to edge
        if (u_int < psf_size_half or u_int >= size_u - psf_size_half or
            v_int < psf_size_half or v_int >= size_v - psf_size_half):
            continue

        # Generate PSF (normalized so peak is 1.0)
        star_psf = psf.eval_rect(
            (psf_size_half * 2 + 1, psf_size_half * 2 + 1),
            offset=(v_frac, u_frac),
            scale=1.0,  # Use scale=1.0 to get normalized PSF
            movement=(0.0, 0.0),
            movement_granularity=1.0
        )

        # Normalize PSF to have peak value of 1.0, then scale by intensity
        # This ensures stars are bright (peak brightness = intensity, not distributed)
        psf_max = np.max(star_psf)
        if psf_max > 0:
            star_psf = star_psf / psf_max * intensities[i]
        else:
            star_psf = star_psf * intensities[i]

        # Add to image (don't overwrite)
        img[v_int - psf_size_half:v_int + psf_size_half + 1,
            u_int - psf_size_half:u_int + psf_size_half + 1] += star_psf

    # Clip to valid range
    img[:] = np.clip(img, 0.0, 1.0)


def render_combined_model(
    sim_params: dict[str, Any],
    *,
    ignore_offset: bool = False
) -> tuple[np.ndarray, dict[str, Any]]:
    """Render stars then bodies from a full sim_params dict. Returns (img, meta).

    ignore_offset = True should be used when rendering the image in the GUI, but not
    when creating the simulated image to navigate.

    Parameters:
        sim_params: The parameters describing the simulated model.
        ignore_offset: Whether to ignore the offset.

    Returns:
        A tuple containing the image and metadata.
    """

    size_v = int(sim_params['size_v'])
    size_u = int(sim_params['size_u'])
    if not ignore_offset:
        offset_v = float(sim_params.get('offset_v', 0.0))
        offset_u = float(sim_params.get('offset_u', 0.0))
    else:
        offset_v = 0.0
        offset_u = 0.0

    img = np.zeros((size_v, size_u), dtype=np.float64)

    # Get random seed for background effects
    random_seed = int(sim_params.get('random_seed', 42))

    # Apply background noise first
    background_noise_intensity = float(
        sim_params.get('background_noise_intensity', 0.0))
    render_background_noise(img, background_noise_intensity, random_seed)

    # Then background stars
    background_stars_num = int(sim_params.get('background_stars_num', 0))
    background_stars_psf_sigma = float(
        sim_params.get('background_stars_psf_sigma', 0.9))
    background_stars_distribution_exponent = float(
        sim_params.get('background_stars_distribution_exponent', 2.5))
    render_background_stars(
        img, background_stars_num, random_seed,
        psf_sigma=background_stars_psf_sigma,
        distribution_exponent=background_stars_distribution_exponent)

    stars_params = sim_params.get('stars', []) or []
    bodies_params = sim_params.get('bodies', []) or []

    img, sim_star_list, star_info = render_stars(img, stars_params, offset_v, offset_u)

    # Pass seed to render_bodies for crater generation
    bodies_result = render_bodies(img, bodies_params, offset_v, offset_u, seed=random_seed)
    img = bodies_result['img']
    body_models = bodies_result['bodies']
    inventory = bodies_result['inventory']
    body_masks = bodies_result['body_masks']
    order_near_to_far = bodies_result['order_near_to_far']
    body_index_map = bodies_result['body_index_map']

    meta: dict[str, Any] = {
        'stars': sim_star_list,
        'bodies': body_models,
        'inventory': inventory,
        'star_info': star_info,
        'body_masks': body_masks,
        'order_near_to_far': order_near_to_far,
        'body_index_map': body_index_map,
    }
    return img, meta

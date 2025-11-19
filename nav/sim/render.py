from typing import Any, cast

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
) -> tuple[
    np.ndarray,
    dict[str, dict[str, Any]],
    dict[str, dict[str, float]],
    list[np.ndarray],
    list[str],
]:
    """Render bodies over img.

    Returns (img, body_models, inventory, body_masks, order_near_to_far_names).
    """
    size_v, size_u = img.shape

    body_models: list[dict[str, Any]] = bodies_params or []
    for body_number, body_params in enumerate(body_models):
        if 'range' not in body_params:
            body_params['range'] = body_number + 1

    # Sort by range: far to near for composition; also prepare near-to-far order for hit-test
    sorted_body_models = sorted(body_models, key=lambda x: x['range'], reverse=True)
    order_near_to_far = [
        bp.get('name', f'SIM-BODY-{i+1}').upper()
        for i, bp in enumerate(
            sorted(bodies_params or [], key=lambda x: x['range'])
        )
    ]

    inventory: dict[str, dict[str, float]] = {}
    body_model_dict: dict[str, dict[str, Any]] = {}
    body_masks: list[np.ndarray] = []

    for body_number, params in enumerate(sorted_body_models):
        body_name = params.get('name', f'SIM-BODY-{body_number+1}').upper()

        center_v = float(params.get('center_v', size_v / 2.0)) + offset_v
        center_u = float(params.get('center_u', size_u / 2.0)) + offset_u

        semi_major_axis = float(params.get('semi_major_axis', 0.0))
        semi_minor_axis = float(params.get('semi_minor_axis', 0.0))
        semi_c_axis = float(params.get('semi_c_axis', min(semi_major_axis, semi_minor_axis)))

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

        sim_body = create_simulated_body(
            size=(size_v, size_u),
            center=(center_v, center_u),
            semi_major_axis=semi_major_axis,
            semi_minor_axis=semi_minor_axis,
            semi_c_axis=semi_c_axis,
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
        )

        # Composition: overwrite where body contributes
        mask = sim_body > 0
        img[mask] = sim_body[mask]
        body_masks.append(mask)

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

    return img, body_model_dict, inventory, body_masks, order_near_to_far


def render_combined_model(
    sim_params: dict[str, Any], ignore_offset: bool = False
) -> tuple[np.ndarray, dict[str, Any]]:
    """Render stars then bodies from a full sim_params dict. Returns (img, meta)."""
    size_v = int(sim_params['size_v'])
    size_u = int(sim_params['size_u'])
    if not ignore_offset:
        offset_v = float(sim_params.get('offset_v', 0.0))
        offset_u = float(sim_params.get('offset_u', 0.0))
    else:
        offset_v = 0.0
        offset_u = 0.0

    img = np.zeros((size_v, size_u), dtype=np.float64)

    stars_params = sim_params.get('stars', []) or []
    bodies_params = sim_params.get('bodies', []) or []

    img, sim_star_list, star_info = render_stars(img, stars_params, offset_v, offset_u)
    img, body_models, inventory, body_masks, order_near_to_far = render_bodies(
        img, bodies_params, offset_v, offset_u
    )

    meta: dict[str, Any] = {
        'stars': sim_star_list,
        'bodies': body_models,
        'inventory': inventory,
        'star_info': star_info,
        'body_masks': body_masks,
        'order_near_to_far': order_near_to_far,
    }
    return img, meta

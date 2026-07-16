"""Scene-radiance stage: compose the noise-free signal image.

Composes the ring/body stack (far to near, nearer objects overwriting) into the
frame's normalized signal plane and the star field (catalog stars plus the
background sky) into the frame's point-source plane, applying the scene's
planted pointing offset and camera roll.  Feature truth (rendered star records,
body masks, inventory, z-order maps) is accumulated into ``frame.truth`` for the
renderer's output metadata.

Stars are point sources: each deposits its total flux
(``zero_point * 10**(-0.4 * vmag) * exposure_sec``) as a sub-pixel point mass in
the detector-native point-source plane (electrons for a CCD, DN for the
vidicon), so the whole-scene optics PSF is the star's only convolution.  The
background sky draws its counts from a cumulative star-count law and renders them
through the same flux/point-mass path.  Body / ring occlusion is mask-overwrite
rather than transparency compositing.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np

from spindoctor.config import DEFAULT_CONFIG
from spindoctor.sim.forward.artifacts_catalog import (
    resolve_detector_defaults,
    resolve_sky_pixel_scale_arcsec,
    resolve_star_flux_zero_point,
)
from spindoctor.sim.forward.body import render_single_body
from spindoctor.sim.forward.optics import effective_psf
from spindoctor.sim.forward.ring import composite_ring
from spindoctor.sim.forward.stages import SimFrame
from spindoctor.sim.forward.star import faint_sky_cutoff_mag, render_sky_counts, render_stars
from spindoctor.sim.instruments import resolve_sim_inst_config
from spindoctor.sim.seeds import derive_effect_seed
from spindoctor.sim.star_records import DEFAULT_PSF_SIZE
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType, NDArrayIntType

__all__ = ['compose_scene_radiance']


def _nominal_read_noise(instrument: str | None, domain: str) -> float:
    """The camera's nominal per-pixel read noise for the faint-sky cutoff.

    Read from the per-instrument catalog regardless of whether the scene enables
    read noise -- the cutoff asks whether a star could ever clear the camera's
    noise floor, a physical property of the detector.

    Parameters:
        instrument: The sim instrument name.
        domain: The zero point's unit domain ('electrons' or 'dn').

    Returns:
        The nominal read noise in the zero point's unit domain.
    """
    catalog = resolve_detector_defaults(instrument)
    if domain == 'dn':
        vidicon = catalog.get('vidicon') or {}
        line = float(vidicon.get('read_noise_line_dn', 1.0))
        pixel = float(vidicon.get('read_noise_pixel_dn', 1.0))
        return float(np.hypot(line, pixel))
    return float(catalog.get('read_noise_e', 1.0))


def compose_scene_radiance(
    frame: SimFrame,
    *,
    params: Mapping[str, Any],
    rng: np.random.Generator,
) -> None:
    """Compose the scene's noise-free radiance into the frame in place.

    Parameters:
        frame: The frame whose signal plane is composed in place; its
            ``truth`` dict receives the renderer output metadata (``stars``,
            ``bodies``, ``rings``, ``inventory``, ``star_info``,
            ``body_masks``, ``ring_masks``, ``order_near_to_far``,
            ``body_index_map``, ``body_mask_map``).
        params: The full scene mapping.
        rng: The stage generator.  Unused directly: this stage's randomized
            sub-effects (background stars, craters) run behind parameter-keyed
            caches that need scalar seeds, so they derive named sub-seeds from
            the scene's ``random_seed`` instead of consuming generator state.
    """
    del rng
    img = frame.signal
    point_e = frame.point_e
    size_v, size_u = img.shape
    # The signal plane is the oversampled grid (V*os, U*os).  Every pixel-space
    # quantity below (offsets, centres, radii, PSF sigmas, anti-aliasing widths)
    # is scaled by ``os`` so the radiance renders at the oversampled resolution;
    # the box downsample after optics returns it to the detector grid.  At
    # ``os == 1`` every scale factor is an exact multiply-by-one, so a scene
    # with no optics block renders identically to a detector-grid render.
    os = int(frame.oversample)
    random_seed = int(params.get('random_seed', 42))
    background_stars_seed = derive_effect_seed(random_seed, 'scene_radiance/background_stars')
    crater_seed = derive_effect_seed(random_seed, 'scene_radiance/craters')
    catalog_scatter_seed = derive_effect_seed(random_seed, 'scene_radiance/catalog_scatter')
    # Scene-level per-star position-scatter sigma (detector pixels), scaled to
    # the oversampled render grid alongside the other star pixel quantities.
    catalog_scatter_px = float(params.get('star_catalog_scatter_px', 0.0)) * os

    offset_v = float(params.get('offset_v', 0.0)) * os
    offset_u = float(params.get('offset_u', 0.0)) * os
    # Camera roll about the boresight, part of the planted pointing error the
    # navigator recovers; suppressed in GUI preview mode alongside the offset.
    offset_rotation_deg = float(params.get('offset_rotation_deg', 0.0))

    # A planted spacecraft-ephemeris error displaces bodies and rings by
    # parallax (1/range), computed at full precision on the oversampled grid.
    spk_error = params.get('spk_error') or {}

    # Resolve the per-instrument config once; stars record the scene PSF sigma
    # (or, absent a PSF, the instrument's configured sigma) for their diagnostics.
    inst_config = resolve_sim_inst_config(
        DEFAULT_CONFIG, params.get('instrument'), params.get('instrument_config')
    )
    star_psf_sigma = float(inst_config['star_psf_sigma'])

    # Stars are point sources deposited into the point-source plane; the scene
    # PSF (an explicit optics.psf block, the navigator-matched form, or the
    # instrument-defaults kernel) is their only convolution.  With no PSF a star
    # is a 1-pixel spike (the undersampled limit); the recorded sigma falls back
    # to the instrument's configured value in that case.
    scene_psf = effective_psf(params)
    if scene_psf is not None:
        sigma_v_psf = float(scene_psf['sigma_v'])
        rendered_sigma_det = max(sigma_v_psf, float(scene_psf.get('sigma_u', sigma_v_psf)))
    else:
        rendered_sigma_det = star_psf_sigma
    rendered_sigma = rendered_sigma_det * os

    # The photometric zero point and its unit domain (electrons for a CCD, DN for
    # the vidicon) drive the flux each star deposits into the point-source plane.
    zero_point, flux_domain = resolve_star_flux_zero_point(params.get('instrument'))
    exposure_sec = float(params.get('exposure_sec', 1.0))

    stars_params = params.get('stars', []) or []
    bodies_params = params.get('bodies', []) or []
    rings_params = params.get('rings', []) or []

    # Star positions and smear vectors are pixel-space, so they scale with the
    # oversampling factor for rendering; at os == 1 the copy is numerically
    # identical to the scene's stars.
    stars_params_scaled = [_scale_star_params(sp, os) for sp in stars_params]

    _render_sky(point_e, params, os=os, seed=background_stars_seed)

    point_e, sim_star_list, star_info = render_stars(
        point_e,
        stars_params_scaled,
        offset_v,
        offset_u=offset_u,
        zero_point=zero_point,
        exposure_sec=exposure_sec,
        rendered_sigma=rendered_sigma,
        rotation_deg=offset_rotation_deg,
        oversample=os,
        catalog_scatter_px=catalog_scatter_px,
        catalog_scatter_seed=catalog_scatter_seed,
    )
    del flux_domain

    # The camera roll and the planted spacecraft-ephemeris parallax are both
    # detector-space displacements; geometry is built in detector coordinates
    # about the detector centre, then multiplied by ``os`` for the oversampled
    # render.  The parallax of an object at physical range R km is the planted
    # image-plane error scaled by reference_range_km / R (near objects move
    # more than far ones); stars carry no such shift.
    roll_cos = float(np.cos(np.radians(offset_rotation_deg)))
    roll_sin = float(np.sin(np.radians(offset_rotation_deg)))
    det_center_v = (size_v / os) / 2.0
    det_center_u = (size_u / os) / 2.0
    spk_dv = float(spk_error.get('dv_px', 0.0))
    spk_du = float(spk_error.get('du_px', 0.0))
    spk_ref_range = float(spk_error.get('reference_range_km', 0.0))

    def _spk_shift(range_km: float) -> tuple[float, float]:
        """Parallax displacement in detector pixels for a body/ring range."""
        if not spk_error or range_km <= 0.0:
            return 0.0, 0.0
        factor = spk_ref_range / range_km
        return spk_dv * factor, spk_du * factor

    # Process rings: assign default z-order ranges, apply parallax, and scale
    # every pixel-space quantity to the oversampled grid.  The original scene
    # entries keep their detector-space values for the truth metadata.
    rings_scaled: list[dict[str, Any]] = []
    for ring_number, ring_params in enumerate(rings_params):
        if 'range' in ring_params:
            ring_params['range'] = float(ring_params['range'])
        else:
            # Default range: start after bodies (assuming bodies use 1, 2, 3, ...)
            # Use a large starting value to ensure rings are behind bodies by default
            ring_params['range'] = float(ring_number + 1000.0)
        ring_scaled = dict(ring_params)
        cv = float(ring_params.get('center_v', det_center_v))
        cu = float(ring_params.get('center_u', det_center_u))
        ring_range_km = ring_params.get('range_km')
        if ring_range_km is not None:
            sdv, sdu = _spk_shift(float(ring_range_km))
            cv += sdv
            cu += sdu
        ring_scaled['center_v'] = cv * os
        ring_scaled['center_u'] = cu * os
        if ring_params.get('shading_distance') is not None:
            ring_scaled['shading_distance'] = float(ring_params['shading_distance']) * os
        if ring_params.get('inner_data') is not None:
            ring_scaled['inner_data'] = _scale_ring_modes(ring_params['inner_data'], os)
        if ring_params.get('outer_data') is not None:
            ring_scaled['outer_data'] = _scale_ring_modes(ring_params['outer_data'], os)
        rings_scaled.append(ring_scaled)

    # Process bodies: assign default ranges, apply parallax and the camera roll,
    # and scale centres and axes to the oversampled grid.  A roll rotates each
    # body's centre about the boresight and adds to its line-of-sight pose
    # (rotation_z), so a body moves and turns under the same pointing rotation
    # the stars do; the body NavModel predicts the unrolled geometry.
    bodies_with_ranges = []
    for body_number, body_params in enumerate(bodies_params):
        body_params_copy = dict(body_params)
        # Positional default name, applied once here so every name-keyed
        # consumer (render order, masks, inventory) sees the same identity
        # and two unnamed bodies cannot collide.
        body_params_copy.setdefault('name', f'SIM-BODY-{body_number + 1}')
        if 'range_km' in body_params_copy:
            range_km = float(body_params_copy['range_km'])
        else:
            range_km = float(body_number + 1)
        body_params_copy['range_km'] = range_km
        cv = float(body_params.get('center_v', det_center_v))
        cu = float(body_params.get('center_u', det_center_u))
        sdv, sdu = _spk_shift(range_km)
        cv += sdv
        cu += sdu
        if offset_rotation_deg != 0.0:
            rv = cv - det_center_v
            ru = cu - det_center_u
            cv = det_center_v + roll_cos * rv - roll_sin * ru
            cu = det_center_u + roll_sin * rv + roll_cos * ru
            body_params_copy['rotation_z'] = (
                float(body_params.get('rotation_z', 0.0)) + offset_rotation_deg
            )
        body_params_copy['center_v'] = cv * os
        body_params_copy['center_u'] = cu * os
        for axis_key in ('axis1', 'axis2', 'axis3'):
            axis_val = body_params.get(axis_key)
            if axis_val is not None:
                body_params_copy[axis_key] = float(axis_val) * os
        bodies_with_ranges.append(body_params_copy)

    # Combine rings and bodies, sort by range (far to near).  A ring that
    # carries a physical 'range_km' (spk_error scenes require one on every
    # ring) sorts by it against the bodies' physical 'range_km', so mixed
    # scenes order physically.  A ring without one falls back to the
    # hint-unit 'range' key; that comparison is meaningful only because ring
    # defaults start at 1000 (dies with the ring-system rework).
    render_items: list[tuple[float, str, Any, int]] = []
    for idx, ring_scaled in enumerate(rings_scaled):
        ring_range_km = ring_scaled.get('range_km')
        sort_range = float(ring_range_km) if ring_range_km is not None else ring_scaled['range']
        render_items.append((sort_range, 'ring', ring_scaled, idx))
    for idx, body_params in enumerate(bodies_with_ranges):
        render_items.append((body_params['range_km'], 'body', body_params, idx))

    # Sort all items by range (far to near)
    render_items.sort(key=lambda x: x[0], reverse=True)

    # Render in range order (far to near)
    time = float(params.get('time', 0.0))
    epoch = float(params.get('ring_epoch', 0.0))
    shade_solid = bool(params.get('shade_solid_rings', False))
    # Track ring masks in original order for click detection
    ring_mask_map: dict[int, NDArrayBoolType] = {}

    # Track body data for final metadata
    body_models_dict: dict[str, dict[str, Any]] = {}
    # Store body masks by original index, not render order
    body_mask_map_by_idx: dict[int, NDArrayBoolType] = {}
    body_mask_map_dict: dict[str, NDArrayBoolType] = {}
    inventory_dict: dict[str, dict[str, float]] = {}
    body_index_map: NDArrayIntType = np.zeros((size_v, size_u), dtype=np.int32)

    ref_center_v = size_v / 2.0
    ref_center_u = size_u / 2.0

    # Build order_near_to_far for bodies (needed for body_index_map)
    sorted_bodies_by_range = sorted(bodies_with_ranges, key=lambda x: x['range_km'])
    order_near_to_far = [str(bp['name']).upper() for bp in sorted_bodies_by_range]

    for _range_val, item_type, item_params, orig_idx in render_items:
        if item_type == 'ring':
            ring_mask_map[orig_idx] = composite_ring(
                img,
                item_params,
                offset_v,
                offset_u=offset_u,
                time=time,
                epoch=epoch,
                shade_solid=shade_solid,
                resolution=float(os),
            )
        elif item_type == 'body':
            # Render single body
            body_mask, body_info = render_single_body(
                img,
                item_params,
                offset_v,
                offset_u=offset_u,
                seed=crater_seed,
                body_index=orig_idx,
                ref_center_v=ref_center_v,
                ref_center_u=ref_center_u,
                oversample=os,
            )
            # Store mask by original index for proper ordering
            body_mask_map_by_idx[orig_idx] = body_mask
            body_mask_map_dict[body_info['name']] = body_mask
            body_models_dict[body_info['name']] = body_info['params']
            inventory_dict[body_info['name']] = body_info['inventory']
            # Index into near-to-far order is 1-based
            near_index = order_near_to_far.index(body_info['name']) + 1
            body_index_map[body_mask] = near_index

    # Build body_masks in original order (matching bodies_params)
    body_masks: list[NDArrayBoolType] = []
    for idx in range(len(bodies_with_ranges)):
        if idx in body_mask_map_by_idx:
            body_masks.append(body_mask_map_by_idx[idx])
        else:
            # Should not happen, but create empty mask if missing
            body_masks.append(np.zeros((size_v, size_u), dtype=np.bool_))

    # Build ring_masks in original order for click detection
    ring_masks: list[NDArrayBoolType] = []
    for idx in range(len(rings_params)):
        if idx in ring_mask_map:
            ring_masks.append(ring_mask_map[idx])
        else:
            # Should not happen, but create empty mask if missing
            ring_masks.append(np.zeros((size_v, size_u), dtype=np.bool_))

    # Differential smear blurs each object class by its own motion, so it needs
    # the per-class radiance in isolation.  Capture the star, body, and ring
    # layers only when the scene asks for it, rendered with the same scaled
    # geometry and z-order as the composite so occlusion is consistent.  The star
    # layer lives in the point-source (electron / DN) domain like the composite
    # ``point_e``; the body and ring layers are intensive-signal layers.
    if _optics_needs_layers(params):
        stars_layer = np.zeros((size_v, size_u), dtype=np.float64)
        _render_sky(stars_layer, params, os=os, seed=background_stars_seed)
        # The layer must be pixel-identical to the primary deposit above (same
        # scatter sigma and seed included): differential smear REPLACES the
        # point-source plane with this layer, so any argument omitted here
        # silently undoes the corresponding effect for smeared-star scenes.
        render_stars(
            stars_layer,
            stars_params_scaled,
            offset_v,
            offset_u=offset_u,
            zero_point=zero_point,
            exposure_sec=exposure_sec,
            rendered_sigma=rendered_sigma,
            rotation_deg=offset_rotation_deg,
            oversample=os,
            catalog_scatter_px=catalog_scatter_px,
            catalog_scatter_seed=catalog_scatter_seed,
        )
        bodies_layer = np.zeros((size_v, size_u), dtype=np.float64)
        rings_layer = np.zeros((size_v, size_u), dtype=np.float64)
        for _range_val, item_type, item_params, orig_idx in render_items:
            if item_type == 'ring':
                composite_ring(
                    rings_layer,
                    item_params,
                    offset_v,
                    offset_u=offset_u,
                    time=time,
                    epoch=epoch,
                    shade_solid=shade_solid,
                    resolution=float(os),
                )
            else:
                render_single_body(
                    bodies_layer,
                    item_params,
                    offset_v,
                    offset_u=offset_u,
                    seed=crater_seed,
                    body_index=orig_idx,
                    ref_center_v=ref_center_v,
                    ref_center_u=ref_center_u,
                    oversample=os,
                )
        frame.truth['radiance_layers'] = {
            'stars': stars_layer,
            'bodies': bodies_layer,
            'rings': rings_layer,
        }

    frame.truth.update(
        {
            'stars': sim_star_list,
            'bodies': body_models_dict,
            'rings': rings_params,
            'inventory': inventory_dict,
            'star_info': star_info,
            'body_masks': body_masks,
            'ring_masks': ring_masks,
            'order_near_to_far': order_near_to_far,
            'body_index_map': body_index_map,
            'body_mask_map': body_mask_map_dict,
        }
    )


def _render_sky(
    plane: NDArrayFloatType,
    params: Mapping[str, Any],
    *,
    os: int,
    seed: int,
) -> None:
    """Render the background-sky star field into a point-source plane in place.

    Reads the scene ``sky_counts`` block (absent = no sky).  The faint cutoff is
    derived from the camera's nominal read noise and the scene PSF core, so the
    count integral stops where a star drops below the sky background.

    Parameters:
        plane: The oversampled point-source plane to deposit into.
        params: The full scene mapping.
        os: The oversampling factor.
        seed: The sky realization sub-seed.
    """
    sky = params.get('sky_counts')
    if not isinstance(sky, Mapping):
        return
    instrument = params.get('instrument')
    zero_point, domain = resolve_star_flux_zero_point(instrument)
    exposure_sec = float(params.get('exposure_sec', 1.0))
    inst_config = resolve_sim_inst_config(
        DEFAULT_CONFIG, instrument, params.get('instrument_config')
    )
    scene_psf = effective_psf(params)
    if scene_psf is not None:
        sigma_v_psf = float(scene_psf['sigma_v'])
        rendered_sigma_det = max(sigma_v_psf, float(scene_psf.get('sigma_u', sigma_v_psf)))
    else:
        rendered_sigma_det = float(inst_config['star_psf_sigma'])
    read_noise = _nominal_read_noise(instrument, domain)
    cutoff = faint_sky_cutoff_mag(
        zero_point=zero_point,
        exposure_sec=exposure_sec,
        read_noise=read_noise,
        psf_sigma=rendered_sigma_det,
    )
    render_sky_counts(
        plane,
        seed=seed,
        a=float(sky.get('a', -3.1)),
        b=float(sky.get('b', 0.34)),
        density_factor=float(sky.get('density_factor', 1.0)),
        pixel_scale_arcsec=resolve_sky_pixel_scale_arcsec(instrument),
        faint_cutoff_mag=cutoff,
        zero_point=zero_point,
        exposure_sec=exposure_sec,
        diffuse_flux_per_px=float(sky.get('diffuse_e_per_px', 0.0)),
        oversample=os,
    )


def _optics_needs_layers(params: Mapping[str, Any]) -> bool:
    """True when the scene's optics require per-class radiance layers.

    Differential smear (a smear entry addressing a single object class rather
    than the whole scene) is the only effect that needs the classes separated.

    Parameters:
        params: The full scene mapping.

    Returns:
        Whether the radiance stage should capture per-class layers.
    """
    optics = params.get('optics')
    if not isinstance(optics, dict):
        return False
    smear = optics.get('smear') or []
    return any(entry.get('object_class', 'all') != 'all' for entry in smear)


def _scale_star_params(star_params: dict[str, Any], os: int) -> dict[str, Any]:
    """Scale a star's pixel-space fields to the oversampled render grid.

    Catalog position, per-star PSF width, smear vector, PSF fitting-window size
    (the record builder's default is materialized so it scales like an explicit
    entry), the planted catalog-error displacement, and the companion separation
    are all pixel-space, so they scale with the oversampling factor.  At
    ``os == 1`` every scaled value equals its input value.

    Parameters:
        star_params: One scene star entry.
        os: The oversampling factor.

    Returns:
        A scaled copy of the star entry.
    """
    scaled = dict(star_params)
    for key in ('v', 'u', 'psf_sigma', 'move_v', 'move_u', 'catalog_error_v', 'catalog_error_u'):
        if star_params.get(key) is not None:
            scaled[key] = float(star_params[key]) * os
    # The record builder's default window is materialized here so a defaulted
    # entry scales exactly like an explicit one: the downsample stage divides
    # every record's psf_size by os, which would otherwise shrink a defaulted
    # window to (11 // os, 11 // os) detector pixels.
    psf_size = star_params.get('psf_size', DEFAULT_PSF_SIZE)
    scaled['psf_size'] = [int(psf_size[0]) * os, int(psf_size[1]) * os]
    companion = star_params.get('companion')
    if isinstance(companion, dict) and companion.get('sep_px') is not None:
        scaled_companion = dict(companion)
        scaled_companion['sep_px'] = float(companion['sep_px']) * os
        scaled['companion'] = scaled_companion
    return scaled


def _scale_ring_modes(modes: Any, os: int) -> list[dict[str, Any]]:
    """Scale each ring-edge mode's radius amplitudes to the oversampled grid.

    The semi-major-axis ``a`` and the eccentric radius ``ae`` are pixel-space
    radii; the longitude and precession-rate fields are angular and unchanged.

    Parameters:
        modes: The ``inner_data`` / ``outer_data`` list of mode mappings.
        os: The oversampling factor.

    Returns:
        A list of scaled copies of the mode mappings.
    """
    scaled_modes: list[dict[str, Any]] = []
    for mode in modes:
        scaled_mode = dict(mode)
        for key in ('a', 'ae'):
            if mode.get(key) is not None:
                scaled_mode[key] = float(mode[key]) * os
        scaled_modes.append(scaled_mode)
    return scaled_modes

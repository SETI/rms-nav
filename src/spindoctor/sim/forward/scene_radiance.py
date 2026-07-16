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
through the same flux/point-mass path.

Occlusion: bodies and the legacy ``rings`` paint far to near by ``range_km``
(overlaps without explicit ranges are a scene error).  The optical-depth
``ring_system`` composites over that stack as a transmission screen, per pixel
and depth-ordered against the bodies (``img = I_ring + exp(-tau/mu) *
img_behind``).  Point sources sit at infinity: the ring screen attenuates
them and an opaque body's silhouette extinguishes them (the painted, lit
silhouette -- a fully dark night side does not occult, a stated
approximation of the mask-based body renderers).
"""

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from scipy import ndimage

from spindoctor.config import DEFAULT_CONFIG
from spindoctor.sim.forward.artifacts_catalog import (
    resolve_detector_defaults,
    resolve_sky_pixel_scale_arcsec,
    resolve_star_flux_zero_point,
)
from spindoctor.sim.forward.body import render_single_body
from spindoctor.sim.forward.optics import effective_psf
from spindoctor.sim.forward.ring import composite_ring
from spindoctor.sim.forward.ring_system import RingSystemMaps, render_ring_system
from spindoctor.sim.forward.stages import SimFrame
from spindoctor.sim.forward.star import faint_sky_cutoff_mag, render_sky_counts, render_stars
from spindoctor.sim.instruments import resolve_sim_inst_config
from spindoctor.sim.scene_schema import SimSceneValidationError
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
            ``bodies``, ``rings``, ``ring_features``, ``inventory``,
            ``star_info``, ``body_masks``, ``ring_masks``,
            ``order_near_to_far``, ``body_index_map``, ``body_mask_map``,
            ``body_occlusion``).
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
    # Sub-effect stream names follow the established 'scene_radiance/<effect>'
    # convention and are frozen once committed scenes draw from them: renaming
    # one (including 'scene_radiance/background_stars') would reseed its stream
    # and change every scene that uses it.  Ring-system randomized effects
    # (spokes, moonlet fields -- later phases) will derive
    # 'scene_radiance/ring_system/<effect>' streams here; the phase-F ring
    # system is fully deterministic and draws none yet.
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

    # Process rings: apply parallax and scale every pixel-space quantity to
    # the oversampled grid.  The original scene entries keep their
    # detector-space values for the truth metadata.
    rings_scaled: list[dict[str, Any]] = []
    for ring_params in rings_params:
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

    # Combine rings and bodies and order far to near.  The ordering key is
    # the physical 'range_km' -- there is no hint-unit fallback.  A ring
    # without one has no depth and sorts as farthest (behind everything);
    # a body without one keeps the positional default (index + 1) that the
    # truth metadata contract materializes.  Either default is only a
    # deterministic placement, never a depth claim: the paint loop below
    # raises when objects without an explicit range_km actually overlap,
    # so an ambiguous stack is a scene-authoring error, not a guess.  The
    # descending (depth, is-ring, index) key renders unranked rings in
    # reverse list order behind the bodies, matching the retired hint
    # defaults exactly.
    render_items: list[tuple[tuple[float, int, int], str, Any, int, bool]] = []
    for idx, ring_scaled in enumerate(rings_scaled):
        ring_range_km = ring_scaled.get('range_km')
        depth = float(ring_range_km) if ring_range_km is not None else float('inf')
        render_items.append(((depth, 1, idx), 'ring', ring_scaled, idx, ring_range_km is not None))
    for idx, (body_params, body_scaled) in enumerate(
        zip(bodies_params, bodies_with_ranges, strict=True)
    ):
        explicit = body_params.get('range_km') is not None
        render_items.append(
            ((float(body_scaled['range_km']), 0, idx), 'body', body_scaled, idx, explicit)
        )

    # Sort all items far to near.
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
    # Bodies in render (far-to-near) order, for the mutual-event occlusion truth.
    rendered_bodies: list[tuple[str, NDArrayBoolType]] = []

    ref_center_v = size_v / 2.0
    ref_center_u = size_u / 2.0

    # Build order_near_to_far for bodies (needed for body_index_map)
    sorted_bodies_by_range = sorted(bodies_with_ranges, key=lambda x: x['range_km'])
    order_near_to_far = [str(bp['name']).upper() for bp in sorted_bodies_by_range]

    # Painted-so-far tracking for the depth-ambiguity check: overlapping
    # objects must ALL carry an explicit scene range_km, or their stacking
    # would be a silent guess.
    painted_items: list[tuple[str, NDArrayBoolType, bool, str]] = []
    # Per-pixel observer distance of the nearest painted body, for the ring
    # system's per-pixel compositing; built only when the scene has one.
    ring_system_params = params.get('ring_system')
    has_ring_system = isinstance(ring_system_params, Mapping)
    body_depth_map: NDArrayFloatType | None = (
        np.full((size_v, size_u), np.inf, dtype=np.float64) if has_ring_system else None
    )

    for sort_key, item_type, item_params, orig_idx, explicit_depth in render_items:
        item_mask: NDArrayBoolType
        if item_type == 'ring':
            item_mask = composite_ring(
                img,
                item_params,
                offset_v,
                offset_u=offset_u,
                time=time,
                epoch=epoch,
                shade_solid=shade_solid,
                resolution=float(os),
            )
            ring_mask_map[orig_idx] = item_mask
            item_label = str(item_params.get('name', f'rings[{orig_idx}]'))
        else:
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
            rendered_bodies.append((body_info['name'], body_mask))
            # Index into near-to-far order is 1-based
            near_index = order_near_to_far.index(body_info['name']) + 1
            body_index_map[body_mask] = near_index
            if body_depth_map is not None:
                # Painting runs far to near, so the last writer per pixel is
                # the nearest body.
                body_depth_map[body_mask] = sort_key[0]
            item_mask = body_mask
            item_label = body_info['name']
        _check_depth_ambiguity(painted_items, item_label, item_mask, explicit_depth)
        painted_items.append((item_label, item_mask, explicit_depth, item_type))

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

    # The optical-depth ring system composites as a transmission screen over
    # the painted stack, per pixel and depth-ordered against the bodies: the
    # far arm passes behind the planet disc, the near arm in front, and
    # everything behind a ring pixel shows through scaled by exp(-tau/mu).
    ring_maps: RingSystemMaps | None = None
    ring_apply: NDArrayBoolType | None = None
    if has_ring_system:
        assert isinstance(ring_system_params, Mapping)
        assert body_depth_map is not None
        ring_maps = _resolve_ring_system_maps(
            ring_system_params,
            (size_v, size_u),
            det_center_v=det_center_v,
            det_center_u=det_center_u,
            offset_v=offset_v,
            offset_u=offset_u,
            roll_deg=offset_rotation_deg,
            spk_shift=_spk_shift,
            time=time,
            epoch=epoch,
            os=os,
        )
        _check_ring_system_ambiguity(
            painted_items,
            ring_maps.mask,
            ring_explicit=ring_system_params.get('range_km') is not None,
        )
        if ring_maps.depth_km is None:
            # No depth relation exists; the ambiguity check above proved the
            # system overlaps nothing painted, so it screens only the sky.
            ring_apply = ring_maps.mask
        else:
            ring_apply = ring_maps.mask & (ring_maps.depth_km < body_depth_map)
        img[ring_apply] = (
            ring_maps.intensity[ring_apply] + ring_maps.transmission[ring_apply] * img[ring_apply]
        )

    # Point sources sit at infinity: the ring screen attenuates them wherever
    # it carries optical depth (a body in front of the ring zeroes them below
    # anyway), and an opaque body extinguishes them entirely.  The legacy
    # painted 'rings' deliberately do not occlude stars: that path keeps its
    # exact historical output until the ring_system conversion retires it.
    body_occlusion_mask: NDArrayBoolType | None = None
    if body_masks and bool(np.any(point_e)):
        body_occlusion_mask = np.zeros((size_v, size_u), dtype=np.bool_)
        for mask in body_masks:
            body_occlusion_mask |= mask
        if not body_occlusion_mask.any():
            body_occlusion_mask = None
    if ring_maps is not None and ring_maps.mask.any():
        point_e *= ring_maps.transmission
    if body_occlusion_mask is not None:
        point_e[body_occlusion_mask] = 0.0

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
        # The star layer carries the same occlusion the primary point-source
        # plane received (ring transmission, then opaque-body extinction).
        if ring_maps is not None and ring_maps.mask.any():
            stars_layer *= ring_maps.transmission
        if body_occlusion_mask is not None:
            stars_layer[body_occlusion_mask] = 0.0
        bodies_layer = np.zeros((size_v, size_u), dtype=np.float64)
        rings_layer = np.zeros((size_v, size_u), dtype=np.float64)
        for _sort_key, item_type, item_params, orig_idx, _explicit in render_items:
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
        # The ring system's emission belongs to the rings class and its screen
        # dims the bodies behind it, so the per-class sum reproduces the
        # composite exactly (a legacy ring under the screen is a scene error,
        # checked above, so the += cannot double-paint).
        if ring_maps is not None and ring_apply is not None:
            bodies_layer[ring_apply] *= ring_maps.transmission[ring_apply]
            rings_layer[ring_apply] += ring_maps.intensity[ring_apply]
        frame.truth['radiance_layers'] = {
            'stars': stars_layer,
            'bodies': bodies_layer,
            'rings': rings_layer,
        }

    # Ring-system scene truth: the per-feature applied orbit error (the
    # planted ephemeris misplacement the navigator must absorb) alongside
    # each feature's identity and navigability, for the recovery harnesses.
    ring_feature_truth: list[dict[str, Any]] = []
    if has_ring_system:
        assert isinstance(ring_system_params, Mapping)
        for feature in ring_system_params.get('features') or []:
            ring_feature_truth.append(
                {
                    'name': feature.get('name'),
                    'kind': feature.get('kind'),
                    'navigable': bool(feature.get('navigable', False)),
                    'orbit_error': dict(feature.get('orbit_error') or {}),
                }
            )

    frame.truth.update(
        {
            'stars': sim_star_list,
            'bodies': body_models_dict,
            'rings': rings_params,
            'ring_features': ring_feature_truth,
            'inventory': inventory_dict,
            'star_info': star_info,
            'body_masks': body_masks,
            'ring_masks': ring_masks,
            'order_near_to_far': order_near_to_far,
            'body_index_map': body_index_map,
            'body_mask_map': body_mask_map_dict,
            'body_occlusion': _body_occlusion_truth(rendered_bodies),
        }
    )


def _check_depth_ambiguity(
    painted_items: list[tuple[str, NDArrayBoolType, bool, str]],
    item_label: str,
    item_mask: NDArrayBoolType,
    explicit_depth: bool,
) -> None:
    """Fail when overlapping objects are stacked without explicit depths.

    ``range_km`` is the compositing depth.  Two objects whose painted pixels
    overlap must both carry an explicit scene ``range_km``, or their stacking
    order would be a silent guess; a scene that leaves it off an overlapping
    object is malformed and fails loudly here.  Non-overlapping objects need
    no ranges: their paint order is unobservable.

    Parameters:
        painted_items: ``(label, painted mask, explicit range_km?, type)`` for
            every object already painted, in render order.
        item_label: Name of the object just painted.
        item_mask: Pixels the object just painted.
        explicit_depth: Whether the object carries an explicit ``range_km``.

    Raises:
        SimSceneValidationError: If the new object overlaps a painted one and
            either of the pair lacks an explicit ``range_km``.
    """
    if not item_mask.any():
        return
    for other_label, other_mask, other_explicit, _other_type in painted_items:
        if explicit_depth and other_explicit:
            continue
        if np.any(item_mask & other_mask):
            raise SimSceneValidationError(
                f'objects {other_label!r} and {item_label!r} overlap but do not both '
                f'carry an explicit range_km; set range_km on both to order them'
            )


def _check_ring_system_ambiguity(
    painted_items: list[tuple[str, NDArrayBoolType, bool, str]],
    ring_mask: NDArrayBoolType,
    *,
    ring_explicit: bool,
) -> None:
    """Fail when the ring system overlaps painted objects without depths.

    Per-pixel depth ordering against a body needs an explicit ``range_km`` on
    both the ring system and the body.  A legacy painted ring has no
    per-pixel depth at all, so overlapping one is always a scene error (the
    conversion to a ``ring_system`` feature is the fix).

    Parameters:
        painted_items: ``(label, painted mask, explicit range_km?, type)`` for
            every painted body and legacy ring.
        ring_mask: Pixels where the ring system carries optical depth.
        ring_explicit: Whether the ring system carries an explicit
            ``range_km``.

    Raises:
        SimSceneValidationError: On any overlap without a defined depth
            relation.
    """
    if not ring_mask.any():
        return
    for label, mask, explicit, item_type in painted_items:
        if not np.any(mask & ring_mask):
            continue
        if item_type == 'ring':
            raise SimSceneValidationError(
                f'ring_system overlaps legacy ring {label!r}, which has no per-pixel '
                f'depth; convert it to a ring_system feature or separate them'
            )
        if not (ring_explicit and explicit):
            raise SimSceneValidationError(
                f'ring_system and body {label!r} overlap but do not both carry an '
                f'explicit range_km; set range_km on both to order them'
            )


def _resolve_ring_system_maps(
    ring_system: Mapping[str, Any],
    shape: tuple[int, int],
    *,
    det_center_v: float,
    det_center_u: float,
    offset_v: float,
    offset_u: float,
    roll_deg: float,
    spk_shift: Callable[[float], tuple[float, float]],
    time: float,
    epoch: float,
    os: int,
) -> RingSystemMaps:
    """Place the ring system on the sky and render its per-pixel maps.

    Resolves the block's sky placement the way the body loop does: the
    spacecraft-ephemeris parallax displaces the center by ``1/range_km``, a
    camera roll rotates the center about the boresight and adds to the node
    angle (rolling the whole projected pattern), and the planted pointing
    offset translates it on the oversampled grid.

    Parameters:
        ring_system: The validated scene ``ring_system`` mapping.
        shape: The oversampled render-grid shape.
        det_center_v: Detector-frame center v (the roll pivot and default
            ring center), in detector pixels.
        det_center_u: Detector-frame center u, in detector pixels.
        offset_v: Planted pointing offset v on the oversampled grid.
        offset_u: Planted pointing offset u on the oversampled grid.
        roll_deg: Planted camera-roll angle in degrees.
        spk_shift: The scene's parallax function (detector pixels per range).
        time: Scene time in TDB seconds.
        epoch: Ring epoch in TDB seconds.
        os: The oversampling factor.

    Returns:
        The rendered :class:`RingSystemMaps`.
    """
    geometry = ring_system.get('geometry') or {}
    center_v = float(geometry.get('center_v', det_center_v))
    center_u = float(geometry.get('center_u', det_center_u))
    node_deg = float(geometry.get('node_deg', 0.0))
    range_km = ring_system.get('range_km')
    if range_km is not None:
        shift_v, shift_u = spk_shift(float(range_km))
        center_v += shift_v
        center_u += shift_u
    if roll_deg != 0.0:
        roll_cos = float(np.cos(np.radians(roll_deg)))
        roll_sin = float(np.sin(np.radians(roll_deg)))
        rel_v = center_v - det_center_v
        rel_u = center_u - det_center_u
        center_v = det_center_v + roll_cos * rel_v - roll_sin * rel_u
        center_u = det_center_u + roll_sin * rel_v + roll_cos * rel_u
        node_deg += roll_deg
    return render_ring_system(
        shape,
        ring_system,
        center_v=center_v * os + offset_v,
        center_u=center_u * os + offset_u,
        node_deg=node_deg,
        time=time,
        epoch=epoch,
        oversample=os,
    )


def _body_occlusion_truth(
    rendered_bodies: list[tuple[str, NDArrayBoolType]],
) -> dict[str, dict[str, float]]:
    """Per-body mutual-event truth measured from the rendered silhouettes.

    For each body, against the union of every NEARER body's silhouette
    (bodies rendered after it in the far-to-near compositing order):

    - ``visible_fraction``: visible pixels / unoccluded silhouette pixels.
    - ``occluded_limb_arc_deg``: the arc of the body's limb hidden by the
      occluders, measured as the occluded fraction of its silhouette-boundary
      pixels times 360.  Boundary pixels of a rasterized silhouette are
      uniform per unit arc length, so for the (near-)circular discs the
      mutual-event scenes plant this is the hidden limb arc; strongly
      elongated silhouettes weight it by arc length rather than angle.

    Ring occlusion is not counted: mutual events are body-body geometry, and
    a body's silhouette clipped by the frame edge counts its on-frame
    boundary only.

    Parameters:
        rendered_bodies: ``(name, unoccluded silhouette mask)`` per body in
            render (far-to-near) order.

    Returns:
        ``{body_name: {'visible_fraction': ..., 'occluded_limb_arc_deg': ...}}``.
    """
    occlusion: dict[str, dict[str, float]] = {}
    if not rendered_bodies:
        return occlusion
    # Suffix unions: nearer_union[i] = union of masks rendered after body i.
    nearer_union = np.zeros_like(rendered_bodies[0][1])
    nearer_by_body: list[NDArrayBoolType] = []
    for _name, mask in reversed(rendered_bodies):
        nearer_by_body.append(nearer_union.copy())
        nearer_union = nearer_union | mask
    nearer_by_body.reverse()

    for (name, mask), nearer in zip(rendered_bodies, nearer_by_body, strict=True):
        unoccluded = int(np.count_nonzero(mask))
        if unoccluded == 0:
            occlusion[name] = {'visible_fraction': 1.0, 'occluded_limb_arc_deg': 0.0}
            continue
        visible = int(np.count_nonzero(mask & ~nearer))
        boundary = mask & ~ndimage.binary_erosion(mask)
        boundary_total = int(np.count_nonzero(boundary))
        boundary_occluded = int(np.count_nonzero(boundary & nearer))
        arc_deg = 360.0 * boundary_occluded / boundary_total if boundary_total else 0.0
        occlusion[name] = {
            'visible_fraction': visible / unoccluded,
            'occluded_limb_arc_deg': arc_deg,
        }
    return occlusion


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

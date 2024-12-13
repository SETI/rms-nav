import logging

import time

import numpy as np
import numpy.ma as ma
import scipy.ndimage.filters as filt
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

import oops
import polymath

import nav.config
from nav.image import (filter_downsample,
                       image_zoom,
                       shift_image)


_LOGGING_NAME = 'cb.' + __name__


# Sometimes the bounding box returned by "inventory" is not quite big enough
BODIES_POSITION_SLOP_FRAC = 0.05


def bodies_create_model(obs, body_name, inventory,
                        cartographic_data={},
                        always_create_model=False,
                        create_overlay=False,
                        label_avoid_mask=None,
                        bodies_config=None,
                        no_model=False):
    """Create a model for a body.

    Inputs:
        obs                    The Observation.
        body_name              The name of the moon.
        inventory              The entry returned from the inventory() method of
                               an Observation for this body. Used to find the
                               clipping rectangle.
        cartographic_data      A dictionary of body names containing
                               cartographic data in lat/lon format. Each entry
                               contains metadata in the format returned by
                               bodies_mosaic_init.
        always_create_model    True to always return a model even if the
                               body is too small or the curvature or limb is
                               bad.
        create_overlay         True to create the text overlay.
        label_avoid_mask       A mask giving places where text labels should
                               not be placed (i.e. labels from another
                               model are already there). None if no mask.
        bodies_config          Configuration parameters.
        no_model               Create a metadata structure but don't make an
                               actual model.

    Returns:
        model, metadata, text

        metadata is a dictionary containing

        'body_name'            The name of the body.
        'size_ok'              True if the body is large enough to bother with.
        'ring_emission_ok'     True if the ring emission angle at the body's
                               location is large enough that the body will be
                               visible and not hidden by the rings. Only
                               relevant for bodies located inside the rings;
                               always True otherwise.
        'inventory'            The inventory entry for this body.
        'cartographic_data_source'
                               The path of the mosaic used to provide the
                               cartographic data. None if no data provided.
        'curvature_ok'         True if sufficient curvature is visible to permit
                               correlation.
        'limb_ok'              True if the limb is sufficiently sharp to permit
                               correlation.
        'entirely_visible'     True if the body is entirely visible even if
                               shifted to the maximum extent specified by
                               extfov_margin. This is based purely on geometry,
                               not looking at whether other objects occult it.
        'occulted_by'          A list of body names or 'RINGS' that occult some
                               or all of this body. This is set in the main
                               offset loop, not in this procedure. None if
                               nothing occults it or it hasn't been processed
                               by the main loop yet.
        'in_saturn_shadow'     True if the body is in Saturn's shadow and only
                               illuminated by Saturn-shine.
        'sub_solar_lon'        The sub-solar longitude (IAU, West).
        'sub_solar_lat'        The sub-solar latitude.
        'sub_observer_lon'     The sub-observer longitude (IAU, West).
        'sub_observer_lat'     The sub-observer latitude.
        'phase_angle'          The phase angle at the body's center.
        'body_blur'            The amount of model blur required to properly
                               correlate low-resolution cartographic data or
                               to not look bumpy when doing a ellipsoidal model.
        'image_blur'           The amount of image blur required to properly
                               correlate low-resolution cartographic data.
        'nav_uncertainty'      The amount of uncertainty (in pixels) to add to
                               the final model uncertainty by due to the
                               uncertainty of the navigation of the source
                               images in the cartographic data.
        'confidence'           The confidence in how well this model will
                               do in correlation.

        'start_time'           The time (s) when bodies_create_model was called.
        'end_time'             The time (s) when bodies_create_model returned.

            These are used for bootstrapping:

        'reproj'               The reprojection data structure at the
                               resolution specified in the config. Filled in
                               after the offset pass is finished. None if
                               not appropriate for bootstrapping or
                               it hasn't been processed by the main loop yet.
    """

    start_time = time.time()

    logger = logging.getLogger(_LOGGING_NAME+'.bodies_create_model')

    if bodies_config is None:
        bodies_config = nav.config.BODIES_DEFAULT_CONFIG

    body_name = body_name.upper()

    metadata = {}
    metadata['body_name'] = body_name
    metadata['inventory'] = inventory
    metadata['cartographic_data_source'] = None
    if cartographic_data and body_name in cartographic_data:
        metadata['cartographic_data_source'] = (
                             cartographic_data[body_name]['full_path'])
    metadata['size_ok'] = None
    metadata['curvature_ok'] = None
    metadata['limb_ok'] = None
    metadata['entirely_visible'] = None
    metadata['occulted_by'] = None
    metadata['in_saturn_shadow'] = None
    metadata['image_blur'] = None
    metadata['body_blur'] = None
    metadata['nav_uncertainty'] = None
    metadata['reproj'] = None
    metadata['has_bad_pixels'] = None
    metadata['start_time'] = start_time
    metadata['end_time'] = None
    metadata['confidence'] = None

    logger.info('*** Modeling %s ***', body_name)

    metadata['sub_solar_lon'] = obs.ext_bp.sub_solar_longitude(
                           body_name,
                           direction=bodies_config['reproj_lon_direction']).vals
    metadata['sub_solar_lat'] = obs.ext_bp.sub_solar_latitude(body_name).vals
    metadata['sub_observer_lon'] = obs.ext_bp.sub_observer_longitude(
                           body_name,
                           direction=bodies_config['reproj_lon_direction']).vals
    metadata['sub_observer_lat'] = obs.ext_bp.sub_observer_latitude(
                                                                body_name).vals
    metadata['phase_angle'] = obs.ext_bp.center_phase_angle(body_name).vals

    logger.info('Sub-solar longitude    %6.2f', metadata['sub_solar_lon']*oops.DPR)
    logger.info('Sub-solar latitude     %6.2f', metadata['sub_solar_lat']*oops.DPR)
    logger.info('Sub-observer longitude %6.2f',
                metadata['sub_observer_lon']*oops.DPR)
    logger.info('Sub-observer latitude  %6.2f',
                metadata['sub_observer_lat']*oops.DPR)
    logger.info('Phase angle            %6.2f', metadata['phase_angle']*oops.DPR)

    # Create a Meshgrid that only covers the center pixel of the body
    ctr_uv = inventory['center_uv']
    ctr_meshgrid = oops.Meshgrid.for_fov(obs.fov,
                                         origin=(ctr_uv[0]+.5, ctr_uv[1]+.5),
                                         limit =(ctr_uv[0]+.5, ctr_uv[1]+.5),
                                         swap  =True)
    ctr_bp = oops.backplane.Backplane(obs, meshgrid=ctr_meshgrid)
    if body_name == 'SATURN':
        metadata['in_saturn_shadow'] = False
    else:
        # We're just going to assume if part of the body is shadowed, the whole
        # thing is
        saturn_shadow = ctr_bp.where_inside_shadow(body_name, 'saturn').vals
        if np.any(saturn_shadow):
            logger.info('Body is in Saturn\'s shadow')
            metadata['in_saturn_shadow'] = True
        else:
            metadata['in_saturn_shadow'] = False

    logger.info('Pixel size %.2f x %.2f',
                inventory['u_pixel_size'], inventory['v_pixel_size'])
    bb_area = inventory['u_pixel_size'] * inventory['v_pixel_size']
    if bb_area >= bodies_config['min_bounding_box_area']:
        metadata['size_ok'] = True
    else:
        metadata['size_ok'] = False
        logger.info(
            'Bounding box (area %.3f pixels) is too small to bother with',
            bb_area)
        if not always_create_model:
            metadata['end_time'] = time.time()
            return None, metadata, None

    if body_name in nav.config.RINGS_BODY_LIST:
        emission = obs.ext_bp.emission_angle('saturn:ring').mvals.astype('float32')
        min_emission = np.min(np.abs(emission-oops.HALFPI))
        if min_emission*oops.DPR < bodies_config['min_emission_ring_body']:
            logger.info('Minimum emission angle %.2f from 90 too close to ring '+
                        'plane - not using', min_emission*oops.DPR)
            metadata['ring_emission_ok'] = False
            metadata['end_time'] = time.time()
            return None, metadata, None

    metadata['ring_emission_ok'] = True

    u_min = int(inventory['u_min_unclipped'])
    u_max = int(inventory['u_max_unclipped'])
    v_min = int(inventory['v_min_unclipped'])
    v_max = int(inventory['v_max_unclipped'])

    logger.debug('Original bounding box U %d to %d V %d to %d',
                 u_min, u_max, v_min, v_max)

    entirely_visible = False
    if (u_min >= obs.extfov_margin[0] and
        u_max <= obs.data_shape_xy[0]-1-obs.extfov_margin[0] and
        v_min >= obs.extfov_margin[1] and
        v_max <= obs.data_shape_xy[1]-1-obs.extfov_margin[1]):
        # Body is entirely visible - no part is off the edge even when shifting
        # the extended FOV
        entirely_visible = True
    metadata['entirely_visible'] = entirely_visible

    # For curvature later
    u_center = int((u_min+u_max)/2)
    v_center = int((v_min+v_max)/2)
    width = u_max-u_min+1
    height = v_max-v_min+1
    curvature_threshold_frac = bodies_config['curvature_threshold_frac']
    curvature_threshold_pix = bodies_config['curvature_threshold_pixels']
    width_threshold = max(width * curvature_threshold_frac,
                          curvature_threshold_pix)
    height_threshold = max(height * curvature_threshold_frac,
                           curvature_threshold_pix)

    u_min -= int((u_max-u_min) * BODIES_POSITION_SLOP_FRAC)
    u_max += int((u_max-u_min) * BODIES_POSITION_SLOP_FRAC)
    v_min -= int((v_max-v_min) * BODIES_POSITION_SLOP_FRAC)
    v_max += int((v_max-v_min) * BODIES_POSITION_SLOP_FRAC)

    u_min = np.clip(u_min, obs.extfov_xy_min[0], obs.extfov_xy_max[0])
    u_max = np.clip(u_max, obs.extfov_xy_min[0], obs.extfov_xy_max[0])
    v_min = np.clip(v_min, obs.extfov_xy_min[1], obs.extfov_xy_max[1])
    v_max = np.clip(v_max, obs.extfov_xy_min[1], obs.extfov_xy_max[1])

    # Things break if the moon is only a single pixel wide or tall
    if u_min == u_max and u_min == obs.extfov_xy_max[0]:
        u_min -= 1
    if u_min == u_max and u_min == obs.extfov_xy_min[0]:
        u_max += 1
    if v_min == v_max and v_min == obs.extfov_xy_max[1]:
        v_min -= 1
    if v_min == v_max and v_min == obs.extfov_xy_min[1]:
        v_max += 1

    logger.debug('Image size %d %d subrect U %d to %d V %d to %d',
                 obs.data_shape_xy[0], obs.data_shape_xy[1],
                 u_min, u_max, v_min, v_max)
    if entirely_visible:
        logger.info('All of body is guaranteed visible')
    else:
        logger.info('Not all of body guaranteed to be visible')

    if no_model:
        metadata['end_time'] = time.time()
        return None, metadata, None

    # Make a new Backplane that only covers the body, but oversample
    # it so we can do anti-aliasing
    restr_oversample_u = max(int(np.floor(
              bodies_config['oversample_edge_limit'] /
              np.ceil(inventory['u_pixel_size']))), 1)
    restr_oversample_v = max(int(np.floor(
              bodies_config['oversample_edge_limit'] /
              np.ceil(inventory['v_pixel_size']))), 1)
    restr_oversample_u = min(restr_oversample_u,
                             bodies_config['oversample_maximum'])
    restr_oversample_v = min(restr_oversample_v,
                             bodies_config['oversample_maximum'])
    logger.debug('Oversampling by %d,%d', restr_oversample_u,
                 restr_oversample_v)
    restr_u_min = u_min + 1./(2*restr_oversample_u)
    restr_u_max = u_max + 1 - 1./(2*restr_oversample_u)
    restr_v_min = v_min + 1./(2*restr_oversample_v)
    restr_v_max = v_max + 1 - 1./(2*restr_oversample_v)
    if not hasattr(obs, 'restr_oversample_bp_dict'):
        obs.restr_oversample_bp_dict = {}
    restr_key = (restr_u_min, restr_v_min, restr_u_max, restr_v_max)
    if restr_key in obs.restr_oversample_bp_dict:
        restr_o_bp = obs.restr_oversample_bp_dict[restr_key]
    else:
        restr_o_meshgrid = oops.Meshgrid.for_fov(obs.fov,
                                                 origin=(restr_u_min, restr_v_min),
                                                 limit =(restr_u_max, restr_v_max),
                                                 oversample=(restr_oversample_u,
                                                             restr_oversample_v),
                                                 swap  =True)
        restr_o_bp = oops.backplane.Backplane(obs, meshgrid=restr_o_meshgrid)
        obs.restr_oversample_bp_dict[restr_key] = restr_o_bp
    restr_o_incidence_mvals = restr_o_bp.incidence_angle(body_name).mvals
    restr_incidence_mvals = filter_downsample(restr_o_incidence_mvals,
                                              restr_oversample_v,
                                              restr_oversample_u)
    restr_incidence = polymath.Scalar(restr_incidence_mvals)

    # Analyze the limb

    restr_body_mask_inv = ma.getmaskarray(restr_incidence_mvals)
    restr_body_mask = np.logical_not(restr_body_mask_inv)

    # If the inv mask is true, but any of its neighbors are false, then
    # this is an edge
    limb_mask = restr_body_mask
    limb_mask_1 = shift_image(restr_body_mask_inv, -1,  0)
    limb_mask_2 = shift_image(restr_body_mask_inv,  1,  0)
    limb_mask_3 = shift_image(restr_body_mask_inv,  0, -1)
    limb_mask_4 = shift_image(restr_body_mask_inv,  0,  1)
    limb_mask_total = np.logical_or(limb_mask_1, limb_mask_2)
    limb_mask_total = np.logical_or(limb_mask_total, limb_mask_3)
    limb_mask_total = np.logical_or(limb_mask_total, limb_mask_4)
    limb_mask = np.logical_and(limb_mask, limb_mask_total)

    masked_restr_incidence = restr_incidence.mask_where(
                                            np.logical_not(limb_mask))

    if not np.any(limb_mask):
        limb_incidence_min = 1e38
        limb_incidence_max = 1e38
        logger.info('No limb')
    else:
        limb_incidence_min = np.min(masked_restr_incidence.mvals)
        limb_incidence_max = np.max(masked_restr_incidence.mvals)
        logger.info('Limb incidence angle min %.2f max %.2f',
                     limb_incidence_min*oops.DPR, limb_incidence_max*oops.DPR)

        limb_threshold = bodies_config['limb_incidence_threshold']
        limb_frac = bodies_config['limb_incidence_frac']

        # Slightly more than half of the body must be visible in each of the
        # horizontal and vertical directions AND also have a good limb in
        # those quadrants

        curvature_ok = False
        limb_ok = False
        l_edge = obs.extfov_margin[0]
        r_edge = obs.data_shape_xy[0]-1-obs.extfov_margin[0]
        t_edge = obs.extfov_margin[1]
        b_edge = obs.data_shape_xy[1]-1-obs.extfov_margin[1]
        inc_r_edge = masked_restr_incidence.vals.shape[1]-1
        inc_b_edge = masked_restr_incidence.vals.shape[0]-1
        for l_moon, r_moon, lr_str in ((u_center-width_threshold,
                                        u_center+width/2, 'L'),
                                      (u_center-width/2,
                                       u_center+width_threshold, 'R')):
            for t_moon, b_moon, tb_str in ((v_center-height_threshold,
                                            v_center+height/2, 'T'),
                                           (v_center-height/2,
                                            v_center+height_threshold, 'B')):
                logger.debug('LMOON %d LEDGE %d / RMOON %d REDGE %d / '+
                             'TMOON %d TEDGE %d / BMOON %d BEDGE %d',
                             l_moon, l_edge, r_moon, r_edge,
                             t_moon, t_edge, b_moon, b_edge)
                if (l_moon >= l_edge and r_moon <= r_edge and
                    t_moon >= t_edge and b_moon <= b_edge):
                    l_moon_clip = int(np.clip(l_moon-u_min, 0, inc_r_edge))
                    r_moon_clip = int(np.clip(r_moon-u_min, 0, inc_r_edge))
                    t_moon_clip = int(np.clip(t_moon-v_min, 0, inc_b_edge))
                    b_moon_clip = int(np.clip(b_moon-v_min, 0, inc_b_edge))
                    sub_inc = masked_restr_incidence[
                                         t_moon_clip:b_moon_clip+1,
                                         l_moon_clip:r_moon_clip+1].mvals
                    ok_masked_incidence = sub_inc[np.where(sub_inc <
                                                           limb_threshold)]
                    logger.debug('%s %s %d %d', tb_str, lr_str,
                                 ok_masked_incidence.compressed().shape[0],
                                 sub_inc.compressed().shape[0])
                    inc_frac = (float(ok_masked_incidence.compressed().shape[0]) /
                                float(sub_inc.compressed().flatten().shape[0]))
                    if inc_frac >= limb_frac:
                        logger.debug('Curvature+limb quadrant %s%s OK',
                                     tb_str, lr_str)
                        curvature_ok = True

        if (not curvature_ok and
            obs.extfov_margin[0] < u_center < obs.data_shape_xy[0]-1-obs.extfov_margin[0] and
            obs.extfov_margin[1] < v_center < obs.data_shape_xy[1]-1-obs.extfov_margin[1]):
            # See if the moon is mostly centered on the image, and just extends
            # past the edges on each side leaving enough curvature behind
            full_inc = ma.zeros((obs.extdata_shape_xy[1], obs.extdata_shape_xy[0]),
                                dtype=np.float32)
            full_inc[:,:] = ma.masked
            full_inc[v_min+obs.extfov_margin[1]:v_max+obs.extfov_margin[1]+1,
                     u_min+obs.extfov_margin[0]:u_max+obs.extfov_margin[0]+1
                    ] = masked_restr_incidence.mvals
            full_inc[:obs.extfov_margin[1]*2,:] = ma.masked
            full_inc[:,:obs.extfov_margin[0]*2] = ma.masked
            full_inc[-obs.extfov_margin[1]*2:,:] = ma.masked
            full_inc[:,-obs.extfov_margin[0]*2:] = ma.masked
            tl_inc = full_inc[:v_center+obs.extfov_margin[1],
                              :u_center+obs.extfov_margin[0]]
            tr_inc = full_inc[:v_center+obs.extfov_margin[1],
                              u_center+obs.extfov_margin[0]:]
            bl_inc = full_inc[v_center+obs.extfov_margin[1]:,
                              :u_center+obs.extfov_margin[0]]
            br_inc = full_inc[v_center+obs.extfov_margin[1]:,
                              u_center+obs.extfov_margin[0]:]
            if ((np.min(tl_inc) < limb_threshold and
                 np.min(br_inc) < limb_threshold) or
                (np.min(tr_inc) < limb_threshold and
                 np.min(bl_inc) < limb_threshold)):
                curvature_ok = True

        metadata['curvature_ok'] = curvature_ok
        if metadata['curvature_ok']:
            metadata['limb_ok'] = True
            logger.info('Curvature+limb OK')
        else:
            logger.info('Curvature+limb BAD')

            if cartographic_data and body_name in cartographic_data:
                logger.info('Limb ignored because cartographic data available')
                metadata['limb_ok'] = True
            else:
                ok_masked_incidence = masked_restr_incidence.mvals[np.where(
                                          masked_restr_incidence.mvals <
                                          limb_threshold)]
                inc_frac = (float(ok_masked_incidence.compressed().shape[0]) /
                            float(masked_restr_incidence.mvals.compressed().
                                  flatten().shape[0]))
                if inc_frac >= limb_frac:
                    metadata['limb_ok'] = True
                    logger.info('Limb alone meets criteria '+
                                '(%.2f%% is less than %.2f deg)',
                                inc_frac*100, limb_threshold*oops.DPR)
                else:
                    logger.info('Limb alone fails criteria '+
                                '(%.2f%% is less than %.2f deg, %.2f%% needed)',
                                inc_frac*100, limb_threshold*oops.DPR,
                                limb_frac*100)
                    if not always_create_model:
                        metadata['end_time'] = time.time()
                        return None, metadata, None

    # Make the actual model

    restr_model = None
    if (np.max(restr_body_mask) == 0 or
        np.min(restr_incidence[restr_body_mask].vals) >= oops.HALFPI):
        logger.debug('Looking only at back side - making a faint glow')
        # Make a slight glow even on the back side
        restr_model = np.zeros(restr_body_mask.shape)
        restr_model[restr_body_mask] = 0.05
    else:
        logger.debug('Making Lambert model')

        if bodies_config['use_lambert']:
            restr_o_lambert = restr_o_bp.lambert_law(body_name).mvals.filled(0.).astype('float32')
            restr_model = filter_downsample(restr_o_lambert,
                                            restr_oversample_v,
                                            restr_oversample_u)
            # XXX FOR PAPER ONLY
            if nav.config.PNG_BODY_OUTLINE:
                for _x in range(-1, 2):
                    for _y in range(-1, 2):
                        restr_model[shift_image(limb_mask, _x, _y)] = 5
            if body_name == 'TITAN':
                # Special case for Titan because of the atmospheric glow at
                # high phase angles. The model won't be used for correlation,
                # only for making the pretty offset PNG.
                restr_model = restr_model+filt.maximum_filter(limb_mask, 3)
            # Make a slight glow even past the terminator
            restr_model = restr_model+0.05
            restr_model[restr_body_mask_inv] = 0.
        else:
            restr_model = restr_body_mask.astype('float32')

        if (bodies_config['use_albedo'] and
            body_name in bodies_config['geometric_albedo']):
            albedo = bodies_config['geometric_albedo'][body_name]
            logger.info('Applying albedo %f', albedo)
            restr_model *= albedo

    used_cartographic = False
    if cartographic_data:
        # Create a Meshgrid that only covers the extent of the body WITHIN
        # THIS IMAGE
        if not hasattr(obs, 'restr_bp_dict'):
            obs.restr_bp_dict = {}
        restr_key = (u_min, v_min, u_max, v_max)
        if restr_key in obs.restr_bp_dict:
            restr_bp = obs.restr_bp_dict[restr_key]
        else:
            restr_meshgrid = oops.Meshgrid.for_fov(obs.fov,
                                                   origin=(u_min+.5, v_min+.5),
                                                   limit =(u_max+.5, v_max+.5),
                                                   swap  =True)
            restr_bp = oops.backplane.Backplane(obs, meshgrid=restr_meshgrid)
            obs.restr_bp_dict[restr_key] = restr_bp
        for cart_body in sorted(cartographic_data.keys()):
            if cart_body == body_name:
                logger.info('Cartographic data provided for %s - USING',
                            cart_body)
                cart_body_data = cartographic_data[body_name]
                (cart_model, image_blur_amt, model_blur_amt,
                 nav_uncertainty) = _bodies_create_cartographic_model(
                                               restr_bp, cart_body_data)
                if cart_model is not None:
                    # Apply the Lambert shading to the cartographic model
                    restr_model *= cart_model
                    metadata['image_blur'] = image_blur_amt
                    metadata['body_blur'] = model_blur_amt
                    metadata['nav_uncertainty'] = nav_uncertainty
                    logger.info('Image blur amount %.5f', image_blur_amt)
                    logger.info('Model blur amount %.5f', model_blur_amt)
                    used_cartographic = True
                else:
                    logger.info('Cartographic model failed')
            else:
                logger.info('Cartographic data provided for %s',
                            cart_body)

    if not used_cartographic:
        if body_name in bodies_config['surface_bumpiness']:
            center_resolution = obs.ext_bp.center_resolution(body_name).vals
            if (center_resolution <
                bodies_config['surface_bumpiness'][body_name]):
                metadata['body_blur'] = (bodies_config[
                                             'surface_bumpiness'][body_name] /
                                       center_resolution)
                metadata['image_blur'] = metadata['body_blur']
                logger.info('Resolution %.2f is too high - limb will look '+
                            'bumpy - need to blur by %.5f',
                            center_resolution,
                            metadata['body_blur'])
            else:
                logger.info('Resolution %.2f is good enough for a sharp edge',
                            center_resolution)


    # Take the full-resolution object and put it back in the right place in a
    # full-size image
    model = np.zeros((obs.data_shape_xy[1]+obs.extfov_margin[1]*2,
                      obs.data_shape_xy[0]+obs.extfov_margin[0]*2),
                     dtype=np.float32)
    if restr_model is not None:
        model[v_min+obs.extfov_margin[1]:v_max+obs.extfov_margin[1]+1,
              u_min+obs.extfov_margin[0]:u_max+obs.extfov_margin[0]+1] = restr_model

    if create_overlay:
        model_text = _bodies_make_label(obs, body_name, model, label_avoid_mask,
                                        bodies_config)
    else:
        model_text = None

    metadata['confidence'] = 1.
    metadata['end_time'] = time.time()
    return model, metadata, model_text

import time
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
# import scipy.ndimage.filters as filt

import oops
import polymath

from nav.annotation import Annotation
from nav.config.global_config import BODIES_CONFIG
from nav.inst import Inst
from nav.util.image import (filter_downsample,
                            shift_array)
from nav.util.types import NDArrayFloatType

from .nav_model import NavModel

# Sometimes the bounding box returned by "inventory" is not quite big enough
BODIES_POSITION_SLOP_FRAC = 0.05


class NavModelBody(NavModel):
    def __init__(self,
                 obs: oops.Observation,
                 body_name: str,
                 *,
                 inventory: Optional[dict[str, Any]] = None,
                 bodies_config: Optional[dict[str, Any]] = None,
                 **kwargs: Any):
        """
                obs                    The Observation.
        body_name              The name of the moon.


        Args:
            obs (Observation): _description_
            body_name (str): _description_
        """

        super().__init__(obs, logger_name='NavModelBody', **kwargs)

        self._body_name = body_name.upper()
        self._config = bodies_config or BODIES_CONFIG

        if inventory is None:
            inventory = self._obs.inventory([self._body_name],
                                            return_type='full')[body_name]
        self._inventory = inventory

    def create_model(self,
                     *,
                     always_create_model: bool = False,
                     never_create_model: bool = False,
                     create_overlay: bool = False
                     ) -> tuple[NDArrayFloatType | None,
                                dict[str, Any],
                                Annotation | None]:
        """Create a model for a body.

        Parameters:
            inventory: The entry returned from the inventory() method of an Observation
                for this body. Used to find the clipping rectangle.

            always_create_model: True to always return a model even if the body is too
                small or the curvature or limb is bad.

            create_overlay: True to create the text overlay.

            never_create_model: Create a metadata structure but don't make an actual
            model.

        Returns:
            model, metadata, text

            metadata is a dictionary containing

            'body_name'            The name of the body. 'size_ok'             True if the
            body is large enough to bother with. 'ring_emission_ok'     True if the ring
            emission angle at the body's
                                location is large enough that the body will be visible and
                                not hidden by the rings. Only relevant for bodies located
                                inside the rings; always True otherwise.
            'inventory'            The inventory entry for this body.
            'cartographic_data_source'
                                The path of the mosaic used to provide the cartographic
                                data. None if no data provided.
            'curvature_ok'         True if sufficient curvature is visible to permit
                                correlation.
            'limb_ok'              True if the limb is sufficiently sharp to permit
                                correlation.
            'entirely_visible'     True if the body is entirely visible even if
                                shifted to the maximum extent specified by extfov_margin.
                                This is based purely on geometry, not looking at whether
                                other objects occult it.
            'occulted_by'          A list of body names or 'RINGS' that occult some
                                or all of this body. This is set in the main offset loop,
                                not in this procedure. None if nothing occults it or it
                                hasn't been processed by the main loop yet.
            'in_saturn_shadow'     True if the body is in Saturn's shadow and only
                                illuminated by Saturn-shine.
            'sub_solar_lon'        The sub-solar longitude (IAU, West). 'sub_solar_lat' The
            sub-solar latitude. 'sub_observer_lon'     The sub-observer longitude (IAU, West).
            'sub_observer_lat'     The sub-observer latitude. 'phase_angle' The phase angle at
            the body's center. 'body_blur'            The amount of model blur required to
            properly
                                correlate low-resolution cartographic data or to not look
                                bumpy when doing a ellipsoidal model.
            'image_blur'           The amount of image blur required to properly
                                correlate low-resolution cartographic data.
            'nav_uncertainty'      The amount of uncertainty (in pixels) to add to
                                the final model uncertainty by due to the uncertainty of
                                the navigation of the source images in the cartographic
                                data.
            'confidence'           The confidence in how well this model will
                                do in correlation.

            'start_time'           The time (s) when bodies_create_model was called.
            'end_time'             The time (s) when bodies_create_model returned.

                These are used for bootstrapping:

            'reproj'               The reprojection data structure at the
                                resolution specified in the config. Filled in after the
                                offset pass is finished. None if not appropriate for
                                bootstrapping or it hasn't been processed by the main loop
                                yet.
        """

        metadata: dict[str, Any] = {}

        metadata['start_time'] = time.time()

        with self._logger.open(f'Create body model for {self._body_name}'):
            ret = self._create_model(metadata,
                                     always_create_model=always_create_model,
                                     never_create_model=never_create_model,
                                     create_overlay=create_overlay)

        metadata['end_time'] = time.time()

        return ret

    def _create_model(self,
                      metadata: dict[str, Any],
                      always_create_model: bool,
                      never_create_model: bool,
                      create_overlay: bool
                      ) -> tuple[NDArrayFloatType | None,
                                 dict[str, Any],
                                 Annotation | None]:
        # These are just shorthand to make later code easier to read
        obs = self._obs
        body_name = self._body_name
        ext_bp = self._obs.ext_bp
        config = self._config
        inventory = self._inventory

        metadata['body_name'] = body_name
        metadata['inventory'] = inventory
        metadata['size_ok'] = None
        metadata['entirely_visible_extfov'] = None
        # metadata['curvature_ok'] = None
        # metadata['limb_ok'] = None
        # metadata['entirely_visible'] = None
        # metadata['occulted_by'] = None
        # metadata['in_saturn_shadow'] = None
        # metadata['image_blur'] = None
        # metadata['body_blur'] = None
        # metadata['nav_uncertainty'] = None
        # metadata['reproj'] = None
        # metadata['has_bad_pixels'] = None
        # metadata['confidence'] = None

        metadata['sub_solar_lon'] = ext_bp.sub_solar_longitude(body_name).vals
        metadata['sub_solar_lat'] = ext_bp.sub_solar_latitude(body_name).vals
        metadata['sub_observer_lon'] = ext_bp.sub_observer_longitude(body_name).vals
        metadata['sub_observer_lat'] = ext_bp.sub_observer_latitude(body_name).vals
        metadata['phase_angle'] = ext_bp.center_phase_angle(body_name).vals

        self._logger.info('Sub-solar longitude      %6.2f',
                          np.degrees(metadata['sub_solar_lon']))
        self._logger.info('Sub-solar latitude       %6.2f',
                          np.degrees(metadata['sub_solar_lat']))
        self._logger.info('Sub-observer longitude   %6.2f',
                          np.degrees(metadata['sub_observer_lon']))
        self._logger.info('Sub-observer latitude    %6.2f',
                          np.degrees(metadata['sub_observer_lat']))
        self._logger.info('Phase angle              %6.2f',
                          np.degrees(metadata['phase_angle']))

        bb_area = inventory['u_pixel_size'] * inventory['v_pixel_size']
        self._logger.info('Pixel size %.2f x %.2fx, bounding box area %.2f',
                          inventory['u_pixel_size'], inventory['v_pixel_size'], bb_area)
        if bb_area >= config['min_bounding_box_area']:
            metadata['size_ok'] = True
        else:
            metadata['size_ok'] = False
            if not always_create_model:
                self._logger.info(
                    'Bounding box is too small to bother with - aborting early')
                return None, metadata, None

        # # Create a Meshgrid that only covers the center pixel of the body
        # ctr_uv = inventory['center_uv']
        # ctr_meshgrid = oops.Meshgrid.for_fov(obs.fov,
        #                                      origin=(ctr_uv[0]+.5, ctr_uv[1]+.5),
        #                                      limit =(ctr_uv[0]+.5, ctr_uv[1]+.5),
        #                                      swap  =True)
        # ctr_bp = oops.backplane.Backplane(obs, meshgrid=ctr_meshgrid)
        # if body_name == obs.planet:
        #     metadata['in_planet_shadow'] = False
        # else:
        #     # We're just going to assume if part of the body is shadowed, the whole
        #     # thing is
        #     saturn_shadow = ctr_bp.where_inside_shadow(body_name, 'saturn').vals
        #     if np.any(saturn_shadow):
        #         self._logger.info(f'Body is in {obs.planet.title()}\'s shadow')
        #         metadata['in_planet_shadow'] = True
        #     else:
        #         metadata['in_planet_shadow'] = False

        # if body_name in nav.config.RINGS_BODY_LIST:
        #     emission = obs.ext_bp.emission_angle('saturn:ring').mvals.astype('float32')
        #     min_emission = np.min(np.abs(emission-oops.HALFPI))
        #     if min_emission*oops.DPR < bodies_config['min_emission_ring_body']:
        #         self._logger.info('Minimum emission angle %.2f from 90 too close to ring '+
        #                     'plane - not using', min_emission*oops.DPR)
        #         metadata['ring_emission_ok'] = False
        #         metadata['end_time'] = time.time()
        #         return None, metadata, None

        # metadata['ring_emission_ok'] = True

        u_min = int(inventory['u_min_unclipped'])
        u_max = int(inventory['u_max_unclipped'])
        v_min = int(inventory['v_min_unclipped'])
        v_max = int(inventory['v_max_unclipped'])

        # For curvature later
        u_center = int((u_min + u_max) / 2)
        v_center = int((v_min + v_max) / 2)
        width = u_max - u_min + 1
        height = v_max - v_min + 1
        curvature_threshold_frac = config['curvature_threshold_frac']
        curvature_threshold_pix = config['curvature_threshold_pixels']
        width_threshold = max(width * curvature_threshold_frac,
                              curvature_threshold_pix)
        height_threshold = max(height * curvature_threshold_frac,
                               curvature_threshold_pix)

        u_min -= int((u_max-u_min) * BODIES_POSITION_SLOP_FRAC)
        u_max += int((u_max-u_min) * BODIES_POSITION_SLOP_FRAC)
        v_min -= int((v_max-v_min) * BODIES_POSITION_SLOP_FRAC)
        v_max += int((v_max-v_min) * BODIES_POSITION_SLOP_FRAC)

        u_min, v_min = obs.clip_extfov(u_min, v_min)
        u_max, v_max = obs.clip_extfov(u_max, v_max)

        # Things break if the moon is only a single pixel wide or tall
        if u_min == u_max and u_min == obs.extfov_u_max:
            u_min -= 1
        if u_min == u_max and u_min == obs.extfov_u_min:
            u_max += 1
        if v_min == v_max and v_min == obs.extfov_v_max:
            v_min -= 1
        if v_min == v_max and v_min == obs.extfov_v_min:
            v_max += 1

        self._logger.debug('Original bounding box U %d to %d, V %d to %d',
                           u_min, u_max, v_min, v_max)
        self._logger.debug('Image size %d x %d; subrect w/slop U %d to %d, V %d to %d',
                           obs.data_shape_u, obs.data_shape_v, u_min, u_max, v_min, v_max)

        entirely_visible_extfov = False
        if (u_min >= obs.extfov_margin_u and
            u_max <= obs.data_shape_u-1 - obs.extfov_margin_u and
            v_min >= obs.extfov_margin_v and
            v_max <= obs.data_shape_v-1 - obs.extfov_margin_v):
            # Body is entirely visible - no part is off the edge even when shifting
            # the extended FOV
            entirely_visible_extfov = True
            self._logger.info('All of body is guaranteed visible even after maximum offset')
        else:
            self._logger.info('Not all of body guaranteed to be visible after maximum offset')
        metadata['entirely_visible_extfov'] = entirely_visible_extfov

        if never_create_model:
            return None, metadata, None

        # Make a new Backplane that only covers the body, but oversample
        # it so we can do anti-aliasing
        restr_oversample_u = max(int(np.floor(config['oversample_edge_limit'] /
                                              np.ceil(inventory['u_pixel_size']))), 1)
        restr_oversample_v = max(int(np.floor(config['oversample_edge_limit'] /
                                              np.ceil(inventory['v_pixel_size']))), 1)
        restr_oversample_u = min(restr_oversample_u, config['oversample_maximum'])
        restr_oversample_v = min(restr_oversample_v, config['oversample_maximum'])
        self._logger.debug('Oversampling by %d x %d',
                           restr_oversample_u, restr_oversample_v)
        restr_u_min = u_min + 1./(2*restr_oversample_u)
        restr_u_max = u_max + 1 - 1./(2*restr_oversample_u)
        restr_v_min = v_min + 1./(2*restr_oversample_v)
        restr_v_max = v_max + 1 - 1./(2*restr_oversample_v)
        restr_o_meshgrid = oops.Meshgrid.for_fov(obs.fov,
                                                 origin=(restr_u_min, restr_v_min),
                                                 limit=(restr_u_max, restr_v_max),
                                                 oversample=(restr_oversample_u,
                                                             restr_oversample_v),
                                                 swap=True)
        restr_o_bp = oops.backplane.Backplane(obs, meshgrid=restr_o_meshgrid)
        restr_o_incidence_mvals = restr_o_bp.incidence_angle(body_name).mvals
        restr_incidence_mvals = filter_downsample(restr_o_incidence_mvals,
                                                  restr_oversample_v,
                                                  restr_oversample_u)
        restr_incidence = polymath.Scalar(restr_incidence_mvals)

        # Analyze the limb

        restr_body_mask_inv = restr_incidence.expand_mask().mask
        restr_body_mask_valid = ~restr_body_mask_inv

        # If the inv mask is true, but any of its neighbors are false, then
        # this is an edge
        restr_limb_mask_total = (shift_array(restr_body_mask_inv, (-1, 0)) |
                                 shift_array(restr_body_mask_inv, (1, 0)) |
                                 shift_array(restr_body_mask_inv, (0, -1)) |
                                 shift_array(restr_body_mask_inv, (0, 1)))
        # This valid mask will be a single series of pixels just inside the limb
        restr_limb_mask = restr_body_mask_valid & restr_limb_mask_total

        if not restr_limb_mask.any():
            limb_incidence_min = 1e38
            limb_incidence_max = 1e38
            self._logger.info('There is no limb')
        else:
            restr_incidence_limb = restr_incidence.mask_where(~restr_limb_mask)
            limb_incidence_min = restr_incidence_limb.min().vals
            limb_incidence_max = restr_incidence_limb.max().vals
            self._logger.info('Limb incidence angle min %.2f, max %.2f',
                              np.degrees(limb_incidence_min),
                              np.degrees(limb_incidence_max))

            # limb_threshold = config['limb_incidence_threshold']
            # limb_frac = config['limb_incidence_frac']

            # # Slightly more than half of the body must be visible in each of the
            # # horizontal and vertical directions AND also have a good limb in
            # # those quadrants

            # curvature_ok = False
            # limb_ok = False
            # l_edge = obs.extfov_margin_u
            # r_edge = obs.data_shape_u-1-obs.extfov_margin_u
            # t_edge = obs.extfov_margin_v
            # b_edge = obs.data_shape_v-1-obs.extfov_margin_v
            # inc_r_edge = restr_incidence_limb.vals.shape[1]-1
            # inc_b_edge = restr_incidence_limb.vals.shape[0]-1
            # for l_moon, r_moon, lr_str in ((u_center-width_threshold,
            #                                 u_center+width/2, 'L'),
            #                             (u_center-width/2,
            #                             u_center+width_threshold, 'R')):
            #     for t_moon, b_moon, tb_str in ((v_center-height_threshold,
            #                                     v_center+height/2, 'T'),
            #                                 (v_center-height/2,
            #                                     v_center+height_threshold, 'B')):
            #         self._logger.debug('LMOON %d LEDGE %d / RMOON %d REDGE %d / '+
            #                     'TMOON %d TEDGE %d / BMOON %d BEDGE %d',
            #                     l_moon, l_edge, r_moon, r_edge,
            #                     t_moon, t_edge, b_moon, b_edge)
            #         if (l_moon >= l_edge and r_moon <= r_edge and
            #             t_moon >= t_edge and b_moon <= b_edge):
            #             l_moon_clip = int(np.clip(l_moon-u_min, 0, inc_r_edge))
            #             r_moon_clip = int(np.clip(r_moon-u_min, 0, inc_r_edge))
            #             t_moon_clip = int(np.clip(t_moon-v_min, 0, inc_b_edge))
            #             b_moon_clip = int(np.clip(b_moon-v_min, 0, inc_b_edge))
            #             sub_inc = masked_restr_incidence[
            #                                 t_moon_clip:b_moon_clip+1,
            #                                 l_moon_clip:r_moon_clip+1].mvals
            #             ok_masked_incidence = sub_inc[np.where(sub_inc <
            #                                                 limb_threshold)]
            #             self._logger.debug('%s %s %d %d', tb_str, lr_str,
            #                         ok_masked_incidence.compressed().shape[0],
            #                         sub_inc.compressed().shape[0])
            #             inc_frac = (float(ok_masked_incidence.compressed().shape[0]) /
            #                         float(sub_inc.compressed().flatten().shape[0]))
            #             if inc_frac >= limb_frac:
            #                 self._logger.debug('Curvature+limb quadrant %s%s OK',
            #                             tb_str, lr_str)
            #                 curvature_ok = True

            # if (not curvature_ok and
            #     obs.extfov_margin[0] < u_center < obs.data_shape_xy[0]-1-obs.extfov_margin[0] and
            #     obs.extfov_margin[1] < v_center < obs.data_shape_xy[1]-1-obs.extfov_margin[1]):
            #     # See if the moon is mostly centered on the image, and just extends
            #     # past the edges on each side leaving enough curvature behind
            #     full_inc = ma.zeros((obs.extdata_shape_xy[1], obs.extdata_shape_xy[0]),
            #                         dtype=np.float32)
            #     full_inc[:,:] = ma.masked
            #     full_inc[v_min+obs.extfov_margin[1]:v_max+obs.extfov_margin[1]+1,
            #             u_min+obs.extfov_margin[0]:u_max+obs.extfov_margin[0]+1
            #             ] = masked_restr_incidence.mvals
            #     full_inc[:obs.extfov_margin[1]*2,:] = ma.masked
            #     full_inc[:,:obs.extfov_margin[0]*2] = ma.masked
            #     full_inc[-obs.extfov_margin[1]*2:,:] = ma.masked
            #     full_inc[:,-obs.extfov_margin[0]*2:] = ma.masked
            #     tl_inc = full_inc[:v_center+obs.extfov_margin[1],
            #                     :u_center+obs.extfov_margin[0]]
            #     tr_inc = full_inc[:v_center+obs.extfov_margin[1],
            #                     u_center+obs.extfov_margin[0]:]
            #     bl_inc = full_inc[v_center+obs.extfov_margin[1]:,
            #                     :u_center+obs.extfov_margin[0]]
            #     br_inc = full_inc[v_center+obs.extfov_margin[1]:,
            #                     u_center+obs.extfov_margin[0]:]
            #     if ((np.min(tl_inc) < limb_threshold and
            #         np.min(br_inc) < limb_threshold) or
            #         (np.min(tr_inc) < limb_threshold and
            #         np.min(bl_inc) < limb_threshold)):
            #         curvature_ok = True

            # metadata['curvature_ok'] = curvature_ok
            # if metadata['curvature_ok']:
            #     metadata['limb_ok'] = True
            #     self._logger.info('Curvature+limb OK')
            # else:
            #     self._logger.info('Curvature+limb BAD')

        # Make the actual model

        restr_model = None
        if (not restr_body_mask_valid.any() or
            restr_incidence[restr_body_mask_valid].min() >= oops.HALFPI):
            self._logger.debug('Looking only at back side - making a faint glow')
            # Make a slight glow even on the back side
            restr_model = np.zeros(restr_body_mask_valid.shape)
            restr_model[restr_body_mask_valid] = 0.05  # XXX
        else:
            self._logger.debug('Making Lambert model')

            if config['use_lambert']:
                # Make an oversampled Lambert, then downsample to get a nice anti-aliased
                # edge
                restr_o_lambert = restr_o_bp.lambert_law(body_name).mvals.filled(0.)
                restr_model = filter_downsample(restr_o_lambert,
                                                restr_oversample_v,
                                                restr_oversample_u)
                # if body_name == 'TITAN':
                #     # Special case for Titan because of the atmospheric glow at
                #     # high phase angles. The model won't be used for correlation,
                #     # only for making the pretty offset PNG.
                #     restr_model = restr_model+filt.maximum_filter(limb_mask, 3)
                # Make a slight glow even past the terminator
                restr_model = restr_model+0.05  # XXX
                restr_model[restr_body_mask_inv] = 0.
            else:
                restr_model = restr_body_mask_valid.as_float()

            if (config['use_albedo'] and
                body_name in config['geometric_albedo']):
                albedo = config['geometric_albedo'][body_name]
                self._logger.info('Applying albedo %.6f', albedo)
                restr_model *= albedo

        # if not used_cartographic:
        #     if body_name in bodies_config['surface_bumpiness']:
        #         center_resolution = obs.ext_bp.center_resolution(body_name).vals
        #         if (center_resolution <
        #             bodies_config['surface_bumpiness'][body_name]):
        #             metadata['body_blur'] = (bodies_config[
        #                                         'surface_bumpiness'][body_name] /
        #                                 center_resolution)
        #             metadata['image_blur'] = metadata['body_blur']
        #             self._logger.info('Resolution %.2f is too high - limb will look '+
        #                         'bumpy - need to blur by %.5f',
        #                         center_resolution,
        #                         metadata['body_blur'])
        #         else:
        #             self._logger.info('Resolution %.2f is good enough for a sharp edge',
        #                         center_resolution)

        # Take the full-resolution object and put it back in the right place in a
        # full-size image
        model = obs.make_extfov_zeros()
        limb_mask = obs.make_extfov_false()
        if restr_model is not None:
            model[v_min+obs.extfov_margin_v:v_max+obs.extfov_margin_v+1,
                  u_min+obs.extfov_margin_u:u_max+obs.extfov_margin_u+1] = restr_model
            limb_mask[v_min+obs.extfov_margin_v:v_max+obs.extfov_margin_v+1,
                      u_min+obs.extfov_margin_u:u_max+obs.extfov_margin_u+1] = \
                restr_limb_mask

        # if create_overlay:
        #     model_text = _bodies_make_label(obs, body_name, model, label_avoid_mask,
        #                                     bodies_config)
        # else:
        model_text = None

        metadata['confidence'] = 1.

        annotation = Annotation(limb_mask, thicken_overlay=0, text_info=model_text)
        return model, metadata, annotation

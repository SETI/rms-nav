import math
from typing import Any, Optional

import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

from oops import Meshgrid, Observation
from oops.backplane import Backplane
import polymath

from nav.annotation import (Annotation,
                            Annotations,
                            AnnotationTextInfo,
                            TextLocInfo,
                            TEXTINFO_LEFT_ARROW,
                            TEXTINFO_RIGHT_ARROW,
                            TEXTINFO_BOTTOM_ARROW,
                            TEXTINFO_TOP_ARROW)
from nav.support.constants import HALFPI
from nav.support.image import (filter_downsample,
                               shift_array)
from nav.support.misc import now_dt, dt_delta_str
from nav.support.types import NDArrayBoolType, NDArrayFloatType

from .nav_model import NavModel

# Sometimes the bounding box returned by "inventory" is not quite big enough
BODIES_POSITION_SLOP_FRAC = 0.05


class NavModelBody(NavModel):
    def __init__(self,
                 obs: Observation,
                 body_name: str,
                 *,
                 inventory: Optional[dict[str, Any]] = None,
                 **kwargs: Any):
        """Creates a navigation model for a planetary body.

        Parameters:
            obs: The observation object containing the image data.
            body_name: The name of the planetary body.
            inventory: Optional dictionary containing inventory information for the body.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """

        super().__init__(obs, logger_name='NavModelBody', **kwargs)

        self._body_name = body_name.upper()

        if inventory is None:
            inventory = self.obs.inventory([self._body_name],
                                           return_type='full')[body_name]
        self._inventory = inventory

    def create_model(self,
                     *,
                     always_create_model: bool = False,
                     never_create_model: bool = False,
                     create_annotations: bool = True
                     ) -> None:
        """Creates a navigation model for a planetary body with optional text overlay.

        Parameters:
            always_create_model: If True, creates a model even if the body is too small or has
                poor limb definition.
            never_create_model: If True, creates metadata but doesn't generate an actual model or
                annotations.
            create_annotations: If True, creates text annotations for the model.
        """

        metadata: dict[str, Any] = {}
        metadata['start_time'] = start_time = now_dt()
        metadata['end_time'] = None
        metadata['elapsed_time'] = None

        self._model = None
        self._model_mask = None
        self._metadata = metadata
        self._annotations = None
        self._uncertainty = 0.

        with self._logger.open(f'CREATE BODY MODEL FOR: {self._body_name}'):
            self._create_model(always_create_model=always_create_model,
                               never_create_model=never_create_model,
                               create_annotations=create_annotations)

        metadata['end_time'] = end_time = now_dt()
        metadata['elapsed_time'] = dt_delta_str(start_time, end_time)
        return

    def _create_model(self,
                      always_create_model: bool,
                      never_create_model: bool,
                      create_annotations: bool
                      ) -> None:
        """Creates the internal model representation for a planetary body.

        Parameters:
            always_create_model: If True, creates a model even if the body is too small.
            never_create_model: If True, only creates metadata without generating a model.
            create_annotations: If True, creates a text overlay with the body name.
        """

        # These are just shorthand to make later code easier to read
        obs = self.obs
        body_name = self._body_name
        ext_bp = self.obs.ext_bp
        config = self._config.bodies
        inventory = self._inventory
        metadata = self._metadata

        ########################################################################
        # Fill in basic metadata
        ########################################################################

        metadata['body_name'] = body_name
        metadata['inventory'] = inventory
        metadata['size_ok'] = None
        metadata['guaranteed_visible_in_fov'] = None
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

        # Single-valued metadata
        metadata['sub_solar_lon'] = np.degrees(ext_bp.sub_solar_longitude(body_name).vals)
        metadata['sub_solar_lat'] = np.degrees(ext_bp.sub_solar_latitude(body_name).vals)
        metadata['sub_observer_lon'] = np.degrees(ext_bp.sub_observer_longitude(body_name).vals)
        metadata['sub_observer_lat'] = np.degrees(ext_bp.sub_observer_latitude(body_name).vals)
        metadata['phase_angle'] = np.degrees(ext_bp.center_phase_angle(body_name).vals)

        self._logger.info(f'Sub-solar longitude      {metadata["sub_solar_lon"]:6.2f}')
        self._logger.info(f'Sub-solar latitude       {metadata["sub_solar_lat"]:6.2f}')
        self._logger.info(f'Sub-observer longitude   {metadata["sub_observer_lon"]:6.2f}')
        self._logger.info(f'Sub-observer latitude    {metadata["sub_observer_lat"]:6.2f}')
        self._logger.info(f'Phase angle              {metadata["phase_angle"]:6.2f}')

        ########################################################################
        # Check the size of the bounding box
        ########################################################################

        bb_area = inventory['u_pixel_size'] * inventory['v_pixel_size']
        self._logger.info(f'Pixel size {inventory["u_pixel_size"]:.2f} x '
                          f'{inventory["v_pixel_size"]:.2f}, bounding box area {bb_area:.2f}')
        if bb_area >= config.min_bounding_box_area:
            metadata['size_ok'] = True
        else:
            metadata['size_ok'] = False
            if not always_create_model:
                self._logger.info('Bounding box is too small to bother with - aborting early')
                return

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

        ########################################################################
        # Figure out if the body is guaranteed to be visible in the FOV
        ########################################################################

        u_min = int(inventory['u_min_unclipped'])
        u_max = int(inventory['u_max_unclipped'])
        v_min = int(inventory['v_min_unclipped'])
        v_max = int(inventory['v_max_unclipped'])

        # For curvature later
        u_center = (u_min + u_max) // 2
        v_center = (v_min + v_max) // 2
        # width = u_max - u_min + 1
        # height = v_max - v_min + 1
        # curvature_threshold_frac = config.curvature_threshold_frac
        # curvature_threshold_pix = config.curvature_threshold_pixels
        # width_threshold = max(width * curvature_threshold_frac,
        #                       curvature_threshold_pix)
        # height_threshold = max(height * curvature_threshold_frac,
        #                        curvature_threshold_pix)

        # Figure out the bounding box with some slop and see whether or not the body is
        # going to be entirely visible even in the case of maximum offset

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

        self._logger.debug(f'Original bounding box U {u_min} to {u_max}, V {v_min} to {v_max}')
        self._logger.debug(f'Image size {obs.data_shape_u} x {obs.data_shape_v}; '
                           f'subrect w/slop U {u_min} to {u_max}, V {v_min} to {v_max}')

        guaranteed_visible_in_fov = False
        if (u_min >= obs.extfov_margin_u and
            u_max <= obs.data_shape_u-1 - obs.extfov_margin_u and
            v_min >= obs.extfov_margin_v and
            v_max <= obs.data_shape_v-1 - obs.extfov_margin_v):
            # Body is entirely visible - no part is off the edge even when shifting
            # the extended FOV
            guaranteed_visible_in_fov = True
            self._logger.info('All of body is guaranteed visible even after maximum offset')
        else:
            self._logger.info('Not all of body guaranteed to be visible after maximum offset')
        metadata['guaranteed_visible_in_fov'] = guaranteed_visible_in_fov

        if never_create_model:
            return

        ########################################################################
        # Make a new Backplane that only covers the body, but oversample it
        # as necessary so we can do anti-aliasing
        ########################################################################

        restr_oversample_u = max(int(np.floor(config.oversample_edge_limit /
                                              max(np.ceil(inventory['u_pixel_size']),
                                                  1))),
                                 1)
        restr_oversample_v = max(int(np.floor(config.oversample_edge_limit /
                                              max(np.ceil(inventory['v_pixel_size']),
                                                  1))),
                                 1)
        restr_oversample_u = min(restr_oversample_u, config.oversample_maximum)
        restr_oversample_v = min(restr_oversample_v, config.oversample_maximum)
        self._logger.debug(f'Oversampling by {restr_oversample_u} x {restr_oversample_v}')
        restr_u_min = u_min + 1./(2*restr_oversample_u)
        restr_u_max = u_max + 1 - 1./(2*restr_oversample_u)
        restr_v_min = v_min + 1./(2*restr_oversample_v)
        restr_v_max = v_max + 1 - 1./(2*restr_oversample_v)
        restr_o_meshgrid = Meshgrid.for_fov(obs.fov,
                                            origin=(restr_u_min, restr_v_min),
                                            limit=(restr_u_max, restr_v_max),
                                            oversample=(restr_oversample_u,
                                                        restr_oversample_v),
                                            swap=True)
        restr_o_bp = Backplane(obs, meshgrid=restr_o_meshgrid)

        ########################################################################
        # Compute the incidence angles
        ########################################################################

        restr_o_incidence_mvals = restr_o_bp.incidence_angle(body_name).mvals
        restr_incidence_mvals = filter_downsample(restr_o_incidence_mvals,
                                                  restr_oversample_v,
                                                  restr_oversample_u)
        restr_incidence = polymath.Scalar(restr_incidence_mvals)

        ########################################################################
        # Analyze the limb
        ########################################################################

        restr_body_mask_invalid = restr_incidence.expand_mask().mask
        restr_body_mask_valid = ~restr_body_mask_invalid

        # If the inv mask is true, but any of its neighbors are false, then
        # this is an edge
        restr_limb_mask_neighbor = (shift_array(restr_body_mask_invalid, (-1,  0)) |
                                    shift_array(restr_body_mask_invalid, ( 1,  0)) |
                                    shift_array(restr_body_mask_invalid, ( 0, -1)) |
                                    shift_array(restr_body_mask_invalid, ( 0,  1)))
        # This valid mask will be a single series of pixels just inside the limb
        restr_limb_mask = restr_body_mask_valid & restr_limb_mask_neighbor

        if not restr_limb_mask.any():
            self._logger.info('There is no limb')
        else:
            restr_incidence_limb = restr_incidence.mask_where(~restr_limb_mask)
            limb_incidence_min = np.degrees(restr_incidence_limb.min().vals)
            limb_incidence_max = np.degrees(restr_incidence_limb.max().vals)
            self._logger.info('Limb incidence angle '
                              f'min {limb_incidence_min:.2f}, '
                              f'max {limb_incidence_max:.2f}')

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

        ########################################################################
        # Make the actual model
        ########################################################################

        restr_model = None
        if (not restr_body_mask_valid.any() or
            restr_incidence[restr_body_mask_valid].min() >= HALFPI):
            self._logger.debug('Looking only at back side - making a faint glow')
            # Make a slight glow even on the back side
            restr_model = np.zeros(restr_body_mask_valid.shape)
            restr_model[restr_body_mask_valid] = 0.01  # TODO XXX
        else:
            self._logger.debug('Making Lambert model')

            if config.use_lambert:
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
            else:
                restr_model = restr_body_mask_valid.as_float()

            if (config.use_albedo and
                body_name in config.geometric_albedo):
                albedo = config.geometric_albedo[body_name]
                self._logger.info(f'Applying albedo {albedo:.6f}')
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

        ########################################################################
        # Put the small model back in the right place in the full-size model
        ########################################################################

        model_slice_0 = slice(v_min+obs.extfov_margin_v, v_max+obs.extfov_margin_v+1)
        model_slice_1 = slice(u_min+obs.extfov_margin_u, u_max+obs.extfov_margin_u+1)
        model_img = obs.make_extfov_zeros()
        model_img[model_slice_0, model_slice_1] = restr_model
        limb_mask = obs.make_extfov_false()
        limb_mask[model_slice_0, model_slice_1] = restr_limb_mask
        body_mask = obs.make_extfov_false()
        body_mask[:, :] = model_img != 0

        # This is much faster than calculating the range at each pixel and we never
        # need to know the precise range at that level of detail
        self._range = obs.make_extfov_zeros()
        self._range[:, :] = inventory['range'] * body_mask
        self._range[self._range == 0] = math.inf

        metadata['confidence'] = 1.

        ########################################################################
        # Figure out all the location where we might want to label the body
        ########################################################################

        if create_annotations:
            self._annotations = self._create_annotations(u_center, v_center, model_img,
                                                         limb_mask, body_mask)

        self._model_img = model_img
        self._model_mask = body_mask

        self._logger.debug(f'  Body model min: {np.min(self._model_img)}, '
                           f'max: {np.max(self._model_img)}')

    def _create_annotations(self,
                            u_center: int,
                            v_center: int,
                            model: NDArrayFloatType,
                            limb_mask: NDArrayBoolType,
                            body_mask: NDArrayBoolType) -> Annotations:
        """Creates annotation objects for labeling the planetary body in visualizations.

        Parameters:
            u_center: The center U coordinate of the body in the image.
            v_center: The center V coordinate of the body in the image.
            model: The model image array for the body.
            limb_mask: Boolean mask indicating the limb (edge) of the body.
            body_mask: Boolean mask indicating the visible portion of the body.

        Returns:
            A collection of annotations for the body.
        """

        obs = self._obs
        body_name = self._body_name
        config = self._config.bodies

        text_loc: list[TextLocInfo] = []
        v_center_extfov = v_center + obs.extfov_margin_v
        u_center_extfov = u_center + obs.extfov_margin_u

        v_center_extfov_clipped = np.clip(v_center_extfov, 0, body_mask.shape[0]-1)
        u_center_extfov_clipped = np.clip(u_center_extfov, 0, body_mask.shape[1]-1)
        body_mask_u_min = int(np.argmax(body_mask[v_center_extfov_clipped]))
        body_mask_u_max = int((body_mask.shape[1] -
                              np.argmax(body_mask[v_center_extfov_clipped, ::-1]) - 1))
        body_mask_v_min = int(np.argmax(body_mask[:, u_center_extfov_clipped]))
        body_mask_v_max = int((body_mask.shape[0] -
                              np.argmax(body_mask[::-1, u_center_extfov_clipped]) - 1))
        body_mask_u_ctr = (body_mask_u_min + body_mask_u_max) // 2
        body_mask_v_ctr = (body_mask_v_min + body_mask_v_max) // 2

        # Scan the body in +/- V starting at the center of the body in the FOV. For each
        # V, find the leftmost and rightmost U of the limb and place a label there. This
        # gives preference to labels centered vertically on the body.
        # For each label, decide whether it should be a horizontal arrow or a vertical
        # arrow based on the local curvature of the limb, computed as the angle relative
        # to the absolute center of the body.
        for orig_dist in range(0, max(body_mask_v_ctr - body_mask_v_min,
                                      config.label_scan_v)):
            for neg in [-1, 1]:
                dist = orig_dist * neg
                v = body_mask_v_ctr + dist
                if not 0 <= v < body_mask.shape[0]:
                    continue

                # Left side
                u = int(np.argmax(body_mask[v]))
                if u > 0:  # u == 0 if body runs off left side
                    angle = np.rad2deg(
                        np.arctan2(v-v_center_extfov, u-u_center_extfov)) % 360
                    if 135 < angle < 225:  # Left side
                        text_loc.append(TextLocInfo(TEXTINFO_LEFT_ARROW,
                                                    v,
                                                    u - config.label_horiz_gap))
                    elif angle >= 225:  # Top side
                        text_loc.append(TextLocInfo(TEXTINFO_TOP_ARROW,
                                                    v - config.label_vert_gap,
                                                    u))
                    else:  # Bottom side
                        text_loc.append(TextLocInfo(TEXTINFO_BOTTOM_ARROW,
                                                    v + config.label_vert_gap,
                                                    u))

                # Right side
                u = body_mask.shape[1] - int(np.argmax(body_mask[v, ::-1])) - 1
                if u < body_mask.shape[1]-1:  # if body runs off right side
                    angle = np.rad2deg(
                        np.arctan2(v-v_center_extfov, u-u_center_extfov)) % 360
                    if angle > 315 or angle < 45:  # Right side
                        text_loc.append(TextLocInfo(TEXTINFO_RIGHT_ARROW,
                                                    v,
                                                    u + config.label_horiz_gap))
                    elif angle >= 225:  # Top side
                        text_loc.append(TextLocInfo(TEXTINFO_TOP_ARROW,
                                                    v - config.label_vert_gap,
                                                    u))
                    else:  # Bottom side
                        text_loc.append(TextLocInfo(TEXTINFO_BOTTOM_ARROW,
                                                    v + config.label_vert_gap,
                                                    u))

                if orig_dist == 0:
                    # Add in the very top and very bottom here to give them
                    # priority
                    text_loc.append(TextLocInfo(TEXTINFO_TOP_ARROW,
                                                body_mask_v_min - config.label_vert_gap,
                                                body_mask_u_ctr))
                    text_loc.append(TextLocInfo(TEXTINFO_BOTTOM_ARROW,
                                                body_mask_v_max + config.label_vert_gap,
                                                body_mask_u_ctr))
                    break  # No need to do +/- with zero

        # Finally, it's possible none of the above worked, especially it the body is
        # really large in the FOV. So scan through the FOV on a coarse grid, check
        # if each point is in the body, and add it to the list if so. We work hard
        # to give priority to points that are near the center of the body in the FOV.

        for v_orig_dist in range(0, body_mask_v_ctr - body_mask_v_min,
                                 config.label_grid_v):
            for v_neg in [-1, 1]:
                v_dist = v_orig_dist * v_neg
                v = body_mask_v_ctr + v_dist
                if not 0 <= v < body_mask.shape[0]:
                    continue
                for u_orig_dist in range(0, body_mask_u_ctr - body_mask_u_min,
                                         config.label_grid_u):
                    for u_neg in [-1, 1]:
                        u_dist = u_orig_dist * u_neg
                        u = body_mask_u_ctr + u_dist
                        if not 0 <= u < body_mask.shape[1]:
                            continue
                        if not body_mask[v, u]:
                            continue
                        if u < model.shape[1] // 2:
                            text_loc.append(TextLocInfo(TEXTINFO_LEFT_ARROW, v, u))
                        else:
                            text_loc.append(TextLocInfo(TEXTINFO_RIGHT_ARROW, v, u))
                if v_orig_dist == 0:
                    break

        text_info = AnnotationTextInfo(body_name, text_loc=text_loc,
                                       ref_vu=None,
                                       font=config.label_font,
                                       font_size=config.label_font_size,
                                       color=config.label_font_color)

        # Make the avoid mask a little larger than the body mask, so that any text that
        # we place later won't be right up against this body
        text_avoid_mask = ndimage.maximum_filter(body_mask,
                                                 config.label_mask_enlarge)

        annotation = Annotation(obs, limb_mask,
                                config.label_limb_color,
                                thicken_overlay=config.outline_thicken,
                                avoid_mask=text_avoid_mask,
                                text_info=text_info, config=self._config)

        annotations = Annotations()
        annotations.add_annotations(annotation)

        return annotations

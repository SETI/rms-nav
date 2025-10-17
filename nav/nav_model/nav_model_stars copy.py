# mypy: ignore-errors

# import logging

import copy
# import os
# import time

# import numpy.ma as ma
# import scipy.ndimage.filters as filt
# from PIL import Image, ImageDraw, ImageFont

import tkinter as tk
from imgdisp import ImageDisp

# from starcat import (UCAC4StarCatalog,
#                      YBSCStarCatalog,
#                      SCLASS_TO_B_MINUS_V,
#                      SCLASS_TO_SURFACE_TEMP)
# from starcat.starcatalog import (Star,

# import nav.config
# from nav.correlate import (corr_analyze_peak,
#                            corr_log_xy_err,
#                            find_correlation_and_offset)
# import nav.titan
# from nav.flux import (calibrate_iof_image_as_dn,
#                       clean_sclass,
#                       compute_dn_from_star)
from nav.support.image import (#draw_circle,
                       draw_rect)
                    #    filter_sub_median)
# import nav.plot3d

# _LOGGING_NAME = "nav." + __name__

# DEBUG_STARS_FILTER_IMGDISP = False
_DEBUG_STARS_MODEL_IMGDISP = False
# DEBUG_STARS_PSF_3D = False

import time
from typing import Any, Optional, cast

import numpy as np

from oops import Event, Meshgrid, Observation
from oops.backplane import Backplane
import polymath
from psfmodel.gaussian import GaussianPSF
from starcat import (SCLASS_TO_SURFACE_TEMP,
                     SCLASS_TO_B_MINUS_V,
                     Star,
                     UCAC4StarCatalog,
                     YBSCStarCatalog)

from nav.annotation import (Annotation,
                            Annotations,
                            AnnotationTextInfo,
                            TEXTINFO_LEFT,
                            TEXTINFO_RIGHT,
                            TEXTINFO_BOTTOM,
                            TEXTINFO_TOP)
from nav.support.flux import clean_sclass
from nav.support.types import NDArrayFloatType

from .nav_model import NavModel


_STAR_CATALOG_UCAC4 = None
_STAR_CATALOG_YBSC = None


def _get_star_catalog_ucac4():
    """Get UCAC4 star catalog, creating it lazily."""
    global _STAR_CATALOG_UCAC4
    if _STAR_CATALOG_UCAC4 is None:
        _STAR_CATALOG_UCAC4 = UCAC4StarCatalog()
    return _STAR_CATALOG_UCAC4


def _get_star_catalog_ybsc():
    """Get YBSC star catalog, creating it lazily."""
    global _STAR_CATALOG_YBSC
    if _STAR_CATALOG_YBSC is None:
        _STAR_CATALOG_YBSC = YBSCStarCatalog()
    return _STAR_CATALOG_YBSC


class NavModelStars(NavModel):
    def __init__(self,
                 obs: Observation,
                 config: Optional[Config] = None,
                 **kwargs: Any) -> None:
        """Creates a navigation model for stars.

        Parameters:
            obs: The Observation object containing image data.
            config: Configuration object to use. If None, uses DEFAULT_CONFIG.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """

        super().__init__(obs, config=config, **kwargs)

        self._obs = obs
        self._conflict_body_list = None

    def _aberrate_star(self, star: Star) -> None:
        """Update the RA,DEC position of a star with stellar aberration."""

        obs = self._obs
        event = Event(obs.midtime, (polymath.Vector3.ZERO,
                                    polymath.Vector3.ZERO),
                      obs.path, obs.frame)

        event.neg_arr_j2000 = polymath.Vector3.from_ra_dec_length(
            star.ra, star.dec, 1., recursive=False)
        (abb_ra, abb_dec, _) = event.neg_arr_ap_j2000.to_ra_dec_length(recursive=False)

        star.ra = abb_ra.vals
        star.dec = abb_dec.vals

    @staticmethod
    def _star_short_info(star: Star) -> str:
        return ('Star %9d U %8.3f~%7.3f V %8.3f~%7.3f MAG %6.3f BMAG %6.3f '
                'VMAG %6.3f SCLASS %3s TEMP %6d DN %7.2f CONFLICT %s') % (
                    star.unique_number,
                    star.u, abs(star.move_u),
                    star.v, abs(star.move_v),
                    star.vmag,
                    0 if star.johnson_mag_b is None else star.johnson_mag_b,
                    0 if star.johnson_mag_v is None else star.johnson_mag_v,
                    clean_sclass(star.spectral_class),
                    0 if star.temperature is None else star.temperature,
                    0,
                    star.conflicts) # TODO star.dn)

    # def _compute_dimmest_visible_star_vmag(obs, stars_config):
    #     """Compute the VMAG of the dimmest star likely visible."""
    #     min_dn = stars_config[("min_detectable_dn", obs.clean_detector)] # phot/pix
    #     if min_dn == 0:
    #         return 1000
    #     fake_star = Star()
    #     fake_star.temperature = SCLASS_TO_SURFACE_TEMP["G0"]
    #     min_mag = stars_config["min_vmag"]
    #     max_mag = stars_config["max_vmag"]
    #     mag_increment = stars_config["vmag_increment"]
    #     for mag in np.arange(min_mag, max_mag+1e-6, mag_increment):
    #         fake_star.unique_number = 0
    #         fake_star.johnson_mag_v = mag
    #         fake_star.johnson_mag_b = mag+SCLASS_TO_B_MINUS_V["G0"]
    #         fake_star.spectral_class = "G5"
    #         dn = compute_dn_from_star(obs, fake_star)
    #         if dn < min_dn:
    #             return mag # This is conservative
    #     return mag

    def _stars_list_for_obs(self,
                            ra_min: float,
                            ra_max: float,
                            dec_min: float,
                            dec_max: float,
                            mag_min: float,
                            mag_max: float,
                            radec_movement: Optional[tuple[float, float]] = None,
                            **kwargs: Any) -> list[Star]:
        """Return a list of stars with the given constraints.

        See stars_list_for_obs for full details.
        """

        obs = self._obs
        config = self._config.stars

        self._logger.debug('Retrieving stars: Mag range %7.4f to %7.4f', mag_min, mag_max)

        # min_dn = self._config[('min_detectable_dn', obs.clean_detector)]

        # Get a list of all reasonable stars with the given magnitude range.

        star_list1 = []

        if mag_min < 8:  # YBSC maximum is 7.96
            star_list1 = list(
                _get_star_catalog_ybsc().find_stars(
                    allow_double=True,
                    ra_min=ra_min, ra_max=ra_max,
                    dec_min=dec_min, dec_max=dec_max,
                    vmag_min=mag_min, vmag_max=mag_max,
                    **kwargs))
            for star in star_list1:
                star.johnson_mag_v = star.vmag
                if star.b_v is None:
                    star.johnson_mag_b = star.johnson_mag_v
                else:
                    star.johnson_mag_b = star.vmag + star.b_v

        ucac_star_list = list(
                _get_star_catalog_ucac4().find_stars(
                    allow_double=True,
                    allow_galaxy=False,
                    ra_min=ra_min, ra_max=ra_max,
                    dec_min=dec_min, dec_max=dec_max,
                    vmag_min=mag_min, vmag_max=mag_max,
                    **kwargs))

        star_list1 += ucac_star_list

        # This copy is required because we mutate the star object and if we ever
        # reuse the same star object (like trying multiple navigations), we want
        # it to start fresh.
        # TODO It's not clear why we need this. Without this the find_fov_twist
        # program doesn't work. It should be investigated.
        star_list1 = [copy.deepcopy(star) for star in star_list1]

        # Fake the temperature if it's not known, and eliminate stars we just
        # don't want to deal with.

        discard_class = 0
        discard_dn = 0

        default_star_class = cast(str, config.default_star_class)

        star_list2 = []
        for star in star_list1:
            if star.ra is None or star.dec is None:
                continue
            star.conflicts = None
            star.temperature_faked = False
            star.johnson_mag_faked = False
            star.integrated_dn = 0.
            star.overlay_box_width = 0
            star.overlay_box_thickness = 0
            star.psf_delta_u = None
            star.psf_delta_v = None
            star.psf_sigma_x = None
            star.psf_sigma_y = None
            if star.temperature is None:
                star.temperature_faked = True
                star.temperature = SCLASS_TO_SURFACE_TEMP[default_star_class]
                star.spectral_class = default_star_class
            # TODO Validate this
            if star.johnson_mag_v is None or star.johnson_mag_b is None:
                    star.johnson_mag_v = (star.vmag -
                        SCLASS_TO_B_MINUS_V[clean_sclass(star.spectral_class)] / 2.)
                    star.johnson_mag_b = (star.vmag +
                        SCLASS_TO_B_MINUS_V[clean_sclass(star.spectral_class)] / 2.)
                    star.johnson_mag_faked = True
            star.dn = 2 ** -(star.vmag-4)  # TODO
            # star.dn = compute_dn_from_star(obs, star)
            # if star.dn < min_dn:
            #     discard_dn += 1
            #     continue
    #         if star.spectral_class[0] == "M":
    #             # M stars are too dim and too red to be seen
    #             discard_class += 1
    #             continue
            if config.stellar_aberration:
                self._aberrate_star(star)
            star_list2.append(star)

        # Eliminate stars that are not actually in the FOV, including some
        # margin beyond the edge

        ra_dec_list = [x.ra_dec_with_pm(obs.midtime) for x in star_list2]
        ra_list = polymath.Scalar([x[0] for x in ra_dec_list])
        dec_list = polymath.Scalar([x[1] for x in ra_dec_list])

        uv = obs.uv_from_ra_and_dec(ra_list, dec_list, apparent=True)
        u_list, v_list = uv.to_scalars()
        u_list = u_list.vals
        v_list = v_list.vals

        if radec_movement is not None:
            uv1 = obs.uv_from_ra_and_dec(ra_list - radec_movement[0],
                                         dec_list - radec_movement[1],
                                         apparent=True)
            uv2 = obs.uv_from_ra_and_dec(ra_list + radec_movement[0],
                                         dec_list + radec_movement[1],
                                         apparent=True)
        else:
            uv1 = obs.uv_from_ra_and_dec(ra_list, dec_list, tfrac=0., apparent=True)
            uv2 = obs.uv_from_ra_and_dec(ra_list, dec_list, tfrac=1., apparent=True)
        u1_list, v1_list = uv1.to_scalars()
        u1_list = u1_list.vals
        v1_list = v1_list.vals
        u2_list, v2_list = uv2.to_scalars()
        u2_list = u2_list.vals
        v2_list = v2_list.vals

        star_list3 = []
        discard_uv = 0
        for star, u, v, u1, v1, u2, v2 in zip(star_list2,
                                              u_list, v_list,
                                              u1_list, v1_list,
                                              u2_list, v2_list):
#     # Check for off the edge
#     psf_size = max(_find_photometry_boxsize(star, stars_config),
#                    _find_psf_boxsize(star, stars_config))
#     psf_size_half_u = int(psf_size + np.round(abs(star.move_u))) // 2
#     psf_size_half_v = int(psf_size + np.round(abs(star.move_v))) // 2

#     if (star.u+offset[0] < psf_size_half_u or
#         star.u+offset[0] >= obs.data_shape_xy[0]-psf_size_half_u or
#         star.v+offset[1] < psf_size_half_v or
#         star.v+offset[1] >= obs.data_shape_xy[1]-psf_size_half_v):
#         logger.debug("Star %9d U %8.3f V %8.3f is off the edge",
#                      star.unique_number, star.u, star.v)
#         star.conflicts = "EDGE"
#         return True


            if (v < obs.extfov_v_min or v > obs.extfov_v_max or
                u < obs.extfov_u_min or u > obs.extfov_u_max):
                discard_uv += 1
                continue

            star.u = u
            star.v = v
            star.move_u = u2-u1
            star.move_v = v2-v1

            star_list3.append(star)

        self._logger.debug(
            'Found %d stars, discarded because of CLASS %d, LOW DN %d, BAD UV %d',
            len(star_list3), discard_class, discard_dn, discard_uv)

        return star_list3

    def stars_list_for_obs(self,
                           radec_movement: Optional[tuple[float, float]] = None,
                           **kwargs) -> list[Star]:
        """Return a list of stars in the FOV of the obs.

        Parameters:
            radec_movement: A tuple (dra,ddec) that gives the movement of the camera in
                each half of the exposure. None if no movement is available.
            **kwargs: Passed to find_stars to restrict the types of stars returned.

        Returns: The list of Star objects with additional attributes for each Star:

            - ``.u`` and ``.v``: The U,V coordinate including stellar aberration.
            - ``.faked_temperature``: A bool indicating if the temperature and spectral class
              had to be faked.
            - ``.dn``: The estimated integrated DN count given the star's class,
              magnitude, and the filters being used.
        """

        config = self._config.stars

        max_stars = config.max_stars

        ra_min, ra_max, dec_min, dec_max = self._obs.ra_dec_limits_ext()

        # Try to be efficient by limiting the magnitudes searched so we don't
        # return a huge number of dimmer stars and then only need a few of them.
        magnitude_list = [0., 12., 13., 14., 15., 16., 17.]

        # TODO mag_vmax = _compute_dimmest_visible_star_vmag(obs, stars_config)+1
        mag_vmax = 16

        if radec_movement is None:
            self._logger.debug('Retrieving star list with max detectable VMAG %.4f',
                               mag_vmax)
        else:
            self._logger.debug('Retrieving star list with RA/DEC movement %.5f, %.5f, max '
                            'detectable VMAG %.4f',
                            radec_movement[0], radec_movement[1], mag_vmax)

        full_star_list = []

        for mag_min, mag_max in zip(magnitude_list[:-1], magnitude_list[1:]):
            if mag_min > mag_vmax:
                break
            mag_max = min(mag_max, mag_vmax)

            star_list = self._stars_list_for_obs(ra_min, ra_max, dec_min, dec_max,
                                                 mag_min, mag_max,
                                                 radec_movement,
                                                 **kwargs)
            full_star_list += star_list

            self._logger.debug('New total stars %d', len(full_star_list))

            # Remove stars that are on top of each other and keep the brighter one
            if len(full_star_list) > 1:
                new_full_star_list = []
                for i in range(len(full_star_list)):
                    good_star = True
                    for j in range(len(full_star_list)):
                        if i == j:
                            continue
                        if (abs(full_star_list[i].v - full_star_list[j].v) < 1 and
                            abs(full_star_list[i].u - full_star_list[j].u) < 1 and
                            full_star_list[i].vmag < full_star_list[j].vmag):
                            good_star = False
                            self._logger.debug('Removing overlapping star:')
                            self._logger.debug('  %s; keeping ',
                                               self._star_short_info(full_star_list[i]))
                            self._logger.debug('  %s',
                                               self._star_short_info(full_star_list[j]))
                            break
                    if good_star:
                        new_full_star_list.append(full_star_list[i])

                if len(full_star_list) != len(new_full_star_list):
                    self._logger.debug('After removing overlapping stars, total stars %d',
                                       len(new_full_star_list))

                full_star_list = new_full_star_list

            if len(full_star_list) >= max_stars:
                break

        # Sort the list with the brightest stars first.
        # TODO Was DN
        full_star_list.sort(key=lambda x: x.vmag, reverse=False)

        full_star_list = full_star_list[:max_stars]

        for star in full_star_list:
            # Mark all the bodies (or rings) that are conflicting
            rings_can_conflict = False  # TODO
            self._mark_conflicts_obj(star, rings_can_conflict)

        self._logger.info('Star list (total %d):', len(full_star_list))
        for star in full_star_list:
            self._logger.info('  %s', self._star_short_info(star))

        return full_star_list

#===============================================================================
#
# CREATE A MODEL OR OVERLAY FOR STARS IN THE FOV.
#
#===============================================================================

    def create_model(self,
                     *,
                     ra_dec_predicted: Optional[tuple[float, ...]] = None,
                     ignore_conflicts: bool = False
                     ) -> tuple[NDArrayFloatType | None,
                                dict[str, Any],
                                Annotation | None]:
        """Create a model containing nothing but stars.

        Individual stars are modeled using the PSF specified by the associated Inst class.

        Parameters:
            ignore_conflicts: True to include stars that have a conflict with a body or
                rings.

        Returns: The model.
        """

        metadata: dict[str, Any] = {}

        metadata['start_time'] = time.time()

        with self._logger.open(f'CREATE STARS MODEL'):
            ret = self._create_model(metadata, ra_dec_predicted, ignore_conflicts)

        metadata['end_time'] = time.time()

        return ret

    def _create_model(self,
                      metadata: dict[str, Any],
                      ra_dec_predicted: Optional[tuple[float, ...]],
                      ignore_conflicts
                      ) -> tuple[NDArrayFloatType | None,
                                 dict[str, Any],
                                 Annotation | None]:

        config = self._config.stars

        max_move_steps = config.max_movement_steps

        radec_movement = None

        if ra_dec_predicted is not None:
            radec_movement = (ra_dec_predicted[6] * self._obs.texp/2,
                              ra_dec_predicted[7] * self._obs.texp/2)

        model = self._obs.make_extfov_zeros()

        star_list = self.stars_list_for_obs(radec_movement)
        metadata['star_list'] = star_list

        for star in star_list:
            if star.conflicts and not ignore_conflicts:
                continue
            u_idx = star.u + self._obs.extfov_margin_u
            v_idx = star.v + self._obs.extfov_margin_v
            u_int = int(u_idx)
            v_int = int(v_idx)
            u_frac = u_idx - u_int
            v_frac = v_idx - v_int

            psf_size = 5
            # psf_size = _find_psf_boxsize(star, stars_config)

            psf_size_half_u = int(psf_size + np.round(abs(star.move_u))) // 2
            psf_size_half_v = int(psf_size + np.round(abs(star.move_v))) // 2

            move_gran = max(abs(star.move_u) / max_move_steps,
                            abs(star.move_v) / max_move_steps)
            move_gran = np.clip(move_gran, 0.1, 1.0)

            # sigma = nav.config.PSF_SIGMA[obs.clean_detector]
            # if star.dn >= stars_config["psf_gain"][0]:
            #     sigma *= stars_config["psf_gain"][1]

            psf = self.obs.inst.star_psf()

            if (u_int < psf_size_half_u or
                u_int >= model.shape[1]-psf_size_half_u or
                v_int < psf_size_half_v or
                v_int >= model.shape[0]-psf_size_half_v):
                continue

            psf = psf.eval_rect((psf_size_half_v*2+1, psf_size_half_u*2+1),
                                 offset=(v_frac, u_frac),
                                 scale=star.dn,
                                 movement=(star.move_v, star.move_u),
                                 movement_granularity=move_gran)

            model[v_int-psf_size_half_v:v_int+psf_size_half_v+1,
                  u_int-psf_size_half_u:u_int+psf_size_half_u+1] += psf

        if _DEBUG_STARS_MODEL_IMGDISP:
            ImageDisp([model],
                    canvas_size=(1024,1024),
                    enlarge_limit=10,
                    auto_update=True)
            tk.mainloop()

        # Create the text labels

        text_info_list = []
        star_avoid_mask = self._obs.make_extfov_false()
        star_overlay = self._obs.make_extfov_false()

        stretch_regions = []

        for star in star_list:
            if star.conflicts:
                continue

            # Should NOT be rounded for plotting, since all of coord
            # X to X+0.9999 is the same pixel
            u = int(star.u + self._obs.extfov_margin_u)
            v = int(star.v + self._obs.extfov_margin_v)

            # width = star.overlay_box_width
            # thickness = star.overlay_box_thickness
            width = 5
            # thickness = 1
            # if width == 0:
            #     continue

            # width += thickness-1
            # if (not width <= u_idx < overlay.shape[1]-width or
            #     not width <= v_idx < overlay.shape[0]-width):
            #     continue

            u_min = u - width
            v_min = v - width
            u_max = u + width
            v_max = v + width
            u_min, v_min = self._obs.clip_extfov(u_min, v_min)
            u_max, v_max = self._obs.clip_extfov(u_max, v_max)

            star_avoid_mask[v_min:v_max+1, u_min:u_max+1] = True
            draw_rect(star_overlay, True, u, v, width, width)

            stretch_region = self._obs.make_extfov_false()
            stretch_region[v_min:v_max+1, u_min:u_max+1] = True
            compressed_stretch_region = np.packbits(stretch_region, axis=0)
            stretch_regions.append(compressed_stretch_region)

            star_str1 = None
            try:
                star_str1 = star.name[:10]
            except AttributeError:
                pass

            if star_str1 is None or star_str1 == "":
                star_str1 = f'{star.unique_number:09d}'

            star_str2 = f'{star.vmag:.3f} {clean_sclass(star.spectral_class)}'

            text_loc = []

            label_margin = width + 3

            text_loc.append((TEXTINFO_BOTTOM, v + label_margin, u))
            text_loc.append((TEXTINFO_TOP, v - label_margin, u))
            text_loc.append((TEXTINFO_LEFT, v, u - label_margin))
            text_loc.append((TEXTINFO_RIGHT, v, u + label_margin))

            text_info = AnnotationTextInfo(f'{star_str1}\n{star_str2}',
                                           ref_vu=(v, u),
                                           text_loc=text_loc,
                                           font=config.label_font,
                                           font_size=config.label_font_size,
                                           color=config.label_font_color)
            text_info_list.append(text_info)

        annotation = Annotation(self.obs, star_overlay, config.label_star_color,
                                thicken_overlay=0,
                                avoid_mask=star_avoid_mask,
                                text_info=text_info_list)
        annotations = Annotations()
        annotations.add_annotations(annotation)

        self._model_img = model
        self._model_mask = None
        self._range = np.inf
        self._uncertainty = 0.
        self._stretch_regions = stretch_regions
        self._annotations = annotations
        self._metadata = metadata

        self._logger.debug(f'  Star model min: {np.min(self._model_img)}, max: {np.max(self._model_img)}')

    def stars_make_good_bad_overlay(self,
                                    obs: Observation,
                                    star_list: list[Star],
                                    offset: tuple[float, float],
                                    *,
                                    use_extfov: bool = True,
                                    show_streaks: bool = False,
                                    label_avoid_mask: Optional[NDArrayFloatType] = None,
                                    stars_config: Optional[dict[str, Any]] = None) -> tuple[NDArrayFloatType, NDArrayFloatType]:
        """Create an overlay with high and low confidence stars marked.

        Parameters:
            obs: The observation.
            star_list: The list of Star objects.
            offset: The amount to offset a star's position in the (U,V) directions.
            use_extfov: If True, use the extfov, otherwise use the normal fov size.
            show_streaks: If True, draw the streak from the star's PSF in addition to the box or
                circle.
            label_avoid_mask: A mask giving places where text labels should not be placed (i.e.
                labels from another model are already there). None if no mask.
            stars_config: Configuration parameters.

        Returns: The overlay.

            - Star excluded by brightness or conflict: circle
            - Star bad photometry: thin square
            - Star good photometry: thick square
        """
        if stars_config is None:
            stars_config = self._config.stars

        offset_u = 0
        offset_v = 0
        if offset is not None:
            offset_u, offset_v = offset

        if use_extfov:
            overlay = obs.make_extfov_zeros(dtype=np.uint8)
            text = obs.make_extfov_zeros(dtype=np.uint8)
        else:
            overlay = obs.make_fov_zeros(dtype=np.uint8)
            text = obs.make_fov_zeros(dtype=np.uint8)
        text_im = Image.frombuffer("L", (text.shape[1], text.shape[0]), text,
                                "raw", "L", 0, 1)
        text_draw = ImageDraw.Draw(text_im)
        font = None
        extfov_margin = (obs.extfov_margin if use_extfov else (0,0))
        if stars_config["font"] is not None:
            if isinstance(stars_config['font'], dict):
                font_info = stars_config["font"][obs.data_shape_xy[1]]
            else:
                font_info = stars_config['font']
            font = ImageFont.truetype(font_info[0], font_info[1])

        if show_streaks:
            max_move_steps = stars_config["max_movement_steps"]

            for star in star_list:
                u_idx = star.u+offset_u+extfov_margin[0]
                v_idx = star.v+offset_v+extfov_margin[1]
                u_int = int(u_idx)
                v_int = int(v_idx)
                u_frac = u_idx-u_int
                v_frac = v_idx-v_int

                psf_size = _find_psf_boxsize(star, stars_config)

                psf_size_half_u = int(psf_size + np.round(abs(star.move_u))) // 2
                psf_size_half_v = int(psf_size + np.round(abs(star.move_v))) // 2

                move_gran = max(abs(star.move_u)/max_move_steps,
                                abs(star.move_v)/max_move_steps)
                move_gran = np.clip(move_gran, 0.1, 1.0)

                gausspsf = GaussianPSF(sigma=nav.config.PSF_SIGMA[obs.clean_detector],
                                    movement=(star.move_v,star.move_u),
                                    movement_granularity=move_gran)

                if (u_int < psf_size_half_u or
                    u_int >= overlay.shape[1]-psf_size_half_u or
                    v_int < psf_size_half_v or
                    v_int >= overlay.shape[0]-psf_size_half_v):
                    continue

                psf = gausspsf.eval_rect((psf_size_half_v*2+1,psf_size_half_u*2+1),
                                        offset=(v_frac,u_frac),
                                        scale=1.)
                psf = psf / np.max(psf) * 255
                psf = psf.astype("uint8")

                overlay[v_int-psf_size_half_v:v_int+psf_size_half_v+1,
                        u_int-psf_size_half_u:u_int+psf_size_half_u+1] += psf

        # First go through and draw the circles and squares. We have to put the
        # squares in the right place. There"s no way to avoid anything in
        # the label_avoid_mask.
        for star in star_list:
            # Should NOT be rounded for plotting, since all of coord
            # X to X+0.9999 is the same pixel
            u_idx = int(star.u+offset_u+extfov_margin[0])
            v_idx = int(star.v+offset_v+extfov_margin[1])
            star.overlay_box_width = 0
            star.overlay_box_thickness = 0
            if not star.conflicts or star.conflicts == "RINGS":
                if not star.is_bright_enough or not star.is_dim_enough:
                    width = 3
                    if (width < u_idx < overlay.shape[1]-width and
                        width < v_idx < overlay.shape[0]-width):
                        star.overlay_box_width = width
                        star.overlay_box_thickness = 1
                        draw_circle(overlay, u_idx, v_idx, width, 255)
                else:
                    if star.integrated_dn == 0:
                        width = 3
                    else:
                        width = int(star.photometry_box_size // 2) + 1
                    thickness = 1
                    if star.photometry_confidence >= stars_config["min_confidence"]:
                        thickness = 3
                    if (width+thickness-1 <= u_idx <
                        overlay.shape[1]-width-thickness+1 and
                        width+thickness-1 <= v_idx <
                        overlay.shape[0]-width-thickness+1):
                        star.overlay_box_width = width
                        star.overlay_box_thickness = thickness
                        draw_rect(overlay, u_idx, v_idx,
                                width, width, 255, thickness)

        # # Now go through a second time to do the text labels. This way the labels
        # # can avoid overlapping with the squares.
        # for star in star_list:
        #     # Should NOT be rounded for plotting, since all of coord
        #     # X to X+0.9999 is the same pixel
        #     u_idx = int(star.u+offset_u+extfov_margin[0])
        #     v_idx = int(star.v+offset_v+extfov_margin[1])

        #     width = star.overlay_box_width
        #     thickness = star.overlay_box_thickness
        #     if width == 0:
        #         continue

        #     width += thickness-1
        #     if (not width <= u_idx < overlay.shape[1]-width or
        #         not width <= v_idx < overlay.shape[0]-width):
        #         continue

        #     star_str1 = None
        #     try:
        #         star_str1 = star.name[:10]
        #     except AttributeError:
        #         pass

        #     if star_str1 is None or star_str1 == "":
        #         star_str1 = "%09d" % (star.unique_number)
        #     star_str2 = "%.3f %s" % (star.vmag, clean_sclass(star.spectral_class))
        #     text_size = text_draw.textbbox((0,0), star_str1, font=font)[2:]
        #     locations = []
        #     v_text = v_idx-text_size[1] # Whole size because we're doing two lines
        #     if u_idx >= overlay.shape[1]//2:
        #         # Star is on right side of image - default to label on left
        #         if v_text+text_size[1]*2 < overlay.shape[0]:
        #             locations.append((u_idx-width-6-text_size[0], v_text))
        #         if v_text >= 0:
        #             locations.append((u_idx+width+6, v_text))
        #     else:
        #         # Star is on left side of image - default to label on right
        #         if v_text >= 0:
        #             locations.append((u_idx+width+6, v_text))
        #         if v_text+text_size[1]*2 < overlay.shape[0]:
        #             locations.append((u_idx-width-6-text_size[0], v_text))
        #     # Next try below star
        #     u_text = u_idx-text_size[0]//2
        #     v_text = v_idx+width+6
        #     if v_text+text_size[1]*2 < overlay.shape[0]:
        #         locations.append((u_text, v_text))
        #     # And above the star
        #     v_text = v_idx-width-3-text_size[1]*2
        #     if v_text >= 0:
        #         locations.append((u_text, v_text))
        #     # One last gasp effort...try a little further above or below
        #     u_text = u_idx-text_size[0]//2
        #     v_text = v_idx+width+12
        #     if v_text+text_size[1]*2 < overlay.shape[0]:
        #         locations.append((u_text, v_text))
        #     v_text = v_idx-width-12-text_size[1]
        #     if v_text >= 0:
        #         locations.append((u_text, v_text))
        #     # And to the side but further above
        #     v_text = v_idx-text_size[1]-6
        #     if v_text+text_size[1]*2 < overlay.shape[0]:
        #         locations.append((u_idx-width-6-text_size[0], v_text))
        #     if v_text >= 0:
        #         locations.append((u_idx+width+6, v_text))
        #     # And to the side but further below
        #     v_text = v_idx-text_size[1]+6
        #     if v_text+text_size[1]*2 < overlay.shape[0]:
        #         locations.append((u_idx-width-6-text_size[0], v_text))
        #     if v_text >= 0:
        #         locations.append((u_idx+width+6, v_text))

        #     good_u = None
        #     good_v = None
        #     preferred_u = None
        #     preferred_v = None
        #     text = np.array(text_im.getdata()).reshape(text.shape)
        #     for u, v in locations:
        #         if (not np.any(
        #                 text[
        #                 max(v-3,0):
        #                 min(v+text_size[1]*2+3, overlay.shape[0]),
        #                 max(u-3,0):
        #                 min(u+text_size[0]+3, overlay.shape[1])
        #                 ]) and
        #             (label_avoid_mask is None or
        #             not np.any(
        #                 label_avoid_mask[
        #                 max(v-3,0):
        #                 min(v+text_size[1]*2+3, overlay.shape[0]),
        #                 max(u-3,0):
        #                 min(u+text_size[0]+3, overlay.shape[1])
        #                 ]))):
        #             if good_u is None:
        #                 # Give precedence to earlier choices - they"re prettier
        #                 good_u = u
        #                 good_v = v
        #             # But we"d really rather the text not overlap with the squares
        #             # either, if possible
        #             if (preferred_u is None and not np.any(
        #             overlay[max(v-3,0):
        #                     min(v+text_size[1]*2+3, overlay.shape[0]),
        #                     max(u-3,0):
        #                     min(u+text_size[0]+3, overlay.shape[1])])):
        #                 preferred_u = u
        #                 preferred_v = v
        #                 break

        #     if preferred_u is not None:
        #         good_u = preferred_u
        #         good_v = preferred_v

        #     if good_u is not None:
        #         text_draw.text((good_u,good_v), star_str1,
        #                     fill=255, font=font)
        #         text_draw.text((good_u,good_v+text_size[1]), star_str2,
        #                     fill=255, font=font)

        # text = np.array(text_im.getdata()).astype("uint8").reshape(text.shape)

        return overlay, text

#===============================================================================
#
# PERFORM PHOTOMETRY.
#
#===============================================================================

# def _trust_star_dn(obs, star, stars_config):
#     dn_ok = star.dn < stars_config["max_star_dn"]
#     # filter_ok = obs.filter1 == "CL1" and obs.filter2 == "CL2"
#     return dn_ok #and filter_ok

#def _stars_perform_photometry(obs, calib_data, star, offset,
#                              extfov_margin, stars_config):
#    """Perform photometry on a single star.
#
#    See star_perform_photometry for full details.
#    """
#    # calib_data is calibrated in DN and extended
#    u = int(np.round(star.u)) + offset[0] + extfov_margin[0]
#    v = int(np.round(star.v)) + offset[1] + extfov_margin[1]
#
#    if star.dn > stars_config["photometry_boxsize_1"][0]:
#        boxsize = stars_config["photometry_boxsize_1"][1]
#    elif star.dn > stars_config["photometry_boxsize_2"][0]:
#        boxsize = stars_config["photometry_boxsize_2"][1]
#    else:
#        boxsize = stars_config["photometry_boxsize_default"]
#
#    star.photometry_box_width = boxsize
#
#    box_halfsize = boxsize // 2
#
#    # Don't process stars that are off the edge of the real (not extended)
#    # data.
#    if (u-extfov_margin[0] < box_halfsize or
#        u-extfov_margin[0] > calib_data.shape[1]-2*extfov_margin[0]-box_halfsize-1 or
#        v-extfov_margin[1] < box_halfsize or
#        v-extfov_margin[1] > calib_data.shape[0]-2*extfov_margin[1]-box_halfsize-1):
#        return None
#
#    subimage = calib_data[v-box_halfsize:v+box_halfsize+1,
#                          u-box_halfsize:u+box_halfsize+1]
#    subimage = subimage.view(ma.MaskedArray)
#    subimage[1:-1, 1:-1] = ma.masked # Mask out the center
#
#    bkgnd = ma.mean(subimage)
#    bkgnd_std = ma.std(subimage)
#
#    subimage.mask = ~subimage.mask # Mask out the edge
#    integrated_dn = np.sum(subimage-bkgnd)
#    integrated_std = np.std(subimage-bkgnd)
#
#    return integrated_dn, bkgnd, integrated_std, bkgnd_std

# def _find_photometry_boxsize(star, stars_config):
#     for dn, size in stars_config["photometry_boxsizes"]:
#         if star.dn >= dn:
#             return size
#     return size # Default to smallest

# def _find_psf_boxsize(star, stars_config):
#     for dn, size in stars_config["psf_boxsizes"]:
#         if star.dn >= dn:
#             return size
#     return size # Default to smallest

# def _stars_perform_photometry(obs, calib_data, star, offset,
#                               stars_config):
#     """Perform photometry on a single star.

#     See star_perform_photometry for full details.
#     """
#     logger = logging.getLogger(_LOGGING_NAME+"._stars_perform_photometry")
#     _debug_box_sizes = False
#     # calib_data is calibrated in DN and extended
#     u = int(np.round(star.u)) + offset[0] + obs.extfov_margin[0]
#     v = int(np.round(star.v)) + offset[1] + obs.extfov_margin[1]

#     boxsize = _find_photometry_boxsize(star, stars_config)

#     boxsize /= (1024 / obs.data_shape_xy[1]) # Keep same angular size
#     boxsize = max(boxsize, 5)

#     boxsizes = [boxsize]
#     if _debug_box_sizes:
#         boxsizes = list(range(3,21,2))

#     for boxsize in boxsizes:
#         star.photometry_box_size = boxsize

#         # Expand the PSF size by the size of the streak
#         psf_size_half_u = int(boxsize + np.round(abs(star.move_u))) // 2
#         psf_size_half_v = int(boxsize + np.round(abs(star.move_v))) // 2

#         # Don't process stars that are off the edge of the real (not extended)
#         # data.
#         if (u-obs.extfov_margin[0] < psf_size_half_u or
#             u-obs.extfov_margin[0] > calib_data.shape[1]-2*obs.extfov_margin[0]-psf_size_half_u-1 or
#             v-obs.extfov_margin[1] < psf_size_half_v or
#             v-obs.extfov_margin[1] > calib_data.shape[0]-2*obs.extfov_margin[1]-psf_size_half_v-1):
#             return None

#         max_move_steps = stars_config["max_movement_steps"]
#         move_gran = max(abs(star.move_u)/max_move_steps,
#                         abs(star.move_v)/max_move_steps)
#         move_gran = np.clip(move_gran, 0.1, 1.0)

#         # Make a nice Gaussian rectangle of the extended PSF size including
#         # motion
#         gausspsf = GaussianPSF(sigma=nav.config.PSF_SIGMA[obs.clean_detector],
#                                movement=(star.move_v,star.move_u),
#                                movement_granularity=move_gran)

#         psf = gausspsf.eval_rect((psf_size_half_v*2+1,psf_size_half_u*2+1),
#                                  offset=(0.5,0.5))

#         center_u = int(psf.shape[1] // 2)
#         center_v = int(psf.shape[0] // 2)
#         boxsize_2 = int(boxsize//2)

#         # Anything outside of the streak will be zero, so find the minimum
#         # value inside the original box size (minus a one pixel border)
#         # as a way of figuring out where the streak is
#         subpsf = psf[center_v-boxsize_2+1:center_v+boxsize_2,
#                      center_u-boxsize_2+1:center_u+boxsize_2]
#         # We don't allow zero here because it confuses the logic below
#         min_allowed_val = max(np.min(subpsf), 1e-6)

#         # Now do the same thing to a slightly larger box size giving us
#         # the minimum value including the border
#         subpsf = psf[center_v-boxsize_2:center_v+boxsize_2+1,
#                      center_u-boxsize_2:center_u+boxsize_2+1]
#         min_bkgnd_val = np.min(subpsf)

#         # Extract the whole big rectangle from the image
#         subimage = calib_data[v-psf_size_half_v:v+psf_size_half_v+1,
#                               u-psf_size_half_u:u+psf_size_half_u+1]

#         # plot3d(subimage, title="Sub image")

#         # Find the part of the image we care about and extract it
#         streak_bool = psf >= min_allowed_val
#         streak_data = subimage[streak_bool]

#         # Then find the background region and extract it
#         bkgnd_bool = np.logical_and(filt.maximum_filter(streak_bool, 3),
#                                     np.logical_not(streak_bool))
#         bkgnd_data = subimage[bkgnd_bool]

#         bkgnd = ma.mean(bkgnd_data)
#         bkgnd_std = ma.std(bkgnd_data)

#         integrated_dn = np.sum(streak_data-bkgnd)
#         integrated_std = np.std(streak_data-bkgnd)

#         if _debug_box_sizes:
#             logger.debug("Star %9d %2s UV %4d %4d BOX %d PRED %7.2f MEAS %7.2f"+
#                          " BKGND %9.4f BKGND_STD %9.4f",
#                          star.unique_number, clean_sclass(star.spectral_class),
#                          u, v, boxsize, star.dn, integrated_dn, bkgnd, bkgnd_std)

#     return integrated_dn, bkgnd, integrated_std, bkgnd_std

# def stars_perform_photometry(obs, calib_data, star_list, offset=None, stars_config=None):
#     """Perform photometry on a list of stars.

#     Inputs:
#         obs                The observation.
#         calib_data         obs.data calibrated as DN.
#         star_list          The list of Star objects.
#         offset             The amount to offset a star"s position in the (U,V)
#                            directions.
#         stars_config       Configuration parameters.

#     Returns:
#         good_stars         The number of good stars.
#                            (star.photometry_confidence >= STARS_MIN_CONFIDENCE)

#         Each Star is populated with:

#             .integrated_dn            The actual integrated dn measured.
#             .photometry_confidence    The confidence in the result. Currently
#                                       adds 0.5 for a non-noisy background and
#                                       adds 0.5 for the DN within range.
#     """
#     logger = logging.getLogger(_LOGGING_NAME+".stars_perform_photometry")

#     if stars_config is None:
#         stars_config = nav.config.STARS_DEFAULT_CONFIG

#     offset_u = 0
#     offset_v = 0
#     if offset is not None:
#         offset_u, offset_v = offset

#     image = obs.data
#     min_dn = stars_config[("min_detectable_dn", obs.clean_detector)]
#     photometry_slop = stars_config[("photometry_slop", obs.clean_detector)]

#     for star in star_list:
#         u = int(np.round(star.u)) + offset_u
#         v = int(np.round(star.v)) + offset_v
#         if star.conflicts:
#             # Stars that conflict with bodies are ignored
#             star.integrated_dn = 0.
#             star.photometry_confidence = 0.
#             logger.debug("Star %9d %2s UV %4d %4d IGNORED %s",
#                          star.unique_number, clean_sclass(star.spectral_class),
#                          u, v, star.conflicts)
#             continue
#         ret = _stars_perform_photometry(obs, calib_data, star,
#                                         (offset_u, offset_v), stars_config)
#         if ret is None:
#             integrated_dn = 0.
#             confidence = 0.
#         else:
#             integrated_dn, bkgnd, integrated_std, bkgnd_std = ret
#             if (star.johnson_mag_faked or
#                 not _trust_star_dn(obs, star, stars_config)):
#                 # Really the only thing we can do here is see if we detected
#                 # something at all, because we can't trust the photometry
#                 if integrated_dn < 0:
#                     confidence = 0.
#                 else:
#                     confidence = ((integrated_dn >= min_dn)*0.5 +
#                                   (integrated_std >= bkgnd_std*2)*0.5)
#             else:
#                 confidence = ((star.dn/photometry_slop <
#                                  integrated_dn <
#                                star.dn*photometry_slop)*0.5 +
#                               (integrated_std >= bkgnd_std*1.5)*0.5)

#         star.integrated_dn = integrated_dn
#         star.photometry_confidence = confidence

#         logger.debug(
#             "Star %9d %2s UV %4d %4d PRED %7.2f MEAS %7.2f CONF %4.2f",
#             star.unique_number, clean_sclass(star.spectral_class),
#             u, v, star.dn, star.integrated_dn, star.photometry_confidence)

#     good_stars = 0
#     for star in star_list:
#         if star.photometry_confidence >= stars_config["min_confidence"]:
#             good_stars += 1

#     return good_stars


#===============================================================================
#
# FIND THE IMAGE OFFSET BASED ON STARS.
#
#===============================================================================

    def _mark_conflicts_obj(self,
                            star: Star,
                            rings_can_conflict: bool) -> None:
        """Check if a star conflicts with known bodies or rings.

        Sets star.conflicts to a string describing why the Star conflicted.

        Returns True if the star conflicted, False if the star didn't.
        """

        obs = self._obs
        config = self._config

        if self._conflict_body_list is None:
            self._conflict_body_list = ([obs.closest_planet] +
                                        config.satellites(obs.closest_planet))

        # Create a Meshgrid for the area around the star. Give slop on each side - we
        # don't want a star to even be close to a large object.
        star_slop = config.stars.body_conflict_margin
        meshgrid = Meshgrid.for_fov(obs.fov,
                                    origin=(star.u-star_slop,
                                            star.v-star_slop),
                                    limit =(star.u+star_slop,
                                            star.v+star_slop))
        backplane = Backplane(obs, meshgrid)

        # Check for planet and moons
        for body_name in self._conflict_body_list:
            intercepted = backplane.where_intercepted(body_name)
            if intercepted.any():
                self._logger.debug(f'Star {star.unique_number:9d} U {star.u:8.3f} V '
                                   f'{star.v:8.3f} conflicts with {body_name}')
                star.conflicts = f'BODY: {body_name}'
                return True

        # Check for rings
        # if rings_can_conflict:
        #     ring_radius = obs.ext_bp.ring_radius("saturn:ring").mvals.astype("float")
        #     ring_longitude = (obs.ext_bp.ring_longitude("saturn:ring").vals.
        #                     astype("float"))
        #     rad = ring_radius[int(star.v+obs.extfov_margin[1]),
        #                     int(star.u+obs.extfov_margin[0])]
        #     long = ring_longitude[int(star.v+obs.extfov_margin[1]),
        #                         int(star.u+obs.extfov_margin[0])]

        #     # TODO We might want to improve this to support the known position of the
        #     # F ring core.
        #     # C to A rings and F ring
        #     if ((oops.body.SATURN_C_RING[0] <= rad <= oops.body.SATURN_A_RING[1]) or
        #         (139890 <= rad <= 140550)): # F ring
        #         logger.debug("Star %9d U %8.3f V %8.3f conflicts with rings radius "
        #                     "%.1f",
        #                     star.unique_number, star.u, star.v, rad)
        #         star.conflicts = "RINGS"
        #         return True

        return False

# def _stars_optimize_offset_list(offset_list, tolerance=1):
#     """Remove bad offsets.

#     A bad offset is defined as an offset that makes a line with two other
#     offsets in the list. We remove these because when 2-D correlation is
#     performed on certain times of images, there is a "line" of correlation
#     peaks through the image, none of which are actually correct. When we
#     are finding a limited number of peaks, they all get eaten up by this
#     line and we never look elsewhere.
#     """
#     logger = logging.getLogger(_LOGGING_NAME+"._stars_optimize_offset_list")

#     mark_for_deletion = [False] * len(offset_list)
#     for idx1 in range(len(offset_list)-2):
#         for idx2 in range(idx1+1,len(offset_list)-1):
#             for idx3 in range(idx2+1,len(offset_list)):
#                 u1 = offset_list[idx1][0][0]
#                 u2 = offset_list[idx2][0][0]
#                 u3 = offset_list[idx3][0][0]
#                 v1 = offset_list[idx1][0][1]
#                 v2 = offset_list[idx2][0][1]
#                 v3 = offset_list[idx3][0][1]
#                 if (u1 is None or u2 is None or u3 is None or
#                     v1 is None or v2 is None or v3 is None):
#                     continue
#                 if u1 == u2: # Vertical line
#                     if abs(u3-u1) <= tolerance:
# #                         logger.debug("Points %d (%d,%d) %d (%d,%d) %d (%d,%d) "+
# #                                      "in a line",
# #                                      idx1+1, u1, v1,
# #                                      idx2+1, u2, v2,
# #                                      idx3+1, u3, v3)
#                         mark_for_deletion[idx2] = True
#                         mark_for_deletion[idx3] = True
#                 else:
#                     slope = float(v1-v2)/float(u1-u2)
#                     if abs(slope) < 0.5:
#                         v_intercept = slope * (u3-u1) + v1
#                         diff = abs(v3-v_intercept)
#                     else:
#                         u_intercept = 1/slope * (v3-v1) + u1
#                         diff = abs(u3-u_intercept)
#                     if diff <= tolerance:
# #                         logger.debug("Points %d (%d,%d) %d (%d,%d) %d (%d,%d) "+
# #                                      "in a line",
# #                                      idx1+1, u1, v1,
# #                                      idx2+1, u2, v2,
# #                                      idx3+1, u3, v3)
#                         mark_for_deletion[idx2] = True
#                         mark_for_deletion[idx3] = True
#     new_offset_list = []
#     for i in range(len(offset_list)):
#         if not mark_for_deletion[i]:
#             new_offset_list.append(offset_list[i])

#     return new_offset_list

# def _stars_find_offset(obs, filtered_data, star_list, min_stars,
#                        search_multiplier, max_offsets, already_tried,
#                        debug_level, perform_photometry,
#                        rings_can_conflict,
#                        radec_movement, stars_config):
#     """Internal helper for stars_find_offset so the loops don't get too deep.

#     Returns:
#         (offset, good_stars, corr, keep_searching, no_peaks)

#         offset            The offset if found, otherwise None.
#         good_stars        If the offset is found, the number of good stars.
#         corr_val          The correlation value for this offset.
#         corr_details      The correlation details (corr, u, v) for this offset.
#         keep_searching    Even if an offset was found, we don't entirely
#                           trust it, so add it to the list and keep searching.
#         no_peaks          The correlation utterly failed and there are no
#                           peaks. There"s no point in continuing to look.
#     """
#     # 1) Find an offset
#     # 2) Remove any stars that are on top of a moon, planet, or opaque part of
#     #    the rings
#     # 3) Repeat until convergence

#     logger = logging.getLogger(_LOGGING_NAME+"._stars_find_offset")

#     min_brightness_guaranteed_vis = stars_config["min_brightness_guaranteed_vis"]
#     min_confidence = stars_config["min_confidence"]

#     # Restrict the search size
#     search_size_max_u, search_size_max_v = obs.extfov_margin
#     limit_size_max_u, limit_size_max_v = obs.search_limits
#     search_size_max_u = int(search_size_max_u*search_multiplier)
#     search_size_max_v = int(search_size_max_v*search_multiplier)

#     for star in star_list:
#         if star.conflicts == "EDGE":
#             star.conflicts = None

#     peak_margin = 3 # Amount on each side of a correlation peak to black out
#     # Make sure we have peaks that can cover 2 complete "lines" in the
#     # correlation
#     if perform_photometry:
#         trial_max_offsets = (max(2*search_size_max_u+1,
#                                  2*search_size_max_v+1) //
#                              (peak_margin*2+1)) + 4
#     else:
#         # No point in doing more than one offset if we"re not going to do
#         # photometry
#         trial_max_offsets = 1

#     # Find the best offset using the current star list.
#     # Then look to see if any of the stars correspond to known
#     # objects like moons, planets, or opaque parts of the ring.
#     # If so, get rid of those stars and iterate.

#     model = stars_create_model(obs, star_list, stars_config=stars_config)

#     offset_list = find_correlation_and_offset(
#                     filtered_data, model, search_size_min=0,
#                     search_size_max=(search_size_max_u, search_size_max_v),
#                     max_offsets=trial_max_offsets,
#                     extfov_margin=obs.extfov_margin)

#     offset_list = _stars_optimize_offset_list(offset_list)
#     offset_list = offset_list[:max_offsets]

#     new_offset_list = []
#     new_peak_list = []
#     new_corr_details_list = []
#     for i in range(len(offset_list)):
#         if (abs(offset_list[i][0][0]) > limit_size_max_u or
#             abs(offset_list[i][0][1]) > limit_size_max_v):
#             logger.debug("Offset %d,%d is outside allowable limits",
#                          offset_list[i][0][0], offset_list[i][0][1])
#         elif offset_list[i][0] not in already_tried:
#             new_offset_list.append(offset_list[i][0])
#             new_peak_list.append(offset_list[i][1])
#             new_corr_details_list.append(offset_list[i][2])
#             # Nobody else gets to try these before we do
#             already_tried.append(offset_list[i][0])
#         else:
#             logger.debug("Offset %d,%d already tried (or reserved)",
#                          offset_list[i][0][0], offset_list[i][0][1])

#     if len(new_offset_list):
#         logger.debug("Final peak list:")
#         for i in range(len(new_offset_list)):
#             logger.debug("Peak %d U,V %d,%d VAL %f", i+1,
#                          new_offset_list[i][0], new_offset_list[i][1],
#                          new_peak_list[i])

#     if len(new_offset_list) == 0:
#         # No peaks found at all - tell the top-level loop there"s no point
#         # in trying more
#         return None, None, None, None, False, True

#     for peak_num in range(len(new_offset_list)):
#         #
#         #            *** LEVEL 4+n ***
#         #
#         # At this level we actually find the offsets for the set of stars
#         # previously restricted and the restricted search space.
#         #
#         # If one of those offsets gives good photometry, we"re done.
#         #
#         # Otherwise, we mark conflicting stars and recurse with a new list
#         # of non-conflicting stars.
#         #
#         offset = new_offset_list[peak_num]
#         peak = new_peak_list[peak_num]
#         corr_details = new_corr_details_list[peak_num]

#         logger.debug("** LEVEL %d: Peak %d - Trial offset U,V %d,%d",
#                      debug_level, peak_num+1, offset[0], offset[1])

#         # First try the star list as given to us.

#         something_conflicted = False
#         for star in star_list:
#             if star.conflicts == "EDGE" or star.conflicts is None:
#                 star.conflicts = None
#                 res = _stars_mark_conflicts_edge(obs, star, offset,
#                                                  stars_config)
#                 something_conflicted = something_conflicted or res

#         if not perform_photometry:
#             good_stars = 0
#             for star in star_list:
#                 star.integrated_dn = 0.
#                 if star.conflicts:
#                     star.photometry_confidence = 0.
#                 else:
#                     star.photometry_confidence = 1.
#                     good_stars += 1
#             logger.debug("Photometry NOT performed")
#             photometry_str = "WITHOUT"
#         else:
#             good_stars = stars_perform_photometry(obs,
#                                                   obs.calib_dn_extdata,
#                                                   star_list,
#                                                   offset=offset,
#                                                   stars_config=stars_config)
#             logger.debug("Photometry found %d good stars", good_stars)
#             photometry_str = "with"

#         # We have to see at least 2/3 of the really bright stars to
#         # fully believe the result. If we don't see this many, it
#         # triggers some more aggressive searching.
#         bright_stars = 0
#         seen_bright_stars = 0
#         for star in star_list:
#             if (star.dn >= min_brightness_guaranteed_vis and
#                 not star.conflicts and
#                 star.integrated_dn != 0.): # Photometry failed if == 0.
#                 bright_stars += 1
#                 if star.photometry_confidence >= min_confidence:
#                     seen_bright_stars += 1
#         if good_stars >= min_stars:
#             if bright_stars > 0 and seen_bright_stars < bright_stars*2//3:
#                 logger.info("***** Enough good stars (%s photometry), "+
#                             "but only saw %d "+
#                             "out of %d bright stars - possibly bad "+
#                             "offset U,V %d,%d",
#                             photometry_str,
#                             seen_bright_stars, bright_stars,
#                             offset[0], offset[1])
#                 # Return True so the top-level loop keeps searching
#                 return offset, good_stars, peak, corr_details, True, False
#             logger.info("***** Enough good stars (%s photometry) - "+
#                         "final offset U,V %d,%d",
#                         photometry_str,
#                         offset[0], offset[1])
#             # Return False so the top-level loop gives up
#             return offset, good_stars, peak, corr_details, False, False

#         if not something_conflicted:
#             # No point in trying again - we"d just have the same stars!
#             logger.debug("No stars off edge and photometry failed - "+
#                          "continuing to next peak")
#             continue

#         # Get rid of the conflicting stars and recurse until there are no
#         # conflicts

#         # Create the current non-conflicting star list
#         non_conf_star_list = [x for x in star_list if not x.conflicts]

#         logger.debug("After conflict - # stars %d", len(non_conf_star_list))

#         if len(non_conf_star_list) < min_stars:
#             logger.debug("Fewer than %d stars left (%d)", min_stars,
#                          len(non_conf_star_list))
#             continue

#         # And recurse using this limited star list
#         ret = _stars_find_offset(obs, filtered_data, non_conf_star_list,
#                                  min_stars, search_multiplier,
#                                  max_offsets, already_tried,
#                                  debug_level+1, perform_photometry,
#                                  rings_can_conflict, radec_movement,
#                                  stars_config)
#         if ret[0] is not None:
#             return ret
#         # We know that everything in non_conf_star_list is not
#         # conflicting at this level, but they were probably mutated by
#         # _stars_find_offset, so reset them
#         for star in non_conf_star_list:
#             if star.conflicts == "EDGE":
#                 star.conflicts = False

#     logger.debug("Exhausted all peaks - No offset found")

#     return None, None, None, None, False, False

# def _stars_refine_offset(obs, calib_data, star_list, offset,
#                          stars_config):
#     """Perform astrometry to refine the final offset."""
#     logger = logging.getLogger(_LOGGING_NAME+"._stars_refine_offset")
#
#     logger.info("Refining star fit:")
#
#     float_psf_sigma = stars_config["float_psf_sigma"]
#     if float_psf_sigma:
#         master_sigma = None
#         sx_list = []
#         sy_list = []
#     else:
#         master_sigma = nav.config.PSF_SIGMA[obs.clean_detector]
#
#     u_list = []
#     v_list = []
#     delta_u_list = []
#     delta_v_list = []
#     for star in star_list:
#         if star.conflicts:
#             continue
#         if star.photometry_confidence < stars_config["min_confidence"]:
#             continue
#         if abs(star.move_u) > 1 or abs(star.move_v) > 1:
#             logger.info("Aborting refine fit due to excessive streaking")
#             return (offset[0], offset[1]), None
#         u = star.u + offset[0]
#         v = star.v + offset[1]
#
#         sigma = master_sigma
#         search_limit = (1.5, 1.5)
#         if star.dn > stars_config["psf_boxsizes"][0][0]:
#             psf_size = stars_config["psf_boxsizes"][0][1]
#             # In the case of really bright stars, we let sigma float and expand
#             # the search range a lot
#             s = psf_size//2-1.5
#             search_limit = (s,s)
#             sigma = None
#         else:
#             psf_size = _find_psf_boxsize(star, stars_config)
#
#         psf_size_u = int(psf_size + np.round(abs(star.move_u)))
#         psf_size_u = (psf_size_u // 2) * 2 + 1
#         psf_size_v = int(psf_size + np.round(abs(star.move_v)))
#         psf_size_v = (psf_size_v // 2) * 2 + 1
#         gausspsf = GaussianPSF(sigma=sigma,
#                                movement=(star.move_v,star.move_u))
#         ret = gausspsf.find_position(calib_data, (psf_size_v,psf_size_u),
#                       (v,u), search_limit=search_limit,
#                       bkgnd_degree=2,
#                       bkgnd_ignore_center=(2,2),
#                       bkgnd_num_sigma=5,
#                       tolerance=1e-5, num_sigma=10,
#                       max_bad_frac=0.2,
#                       allow_nonzero_base=True)
#         if ret is None:
#             logger.info("Star %9d UV %7.2f %7.2f Gaussian fit failed - "+
#                         "ignoring",
#                         star.unique_number, u, v)
#             continue
#         pos_v, pos_u, metadata = ret
#
#         if DEBUG_STARS_PSF_3D:
#             plot3d(metadata["subimg-gradient"], metadata["scaled_psf"],
#                    "Image vs. PSF")
#         if (abs(pos_u-u) > search_limit[1]-.5 or
#             abs(pos_v-v) > search_limit[0]-.5):
#             logger.info("Star %9d UV %7.2f %7.2f refined to %7.2f %7.2f - "+
#                         "Gaussian fit too far away - ignoring",
#                         star.unique_number, u, v, pos_u, pos_v)
#             continue
#
#         if float_psf_sigma:
#             sx = metadata["sigma_x"]
#             sy = metadata["sigma_y"]
#         else:
#             sx = None
#             sy = None
#         if stars_config["float_psf_sigma"]:
#             logger.info("Star %9d UV %7.2f %7.2f refined to %7.2f %7.2f "+
#                         "(%5.2f %5.2f) PSF Sigma %5.3f %5.3f",
#                         star.unique_number, u, v, pos_u, pos_v,
#                         pos_u-u, pos_v-v, sx, sy)
#         else:
#             logger.info("Star %9d UV %7.2f %7.2f refined to %7.2f %7.2f "+
#                         "(%5.2f %5.2f)",
#                         star.unique_number, u, v, pos_u, pos_v,
#                         pos_u-u, pos_v-v)
#         star.psf_delta_u = pos_u-u
#         star.psf_delta_v = pos_v-v
#         if float_psf_sigma:
#             star.psf_sigma_x = sx
#             star.psf_sigma_y = sy
#             sx_list.append(sx)
#             sy_list.append(sy)
#         u_list.append(pos_u)
#         v_list.append(pos_v)
#         delta_u_list.append(pos_u-u)
#         delta_v_list.append(pos_v-v)
#
#     if len(delta_u_list) == 0:
#         logger.info("No refined fits available")
#         return (offset[0], offset[1]), None
#
#     du_mean = np.mean(delta_u_list)
#     dv_mean = np.mean(delta_v_list)
#
#     for star in star_list:
#         if star.psf_delta_u is not None:
#             star.psf_delta_u -= du_mean
#             star.psf_delta_v -= dv_mean
#
#     norm_delta_u = np.array(delta_u_list) - du_mean
#     norm_delta_v = np.array(delta_v_list) - dv_mean
#
#     logger.info("Mean dU,dV %7.2f +/- %4.2f, %7.2f +/- %4.2f",
#                 du_mean, np.std(delta_u_list), dv_mean, np.std(delta_v_list))
#     logger.info("After normalization - Min dU,dV %5.3f %5.3f Max dU,dV %5.3f %5.3f",
#                 np.min(norm_delta_u), np.min(norm_delta_v),
#                 np.max(norm_delta_u), np.max(norm_delta_v))
#     if len(u_list) > 1:
#         logger.info("Correlation U vs. dU %7.5f",
#                     np.corrcoef(u_list, delta_u_list)[0,1])
#         logger.info("Correlation V vs. dV %7.5f",
#                     np.corrcoef(v_list, delta_v_list)[0,1])
#
#     if float_psf_sigma:
#         logger.info("Mean PSF SU,SV %5.3f +/- %5.3f, %5.3f +/- %5.3f",
#                     np.mean(sx_list), np.std(sx_list),
#                     np.mean(sy_list), np.std(sy_list))
#
#     max_error = nav.config.MAX_POINTING_ERROR[obs.data.shape, obs.detector]
#
#     if stars_config["allow_fractional_offsets"]:
#         ret = (offset[0]+du_mean, offset[1]+dv_mean)
#     else:
#         ret = (int(np.round(offset[0]+du_mean)),
#                int(np.round(offset[1]+dv_mean)))
#
#     if abs(ret[0]) > max_error[0] or abs(ret[1]) > max_error[1]:
#         logger.info("Resulting star offset is beyond maximum allowable "+
#                     "offset - aborting")
#         ret = None
#
#     return ret, (np.std(delta_u_list), np.std(delta_v_list))

################################################################################

# def stars_find_offset(obs, ra_dec_predicted,
#                       stars_only=False,
#                       stars_config=None):
#     """Find the image offset based on stars.

#     Inputs:
#         obs                The observation.
#         ra_dec_predicted   A tuple
#                            (ra0,ra1,ra2,dec0,dec1,dec2,dra/dt,ddec/dt)
#                            giving the navigation information from the
#                            more-precise predicted kernels. This is used to
#                            make accurate star streaks.
#         stars_only         True if there is nothing except stars in the FOV.
#                            In this case we will allow navigation with only
#                            a single star and no photometry, if necessary.
#         stars_config       Configuration parameters. None uses the default.

#     Returns:
#         metadata           A dictionary containing information about the
#                            offset result:
#             "offset"            The (U,V) offset.
#             "corr_psf_details"  Correlation details.
#             "confidence"        The confidence (0-1) in the result.
#             "full_star_list"    The list of Stars in the FOV.
#             "num_stars",        The number of Stars in the FOV.
#             "num_good_stars"    The number of Stars that photometrically match.
#             "smear"             The amount of star smear (u,v in pixels).
#             "rings_subtracted"  True if the rings were subtracted from the
#                                 image.
#             "start_time"        The time (s) when stars_find_offset was called.
#             "end_time"          The time (s) when stars_find_offset returned.
#     """

#     start_time = time.time()

#     logger = logging.getLogger(_LOGGING_NAME+".stars_find_offset")

#     if stars_config is None:
#         stars_config = nav.config.STARS_DEFAULT_CONFIG

#     limit_size_max_u, limit_size_max_v = obs.search_limits

#     # The minimum detectable DN
#     min_dn = stars_config[("min_detectable_dn", obs.clean_detector)]

#     # The relationship between the number of good stars and the confidence
#     # level as a line between two points
#     min_stars, min_stars_conf = stars_config["min_stars_low_confidence"]
#     min_stars_hc, min_stars_hc_conf = stars_config["min_stars_high_confidence"]

#     perform_photometry = stars_config["perform_photometry"]
#     try_without_photometry = stars_config["try_without_photometry"]

#     if obs.instrument_host != "cassini":
#         perform_photometry = False

#     stars_only = True
#     if stars_only:
#         try_without_photometry = True

#     radec_movement = None

#     if ra_dec_predicted is not None:
#         radec_movement = (ra_dec_predicted[6] * obs.texp/2,
#                           ra_dec_predicted[7] * obs.texp/2)

#     metadata = {}
#     metadata["start_time"] = start_time
#     metadata["end_time"] = None

#     obs.star_body_list = None # Body inventory cache

#     if stars_config["calibrated_data"]:
#         obs.calib_dn_extdata = None # DN-calibrated, extended data
#     else:
#         obs.calib_dn_extdata = obs.extdata
#         filtered_data = obs.calib_dn_extdata

#     # Get the Star list and initialize our new fields
#     star_list = stars_list_for_obs(obs, radec_movement,
#                                    stars_config=stars_config)
#     for star in star_list:
#         star.photometry_confidence = 0.
#         star.is_bright_enough = False
#         star.is_dim_enough = True

#     metadata["offset"] = None
#     metadata["uncertainty"] = None
#     metadata["confidence"] = 0.
#     metadata["corr_psf_details"] = None
#     metadata["full_star_list"] = star_list
#     metadata["num_stars"] = len(star_list)
#     metadata["num_good_stars"] = 0
#     metadata["smear"] = (0., 0.)
#     metadata["rings_subtracted"] = False

#     if len(star_list) == 0:
#         logger.debug("No stars available - giving up")
#         metadata["end_time"] = time.time()
#         return metadata

#     if len(star_list) > 0:
#         first_star = star_list[0]
#         metadata["smear"] = (first_star.move_u, first_star.move_v)
#         smear_amt = np.sqrt(first_star.move_u**2+first_star.move_v**2)
#         if smear_amt > stars_config["max_smear"]:
#             logger.debug(
#                 "FAILED to find a valid star offset - star smear %.2f is too great",
#                  smear_amt)
#             metadata["end_time"] = time.time()
#             return metadata

#     rings_can_conflict = True

#     if obs.calib_dn_extdata is None:
#         # For star use only, need data in DN for photometry
#         obs.calib_dn_extdata = calibrate_iof_image_as_dn(
#                                                   obs, data=obs.extdata)
#         filtered_data = obs.calib_dn_extdata

#         if False:
#             calib_data = unpad_image(obs.calib_dn_extdata, extfov_margin)
#             rings_radial_model = rings_create_model_from_image(
#                                                    obs, data=calib_data,
#                                                    extfov_margin=extfov_margin)
#             if rings_radial_model is not None:
#                 assert _TKINTER_AVAILABLE
#                 imdisp = ImageDisp([calib_data, rings_radial_model,
#                                     calib_data-rings_radial_model],
#                                    canvas_size=(512,512),
#                                    allow_enlarge=True, enlarge_limit=10,
#                                    auto_update=True)
#                 tk.mainloop()

#                 filtered_data = pad_image(calib_data-rings_radial_model,
#                                           extfov_margin)
#                 obs.data[:,:] = calib_data-rings_radial_model
#                 obs.extdata[:,:] = filtered_data
#                 rings_can_conflict = False

#         # Filter that calibrated data.
#         # 1) Subtract the local background (median)
#         # 2) Eliminate anything that is < 0
#         # 3) Eliminate any portions of the image that are near a pixel
#         #    that is brighter than the maximum star DN
#         #    Note that since a star"s photons are spread out over the PSF,
#         #    you have to have a really bright single pixel to be brighter
#         #    than the entire integrated DN of the brightest star in the
#         #    FOV.
#         filtered_data = filter_sub_median(obs.calib_dn_extdata,
#                                           median_boxsize=11)

#         filtered_data[filtered_data < 0.] = 0.

#         # If we trust the DN values, then we can eliminate any pixels that
#         # are way too bright.
# #         if _trust_star_dn(obs):
# #             max_dn = star_list[0].dn # Star list is sorted by DN
# #             mask = filtered_data > max_dn*2
# #             mask = filt.maximum_filter(mask, 11)
# #             filtered_data[mask] = 0.

#         if DEBUG_STARS_FILTER_IMGDISP:
#             ImageDisp([filtered_data],
#                       canvas_size=(512,512),
#                       enlarge_limit=10,
#                       auto_update=True)
#             tk.mainloop()

#     # Cache the body inventory
#     if obs.star_body_list is None:
#         obs.star_body_list = obs.inventory(
#                                     # nav.config.LARGE_BODY_LIST_TITAN_ATMOS) # TODO
#                                     nav.config.LARGE_BODY_LIST[obs.planet.upper()])

#     # A list of offsets that we have already tried so we don't waste time
#     # trying them a second time.
#     already_tried = []

#     # A list of offset results so we can choose the best one at the very end.
#     saved_offsets = []


#     while True:
#         #
#         #            *** LEVEL 1 ***
#         #
#         # At this level we delete stars that are too bright. These stars may
#         # "latch on" to portions of the image that aren't really stars.
#         #
#         star_count = 0
#         first_good_star = None
#         for star in star_list:
#             if star.is_dim_enough:
#                 star_count += 1
#                 if first_good_star is None:
#                     first_good_star = star

#         logger.debug("** LEVEL 1: Trying star list with %d stars", star_count)

#         if star_count < min_stars:
#             # There"s no point in continuing if there aren't enough stars
#             logger.debug("FAILED to find a valid star offset - too few total stars")
#             break

#         if first_good_star.dn < min_dn:
#             # There"s no point in continuing if the brightest star is below the
#             # detection threshold
#             logger.debug(
#                  "FAILED to find a valid star offset - brightest star is too dim")
#             metadata["end_time"] = time.time()
#             return metadata

#         offset = None
#         good_stars = 0
#         confidence = 0.
#         corr_psf_details = None

#         got_it = False

#         # Try with the brightest stars, then go down to the dimmest
#         for min_dn_gain in (1.,):#[4., 2.5, 1.]:
#             #
#             #            *** LEVEL 2 ***
#             #
#             # At this level we delete stars that are too dim. These stars may
#             # have poor detections and just confuse the correlation.
#             #
#             logger.debug("** LEVEL 2: Trying MinDN gain %.1f", min_dn_gain)
#             new_star_list = []
#             for star in star_list:
#                 star.photometry_confidence = 0.
#                 star.is_bright_enough = False
#                 if not star.is_dim_enough:
#                     continue
#                 if star.dn < min_dn*min_dn_gain:
#                     continue
#                 star.is_bright_enough = True
#                 new_star_list.append(star)

#             logger.debug("Using %d stars", len(new_star_list))

#             if len(new_star_list) < min_stars:
#                 logger.debug("Not enough stars: %d (%d required)",
#                              len(new_star_list), min_stars)
#                 continue

#             # Try with a small search area, then enlarge
#             for search_multipler in stars_config["search_multipliers"]:
#                 #
#                 #            *** LEVEL 3 ***
#                 #
#                 # At this level we restrict the search space so we first try
#                 # smaller offsets.
#                 #
#                 logger.debug("** LEVEL 3: Trying search multiplier %.2f",
#                              search_multipler)

#                 # The remaining search levels are inside the subroutine
#                 ret = _stars_find_offset(obs, filtered_data, new_star_list,
#                                          min_stars, search_multipler,
#                                          5, already_tried, 4,
#                                          perform_photometry,
#                                          rings_can_conflict,
#                                          radec_movement,
#                                          stars_config)

#                 # Save the offset and maybe continue iterating
#                 (offset, good_stars, corr_val, corr_details,
#                  keep_searching, no_peaks) = ret

#                 if no_peaks and search_multipler == 1.:
#                     logger.debug("No peaks found at largest search range - "+
#                                  "aborting star offset finding")
#                     got_it = True
#                     break

#                 if offset is None:
#                     logger.debug("No valid offset found - iterating")
#                     continue

#                 logger.debug("Found valid offset U,V %d,%d STARS %d CORR %f",
#                              offset[0], offset[1], good_stars, corr_val)
#                 saved_star_list = copy.deepcopy(star_list)
#                 saved_offsets.append((offset, good_stars, corr_val,
#                                       corr_details, saved_star_list))
#                 if not keep_searching:
#                     got_it = True
#                     break

#                 # End of LEVEL 3 - restrict search area

#             if got_it:
#                 break

#             if len(new_star_list) == len(star_list):
#                 # No point in trying smaller MinDNs if we already are looking at
#                 # all stars
#                 logger.debug(
#                      "Already looking at all stars - ignoring other MinDNs")
#                 break

#             # End of LEVEL 2 - eliminate dim stars

#         if got_it:
#             break

#         # Get rid of an unusually bright star - these sometimes get stuck
#         # correlating with non-star objects, like parts of the F ring
#         still_good_stars_left = False
#         for i in range(len(star_list)):
#             if not star_list[i].is_dim_enough:
#                 continue
#             # First bright star we used last time
#             if i == len(star_list)-1:
#                 # It was the last star - nothing to compare against
#                 logger.debug("No dim enough stars left - giving up")
#                 break
#             too_bright_dn = stars_config["too_bright_dn"]
#             too_bright_factor = stars_config["too_bright_factor"]
#             if ((too_bright_dn and star_list[i].dn > too_bright_dn) or
#                 (too_bright_factor and
#                  star_list[i].dn > star_list[i+1].dn*too_bright_factor)):
#                 # Star is too bright - get rid of it
#                 logger.debug("Star %9d (DN %7.2f) is too bright - "+
#                              "ignoring and iterating",
#                              star_list[i].unique_number, star_list[i].dn)
#                 star_list[i].is_dim_enough = False
#                 still_good_stars_left = True
#             break

#         if not still_good_stars_left:
#             break

#         # End of LEVEL 1 - eliminate bright stars

#     used_photometry = perform_photometry

#     if len(saved_offsets) == 0:
#         offset = None
#         corr_psf_details = None
#         good_stars = 0
#         logger.info("FAILED to find a valid star offset")

#         if (stars_only or (perform_photometry and try_without_photometry)):
#             for star in star_list:
#                 star.photometry_confidence = 0.
#                 star.is_dim_enough = True
#                 star.is_bright_enough = False
#                 if star.dn >= min_dn:
#                     star.is_bright_enough = True

#         if (stars_only or
#             star_list[0].dn > stars_config["min_dn_force_one_star"]):
#             logger.info("Trying again with only one star required")
#             ret = _stars_find_offset(obs, filtered_data, star_list,
#                                      1, 1.,
#                                      1, [], 4,
#                                      perform_photometry, rings_can_conflict,
#                                      radec_movement, stars_config)
#             (offset, good_stars, corr_val, corr_details,
#              keep_searching, no_peaks) = ret
#             if no_peaks:
#                 offset = None
#                 corr_psf_details = None

#         if offset is None and perform_photometry and try_without_photometry:
#             logger.info("Trying again with photometry disabled")
#             ret = _stars_find_offset(obs, filtered_data, star_list,
#                                      min_stars, 1.,
#                                      1, [], 4,
#                                      False, rings_can_conflict,
#                                      radec_movement, stars_config)
#             (offset, good_stars, corr_val, corr_details,
#              keep_searching, no_peaks) = ret
#             if no_peaks:
#                 offset = None
#                 corr_psf_details = None
#             used_photometry = False

#         if (offset is None and stars_only and
#             perform_photometry and try_without_photometry):
#             logger.info("Trying again with only one star required AND "+
#                         "photometry disabled")
#             ret = _stars_find_offset(obs, filtered_data, star_list,
#                                      1, 1.,
#                                      1, [], 4,
#                                      False, rings_can_conflict,
#                                      radec_movement, stars_config)
#             (offset, good_stars, corr_val, corr_details,
#              keep_searching, no_peaks) = ret
#             if no_peaks:
#                 offset = None
#                 corr_psf_details = None
#             used_photometry = False

#     else:
#         best_offset = None
#         best_star_list = None
#         best_good_stars = -1
#         best_corr_val = -1
#         best_corr_details = None
#         for (offset, good_stars, corr_val, corr_details,
#              saved_star_list) in saved_offsets:
#             if len(saved_offsets) > 1:
#                 logger.info("Saved offset U,V %d,%d / Good stars %d / Corr %f",
#                             offset[0], offset[1], good_stars, corr_val)
#             if (good_stars > best_good_stars or
#                 (good_stars == best_good_stars and
#                  corr_val > best_corr_val)):
#                 best_offset = offset
#                 best_good_stars = good_stars
#                 best_corr_val = corr_val
#                 best_corr_details = corr_details
#                 best_star_list = saved_star_list
#         offset = best_offset
#         good_stars = best_good_stars
#         corr_val = best_corr_val
#         corr_psf_details = best_corr_details
#         star_list = saved_star_list

#     uncertainty = None
#     if offset is not None:
#         logger.info("Trial final offset U,V %d,%d / Good stars %d / Corr %f",
#                      offset[0], offset[1], good_stars, corr_val)

#         corr_psf_details = corr_analyze_peak(*corr_details,
#                                              large_psf=False)
#         if corr_psf_details is None:
#             logger.info("Correlation peak analysis failed")
#         else:
#             corr_log_xy_err(logger, corr_psf_details)
#             if (corr_psf_details["x"] is not None and
#                 corr_psf_details["y"] is not None):
#                 offset = (offset[0]+corr_psf_details["x"],
#                           offset[1]+corr_psf_details["y"])
#                 if (abs(offset[0]) > limit_size_max_u or
#                     abs(offset[1]) > limit_size_max_v):
#                     logger.info("Resulting star offset is beyond maximum allowable "+
#                                 "offset")
#                     offset = None
#                     uncertainty = None
#                 else:
#                     uncertainty = (corr_psf_details["xy_rot_err_1"],
#                                    corr_psf_details["xy_rot_err_2"],
#                                    corr_psf_details["xy_rot_angle"])

#     # Compute confidence
#     if offset is None:
#         confidence = 0.
#     else:
#         if good_stars < min_stars:
#             confidence = 0.3 * good_stars / min_stars
#             # Give a boost if the star is really bright
#             if star_list[0].vmag < 3:
#                 confidence += 0.2
#         else:
#             confidence = ((good_stars-min_stars) *
#                             (float(min_stars_hc_conf)-min_stars_conf)/
#                             (float(min_stars_hc-min_stars)) +
#                           min_stars_conf)
#         if len(star_list) > 0:
#             movement = np.sqrt(star_list[0].move_u**2 + star_list[0].move_v**2)
#             if movement > 1:
#                 confidence /= (movement-1)/4+1
#         if not used_photometry:
#             confidence *= 0.5
#         confidence = np.clip(confidence, 0., 1.)

#     metadata["offset"] = offset
#     metadata["uncertainty"] = uncertainty
#     metadata["corr_psf_details"] = corr_psf_details
#     metadata["confidence"] = confidence
#     metadata["full_star_list"] = star_list
#     metadata["num_stars"] = len(star_list)
#     metadata["num_good_stars"] = good_stars
#     metadata["rings_subtracted"] = not rings_can_conflict

#     if offset is not None:
#         if (uncertainty is not None and
#             uncertainty[0] is not None and
#             uncertainty[1] is not None):
#             logger.info("Returning final offset U,V %.2f,%.2f (%.2fx%.2f @ %.2f) / Good stars %d / "+
#                         "Corr %f / Conf %f / Rings sub %s",
#                         offset[0], offset[1],
#                         uncertainty[0], uncertainty[1], uncertainty[2],
#                         good_stars, corr_val, confidence,
#                         str(not rings_can_conflict))
#         else:
#             logger.info("Returning final offset U,V %.2f,%.2f / Good stars %d / "+
#                         "Corr %f / Conf %f / Rings sub %s",
#                         offset[0], offset[1], good_stars, corr_val, confidence,
#                         str(not rings_can_conflict))


#     offset_x = 0
#     offset_y = 0
#     if offset is not None:
#         offset_x = offset[0]
#         offset_y = offset[1]

#     for star in star_list:
#         _stars_mark_conflicts_edge(obs, star, (offset_x, offset_y),
#                                    stars_config)

#     logger.info("Final star list after offset:")
#     for star in star_list:
#         logger.info("Star %9d U %8.3f+%7.3f V %8.3f+%7.3f MAG %6.3f "+
#                     "SCLASS %3s TEMP %6d DN %7.2f MEAS %7.2f CONF %4.2f CONFLICTS %s",
#                     star.unique_number,
#                     star.u+offset_x, abs(star.move_u),
#                     star.v+offset_y, abs(star.move_v),
#                     star.vmag,
#                     clean_sclass(star.spectral_class),
#                     0 if star.temperature is None else star.temperature,
#                     star.dn, star.integrated_dn, star.photometry_confidence,
#                     str(star.conflicts))

#     metadata["end_time"] = time.time()
#     return metadata

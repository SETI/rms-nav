# mypy: ignore-errors

import copy

import tkinter as tk
from imgdisp import ImageDisp

from nav.support.image import draw_rect
_DEBUG_STARS_MODEL_IMGDISP = False

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
from nav.config import Config
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
                 *,
                 config: Optional[Config] = None) -> None:
        """Creates a navigation model for stars.

        Parameters:
            obs: The Observation object containing image data.
            config: Configuration object to use. If None, uses DEFAULT_CONFIG.
        """

        super().__init__(obs, config=config)

        self._obs = obs
        self._conflict_body_list = None
        self._star_list = None

    @property
    def star_list(self) -> list[Star]:
        """Return the list of stars in the model."""
        return self._star_list

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
        return ('Star %9d U %8.3f+/-%7.3f V %8.3f+/-%7.3f MAG %6.3f BMAG %6.3f '
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
            star.psf_size = obs.star_psf_size(star)
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
            if config.stellar_aberration:
                self._aberrate_star(star)
            star_list2.append(star)

        # Eliminate stars that are not actually in the FOV, including the PSF size
        # margin beyond the edge

        if config.proper_motion:
            ra_dec_list = [x.ra_dec_with_pm(obs.midtime) for x in star_list2]
        else:
            ra_dec_list = [(x.ra, ra.dec) for x in star_list2]
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
            psf_size_half_u = star.psf_size[1] // 2
            psf_size_half_v = star.psf_size[0] // 2

            # TODO Need to test star movement here as well
            if (u <= obs.extfov_u_min+psf_size_half_u or
                u >= obs.extfov_u_max-psf_size_half_u or
                v <= obs.extfov_v_min+psf_size_half_v or
                v >= obs.extfov_v_max-psf_size_half_v or
                u1 <= obs.extfov_u_min+psf_size_half_u or
                u1 >= obs.extfov_u_max-psf_size_half_u or
                v1 <= obs.extfov_v_min+psf_size_half_v or
                v1 >= obs.extfov_v_max-psf_size_half_v or
                u2 <= obs.extfov_u_min+psf_size_half_u or
                u2 >= obs.extfov_u_max-psf_size_half_u or
                v2 <= obs.extfov_v_min+psf_size_half_v or
                v2 >= obs.extfov_v_max-psf_size_half_v):
                self._logger.debug("Star %9d U %8.3f V %8.3f is off the edge",
                                   star.unique_number, u, v)
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

            # Remove stars that are on top of each other
            if len(full_star_list) > 1:
                new_full_star_list = []
                for i in range(len(full_star_list)):
                    if full_star_list[i].conflicts == 'STAR':
                        continue
                    for j in range(i+1, len(full_star_list)):
                        u_gap = (full_star_list[i].psf_size[1] / 2 +
                                 full_star_list[j].psf_size[1] / 2)
                        v_gap = (full_star_list[i].psf_size[0] / 2 +
                                 full_star_list[j].psf_size[0] / 2)
                        if (abs(full_star_list[i].v - full_star_list[j].v) < v_gap and
                            abs(full_star_list[i].u - full_star_list[j].u) < u_gap):
                            self._logger.debug('Removing overlapping stars:')
                            self._logger.debug('  %s',
                                               self._star_short_info(full_star_list[i]))
                            self._logger.debug('  %s',
                                               self._star_short_info(full_star_list[j]))
                            full_star_list[j].conflicts = 'STAR'
                            break
                    else:
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

        log_level = self._config.general.get('model_stars_log_level')
        with self._logger.open(f'CREATE STARS MODEL', level=log_level):
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
        self._star_list = star_list
        # TODO Should break this apart into a dictionary for the metadata
        metadata['star_list'] = [self._star_short_info(star) for star in star_list]

        for star in star_list:
            if star.conflicts and not ignore_conflicts:
                continue
            u_idx = star.u + self._obs.extfov_margin_u
            v_idx = star.v + self._obs.extfov_margin_v
            u_int = int(u_idx)
            v_int = int(v_idx)
            u_frac = u_idx - u_int
            v_frac = v_idx - v_int

            # psf_size = _find_psf_boxsize(star, stars_config)

            psf_size_half_u = int(star.psf_size[1] + np.round(abs(star.move_u))) // 2
            psf_size_half_v = int(star.psf_size[0] + np.round(abs(star.move_v))) // 2

            move_gran = max(abs(star.move_u) / max_move_steps,
                            abs(star.move_v) / max_move_steps)
            move_gran = np.clip(move_gran, 0.1, 1.0)

            # sigma = nav.config.PSF_SIGMA[obs.clean_detector]
            # if star.dn >= stars_config["psf_gain"][0]:
            #     sigma *= stars_config["psf_gain"][1]

            psf = self.obs.star_psf()

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
            v_halfsize = (star.psf_size[0] // 2) + 2
            u_halfsize = (star.psf_size[1] // 2) + 2

            # width += thickness-1
            # if (not width <= u_idx < overlay.shape[1]-width or
            #     not width <= v_idx < overlay.shape[0]-width):
            #     continue

            u_min = u - u_halfsize
            v_min = v - v_halfsize
            u_max = u + u_halfsize
            v_max = v + v_halfsize
            u_min, v_min = self._obs.clip_extfov(u_min, v_min)
            u_max, v_max = self._obs.clip_extfov(u_max, v_max)

            star_avoid_mask[v_min:v_max+1, u_min:u_max+1] = True
            draw_rect(star_overlay, True, u, v, u_halfsize, v_halfsize)

            stretch_region = self._obs.make_extfov_false()
            stretch_region[v_min:v_max+1, u_min:u_max+1] = True
            # Note that packbits pads the array with zeros to the nearest multiple of 8
            # in each dimension, so when unpacking the array we have to clip the array
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

            label_margin = u_halfsize + 3

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
        self._model_mask = self._model_img != 0
        self._range = np.inf
        self._blur_amount = None  # np.eye(2, 2) * 5.
        self._uncertainty = 0.
        self._confidence = 1.
        self._stretch_regions = stretch_regions
        self._annotations = annotations
        self._metadata = metadata

        self._logger.debug(f'  Star model min: {np.min(self._model_img)}, max: {np.max(self._model_img)}')

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

        return False

import copy
from typing import TYPE_CHECKING, cast

import numpy as np

from nav.config import Config
from nav.nav_model import NavModelCombined, NavModelStars
from nav.support.correlate import navigate_with_pyramid_kpeaks
from nav.support.misc import mad_std
from nav.support.types import MutableStar, NDArrayFloatType, NDArrayIntType

from .nav_technique import NavTechnique

if TYPE_CHECKING:
    from nav.nav_master import NavMaster


class NavTechniqueCorrelateAll(NavTechnique):
    """Implements navigation technique using correlation across all available models."""

    def __init__(self, nav_master: 'NavMaster', *, config: Config | None = None) -> None:
        """Initializes a navigation technique using correlation across all available models.

        Parameters:
            nav_master: The navigation master instance.
            config: Configuration object to use. If None, uses DEFAULT_CONFIG.
        """

        super().__init__(nav_master, config=config)

        self._combined_model: NavModelCombined | None = None

    # # Bodies: limb annulus extraction + gradient model
    # # Set annulus half-width in pixels from crater_height/resolution
    # base = 1.5  # pixels
    # k_bumpy = 2.0  # scale factor
    # for b in body_components or []:
    #     B = normalize(b.array)
    #     # Limb emphasis: gradient magnitude of a slightly blurred silhouette
    #     Bg = gaussian_blur_cov(B, b.U)  # optional extra uncertainty blur
    #     G = gradient_magnitude(Bg)
    #     # limb mask = thresholded gradient band, dilated by width ~ base + k*H/R
    #     thr = 0.5 * np.max(G) if np.max(G) > 0 else 0.0
    #     limb = (G >= thr).astype(np.float64)
    #     width = base + k_bumpy * (b.crater_height / max(b.resolution, 1e-12))
    #     # approximate dilation by Gaussian blur of the binary mask then clip
    #     limb_band = gaussian_blur_cov(limb, np.diag([width**2, width**2]))
    #     limb_band = (limb_band > 0.1).astype(np.float64)

    #     # model contribution = gradient magnitude within limb band
    #     M_eff += b.confidence * (G * limb_band)
    #     W_eff += b.confidence * limb_band
    #     total_w += b.confidence

    # # Rings
    # for r in ring_components or []:
    #     R = normalize(r.array)
    #     Rb = gaussian_blur_cov(R, r.U)  # anisotropic radial blur encoded in U
    #     M_eff += r.confidence * Rb
    #     W_eff += r.confidence * r.W
    #     total_w += r.confidence

    # Create a single model which, for each pixel, has the element from the model with
    # the smallest range (to the observer), and is thus in front.

    def combined_model(self) -> NavModelCombined | None:
        """Returns the final combined model."""
        return self._combined_model

    def navigate(self) -> None:
        """Performs navigation using correlation across all available models.

        Attempts to find correlation between observed image and combined model,
        computing the offset if successful.
        """

        # offset, uncertainty, confidence, and metadata start out as None/empty

        try:
            log_level = self.config.general.log_level_nav_correlate_all
        except AttributeError:
            log_level = None
        with self.logger.open('NAVIGATION PASS: ALL MODELS CORRELATION', log_level=log_level):
            obs = self.nav_master.obs

            #
            # The first thing we do is create the combined model with ALL models available
            #
            used_model_names = ', '.join([x.name for x in self.nav_master.all_models])
            self.logger.info(f'Initial correlation using all models: {used_model_names}')
            combined_model = self._combine_models(['*'])
            self._combined_model = combined_model
            if combined_model is None:
                self.logger.info('correlate_all navigation technique failed - no models available')
                return

            if (
                len(combined_model.models) == 0
                or combined_model.models[0].model_img is None
                or combined_model.models[0].model_mask is None
            ):
                raise ValueError('Combined model has no result or missing image/mask')

            result = navigate_with_pyramid_kpeaks(
                obs.extdata,
                combined_model.models[0].model_img,
                combined_model.models[0].model_mask,
                upsample_factor=self.config.offset.correlation_fft_upsample_factor,
            )

            corr_offset = (float(result['offset'][0]), float(result['offset'][1]))

            if not (
                -obs.extfov_margin_u + 1 < corr_offset[1] < obs.extfov_margin_u - 1
                and -obs.extfov_margin_v + 1 < corr_offset[0] < obs.extfov_margin_v - 1
            ):
                self.logger.info('Correlation offset is outside the extended FOV')
                return

            self.logger.debug(
                f'Correlation offset: dU {corr_offset[1]:.3f}, dV {corr_offset[0]:.3f}'
            )
            self.logger.debug(f'Correlation quality: {float(result["quality"]):.3f}')

            self._offset = corr_offset
            # TODO
            self._uncertainty = (
                float(result['sigma_xy'][0]),
                float(result['sigma_xy'][1]),
            )
            self._confidence = 1

            star_models = self._filter_models(['stars'])
            if len(star_models) == 1:
                ret = self._refine_stars(cast(NavModelStars, star_models[0]), corr_offset)
                if ret is not None:
                    self._offset, self._uncertainty = ret

        if not (
            -obs.extfov_margin_u + 1 < self._offset[1] < obs.extfov_margin_u - 1
            and -obs.extfov_margin_v + 1 < self._offset[0] < obs.extfov_margin_v - 1
        ):
            self.logger.info('Final offset is outside the extended FOV')
            self._offset = None
            self._uncertainty = None
            self._confidence = None
            return

        self._metadata['offset'] = self._offset
        self._metadata['uncertainty'] = self._uncertainty
        self._metadata['confidence'] = self._confidence

    def _refine_stars(
        self, star_model: NavModelStars, offset: tuple[float, float]
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """Refine the offset using the position of stars in the image based.

        Parameters:
            star_model: The star model containing the list of stars.
            offset: The initial offset to refine.

        Returns:
            A tuple containing the refined offset and uncertainty.
        """

        def detect_outliers(
            data: list[float], reliability: list[float], threshold: float
        ) -> NDArrayIntType:
            data_array = cast(NDArrayFloatType, np.asarray(data))
            n = data_array.size
            if n < 3:
                return cast(NDArrayIntType, np.array([], dtype=int))

            # Robust center and scale
            median = np.median(data_array)
            mad = np.median(np.abs(data_array - median))
            if mad == 0:  # degenerate case
                return np.array([], dtype=int)

            # Robust z-scores (standard deviation units)
            z = 0.6745 * (data_array - median) / mad  # 0.6745 ~ Phi^-1(0.75)

            # Make later points “more suspicious” by dividing by reliability
            score = np.abs(z) / reliability

            outliers = np.where(score > threshold)[0]
            return outliers

        if not self.config.offset.star_refinement_enabled or len(star_model.star_list) == 0:
            return None

        obs = self.nav_master.obs
        img = obs.data
        psf = obs.star_psf()

        self.logger.info('Starting star position optimization process')
        self.logger.info(f'Initial offset: dU {offset[1]:.3f}, dV {offset[0]:.3f}')
        self.logger.info(f'Initial number of stars: {len(star_model.star_list)}')

        u_diff_list = []
        v_diff_list = []
        uv_star_list = []
        new_star_list: list[MutableStar] = copy.deepcopy(star_model.star_list)
        for star in new_star_list:
            if star.conflicts:
                continue
            psf_size = obs.star_psf_size(star)
            star_u = star.u + offset[1]
            star_v = star.v + offset[0]
            if (
                star_u < psf_size[1]
                or star_u > img.shape[1] - psf_size[1]
                or star_v < psf_size[0]
                or star_v > img.shape[0] - psf_size[0]
            ):
                self.logger.debug(
                    f'Star {star.pretty_name:9s} VMAG {star.vmag:6.3f} '
                    f'U {star_u:8.3f}, V {star_v:8.3f} too '
                    'close to edge or outside image'
                )
                star.conflicts = 'REFINEMENT EDGE'
                continue
            ret = psf.find_position(
                img,
                psf_size,
                (star_v, star_u),
                search_limit=self.config.offset.star_refinement_search_limit,
            )
            if ret is None:
                self.logger.debug(
                    f'Star {star.pretty_name:9s} VMAG {star.vmag:6.3f} '
                    f'U {star_u:8.3f}, V {star_v:8.3f} failed '
                    'to find position'
                )
                star.conflicts = 'REFINEMENT FAILED'
                continue
            opt_v, opt_u, _opt_metadata = ret
            self.logger.debug(
                f'Star {star.pretty_name:9s} VMAG {star.vmag:6.3f} '
                f'Searched at {star_u:8.3f}, {star_v:8.3f} '
                f'found at {opt_u:8.3f}, {opt_v:8.3f} '
                f'diff {opt_u - star_u:6.3f}, {opt_v - star_v:6.3f}'
            )
            # TODO Implement edge clipping for Voyager and Galileo
            # if opt_u < clip or opt_u > img.shape[1] - clip or opt_v < clip or
            # opt_v > img.shape[0] - clip:
            #     if verbose:
            #         print(f'Star {star.unique_number} VMAG {star.vmag} clipped')
            #     return False
            star.diff_u = float(opt_u - star_u)
            star.diff_v = float(opt_v - star_v)
            u_diff_list.append(star.diff_u)
            v_diff_list.append(star.diff_v)
            uv_star_list.append(star)

        if len(u_diff_list) == 0:
            self.logger.info('No stars found to refine')
            return None

        u_diff_min = np.min(u_diff_list)
        u_diff_max = np.max(u_diff_list)
        u_diff_mean = np.mean(u_diff_list)
        u_diff_std = np.std(u_diff_list)
        u_diff_median = np.median(u_diff_list)
        u_diff_mad = mad_std(u_diff_list)
        v_diff_min = np.min(v_diff_list)
        v_diff_max = np.max(v_diff_list)
        v_diff_mean = np.mean(v_diff_list)
        v_diff_std = np.std(v_diff_list)
        v_diff_median = np.median(v_diff_list)
        v_diff_mad = mad_std(v_diff_list)

        self.logger.info(f'Number of stars: {len(u_diff_list)}')
        self.logger.info(
            f'U diff: min {u_diff_min:6.3f}, max {u_diff_max:6.3f}, '
            f'mean {u_diff_mean:6.3f} +/- {u_diff_std:6.3f}, '
            f'median {u_diff_median:6.3f} +/- {u_diff_mad:6.3f}'
        )
        self.logger.info(
            f'V diff: min {v_diff_min:6.3f}, max {v_diff_max:6.3f}, '
            f'mean {v_diff_mean:6.3f} +/- {v_diff_std:6.3f}, '
            f'median {v_diff_median:6.3f} +/- {v_diff_mad:6.3f}'
        )

        nsigma = self.config.offset.star_refinement_nsigma

        # Roughly mark dimmer stars as less reliable and thus more likely to be outliers
        min_vmag = 6  # TODO Fix this
        max_vmag = obs.star_max_usable_vmag()
        vmag_spread = max_vmag - min_vmag
        # Convert vmag to a reliability between 1 and 0.5.
        # Note vmag is guaranteed to have a value because of if it doesn't the star
        # isn't added to the original star list.
        # TODO clean this up
        reliability = [1 - (cast(float, x.vmag) - min_vmag) / vmag_spread / 2 for x in uv_star_list]
        u_outliers = detect_outliers(u_diff_list, reliability, nsigma)
        v_outliers = detect_outliers(v_diff_list, reliability, nsigma)
        final_u_diff_list = []
        final_v_diff_list = []
        final_uv_star_list = []
        for idx, (u_diff, v_diff, star) in enumerate(
            zip(u_diff_list, v_diff_list, uv_star_list, strict=True)
        ):
            if idx in u_outliers or idx in v_outliers:
                self.logger.debug(
                    f'Star {star.pretty_name:9s} VMAG {star.vmag:6.3f} '
                    f'U {star.u:8.3f}, V {star.v:8.3f} diff '
                    f'{star.diff_u:6.3f}, {star.diff_v:6.3f} '
                    'marked as an outlier'
                )
                star.conflicts = 'REFINEMENT OUTLIER'
            else:
                final_u_diff_list.append(u_diff)
                final_v_diff_list.append(v_diff)
                final_uv_star_list.append(star)

        if len(final_u_diff_list) == 0:
            self.logger.info('No remaining stars after removing outliers')
            return None

        final_u_diff_min = np.min(final_u_diff_list)
        final_u_diff_max = np.max(final_u_diff_list)
        final_u_diff_mean = np.mean(final_u_diff_list)
        final_u_diff_std = np.std(final_u_diff_list)
        final_u_diff_median = np.median(final_u_diff_list)
        final_u_diff_mad = mad_std(final_u_diff_list)
        final_v_diff_min = np.min(final_v_diff_list)
        final_v_diff_max = np.max(final_v_diff_list)
        final_v_diff_mean = np.mean(final_v_diff_list)
        final_v_diff_std = np.std(final_v_diff_list)
        final_v_diff_median = np.median(final_v_diff_list)
        final_v_diff_mad = mad_std(final_v_diff_list)

        self.logger.info(f'Refined number of stars: {len(final_u_diff_list)}')
        self.logger.info(
            f'Refined U diff: min {final_u_diff_min:6.3f}, '
            f'max {final_u_diff_max:6.3f}, '
            f'mean {final_u_diff_mean:6.3f} +/- {final_u_diff_std:6.3f}, '
            f'median {final_u_diff_median:6.3f} +/- {final_u_diff_mad:6.3f}'
        )
        self.logger.info(
            f'Refined V diff: min {final_v_diff_min:6.3f}, '
            f'max {final_v_diff_max:6.3f}, '
            f'mean {final_v_diff_mean:6.3f} +/- {final_v_diff_std:6.3f}, '
            f'median {final_v_diff_median:6.3f} +/- {final_v_diff_mad:6.3f}'
        )

        # Update the offset with the median difference
        refined_offset = (
            float(offset[0] + final_v_diff_median),
            float(offset[1] + final_u_diff_median),
        )
        refined_sigma = (
            float(np.std(final_v_diff_list)),
            float(np.std(final_u_diff_list)),
        )

        self.logger.info(
            'Refined offset: '
            f'dU {refined_offset[1]:.3f} +/- {refined_sigma[1]:.3f}, '
            f'dV {refined_offset[0]:.3f} +/- {refined_sigma[0]:.3f}'
        )
        return refined_offset, refined_sigma

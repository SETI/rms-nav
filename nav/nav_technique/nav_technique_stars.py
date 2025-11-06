import copy
from typing import Optional, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from .nav_technique import NavTechnique
from nav.config import Config
from nav.nav_model.nav_model_stars import NavModelStars
from nav.support.correlate import navigate_with_pyramid_kpeaks
from nav.support.image import gaussian_blur_cov

if TYPE_CHECKING:
    from nav.nav_master import NavMaster


class NavTechniqueStars(NavTechnique):
    """Implements navigation technique using star field correlation.

    Parameters:
        *args: Variable length argument list passed to parent class
        **kwargs: Arbitrary keyword arguments passed to parent class
    """
    def __init__(self,
                 nav_master: 'NavMaster',
                 *,
                 config: Optional[Config] = None) -> None:
        super().__init__(nav_master, config=config)

    def navigate(self) -> None:
        """Performs navigation using star field correlation.

        Attempts to find correlation between observed image and star model,
        computing the offset if successful.
        """

        log_level = self._config.general.get('log_level_nav_stars')
        with self.logger.open('NAVIGATION PASS: STARS', level=log_level):
            obs = self.nav_master.obs
            star_models = self.nav_master.star_models
            if len(star_models) == 0:
                self.logger.info('No star models available')
                return
            if len(star_models) > 1:
                self.logger.error('More than one star model available')
                return
            star_model = star_models[0]

            if self.config.stars.max_refinement_iterations <= 0:
                raise ValueError('max_refinement_iterations must be greater than 0')

            for pass_num in range(self.config.stars.max_refinement_iterations):
                self.logger.debug(f'Starting refinement pass {pass_num+1}')

                model_img = star_model.model_img
                # plt.imshow(model_img)
                if star_model.blur_amount is not None:
                    model_img = gaussian_blur_cov(model_img, star_model.blur_amount)
                    # plt.figure()
                    # plt.imshow(model_img)
                    # plt.show()

                result = navigate_with_pyramid_kpeaks(
                    obs.extdata, star_model.model_img, star_model.model_mask,
                    upsample_factor=self.config.stars.correlation_fft_upsample_factor)
                # TODO Handle failure
                corr_offset = (-float(result['offset'][0]), -float(result['offset'][1]))

                self.logger.debug('Correlation offset: '
                                  f'dU {corr_offset[1]:.3f}, dV {corr_offset[0]:.3f}')
                self.logger.debug('Correlation quality: '
                                  f'{float(result['quality']):.3f}')

                img = obs.data
                psf = obs.star_psf()

                self.logger.info('Starting star position optimization process')
                u_diff_list = []
                v_diff_list = []
                new_star_list = copy.deepcopy(star_model.star_list)
                for star in new_star_list:
                    if star.conflicts:
                        continue
                    psf_size = obs.star_psf_size(star)
                    star_u = star.u - corr_offset[1]
                    star_v = star.v - corr_offset[0]
                    if (star_u < psf_size[1] or star_u > img.shape[1] - psf_size[1] or
                        star_v < psf_size[0] or star_v > img.shape[0] - psf_size[0]):
                        self.logger.debug(f'Star {star.pretty_name:9s} VMAG {star.vmag:6.3f} '
                                          f'U {star_u:8.3f}, V {star_v:8.3f} too '
                                          'close to edge or outside image')
                        star.conflicts = 'REFINEMENT EDGE'
                        continue
                    ret = psf.find_position(img, psf_size, (star_v, star_u),
                                            search_limit=self.config.stars.refinement_search_limit)
                    if ret is None:
                        self.logger.debug(f'Star {star.pretty_name:9s} VMAG {star.vmag:6.3f} '
                                          f'U {star_u:8.3f}, V {star_v:8.3f} failed '
                                          'to find position')
                        star.conflicts = 'REFINEMENT FAILED'
                        continue
                    opt_v, opt_u, opt_metadata = ret
                    self.logger.debug(f'Star {star.pretty_name:9s} VMAG {star.vmag:6.3f} '
                                      f'Searched at {star_u:8.3f}, {star_v:8.3f} '
                                      f'found at {opt_u:8.3f}, {opt_v:8.3f} '
                                      f'diff {opt_u-star_u:6.3f}, {opt_v-star_v:6.3f}')
                    # TODO Implement edge clipping for Voyager and Galileo
                    # if opt_u < clip or opt_u > img.shape[1] - clip or opt_v < clip or
                    # opt_v > img.shape[0] - clip:
                    #     if verbose:
                    #         print(f'Star {star.unique_number} VMAG {star.vmag} clipped')
                    #     return False
                    star.diff_u = float(opt_u-star_u)
                    star.diff_v = float(opt_v-star_v)
                    u_diff_list.append(star.diff_u)
                    v_diff_list.append(star.diff_v)

                u_diff_min = np.min(u_diff_list)
                u_diff_max = np.max(u_diff_list)
                u_diff_mean = np.mean(u_diff_list)
                u_diff_std = np.std(u_diff_list)
                u_diff_median = np.median(u_diff_list)
                v_diff_min = np.min(v_diff_list)
                v_diff_max = np.max(v_diff_list)
                v_diff_mean = np.mean(v_diff_list)
                v_diff_std = np.std(v_diff_list)
                v_diff_median = np.median(v_diff_list)

                self.logger.info(f'U diff: min {u_diff_min:6.3f}, max {u_diff_max:6.3f}, '
                                 f'mean {u_diff_mean:6.3f}, std {u_diff_std:6.3f}, '
                                 f'median {u_diff_median:6.3f}')
                self.logger.info(f'V diff: min {v_diff_min:6.3f}, max {v_diff_max:6.3f}, '
                                 f'mean {v_diff_mean:6.3f}, std {v_diff_std:6.3f}, '
                                 f'median {v_diff_median:6.3f}')

                refined_offset = (corr_offset[0] + v_diff_mean, corr_offset[1] + u_diff_mean)
                refined_sigma = (v_diff_std, u_diff_std)

                if pass_num == self.config.stars.max_refinement_iterations - 1:
                    break

                nsigma = self.config.stars.refinement_nsigma

                # Mark the outliers for the next iteration
                marked_outlier = False
                for star in new_star_list:
                    if star.conflicts:
                        continue
                    if (abs(star.diff_u) > nsigma * u_diff_std or
                        abs(star.diff_v) > nsigma * v_diff_std):
                        self.logger.debug(f'Star {star.pretty_name:9s} VMAG {star.vmag:6.3f} '
                                          f'U {star.u:8.3f}, V {star.v:8.3f} diff '
                                          f'{star.diff_u:6.3f}, {star.diff_v:6.3f} '
                                          'marked as an outlier')
                        star.conflicts = 'REFINEMENT OUTLIER'
                        marked_outlier = True

                if not marked_outlier:
                    self.logger.debug('No outliers marked - stopping refinement')
                    break

                star_model = NavModelStars(obs, star_list=new_star_list)
                star_model.create_model()

            # At this point star_model is the updated star model with various markings for
            # refinement so we want to replace the original star model with the updated one.
            self.nav_master.star_models[0] = star_model

            self._offset = refined_offset
            self._uncertainty = refined_sigma
            self._confidence = float(result['quality'])
            self.logger.info('Final offset: '
                             f'dU {self._offset[1]:.3f} +/- {self._uncertainty[1]:.3f}, '
                             f'dV {self._offset[0]:.3f} +/- {self._uncertainty[0]:.3f}')
            self.logger.info('Final confidence: '
                             f'{self._confidence:.3f}')

        self._metadata['offset'] = self._offset
        self._metadata['uncertainty'] = self._uncertainty
        self._metadata['confidence'] = self._confidence

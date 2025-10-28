from typing import Optional, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from .nav_technique import NavTechnique
from nav.config import Config
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

        log_level = self._config.general.get('nav_stars_log_level')
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
            model_img = star_model.model_img
            # plt.imshow(model_img)
            if star_model.blur_amount is not None:

                model_img = gaussian_blur_cov(model_img, star_model.blur_amount)
                # plt.figure()
                # plt.imshow(model_img)
                # plt.show()

            result = navigate_with_pyramid_kpeaks(obs.extdata,
                                                  star_model.model_img,
                                                  star_model.model_mask)
            # TODO Handle failure
            corr_offset = (-float(result['offset'][0]), -float(result['offset'][1]))

            self.logger.debug('Correlation offset: '
                              f'{corr_offset[0]:.2f}, {corr_offset[1]:.2f}')
            self.logger.debug('Correlation quality: '
                              f'{float(result['quality']):.2f}')

            img = obs.data
            psf = obs.star_psf()

            self.logger.info('Starting star position optimization process')
            u_diff_list = []
            v_diff_list = []
            for star in star_model.star_list:
                if star.conflicts:
                    continue
                psf_size = obs.star_psf_size(star)
                star_u = star.u - corr_offset[1]
                star_v = star.v - corr_offset[0]
                if (star_u < psf_size[1] or star_u > img.shape[1] - psf_size[1] or
                    star_v < psf_size[0] or star_v > img.shape[0] - psf_size[0]):
                    self.logger.debug(f'Star {star.unique_number} VMAG {star.vmag} outside image')
                    continue
                ret = psf.find_position(img, psf_size, (star_v, star_u), search_limit=(2.5, 2.5))
                if ret is None:
                    self.logger.info(f'Star {star.unique_number} VMAG {star.vmag} failed '
                                     'to find position')
                    continue
                opt_v, opt_u, opt_metadata = ret
                self.logger.debug(f'Star {star.unique_number:9d} VMAG {star.vmag:6.3f} '
                                  f'Searched at {star_u:8.3f}, {star_v:8.3f} '
                                  f'found at {opt_u:8.3f}, {opt_v:8.3f} '
                                  f'diff {opt_u-star_u:6.3f}, {opt_v-star_v:6.3f}')
                # TODO Implement edge clipping for Voyager and Galileo
                # if opt_u < clip or opt_u > img.shape[1] - clip or opt_v < clip or
                # opt_v > img.shape[0] - clip:
                #     if verbose:
                #         print(f'Star {star.unique_number} VMAG {star.vmag} clipped')
                #     return False
                diff_u = float(opt_u-star_u)
                diff_v = float(opt_v-star_v)
                if abs(diff_u) > 1.5 or abs(diff_v) > 1.5:
                    self.logger.info(f'  Diff too large - ignoring star')
                    continue
                u_diff_list.append(diff_u)
                v_diff_list.append(diff_v)

            # TODO Handle empty lists
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

            self.logger.info(f'U diff: min {u_diff_min:5.2f}, max {u_diff_max:5.2f}, '
                             f'mean {u_diff_mean:5.2f}, std {u_diff_std:5.2f}, '
                             f'median {u_diff_median:5.2f}')
            self.logger.info(f'V diff: min {v_diff_min:5.2f}, max {v_diff_max:5.2f}, '
                             f'mean {v_diff_mean:5.2f}, std {v_diff_std:5.2f}'
                             f'median {v_diff_median:5.2f}')

            self._offset = corr_offset
            self._confidence = float(result['quality'])
            self.logger.info('Final offset: '
                             f'dU {self._offset[1]:.2f}, dV {self._offset[0]:.2f}')
            self.logger.info('Final confidence: '
                             f'{self._confidence:.2f}')

        self._metadata['offset'] = self._offset
        self._metadata['confidence'] = self._confidence

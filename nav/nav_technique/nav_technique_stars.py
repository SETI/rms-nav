from typing import Any

from nav.support.correlate import find_correlation_and_offset

from .nav_technique import NavTechnique


class NavTechniqueStars(NavTechnique):
    """Implements navigation technique using star field correlation.

    Parameters:
        *args: Variable length argument list passed to parent class
        **kwargs: Arbitrary keyword arguments passed to parent class
    """
    def __init__(self,
                 *args: Any,
                 **kwargs: Any) -> None:
        super().__init__(*args, logger_name='NavTechniqueStars', **kwargs)

    def navigate(self) -> None:
        """Performs navigation using star field correlation.

        Attempts to find correlation between observed image and star model,
        computing the offset if successful.
        """

        with self.logger.open('NAVIGATION PASS: STARS CORRELATION'):
            obs = self.nav_master.obs
            star_models = self.nav_master.star_models
            if len(star_models) == 0:
                self.logger.info('No star models available')
                return
            if len(star_models) > 1:
                self.logger.fatal('More than one star model available')
                return
            final_model = star_models[0].model_img

            assert final_model is not None

            model_offset_list = find_correlation_and_offset(
                obs.extdata, final_model, extfov_margin_vu=obs.extfov_margin_vu,
                logger=self.logger)

            if len(model_offset_list) > 0:
                offset = (-model_offset_list[0][0][0], -model_offset_list[0][0][1])
                self._offset = offset
                self._confidence = model_offset_list[0][1]
                self.logger.info('Star navigation technique final offset: '
                                 f'{offset[0]:.2f}, {offset[1]:.2f}')
            else:
                self._offset = None
                self._confidence = None
                self.logger.info('Star navigation technique failed')

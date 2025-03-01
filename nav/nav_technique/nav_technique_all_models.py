from typing import Any

from nav.support.correlate import find_correlation_and_offset

from .nav_technique import NavTechnique


class NavTechniqueAllModels(NavTechnique):
    """Implements navigation technique using correlation across all available models.
    
    Parameters:
        *args: Variable length argument list passed to parent class
        **kwargs: Arbitrary keyword arguments passed to parent class
    """
    def __init__(self,
                 *args: Any,
                 **kwargs: Any) -> None:
        super().__init__(*args, logger_name='NavTechniqueCorrelation', **kwargs)

    def navigate(self) -> None:
        """Performs navigation using correlation across all available models.
        
        Attempts to find correlation between observed image and combined model,
        computing the offset if successful.
        """


        with self.logger.open('NAVIGATION PASS: ALL MODELS CORRELATION'):
            obs = self.nav_master.obs
            final_model = self.nav_master.combined_model

            model_offset_list = find_correlation_and_offset(
                obs.extdata, final_model, extfov_margin_vu=obs.extfov_margin_vu,
                logger=self.logger)

            if len(model_offset_list) > 0:
                offset = (-model_offset_list[0][0][0], -model_offset_list[0][0][1])
                self._offset = offset
                self.logger.info('All models navigation technique final offset: '
                                 f'{offset[0]:.2f}, {offset[1]:.2f}')
            else:
                self._offset = None
                self.logger.info('All models natechnique failed')

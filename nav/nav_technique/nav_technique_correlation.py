from typing import Any

from nav.support.correlate import find_correlation_and_offset

from .nav_technique import NavTechnique


class NavTechniqueCorrelation(NavTechnique):
    def __init__(self,
                 nav_master: 'NavMaster',
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, logger_name='NavTechniqueCorrelation', **kwargs)

        self._nav_master = nav_master

    @property
    def nav_master(self) -> 'NavMaster':
        return self._nav_master

    def navigate(self) -> None:
        obs = self.nav_master.obs
        final_model = self.nav_master._create_combined_model()

        model_offset_list = find_correlation_and_offset(
            obs.extdata, final_model, extfov_margin_vu=obs.extfov_margin_vu)

        if len(model_offset_list) > 0:
            offset = (-model_offset_list[0][0][0], -model_offset_list[0][0][1])
            self._offset = offset
            self.logger.info(
                f'Correlation technique final offset: {offset[0]:.2f}, {offset[1]:.2f}')
        else:
            self._offset = None
            self.logger.info('Correlation technique failed')

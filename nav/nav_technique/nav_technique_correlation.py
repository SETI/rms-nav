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

    def navigate(self):
        obs = self.nav_master.obs
        final_model = self.nav_master._create_combined_model()

        model_offset_list = find_correlation_and_offset(obs.extdata, final_model)

        offset = (-model_offset_list[0][0][1], -model_offset_list[0][0][0])
        self._offset = offset

        self.logger.info('Correlation technique final offset: {offset:.2f}, {offset:.2f}')

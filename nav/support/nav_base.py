from typing import Optional

from pdslogger import PdsLogger

from nav.config import Config, DEFAULT_CONFIG


class NavBase:
    def __init__(self,
                 *,
                 config: Optional[Config] = None,
                 logger_name: Optional[str] = None) -> None:

        self._config = config or DEFAULT_CONFIG
        self._logger = self._config.logger
        if logger_name is not None:
            self._logger = self._logger.get_logger(logger_name)

    @property
    def config(self) -> Config:
        return self._config

    @property
    def logger(self) -> PdsLogger:
        return self._logger

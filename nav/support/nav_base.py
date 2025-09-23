from typing import Optional

from pdslogger import PdsLogger

from nav.config import Config, DEFAULT_CONFIG, DEFAULT_LOGGER


class NavBase:
    """Provides a base class with configuration and logging capabilities for navigation components.

    Serves as the foundation for navigation-related classes by providing common functionality
    for configuration management and logging.

    Parameters:
        config: Configuration object for this instance. Uses DEFAULT_CONFIG if not provided.
        logger_name: Name for the logger instance. If provided, creates a child logger.
    """

    def __init__(self,
                 *,
                 config: Optional[Config] = None,
                 logger_name: Optional[str] = None) -> None:

        self._config = config or DEFAULT_CONFIG
        self._logger = DEFAULT_LOGGER
        if logger_name is not None:
            self._logger = self._logger.get_logger(logger_name, lognames=False, digits=3)

    @property
    def config(self) -> Config:
        """Returns the configuration object associated with this instance."""
        return self._config

    @property
    def logger(self) -> PdsLogger:
        """Returns the logger instance associated with this object."""
        return self._logger

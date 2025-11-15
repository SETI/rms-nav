from typing import Any, Optional

from pdslogger import PdsLogger

from nav.config import Config, DEFAULT_CONFIG, DEFAULT_LOGGER


class NavBase:
    """Provides a base class with configuration and logging capabilities for navigation components.

    Serves as the foundation for navigation-related classes by providing common functionality
    for configuration management and logging.

    Parameters:
        config: Configuration object for this instance. Uses DEFAULT_CONFIG if not provided.
    """

    def __init__(self,
                 *,
                 config: Optional[Config] = None,
                 **kwargs: Any) -> None:
        """Initializes a new NavBase instance.

        Parameters:
            config: Configuration object to use. If None, uses DEFAULT_CONFIG.
            **kwargs: Additional keyword arguments used by subclasses.
        """

        self._config = config or DEFAULT_CONFIG
        self._logger = DEFAULT_LOGGER

    @property
    def config(self) -> Config:
        """Returns the configuration object associated with this instance."""
        return self._config

    @property
    def logger(self) -> PdsLogger:
        """Returns the logger instance associated with this object."""
        return self._logger

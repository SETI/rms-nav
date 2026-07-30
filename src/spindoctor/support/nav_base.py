from typing import Any, ClassVar

from pdslogger import PdsLogger

from spindoctor.config import DEFAULT_CONFIG, IMAGE_LOGGER, MAIN_LOGGER, Config, LogRole


class NavBase:
    """Provides a base class with configuration and logging capabilities for navigation components.

    Serves as the foundation for navigation-related classes by providing common functionality
    for configuration management and logging.

    Which logger a subclass writes to is declared, not inferred.  Most
    navigation components work on one image and keep the default
    ``LogRole.IMAGE``; a component whose work spans a whole run, such as
    enumerating a dataset, sets ``log_role = LogRole.MAIN`` so its records go
    to the run's log rather than into whichever image happens to be open.

    Class attributes:
        log_role: Which logger this component's records belong to.

    Parameters:
        config: Configuration object for this instance. Uses DEFAULT_CONFIG if not provided.
    """

    log_role: ClassVar[LogRole] = LogRole.IMAGE

    def __init__(self, *, config: Config | None = None, **kwargs: Any) -> None:
        """Initializes a new NavBase instance.

        Parameters:
            config: Configuration object to use. If None, uses DEFAULT_CONFIG.
            **kwargs: Additional keyword arguments used by subclasses.
        """

        self._config = config or DEFAULT_CONFIG
        self._logger = MAIN_LOGGER if self.log_role is LogRole.MAIN else IMAGE_LOGGER

    @property
    def config(self) -> Config:
        """Returns the configuration object associated with this instance."""
        return self._config

    @property
    def logger(self) -> PdsLogger:
        """Returns the logger instance associated with this object."""
        return self._logger

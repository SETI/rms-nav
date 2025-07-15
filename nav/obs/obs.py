from abc import ABC
from typing import Optional, TYPE_CHECKING

from pdslogger import PdsLogger

from nav.config import Config, DEFAULT_CONFIG, DEFAULT_LOGGER

if TYPE_CHECKING:
    from nav.inst import Inst


class Obs(ABC):
    """Represents an observation in the navigation system.

    This abstract base class provides common functionality for all observation types,
    including configuration and logging capabilities.
    """

    def __init__(self,
                 *,
                 config: Optional[Config] = None,
                 logger_name: Optional[str] = None) -> None:
        """Initializes a new observation instance.

        Parameters:
            config: Configuration object to use. If None, uses DEFAULT_CONFIG.
            logger_name: Name for the logger. If provided, creates a child logger with this name.
        """

        self._config = config or DEFAULT_CONFIG
        self._logger = DEFAULT_LOGGER
        self._inst: Optional[Inst] = None

        if logger_name is not None:
            self._logger = self._logger.get_logger(logger_name)

    @property
    def config(self) -> Config:
        """Returns the configuration object associated with this observation."""

        return self._config

    @property
    def logger(self) -> PdsLogger:
        """Returns the logger instance for this observation."""

        return self._logger

    def set_inst(self, inst: 'Inst') -> None:
        """Sets the instrument associated with this observation.

        Parameters:
            inst: The instrument instance to associate with this observation.
        """

        self._inst = inst

    @property
    def inst(self) -> 'Inst':
        """Returns the instrument associated with this observation."""

        # Can't get casting to work here with Inst having a circular import
        return self._inst  # type: ignore

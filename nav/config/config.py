from logging import Logger
from pathlib import Path
from typing import Any, Optional, cast

from pdslogger import PdsLogger, STDOUT_HANDLER
from ruamel.yaml import YAML

from nav.support.attrdict import AttrDict


class Config:
    """Manages configuration settings for the navigation system.

    This class handles loading, updating, and accessing configuration settings from YAML files.
    It provides access to various configuration sections through properties and methods.
    """

    def __init__(self) -> None:
        """Initializes a new Config instance with empty configuration containers."""

        self._logger: PdsLogger | None = None
        self._config_dict: dict[str, Any] = {}
        self._config_offset: dict[str, Any] = AttrDict({})
        self._config_bodies: dict[str, Any] = AttrDict({})
        self._config_general: dict[str, Any] = AttrDict({})
        self._config_rings: dict[str, Any] = AttrDict({})
        self._config_stars: dict[str, Any] = AttrDict({})
        self._config_titan: dict[str, Any] = AttrDict({})
        self._config_bootstrap: dict[str, Any] = AttrDict({})

    def set_logger(self,
                   logger: Logger) -> None:
        """Sets the logger for this configuration instance.

        Parameters:
            logger: The logger instance to use for logging configuration-related messages.
        """

        self._logger = PdsLogger.as_pdslogger(logger)

    def _maybe_read_config(self) -> None:
        """Reads the configuration file if it hasn't been read already.

        Ensures configuration is loaded before attempting to access configuration values.
        """

        if not self._config_dict:
            self.read_config()

    def _update_attrdicts(self) -> None:
        """Updates all attribute dictionaries from the main configuration dictionary.

        Converts dictionary sections to AttrDict instances for convenient attribute-style access.
        """

        self._config_offset = AttrDict(self._config_dict['offset'])
        self._config_bodies = AttrDict(self._config_dict['bodies'])
        self._config_general = AttrDict(self._config_dict['general'])
        self._config_rings = AttrDict(self._config_dict['rings'])
        self._config_stars = AttrDict(self._config_dict['stars'])
        self._config_titan = AttrDict(self._config_dict['titan'])
        self._config_bootstrap = AttrDict(self._config_dict['bootstrap'])

    def read_config(self,
                    config_path: Optional[str | Path] = None) -> None:
        """Reads configuration from the specified YAML file.

        Parameters:
            config_path: Path to the configuration file. If None, uses the default config file.
        """

        if config_path is None:
            config_path = Path(__file__).resolve().parent / 'default_config.yaml'
        yaml = YAML(typ='safe')
        self._config_dict = yaml.load(config_path)
        self._update_attrdicts()

    def update_config(self,
                      config_path: str | Path) -> None:
        """Updates the current configuration with values from the specified YAML file.

        Parameters:
            config_path: Path to the configuration file containing update values.
        """

        self._maybe_read_config()
        yaml = YAML(typ='safe')
        new_config = yaml.load(config_path)
        for key in ('planets',
                    'satellites',
                    'fuzzy_satellites',
                    'ring_satellites',
                    'offset',
                    'bodies',
                    'rings',
                    'stars',
                    'titan',
                    'bootstrap'):
            if key in new_config:
                self._config_dict[key].update(new_config[key])
        self._update_attrdicts()

    @property
    def logger(self) -> PdsLogger:
        """Returns the logger instance, creating one if it doesn't exist."""

        if self._logger is None:
            self._logger = PdsLogger('nav', lognames=False, digits=3)
            self._logger.add_handler(STDOUT_HANDLER)
            self._logger.info('*** START OF LOG ***')
        return self._logger

    @property
    def planets(self) -> list[str]:
        """Returns the list of configured planet names."""

        self._maybe_read_config()
        return cast(list[str], self._config_dict['planets'])

    def satellites(self,
                   planet: str) -> list[str]:
        """Returns the list of satellites for the specified planet.

        Parameters:
            planet: The name of the planet to get satellites for.

        Returns:
            A list of satellite names for the specified planet.
        """

        self._maybe_read_config()
        return cast(list[str], self._config_dict['satellites'].get(planet.upper(), []))

    def fuzzy_satellites(self,
                         planet: str) -> list[str]:
        """Returns the list of fuzzy satellites for the specified planet.

        Parameters:
            planet: The name of the planet to get fuzzy satellites for.

        Returns:
            A list of fuzzy satellite names for the specified planet.
        """

        self._maybe_read_config()
        return cast(list[str], self._config_dict['fuzzy_satellites'][planet.upper()])

    def ring_satellites(self,
                        planet: str) -> list[str]:
        """Returns the list of ring satellites for the specified planet.

        Parameters:
            planet: The name of the planet to get ring satellites for.

        Returns:
            A list of ring satellite names for the specified planet.
        """

        self._maybe_read_config()
        return cast(list[str], self._config_dict['ring_satellites'][planet.upper()])

    @property
    def general(self) -> Any:
        """Returns the general configuration settings."""

        self._maybe_read_config()
        return self._config_general

    @property
    def offset(self) -> Any:
        """Returns the offset configuration settings."""

        self._maybe_read_config()
        return self._config_offset

    @property
    def bodies(self) -> Any:
        """Returns the celestial bodies configuration settings."""

        self._maybe_read_config()
        return self._config_bodies

    @property
    def rings(self) -> Any:
        """Returns the planetary rings configuration settings."""

        self._maybe_read_config()
        return self._config_rings

    @property
    def stars(self) -> Any:
        """Returns the stars configuration settings."""

        self._maybe_read_config()
        return self._config_stars

    @property
    def titan(self) -> Any:
        """Returns the Titan-specific configuration settings."""

        self._maybe_read_config()
        return self._config_titan

    @property
    def bootstrap(self) -> Any:
        """Returns the bootstrap configuration settings."""

        self._maybe_read_config()
        return self._config_bootstrap


DEFAULT_CONFIG = Config()

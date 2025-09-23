from pathlib import Path
from typing import Any, Optional, cast

from ruamel.yaml import YAML

from nav.support.attrdict import AttrDict


class Config:
    """Manages configuration settings for the navigation system.

    This class handles loading, updating, and accessing configuration settings from YAML files.
    It provides access to various configuration sections through properties and methods.
    """

    def __init__(self) -> None:
        """Initializes a new Config instance with empty configuration containers."""

        self._config_dict: dict[str, Any] = {}
        self._config_offset: dict[str, Any] = AttrDict({})
        self._config_bodies: dict[str, Any] = AttrDict({})
        self._config_general: dict[str, Any] = AttrDict({})
        self._config_rings: dict[str, Any] = AttrDict({})
        self._config_stars: dict[str, Any] = AttrDict({})
        self._config_titan: dict[str, Any] = AttrDict({})
        self._config_bootstrap: dict[str, Any] = AttrDict({})

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

    def _load_yaml(self,
                   config_path: str | Path) -> dict[str, Any]:
        """Loads a YAML file and returns a dictionary mapping.
        """

        yaml = YAML(typ='safe')
        with open(config_path, 'r', encoding='utf-8') as f:
            loaded = yaml.load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f'Config "{config_path}" did not parse to a dictionary mapping')
        return loaded

    def read_config(self,
                    config_path: Optional[str | Path] = None,
                    reread: bool = False) -> None:
        """Reads configuration from the specified YAML file.

        Parameters:
            config_path: Path to the configuration file. If None, uses the default config files.
            reread: Whether to reread the configuration file if it has already been read.
        """

        if not reread and self._config_dict:
            return

        if config_path is None:
            config_dir = Path(__file__).resolve().parent.parent / 'config_files'
            for filename in sorted(config_dir.glob('*.yaml')):
                self.update_config(filename, read_default=False)
            return

        self._config_dict = self._load_yaml(config_path)
        self._update_attrdicts()

    def update_config(self,
                      config_path: str | Path,
                      read_default: bool = True) -> None:
        """Updates the current configuration with values from the specified YAML file.

        Parameters:
            config_path: Path to the configuration file containing update values.
            read_default: Whether to read the default configuration file if no config
                has been previously read.
        """

        if read_default:
            self.read_config()
        new_config = self._load_yaml(config_path)
        for key in new_config:
            if key in self._config_dict:
                self._config_dict[key].update(new_config[key])
            else:
                self._config_dict[key] = new_config[key]
        self._update_attrdicts()

    @property
    def planets(self) -> list[str]:
        """Returns the list of configured planet names."""

        self.read_config()
        return cast(list[str], self._config_dict.get('planets', []))

    def satellites(self,
                   planet: str) -> list[str]:
        """Returns the list of satellites for the specified planet.

        Parameters:
            planet: The name of the planet to get satellites for.

        Returns:
            A list of satellite names for the specified planet.
        """

        self.read_config()
        return cast(list[str], self._config_dict['satellites', {}].get(planet.upper(), []))

    def fuzzy_satellites(self,
                         planet: str) -> list[str]:
        """Returns the list of fuzzy satellites for the specified planet.

        Parameters:
            planet: The name of the planet to get fuzzy satellites for.

        Returns:
            A list of fuzzy satellite names for the specified planet.
        """

        self.read_config()
        return cast(list[str], self._config_dict['fuzzy_satellites', {}][planet.upper(), []])

    def ring_satellites(self,
                        planet: str) -> list[str]:
        """Returns the list of ring satellites for the specified planet.

        Parameters:
            planet: The name of the planet to get ring satellites for.

        Returns:
            A list of ring satellite names for the specified planet.
        """

        self.read_config()
        return cast(list[str], self._config_dict['ring_satellites', {}][planet.upper(), []])

    @property
    def general(self) -> Any:
        """Returns the general configuration settings."""

        self.read_config()
        return self._config_general

    @property
    def offset(self) -> Any:
        """Returns the offset configuration settings."""

        self.read_config()
        return self._config_offset

    @property
    def bodies(self) -> Any:
        """Returns the celestial bodies configuration settings."""

        self.read_config()
        return self._config_bodies

    @property
    def rings(self) -> Any:
        """Returns the planetary rings configuration settings."""

        self.read_config()
        return self._config_rings

    @property
    def stars(self) -> Any:
        """Returns the stars configuration settings."""

        self.read_config()
        return self._config_stars

    @property
    def titan(self) -> Any:
        """Returns the Titan-specific configuration settings."""

        self.read_config()
        return self._config_titan

    @property
    def bootstrap(self) -> Any:
        """Returns the bootstrap configuration settings."""

        self.read_config()
        return self._config_bootstrap


DEFAULT_CONFIG = Config()

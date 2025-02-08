from logging import Logger
from pathlib import Path
from typing import Any, Optional, cast

from pdslogger import PdsLogger
from ruamel.yaml import YAML


class Config:
    _config: dict[str, Any]

    def __init__(self) -> None:
        self._logger: PdsLogger | None = None
        self._config_dict: dict[str, Any] = {}

    def set_logger(self,
                   logger: Logger) -> None:
        self._logger = PdsLogger.as_pdslogger(logger)

    def _maybe_read_config(self) -> None:
        if not self._config_dict:
            self.read_config()

    def read_config(self,
                    config_path: Optional[str | Path] = None) -> None:
        if config_path is None:
            config_path = Path(__file__).resolve().parent / 'default_config.yaml'
        yaml = YAML(typ='safe')
        self._config_dict = yaml.load(config_path)

    def update_config(self,
                      config_path: str | Path) -> None:
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

    @property
    def logger(self) -> PdsLogger:
        if self._logger is None:
            self._logger = PdsLogger('nav', lognames=False)
            self._logger.info('Starting')
        return self._logger

    @property
    def planets(self) -> list[str]:
        self._maybe_read_config()
        return cast(list[str], self._config_dict['planets'])

    def satellites(self,
                   planet: str) -> list[str]:
        self._maybe_read_config()
        return cast(list[str], self._config_dict['satellites'][planet.upper()])

    def fuzzy_satellites(self,
                         planet: str) -> list[str]:
        self._maybe_read_config()
        return cast(list[str], self._config_dict['fuzzy_satellites'][planet.upper()])

    def ring_satellites(self,
                        planet: str) -> list[str]:
        self._maybe_read_config()
        return cast(list[str], self._config_dict['ring_satellites'][planet.upper()])

    def general(self,
               key: str) -> Any:
        self._maybe_read_config()
        return self._config_dict['general'][key]

    @property
    def general_config(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._config_dict['general'])

    def offset(self,
               key: str) -> Any:
        self._maybe_read_config()
        return self._config_dict['offset'][key]

    @property
    def offset_config(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._config_dict['offset'])

    def bodies(self,
               key: str) -> Any:
        self._maybe_read_config()
        return self._config_dict['bodies'][key]

    @property
    def bodies_config(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._config_dict['bodies'])

    def rings(self,
              key: str) -> Any:
        self._maybe_read_config()
        return self._config_dict['rings'][key]

    @property
    def rings_config(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._config_dict['rings'])

    def stars(self,
              key: str) -> Any:
        self._maybe_read_config()
        return self._config_dict['stars'][key]

    @property
    def stars_config(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._config_dict['stars'])

    def titan(self,
              key: str) -> Any:
        self._maybe_read_config()
        return self._config_dict['titan'][key]

    @property
    def titan_config(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._config_dict['titan'])

    def bootstrap(self,
                  key: str) -> Any:
        self._maybe_read_config()
        return self._config_dict['bootstrap'][key]

    @property
    def bootstrap_config(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._config_dict['bootstrap'])


DEFAULT_CONFIG = Config()

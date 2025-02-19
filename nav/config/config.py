from logging import Logger
from pathlib import Path
from typing import Any, Optional, cast

from pdslogger import PdsLogger
from ruamel.yaml import YAML


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Config:
    def __init__(self) -> None:
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
        self._logger = PdsLogger.as_pdslogger(logger)

    def _maybe_read_config(self) -> None:
        if not self._config_dict:
            self.read_config()

    def _update_attrdicts(self):
        self._config_offset: dict[str, Any] = AttrDict(self._config_dict['offset'])
        self._config_bodies: dict[str, Any] = AttrDict(self._config_dict['bodies'])
        self._config_general: dict[str, Any] = AttrDict(self._config_dict['general'])
        self._config_rings: dict[str, Any] = AttrDict(self._config_dict['rings'])
        self._config_stars: dict[str, Any] = AttrDict(self._config_dict['stars'])
        self._config_titan: dict[str, Any] = AttrDict(self._config_dict['titan'])
        self._config_bootstrap: dict[str, Any] = AttrDict(self._config_dict['bootstrap'])

    def read_config(self,
                    config_path: Optional[str | Path] = None) -> None:
        if config_path is None:
            config_path = Path(__file__).resolve().parent / 'default_config.yaml'
        yaml = YAML(typ='safe')
        self._config_dict = yaml.load(config_path)
        self._update_attrdicts()

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
        self._update_attrdicts()

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

    @property
    def general(self) -> Any:
        self._maybe_read_config()
        return self._config_general

    @property
    def offset(self) -> Any:
        self._maybe_read_config()
        return self._config_offset

    @property
    def bodies(self) -> Any:
        self._maybe_read_config()
        return self._config_bodies

    @property
    def rings(self) -> Any:
        self._maybe_read_config()
        return self._config_rings

    @property
    def stars(self) -> Any:
        self._maybe_read_config()
        return self._config_stars

    @property
    def titan(self) -> Any:
        self._maybe_read_config()
        return self._config_titan

    @property
    def bootstrap(self) -> Any:
        self._maybe_read_config()
        return self._config_bootstrap


DEFAULT_CONFIG = Config()

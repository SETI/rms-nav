from abc import ABC, abstractmethod
from typing import Any, Optional

import oops
from pdslogger import PdsLogger

from nav.annotation import Annotation
from nav.config import Config, DEFAULT_CONFIG
from nav.inst import Inst
from nav.obs import ObsSnapshot
from nav.util.types import NDArrayFloatType, NDArrayBoolType


class NavModel(ABC):
    def __init__(self,
                 obs: ObsSnapshot,
                 *,
                 config: Optional[Config] = None,
                 logger_name: Optional[str] = None) -> None:

        self._config = config or DEFAULT_CONFIG
        self._obs = obs

        self._logger = self._config.logger
        if logger_name is not None:
            self._logger = self._logger.get_logger(logger_name)

        self._model = None
        self._model_mask = None
        self._metadata = None
        self._annotations = None

    @property
    def config(self) -> Config:
        return self._config

    @property
    def logger(self) -> PdsLogger:
        return self._logger

    @property
    def obs(self) -> ObsSnapshot:
        return self._obs

    @property
    def inst(self) -> Inst:
        return self.obs.inst

    @property
    def model(self) -> NDArrayFloatType:
        return self._model

    @property
    def model_mask(self) -> NDArrayBoolType:
        return self._model_mask

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @property
    def annotations(self) -> list[Annotation] | None:
        return self._annotations

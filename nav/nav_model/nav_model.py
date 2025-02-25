from abc import ABC
from typing import Any, Optional

from pdslogger import PdsLogger

from nav.annotation import Annotations
from nav.config import Config
from nav.inst import Inst
from nav.obs import ObsSnapshot
from nav.support.nav_base import NavBase
from nav.support.types import NDArrayFloatType, NDArrayBoolType


class NavModel(ABC, NavBase):
    def __init__(self,
                 obs: ObsSnapshot,
                 *,
                 config: Optional[Config] = None,
                 logger_name: Optional[str] = None) -> None:

        super().__init__(config=config, logger_name=logger_name)

        self._obs = obs
        self._model_img: NDArrayFloatType | None = None
        self._model_mask: NDArrayBoolType | None = None
        self._metadata: dict[str, Any] = {}
        self._range: NDArrayFloatType | None = None
        self._annotations: Annotations | None = None

    @property
    def obs(self) -> ObsSnapshot:
        return self._obs

    @property
    def inst(self) -> Inst:
        return self.obs.inst

    @property
    def model_img(self) -> NDArrayFloatType | None:
        return self._model_img

    @property
    def model_mask(self) -> NDArrayBoolType | None:
        return self._model_mask

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @property
    def range(self) -> NDArrayFloatType | float | None:
        return self._range

    @property
    def annotations(self) -> Annotations | None:
        return self._annotations

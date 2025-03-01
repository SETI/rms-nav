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
    """Base class for navigation models used to generate synthetic images."""
    
    def __init__(self,
                 obs: ObsSnapshot,
                 *,
                 config: Optional[Config] = None,
                 logger_name: Optional[str] = None) -> None:
        """Initializes a navigation model with observation data.
        
        Parameters:
            obs: Observation snapshot containing image and metadata.
            config: Configuration object to use. If None, uses DEFAULT_CONFIG.
            logger_name: Name for the logger. If None, uses class name.
        """

        super().__init__(config=config, logger_name=logger_name)

        self._obs = obs
        self._model_img: NDArrayFloatType | None = None
        self._model_mask: NDArrayBoolType | None = None
        self._metadata: dict[str, Any] = {}
        self._range: NDArrayFloatType | None = None
        self._annotations: Annotations | None = None

    @property
    def obs(self) -> ObsSnapshot:
        """Returns the observation snapshot associated with this model."""
        return self._obs

    @property
    def inst(self) -> Inst:
        """Returns the instrument associated with the observation."""
        return self.obs.inst

    @property
    def model_img(self) -> NDArrayFloatType | None:
        """Returns the model image array if available, or None if not generated yet."""
        return self._model_img

    @property
    def model_mask(self) -> NDArrayBoolType | None:
        """Returns the model mask array if available, or None if not generated yet."""
        return self._model_mask

    @property
    def metadata(self) -> dict[str, Any]:
        """Returns the metadata dictionary associated with this model."""
        return self._metadata

    @property
    def range(self) -> NDArrayFloatType | float | None:
        """Returns the range to the target body, either as a single value or per-pixel array."""
        return self._range

    @property
    def annotations(self) -> Annotations | None:
        """Returns the annotations object associated with this model if available."""
        return self._annotations

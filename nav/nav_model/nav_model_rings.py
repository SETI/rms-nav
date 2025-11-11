from typing import Optional

# import numpy as np

from oops import Observation
# import polymath

# from nav.annotation import (Annotation,
#                             Annotations,
#                             AnnotationTextInfo,
#                             TEXTINFO_LEFT,
#                             TEXTINFO_RIGHT,
#                             TEXTINFO_BOTTOM,
#                             TEXTINFO_TOP)
# from nav.support.types import NDArrayFloatType

from nav.config import Config
from .nav_model import NavModel


class NavModelRings(NavModel):
    def __init__(self,
                 name: str,
                 obs: Observation,
                 *,
                 config: Optional[Config] = None) -> None:
        """Creates a navigation model for planetary rings.

        Parameters:
            name: The name of the model.
            obs: The Observation object containing image data.
            config: Configuration object to use. If None, uses DEFAULT_CONFIG.
        """

        super().__init__(name, obs, config=config)

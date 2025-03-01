from typing import Any, Optional, cast

import numpy as np

from oops import Observation
import polymath

from nav.annotation import (Annotation,
                            Annotations,
                            AnnotationTextInfo,
                            TEXTINFO_LEFT,
                            TEXTINFO_RIGHT,
                            TEXTINFO_BOTTOM,
                            TEXTINFO_TOP)
from nav.support.types import NDArrayFloatType

from .nav_model import NavModel


class NavModelTitan(NavModel):
    def __init__(self,
                 obs: Observation,
                 **kwargs: Any) -> None:
        """Creates a navigation model for Titan with specialized features.
        
        Parameters:
            obs: The observation object containing the image data.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """

        super().__init__(obs, logger_name='NavModelTitan', **kwargs)

        self._obs = obs

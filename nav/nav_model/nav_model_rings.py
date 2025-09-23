from typing import Any

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

from .nav_model import NavModel


class NavModelRings(NavModel):
    def __init__(self,
                 obs: Observation,
                 **kwargs: Any) -> None:
        """Creates a navigation model for planetary rings.

        Parameters:
            obs: The Observation object containing image data.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """

        super().__init__(obs, logger_name='NavModelRings', **kwargs)

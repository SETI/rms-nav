from typing import Any

from oops import Observation

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

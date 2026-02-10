from typing import Optional

from oops import Observation

from nav.config import Config
from .nav_model import NavModel


class NavModelTitan(NavModel):
    def __init__(
        self, name: str, obs: Observation, *, config: Optional[Config] = None
    ) -> None:
        """Creates a navigation model for Titan with specialized features.

        Parameters:
            name: The name of the model.
            obs: The observation object containing the image data.
            config: Configuration object to use. If None, uses DEFAULT_CONFIG.
        """
        super().__init__(name, obs, config=config)

        self._obs = obs

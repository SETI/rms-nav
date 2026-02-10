from abc import ABC
from typing import Any, Optional

from nav.config import Config
from nav.support.nav_base import NavBase


class Obs(ABC, NavBase):
    """Represents an observation in the navigation system.

    This abstract base class provides common functionality for all observation types,
    including configuration and logging capabilities.
    """

    def __init__(self, *, config: Optional[Config] = None, **kwargs: Any) -> None:
        """Initializes a new observation instance.

        Parameters:
            config: Configuration object to use. If None, uses DEFAULT_CONFIG.
            **kwargs: Additional keyword arguments used by subclasses.
        """
        super().__init__(config=config, **kwargs)

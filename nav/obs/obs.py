from abc import ABC
from typing import Optional, TYPE_CHECKING

from nav.config import Config
from nav.support.nav_base import NavBase

if TYPE_CHECKING:
    from nav.inst import Inst


class Obs(ABC, NavBase):
    """Represents an observation in the navigation system.

    This abstract base class provides common functionality for all observation types,
    including configuration and logging capabilities.
    """

    def __init__(self,
                 *,
                 config: Optional[Config] = None) -> None:
        """Initializes a new observation instance.

        Parameters:
            config: Configuration object to use. If None, uses DEFAULT_CONFIG.
        """

        super().__init__(config=config)

        self._inst: Optional[Inst] = None

    def set_inst(self, inst: 'Inst') -> None:
        """Sets the instrument associated with this observation.

        Parameters:
            inst: The instrument instance to associate with this observation.
        """

        self._inst = inst

    @property
    def inst(self) -> 'Inst':
        """Returns the instrument associated with this observation."""

        if self._inst is None:
            raise ValueError('Instrument not set')

        return self._inst

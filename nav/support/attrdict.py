from typing import Any


class AttrDict(dict[str, Any]):
    """Implements a dictionary that allows attribute-style access to its key-value pairs.

    A dictionary subclass that exposes its keys as attributes, allowing dict items to be
    accessed using attribute notation (dict.key) in addition to the normal dictionary
    lookup (dict[key]).

    Parameters:
        *args: Variable length argument list passed to dict constructor.
        **kwargs: Arbitrary keyword arguments passed to dict constructor.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

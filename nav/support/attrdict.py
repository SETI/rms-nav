from typing import Any, no_type_check


@no_type_check
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

    # This is a stupid thing to do, but it's necessary to avoid mypy from complaining
    # about missing attributes. mypy ignores attributes for classes that have a
    # __getattr__ method.
    @no_type_check
    def __getattr__(self, name: str) -> Any:
        return super().__getattr__(name)

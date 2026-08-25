from typing import Any, no_type_check


@no_type_check
class AttrDict(dict[str, Any]):
    """Implements a dictionary that allows attribute-style access to its key-value pairs.

    A dictionary subclass that exposes its keys as attributes, allowing dict items to be
    accessed using attribute notation (dict.key) in addition to the normal dictionary
    lookup (dict[key]).

    Because the instance dict and the mapping are the same object, setting an attribute
    inserts a key.  The class therefore advertises the marker a library uses to recognize
    an object it must not annotate; see the comment on it below.

    Parameters:
        *args: Variable length argument list passed to dict constructor.
        **kwargs: Arbitrary keyword arguments passed to dict constructor.
    """

    # This class publishes its own dict as its ``__dict__``, so an attribute set
    # on an instance is a key inserted into the data.  That is the point for our
    # own writes and a hazard for anyone else's: a library that caches
    # bookkeeping on the objects it inspects silently adds keys that no
    # configuration file declares.  oops does exactly that -- building a
    # Backplane walks everything reachable from the observation, which includes
    # the shared Config and so every section of it -- and it skips any object
    # advertising this marker.  The marker lives on the class, so it is visible
    # to hasattr without occupying the mapping.
    #
    # The opt-out is advisory: it protects the mapping from a library that
    # honors it, not from one that writes unconditionally.  Remove it once oops
    # keeps its bookkeeping on its own objects.  See #552.
    _IS_IMMUTABLE = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    # This is a stupid thing to do, but it's necessary to avoid mypy from complaining
    # about missing attributes. mypy ignores attributes for classes that have a
    # __getattr__ method.
    @no_type_check
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(f"Attribute '{name}' not found") from exc

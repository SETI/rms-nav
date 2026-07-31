from typing import Any, ClassVar

from pdslogger import PdsLogger

from spindoctor.config import (
    DEFAULT_CONFIG,
    IMAGE_LOGGER,
    MAIN_LOGGER,
    Config,
    LogRole,
    log_key_for,
    log_levels,
)


class NavBase:
    """Provides a base class with configuration and logging capabilities for navigation components.

    Serves as the foundation for navigation-related classes by providing common functionality
    for configuration management and logging.

    Which logger a subclass writes to is declared, not inferred.  Most
    navigation components work on one image and keep the default
    ``LogRole.IMAGE``; a component whose work spans a whole run, such as
    enumerating a dataset, sets ``log_role = LogRole.MAIN`` so its records go
    to the run's log rather than into whichever image happens to be open.

    A subclass opens its own section of the log with :meth:`log_section`,
    which applies whatever level the configuration gives that component.  The
    component is named by ``log_key``, which defaults to the snake_case form
    of the class name.  Unlike ``log_role``, ``log_key`` is inherited
    normally, so a family of classes that should share one key -- every
    observation subclass, say -- declares it once on their base.

    Class attributes:
        log_role: Which logger this component's records belong to.
        log_key: The name this component is configured under, or None to derive it.

    Parameters:
        config: Configuration object for this instance. Uses DEFAULT_CONFIG if not provided.
    """

    log_role: ClassVar[LogRole] = LogRole.IMAGE
    log_key: ClassVar[str | None] = None

    def __init__(self, *, config: Config | None = None, **kwargs: Any) -> None:
        """Initializes a new NavBase instance.

        Parameters:
            config: Configuration object to use. If None, uses DEFAULT_CONFIG.
            **kwargs: Additional keyword arguments used by subclasses.
        """

        self._config = config or DEFAULT_CONFIG
        self._logger = MAIN_LOGGER if self.log_role is LogRole.MAIN else IMAGE_LOGGER

    @property
    def config(self) -> Config:
        """Returns the configuration object associated with this instance."""
        return self._config

    @property
    def logger(self) -> PdsLogger:
        """Returns the logger instance associated with this object."""
        return self._logger

    @property
    def resolved_log_key(self) -> str:
        """The name this component is configured under.

        Returns:
            The declared :attr:`log_key`, else the key derived from the class
            name by :func:`spindoctor.config.log_key_for`.
        """
        return self.log_key if self.log_key is not None else log_key_for(type(self))

    def log_section(self, title: str, *args: Any, **kwargs: Any) -> Any:
        """Open a section of the log at this component's configured level.

        The level comes from the ``logging`` configuration section, so one
        component can be made verbose or quiet without affecting the rest.
        Use this rather than ``self.logger.open`` so that the component's
        configured level is actually applied.  A main-role component takes the
        main level; it has no per-module key, because the run's log is not
        divided by component.

        Parameters:
            title: Title of the section.
            *args: Passed to ``PdsLogger.open``.
            **kwargs: Passed to ``PdsLogger.open``.  An explicit ``level``
                overrides the configured one, including ``None``, which asks
                the section to inherit the level enclosing it.

        Returns:
            The context manager returned by ``PdsLogger.open``.
        """
        if 'level' not in kwargs:
            # "level" is absent rather than None: an explicit None means
            # "inherit the enclosing section", which is a different request
            # from not having said anything, and setdefault cannot tell them
            # apart.
            levels = log_levels()
            kwargs['level'] = (
                levels.main_section_level()
                if self.log_role is LogRole.MAIN
                else levels.section_level_for(self.resolved_log_key)
            )
        return self.logger.open(title, *args, **kwargs)

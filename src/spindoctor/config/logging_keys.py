"""Key namespace and validation for the ``logging`` configuration section.

The ``logging`` section names each configurable module by a snake_case key.
For navigation techniques and models those keys are derived from the class
name by :func:`log_key_for`, so a key is always recoverable from the source
and never has to be maintained as a parallel list that can fall out of date.
A class may override its derived key by declaring a ``log_key`` class
attribute.

:func:`validate_logging_config` rejects an unrecognized module key, program
name, or level name when the configuration is loaded, so a typo fails at
startup instead of silently having no effect.  It lives here rather than on
:class:`~spindoctor.config.config.Config` because it needs the technique and
model registries, and those packages import ``spindoctor.config``; the
registry imports are therefore function-local and happen only when validation
runs.
"""

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import Config

__all__ = [
    'CATEGORY_KEYS',
    'LOGGER_KEYS',
    'LOG_LEVEL_NAMES',
    'OTHER_LOG_KEYS',
    'log_key_for',
    'model_log_keys',
    'technique_log_keys',
    'validate_logging_config',
]


LOG_LEVEL_NAMES = frozenset({'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', 'NONE'})
"""Level names accepted anywhere in the ``logging`` section."""

LOGGER_KEYS = frozenset({'main', 'image'})
"""The two per-logger global defaults."""

CATEGORY_KEYS = frozenset({'techniques', 'models', 'other'})
"""Categories that group per-module overrides."""

OTHER_LOG_KEYS = frozenset(
    {
        'annotate',
        'ensemble',
        'image_derivatives',
        'nav_correlate_all',
        'obs',
        'orchestrator',
        'provenance',
    }
)
"""Image-scoped modules that are neither a technique nor a model.

A per-image backend has no key here.  Each program drives at most one backend,
so that backend's verbosity is the program's ``image`` level.
"""

_CATEGORY_DEFAULT_KEY = 'default'
_PROGRAMS_KEY = 'programs'
_STRICT_SCOPE_KEY = 'strict_scope'

_CAMEL_BOUNDARY = re.compile(r'(?<!^)(?=[A-Z])')
_CLASS_NAME_PREFIXES = ('NavTechnique', 'NavModel')
_CLASS_NAME_SUFFIXES = ('Simulated', 'Nav')


def log_key_for(cls: type) -> str:
    """Return the ``logging`` configuration key naming ``cls``.

    The key is the class name with a ``NavTechnique`` or ``NavModel`` prefix
    removed, then a trailing ``Simulated`` and a trailing ``Nav`` removed, then
    converted from CamelCase to snake_case.  ``TitanHazeNav`` becomes
    ``titan_haze`` and ``NavTechniqueManual`` becomes ``manual``.  A simulated
    model shares its real sibling's key, because the two are one component
    differing only in where their inputs come from, so both ``NavModelRings``
    and ``NavModelRingsSimulated`` become ``rings``.

    A class whose derived key would be wrong may declare a ``log_key`` class
    attribute instead.  Only a key declared on ``cls`` itself is honored; an
    inherited one is ignored, so a subclass always gets a key derived from its
    own name unless it declares one.

    Parameters:
        cls: The technique or model class to name.

    Returns:
        The snake_case configuration key for ``cls``.
    """
    declared = cls.__dict__.get('log_key')
    if declared is not None:
        return str(declared)

    name = cls.__name__
    for prefix in _CLASS_NAME_PREFIXES:
        if name.startswith(prefix) and len(name) > len(prefix):
            name = name[len(prefix) :]
            break
    for suffix in _CLASS_NAME_SUFFIXES:
        if name.endswith(suffix) and len(name) > len(suffix):
            name = name[: -len(suffix)]
    return _CAMEL_BOUNDARY.sub('_', name).lower()


def technique_log_keys() -> frozenset[str]:
    """Return the configuration key for every navigation technique.

    Covers the autonomous techniques in ``NavTechnique._registry`` plus the
    interactive ``NavTechniqueManual``, which logs like the others but is
    deliberately excluded from the autonomous registry.

    Returns:
        The set of valid keys for the ``logging.techniques`` category.
    """
    from spindoctor.nav_technique.nav_technique import NavTechnique
    from spindoctor.nav_technique.nav_technique_manual import NavTechniqueManual

    keys = {log_key_for(cls) for cls in NavTechnique._registry}
    keys.add(log_key_for(NavTechniqueManual))
    return frozenset(keys)


def model_log_keys() -> frozenset[str]:
    """Return the configuration key for every navigation model.

    Returns:
        The set of valid keys for the ``logging.models`` category.  Simulated
        models collapse onto their real sibling's key, so the result names
        model families rather than classes.
    """
    from spindoctor.nav_model.nav_model import NavModel

    return frozenset(log_key_for(cls) for cls in NavModel._registry)


def _validate_level(value: Any, location: str) -> None:
    """Raise if ``value`` is not a recognized level name.

    Parameters:
        value: The configured value, expected to be a level-name string.
        location: Dotted path of the key, used in the error message.

    Raises:
        ValueError: If ``value`` is not a string or names no known level.
    """
    if not isinstance(value, str):
        raise ValueError(f'{location} must be a level name string, got {type(value).__name__}')
    if value.strip().upper() not in LOG_LEVEL_NAMES:
        raise ValueError(f'{location} is {value!r}; expected one of {sorted(LOG_LEVEL_NAMES)}')


def _validate_category(
    block: Any, category: str, valid_keys: frozenset[str], location: str
) -> None:
    """Raise if a category block holds an unknown module key or a bad level.

    Parameters:
        block: The category's configured value, expected to be a mapping.
        category: Category name, one of :data:`CATEGORY_KEYS`.
        valid_keys: Module keys this category accepts.
        location: Dotted path of the category, used in error messages.

    Raises:
        ValueError: If ``block`` is not a mapping, names an unknown module, or
            holds a value that is not a level name.
    """
    if not isinstance(block, dict):
        raise ValueError(f'{location} must be a mapping, got {type(block).__name__}')
    for key, value in block.items():
        if key != _CATEGORY_DEFAULT_KEY and key not in valid_keys:
            raise ValueError(
                f'{location}.{key} is not a known {category} key; '
                f'expected {_CATEGORY_DEFAULT_KEY!r} or one of {sorted(valid_keys)}'
            )
        _validate_level(value, f'{location}.{key}')


def _validate_block(block: dict[str, Any], location: str, *, allow_strict_scope: bool) -> None:
    """Validate one logging block: the top-level section or one program's.

    Parameters:
        block: The mapping to validate.
        location: Dotted path of the block, used in error messages.
        allow_strict_scope: Whether ``strict_scope`` is permitted here.  It is
            a global switch, so a per-program block rejects it.

    Raises:
        ValueError: If the block holds an unknown key or a malformed value.
    """
    categories = {
        'techniques': technique_log_keys(),
        'models': model_log_keys(),
        'other': OTHER_LOG_KEYS,
    }
    for key, value in block.items():
        if key == _PROGRAMS_KEY:
            continue  # handled by the caller
        if key == _STRICT_SCOPE_KEY:
            if not allow_strict_scope:
                raise ValueError(
                    f'{location}.{_STRICT_SCOPE_KEY} is a global setting and cannot be '
                    f'set for one program'
                )
            if not isinstance(value, bool):
                raise ValueError(
                    f'{location}.{_STRICT_SCOPE_KEY} must be true or false, '
                    f'got {type(value).__name__}'
                )
        elif key in LOGGER_KEYS:
            _validate_level(value, f'{location}.{key}')
        elif key in CATEGORY_KEYS:
            _validate_category(value, key, categories[key], f'{location}.{key}')
        else:
            allowed = sorted(LOGGER_KEYS | CATEGORY_KEYS)
            raise ValueError(
                f'{location}.{key} is not a known logging key; expected one of {allowed}'
            )


def validate_logging_config(config: 'Config') -> None:
    """Validate the ``logging`` section of ``config``.

    Checks every key in the top-level block and in each ``programs`` block
    against the technique and model registries, the fixed set of other
    image-scoped modules, and the registered program identities, and checks
    that every level is a recognized name.

    Parameters:
        config: The configuration to validate.

    Raises:
        ValueError: On an unknown logging key, category, module key, program
            name, or level name.  The message names the offending dotted path.
    """
    from spindoctor.cli.program_names import PROGRAM_NAMES

    section = config.logging
    if not section:
        return

    block = dict(section)
    _validate_block(block, 'logging', allow_strict_scope=True)

    programs = block.get(_PROGRAMS_KEY) or {}
    if not isinstance(programs, dict):
        raise ValueError(
            f'logging.{_PROGRAMS_KEY} must be a mapping, got {type(programs).__name__}'
        )
    for program_name, program_block in programs.items():
        location = f'logging.{_PROGRAMS_KEY}.{program_name}'
        if program_name not in PROGRAM_NAMES:
            raise ValueError(
                f'{location} is not a known program; expected one of {sorted(PROGRAM_NAMES)}'
            )
        if not isinstance(program_block, dict):
            raise ValueError(f'{location} must be a mapping, got {type(program_block).__name__}')
        _validate_block(dict(program_block), location, allow_strict_scope=False)

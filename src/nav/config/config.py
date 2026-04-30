"""YAML-backed configuration for RMS-NAV.

Loads and merges settings from bundled defaults and optional user YAML files
using :class:`ruamel.yaml.YAML` (safe typ), then exposes sections as
:class:`nav.support.attrdict.AttrDict` for attribute-style access. The
:class:`Config` class is the main public entry; helpers such as
:func:`_as_str_list` validate list-shaped YAML fragments used when building
config structures.
"""

from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from nav.support.attrdict import AttrDict


def _strip_underscore_keys(value: Any) -> Any:
    """Recursively drop mapping keys whose name starts with ``_``.

    Used by :meth:`Config._load_yaml` to remove documentation-only
    ``_sources`` blocks before merging.  Lists are walked element-wise;
    scalars pass through unchanged.

    Parameters:
        value: A YAML-parsed value (mapping, list, or scalar).

    Returns:
        ``value`` with every underscore-prefixed mapping key removed.
    """
    if isinstance(value, dict):
        return {
            k: _strip_underscore_keys(v)
            for k, v in value.items()
            if not (isinstance(k, str) and k.startswith('_'))
        }
    if isinstance(value, list):
        return [_strip_underscore_keys(v) for v in value]
    return value


def _as_str_list(value: Any, *, location: str) -> list[str]:
    """Coerce a YAML list value to ``list[str]``.

    Parameters:
        value: Parsed YAML fragment expected to be a list of strings.
        location: Human-readable path (e.g. config key path) for error messages.

    Returns:
        A new ``list[str]`` containing each element of ``value`` as ``str``.

    Raises:
        TypeError: If ``value`` is not a list, or if any element is not a ``str``.
    """

    if not isinstance(value, list):
        raise TypeError(f'{location}: expected a list of strings, got {type(value).__name__}')
    out: list[str] = []
    for i, item in enumerate(value):
        if not isinstance(item, str):
            raise TypeError(f'{location}[{i}]: expected str, got {type(item).__name__}')
        out.append(item)
    return out


class Config:
    """Manages configuration settings for the navigation system.

    This class handles loading, updating, and accessing configuration settings from YAML files.
    It provides access to various configuration sections through properties and methods.
    """

    def __init__(self) -> None:
        """Initializes a new Config instance with empty configuration containers."""

        self._config_dict: dict[str, Any] = {}
        self._config_environment: dict[str, Any] = AttrDict({})
        self._config_general: dict[str, Any] = AttrDict({})
        self._config_offset: dict[str, Any] = AttrDict({})
        self._config_bodies: dict[str, Any] = AttrDict({})
        self._config_body_shape: dict[str, Any] = AttrDict({})
        self._config_rings: dict[str, Any] = AttrDict({})
        self._config_stars: dict[str, Any] = AttrDict({})
        self._config_titan: dict[str, Any] = AttrDict({})
        self._config_bootstrap: dict[str, Any] = AttrDict({})
        self._config_backplanes: dict[str, Any] = AttrDict({})
        self._config_pds4: dict[str, Any] = AttrDict({})

    @property
    def is_loaded(self) -> bool:
        """Whether merged YAML is present (after ``read_config`` / ``update_config``)."""

        return bool(self._config_dict)

    def ensure_loaded(self) -> None:
        """Load bundled default YAML if not already loaded.

        Safe to call repeatedly; delegates to :meth:`read_config` with no path
        (same early-return behavior when data is already present).
        """

        self.read_config()

    def _update_attrdicts(self) -> None:
        """Updates all attribute dictionaries from the main configuration dictionary.

        Converts dictionary sections to AttrDict instances for convenient attribute-style access.
        """

        self._config_environment = AttrDict(self._config_dict.get('environment', {}))
        self._config_general = AttrDict(self._config_dict.get('general', {}))
        self._config_offset = AttrDict(self._config_dict.get('offset', {}))
        self._config_bodies = AttrDict(self._config_dict.get('bodies', {}))
        self._config_body_shape = AttrDict(self._config_dict.get('body_shape', {}))
        self._config_rings = AttrDict(self._config_dict.get('rings', {}))
        self._config_stars = AttrDict(self._config_dict.get('stars', {}))
        self._config_titan = AttrDict(self._config_dict.get('titan', {}))
        self._config_bootstrap = AttrDict(self._config_dict.get('bootstrap', {}))
        self._config_backplanes = AttrDict(self._config_dict.get('backplanes', {}))
        self._config_pds4 = AttrDict(self._config_dict.get('pds4', {}))

    def _load_yaml(self, config_path: str | Path) -> dict[str, Any]:
        """Loads a YAML file and returns a dictionary mapping.

        Documentation-only ``_sources`` blocks (and any other top-level or
        nested key that starts with an underscore) are stripped at load
        time so citations can live alongside values for human review
        without bloating the parsed config.
        """

        yaml = YAML(typ='safe')
        with open(config_path, encoding='utf-8') as f:
            loaded = yaml.load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f'Config "{config_path}" did not parse to a dictionary mapping')
        stripped = _strip_underscore_keys(loaded)
        # ``_strip_underscore_keys`` returns ``Any`` because it walks
        # arbitrary YAML; the top-level shape is guaranteed dict because
        # ``loaded`` was checked above.
        assert isinstance(stripped, dict)
        return stripped

    def read_config(self, config_path: str | Path | None = None, reread: bool = False) -> None:
        """Reads configuration from the specified YAML file.

        Parameters:
            config_path: Path to the configuration file. If None, uses the default
                config files.
            reread: Whether to reread the configuration file if it has already been read.

        Raises:
            ValueError: If any registered NavTechnique's confidence spec
                references an attribute the technique does not declare.
                Validation runs once per ``read_config`` invocation.
        """

        if not reread and self._config_dict:
            return

        if config_path is None:
            config_dir = Path(__file__).resolve().parent.parent / 'config_files'
            for filename in sorted(config_dir.glob('*.yaml')):
                self.update_config(filename, read_default=False)
            self._validate_registered_techniques()
            return

        self._config_dict = self._load_yaml(config_path)
        self._update_attrdicts()
        self._validate_registered_techniques()

    def _validate_registered_techniques(self) -> None:
        """Load + validate every registered NavTechnique's confidence spec.

        Builds each technique's :class:`ConfidenceSpec` from the
        ``techniques.<technique_name>`` block in
        ``config_510_techniques.yaml`` and assigns it to the class
        attribute, then asserts every term references an attribute the
        technique has declared in ``confidence_attributes``.  Test-only
        registry entries whose ``name`` starts with ``_`` are exempt
        from the YAML requirement (the test fixture supplies its own
        spec or uses ``None`` to opt out of confidence evaluation).

        Imported inside the method because ``nav.nav_technique.nav_technique``
        imports ``nav.support.nav_base`` which imports ``nav.config``;
        promoting these imports to the module top would fail with
        ``ImportError: cannot import name 'DEFAULT_CONFIG' from
        partially initialized module 'nav.config'``.  This lazy-import
        site is the minimum cycle break — the other modules can keep
        their top-of-file imports.
        """
        from nav.nav_technique.confidence_config import (
            ConfidenceConfigError,
            load_confidence_spec,
            load_technique_tuning,
        )
        from nav.nav_technique.nav_technique import (
            NavTechnique,
            validate_registered_confidence_specs,
        )

        techniques_block = dict(self._config_dict.get('techniques', {}))
        for cls in NavTechnique._registry:
            try:
                cls.confidence_spec = load_confidence_spec(techniques_block, cls.name)
            except ConfidenceConfigError:
                if cls.name.startswith('_'):
                    # Test-only technique: leave spec as inherited (None)
                    # so the test fixture controls it.
                    continue
                raise
            cls.tuning = load_technique_tuning(techniques_block, cls.name)
        validate_registered_confidence_specs()

    def update_config(self, config_path: str | Path, read_default: bool = True) -> None:
        """Updates the current configuration with values from the specified YAML file.

        Parameters:
            config_path: Path to the configuration file containing update values.
            read_default: Whether to read the default configuration file if no config
                has been previously read.
        """

        if read_default:
            self.read_config()
        new_config = self._load_yaml(config_path)
        for key in new_config:
            if key in self._config_dict:
                self._config_dict[key].update(new_config[key])
            else:
                self._config_dict[key] = new_config[key]
        self._update_attrdicts()

    def category(self, category: str) -> AttrDict:
        """Returns the configuration settings for the specified category."""

        self.read_config()
        return AttrDict(self._config_dict.get(category, {}))

    @property
    def general(self) -> Any:
        """Returns the general configuration settings."""

        self.read_config()
        return self._config_general

    @property
    def environment(self) -> Any:
        """Returns the environment configuration settings."""

        self.read_config()
        return self._config_environment

    @property
    def planets(self) -> list[str]:
        """Returns the list of configured planet names."""

        self.read_config()
        return _as_str_list(self._config_dict.get('planets', []), location='config.planets')

    def satellites(self, planet: str) -> list[str]:
        """Returns the list of satellites for the specified planet.

        Parameters:
            planet: The name of the planet to get satellites for.

        Returns:
            A list of satellite names for the specified planet.
        """

        self.read_config()
        block = self._config_dict.get('satellites', {})
        if not isinstance(block, dict):
            raise TypeError(
                f'config key "satellites" must be a mapping, got {type(block).__name__}'
            )
        return _as_str_list(
            block.get(planet.upper(), []),
            location=f'config.satellites[{planet.upper()!r}]',
        )

    def fuzzy_satellites(self, planet: str) -> list[str]:
        """Returns the list of fuzzy satellites for the specified planet.

        Parameters:
            planet: The name of the planet to get fuzzy satellites for.

        Returns:
            A list of fuzzy satellite names for the specified planet.
        """

        self.read_config()
        block = self._config_dict.get('fuzzy_satellites', {})
        if not isinstance(block, dict):
            raise TypeError(
                f'config key "fuzzy_satellites" must be a mapping, got {type(block).__name__}'
            )
        return _as_str_list(
            block.get(planet.upper(), []),
            location=f'config.fuzzy_satellites[{planet.upper()!r}]',
        )

    def ring_satellites(self, planet: str) -> list[str]:
        """Returns the list of ring satellites for the specified planet.

        Parameters:
            planet: The name of the planet to get ring satellites for.

        Returns:
            A list of ring satellite names for the specified planet.
        """

        self.read_config()
        block = self._config_dict.get('ring_satellites', {})
        if not isinstance(block, dict):
            raise TypeError(
                f'config key "ring_satellites" must be a mapping, got {type(block).__name__}'
            )
        return _as_str_list(
            block.get(planet.upper(), []),
            location=f'config.ring_satellites[{planet.upper()!r}]',
        )

    @property
    def offset(self) -> Any:
        """Returns the offset configuration settings."""

        self.read_config()
        return self._config_offset

    @property
    def bodies(self) -> Any:
        """Returns the celestial bodies configuration settings."""

        self.read_config()
        return self._config_bodies

    @property
    def body_shape(self) -> Any:
        """Returns the per-body shape catalogue (``config_220_body_shape.yaml``).

        Each entry is keyed by SPICE body name (e.g. ``MIMAS``) and exposes
        ``radii_km``, ``ellipsoid_rms_residual_km``, ``crater_scale_km``,
        ``albedo_mean``, ``albedo_variation``, and ``shape_class_hint``.
        Missing-body lookups return ``None`` from ``AttrDict.get``; downstream
        code applies the conservative fallback (10% radius default,
        reliability cap 0.3).
        """

        self.read_config()
        return self._config_body_shape

    @property
    def rings(self) -> Any:
        """Returns the planetary rings configuration settings."""

        self.read_config()
        return self._config_rings

    @property
    def stars(self) -> Any:
        """Returns the stars configuration settings."""

        self.read_config()
        return self._config_stars

    @property
    def titan(self) -> Any:
        """Returns the Titan-specific configuration settings."""

        self.read_config()
        return self._config_titan

    @property
    def bootstrap(self) -> Any:
        """Returns the bootstrap configuration settings."""

        self.read_config()
        return self._config_bootstrap

    @property
    def backplanes(self) -> Any:
        """Returns backplanes configuration, including bodies, rings, and target LIDs."""

        self.read_config()
        return self._config_backplanes

    @property
    def pds4(self) -> Any:
        """Returns PDS4 bundle generation configuration."""

        self.read_config()
        return self._config_pds4


DEFAULT_CONFIG = Config()

"""Provenance — reproducibility metadata attached to every NavResult.

Two navigations with identical inputs produce byte-identical
``Provenance`` *except* for ``pipeline_run_iso8601``, which is wall-clock
by construction; regression-baseline comparison strips that field before
comparing.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

__all__ = ['Provenance']


@dataclass(frozen=True)
class Provenance:
    """Reproducibility envelope written into every NavResult.

    Parameters:
        rms_nav_version: ``__version__`` string (e.g. ``'0.5.2'``).
        rms_nav_git_sha: Short git SHA, ``'dirty'``, or ``None`` if neither
            can be determined.
        spice_kernels: Sorted tuple of SPICE kernel filenames actually
            loaded (from ``spice.ktotal`` / ``spice.kdata``).
        static_data_hashes: Mapping ``filename -> sha256(raw bytes)`` for
            static-data YAMLs (``config_220_body_shape.yaml``, every
            ``config_3N0_*_rings.yaml``, every ``config_4N0_inst_*.yaml``).
            Comments and whitespace are included in the hashed bytes.
            Stored as a read-only ``MappingProxyType`` after construction.
        technique_names: Sorted tuple of registered technique class names.
        extractor_names: Sorted tuple of registered extractor class names.
        image_et: Observation midtime ET (TDB seconds past J2000).
        pipeline_run_iso8601: UTC timestamp when the run began.  Excluded
            from byte-identical regression-baseline comparison because it
            varies wall-clock-to-wall-clock for identical inputs.

    The non-init field ``spice_kernel_count`` is derived from
    ``len(spice_kernels)`` in ``__post_init__``.
    """

    rms_nav_version: str
    image_et: float
    pipeline_run_iso8601: str
    rms_nav_git_sha: str | None = None
    spice_kernels: tuple[str, ...] = ()
    static_data_hashes: Mapping[str, str] = field(default_factory=dict)
    technique_names: tuple[str, ...] = ()
    extractor_names: tuple[str, ...] = ()
    spice_kernel_count: int = field(init=False)

    def __post_init__(self) -> None:
        """Normalize sequences, derive count, freeze ``static_data_hashes``."""
        # Normalize the three sequence fields to deterministic sorted
        # tuples so callers' mutable or unsorted inputs cannot leak in.
        # ``object.__setattr__`` is required because the dataclass is frozen.
        object.__setattr__(self, 'spice_kernels', tuple(sorted(self.spice_kernels)))
        object.__setattr__(self, 'technique_names', tuple(sorted(self.technique_names)))
        object.__setattr__(self, 'extractor_names', tuple(sorted(self.extractor_names)))
        object.__setattr__(self, 'spice_kernel_count', len(self.spice_kernels))
        if not isinstance(self.static_data_hashes, MappingProxyType):
            object.__setattr__(
                self,
                'static_data_hashes',
                MappingProxyType(dict(self.static_data_hashes)),
            )

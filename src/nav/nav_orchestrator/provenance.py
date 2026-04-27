"""Provenance — reproducibility metadata attached to every NavResult.

Two navigations with identical inputs produce byte-identical
``Provenance`` *except* for ``pipeline_run_iso8601``, which is wall-clock
by construction; regression-baseline comparison strips that field before
comparing.
"""

from dataclasses import dataclass, field

__all__ = ['Provenance']


@dataclass(frozen=True)
class Provenance:
    """Reproducibility envelope written into every NavResult.

    Parameters:
        rms_nav_version: ``__version__`` string (e.g. ``'0.5.2'``).
        rms_nav_git_sha: Short git SHA, ``'dirty'``, or ``None`` if neither
            can be determined.
        spice_kernels: Sorted list of SPICE kernel filenames actually
            loaded (from ``spice.ktotal`` / ``spice.kdata``).
        spice_kernel_count: Convenience field equal to
            ``len(spice_kernels)``.
        static_data_hashes: Mapping ``filename -> sha256(raw bytes)`` for
            static-data YAMLs (``config_220_body_shape.yaml``, every
            ``config_3N0_*_rings.yaml``, every ``config_4N0_inst_*.yaml``).
            Comments and whitespace are included in the hashed bytes.
        technique_names: Sorted list of registered technique class names.
        extractor_names: Sorted list of registered extractor class names.
        image_et: Observation midtime ET (TDB seconds past J2000).
        pipeline_run_iso8601: UTC timestamp when the run began.  Excluded
            from byte-identical regression-baseline comparison because it
            varies wall-clock-to-wall-clock for identical inputs.
    """

    rms_nav_version: str
    image_et: float
    pipeline_run_iso8601: str
    rms_nav_git_sha: str | None = None
    spice_kernels: tuple[str, ...] = ()
    spice_kernel_count: int = 0
    static_data_hashes: dict[str, str] = field(default_factory=dict)
    technique_names: tuple[str, ...] = ()
    extractor_names: tuple[str, ...] = ()

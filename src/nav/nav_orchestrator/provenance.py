"""Provenance — reproducibility metadata attached to every NavResult.

Two navigations with identical inputs produce byte-identical
``Provenance`` *except* for ``pipeline_run_iso8601``, which is wall-clock
by construction; regression-baseline comparison strips that field before
comparing.

The :func:`collect_provenance_metadata` helper produces the per-image
``rms_nav_git_sha``, loaded-SPICE-kernel list, and the static-data hash
dictionary at navigate time so the orchestrator can populate the
``Provenance`` envelope without each caller re-implementing the lookups.
"""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

__all__ = [
    'Provenance',
    'ProvenanceMetadata',
    'collect_provenance_metadata',
]


_STATIC_DATA_PREFIXES: tuple[str, ...] = (
    'config_220_',  # body shape catalogue (Phase 3+)
    'config_3',  # ring catalogues (300_*_rings.yaml)
    'config_4',  # per-instrument blocks (400_inst_coiss.yaml ...)
)
"""Filename prefixes counted as static-data YAML for hashing.

Anything in ``src/nav/config_files`` whose name starts with one of these
prefixes is sha256-hashed and recorded in ``Provenance.static_data_hashes``.
"""


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


@dataclass(frozen=True)
class ProvenanceMetadata:
    """The per-image runtime-derived provenance fields.

    Parameters:
        git_sha: Short git SHA of the repository, ``'dirty'`` if there are
            uncommitted changes, or ``None`` if not available.
        spice_kernels: Sorted tuple of SPICE kernel filenames actually
            loaded.
        static_data_hashes: Mapping of static-data YAML filename to
            sha256-hex digest of the file's raw bytes.
    """

    git_sha: str | None
    spice_kernels: tuple[str, ...]
    static_data_hashes: Mapping[str, str]


def _resolve_git_sha() -> str | None:
    """Return the short git SHA at the head of the working tree or ``None``.

    Uses ``git rev-parse HEAD`` to read the SHA and ``git status
    --porcelain`` to detect uncommitted changes (returning ``'dirty'`` in
    that case).  Returns ``None`` when the tree is not inside a git
    repository or git is unavailable.
    """
    repo_root = Path(__file__).resolve().parents[3]
    try:
        sha = subprocess.run(
            ['git', '-C', str(repo_root), 'rev-parse', '--short', 'HEAD'],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    if not sha:
        return None
    try:
        status = subprocess.run(
            ['git', '-C', str(repo_root), 'status', '--porcelain'],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return sha
    return f'{sha}-dirty' if status.strip() else sha


def _resolve_spice_kernels() -> tuple[str, ...]:
    """Return the sorted tuple of currently-loaded SPICE kernel basenames.

    Uses ``cspyce`` (the SPICE binding shared with ``oops``) when it is
    available; returns an empty tuple when SPICE is not loaded.  The
    tuple holds *basenames* only so the hash and JSON output stay
    deterministic across machines with different kernel install roots.
    """
    try:
        import cspyce
    except ImportError:
        return ()
    try:
        ktotal = int(cspyce.ktotal('ALL'))
    except Exception:  # pragma: no cover - cspyce diagnostic edge case
        return ()
    kernels: list[str] = []
    for index in range(ktotal):
        try:
            file_name, _, _, _ = cspyce.kdata(index, 'ALL')
        except Exception:  # pragma: no cover - cspyce diagnostic edge case
            continue
        if file_name:
            kernels.append(Path(str(file_name)).name)
    return tuple(sorted(kernels))


def _resolve_static_data_hashes() -> Mapping[str, str]:
    """Return ``{filename: sha256_hex(raw bytes)}`` for shipped static data.

    Walks ``src/nav/config_files`` and hashes any file whose name starts
    with one of the recognised static-data prefixes
    (``config_220_``, ``config_3``, ``config_4``).  Returns the mapping
    sorted by filename so equality testing is stable.
    """
    config_dir = Path(__file__).resolve().parent.parent / 'config_files'
    hashes: dict[str, str] = {}
    for path in sorted(config_dir.glob('*.yaml')):
        name = path.name
        if not any(name.startswith(prefix) for prefix in _STATIC_DATA_PREFIXES):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[name] = digest
    return MappingProxyType(hashes)


def collect_provenance_metadata() -> ProvenanceMetadata:
    """Gather process-wide provenance metadata at navigate time.

    Returns:
        A :class:`ProvenanceMetadata` instance populated with the current
        git SHA, loaded SPICE kernel list, and static-data hashes.
    """
    return ProvenanceMetadata(
        git_sha=_resolve_git_sha(),
        spice_kernels=_resolve_spice_kernels(),
        static_data_hashes=_resolve_static_data_hashes(),
    )

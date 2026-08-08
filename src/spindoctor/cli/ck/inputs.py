"""What a generator run reads before it can write anything.

The kernel writer's other modules each answer a question about one image.  This
one gathers the run: the per-image metadata documents a navigation pass left
under its results root, the kernel directories those images name, and the
spacecraft clock kernel each of the run's clocks is encoded against.

Nothing here logs.  The driver reports what these functions return, so that the
part of the run that touches only files and SPICE stays testable without a
logger and stays inside the writer package, which may import neither oops nor
anything from ``spindoctor.support``.
"""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cspyce
from filecache import FCPath

from spindoctor.cli.ck.clocks import SCLK_SUFFIX, select_sclk_kernel
from spindoctor.cli.ck.frames import FK_SUFFIXES, require_one_frame_kernel_per_frame
from spindoctor.cli.ck.images import ImageEntry
from spindoctor.cli.ck.segment import resolve_sclk_id

METADATA_SUFFIX = '_metadata.json'
"""What a per-image navigation metadata document is named."""

LSK_SUFFIXES = frozenset({'.tls'})
"""The extension a leapseconds kernel carries in the holdings."""


@dataclass(frozen=True)
class Document:
    """One image's navigation metadata, as the run read it.

    Parameters:
        path: The metadata file.
        stub: The image's results path stub, which names its log.
        metadata: The document itself.
    """

    path: FCPath
    stub: str
    metadata: dict[str, Any]


################################################################################
#
# READING WHAT THE NAVIGATION RUN LEFT
#
################################################################################


def read_documents(root: FCPath, mission: str) -> tuple[list[Document], list[tuple[FCPath, str]]]:
    """Read every metadata document of one mission under a results root.

    A file that cannot be read as JSON, or that holds JSON that is not a
    document, is returned for the caller to report rather than raised on: it
    names no image, so there is nothing for the report to say about it and
    nothing an omission reason could be recorded against.  A document of
    another mission is simply not this run's business and is passed over
    silently -- but only a document that *names* a mission can be another
    mission's.  One with no readable instrument at all is unreadable, not
    foreign: skipping it silently would let a truncated or corrupted document
    vanish from every mission's run without a trace.

    Parameters:
        root: The navigation results root.
        mission: The instrument identity to keep.

    Returns:
        The mission's documents, ordered by path, and one entry per file that
        could not be read at all, pairing it with why.
    """
    documents: list[Document] = []
    unreadable: list[tuple[FCPath, str]] = []
    for path in sorted(root.rglob(f'*{METADATA_SUFFIX}'), key=lambda entry: entry.as_posix()):
        stub = _stub_for(root, path)
        try:
            metadata = _read_document(path)
        except (OSError, ValueError) as exc:
            unreadable.append((path, str(exc)))
            continue
        observation = metadata.get('observation')
        instrument = observation.get('instrument') if isinstance(observation, dict) else None
        if not isinstance(instrument, str):
            unreadable.append((path, 'names no instrument to attribute it to a mission'))
            continue
        if instrument != mission:
            continue
        documents.append(Document(path=path, stub=stub, metadata=metadata))
    return documents, unreadable


def _read_document(path: FCPath) -> dict[str, Any]:
    """Read one metadata document.

    Parameters:
        path: The file to read.

    Returns:
        The document.

    Raises:
        ValueError: if the file does not hold a JSON object.
        OSError: if it cannot be read.
    """
    document = json.loads(path.read_text())
    if not isinstance(document, dict):
        raise ValueError(f'holds a {type(document).__name__}, not a JSON object')
    return cast(dict[str, Any], document)


def _stub_for(root: FCPath, path: FCPath) -> str:
    """Return the results path stub naming one image's log.

    Parameters:
        root: The navigation results root.
        path: The image's metadata file.

    Returns:
        The file's path relative to the root, without the metadata suffix.
        The full path is used when it does not lie under the root, which
        cannot happen for a document the root's own listing produced.
    """
    relative = path.as_posix().removeprefix(root.as_posix()).lstrip('/')
    return relative.removesuffix(METADATA_SUFFIX)


def select_by_time(
    documents: Sequence[Document], start_et: float | None, stop_et: float | None
) -> tuple[list[Document], int]:
    """Keep the documents whose exposure midtime lies within a time range.

    Parameters:
        documents: The documents to filter.
        start_et: Earliest midtime to keep, or ``None`` for no lower bound.
        stop_et: Latest midtime to keep, or ``None`` for no upper bound.

    Returns:
        The selected documents and how many were dropped for recording no
        midtime.  An image with no midtime is kept when no bound is given and
        dropped when either is, since it cannot be shown to satisfy one.

    Raises:
        ValueError: if both bounds are given with the start after the stop.  A
            swapped pair would select nothing, and a run that writes nothing
            for that reason would be indistinguishable from a clean run over a
            quiet span.
    """
    if start_et is not None and stop_et is not None and start_et > stop_et:
        raise ValueError(
            f'the time range is inverted: its start {start_et!r} is after its stop {stop_et!r}'
        )
    if start_et is None and stop_et is None:
        return list(documents), 0
    selected: list[Document] = []
    undated = 0
    for document in documents:
        midtime = _midtime_of(document)
        if midtime is None:
            undated += 1
            continue
        if start_et is not None and midtime < start_et:
            continue
        if stop_et is not None and midtime > stop_et:
            continue
        selected.append(document)
    return selected, undated


def _midtime_of(document: Document) -> float | None:
    """Return one document's recorded exposure midtime.

    Parameters:
        document: The document.

    Returns:
        The midtime in TDB seconds past J2000, or ``None`` when the document
        records none or records something that is not a finite number.  A
        non-finite value is read as none rather than passed on, because every
        comparison against a NaN is False: a NaN midtime would fall inside
        every time range at once, and an infinite one would fall inside a
        half-bounded range it can have no business in.
    """
    result = document.metadata.get('navigation_result')
    if not isinstance(result, dict):
        return None
    times = result.get('times')
    if not isinstance(times, dict):
        return None
    midtime = times.get('midtime_et')
    if isinstance(midtime, bool) or not isinstance(midtime, int | float):
        return None
    if not math.isfinite(midtime):
        return None
    return float(midtime)


################################################################################
#
# THE KERNEL POOL
#
################################################################################


def kernel_paths(directories: Sequence[str]) -> dict[str, tuple[FCPath, ...]]:
    """Index every file in the run's kernel directories by basename.

    Parameters:
        directories: The directories to list, without recursion.

    Returns:
        One entry per basename, holding every path of that name.  A basename
        found in two directories keeps both, so that a caller needing exactly
        one file can say which two it could not choose between.

    Raises:
        ValueError: if a directory does not exist or is not a directory.
    """
    found: dict[str, list[FCPath]] = {}
    for directory in directories:
        root = FCPath(directory)
        try:
            entries = sorted(root.iterdir(), key=lambda entry: entry.as_posix())
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise ValueError(
                f'kernel directory {root.as_posix()!r} does not exist or is not a directory'
            ) from exc
        for entry in entries:
            if entry.is_file():
                found.setdefault(entry.name, []).append(entry)
    return {name: tuple(paths) for name, paths in found.items()}


def resolve_one(basename: str, paths: Mapping[str, tuple[FCPath, ...]]) -> FCPath:
    """Return the one file of a given basename among the run's directories.

    Parameters:
        basename: The name to resolve.
        paths: The indexed kernel directories.

    Returns:
        The file's path.

    Raises:
        ValueError: if no directory holds that name, or if more than one does,
            since the two are different files and nothing in the record says
            which was furnished.
    """
    if basename not in paths:
        raise ValueError(
            f'{basename!r} is named by an image and is not in any of the kernel directories'
        )
    candidates = paths[basename]
    if len(candidates) > 1:
        raise ValueError(
            f'{basename!r} is in more than one kernel directory: '
            f'{[path.as_posix() for path in candidates]}; which one was furnished is not in the '
            f'record'
        )
    return candidates[0]


def recorded_basenames(documents: Sequence[Document]) -> tuple[str, ...]:
    """Return every SPICE kernel basename the documents record, sorted.

    Parameters:
        documents: The documents to read.

    Returns:
        The basenames, without repeats.
    """
    names: set[str] = set()
    for document in documents:
        result = document.metadata.get('navigation_result')
        if not isinstance(result, dict):
            continue
        provenance = result.get('provenance')
        if not isinstance(provenance, dict):
            continue
        recorded = provenance.get('spice_kernels')
        if not isinstance(recorded, list):
            continue
        names.update(name for name in recorded if isinstance(name, str))
    return tuple(sorted(names))


def furnish_supporting_kernels(
    basenames: Sequence[str], paths: Mapping[str, tuple[FCPath, ...]], suffixes: frozenset[str]
) -> tuple[str, ...]:
    """Furnish every recorded kernel of the given kinds that the run can resolve.

    A basename the run's directories do not hold is skipped rather than
    refused: provenance accumulates every kernel a batch ever furnished, so it
    names kernels this mission's images never needed.  A kernel that is
    genuinely missing surfaces where it is used -- a frame the images name and
    the pool does not define is refused by the assignment step, by name.

    Parameters:
        basenames: The recorded basenames.
        paths: The indexed kernel directories.
        suffixes: The extensions to furnish.

    Returns:
        The basenames furnished, in the order they were furnished.

    Raises:
        ValueError: if a basename resolves to more than one file.
        OSError: if a kernel cannot be furnished.
    """
    furnished: list[str] = []
    for basename in basenames:
        if Path(basename).suffix.lower() not in suffixes:
            continue
        if basename not in paths:
            continue
        local = str(cast(Path, resolve_one(basename, paths).retrieve()))
        cspyce.furnsh(local)
        furnished.append(basename)
    return tuple(furnished)


def furnish_frame_kernels(
    entries: Sequence[ImageEntry],
    basenames: Sequence[str],
    paths: Mapping[str, tuple[FCPath, ...]],
) -> tuple[str, ...]:
    """Furnish the run's frame kernels, refusing two that define one frame.

    A mission furnishes several frame kernels at once by design, so they are
    not resolved to one the way a clock kernel is.  What is refused is two of
    them defining a frame this run's images actually name -- the camera frame
    the reproduction test asks through, or the frame naming a corrected object
    -- since the pool would answer with whichever was furnished last and every
    image navigated through the other would be reported as having no baseline.

    Parameters:
        entries: The images the run considered.  Those carrying no pointing
            name no frames.
        basenames: Every kernel basename the run's documents record.
        paths: The indexed kernel directories.

    Returns:
        The basenames furnished, in the order they were furnished.

    Raises:
        ValueError: if a basename resolves to more than one file, if a frame
            the run needs is already defined by a furnished kernel, or if two
            candidates define the same one.
        OSError: if a kernel cannot be furnished.
    """
    candidates = {
        basename: resolve_one(basename, paths)
        for basename in basenames
        if Path(basename).suffix.lower() in FK_SUFFIXES and basename in paths
    }
    pointings = [entry.pointing for entry in entries if entry.pointing is not None]
    require_one_frame_kernel_per_frame(
        candidates,
        camera_frames={pointing.camera_frame for pointing in pointings},
        ck_frame_ids={pointing.ck_frame_id for pointing in pointings},
    )
    return furnish_supporting_kernels(basenames, paths, FK_SUFFIXES)


def clock_kernels(
    entries: Sequence[ImageEntry],
    basenames: Sequence[str],
    paths: Mapping[str, tuple[FCPath, ...]],
) -> dict[int, str]:
    """Choose the clock kernel each spacecraft clock in the run is encoded with.

    A run furnishes one pool for all of its images, so the candidates offered
    are the union of what the run's images record -- and the requirement that
    exactly one of them define a clock is then also the requirement that the
    run's images agree about that clock.  Two images of one spacecraft whose
    records name different versions of its clock kernel leave two candidates
    defining it, and the run refuses rather than encoding one image's time tags
    against the other's kernel, which nothing downstream would report.

    Parameters:
        entries: The images the run considered.  Those carrying no pointing
            encode no time tags and name no clock.
        basenames: Every kernel basename the run's documents record.
        paths: The indexed kernel directories.

    Returns:
        One entry per spacecraft clock the run's eligible images need, holding
        the basename of the kernel that defines it.

    Raises:
        ValueError: if the run's records name no clock kernel for a clock its
            images need, or name more than one.
        OSError: if a candidate kernel cannot be furnished for the probe.
    """
    candidates = {
        basename: resolve_one(basename, paths)
        for basename in basenames
        if Path(basename).suffix.lower() == SCLK_SUFFIX and basename in paths
    }
    needed = sorted(
        {
            resolve_sclk_id(entry.pointing.ck_frame_id)
            for entry in entries
            if entry.pointing is not None
        }
    )
    return {sclk_id: select_sclk_kernel(candidates, sclk_id) for sclk_id in needed}

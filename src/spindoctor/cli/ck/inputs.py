"""What a generator run assembles before it can write anything.

The kernel writer's other modules each answer a question about one image.  This
one gathers the run: which of the navigation records a run was handed belong to
the span it was asked for, the kernel directories those records name, and the
spacecraft clock kernel each of the run's clocks is encoded against.

Where the records themselves come from is not here.  They arrive as the stream
:mod:`spindoctor.nav_records` defines and either storage answers, so that this
program reads a results tree and an ingested index through the same code every
other consumer does.  What is here is the one thing this program does with that
stream that no other consumer does: it holds the whole mission, because a kernel
set cannot be assigned until every image that might claim a baseline has been
seen.

Nothing here logs.  The driver reports what these functions return, so that the
part of the run that touches only files and SPICE stays testable without a
logger.
"""

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import cast

import cspyce
from filecache import FCPath

from spindoctor.cli.ck.clocks import SCLK_SUFFIX, select_sclk_kernel
from spindoctor.cli.ck.frames import FK_SUFFIXES, require_one_frame_kernel_per_frame
from spindoctor.cli.ck.images import ImageEntry
from spindoctor.cli.ck.segment import resolve_sclk_id
from spindoctor.nav_records import NavRecord, UnreadableFile
from spindoctor.results_index import IMAGES
from spindoctor.support.nav_record import record_midtime_et

LSK_SUFFIXES = frozenset({'.tls'})
"""The extension a leapseconds kernel carries in the holdings."""

RECORD_COLUMNS = (
    IMAGES.c.image_name,
    IMAGES.c.instrument,
    IMAGES.c.camera,
    IMAGES.c.shutter_mode,
    IMAGES.c.status,
    IMAGES.c.status_reason,
    IMAGES.c.offset_dv,
    IMAGES.c.offset_du,
    IMAGES.c.sigma_dv,
    IMAGES.c.sigma_du,
    IMAGES.c.rotation_deg,
    IMAGES.c.confidence,
    IMAGES.c.confidence_rank,
    IMAGES.c.spice_kernels,
    IMAGES.c.start_et,
    IMAGES.c.stop_et,
    IMAGES.c.midtime_et,
    IMAGES.c.exposure_s,
    IMAGES.c.sclk_midtime,
    IMAGES.c.camera_frame,
    IMAGES.c.camera_frame_id,
    IMAGES.c.ck_frame_id,
    IMAGES.c.cmatrix,
    IMAGES.c.cmatrix_original,
)
"""Every column this program reads, and the whole of what a bulk read selects.

A row is only cheaper than a document while it carries less, so this is a
declaration rather than a convenience: what is not here is not read, and what is
read is here.  A test holds the list to the fields
:mod:`spindoctor.results_index.rebuild` knows a place for, since a column
selected that no field is rebuilt from would be paid for and dropped.

``camera_frame`` is here and is not among the columns the reprojection and
backplane stages select, which is the whole of the difference between what the
two consumers read: a kernel writer looks the frame up among the frame kernels it
furnishes, and a reader gating an attitude against an observation takes the frame
identity from the observation instead.
"""


def read_whole_mission(
    stream: Iterable[NavRecord | UnreadableFile],
) -> tuple[list[NavRecord], list[UnreadableFile]]:
    """Collect one mission's stream, separating the files that held no record.

    The stream is held whole rather than consumed as it arrives.  This program
    assigns every corrected attitude to one of the original kernels the run
    indexes, and cannot know which originals it needs until it has seen every
    image, so a mission's records are in memory whatever this function does with
    them.

    Both lists are ordered by the path of the file each entry stands for, which
    is an order neither storage promises: a walk yields in the order its
    directory listings return, and a server sorts text under a collation of its
    own.  Imposing it here is what makes a run's kernels, its report and its log
    identical whichever storage answered.

    Parameters:
        stream: What the record source yielded, records and unreadable files
            mixed in the order it found them.

    Returns:
        The records, and the files no record could be read out of, each in the
        order of the document it stands for.
    """
    records: list[NavRecord] = []
    unreadable: list[UnreadableFile] = []
    for found in stream:
        if isinstance(found, UnreadableFile):
            unreadable.append(found)
        else:
            records.append(found)
    records.sort(key=lambda record: record.path.as_posix())
    unreadable.sort(key=lambda entry: entry.path.as_posix())
    return records, unreadable


def select_by_time(
    records: Sequence[NavRecord], start_et: float | None, stop_et: float | None
) -> tuple[list[NavRecord], int]:
    """Keep the records whose exposure midtime lies within a time range.

    Parameters:
        records: The records to filter.
        start_et: Earliest midtime to keep, or ``None`` for no lower bound.
        stop_et: Latest midtime to keep, or ``None`` for no upper bound.

    Returns:
        The selected records and how many were dropped for recording no
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
        return list(records), 0
    selected: list[NavRecord] = []
    undated = 0
    for record in records:
        midtime = record_midtime_et(record.metadata)
        if midtime is None:
            undated += 1
            continue
        if start_et is not None and midtime < start_et:
            continue
        if stop_et is not None and midtime > stop_et:
            continue
        selected.append(record)
    return selected, undated


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


def recorded_basenames(records: Sequence[NavRecord]) -> tuple[str, ...]:
    """Return every SPICE kernel basename the records name, sorted.

    Parameters:
        records: The records to read.

    Returns:
        The basenames, without repeats.
    """
    names: set[str] = set()
    for record in records:
        result = record.metadata.get('navigation_result')
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
        basenames: Every kernel basename the run's records name.
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
        basenames: Every kernel basename the run's records name.
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

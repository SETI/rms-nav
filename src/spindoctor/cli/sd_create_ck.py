#!/usr/bin/env python3
################################################################################
# sd_create_ck.py
#
# Turn the corrected pointing a navigation run recorded into SPICE C-kernels.
# One mission per invocation: every per-image metadata document under the
# navigation results root is read, each eligible image is paired with the
# original kernel it navigated against, and one corrected kernel is written per
# original, mirroring its name with "_nav" before the extension.  Beside them go
# a meta-kernel that furnishes the set in the order that makes a correction take
# precedence, and a CSV report saying what became of every image considered.
################################################################################

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cspyce
import pdslogger
from filecache import FCPath, FileCache

# Make CLI runnable from source tree with
#    python src/package
package_source_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, package_source_path)

from spindoctor import __version__
from spindoctor.cli.ck.assignment import Assignment, assign_images, group_for_output
from spindoctor.cli.ck.clocks import SCLK_SUFFIX, select_sclk_kernel
from spindoctor.cli.ck.comments import CommentArea, build_comment_lines
from spindoctor.cli.ck.images import ImageEntry, OmissionReason
from spindoctor.cli.ck.index import build_ck_index
from spindoctor.cli.ck.kernel_file import write_ck_file
from spindoctor.cli.ck.metakernel import write_meta_kernel
from spindoctor.cli.ck.report import ImageFacts, ReportRow, read_image_facts, write_report
from spindoctor.cli.ck.segment import CkSegment, build_segment, resolve_sclk_id
from spindoctor.cli.logging_args import add_logging_arguments, reporting_logging_errors
from spindoctor.config import (
    DEFAULT_CONFIG,
    IMAGE_LOGGER,
    MAIN_LOGGER,
    RunLogging,
    build_image_log_handlers,
    build_run_logging,
    get_nav_results_root,
    load_default_and_user_config,
)
from spindoctor.config.program_names import SD_CREATE_CK
from spindoctor.support.misc import log_run_environment

PROGRAM_NAME = SD_CREATE_CK
"""Program identity: names the main log directory and the
``logging.programs`` configuration block for this program."""

BACKEND = 'ck'
"""The per-image log subtree this program writes under."""

MISSIONS = ('coiss', 'gossi', 'nhlorri', 'vgiss')
"""The instrument identities whose images have a C-kernel object to correct.

Spelled here rather than read from the observation registry, which is the
authority on the names: reading it would import oops into a program whose whole
point is to write kernels without the geometry stack.  A mission whose images
carry no corrected attitude at all -- simulated images -- is deliberately not
offered.
"""

METADATA_SUFFIX = '_metadata.json'
"""What a per-image navigation metadata document is named."""

REPORT_SUFFIX = '_ck_report.csv'
META_KERNEL_SUFFIX = '_nav.tm'

# The kernel kinds the writer needs furnished, by the extension the holdings
# store them under.  The spacecraft clock is not among them: which clock kernel
# a run may furnish is decided per image, from the provenance, and is the one
# thing here that a version mismatch would corrupt silently.
_LSK_SUFFIXES = frozenset({'.tls'})
_FK_SUFFIXES = frozenset({'.tf', '.tk'})


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
# ARGUMENT PARSING
#
################################################################################


def parse_args(command_list: list[str]) -> argparse.Namespace:
    """Build the parser and parse one command line.

    Parameters:
        command_list: The arguments, the mission first.

    Returns:
        The parsed arguments.
    """
    cmdparser = argparse.ArgumentParser(
        description='Write corrected-pointing C-kernels from navigated images.',
        epilog="""Each corrected kernel mirrors one original and carries a segment per
                navigated exposure; the originals stay required, since the corrections
                cover only the exposures that were navigated.""",
    )

    cmdparser.add_argument(
        'mission',
        type=str.lower,
        choices=MISSIONS,
        help='The mission whose navigated images to write kernels for',
    )

    environment_group = cmdparser.add_argument_group('Environment')
    environment_group.add_argument(
        '--config-file',
        action='append',
        default=None,
        help="""The configuration file(s) to use to override default settings;
        may be specified multiple times. If not provided, attempts to load
        ./nav_default_config.yaml if present.""",
    )
    environment_group.add_argument(
        '--nav-results-root',
        type=str,
        default=None,
        help="""The root directory of the navigation results to read metadata from;
        overrides the NAV_RESULTS_ROOT environment variable and the nav_results_root
        configuration variable""",
    )
    environment_group.add_argument(
        '--kernel-dir',
        action='append',
        required=True,
        metavar='DIR',
        help="""A directory of SPICE kernels; may be specified multiple times, and at
        least one is required. Every directory is scanned for C-kernels to pair images
        against, and all of them together resolve the kernel basenames each image's
        provenance records, so the leapseconds, frame and spacecraft clock kernels
        navigation used must be among them. Directories are not searched recursively.""",
    )

    selection_group = cmdparser.add_argument_group('Image selection')
    selection_group.add_argument(
        '--start-time',
        type=str,
        default=None,
        metavar='UTC',
        help="""Ignore images whose exposure midtime is before this UTC time. An image
        that recorded no midtime is ignored whenever either bound is given, since it
        cannot be placed in time.""",
    )
    selection_group.add_argument(
        '--stop-time',
        type=str,
        default=None,
        metavar='UTC',
        help='Ignore images whose exposure midtime is after this UTC time.',
    )

    output_group = cmdparser.add_argument_group('Output')
    output_group.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help="""Directory the corrected kernels, the meta-kernel and the report are
        written to. It is created if it does not exist.""",
    )

    add_logging_arguments(cmdparser)

    return cmdparser.parse_args(command_list)


################################################################################
#
# READING WHAT THE NAVIGATION RUN LEFT
#
################################################################################


def read_documents(root: FCPath, mission: str) -> tuple[list[Document], int]:
    """Read every metadata document of one mission under a results root.

    A document that cannot be read as JSON, or that does not say which
    instrument it is from, is reported and skipped: it names no image, so there
    is nothing for the report to say about it.

    Parameters:
        root: The navigation results root.
        mission: The instrument identity to keep.

    Returns:
        The mission's documents, ordered by path, and how many documents could
        not be read at all.
    """
    documents: list[Document] = []
    unreadable = 0
    for path in sorted(root.rglob(f'*{METADATA_SUFFIX}'), key=lambda entry: entry.as_posix()):
        stub = _stub_for(root, path)
        try:
            metadata = _read_document(path)
        except (OSError, ValueError) as exc:
            MAIN_LOGGER.error('Could not read %s: %s', path.as_posix(), exc)
            unreadable += 1
            continue
        observation = metadata.get('observation')
        if not isinstance(observation, dict) or observation.get('instrument') != mission:
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
    """
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


@contextmanager
def furnished(path: FCPath) -> Iterator[None]:
    """Furnish one kernel for the duration of a block and unload it after.

    Parameters:
        path: The kernel to furnish, local or remote.

    Yields:
        Nothing; the kernel is furnished for the body of the block.
    """
    local = str(cast(Path, path.retrieve()))
    cspyce.furnsh(local)
    try:
        yield
    finally:
        cspyce.unload(local)


################################################################################
#
# WRITING
#
################################################################################


def report_rows(
    documents: Sequence[Document], assignments: Sequence[Assignment]
) -> tuple[list[ReportRow], dict[str, ImageFacts]]:
    """Build the report, one row per image the run considered.

    Parameters:
        documents: The documents the run considered, in report order.
        assignments: What became of each of them.

    Returns:
        The rows, and the facts of each image keyed by name so the comment
        areas can carry the same numbers the report does.

    Raises:
        ValueError: if a document holds a value the report cannot render, or
            if the two sequences are of different lengths -- the assignment
            step answers one assignment per image in the order it was given
            them, so a length mismatch means they are not the same images.
    """
    rows: list[ReportRow] = []
    facts_by_name: dict[str, ImageFacts] = {}
    for document, assignment in zip(documents, assignments, strict=True):
        facts = read_image_facts(document.metadata)
        facts_by_name[facts.image_name] = facts
        rows.append(
            ReportRow(
                facts=facts,
                source_bc=assignment.output_name,
                omission_reason=assignment.omission_reason,
            )
        )
    return rows, facts_by_name


def write_output_files(
    assignments: Sequence[Assignment],
    facts_by_name: Mapping[str, ImageFacts],
    output_dir: FCPath,
    *,
    sclk_basenames: Mapping[int, str],
    configuration_hash: str,
) -> list[tuple[FCPath, FCPath]]:
    """Write one corrected kernel per original the run's images navigated against.

    Parameters:
        assignments: What became of each image the run considered.
        facts_by_name: The reported facts of each image, by name.
        output_dir: Where the corrected kernels go.
        sclk_basenames: The clock kernel each spacecraft clock is encoded with.
        configuration_hash: Digest of the configuration this run used.

    Returns:
        One entry per file written, pairing the original with the correction.

    Raises:
        OSError: if a baseline kernel supplies no pointing at a record epoch,
            which is an image whose exposure its own baseline does not cover.
        ValueError: if a segment cannot be built or a file cannot be written.
    """
    written: list[tuple[FCPath, FCPath]] = []
    for group in group_for_output(assignments):
        output = output_dir / group.name
        with furnished(group.baseline.path):
            segments = [_segment_for(assignment) for assignment in group.assignments]
        images = tuple(facts_by_name[assignment.image_name] for assignment in group.assignments)
        area = CommentArea(
            generator_version=__version__,
            configuration_hash=configuration_hash,
            baseline_basenames=(group.baseline.basename,),
            sclk_basename=sclk_basenames[_clock_of(group.assignments[0])],
            images=images,
        )
        write_ck_file(local_output_path(output), segments, build_comment_lines(area))
        MAIN_LOGGER.info(
            'Wrote %s: %d segment(s) correcting %s',
            output.as_posix(),
            len(segments),
            group.baseline.basename,
        )
        written.append((group.baseline.path, output))
    return written


def local_output_path(path: FCPath) -> Path:
    """Return the local path SPICE writes a kernel to.

    Parameters:
        path: The output path.

    Returns:
        The same path as a local one.

    Raises:
        ValueError: if it is not local, since SPICE creates a file by name on
            the local filesystem and cannot write to a remote root.
    """
    local = Path(path.as_posix())
    if '://' in path.as_posix():
        raise ValueError(
            f'{path.as_posix()!r} is not a local directory; SPICE creates a kernel by name on the '
            f'local filesystem, so a corrected kernel cannot be written to a remote root'
        )
    local.parent.mkdir(parents=True, exist_ok=True)
    return local


def _segment_for(assignment: Assignment) -> CkSegment:
    """Build the corrected segment of one assigned image.

    Parameters:
        assignment: The image and the baseline it corrects.

    Returns:
        The segment.

    Raises:
        ValueError: if the image carries no pointing, which an assignment with
            a baseline cannot.
        OSError: if the baseline supplies no pointing at a record epoch.
    """
    pointing = assignment.entry.pointing
    if pointing is None:
        raise ValueError(f'{assignment.image_name} has a baseline and no pointing to write')
    try:
        return build_segment(pointing)
    except (OSError, KeyError, ValueError) as exc:
        # Reported to both logs before it stops the run: the reason set the
        # report may use has no entry for an image whose baseline reproduced
        # its attitude and then could not supply a record epoch, and inventing
        # one would be a schema change for every consumer of the report.
        IMAGE_LOGGER.error('Could not build the corrected segment: %s', exc)
        MAIN_LOGGER.error(
            '%s: could not build the corrected segment: %s', assignment.image_name, exc
        )
        raise


def _clock_of(assignment: Assignment) -> int:
    """Return the spacecraft clock an assigned image's time tags are encoded with.

    Parameters:
        assignment: The image.

    Returns:
        The clock id.

    Raises:
        ValueError: if the image carries no pointing.
    """
    pointing = assignment.entry.pointing
    if pointing is None:
        raise ValueError(f'{assignment.image_name} has a baseline and no pointing to write')
    return resolve_sclk_id(pointing.ck_frame_id)


################################################################################
#
# REPORTING WHAT HAPPENED
#
################################################################################


def log_dispositions(
    documents: Sequence[Document], assignments: Sequence[Assignment], run_logging: RunLogging
) -> Counter[str]:
    """Report what became of each image, to that image's log and to the run's.

    An operator watching a batch must not have to open a per-image log to learn
    that corrections stopped being written, so every omission is one line in
    the run log as well as the detail in the image's own, and the counts are
    reported once at the end.

    Parameters:
        documents: The documents the run considered.
        assignments: What became of each of them.
        run_logging: The run's logging configuration.

    Returns:
        How many images each omission reason accounted for, with the images
        that received a segment counted under ``'corrected'``.

    Raises:
        ValueError: if the two sequences are of different lengths, which means
            they are not the same images.
    """
    totals: Counter[str] = Counter()
    for document, assignment in zip(documents, assignments, strict=True):
        totals[_disposition_of(assignment)] += 1
        _log_one_disposition(document, assignment, run_logging)
    return totals


def _disposition_of(assignment: Assignment) -> str:
    """Return the one word summarizing what became of one image.

    Parameters:
        assignment: The image's assignment.

    Returns:
        The omission reason's value, or ``'corrected'``.
    """
    if assignment.omission_reason is None:
        return 'corrected'
    return assignment.omission_reason.value


def _log_one_disposition(
    document: Document, assignment: Assignment, run_logging: RunLogging
) -> None:
    """Write one image's disposition to its own log, and any omission to the run's.

    Parameters:
        document: The image's metadata document.
        assignment: What became of it.
        run_logging: The run's logging configuration.
    """
    handlers, log_path = build_image_log_handlers(
        BACKEND,
        document.stub,
        run_logging.sinks,
        run_logging.levels,
        timestamp=run_logging.timestamp,
    )
    try:
        with IMAGE_LOGGER.open(
            document.path.as_posix(),
            handler=handlers,
            level=run_logging.levels.image_section_level(),
        ):
            if assignment.omission_reason is None:
                IMAGE_LOGGER.info(
                    'Corrected in %s, from the baseline %s',
                    assignment.output_name,
                    assignment.baseline.basename if assignment.baseline is not None else '',
                )
            else:
                IMAGE_LOGGER.warning(
                    'No corrected segment written: %s', assignment.omission_reason.value
                )
    finally:
        for handler in handlers:
            if handler is not pdslogger.NULL_HANDLER:
                handler.close()
    if assignment.omission_reason is not None:
        MAIN_LOGGER.warning(
            '%s: no corrected segment written (%s)%s',
            assignment.image_name,
            assignment.omission_reason.value,
            '' if log_path is None else f'; see {log_path}',
        )


def log_totals(totals: Counter[str]) -> None:
    """Report how many images each disposition accounted for.

    Parameters:
        totals: The counts, keyed by disposition.
    """
    MAIN_LOGGER.info('Images corrected %d', totals.get('corrected', 0))
    for reason in OmissionReason:
        MAIN_LOGGER.info('Images omitted, %s: %d', reason.value, totals.get(reason.value, 0))


################################################################################
#
# MAIN
#
################################################################################


def main() -> None:
    """Write one mission's corrected C-kernels, meta-kernel and report."""
    command_list = sys.argv[1:]
    arguments = parse_args(command_list)

    with reporting_logging_errors():
        load_default_and_user_config(arguments, DEFAULT_CONFIG)

    nav_results_root = FileCache(None).new_path(get_nav_results_root(arguments, DEFAULT_CONFIG))
    output_dir = FileCache(None).new_path(arguments.output_dir)

    with reporting_logging_errors():
        run_logging = build_run_logging(PROGRAM_NAME, arguments, DEFAULT_CONFIG)

    start_time = time.time()
    MAIN_LOGGER.info('*************************************')
    MAIN_LOGGER.info('*** BEGINNING C-KERNEL GENERATION ***')
    MAIN_LOGGER.info('*************************************')
    MAIN_LOGGER.info('')
    log_run_environment(MAIN_LOGGER, command_list)

    documents, unreadable = read_documents(nav_results_root, arguments.mission)
    MAIN_LOGGER.info(
        'Read %d %s metadata document(s) under %s',
        len(documents),
        arguments.mission,
        nav_results_root.as_posix(),
    )
    if unreadable > 0:
        MAIN_LOGGER.error('Could not read %d metadata document(s)', unreadable)

    paths = kernel_paths(arguments.kernel_dir)
    # The leapseconds kernel is furnished before the time filter, because the
    # filter's own bounds are UTC and there is nothing to convert them with
    # until it is.  Its candidates are therefore the whole mission's, not the
    # selection's; that is safe where a clock kernel would not be, since
    # leap seconds are the same fact in every version of the kernel that
    # states them.
    lsk = furnish_supporting_kernels(recorded_basenames(documents), paths, _LSK_SUFFIXES)
    MAIN_LOGGER.info('Furnished leapseconds kernel(s): %s', ', '.join(lsk))

    documents, undated = select_by_time(
        documents,
        None if arguments.start_time is None else float(cspyce.utc2et(arguments.start_time)),
        None if arguments.stop_time is None else float(cspyce.utc2et(arguments.stop_time)),
    )
    if undated > 0:
        MAIN_LOGGER.warning(
            'Ignored %d image(s) that recorded no exposure midtime to place in time', undated
        )
    if len(documents) == 0:
        MAIN_LOGGER.warning('No images selected; nothing to write')
        sys.exit(0)
    MAIN_LOGGER.info('Selected %d image(s)', len(documents))

    # Read from the selected images only.  A run restricted to a time range
    # must not be refused for a disagreement between two kernel versions that
    # only images outside that range recorded.
    basenames = recorded_basenames(documents)
    frames = furnish_supporting_kernels(basenames, paths, _FK_SUFFIXES)
    MAIN_LOGGER.info('Furnished frame kernel(s): %s', ', '.join(frames))

    entries = [_entry_for(document) for document in documents]
    sclk_basenames = clock_kernels(entries, basenames, paths)
    for sclk_id, basename in sorted(sclk_basenames.items()):
        MAIN_LOGGER.info('Spacecraft clock %d is encoded with %s', sclk_id, basename)
        cspyce.furnsh(str(cast(Path, resolve_one(basename, paths).retrieve())))

    index = build_ck_index(arguments.kernel_dir)
    MAIN_LOGGER.info('Indexed %d candidate C-kernel(s)', len(index.files))

    assignments = assign_images(entries, index)
    rows, facts_by_name = report_rows(documents, assignments)

    written = write_output_files(
        assignments,
        facts_by_name,
        output_dir,
        sclk_basenames=sclk_basenames,
        configuration_hash=DEFAULT_CONFIG.resolved_config_hash(),
    )

    if len(written) > 0:
        meta_kernel = output_dir / f'{arguments.mission}{META_KERNEL_SUFFIX}'
        write_meta_kernel(
            meta_kernel,
            originals=[str(original) for original, _correction in written],
            corrections=[str(correction) for _original, correction in written],
        )
        MAIN_LOGGER.info('Wrote meta-kernel %s', meta_kernel.as_posix())
    else:
        MAIN_LOGGER.warning('No image received a corrected segment; no meta-kernel written')

    report = output_dir / f'{arguments.mission}{REPORT_SUFFIX}'
    write_report(report, rows)
    MAIN_LOGGER.info('Wrote report %s with %d row(s)', report.as_posix(), len(rows))

    log_totals(log_dispositions(documents, assignments, run_logging))
    MAIN_LOGGER.info('Total elapsed time %.2f sec', time.time() - start_time)
    sys.exit(0)


def _entry_for(document: Document) -> ImageEntry:
    """Read one document's generator entry, naming the image if it cannot be read.

    Parameters:
        document: The document.

    Returns:
        The entry.

    Raises:
        TypeError: if the document holds a value of the wrong kind.
        ValueError: if a field the generator needs is absent or unusable.
    """
    try:
        return ImageEntry.from_metadata(document.metadata)
    except (TypeError, ValueError) as exc:
        MAIN_LOGGER.error(
            '%s: cannot be read as a navigated image: %s', document.path.as_posix(), exc
        )
        raise


if __name__ == '__main__':
    main()

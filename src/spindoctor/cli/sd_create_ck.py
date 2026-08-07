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
import os
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cspyce
import pdslogger
from filecache import FCPath, FileCache

# Make CLI runnable from source tree with
#    python src/package
package_source_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, package_source_path)

from spindoctor import __version__
from spindoctor.cli.ck.assignment import Assignment, assign_images, group_for_output
from spindoctor.cli.ck.comments import CommentArea, build_comment_lines
from spindoctor.cli.ck.images import ImageEntry, OmissionReason
from spindoctor.cli.ck.index import CkFile, build_ck_index
from spindoctor.cli.ck.inputs import (
    LSK_SUFFIXES,
    Document,
    clock_kernels,
    furnish_frame_kernels,
    furnish_supporting_kernels,
    furnished,
    kernel_paths,
    read_documents,
    recorded_basenames,
    resolve_one,
    select_by_time,
)
from spindoctor.cli.ck.kernel_file import write_ck_file
from spindoctor.cli.ck.metakernel import write_meta_kernel
from spindoctor.cli.ck.pointing import ImagePointing
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

REPORT_SUFFIX = '_ck_report.csv'
META_KERNEL_SUFFIX = '_nav.tm'


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
# BUILDING, THEN WRITING
#
################################################################################


@dataclass(frozen=True)
class PreparedFile:
    """One corrected C-kernel, built in memory and not yet written.

    Parameters:
        baseline: The original C-kernel this file mirrors.
        output: Where the corrected file goes.
        segments: Its segments, in the order the images were assigned.
        comment_lines: The comment area to attach to it.
    """

    baseline: CkFile
    output: FCPath
    segments: tuple[CkSegment, ...]
    comment_lines: tuple[str, ...]


def image_facts(documents: Sequence[Document]) -> dict[str, ImageFacts]:
    """Read the facts the report and the comment areas both carry, by image name.

    Parameters:
        documents: The documents the run considered.

    Returns:
        One entry per image, keyed by the name recorded for it.

    Raises:
        ValueError: if a document holds a value the report cannot render, or if
            two documents name the same image, whose facts could not then be
            told apart and one of which would silently replace the other in
            whichever kernel's comment area carried it.
    """
    facts_by_name: dict[str, ImageFacts] = {}
    for document in documents:
        facts = read_image_facts(document.metadata)
        if facts.image_name in facts_by_name:
            raise ValueError(
                f'two documents name the image {facts.image_name!r}; their reported facts cannot '
                f'be told apart'
            )
        facts_by_name[facts.image_name] = facts
    return facts_by_name


def report_rows(
    assignments: Sequence[Assignment], facts_by_name: Mapping[str, ImageFacts]
) -> list[ReportRow]:
    """Build the report, one row per image the run considered.

    Parameters:
        assignments: What became of each image the run considered, in report
            order.
        facts_by_name: The reported facts of each image, by name.

    Returns:
        The rows.

    Raises:
        KeyError: if an assignment names an image the facts do not hold, which
            means the two were read from different sets of documents.
    """
    return [
        ReportRow(
            facts=facts_by_name[assignment.image_name],
            source_bc=assignment.output_name,
            omission_reason=assignment.omission_reason,
        )
        for assignment in assignments
    ]


def build_output_files(
    assignments: Sequence[Assignment],
    facts_by_name: Mapping[str, ImageFacts],
    output_dir: FCPath,
    *,
    sclk_basenames: Mapping[int, str],
    configuration_hash: str,
) -> tuple[PreparedFile, ...]:
    """Build every corrected kernel a run will write, and write none of them.

    Nothing here reaches the filesystem, so a failure part way through leaves
    the output directory as it found it.  That is the point of separating the
    two phases: the alternative is a run that dies with some kernels written,
    no meta-kernel, and no report -- the one artifact that says what is in the
    files that did get written.

    Parameters:
        assignments: What became of each image the run considered.
        facts_by_name: The reported facts of each image, by name.
        output_dir: Where the corrected kernels will go.
        sclk_basenames: The clock kernel each spacecraft clock is encoded with.
        configuration_hash: Digest of the configuration this run used.

    Returns:
        The files to write, ordered by name.

    Raises:
        ValueError: if a segment cannot be built: a baseline supplying angular
            velocity at only some of an exposure's records, an exposure needing
            more records than a segment holds, or an image carrying no
            pointing.
        KeyError: if a frame the correction is expressed between is not defined
            by the furnished kernels.
        OSError: if a baseline kernel cannot be furnished, or supplies no
            pointing at a record epoch.
    """
    # Every segment of the run is resident at once, which is what buys the
    # separation.  Measured on this writer's own segments, one costs 678 bytes
    # for the three records an ordinary exposure carries and 1826 bytes for the
    # 21 a twenty-second one does, so the 50,000-image batch this tool is sized
    # for holds about 34 MB of segments -- a few percent of the per-image
    # metadata such a batch already has resident, and flat in the number of
    # output files.  A batch of exposures long enough to hit the record cap
    # would be a different story: 50,000 segments of 10,000 records each is
    # 32 GB, and no instrument commands such an exposure.
    prepared: list[PreparedFile] = []
    for group in group_for_output(assignments):
        with furnished(group.baseline.path):
            segments = tuple(_segment_for(assignment) for assignment in group.assignments)
        area = CommentArea(
            generator_version=__version__,
            configuration_hash=configuration_hash,
            baseline_basenames=(group.baseline.basename,),
            sclk_basename=sclk_basenames[_clock_of(group.assignments[0])],
            images=tuple(facts_by_name[assignment.image_name] for assignment in group.assignments),
        )
        prepared.append(
            PreparedFile(
                baseline=group.baseline,
                output=output_dir / group.name,
                segments=segments,
                comment_lines=tuple(build_comment_lines(area)),
            )
        )
    return tuple(prepared)


def write_output_files(files: Sequence[PreparedFile]) -> list[tuple[FCPath, FCPath]]:
    """Write the corrected kernels a run has already built.

    Parameters:
        files: The built files, in the order they are to be written.

    Returns:
        One entry per file written, pairing the original with the correction.

    Raises:
        ValueError: if SPICE refuses a file's records or its comment area, or
            if a corrected kernel of that name already exists.
        RuntimeError: if SPICE cannot create a file.
        OSError: if SPICE refuses a write for an operating-system reason.
    """
    written: list[tuple[FCPath, FCPath]] = []
    for prepared in files:
        write_ck_file(local_output_path(prepared.output), prepared.segments, prepared.comment_lines)
        MAIN_LOGGER.info(
            'Wrote %s: %d segment(s) correcting %s',
            prepared.output.as_posix(),
            len(prepared.segments),
            prepared.baseline.basename,
        )
        written.append((prepared.baseline.path, prepared.output))
    return written


def absolute_directory(directory: str) -> str:
    """Return one directory named on the command line, resolved to absolute.

    A remote directory is passed through: it is already absolute, and there is
    no local working directory to resolve it against.

    Parameters:
        directory: The directory as it was typed.

    Returns:
        The same directory, absolute if it is local.
    """
    if '://' in directory:
        return directory
    return str(Path(directory).resolve())


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
            a baseline cannot; if the baseline supplies angular velocity at
            only some of the record epochs, which no segment can express; or if
            the recorded exposure would need more records than a segment holds.
        OSError: if the baseline supplies no pointing at a record epoch.
    """
    pointing = pointing_of(assignment)
    try:
        return build_segment(pointing)
    except (OSError, KeyError, ValueError) as exc:
        # The run log only, deliberately.  This stops the run, and the image
        # log it would otherwise be written to is opened by the reporting pass
        # that never gets to run; logging through the image logger with no
        # image scope open is a defect rather than a fallback, and under
        # strict_scope it would replace this error with a scope error and
        # demote the real one to a context.  It stops the run because the
        # reason set the report may use has no entry for an image whose
        # baseline reproduced its attitude and then could not supply a record
        # epoch or its angular velocity, and inventing one would be a schema
        # change for every consumer of the report.
        MAIN_LOGGER.exception(
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
    return resolve_sclk_id(pointing_of(assignment).ck_frame_id)


def pointing_of(assignment: Assignment) -> ImagePointing:
    """Return the recorded pointing of an image that is getting a segment.

    Parameters:
        assignment: The image and the baseline it corrects.

    Returns:
        Its recorded pointing.

    Raises:
        ValueError: if the image carries none.  An assignment that names a
            baseline cannot, so this is the guard on a caller that passed one
            that names none instead.
    """
    pointing = assignment.entry.pointing
    if pointing is None:
        raise ValueError(f'{assignment.image_name} has no pointing to write')
    return pointing


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
    # Resolved to absolute, both of them, because the meta-kernel names the
    # kernels it furnishes by these paths and SPICE resolves a relative name
    # against the *consumer's* working directory.  A meta-kernel written with
    # relative names works only from the directory that generated it, and
    # elsewhere fails on the first correction -- after the originals have
    # already loaded, so the pool is left uncorrected rather than empty.
    output_dir = FileCache(None).new_path(absolute_directory(arguments.output_dir))
    kernel_dirs = [absolute_directory(directory) for directory in arguments.kernel_dir]

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
    for path, reason in unreadable:
        MAIN_LOGGER.error('Could not read %s: %s', path.as_posix(), reason)
    if len(unreadable) > 0:
        MAIN_LOGGER.error('Could not read %d metadata file(s)', len(unreadable))

    paths = kernel_paths(kernel_dirs)
    # The leapseconds kernel is furnished before the time filter, because the
    # filter's own bounds are UTC and there is nothing to convert them with
    # until it is.  Its candidates are therefore the whole mission's, not the
    # selection's; that is safe where a clock kernel would not be, since
    # leap seconds are the same fact in every version of the kernel that
    # states them.
    lsk = furnish_supporting_kernels(recorded_basenames(documents), paths, LSK_SUFFIXES)
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
    # The entries are read before any frame kernel is furnished, because they
    # need no SPICE and they name the frames the frame kernels are checked
    # against.
    entries = [_entry_for(document) for document in documents]
    frames = furnish_frame_kernels(entries, basenames, paths)
    MAIN_LOGGER.info('Furnished frame kernel(s): %s', ', '.join(frames))

    sclk_basenames = clock_kernels(entries, basenames, paths)
    for sclk_id, basename in sorted(sclk_basenames.items()):
        MAIN_LOGGER.info('Spacecraft clock %d is encoded with %s', sclk_id, basename)
        cspyce.furnsh(str(cast(Path, resolve_one(basename, paths).retrieve())))

    index = build_ck_index(kernel_dirs)
    MAIN_LOGGER.info('Indexed %d candidate C-kernel(s)', len(index.files))

    assignments = assign_images(entries, index)
    facts_by_name = image_facts(documents)

    # Everything is built before anything is written.  A run that dies part way
    # through the build then leaves the output directory untouched, rather than
    # some corrected kernels, no meta-kernel, and no report saying what is in
    # them.
    prepared = build_output_files(
        assignments,
        facts_by_name,
        output_dir,
        sclk_basenames=sclk_basenames,
        configuration_hash=DEFAULT_CONFIG.resolved_config_hash(),
    )
    rows = report_rows(assignments, facts_by_name)
    written = write_output_files(prepared)

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
    # Non-zero when anything the run was pointed at could not be read, so a
    # batch wrapper can tell a clean run from one that silently skipped its
    # input.  An image the run considered and omitted for a reason is not that:
    # it is in the report, which is the answer, and it exits zero.
    sys.exit(1 if len(unreadable) > 0 else 0)


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
        # The run log only, for the same reason the segment failure reports
        # there: this stops the run, and no image scope is open to write to.
        MAIN_LOGGER.exception(
            '%s: cannot be read as a navigated image: %s', document.path.as_posix(), exc
        )
        raise


if __name__ == '__main__':
    main()

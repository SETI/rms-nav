#!/usr/bin/env python3
"""Read one or more navigation results roots into the results index.

Dispatch script for the ``sd_stats_ingest`` console entry point.  See
``spindoctor.cli.stats`` for the statistics-system overview.

The roots are resolved the way every consumer resolves a navigation results
root -- the command line, then ``environment.nav_results_root``, then
``NAV_RESULTS_ROOT`` -- because the root is half of the key every row is stored
under.  Ingesting a subdirectory of a results root would produce stubs no
consumer's lookup can match.

Ingest is never automatic: no batch driver runs it as a side effect, and the
index it writes is a snapshot of the tree as of this run.

One pass may also be spread over a queue of workers.  ``--output-cloud-tasks-file``
lists each root once, removes the rows whose documents have left the tree, and
writes out the shares for ``sd_stats_ingest_cloud_tasks`` to read; when those
have run, ``--complete-cloud-tasks-file`` adds their tallies up and stamps each
root's ingest as finished.  Until that last step a consumer treats the roots as
ones nobody has ingested, which is what keeps absence of a row from being read
as an answer while the workers are still writing.
"""

import argparse
import os
import sys

import sqlalchemy
from filecache import FCPath

# Make CLI runnable from source tree with
#    python src/package
package_source_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, package_source_path)

from spindoctor.cli.logging_args import add_logging_arguments, reporting_logging_errors
from spindoctor.cli.stats.ingest import (
    IngestCounts,
    TaskCompletion,
    complete_ingest_tasks,
    fan_out_ingest_tasks,
    ingest_metadata_files,
    task_results_from_event_log,
)
from spindoctor.config import (
    DEFAULT_CONFIG,
    MAIN_LOGGER,
    build_run_logging,
    get_nav_results_root,
    get_results_db_url,
    load_default_and_user_config,
)
from spindoctor.config.program_names import SD_STATS_INGEST
from spindoctor.results_index import masked_url, open_index
from spindoctor.support.file import json_as_string

PROGRAM_NAME = SD_STATS_INGEST
"""Program identity: names the main log directory and the
``logging.programs`` configuration block for this program."""

URL_OPTIONS = ('--results-db',)
"""Options whose value is a connection URL and can therefore carry a password.

Only these are masked in the logged command line.  A results root is not a
connection URL: it has no credentials to hide, and it is the one word of the
command line an operator reads the run log to correct, so masking one would
corrupt the string and protect nothing.
"""


def _names_a_url_option(word: str) -> bool:
    """Whether a command-line word names an option whose value is a URL.

    Any distinguishing prefix of a long option is the option: argparse accepts
    ``--results-d`` for ``--results-db`` and consumes the URL after it just the
    same, so matching the full spelling alone would leave the abbreviated
    command line unmasked.  A prefix that argparse would have rejected never
    reaches here, since parsing runs first and exits on one.

    Parameters:
        word: One word of the command line, without any ``=value`` part.

    Returns:
        True when the word names one of :data:`URL_OPTIONS`.
    """
    if not word.startswith('--') or word == '--':
        return False
    return any(option.startswith(word) for option in URL_OPTIONS)


def parse_args(command_list: list[str]) -> argparse.Namespace:
    """Build the parser and read the command line.

    Parameters:
        command_list: Arguments, without the program name.

    Returns:
        The parsed arguments.
    """
    cmdparser = argparse.ArgumentParser(
        description='Read navigation metadata documents into the results index.'
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
        action='append',
        dest='nav_results_roots',
        default=None,
        metavar='ROOT',
        help="""Root directory of a navigation results tree to read (a local
        directory or any URL the filecache layer accepts); may be specified
        multiple times. Overrides NAV_RESULTS_ROOT and the nav_results_root
        configuration variable.""",
    )
    environment_group.add_argument(
        '--results-db',
        default=None,
        metavar='URL',
        help="""Connection URL of the results index to write (a sqlite: URL
        naming a local path, or a postgresql+psycopg: URL naming a server);
        overrides NAV_RESULTS_DB and the environment.results_db configuration
        variable. The tables are created if they are absent.""",
    )

    ingest_group = cmdparser.add_argument_group('Ingest')
    ingest_group.add_argument(
        '--force',
        action='store_true',
        default=False,
        help="""Read every document, including ones whose recorded size and
        modification time still match the tree. Refused with
        --complete-cloud-tasks-file, which reads no document.""",
    )

    cloud_group = cmdparser.add_argument_group('Cloud tasks')
    cloud_mode = cloud_group.add_mutually_exclusive_group()
    cloud_mode.add_argument(
        '--output-cloud-tasks-file',
        default=None,
        metavar='PATH',
        help="""Write a JSON task descriptions file suitable for loading into a
        cloud_tasks queue (consumed by sd_stats_ingest_cloud_tasks) and read no
        document here. Each root is still listed once, and the rows whose
        documents have left it are still removed; its ingest stays unfinished
        until --complete-cloud-tasks-file adds up what the workers did.""",
    )
    cloud_mode.add_argument(
        '--complete-cloud-tasks-file',
        default=None,
        metavar='PATH',
        help="""Read the cloud_tasks event log the workers wrote, add up what
        their tasks did, and record it against each named root's ingest run.
        A root whose tasks do not account for every file its listing found is
        left unfinished and named.""",
    )

    add_logging_arguments(cmdparser, has_image_logger=False)

    return cmdparser.parse_args(command_list)


def masked_command_line(command_list: list[str]) -> list[str]:
    """Return a command line with the value of every connection-URL option masked.

    The command line is logged because which of the command line, the
    configuration file and the environment supplied a value is exactly what a
    reader of a failed run needs to know, and one of its words can be a database
    password.  Every spelling argparse accepts is covered: the value as a
    separate word, the value joined to the option by ``=``, and either of those
    under an abbreviation of the option's name.

    Parameters:
        command_list: The arguments, without the program name.

    Returns:
        The arguments, with every connection URL among them masked.
    """
    masked: list[str] = []
    value_of_url_option = False
    for word in command_list:
        if value_of_url_option:
            masked.append(masked_url(word))
            value_of_url_option = False
            continue
        option, separator, value = word.partition('=')
        if separator and _names_a_url_option(option):
            masked.append(f'{option}={masked_url(value)}')
            continue
        masked.append(word)
        value_of_url_option = _names_a_url_option(word)
    return masked


def _log_outcome(counts: IngestCounts) -> None:
    """Write the closing summary of an ingest pass to the main log.

    The failures are tallied by reason as well as counted, because a results
    tree holds many ``*_metadata.json`` files that were never navigation
    documents.  Several hundred of those are ordinary; several hundred
    navigation results that would not parse are not, and the tally is what
    tells the two apart at a glance.  Each reason names one file that carried
    it, because a reason is a field-level diagnosis and one look at a real file
    is what turns it into a judgement about the tree.

    Parameters:
        counts: What the pass did.
    """
    MAIN_LOGGER.info('Metadata files seen: %d', counts.files_seen)
    MAIN_LOGGER.info('Ingested: %d', counts.files_ingested)
    MAIN_LOGGER.info('Skipped as unchanged: %d', counts.files_skipped)
    MAIN_LOGGER.info('Rows removed, their document gone from the tree: %d', counts.files_removed)
    MAIN_LOGGER.info('Not ingestible: %d', counts.files_failed)
    for reason in sorted(counts.failures_by_reason):
        MAIN_LOGGER.info(
            '    %s: %d file(s), for example %s',
            reason,
            counts.failures_by_reason[reason],
            counts.example_by_reason.get(reason, '(none recorded)'),
        )
    if counts.directories_missed:
        MAIN_LOGGER.warning(
            'Directories not listed, whose files were therefore never seen: %d. Absence of '
            'a row under one of them is not evidence that its image was never navigated.',
            counts.directories_missed,
        )
    if counts.roots_unreadable:
        MAIN_LOGGER.error(
            'Roots that could not be listed and are therefore not ingested: %d',
            counts.roots_unreadable,
        )


def _log_completion(completion: TaskCompletion) -> None:
    """Write the closing summary of a cloud-task completion to the main log.

    Parameters:
        completion: What adding the shares up did.
    """
    _log_outcome(completion.counts)
    MAIN_LOGGER.info('Ingest runs completed: %d', completion.runs_completed)
    for root in completion.roots_unaccounted:
        MAIN_LOGGER.error(
            'Left unfinished, because its tasks did not account for every file its listing '
            'found: %s',
            root,
        )
    for root in completion.roots_unlisted:
        MAIN_LOGGER.error(
            'Left unfinished, because its ingest run never recorded what its listing found, '
            'so nothing says what this root holds: %s. Divide the root up with '
            '--output-cloud-tasks-file, run its tasks, and complete that run.',
            root,
        )
    for root in completion.roots_without_a_run:
        MAIN_LOGGER.error(
            'No unfinished ingest run to complete for %s: run --output-cloud-tasks-file over '
            'that root first, and complete the run that fanned out',
            root,
        )
    if completion.results_failed:
        MAIN_LOGGER.error(
            'Tasks that reported an error instead of a share: %d', completion.results_failed
        )
    if completion.results_unreadable:
        MAIN_LOGGER.error(
            'Task results that are not the shape a share reports: %d',
            completion.results_unreadable,
        )
    if completion.results_unclaimed:
        MAIN_LOGGER.warning(
            'Task results naming an ingest run none of these roots is waiting on: %d. They '
            'belong to another fan-out, and are left for whoever completes it.',
            completion.results_unclaimed,
        )
    if completion.results_of_another_root:
        MAIN_LOGGER.error(
            'Task results naming a run being completed here but reporting rows under a '
            "different root: %d. They are not that run's shares and are not counted toward "
            'it: a run number is only unique within the index that minted it, so a task file '
            'that outlived its index names a run of whatever was built next.',
            completion.results_of_another_root,
        )
    if completion.results_superseded:
        MAIN_LOGGER.info(
            'Task results superseded by a later report of the same task: %d. A queue '
            'delivers a task again whenever it could not see the last delivery '
            'acknowledged, and one share reported twice is still one share.',
            completion.results_superseded,
        )
    if completion.results_unidentified:
        MAIN_LOGGER.error(
            'Task results naming no task: %d. One of them cannot be told from a repeat of '
            'another, so none of them is counted toward a run.',
            completion.results_unidentified,
        )


def _run_ingest(engine: sqlalchemy.Engine, roots: list[str], *, force: bool) -> int:
    """Read every document under each root and write its rows.

    Parameters:
        engine: The open index.
        roots: The navigation results roots to walk.
        force: Whether to re-read every document.

    Returns:
        The exit status: 0 when every named root was walked, 1 when one could
        not be listed.
    """
    counts = ingest_metadata_files(engine, roots, force=force, logger=MAIN_LOGGER)
    _log_outcome(counts)
    # Whether the run completed, not what it found.  A count of documents flips
    # between two passes over one unchanged tree -- what one pass ingests the
    # next one skips, and what one pass refuses the next one skips too -- so a
    # status read from a count tells a scheduled run that a tree it has already
    # accounted for has gone wrong.  A root that could not be listed is the
    # failure: nothing under it was walked, and every later root of the same
    # pass is still walked, so the status is the only place it shows.
    return 1 if counts.roots_unreadable else 0


def _write_cloud_tasks(
    engine: sqlalchemy.Engine, roots: list[str], *, force: bool, path: str
) -> int:
    """List each root once and write out the shares its documents divide into.

    Parameters:
        engine: The open index.
        roots: The navigation results roots to walk.
        force: Whether the workers should re-read every document.
        path: Where to write the task descriptions.

    Returns:
        The exit status: 0 when every named root was listed, 1 when one could
        not be.
    """
    MAIN_LOGGER.info('Writing cloud_tasks file to %s', path)
    fan_out = fan_out_ingest_tasks(engine, roots, force=force, logger=MAIN_LOGGER)
    with FCPath(path).open('w') as file:
        file.write(json_as_string(fan_out.tasks))
    MAIN_LOGGER.info('Wrote %d task(s) to %s', len(fan_out.tasks), path)
    MAIN_LOGGER.info('Metadata files seen: %d', fan_out.counts.files_seen)
    MAIN_LOGGER.info(
        'Rows removed, their document gone from the tree: %d', fan_out.counts.files_removed
    )
    if fan_out.counts.directories_missed:
        MAIN_LOGGER.warning(
            'Directories not listed, whose files were therefore never seen: %d. Absence of '
            'a row under one of them is not evidence that its image was never navigated.',
            fan_out.counts.directories_missed,
        )
    if fan_out.counts.roots_unreadable:
        MAIN_LOGGER.error(
            'Roots that could not be listed and are therefore not ingested: %d',
            fan_out.counts.roots_unreadable,
        )
    MAIN_LOGGER.info(
        'Each root stays unfinished until the workers have run and '
        '--complete-cloud-tasks-file has added up what they did'
    )
    return 1 if fan_out.counts.roots_unreadable else 0


def _complete_cloud_tasks(engine: sqlalchemy.Engine, roots: list[str], *, path: str) -> int:
    """Add up what the workers did and stamp the runs they completed.

    Parameters:
        engine: The open index.
        roots: The navigation results roots whose runs are being completed.
        path: The cloud_tasks event log the workers wrote.

    Returns:
        The exit status: 0 when every named root's ingest run was completed, 1
        when one was not, and 1 when the event log could not be read.
    """
    MAIN_LOGGER.info('Reading task results from %s', path)
    try:
        found = task_results_from_event_log(FCPath(path))
    except (OSError, UnicodeDecodeError) as exc:
        # An ordinary mistyped path, which the pass enumerates and charges to
        # the file rather than letting out as a traceback.  A path naming a file
        # that is not text -- a gzipped log, a database, an image -- is the same
        # error and is charged the same way; it raises a UnicodeDecodeError,
        # which is a ValueError rather than an OSError.  Every named root keeps
        # its unfinished run, so no consumer reads absence under one of them as
        # an answer.
        MAIN_LOGGER.fatal('Cannot read the task event log %s: %s', path, exc)
        return 1
    MAIN_LOGGER.info('Task results read: %d', len(found.results))
    if found.lines_unread:
        MAIN_LOGGER.warning(
            'Lines of %s that are not events: %d. A log being appended to while it is read '
            'ends in a partial line; many of them say this is not an event log.',
            path,
            found.lines_unread,
        )
    if found.tasks_unfinished:
        MAIN_LOGGER.error(
            'Tasks that ended without returning a share: %d. Their documents were never '
            'read, so the roots they belong to cannot be completed.',
            found.tasks_unfinished,
        )
    completion = complete_ingest_tasks(engine, roots, found.results, logger=MAIN_LOGGER)
    _log_completion(completion)
    unfinished = (
        completion.roots_unaccounted + completion.roots_unlisted + completion.roots_without_a_run
    )
    return 1 if unfinished else 0


def main() -> None:
    """Console entry point for ``sd_stats_ingest``.

    Resolves the index URL and the results roots and, according to the mode
    named on the command line, reads every document under those roots, divides
    them into cloud tasks, or adds up what those tasks did.  The outcome goes to
    the main log either way.

    Raises:
        SystemExit: Always, since this is a console entry point.  The status
            says whether the pass completed, not what it found: 0 when every
            named root was walked, whatever mix of documents was read, skipped
            and refused, and 1 when the run could not complete -- no index or no
            root could be resolved, the index could not be opened, or a root
            could not be listed.  A tree of files that are not navigation
            documents is a completed pass and exits 0, and exits 0 again on the
            next pass over the same tree, so a scheduled run's status means the
            same thing every time it is read.  Completing a fan-out exits 1 when
            the event log cannot be read, when a named root has no unfinished
            run, when its run never recorded what its listing found, or when its
            tasks did not account for every file that listing found.
    """
    command_list = sys.argv[1:]
    arguments = parse_args(command_list)
    # One pass may cover several roots, which is what makes ingest different
    # from every other program; the shared root resolver reads one attribute,
    # so the first named root is the one this run is logged under and the rest
    # are further trees to walk.  With none named, the resolver falls through to
    # the configuration variable and the environment as it does everywhere else.
    arguments.nav_results_root = (arguments.nav_results_roots or [None])[0]

    # Read configuration files
    with reporting_logging_errors():
        load_default_and_user_config(arguments, DEFAULT_CONFIG)
    with reporting_logging_errors():
        build_run_logging(PROGRAM_NAME, arguments, DEFAULT_CONFIG)

    url = get_results_db_url(arguments, DEFAULT_CONFIG)
    if url is None:
        MAIN_LOGGER.fatal(
            'No results index was named. Give one with --results-db, the '
            'environment.results_db configuration variable, or NAV_RESULTS_DB.'
        )
        sys.exit(1)

    try:
        roots = arguments.nav_results_roots or [get_nav_results_root(arguments, DEFAULT_CONFIG)]
    except ValueError as exc:
        MAIN_LOGGER.fatal('No navigation results root was named: %s', exc)
        sys.exit(1)

    # Completing a fan-out reads runs the fan-out already recorded, so an index
    # that is not there is a wrong URL rather than a first run: creating an
    # empty one would answer "no run to complete" for a root whose run is
    # sitting in the index the operator meant to name.
    completing = arguments.complete_cloud_tasks_file is not None

    MAIN_LOGGER.info('Starting results index ingest')
    MAIN_LOGGER.info('Roots: %s', ', '.join(roots))
    if not completing:
        MAIN_LOGGER.info('Force: %s', arguments.force)
    MAIN_LOGGER.info('Arguments: %s', masked_command_line(command_list))

    if completing and arguments.force:
        # Refused rather than ignored.  Completion reads no document, so there
        # is nothing for --force to re-read; an operator who typed it meant the
        # shares to be read again, and that is a property of the fan-out that
        # cut them, decided one step earlier.
        MAIN_LOGGER.fatal(
            '--force has no meaning when adding up what the workers did, since no document '
            'is read here. Re-run --output-cloud-tasks-file with --force and run the tasks '
            'it writes.'
        )
        sys.exit(1)
    try:
        engine = open_index(url, create=not completing)
    except ValueError as exc:
        MAIN_LOGGER.fatal('Cannot open the results index: %s', exc)
        sys.exit(1)
    try:
        if arguments.output_cloud_tasks_file is not None:
            status = _write_cloud_tasks(
                engine, roots, force=arguments.force, path=arguments.output_cloud_tasks_file
            )
        elif completing:
            status = _complete_cloud_tasks(engine, roots, path=arguments.complete_cloud_tasks_file)
        else:
            status = _run_ingest(engine, roots, force=arguments.force)
    except Exception as exc:
        # The pass enumerates every failure it expects and charges it to one
        # file or one root.  Anything still escaping is a failure nobody
        # enumerated, and a console entry point owes its caller a message and a
        # status for one rather than a traceback: the run rows of the roots it
        # did not reach keep their NULL finish times, so no consumer reads
        # absence under them as an answer.
        MAIN_LOGGER.fatal('Ingest could not complete (%s: %s)', type(exc).__name__, exc)
        MAIN_LOGGER.exception(exc)
        sys.exit(1)
    finally:
        engine.dispose()
    sys.exit(status)


if __name__ == '__main__':
    main()

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

``--drop-index`` is the opposite operation and shares only the URL with the
rest: it removes the index's own tables from the database that URL names and
stops there, walking no tree.  It is what makes emptying an index and starting
over something an operator can reach without hand-written SQL, on a shared
PostgreSQL server as well as on a file -- which the schema version gate depends
on, since a version bump is deliberately not migrated and rebuilding is the
whole of the remedy.  It drops from one schema, the one this database's own
stamp table was found in, and only after finding that stamp: a table called
``images`` is not evidence of anything.

The two halves rest on one rule.  An ingest builds an index only in a schema
that holds nothing, or one already carrying a stamp of SpinDoctor's, and refuses
a schema holding a table it did not create -- so a stamp never comes to stand
over somebody else's table, and the six names the drop removes from a stamped
schema are six tables SpinDoctor created.

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
from spindoctor.cli.stats.drop import drop_results_index
from spindoctor.cli.stats.ingest import (
    IngestCounts,
    TaskCompletion,
    UnlistableDirectoryError,
    complete_ingest_tasks,
    distinct_roots,
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
from spindoctor.results_index import open_index
from spindoctor.support.command_line import masked_command_line
from spindoctor.support.file import json_as_string

PROGRAM_NAME = SD_STATS_INGEST
"""Program identity: names the main log directory and the
``logging.programs`` configuration block for this program."""


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
        overrides the environment.results_db configuration variable and
        NAV_RESULTS_DB. The tables are created if they are absent, in the schema this
        database's own schema_meta stamp was found in or, where there is no such
        stamp, the one a table created without a schema name lands in. That
        schema is refused, and nothing is created or stamped in it, when it
        already holds any table the index does not own or any table of the
        index's own names that no stamp of ours stands over.""",
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

    drop_group = cmdparser.add_argument_group('Drop')
    drop_group.add_argument(
        '--drop-index',
        action='store_true',
        default=False,
        help="""Remove the results index's own tables from the database
        --results-db names, and stop: no results root is read and no document is
        ingested. The tables that go are the index's own six names, from the one
        schema this database's own schema_meta stamp was found in; no other
        table of that schema, and no other schema, is touched. What makes those
        six SpinDoctor's own is that an ingest refuses to build an index in a
        schema holding anything it did not create, so a schema carrying that
        stamp holds this index and nothing else. A database holding none of
        those tables is left alone and said to be, and one holding tables of
        those names that no stamp of ours stands over is refused. Refused
        together with --force, --nav-results-root and either cloud-tasks mode,
        none of which this does.""",
    )
    drop_group.add_argument(
        '--yes',
        action='store_true',
        default=False,
        help="""Drop without asking for confirmation, for a run with nobody at
        the terminal. Refused without --drop-index, which is the only thing this
        program asks about.""",
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
        A root whose tasks do not account for exactly the files its listing
        found is left unfinished and named.""",
    )

    add_logging_arguments(cmdparser, has_image_logger=False)

    return cmdparser.parse_args(command_list)


def _log_outcome(counts: IngestCounts) -> None:
    """Write the closing summary of an ingest pass to the main log.

    The lines themselves are the pass's own, so that the report over a results
    tree -- which runs this same pass into a temporary index -- tells an operator
    about it in the same words.

    Parameters:
        counts: What the pass did.
    """
    summary = counts.summary()
    for line in summary.lines:
        MAIN_LOGGER.info('%s', line)
    for line in summary.failures:
        MAIN_LOGGER.error('%s', line)


def _log_completion(completion: TaskCompletion) -> None:
    """Write the closing summary of a cloud-task completion to the main log.

    Parameters:
        completion: What adding the shares up did.
    """
    _log_outcome(completion.counts)
    MAIN_LOGGER.info('Ingest runs completed: %d', completion.runs_completed)
    for root in completion.roots_unaccounted:
        MAIN_LOGGER.error(
            'Left unfinished, because its tasks did not account for exactly the files its '
            'listing found: %s',
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


def _mode_refusal(arguments: argparse.Namespace) -> str | None:
    """Return why a command line cannot be run as it stands, or None.

    Refused rather than ignored, on the same grounds as ``--force`` under a
    completion below: an operator who typed an option meant something by it, and
    a program that silently does one of the two things asked of it has decided
    which one on their behalf.  ``--drop-index`` removes the index and stops, so
    every option describing an ingest is at odds with it rather than modifying
    it; and ``--yes`` answers a question only the drop asks.

    ``--nav-results-root`` is refused only when it was typed.  The same root
    reaches this program from the configuration and from the environment, where
    it is a machine's standing setting rather than a request, and refusing a
    drop because the machine has a results tree would refuse it nearly
    everywhere.

    Parameters:
        arguments: The parsed command line.

    Returns:
        The refusal to report, or None when the arguments agree with each other.
    """
    if not arguments.drop_index:
        if arguments.yes:
            return (
                '--yes says not to ask before dropping the results index, and this command '
                'line asks for no drop. Add --drop-index, or leave --yes off.'
            )
        return None
    conflicting = [
        name
        for name, given in (
            ('--force', arguments.force),
            ('--nav-results-root', arguments.nav_results_roots is not None),
            ('--output-cloud-tasks-file', arguments.output_cloud_tasks_file is not None),
            ('--complete-cloud-tasks-file', arguments.complete_cloud_tasks_file is not None),
        )
        if given
    ]
    if conflicting:
        return (
            f'--drop-index removes the index and stops, reading no document, so it has '
            f'nothing to do with {", ".join(conflicting)}. Drop first, then run the ingest '
            f'you want against what it left behind.'
        )
    return None


def main() -> None:
    """Console entry point for ``sd_stats_ingest``.

    Resolves the index URL and, unless the command line asks for a drop, the
    results roots as well; then, according to the mode named on the command
    line, removes the index's tables, reads every document under those roots,
    divides them into cloud tasks, or adds up what those tasks did.  The outcome
    goes to the main log whichever mode ran.

    Raises:
        SystemExit: Always, since this is a console entry point.  The status
            says whether the pass completed, not what it found: 0 when every
            named root was walked, whatever mix of documents was read, skipped
            and refused, and 1 when the run could not complete -- no index or no
            root could be resolved, a named root is not a location that can be
            read, the index could not be opened, a root could not be listed, or
            a directory under one could not be, which stops the pass where it
            is found and leaves every root from there on unfinished.
            A tree of files that are not navigation documents is a completed
            pass and exits 0, and exits 0 again on the next pass over the same
            tree, so a scheduled run's status means the same thing every time it
            is read.  Completing a fan-out exits 1 when the event log cannot be
            read, when a named root has no unfinished run, when its run never
            recorded what its listing found, or when its tasks did not account
            for exactly the files that listing found.  A drop exits 0 when the
            tables went and 0 again when the database held none of them, and 1
            when the database could not be opened or read, when it holds tables
            of those names that nothing proves are the index's, when a table
            would not drop, or when whoever was asked said anything but yes.
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

    refusal = _mode_refusal(arguments)
    if refusal is not None:
        MAIN_LOGGER.fatal('%s', refusal)
        sys.exit(1)

    url = get_results_db_url(arguments, DEFAULT_CONFIG)
    if url is None:
        MAIN_LOGGER.fatal(
            'No results index was named. Give one with --results-db, the '
            'environment.results_db configuration variable, or NAV_RESULTS_DB.'
        )
        sys.exit(1)

    if arguments.drop_index:
        # Before the roots are resolved, and instead of them: a drop is about
        # the database alone, and requiring a results root for it would refuse
        # the command on a machine that has the index and not the tree.
        MAIN_LOGGER.info('Starting results index drop')
        MAIN_LOGGER.info('Arguments: %s', masked_command_line(command_list))
        sys.exit(drop_results_index(url, assume_yes=arguments.yes, logger=MAIN_LOGGER))

    try:
        named = arguments.nav_results_roots or [get_nav_results_root(arguments, DEFAULT_CONFIG)]
    except ValueError as exc:
        MAIN_LOGGER.fatal('No navigation results root was named: %s', exc)
        sys.exit(1)

    # Normalized and de-duplicated here rather than in each mode, so that the
    # roots this run reports on are the roots it works over: every later message
    # names the normalized spelling, and a command line naming one root two ways
    # would otherwise open by listing two and then account for one, which reads
    # as a root having gone missing.
    try:
        roots = distinct_roots(named)
    except ValueError as exc:
        MAIN_LOGGER.fatal('A navigation results root is not a location that can be read: %s', exc)
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
    except UnlistableDirectoryError as exc:
        # The one failure a pass stops for rather than charging to a file or a
        # root.  A directory nobody listed holds documents nobody recorded, and
        # absence of a row is what every consumer reads as "this image was
        # never navigated", so the alternative to stopping is a completed pass
        # that answers wrongly and goes on answering wrongly.
        MAIN_LOGGER.fatal(
            'Ingest stopped: %s. This root and any named after it have no completed ingest '
            'run, so no consumer reads absence under them as an answer. Make that directory '
            'readable and run the ingest again.',
            exc,
        )
        status = 1
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

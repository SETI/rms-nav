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
"""

import argparse
import os
import sys

# Make CLI runnable from source tree with
#    python src/package
package_source_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, package_source_path)

from spindoctor.cli.logging_args import add_logging_arguments, reporting_logging_errors
from spindoctor.cli.stats.ingest import IngestCounts, ingest_metadata_files
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
        overrides NAV_RESULTS_DB and the environment.results_db configuration
        variable. The tables are created if they are absent.""",
    )

    ingest_group = cmdparser.add_argument_group('Ingest')
    ingest_group.add_argument(
        '--force',
        action='store_true',
        default=False,
        help="""Read every document, including ones whose recorded size and
        modification time still match the tree.""",
    )

    add_logging_arguments(cmdparser, has_image_logger=False)

    return cmdparser.parse_args(command_list)


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
    if counts.roots_unreadable:
        MAIN_LOGGER.error(
            'Roots that could not be listed and are therefore not ingested: %d',
            counts.roots_unreadable,
        )


def main() -> None:
    """Console entry point for ``sd_stats_ingest``.

    Resolves the index URL and the results roots, walks each root once, and
    writes what it read into the index, reporting the outcome to the main log.

    Raises:
        SystemExit: Always, since this is a console entry point.  The status is
            0 when the pass accounted for at least one document -- read now or
            skipped as unchanged since the last pass -- and 1 when no index or
            no root could be resolved, when the index could not be opened, and
            when the pass accounted for nothing at all, which is what a root
            holding no results looks like.
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

    MAIN_LOGGER.info('Starting results index ingest')
    # Masked, not verbatim: a results root can be a signed cloud URL and the
    # index URL can carry a database password, and a run log is read by whoever
    # is handed one.
    MAIN_LOGGER.info('Roots: %s', ', '.join(masked_url(root) for root in roots))
    MAIN_LOGGER.info('Force: %s', arguments.force)
    MAIN_LOGGER.info('Arguments: %s', [masked_url(value) for value in command_list])

    try:
        engine = open_index(url, create=True)
    except ValueError as exc:
        MAIN_LOGGER.fatal('Cannot open the results index: %s', exc)
        sys.exit(1)
    try:
        counts = ingest_metadata_files(engine, roots, force=arguments.force, logger=MAIN_LOGGER)
    finally:
        engine.dispose()
    _log_outcome(counts)
    # Nothing ingested is a failure only when nothing was skipped either: a
    # second pass over an unchanged tree legitimately reads no document at all.
    sys.exit(0 if counts.files_ingested + counts.files_skipped > 0 else 1)


if __name__ == '__main__':
    main()

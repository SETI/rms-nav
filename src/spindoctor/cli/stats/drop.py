"""Dropping a results index from the command line.

The destructive half of the statistics system.  It opens the database the URL
names, says what of the index is in it, asks whoever typed the command whether
that is what they meant, removes those tables and stops.  It never walks a
results tree: dropping is a deliberate act, not the opening move of a long
ingest, and a command that did both would make a mistyped URL expensive twice
over.

The account of what is about to go is written to the log, which is where
everything else ``sd_stats_ingest`` does is written.  The one thing this module
puts on the terminal itself is the question, which ``input`` writes: a prompt is
a dialogue rather than a report, and carrying it in the log alone would leave a
run whose log was routed to a file waiting silently for an answer.

Two questions this answers by reporting rather than by refusing:

**An ingest run that has not finished does not stop a drop.**  Such a run is
either a pass writing the index at this moment or one that died, and nothing in
the index tells the two apart -- there is no heartbeat and no process to ask.
A pass that died is also the commonest reason to want a drop, so refusing on
that evidence would withhold the command from the case that needs it most, in
order to guard a case the confirmation already guards.  It is therefore counted
and said out loud, before the question is asked, so that the person who is about
to shoot a live ingest is told so while there is still an answer to give.  What
a drop under a live pass costs is that pass, which fails on a table that has
gone and leaves nothing behind: an unfinished run already reads as "not
ingested" to every consumer, so no reader is told anything different before and
after.

**Another process holding the database does not stop a drop either.**  Neither
backend can be asked the question honestly: SQLite's readers take no lock to
observe under write-ahead logging, and a PostgreSQL role need not be allowed to
read the server's activity view.  What can be done is to make the attempt fail
rather than hang, which
:data:`~spindoctor.results_index.drop.DROP_LOCK_TIMEOUT_MS` does, and to leave
nothing half-finished when it does, which the drop order does.  So the database
itself decides, promptly and per table, instead of a guess deciding beforehand.
"""

from pdslogger import PdsLogger
from sqlalchemy.exc import SQLAlchemyError

from spindoctor.results_index import (
    IndexContents,
    drop_index_tables,
    index_contents,
    masked_url,
    open_database,
)

__all__ = ['AGREEMENT', 'drop_results_index']

AGREEMENT = ('y', 'yes')
"""The answers that mean yes.  Anything else, including nothing, means no."""

_PROMPT = 'Drop {tables} table(s) and {rows} row(s) from {url}?{unfinished} [y/N] '
"""What the operator is asked.

The question carries the facts an answer turns on rather than only the verb.
The account above it goes to the log, which is where the whole of what this run
did belongs and which is on the terminal in an ordinary run -- but a run whose
log has been routed to a file still puts this line in front of the person
typing, and a question that read only "are you sure?" would be one they had to
answer out of memory.

``sd_stats_ingest`` carries a main logger and reports through it; this is a
dialogue rather than a report, and the prompt is written by ``input`` itself.
"""

_UNFINISHED = ' {count} ingest run(s) have not finished.'
"""The one fact from the account that is repeated into the question.

A pass may be writing the index at this moment, and the drop would end it.  It
is the only thing here that a person answering could not have known from the
command they typed.
"""


def _summary(safe_url: str, contents: IndexContents) -> list[str]:
    """Return the account of what a drop is about to remove.

    Written for a person deciding whether to answer yes, so it leads with what
    is lost rather than with what is done: the tables by name, how many rows are
    in each of them, and the two facts that change the answer -- a schema
    version that is not the one this code reads, which is the case the drop
    exists for, and an ingest run that nothing has finished.

    Parameters:
        safe_url: The index URL with its credentials already masked.
        contents: What the database holds of the index.

    Returns:
        The lines, in the order they are said.
    """
    lines = [f'About to drop the SpinDoctor results index tables from {safe_url}']
    lines.extend(f'    {table.name}: {table.rows} row(s)' for table in contents.tables)
    lines.append(f'    {len(contents.tables)} table(s), {contents.rows} row(s) in all')
    if contents.schema_version is None:
        lines.append('This database carries no readable schema version stamp.')
    else:
        lines.append(f'This database is stamped with schema version {contents.schema_version}.')
    if contents.unfinished_runs:
        lines.append(
            f'{contents.unfinished_runs} ingest run(s) have begun and not finished. Either a '
            f'pass is writing this index now, which the drop would end, or one died; nothing '
            f'recorded here tells the two apart.'
        )
    lines.append(
        'Nothing else in this database is touched. The metadata documents are the source of '
        'truth, so a dropped index is rebuilt by running sd_stats_ingest again.'
    )
    return lines


def _answer(safe_url: str, contents: IndexContents) -> str | None:
    """Put the question to whoever typed the command.

    Parameters:
        safe_url: The index URL with its credentials already masked.
        contents: What the database holds of the index.

    Returns:
        The line that was typed, or None when there was nobody to type one.
        Three ways of having no standard input are read the same way, because
        the drop's answer to all of them is the same: one at its end, which is
        what a scheduled run redirected from nowhere has; one whose file has
        been closed under it; and none at all, which is what a service manager
        that hands its children no input leaves behind.
    """
    unfinished = (
        _UNFINISHED.format(count=contents.unfinished_runs) if contents.unfinished_runs else ''
    )
    question = _PROMPT.format(
        tables=len(contents.tables), rows=contents.rows, url=safe_url, unfinished=unfinished
    )
    try:
        return input(question)
    except (EOFError, OSError, RuntimeError):
        return None


def drop_results_index(url: str, *, assume_yes: bool, logger: PdsLogger) -> int:
    """Remove every table of a results index, having said what that costs.

    Which index is being dropped is recorded before it is opened, since a run
    log that says only that a database could not be opened leaves out the one
    thing a reader has to check: which of the three resolution levels supplied
    the URL, and which URL it was.

    Parameters:
        url: Connection URL of the index to drop.
        assume_yes: Whether to drop without asking.  A run with nobody at the
            terminal needs this; without it, a standard input that is at its end
            is treated as a refusal rather than as consent.
        logger: Logger the whole account is written to, whether or not anybody
            was asked.

    Returns:
        The exit status: 0 when the tables were dropped and 0 again when there
        were none to drop, since an index that is already gone is the state
        asked for; 1 when the database could not be opened or read, when a
        table would not drop, and when the operator answered anything but yes.
    """
    safe_url = masked_url(url)
    logger.info('Results index to drop the tables of: %s', safe_url)
    try:
        engine = open_database(url)
    except ValueError as exc:
        logger.fatal('Cannot open the database to drop the results index from: %s', exc)
        return 1
    try:
        try:
            contents = index_contents(engine)
        except SQLAlchemyError as exc:
            logger.fatal(
                'Nothing was dropped: %s holds tables of the results index that could not be '
                'read (%s: %s). Check that the account this URL opens may read every table '
                'it is being asked to drop.',
                safe_url,
                type(exc).__name__,
                exc,
            )
            return 1
        if not contents.tables:
            logger.info(
                '%s holds none of the results index tables, so nothing was dropped. An index '
                'that is not there and one that has been dropped are the same thing to every '
                'program that reads one.',
                safe_url,
            )
            return 0
        for line in _summary(safe_url, contents):
            logger.info('%s', line)
        if not assume_yes:
            answer = _answer(safe_url, contents)
            if answer is None:
                logger.fatal(
                    'Nothing was dropped: there is nobody to confirm with, since standard '
                    'input is at its end. Pass --yes to drop without being asked.'
                )
                return 1
            if answer.strip().lower() not in AGREEMENT:
                logger.info('Nothing was dropped: the answer was %r rather than yes.', answer)
                return 1
        try:
            dropped = drop_index_tables(engine)
        except SQLAlchemyError as exc:
            logger.fatal(
                'The drop of %s did not complete (%s: %s). Another session may be holding one '
                'of these tables; nothing else in the database was touched.',
                safe_url,
                type(exc).__name__,
                exc,
            )
            return 1
        logger.info('Dropped from %s: %s', safe_url, ', '.join(dropped))
        logger.info(
            'That index is now what one nobody has ingested into looks like. Run '
            'sd_stats_ingest to build it again.'
        )
    finally:
        engine.dispose()
    return 0

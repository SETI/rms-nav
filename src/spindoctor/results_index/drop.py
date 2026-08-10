"""Removing the results index's own tables, and nothing else.

Starting a results tree over is an ordinary operation -- delete the results,
navigate again, ingest again -- and the index needs a counterpart to it.  The
schema version gate makes that need sharper rather than softer: a version bump
is deliberately not migrated, so emptying the database and ingesting again is
the whole of what the gate leaves an operator to do, and it is what the gate's
own refusal sends them here for.

**A name is not evidence.**  The tables this removes are called ``images``,
``techniques``, ``feature_sources``, ``failed_files``, ``schema_meta`` and
``ingest_runs``, which are among the commonest table names there are, so a
database is never taken to hold a SpinDoctor index because a table of one of
those names is in it.  What proves the index is its own stamp table: a
``schema_meta`` carrying the marks SpinDoctor's stamp carries.  Without that
this reports what it found and removes nothing, which is the same answer the
drop owes a URL naming something that is not an index at all.

**The evidence names the schema too.**  A server resolves an unqualified table
name through a search path that may cross several schemas, and resolving six
names independently is how one drop comes to span two of them -- destroying a
stranger's table in the first while leaving the index's own in the second.  So
the stamp is looked for once, the schema it was found in is the schema every
later statement names explicitly, and a table of one of these names in any other
schema is not this index's and is never touched.

**The transaction is the guarantee, on both backends.**  PostgreSQL rolls DDL
back with everything else.  The SQLite driver opens no transaction of its own
for ``DROP TABLE``, which would leave each drop committed on its own -- so this
opens one itself, with ``BEGIN IMMEDIATE``, and SQLite's own DDL is
transactional inside it.  An interrupted drop therefore leaves the database
exactly as it was on either backend, rather than a state something else has to
be able to read.

**The order is the second line.**  ``schema_meta`` goes first, before the tables
it stamps, so that the one state which must never be reached is the one the
statement order cannot produce: a stamp still standing over tables that have
gone, which the version gate reads as a healthy index and inside which every
consumer's first query fails.
"""

from dataclasses import dataclass

import sqlalchemy
from sqlalchemy.engine import Engine

from spindoctor.results_index.schema import INGEST_RUNS, METADATA, SCHEMA_META, SCHEMA_VERSION
from spindoctor.results_index.scope import (
    carries_the_stamp_marks,
    index_tables_in,
    resolved_schema,
)

__all__ = [
    'DROP_LOCK_TIMEOUT_MS',
    'IndexContents',
    'TableContents',
    'drop_index_tables',
    'index_contents',
    'index_table_names',
]

DROP_LOCK_TIMEOUT_MS = 30000
"""How long the drop waits for a PostgreSQL lock before giving the table up.

``DROP TABLE`` takes a lock no other statement shares, so a transaction that has
so much as read the table holds the drop off for as long as it stays open -- an
ingest in flight, or a session somebody left sitting in ``psql``.  Without a
bound the drop waits for that silently and forever, which reads as a hung
command rather than as a table somebody else is using.

Applied to the reading that precedes the drop as well as to the drop itself.
Counting the rows of a table takes a lock too, and a bound that began only after
the confirmation would leave the command hanging before anybody was asked
anything -- which is the one place a hang is hardest to account for.

Matched to :data:`~spindoctor.results_index.engine.SQLITE_BUSY_TIMEOUT_MS`, so
that a contended drop gives up after the same wait on either backend.  SQLite
needs nothing more: the busy timeout applies to every connection already.
"""

_POSTGRES_DIALECT = 'postgresql'
"""Dialect name of every PostgreSQL driver, whichever one the URL asked for."""

_SQLITE_DIALECT = 'sqlite'
"""Dialect name of the SQLite driver."""


@dataclass(frozen=True)
class TableContents:
    """One table of the index, and how much of it there is to lose.

    Parameters:
        name: The table's name, as the database holds it.
        rows: How many rows it holds.
    """

    name: str
    rows: int


@dataclass(frozen=True)
class IndexContents:
    """What of SpinDoctor's own index a database holds, and where.

    Read before the drop, so that a destructive command can say what it is about
    to destroy, so that a database holding none of it is answered without being
    touched, and so that one holding tables that merely share these names is
    refused rather than emptied.

    Parameters:
        schema: The schema SpinDoctor's own stamp table was found in, which is
            the schema every statement of the drop then names.  None when
            nothing in this database proves it holds a SpinDoctor index, in
            which case nothing is to be dropped from it whatever it is called.
        tables: The index's tables that are present in that schema, in the order
            they would be dropped, each with its row count.  Empty when the
            database holds none of them, and empty whenever ``schema`` is None.
        schema_version: The version the database is stamped with, or None when
            it carries no readable stamp.  It is reported rather than required:
            a database this command is pointed at is quite likely to be one no
            opener would accept.
        unfinished_runs: How many ingest runs have begun and not finished, or
            None when the question cannot be put to this database -- there is no
            run table, or its stamp is not the version whose columns this code
            knows.
        unproven: Names of tables the database holds that the index also uses,
            reported only when ``schema`` is None.  They are what a refusal
            names: either somebody else's tables that happen to be called this,
            or the remains of an index whose stamp has gone, and nothing here
            can tell those apart.
    """

    schema: str | None
    tables: tuple[TableContents, ...]
    schema_version: int | None
    unfinished_runs: int | None
    unproven: tuple[str, ...]

    @property
    def rows(self) -> int:
        """Total rows across every table of the index that is present.

        Returns:
            The sum of the per-table counts.
        """
        return sum(table.rows for table in self.tables)


def _drop_order() -> tuple[sqlalchemy.Table, ...]:
    """Return every table of the index, in the order the drop removes them.

    Dependents come before what they depend on, because a table another one
    references cannot be dropped while that reference stands.  ``schema_meta``
    is lifted to the front of that order: it references nothing and nothing
    references it, so it is free to go first, and going first is what keeps a
    stamp from ever standing over tables that have gone.

    Returns:
        The tables, in drop order.
    """
    dependents_first = reversed(METADATA.sorted_tables)
    return (SCHEMA_META, *(table for table in dependents_first if table is not SCHEMA_META))


def index_table_names() -> tuple[str, ...]:
    """Return the name of every table the index owns.

    The whole of what a drop removes, and the only names it ever utters.

    Returns:
        The table names, in the order the drop removes them.
    """
    return tuple(table.name for table in _drop_order())


def _index_schema(connection: sqlalchemy.Connection) -> str | None:
    """Return the schema this database's own results index lives in, if it has one.

    Parameters:
        connection: An open connection to the database.

    Returns:
        The schema holding a ``schema_meta`` that carries SpinDoctor's own
        marks, or None when this database offers no evidence of holding an
        index of ours.
    """
    schema = resolved_schema(connection)
    if schema is None:
        return None
    return schema if carries_the_stamp_marks(connection, schema) else None


def _present_tables(connection: sqlalchemy.Connection, schema: str) -> tuple[sqlalchemy.Table, ...]:
    """Return the index's tables that this schema actually holds.

    Tables only.  A view of one of these names is not a table this drops -- a
    ``DROP TABLE`` refuses one, and refusing it in the middle of the drop would
    take back the whole of it -- so it is left out of the account rather than
    counted into a drop that could not then finish.

    Parameters:
        connection: An open connection to the database.
        schema: The schema the index's own stamp was found in.

    Returns:
        The tables that are there, in drop order.
    """
    held = set(sqlalchemy.inspect(connection).get_table_names(schema=schema))
    bound = index_tables_in(schema)
    return tuple(bound[name] for name in index_table_names() if name in held)


def _unproven_tables(connection: sqlalchemy.Connection) -> tuple[str, ...]:
    """Return the index's table names that this database holds somewhere reachable.

    Asked only of a database that proved nothing, and only so that the refusal
    can say what it saw.  The names are resolved the way the server resolves
    them, because those are the tables a drop deciding from names alone would
    have destroyed.

    Parameters:
        connection: An open connection to the database.

    Returns:
        The names, in drop order.
    """
    inspector = sqlalchemy.inspect(connection)
    return tuple(name for name in index_table_names() if inspector.has_table(name))


def _row_count(connection: sqlalchemy.Connection, table: sqlalchemy.Table) -> int:
    """Return how many rows a table holds.

    Counted through the table's name alone and none of its columns, so that a
    table belonging to a schema version this code does not read is still counted
    rather than refused.

    Parameters:
        connection: An open connection to the database.
        table: The table to count, named in its schema.

    Returns:
        The row count.
    """
    counted = connection.execute(
        sqlalchemy.select(sqlalchemy.func.count()).select_from(table)
    ).scalar()
    return 0 if counted is None else int(counted)


def _stamp(connection: sqlalchemy.Connection, stamp_table: sqlalchemy.Table) -> int | None:
    """Return the schema version a database is stamped with, if it can be read.

    The stamp is a fact this reports rather than one it requires.  A database
    the drop is pointed at may be malformed in ways no opener would accept --
    a ``schema_meta`` whose version column holds text, or nothing at all -- and
    refusing to say what a database holds because the stamp would not come out
    of it would withhold the drop from one of the cases it exists for.  So every
    way of not being a version is answered the same way, including the ones that
    are not database errors at all.

    The read is taken inside a savepoint, because a statement PostgreSQL refuses
    ends the transaction around it, and everything else this reads would then
    fail with the first failure's shadow rather than with its own answer.

    Parameters:
        connection: An open connection to the database.
        stamp_table: The index's stamp table, named in its schema.

    Returns:
        The stamped version, or None when there is none or it will not be read.
    """
    try:
        with connection.begin_nested():
            row = connection.execute(sqlalchemy.select(stamp_table.c.schema_version)).first()
            return None if row is None else int(row.schema_version)
    except (sqlalchemy.exc.SQLAlchemyError, TypeError, ValueError):
        return None


def _unfinished_runs(
    connection: sqlalchemy.Connection,
    present: tuple[sqlalchemy.Table, ...],
    version: int | None,
    runs_table: sqlalchemy.Table,
) -> int | None:
    """Return how many ingest runs have begun and not finished.

    An unfinished run is either a pass writing this index at this moment or one
    that died, and nothing in the index tells the two apart.  It is therefore
    reported rather than acted on: it is what a person about to drop an index
    someone else may be filling needs to see, and what nothing can conclude from
    on its own.

    Asked only of a database stamped with the version this code reads, because
    the question is phrased in a column, and a column is exactly what a version
    is free to have changed.

    The stamp is not proof of the column all the same, so the count is taken
    inside a savepoint and every failure of it is answered as "the question
    cannot be put to this database".  A stamp says which version wrote the
    database, not that nothing has happened to it since, and a drop that
    refused a database because one column of ``ingest_runs`` was not where the
    stamp implied would refuse the database every other program opens without
    complaint.  The savepoint is what keeps such a failure from ending the
    transaction the rest of this account is read in.

    Parameters:
        connection: An open connection to the database.
        present: The index's tables that this database holds.
        version: The version the database is stamped with, if any.
        runs_table: The index's run table, named in its schema.

    Returns:
        The count, or None when the question cannot be put to this database.
    """
    if version != SCHEMA_VERSION or not any(table.name == INGEST_RUNS.name for table in present):
        return None
    try:
        with connection.begin_nested():
            counted = connection.execute(
                sqlalchemy.select(sqlalchemy.func.count())
                .select_from(runs_table)
                .where(runs_table.c.finished_utc.is_(None))
            ).scalar()
    except (sqlalchemy.exc.SQLAlchemyError, TypeError, ValueError):
        return None
    return 0 if counted is None else int(counted)


def _bound_the_lock_wait(connection: sqlalchemy.Connection) -> None:
    """Stop a statement from waiting forever on a lock somebody else holds.

    ``SET LOCAL`` lasts exactly as long as the transaction it is issued in, so
    the bound applies to the work this transaction carries and to no other
    statement the connection is later handed.

    SQLite is left alone: its busy timeout is applied to every connection when
    the engine is built, and it bounds the same wait for the same reason.

    Parameters:
        connection: The connection the work runs on, inside its transaction.
    """
    if connection.dialect.name != _POSTGRES_DIALECT:
        return
    # Rendered rather than bound: a session setting takes a literal, and no
    # dialect accepts a parameter here.  The value is an integer of this
    # module's own, forced to one so that nothing else can be spelled into it.
    connection.exec_driver_sql(f"SET LOCAL lock_timeout = '{int(DROP_LOCK_TIMEOUT_MS)}ms'")


def _open_one_transaction(connection: sqlalchemy.Connection) -> None:
    """Make everything that follows one transaction, on the SQLite driver too.

    SQLite's own DDL is transactional; its Python driver is what is not.  The
    driver opens a transaction when it sees an INSERT, UPDATE or DELETE and for
    nothing else, so a run of ``DROP TABLE`` statements is issued in autocommit
    and each one stands on its own the moment it returns.  Issuing ``BEGIN``
    here is what puts them inside a transaction that a failure or an interrupt
    takes back whole.

    ``IMMEDIATE`` rather than a plain begin, so the write lock is taken at the
    start: a drop that would have to give way to another writer gives way before
    it has dropped anything, and the busy timeout bounds that wait exactly as
    the lock timeout bounds PostgreSQL's.

    Parameters:
        connection: The connection the drop runs on, at the start of its
            transaction and before any statement of its own.
    """
    if connection.dialect.name != _SQLITE_DIALECT:
        return
    connection.exec_driver_sql('BEGIN IMMEDIATE')


def index_contents(engine: Engine) -> IndexContents:
    """Return what of SpinDoctor's own index a database holds, and where.

    Parameters:
        engine: An open database, from
            :func:`~spindoctor.results_index.engine.open_database`, since one
            holding a version no opener accepts is a database this is asked
            about rather than one it is not.

    Returns:
        The schema the index's own stamp proves it lives in, the tables of that
        schema that are present with their row counts, the stamp, and how many
        ingest runs are outstanding -- or, for a database that proves nothing,
        a report naming whatever tables of these names it does hold.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: If a table that is there cannot be
            counted.  A table this account may not read is one it is unlikely to
            be able to drop, and finding that out before anything is dropped is
            what keeps a refusal from becoming a half-finished drop.
    """
    # A connection rather than a transaction that commits: this reads and writes
    # nothing, and what it leaves behind on the server is a transaction rolled
    # back rather than one committed.  The lock bound still applies, since the
    # first statement opens the transaction ``SET LOCAL`` is scoped to.
    with engine.connect() as connection:
        _bound_the_lock_wait(connection)
        schema = _index_schema(connection)
        if schema is None:
            return IndexContents(
                schema=None,
                tables=(),
                schema_version=None,
                unfinished_runs=None,
                unproven=_unproven_tables(connection),
            )
        bound = index_tables_in(schema)
        present = _present_tables(connection, schema)
        version = _stamp(connection, bound[SCHEMA_META.name])
        tables = tuple(
            TableContents(name=table.name, rows=_row_count(connection, table)) for table in present
        )
        unfinished = _unfinished_runs(connection, present, version, bound[INGEST_RUNS.name])
    return IndexContents(
        schema=schema,
        tables=tables,
        schema_version=version,
        unfinished_runs=unfinished,
        unproven=(),
    )


def drop_index_tables(engine: Engine, contents: IndexContents) -> tuple[str, ...]:
    """Remove the tables a reading of this database found, and nothing else.

    The reading is passed in rather than taken again, so that the tables which
    go are the tables somebody was shown and agreed to.  Between one reading and
    another the answer is free to change -- a creating open in another process
    puts them back -- and a destructive command must not act on a list nobody
    saw.

    Idempotent: contents holding no table of the index are not written at all,
    and the empty answer is what says so.

    Parameters:
        engine: An open database, from
            :func:`~spindoctor.results_index.engine.open_database`.
        contents: What :func:`index_contents` read from that database, naming
            the schema to drop from and the tables to drop.

    Returns:
        The names of the tables that were dropped, in the order they went, and
        an empty tuple when there was nothing of the index to drop.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: If a table cannot be dropped, including
            because another session holds a lock on it for longer than
            :data:`DROP_LOCK_TIMEOUT_MS`.  The transaction rolls back on both
            backends and the database is left exactly as it was.
    """
    if contents.schema is None or not contents.tables:
        return ()
    bound = index_tables_in(contents.schema)
    going = tuple(bound[table.name] for table in contents.tables)
    with engine.begin() as connection:
        _open_one_transaction(connection)
        _bound_the_lock_wait(connection)
        for table in going:
            connection.execute(sqlalchemy.schema.DropTable(table))
    return tuple(table.name for table in going)

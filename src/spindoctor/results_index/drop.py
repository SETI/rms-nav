"""Removing the results index's own tables, and nothing else.

Starting a results tree over is an ordinary operation -- delete the results,
navigate again, ingest again -- and the index needs a counterpart to it.  The
schema version gate makes that need sharper rather than softer: a version bump
is deliberately not migrated, so emptying the database and ingesting again is
the whole of what the gate leaves an operator to do, and it is what the gate's
own refusal sends them here for.

**Only the tables named here are touched.**  They are the tables of
:data:`~spindoctor.results_index.schema.METADATA`, taken by name.  No schema is
dropped, no pattern is matched and nothing is discovered by looking at what the
database holds: a PostgreSQL server is routinely shared, and an index living in
a database beside somebody else's tables must be removable without a thought
about theirs.

**What is left behind is a database, not a hole.**  Dropping every table of an
index leaves what a database that was never ingested into looks like -- on
PostgreSQL literally so -- which every consumer already reads as "not
ingested", and which the next opener with ``create`` rebuilds.

**The order is the guarantee.**  ``schema_meta`` goes first, before the tables
it stamps.  Every state an interrupted drop can leave is then one with no stamp,
which the version gate reads as "this is not a results index"; the state that
must never be left is the opposite one, a stamp still standing over tables that
have gone, because the gate reads that as a healthy index and every consumer
then fails inside its first query.  The order matters because the transaction
around the drop is not equally strong on both backends: PostgreSQL rolls DDL
back with everything else, while the SQLite driver commits each ``DROP TABLE``
outside the surrounding transaction.
"""

from dataclasses import dataclass

import sqlalchemy
from sqlalchemy.engine import Engine

from spindoctor.results_index.engine import stamped_version
from spindoctor.results_index.schema import INGEST_RUNS, METADATA, SCHEMA_META, SCHEMA_VERSION

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

Matched to :data:`~spindoctor.results_index.engine.SQLITE_BUSY_TIMEOUT_MS`, so
that a contended drop gives up after the same wait on either backend.  SQLite
needs nothing more: the busy timeout applies to every connection already.
"""

_POSTGRES_DIALECT = 'postgresql'
"""Dialect name of every PostgreSQL driver, whichever one the URL asked for."""


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
    """What of SpinDoctor's own index a database holds.

    Read before the drop, so that a destructive command can say what it is about
    to destroy, and so that a database holding none of it is answered without
    being touched.

    Parameters:
        tables: The index's tables that are present, in the order they would be
            dropped, each with its row count.  Empty when the database holds
            none of them.
        schema_version: The version the database is stamped with, or None when
            it carries no readable stamp.  It is reported rather than required:
            a database this command is pointed at is quite likely to be one no
            opener would accept.
        unfinished_runs: How many ingest runs have begun and not finished, or
            None when the question cannot be put to this database -- there is no
            run table, or its stamp is not the version whose columns this code
            knows.
    """

    tables: tuple[TableContents, ...]
    schema_version: int | None
    unfinished_runs: int | None

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
    references it, so it is free to go first, and going first is what makes
    every interruption leave a database with no stamp rather than a stamp with
    no tables.

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


def _present_tables(engine: Engine) -> tuple[sqlalchemy.Table, ...]:
    """Return the index's tables that this database actually holds.

    Parameters:
        engine: The open database.

    Returns:
        The tables that are there, in drop order.
    """
    inspector = sqlalchemy.inspect(engine)
    return tuple(table for table in _drop_order() if inspector.has_table(table.name))


def _row_count(connection: sqlalchemy.Connection, table: sqlalchemy.Table) -> int:
    """Return how many rows a table holds.

    Counted through the table's name alone and none of its columns, so that a
    table belonging to a schema version this code does not read is still counted
    rather than refused.

    Parameters:
        connection: An open connection to the database.
        table: The table to count.

    Returns:
        The row count.
    """
    counted = connection.execute(
        sqlalchemy.select(sqlalchemy.func.count()).select_from(table)
    ).scalar()
    return 0 if counted is None else int(counted)


def _stamp(engine: Engine) -> int | None:
    """Return the schema version a database is stamped with, if it can be read.

    The stamp is a fact this reports rather than one it requires.  A database
    the drop is pointed at may be malformed in ways no opener would accept --
    a ``schema_meta`` left over from a schema whose columns were different -- and
    refusing to say what a database holds because the stamp would not come out
    of it would withhold the drop from one of the cases it exists for.

    Parameters:
        engine: The open database.

    Returns:
        The stamped version, or None when there is none or it will not be read.
    """
    try:
        return stamped_version(engine)
    except sqlalchemy.exc.SQLAlchemyError:
        return None


def _unfinished_runs(
    connection: sqlalchemy.Connection, present: tuple[sqlalchemy.Table, ...], version: int | None
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

    Parameters:
        connection: An open connection to the database.
        present: The index's tables that this database holds.
        version: The version the database is stamped with, if any.

    Returns:
        The count, or None when the question cannot be put to this database.
    """
    if version != SCHEMA_VERSION or INGEST_RUNS not in present:
        return None
    counted = connection.execute(
        sqlalchemy.select(sqlalchemy.func.count())
        .select_from(INGEST_RUNS)
        .where(INGEST_RUNS.c.finished_utc.is_(None))
    ).scalar()
    return 0 if counted is None else int(counted)


def index_contents(engine: Engine) -> IndexContents:
    """Return what of SpinDoctor's own index a database holds.

    Parameters:
        engine: An open database, from
            :func:`~spindoctor.results_index.engine.open_database`, since one
            holding a version no opener accepts is a database this is asked
            about rather than one it is not.

    Returns:
        The tables that are present with their row counts, the stamp, and how
        many ingest runs are outstanding.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: If a table that is there cannot be
            counted.  A table this account may not read is one it is unlikely to
            be able to drop, and finding that out before anything is dropped is
            what keeps a refusal from becoming a half-finished drop.
    """
    present = _present_tables(engine)
    version = _stamp(engine)
    with engine.connect() as connection:
        tables = tuple(
            TableContents(name=table.name, rows=_row_count(connection, table)) for table in present
        )
        unfinished = _unfinished_runs(connection, present, version)
    return IndexContents(tables=tables, schema_version=version, unfinished_runs=unfinished)


def _bound_the_lock_wait(connection: sqlalchemy.Connection) -> None:
    """Stop the drop from waiting forever on a lock somebody else holds.

    ``SET LOCAL`` lasts exactly as long as the transaction it is issued in, so
    the bound applies to this drop and to no other statement the connection
    later carries.

    SQLite is left alone: its busy timeout is applied to every connection when
    the engine is built, and it bounds the same wait for the same reason.

    Parameters:
        connection: The connection the drop runs on, inside its transaction.
    """
    if connection.dialect.name != _POSTGRES_DIALECT:
        return
    # Rendered rather than bound: a session setting takes a literal, and no
    # dialect accepts a parameter here.  The value is an integer of this
    # module's own, forced to one so that nothing else can be spelled into it.
    connection.exec_driver_sql(f"SET LOCAL lock_timeout = '{int(DROP_LOCK_TIMEOUT_MS)}ms'")


def drop_index_tables(engine: Engine) -> tuple[str, ...]:
    """Remove every table of the results index from a database, and nothing else.

    Idempotent: a database holding none of these tables is not written at all,
    and the empty answer is what says so.

    Parameters:
        engine: An open database, from
            :func:`~spindoctor.results_index.engine.open_database`.

    Returns:
        The names of the tables that were dropped, in the order they went, and
        an empty tuple when there was nothing of the index there.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: If a table cannot be dropped, including
            because another session holds a lock on it for longer than
            :data:`DROP_LOCK_TIMEOUT_MS`.  On PostgreSQL the transaction rolls
            back and the database is left exactly as it was; on SQLite the drops
            already committed stand, which the order they are issued in is what
            makes safe.
    """
    present = _present_tables(engine)
    if not present:
        return ()
    with engine.begin() as connection:
        _bound_the_lock_wait(connection)
        for table in present:
            connection.execute(sqlalchemy.schema.DropTable(table))
    return tuple(table.name for table in present)

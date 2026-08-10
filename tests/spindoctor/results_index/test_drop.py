"""Removing the results index's own tables, and nothing else.

Four things are asked here.  That the drop removes exactly the tables the schema
declares and no other object of the database.  That it removes them only from a
database that proved it holds an index of SpinDoctor's, since ``images`` is a
name anybody may use and a drop deciding from names alone destroys strangers'
data.  That it works on the databases no opener will accept, since a version the
gate refuses is the case the drop exists for.  And that an interruption leaves
the database exactly as it was, on the backend whose driver would otherwise
commit each ``DROP TABLE`` on its own.
"""

from pathlib import Path
from typing import Any

import pytest
import sqlalchemy
from tests.spindoctor.results_index.conftest import image_row, opened, sqlite_url_for

from spindoctor.results_index import (
    IMAGES,
    INGEST_RUNS,
    METADATA,
    SCHEMA_META,
    SCHEMA_VERSION,
    IndexContents,
    drop_index_tables,
    index_contents,
    index_table_names,
    open_database,
    open_index,
)

FOREIGN_TABLE = 'somebody_elses_table'
"""A table of the same database that SpinDoctor did not create.

A PostgreSQL server is routinely shared, and a SQLite file is free to hold
anything, so the drop's promise is about the tables it names rather than about
the database around them.
"""

COLLIDING_TABLE = IMAGES.name
"""A table SpinDoctor did not create, under a name SpinDoctor also uses.

The harder half of the same promise, and the one a foreign table called
``somebody_elses_table`` cannot ask: our names are among the commonest there
are, so a database is free to hold one of them for a reason of its own.
"""

SQLITE_SCHEMA = 'main'
"""The one namespace a SQLite database has, which the drop names explicitly."""


def _built(path: Path) -> str:
    """Create an index at a path and return its URL.

    Parameters:
        path: The database file's path.

    Returns:
        The URL of the created index.
    """
    url = sqlite_url_for(path)
    with opened(url, create=True):
        pass
    return url


def _table_names(url: str) -> list[str]:
    """Return every table a database holds, whoever created it.

    Parameters:
        url: The database URL.

    Returns:
        The table names, sorted.
    """
    engine = sqlalchemy.create_engine(url)
    try:
        return sorted(sqlalchemy.inspect(engine).get_table_names())
    finally:
        engine.dispose()


def _contents(url: str) -> IndexContents:
    """Open a database and read what of the index it holds.

    Parameters:
        url: The database URL.

    Returns:
        The contents.
    """
    engine = open_database(url)
    try:
        return index_contents(engine)
    finally:
        engine.dispose()


def _dropped(url: str) -> tuple[str, ...]:
    """Open a database, drop the index tables from it, and close it.

    Parameters:
        url: The database URL.

    Returns:
        The names of the tables that were dropped.
    """
    engine = open_database(url)
    try:
        return drop_index_tables(engine, index_contents(engine))
    finally:
        engine.dispose()


def _execute(url: str, *statements: str) -> None:
    """Run statements against a database as somebody other than the index.

    Parameters:
        url: The database URL.
        statements: The statements to run, in order.
    """
    engine = sqlalchemy.create_engine(url)
    try:
        with engine.begin() as connection:
            for statement in statements:
                connection.exec_driver_sql(statement)
    finally:
        engine.dispose()


def _add_foreign_table(url: str) -> None:
    """Create a table in the database that the index does not own.

    Parameters:
        url: The database URL.
    """
    _execute(url, f'CREATE TABLE {FOREIGN_TABLE} (x INTEGER)')


def _rows_of(url: str, table: str) -> int:
    """Return how many rows a table of a database holds.

    Parameters:
        url: The database URL.
        table: The table's name.

    Returns:
        The row count.
    """
    engine = sqlalchemy.create_engine(url)
    try:
        with engine.connect() as connection:
            counted: Any = connection.exec_driver_sql(f'SELECT count(*) FROM {table}').scalar()
    finally:
        engine.dispose()
    return int(counted)


# ---------------------------------------------------------------------------
# The order the tables go in
# ---------------------------------------------------------------------------


def test_the_drop_covers_every_table_the_schema_declares() -> None:
    """The names come from the metadata, so a table added to it is dropped too.

    A hand-written list would be right on the day it was written and wrong on
    the day the next table was added, and the table it forgot is the one left
    standing over a database that reads as empty.
    """
    assert sorted(index_table_names()) == sorted(METADATA.tables)


def test_the_stamp_is_dropped_before_anything_it_stamps() -> None:
    """The one state that must never be reached is the one the order forbids.

    A stamp still standing over tables that have gone is read by the version
    gate as a healthy index of this version, and every consumer then fails
    inside its first query instead of being told to ingest.
    """
    assert index_table_names()[0] == SCHEMA_META.name


@pytest.mark.parametrize('child', ['techniques', 'feature_sources'])
def test_a_child_table_goes_before_the_table_it_references(child: str) -> None:
    """A table another one references cannot be dropped while that reference stands.

    Parameters:
        child: Name of a table carrying a foreign key into ``images``.
    """
    order = index_table_names()
    assert order.index(child) < order.index(IMAGES.name)


# ---------------------------------------------------------------------------
# What a drop removes
# ---------------------------------------------------------------------------


def test_the_drop_removes_every_table_of_the_index(tmp_path: Path) -> None:
    """The whole schema goes, not the part a query happened to name.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    _dropped(url)
    assert _table_names(url) == []


def test_the_drop_names_what_it_removed(tmp_path: Path) -> None:
    """An operator reads the answer, so the answer is the table names.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    assert _dropped(url) == index_table_names()


def test_a_table_the_index_does_not_own_survives_the_drop(tmp_path: Path) -> None:
    """A shared database is the case this promise is about.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    _add_foreign_table(url)
    _dropped(url)
    assert _table_names(url) == [FOREIGN_TABLE]


def _statements_of_a_drop(url: str) -> list[str]:
    """Return every statement a drop issued, in the order it issued them.

    Parameters:
        url: The database URL.

    Returns:
        The statements, stripped of surrounding space.
    """
    statements: list[str] = []
    engine = open_database(url)
    contents = index_contents(engine)
    sqlalchemy.event.listen(
        engine,
        'before_cursor_execute',
        lambda conn, cursor, statement, *rest: statements.append(statement.strip()),
    )
    try:
        drop_index_tables(engine, contents)
    finally:
        engine.dispose()
    return statements


def test_the_drop_issues_one_drop_table_per_index_table(tmp_path: Path) -> None:
    """Read off the statements themselves, rather than off what survived.

    A database this test can see the whole of would pass on its contents alone
    even if the drop had reached for the schema around them; what says it did
    not is that the only destructive statements it issued were these six, each
    naming one table by name.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    issued = _statements_of_a_drop(url)
    destructive = [statement for statement in issued if statement.upper().startswith('DROP')]
    assert destructive == [f'DROP TABLE {SQLITE_SCHEMA}.{name}' for name in index_table_names()]


def test_every_statement_of_the_drop_names_its_schema(tmp_path: Path) -> None:
    """A bare name is resolved by the server, and a search path may cross schemas.

    Naming the schema is what keeps six names from resolving into two of them,
    which is how a drop comes to destroy one database's table while leaving the
    index's own standing.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    destructive = [
        statement
        for statement in _statements_of_a_drop(url)
        if statement.upper().startswith('DROP')
    ]
    assert all(f' {SQLITE_SCHEMA}.' in statement for statement in destructive)


def test_the_drop_never_utters_a_table_it_does_not_own(tmp_path: Path) -> None:
    """Including in the statements it issues to find out what is there.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    _add_foreign_table(url)
    named = [statement for statement in _statements_of_a_drop(url) if FOREIGN_TABLE in statement]
    assert named == []


def test_a_second_drop_removes_nothing(tmp_path: Path) -> None:
    """An idempotent drop has to be visibly idempotent, not silently so.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    _dropped(url)
    assert _dropped(url) == ()


def test_a_second_drop_leaves_a_foreign_table_alone(tmp_path: Path) -> None:
    """A database holding none of the index keeps whatever else is in it.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    _add_foreign_table(url)
    _dropped(url)
    _dropped(url)
    assert _table_names(url) == [FOREIGN_TABLE]


def test_a_database_holding_none_of_the_index_is_not_written_at_all(tmp_path: Path) -> None:
    """Not "nothing was removed" but "no statement was issued".

    A drop that opened a transaction over an empty list would satisfy every
    assertion about what survived, so what is asserted here is that the drop
    said nothing to the database at all.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = sqlite_url_for(tmp_path / 'index.sqlite3')
    _add_foreign_table(url)
    assert _statements_of_a_drop(url) == []


def test_the_tables_dropped_are_the_tables_that_were_read(tmp_path: Path) -> None:
    """A destructive command must not act on a list nobody was shown.

    Between the reading an operator answers and the drop that follows, another
    process is free to put a table back.  What goes is what was read.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    engine = open_database(url)
    try:
        contents = index_contents(engine)
        _execute(url, f'DROP TABLE {INGEST_RUNS.name}', 'CREATE TABLE ingest_runs_copy (x INTEGER)')
        dropped = drop_index_tables(
            engine,
            IndexContents(
                schema=contents.schema,
                tables=tuple(table for table in contents.tables if table.name != INGEST_RUNS.name),
                schema_version=contents.schema_version,
                unfinished_runs=contents.unfinished_runs,
                unproven=(),
            ),
        )
    finally:
        engine.dispose()
    assert INGEST_RUNS.name not in dropped


# ---------------------------------------------------------------------------
# What proves a database holds an index of ours
# ---------------------------------------------------------------------------


def test_a_table_sharing_one_of_our_names_is_not_evidence_of_an_index(tmp_path: Path) -> None:
    """``images`` is a name anybody may use, and using it is not consent.

    Parameters:
        tmp_path: Directory the database file is written into.
    """
    url = sqlite_url_for(tmp_path / 'other.sqlite3')
    _execute(url, f'CREATE TABLE {COLLIDING_TABLE} (id INTEGER PRIMARY KEY, caption TEXT)')
    assert _contents(url).schema is None


def test_such_a_table_is_named_in_what_the_reading_reports(tmp_path: Path) -> None:
    """So that a refusal can say what it saw rather than only that it refused.

    Parameters:
        tmp_path: Directory the database file is written into.
    """
    url = sqlite_url_for(tmp_path / 'other.sqlite3')
    _execute(url, f'CREATE TABLE {COLLIDING_TABLE} (id INTEGER PRIMARY KEY, caption TEXT)')
    assert _contents(url).unproven == (COLLIDING_TABLE,)


def test_such_a_table_is_not_listed_as_one_of_the_index_tables(tmp_path: Path) -> None:
    """Listing it would be an account of somebody else's rows as ours.

    Parameters:
        tmp_path: Directory the database file is written into.
    """
    url = sqlite_url_for(tmp_path / 'other.sqlite3')
    _execute(url, f'CREATE TABLE {COLLIDING_TABLE} (id INTEGER PRIMARY KEY, caption TEXT)')
    assert _contents(url).tables == ()


def test_such_a_table_is_not_dropped(tmp_path: Path) -> None:
    """The whole of the finding: a stranger's rows are not ours to remove.

    Parameters:
        tmp_path: Directory the database file is written into.
    """
    url = sqlite_url_for(tmp_path / 'other.sqlite3')
    _execute(
        url,
        f'CREATE TABLE {COLLIDING_TABLE} (id INTEGER PRIMARY KEY, caption TEXT)',
        f"INSERT INTO {COLLIDING_TABLE} (caption) VALUES ('somebody elses cat'), ('their dog')",
    )
    _dropped(url)
    assert _rows_of(url, COLLIDING_TABLE) == 2


def test_a_database_of_our_names_with_no_stamp_is_not_written_at_all(tmp_path: Path) -> None:
    """Refused before a transaction is opened, not inside one that drops nothing.

    Parameters:
        tmp_path: Directory the database file is written into.
    """
    url = sqlite_url_for(tmp_path / 'other.sqlite3')
    _execute(url, f'CREATE TABLE {COLLIDING_TABLE} (id INTEGER PRIMARY KEY, caption TEXT)')
    assert _statements_of_a_drop(url) == []


def test_a_stamp_table_without_the_marks_of_ours_is_not_evidence(tmp_path: Path) -> None:
    """A table called ``schema_meta`` is not therefore SpinDoctor's.

    Parameters:
        tmp_path: Directory the database file is written into.
    """
    url = sqlite_url_for(tmp_path / 'other.sqlite3')
    _execute(
        url,
        f'CREATE TABLE {SCHEMA_META.name} (version_num TEXT)',
        f'CREATE TABLE {COLLIDING_TABLE} (id INTEGER)',
    )
    assert _contents(url).schema is None


def test_a_stamp_carrying_the_marks_of_ours_is_evidence(tmp_path: Path) -> None:
    """The pair of columns is the mark, not the whole column set of this version.

    Parameters:
        tmp_path: Directory the database file is written into.
    """
    url = sqlite_url_for(tmp_path / 'other.sqlite3')
    _execute(
        url,
        f'CREATE TABLE {SCHEMA_META.name} '
        f'(singleton INTEGER PRIMARY KEY, schema_version INTEGER, whatever TEXT)',
    )
    assert _contents(url).schema == SQLITE_SCHEMA


def test_a_database_holding_part_of_a_schema_with_its_stamp_can_be_dropped(
    tmp_path: Path,
) -> None:
    """An interrupted creation leaves one, and no opener will touch it.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = sqlite_url_for(tmp_path / 'index.sqlite3')
    engine = sqlalchemy.create_engine(url)
    try:
        SCHEMA_META.create(engine)
        IMAGES.create(engine)
    finally:
        engine.dispose()
    assert _dropped(url) == (SCHEMA_META.name, IMAGES.name)


def test_a_database_holding_part_of_a_schema_without_its_stamp_is_left_alone(
    tmp_path: Path,
) -> None:
    """Because nothing tells that state from a database of somebody else's.

    The cost is named rather than hidden: such a database is one the drop
    declines, and its tables are removed by hand or with the file.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = sqlite_url_for(tmp_path / 'index.sqlite3')
    engine = sqlalchemy.create_engine(url)
    try:
        IMAGES.create(engine)
    finally:
        engine.dispose()
    assert _dropped(url) == ()


# ---------------------------------------------------------------------------
# The databases no opener accepts
# ---------------------------------------------------------------------------


def test_an_index_stamped_with_another_version_can_be_dropped(tmp_path: Path) -> None:
    """The case the drop exists for: the gate's own remedy, made reachable.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    with opened(url) as engine, engine.begin() as connection:
        connection.execute(SCHEMA_META.update().values(schema_version=SCHEMA_VERSION + 1))
    _dropped(url)
    assert _table_names(url) == []


def test_the_contents_report_a_version_this_code_does_not_read(tmp_path: Path) -> None:
    """What is about to go is named by the version it was written under.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    with opened(url) as engine, engine.begin() as connection:
        connection.execute(SCHEMA_META.update().values(schema_version=SCHEMA_VERSION + 1))
    assert _contents(url).schema_version == SCHEMA_VERSION + 1


def test_a_stamp_whose_columns_are_not_this_version_does_not_stop_the_drop(
    tmp_path: Path,
) -> None:
    """The stamp is a fact reported, not a fact required.

    A ``schema_meta`` left over from a schema whose columns were different is
    exactly the shape an old index has, and refusing to say what a database
    holds because its stamp would not come out is withholding the drop from one
    of the cases it is for.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = sqlite_url_for(tmp_path / 'index.sqlite3')
    _execute(
        url,
        f'CREATE TABLE {SCHEMA_META.name} '
        f'(singleton INTEGER PRIMARY KEY, schema_version INTEGER, something_else INTEGER)',
    )
    assert _contents(url).schema_version is None


@pytest.mark.parametrize(
    'stamp',
    ["'v6-beta'", 'NULL'],
    ids=['a-version-that-is-text', 'a-version-that-is-null'],
)
def test_a_stamp_that_is_not_a_version_is_reported_as_none(stamp: str, tmp_path: Path) -> None:
    """Not every way of failing to be an integer is a database error.

    Text where a number belongs, and nothing at all where a number belongs, are
    both refusals the value raises rather than the database, and a destructive
    command that let one of them out would end in a traceback.

    Parameters:
        stamp: The value the stamp column is given.
        tmp_path: Directory the index file is written into.
    """
    url = sqlite_url_for(tmp_path / 'index.sqlite3')
    _execute(
        url,
        f'CREATE TABLE {SCHEMA_META.name} '
        f'(singleton INTEGER PRIMARY KEY, schema_version TEXT, created_utc TEXT)',
        f'INSERT INTO {SCHEMA_META.name} VALUES (1, {stamp}, NULL)',
        f'CREATE TABLE {COLLIDING_TABLE} (x INTEGER)',
    )
    assert _contents(url).schema_version is None


@pytest.mark.parametrize(
    'stamp',
    ["'v6-beta'", 'NULL'],
    ids=['a-version-that-is-text', 'a-version-that-is-null'],
)
def test_a_stamp_that_is_not_a_version_does_not_stop_the_drop(stamp: str, tmp_path: Path) -> None:
    """It is a malformed index, which is one of the states the drop is for.

    Parameters:
        stamp: The value the stamp column is given.
        tmp_path: Directory the index file is written into.
    """
    url = sqlite_url_for(tmp_path / 'index.sqlite3')
    _execute(
        url,
        f'CREATE TABLE {SCHEMA_META.name} '
        f'(singleton INTEGER PRIMARY KEY, schema_version TEXT, created_utc TEXT)',
        f'INSERT INTO {SCHEMA_META.name} VALUES (1, {stamp}, NULL)',
        f'CREATE TABLE {COLLIDING_TABLE} (x INTEGER)',
    )
    _dropped(url)
    assert _table_names(url) == []


# ---------------------------------------------------------------------------
# What an interruption leaves
# ---------------------------------------------------------------------------


def _drop_interrupted_at(url: str, statements_first: int) -> None:
    """Run a drop and raise ``KeyboardInterrupt`` part of the way through it.

    Ctrl-C at the terminal, reproduced where it lands: between two of the
    statements the drop issues, with no source of the drop's own modified.

    Parameters:
        url: The database URL.
        statements_first: How many statements to let through before the
            interrupt.
    """
    seen = 0

    def interrupt_after_a_few(*args: Any) -> None:
        nonlocal seen
        seen += 1
        if seen > statements_first:
            raise KeyboardInterrupt

    engine = open_database(url)
    contents = index_contents(engine)
    sqlalchemy.event.listen(engine, 'before_cursor_execute', interrupt_after_a_few)
    try:
        with pytest.raises(KeyboardInterrupt):
            drop_index_tables(engine, contents)
    finally:
        engine.dispose()


def test_an_interrupted_drop_leaves_every_table(tmp_path: Path) -> None:
    """SQLite's own DDL is transactional; its driver is what does not use it.

    The driver opens a transaction for INSERT, UPDATE and DELETE and for nothing
    else, so a run of ``DROP TABLE`` statements would stand one at a time.  The
    drop opens one itself, and this is what says so.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    _drop_interrupted_at(url, statements_first=3)
    assert _table_names(url) == sorted(METADATA.tables)


def test_an_interrupted_drop_leaves_the_rows_that_were_in_them(tmp_path: Path) -> None:
    """Rows in a table nothing dropped, rather than rows in a rebuilt index.

    An ``images`` left standing by a half-finished drop is the state that is
    never repaired: the next creating open stamps it, and the incremental skip
    then passes over every document whose size and modification time it still
    records.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    with opened(url) as engine, engine.begin() as connection:
        connection.execute(IMAGES.insert(), image_row())
    _drop_interrupted_at(url, statements_first=3)
    assert _rows_of(url, IMAGES.name) == 1


def test_an_interrupted_drop_leaves_the_index_openable(tmp_path: Path) -> None:
    """Which is the whole of "no partially-dropped state": a consumer still reads it.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    _drop_interrupted_at(url, statements_first=3)
    with opened(url) as engine, engine.connect() as connection:
        stamped = connection.execute(sqlalchemy.select(SCHEMA_META.c.schema_version)).scalar()
    assert stamped == SCHEMA_VERSION


def _add_a_reference_into_images(url: str) -> None:
    """Give an outside table a foreign key into ``images``, and a row using it.

    A reference from outside the index is what makes ``DROP TABLE images`` fail
    with foreign keys enforced, which is a database error rather than an
    interrupt and reaches the same place.

    Parameters:
        url: The database URL.
    """
    _execute(
        url,
        'CREATE TABLE their_notes (root_url TEXT, results_path_stub TEXT, note TEXT, '
        'FOREIGN KEY (root_url, results_path_stub) '
        'REFERENCES images (root_url, results_path_stub))',
    )
    row = image_row()
    _execute(
        url,
        f"INSERT INTO their_notes VALUES ('{row['root_url']}', "
        f"'{row['results_path_stub']}', 'mine')",
    )


def test_a_drop_a_reference_refuses_leaves_every_table(tmp_path: Path) -> None:
    """The same guarantee reached by a database error rather than by a key press.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    with opened(url) as engine, engine.begin() as connection:
        connection.execute(IMAGES.insert(), image_row())
    _add_a_reference_into_images(url)
    with pytest.raises(sqlalchemy.exc.SQLAlchemyError):
        _dropped(url)
    assert _table_names(url) == sorted([*METADATA.tables, 'their_notes'])


def test_a_drop_a_reference_refuses_leaves_the_index_openable(tmp_path: Path) -> None:
    """The stamp goes first, so a drop that stood would have unstamped it.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    with opened(url) as engine, engine.begin() as connection:
        connection.execute(IMAGES.insert(), image_row())
    _add_a_reference_into_images(url)
    with pytest.raises(sqlalchemy.exc.SQLAlchemyError):
        _dropped(url)
    with opened(url) as engine, engine.connect() as connection:
        stamped = connection.execute(sqlalchemy.select(SCHEMA_META.c.schema_version)).scalar()
    assert stamped == SCHEMA_VERSION


def test_the_drop_opens_its_transaction_before_it_drops_anything(tmp_path: Path) -> None:
    """The statement that makes the rest of them one thing has to come first.

    ``IMMEDIATE`` rather than a plain begin, so that a drop which has to give way
    to another writer gives way before it has dropped anything, bounded by the
    busy timeout every connection carries.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    assert _statements_of_a_drop(url)[0] == 'BEGIN IMMEDIATE'


# ---------------------------------------------------------------------------
# What the drop leaves behind
# ---------------------------------------------------------------------------


def test_a_database_with_no_stamp_reads_as_one_nobody_ingested(tmp_path: Path) -> None:
    """The state a dropped stamp leaves, put to the gate that reads it.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    with opened(url) as engine, engine.begin() as connection:
        connection.execute(sqlalchemy.schema.DropTable(SCHEMA_META))
    with pytest.raises(ValueError, match='this is not a results index'):
        open_index(url)


def test_a_dropped_index_is_rebuilt_by_the_next_creating_open(tmp_path: Path) -> None:
    """Left usable, rather than left absent: the next ingest builds it again.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    _dropped(url)
    with opened(url, create=True):
        pass
    assert _table_names(url) == sorted(METADATA.tables)


def test_a_rebuilt_index_carries_this_version(tmp_path: Path) -> None:
    """And is stamped, so the gate lets a consumer in again.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    _dropped(url)
    with opened(url, create=True) as engine, engine.connect() as connection:
        stamped = connection.execute(sqlalchemy.select(SCHEMA_META.c.schema_version)).scalar()
    assert stamped == SCHEMA_VERSION


def test_a_rebuilt_index_holds_none_of_the_dropped_rows(tmp_path: Path) -> None:
    """Starting from scratch is the point, so nothing may survive the rebuild.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    with opened(url) as engine, engine.begin() as connection:
        connection.execute(IMAGES.insert(), image_row())
    _dropped(url)
    with opened(url, create=True) as engine, engine.connect() as connection:
        remaining = connection.execute(
            sqlalchemy.select(sqlalchemy.func.count()).select_from(IMAGES)
        ).scalar()
    assert remaining == 0


# ---------------------------------------------------------------------------
# What a drop says it is about to remove
# ---------------------------------------------------------------------------


def test_the_contents_name_the_schema_the_tables_are_in(tmp_path: Path) -> None:
    """Named rather than implied, since it is what every statement then carries.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    assert _contents(url).schema == SQLITE_SCHEMA


def test_the_contents_count_the_rows_of_each_table(tmp_path: Path) -> None:
    """What is at stake is rows, so rows are what the account is in.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    with opened(url) as engine, engine.begin() as connection:
        connection.execute(IMAGES.insert(), image_row())
    counted = {table.name: table.rows for table in _contents(url).tables}
    assert counted[IMAGES.name] == 1


def test_the_contents_total_the_rows_across_the_tables(tmp_path: Path) -> None:
    """One number for a person deciding, beside the per-table breakdown.

    The stamp row is one of them, so a freshly created index holding one image
    totals two.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    with opened(url) as engine, engine.begin() as connection:
        connection.execute(IMAGES.insert(), image_row())
    assert _contents(url).rows == 2


def test_the_contents_report_the_stamp(tmp_path: Path) -> None:
    """A version that is not this one is what makes a drop the right move.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    assert _contents(url).schema_version == SCHEMA_VERSION


def test_the_contents_of_a_database_holding_no_index_are_empty(tmp_path: Path) -> None:
    """Which is what lets a drop answer without writing anything.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = sqlite_url_for(tmp_path / 'index.sqlite3')
    _add_foreign_table(url)
    assert _contents(url).tables == ()


def test_the_contents_of_a_database_holding_no_index_name_nothing_unproven(
    tmp_path: Path,
) -> None:
    """A database with nothing of these names in it is the state asked for.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = sqlite_url_for(tmp_path / 'index.sqlite3')
    _add_foreign_table(url)
    assert _contents(url).unproven == ()


def test_the_contents_count_an_unfinished_ingest_run(tmp_path: Path) -> None:
    """A drop is allowed under one, so the person answering is told about it.

    Nothing in the index tells a pass that is writing it now from one that died,
    so the count is reported rather than acted on.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    with opened(url) as engine, engine.begin() as connection:
        connection.execute(
            INGEST_RUNS.insert(),
            {
                'root_url': '/data/nav-results',
                'started_utc': '2026-08-09T00:00:00+00:00',
                'finished_utc': None,
                'schema_version': SCHEMA_VERSION,
            },
        )
    assert _contents(url).unfinished_runs == 1


def test_a_finished_ingest_run_is_not_counted_as_unfinished(tmp_path: Path) -> None:
    """Otherwise every drop would carry the warning and none would mean it.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    with opened(url) as engine, engine.begin() as connection:
        connection.execute(
            INGEST_RUNS.insert(),
            {
                'root_url': '/data/nav-results',
                'started_utc': '2026-08-09T00:00:00+00:00',
                'finished_utc': '2026-08-09T00:01:00+00:00',
                'schema_version': SCHEMA_VERSION,
            },
        )
    assert _contents(url).unfinished_runs == 0


def test_the_unfinished_count_is_withheld_at_another_schema_version(tmp_path: Path) -> None:
    """The question is phrased in a column, and a version may have changed it.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    with opened(url) as engine, engine.begin() as connection:
        connection.execute(SCHEMA_META.update().values(schema_version=SCHEMA_VERSION + 1))
    assert _contents(url).unfinished_runs is None


# ---------------------------------------------------------------------------
# The opener the drop uses
# ---------------------------------------------------------------------------


def test_a_sqlite_path_that_is_not_there_is_refused(tmp_path: Path) -> None:
    """A database that is not there is refused on both backends alike.

    A PostgreSQL database that does not exist is refused by the server, and a
    typed path deserves the same answer rather than a quiet success over
    nothing.

    Parameters:
        tmp_path: Directory the path names a file in.
    """
    with pytest.raises(ValueError, match='there is no results index at'):
        open_database(sqlite_url_for(tmp_path / 'absent.sqlite3'))


def test_the_absent_refusal_says_nothing_was_dropped(tmp_path: Path) -> None:
    """Rather than telling an operator who is deleting one to build one.

    Parameters:
        tmp_path: Directory the path names a file in.
    """
    with pytest.raises(ValueError, match='Nothing was dropped'):
        open_database(sqlite_url_for(tmp_path / 'absent.sqlite3'))


def test_a_refused_path_is_not_created(tmp_path: Path) -> None:
    """A mistyped URL must not leave an empty database where it pointed.

    Parameters:
        tmp_path: Directory the path names a file in.
    """
    path = tmp_path / 'absent.sqlite3'
    with pytest.raises(ValueError):
        open_database(sqlite_url_for(path))
    assert not path.exists()


def test_a_read_only_database_is_refused_before_a_table_goes(tmp_path: Path) -> None:
    """Refusing beats half-completing, so the question is asked at the open.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    path = tmp_path / 'index.sqlite3'
    url = _built(path)
    path.chmod(0o444)
    try:
        with pytest.raises(ValueError, match='dropping the index has to write it'):
            open_database(url)
    finally:
        path.chmod(0o644)


def test_the_read_only_refusal_does_not_prescribe_ingesting_a_copy(tmp_path: Path) -> None:
    """The remedy is per operation: a copy answers nothing for a drop.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    path = tmp_path / 'index.sqlite3'
    url = _built(path)
    path.chmod(0o444)
    try:
        with pytest.raises(ValueError) as excinfo:
            open_database(url)
    finally:
        path.chmod(0o644)
    assert 'Ingest a writable copy' not in str(excinfo.value)


def test_a_url_no_driver_reads_is_refused_as_a_value_error() -> None:
    """The drop's opener keeps the one-exception-type contract of the other.

    Parameters:
        None.
    """
    with pytest.raises(ValueError, match='no database driver for this URL scheme'):
        open_database('nosuchbackend://host/db')
